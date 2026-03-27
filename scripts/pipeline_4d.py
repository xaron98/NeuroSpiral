#!/usr/bin/env python3
"""NeuroSpiral — 4D Topological Sleep Stage Classification Pipeline.

Implements the full "Spiral 4D" approach:
  1. Load EDF polysomnography + hypnogram
  2. Preprocess: bandpass 0.5–30 Hz, ICA, 30s epoch segmentation
  3. Takens embedding: 1D EEG → 4D phase-space point cloud
  4. Persistent homology: extract topological features (H0, H1, H2)
  5. Combine with spectral features (delta power)
  6. Train RandomForest classifier, evaluate with N3-priority metrics

Usage:
    # Download sample + run full pipeline
    python scripts/pipeline_4d.py --download-sample

    # Custom files
    python scripts/pipeline_4d.py --psg data/raw/SC4001E0-PSG.edf \\
                                   --hyp data/raw/SC4001EC-Hypnogram.edf

    # Skip TDA (spectral-only baseline for comparison)
    python scripts/pipeline_4d.py --download-sample --no-tda

Mathematical foundation:
    Takens' Theorem (1981): A smooth dynamical system on a d-dimensional
    manifold can be reconstructed (up to diffeomorphism) from a single
    scalar observable via delay embedding in m ≥ 2d+1 dimensions.

    For sleep EEG: the underlying "sleep state" lives on a low-dimensional
    manifold. Different stages produce geometrically distinct attractors
    whose topology (connected components, loops, voids) is captured by
    persistent homology and serves as a stage fingerprint.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

# Suppress MNE verbosity
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.edf_loader import load_sleep_edf, extract_epochs_from_annotations
from src.preprocessing.pipeline import preprocess_raw, compute_epoch_quality
from src.features.takens import time_delay_embedding
from src.features.topology import extract_tda_features

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

LABEL_MAPPING = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",      # R&K → AASM
    "Sleep stage R": "REM",
    "Movement time": "W",
    "Sleep stage ?": None,
}

PREPROCESS = {
    "channels": ["EEG Fpz-Cz"],  # Single channel for TDA (cleaner embedding)
    "l_freq": 0.5,
    "h_freq": 30.0,
    "resample_hz": 100,
    "ica": {
        "n_components": 10,
        "method": "fastica",
        "max_iter": 500,
        "random_state": 42,
        "eog_threshold": 0.85,
    },
}

TAKENS = {
    "dimension": 4,
    "tau": None,           # Auto-estimate via mutual information
    "n_bins_mi": 64,
    "max_tau_search": 80,  # ~0.8s at 100 Hz
}

TDA = {
    "max_dim": 2,          # H0 (components), H1 (loops), H2 (voids)
    "max_edge": 2.0,
    "n_subsample": 300,
    "betti_bins": 20,
}

SPECTRAL_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 15.0),
    "beta": (15.0, 30.0),
}

MODEL = {
    "n_estimators": 300,
    "max_depth": 20,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "cv_folds": 5,
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def download_sample(output_dir: Path) -> tuple[Path, Path]:
    """Download Sleep-EDF sample from PhysioNet."""
    import urllib.request

    output_dir.mkdir(parents=True, exist_ok=True)
    base = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
    files = {
        "SC4001E0-PSG.edf": f"{base}/SC4001E0-PSG.edf",
        "SC4001EC-Hypnogram.edf": f"{base}/SC4001EC-Hypnogram.edf",
    }
    paths = {}
    for fname, url in files.items():
        p = output_dir / fname
        if p.exists():
            print(f"  ✓ {fname} (cached)")
        else:
            print(f"  ↓ {fname}...")
            urllib.request.urlretrieve(url, p)
            print(f"  ✓ saved")
        paths[fname] = p
    return paths["SC4001E0-PSG.edf"], paths["SC4001EC-Hypnogram.edf"]


def compute_spectral_features(
    epoch: np.ndarray,
    sfreq: float,
) -> dict[str, float]:
    """Compute relative band powers for one epoch (1D signal)."""
    freqs, psd = scipy_signal.welch(epoch, fs=sfreq, nperseg=min(256, len(epoch)))
    total = np.trapz(psd, freqs)
    if total == 0:
        return {name: 0.0 for name in SPECTRAL_BANDS}
    features = {}
    for name, (lo, hi) in SPECTRAL_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        features[f"spec_{name}"] = float(np.trapz(psd[mask], freqs[mask]) / total)
    # Key ratio: delta/beta (high in N3)
    eps = 1e-10
    features["spec_delta_beta"] = features["spec_delta"] / (features["spec_beta"] + eps)
    return features


def print_stage_distribution(labels: np.ndarray, names: list[str]):
    """Print stage counts with visual bars."""
    for i, name in enumerate(names):
        count = np.sum(labels == i)
        pct = count / len(labels) * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:>4}: {count:>4} ({pct:5.1f}%) {bar}")


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def run_pipeline(
    psg_path: Path,
    hyp_path: Path,
    use_tda: bool = True,
    output_dir: Path | None = None,
):
    """Execute the full 4D topological classification pipeline."""

    t_start = time.time()

    # ── Step 0: Load ─────────────────────────────────────────
    print("\n═══════════════════════════════════════════════════")
    print("  NEUROSPIRAL — 4D Topological Sleep Staging")
    print("═══════════════════════════════════════════════════")

    print("\n[1/6] Loading EDF files...")
    record = load_sleep_edf(
        psg_path=psg_path,
        hypnogram_path=hyp_path,
        channels=PREPROCESS["channels"],
        label_mapping=LABEL_MAPPING,
    )
    print(f"  Subject: {record.subject_id}")
    print(f"  Duration: {record.duration_hours:.1f}h | {record.sfreq} Hz")
    print(f"  Channels: {record.raw.ch_names}")

    # ── Step 1: Preprocess ───────────────────────────────────
    print("\n[2/6] Preprocessing...")
    result = preprocess_raw(
        raw=record.raw,
        l_freq=PREPROCESS["l_freq"],
        h_freq=PREPROCESS["h_freq"],
        resample_hz=PREPROCESS["resample_hz"],
        ica_config=PREPROCESS["ica"],
    )
    sfreq = result.raw.info["sfreq"]
    print(f"  Pipeline: {' → '.join(result.steps_applied)}")
    print(f"  Final sfreq: {sfreq} Hz")

    # Extract epochs
    record.raw = result.raw
    epochs, labels, label_names = extract_epochs_from_annotations(record)
    print(f"  Epochs: {epochs.shape[0]} × {epochs.shape[2]} samples")

    # Quality filter
    quality_mask = compute_epoch_quality(epochs, sfreq)
    epochs = epochs[quality_mask]
    labels = labels[quality_mask]
    print(f"  After quality filter: {epochs.shape[0]} epochs")
    print("\n  Stage distribution:")
    print_stage_distribution(labels, label_names)

    # ── Step 2 & 3: Takens Embedding + TDA ───────────────────
    n_epochs = epochs.shape[0]
    all_features = []
    tau_values = []

    if use_tda:
        print(f"\n[3/6] Takens embedding (d={TAKENS['dimension']}) + TDA...")
        t_tda = time.time()

        for i in range(n_epochs):
            if i % 50 == 0:
                elapsed = time.time() - t_tda
                rate = (i / elapsed) if elapsed > 0 else 0
                eta = ((n_epochs - i) / rate) if rate > 0 else 0
                print(f"  Epoch {i:>4}/{n_epochs} "
                      f"({i/n_epochs*100:5.1f}%) "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

            epoch_1d = epochs[i, 0, :]  # Single channel

            # Takens embedding: 1D → 4D point cloud
            point_cloud, tau = time_delay_embedding(
                epoch_1d,
                dimension=TAKENS["dimension"],
                tau=TAKENS["tau"],
                n_bins=TAKENS["n_bins_mi"],
                max_tau_search=TAKENS["max_tau_search"],
            )
            tau_values.append(tau)

            # TDA features from the 4D point cloud
            tda_feats = extract_tda_features(
                point_cloud,
                max_dim=TDA["max_dim"],
                max_edge=TDA["max_edge"],
                n_subsample=TDA["n_subsample"],
                betti_bins=TDA["betti_bins"],
            )

            # Spectral features (delta power etc.)
            spec_feats = compute_spectral_features(epoch_1d, sfreq)

            # Merge
            combined = {**tda_feats, **spec_feats}
            all_features.append(combined)

        tda_elapsed = time.time() - t_tda
        print(f"  ✓ Done in {tda_elapsed:.1f}s "
              f"({tda_elapsed/n_epochs*1000:.0f}ms/epoch)")
        print(f"  τ distribution: median={np.median(tau_values):.0f}, "
              f"range=[{np.min(tau_values)}, {np.max(tau_values)}] samples")
    else:
        print("\n[3/6] Spectral features only (TDA disabled)...")
        for i in range(n_epochs):
            epoch_1d = epochs[i, 0, :]
            spec_feats = compute_spectral_features(epoch_1d, sfreq)
            all_features.append(spec_feats)
        print(f"  ✓ {n_epochs} epochs processed")

    # Build feature matrix
    feature_names = list(all_features[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in all_features])
    y = labels

    # Handle NaN/Inf
    nan_mask = np.isnan(X) | np.isinf(X)
    if nan_mask.any():
        n_bad = nan_mask.any(axis=1).sum()
        print(f"  ⚠ {n_bad} epochs with NaN/Inf — replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  Feature matrix: {X.shape} ({len(feature_names)} features)")
    if use_tda:
        n_tda = sum(1 for k in feature_names if not k.startswith("spec_"))
        n_spec = sum(1 for k in feature_names if k.startswith("spec_"))
        print(f"    Topological: {n_tda} | Spectral: {n_spec}")

    # ── Step 4: Classification ───────────────────────────────
    print(f"\n[4/6] Training RandomForest (CV={MODEL['cv_folds']})...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=MODEL["n_estimators"],
        max_depth=MODEL["max_depth"],
        class_weight=MODEL["class_weight"],
        random_state=MODEL["random_state"],
        n_jobs=MODEL["n_jobs"],
    )

    cv = StratifiedKFold(
        n_splits=MODEL["cv_folds"],
        shuffle=True,
        random_state=42,
    )

    all_y_true = []
    all_y_pred = []
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fold_f1 = f1_score(y_test, y_pred, average="macro")
        fold_scores.append(fold_f1)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold+1}: F1-macro = {fold_f1:.3f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # ── Step 5: Evaluation ───────────────────────────────────
    print(f"\n[5/6] Evaluation (N3 = glymphatic target)")
    print("─" * 55)

    print(f"\n  Cross-validated F1-macro: {np.mean(fold_scores):.3f} "
          f"± {np.std(fold_scores):.3f}")

    print(f"\n  Classification Report:")
    report = classification_report(
        all_y_true, all_y_pred,
        target_names=label_names,
        digits=3,
        zero_division=0,
    )
    for line in report.split("\n"):
        print(f"    {line}")

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\n  Confusion Matrix:")
    header = "        " + "  ".join(f"{n:>5}" for n in label_names)
    print(f"    {header}")
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>5}" for v in row)
        print(f"    {label_names[i]:>5}   {row_str}")

    # N3-specific metrics (the target for glymphatic stimulation)
    if "N3" in label_names:
        n3_idx = label_names.index("N3")
        n3_mask_true = all_y_true == n3_idx
        n3_mask_pred = all_y_pred == n3_idx

        tp = np.sum(n3_mask_true & n3_mask_pred)
        fp = np.sum(~n3_mask_true & n3_mask_pred)
        fn = np.sum(n3_mask_true & ~n3_mask_pred)

        n3_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        n3_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        n3_f1 = 2 * n3_precision * n3_recall / (n3_precision + n3_recall + 1e-10)

        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  N3 (Slow Wave) — Glymphatic Target     │")
        print(f"  │  Precision: {n3_precision:.3f}  "
              f"(false alarms)             │")
        print(f"  │  Recall:    {n3_recall:.3f}  "
              f"(missed N3 windows)        │")
        print(f"  │  F1-Score:  {n3_f1:.3f}                         │")
        print(f"  └─────────────────────────────────────────┘")

    # ── Step 6: Feature importance ───────────────────────────
    if use_tda:
        print(f"\n[6/6] Feature importance (top 15)...")
        # Retrain on full dataset for importances
        clf.fit(X_scaled, y)
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]

        print(f"\n  {'Feature':<30} {'Importance':>10}  {'Type':>8}")
        print(f"  {'─'*30} {'─'*10}  {'─'*8}")
        for idx in top_idx:
            name = feature_names[idx]
            imp = importances[idx]
            ftype = "TDA" if not name.startswith("spec_") else "Spectral"
            bar = "▓" * int(imp * 200)
            print(f"  {name:<30} {imp:>10.4f}  {ftype:>8}  {bar}")

        # TDA vs Spectral contribution
        tda_imp = sum(importances[i] for i, k in enumerate(feature_names)
                      if not k.startswith("spec_"))
        spec_imp = sum(importances[i] for i, k in enumerate(feature_names)
                       if k.startswith("spec_"))
        total_imp = tda_imp + spec_imp
        print(f"\n  Total importance share:")
        print(f"    Topological: {tda_imp/total_imp*100:.1f}%")
        print(f"    Spectral:    {spec_imp/total_imp*100:.1f}%")

    # ── Save results ─────────────────────────────────────────
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / f"{record.subject_id}_4d_pipeline.npz",
            features=X,
            labels=y,
            label_names=label_names,
            feature_names=feature_names,
            predictions=all_y_pred,
            true_labels=all_y_true,
        )
        print(f"\n  💾 Results saved to {output_dir}/")

    elapsed_total = time.time() - t_start
    print(f"\n{'═'*55}")
    print(f"  Pipeline complete in {elapsed_total:.1f}s")
    print(f"{'═'*55}\n")

    return {
        "f1_macro": np.mean(fold_scores),
        "f1_std": np.std(fold_scores),
        "n3_f1": n3_f1 if "N3" in label_names else None,
        "n_features": len(feature_names),
        "n_epochs": n_epochs,
    }


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NeuroSpiral 4D Topological Sleep Staging Pipeline"
    )
    parser.add_argument("--psg", type=Path, help="PSG .edf file")
    parser.add_argument("--hyp", type=Path, help="Hypnogram .edf file")
    parser.add_argument("--download-sample", action="store_true")
    parser.add_argument(
        "--no-tda", action="store_true",
        help="Skip TDA (spectral-only baseline for comparison)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data/results",
    )
    args = parser.parse_args()

    if args.download_sample:
        print("\n📥 Downloading Sleep-EDF sample...")
        raw_dir = PROJECT_ROOT / "data/raw"
        psg_path, hyp_path = download_sample(raw_dir)
    elif args.psg and args.hyp:
        psg_path, hyp_path = args.psg, args.hyp
    else:
        parser.error("Provide --psg and --hyp, or use --download-sample")
        return

    results = run_pipeline(
        psg_path=psg_path,
        hyp_path=hyp_path,
        use_tda=not args.no_tda,
        output_dir=args.output_dir,
    )

    # If both modes were run, compare
    if not args.no_tda:
        print("\n💡 Tip: Run with --no-tda to get a spectral-only baseline")
        print("   and compare against the topological pipeline.")


if __name__ == "__main__":
    main()
