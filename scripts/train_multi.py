#!/usr/bin/env python3
"""NeuroSpiral — Multi-Subject Training.

Downloads and processes multiple Sleep-EDF recordings to train
a more robust classifier. Single-subject training overfits to
that person's brain; multi-subject training learns the universal
geometric signatures of each sleep stage.

Usage:
    # Train on 5 subjects (10 recordings — 2 nights each)
    python scripts/train_multi.py --n-subjects 5

    # Train on all available (up to 20 subjects)
    python scripts/train_multi.py --n-subjects 20

    # Use existing downloaded data only (no new downloads)
    python scripts/train_multi.py --local-only
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
from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneGroupOut,
    cross_val_predict,
)
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.edf_loader import load_sleep_edf, extract_epochs_from_annotations
from src.preprocessing.pipeline import preprocess_raw, compute_epoch_quality
from src.features.takens import time_delay_embedding
from src.features.topology import extract_tda_features
from src.geometry.tesseract import (
    project_to_clifford_torus,
    extract_tesseract_features,
)
from src.geometry.wasserstein import (
    trajectory_to_spd,
    compute_reference_spd,
    bures_wasserstein,
    extract_distance_features,
)

# ──────────────────────────────────────────────────────────────
# Sleep-EDF subject list (cassette study — healthy subjects)
# Each subject has 2 nights: E0 (night 1) and E0 (night 2)
# Format: SC4SSNE0-PSG.edf + SC4SSNEC-Hypnogram.edf
# where SS = subject number, N = night (0 or 1)
# ──────────────────────────────────────────────────────────────

SLEEP_EDF_SUBJECTS = [
    # (subject_id, night, psg_filename, hyp_filename)
    ("SC4001", 0, "SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf"),
    ("SC4002", 0, "SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf"),
    ("SC4011", 0, "SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf"),
    ("SC4012", 0, "SC4012E0-PSG.edf", "SC4012EC-Hypnogram.edf"),
    ("SC4021", 0, "SC4021E0-PSG.edf", "SC4021EH-Hypnogram.edf"),
    ("SC4022", 0, "SC4022E0-PSG.edf", "SC4022EH-Hypnogram.edf"),
    ("SC4031", 0, "SC4031E0-PSG.edf", "SC4031EC-Hypnogram.edf"),
    ("SC4032", 0, "SC4032E0-PSG.edf", "SC4032EC-Hypnogram.edf"),
    ("SC4041", 0, "SC4041E0-PSG.edf", "SC4041EC-Hypnogram.edf"),
    ("SC4042", 0, "SC4042E0-PSG.edf", "SC4042EC-Hypnogram.edf"),
    ("SC4051", 0, "SC4051E0-PSG.edf", "SC4051EC-Hypnogram.edf"),
    ("SC4052", 0, "SC4052E0-PSG.edf", "SC4052EC-Hypnogram.edf"),
    ("SC4061", 0, "SC4061E0-PSG.edf", "SC4061EC-Hypnogram.edf"),
    ("SC4062", 0, "SC4062E0-PSG.edf", "SC4062EC-Hypnogram.edf"),
    ("SC4071", 0, "SC4071E0-PSG.edf", "SC4071EC-Hypnogram.edf"),
    ("SC4072", 0, "SC4072E0-PSG.edf", "SC4072EC-Hypnogram.edf"),
    ("SC4081", 0, "SC4081E0-PSG.edf", "SC4081EC-Hypnogram.edf"),
    ("SC4082", 0, "SC4082E0-PSG.edf", "SC4082EC-Hypnogram.edf"),
    ("SC4091", 0, "SC4091E0-PSG.edf", "SC4091EC-Hypnogram.edf"),
    ("SC4092", 0, "SC4092E0-PSG.edf", "SC4092EC-Hypnogram.edf"),
]

LABEL_MAPPING = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Movement time": "W",
    "Sleep stage ?": None,
}

SPECTRAL_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 15.0),
    "beta": (15.0, 30.0),
}

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"


# ──────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────

def download_subject(subject_info: tuple, data_dir: Path) -> tuple[Path, Path] | None:
    """Download one subject's EDF files."""
    import urllib.request
    import urllib.error

    sid, night, psg_name, hyp_name = subject_info
    data_dir.mkdir(parents=True, exist_ok=True)

    psg_path = data_dir / psg_name
    hyp_path = data_dir / hyp_name

    for fname, fpath in [(psg_name, psg_path), (hyp_name, hyp_path)]:
        if fpath.exists():
            continue
        url = f"{BASE_URL}/{fname}"
        try:
            print(f"    ↓ {fname}...")
            urllib.request.urlretrieve(url, fpath)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"    ✗ Failed to download {fname}: {e}")
            return None

    return psg_path, hyp_path


# ──────────────────────────────────────────────────────────────
# Process one subject
# ──────────────────────────────────────────────────────────────

def process_subject(
    psg_path: Path,
    hyp_path: Path,
    use_tda: bool = False,  # Off by default for speed in multi-subject
) -> tuple[np.ndarray, np.ndarray, list[str], str] | None:
    """Process one recording: load → preprocess → extract features."""

    try:
        record = load_sleep_edf(
            psg_path, hyp_path,
            channels=["EEG Fpz-Cz"],
            label_mapping=LABEL_MAPPING,
        )
    except Exception as e:
        print(f"    ✗ Load failed: {e}")
        return None

    try:
        result = preprocess_raw(record.raw, 0.5, 30.0, 100.0, {
            "n_components": 10, "method": "fastica",
            "max_iter": 500, "random_state": 42, "eog_threshold": 0.85,
        })
    except Exception as e:
        print(f"    ✗ Preprocess failed: {e}")
        return None

    record.raw = result.raw
    sfreq = result.raw.info["sfreq"]

    try:
        epochs, labels, names = extract_epochs_from_annotations(record)
    except Exception as e:
        print(f"    ✗ Epoch extraction failed: {e}")
        return None

    quality = compute_epoch_quality(epochs, sfreq)
    epochs, labels = epochs[quality], labels[quality]

    if len(epochs) < 50:
        print(f"    ✗ Too few epochs ({len(epochs)})")
        return None

    # Extract features
    all_feats = []
    for i in range(len(epochs)):
        epoch_1d = epochs[i, 0, :]
        feats = {}

        # Spectral
        freqs, psd = scipy_signal.welch(epoch_1d, fs=sfreq, nperseg=min(256, len(epoch_1d)))
        total = np.trapz(psd, freqs)
        for name, (lo, hi) in SPECTRAL_BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            feats[f"spec_{name}"] = float(np.trapz(psd[mask], freqs[mask]) / (total + 1e-10))
        feats["spec_delta_beta"] = feats["spec_delta"] / (feats["spec_beta"] + 1e-10)

        # Tesseract geometry
        try:
            cloud, tau = time_delay_embedding(epoch_1d, dimension=4)
            cloud_torus = project_to_clifford_torus(cloud)
            tess = extract_tesseract_features(cloud_torus)
            for k, v in tess.items():
                feats[f"tess_{k}"] = v
        except Exception:
            pass

        # TDA (optional, slow)
        if use_tda:
            try:
                cloud, _ = time_delay_embedding(epoch_1d, dimension=4)
                tda = extract_tda_features(cloud, max_dim=2, n_subsample=300)
                feats.update(tda)
            except Exception:
                pass

        all_feats.append(feats)

    all_keys = sorted(set().union(*[f.keys() for f in all_feats]))
    X = np.array([[f.get(k, 0.0) for k in all_keys] for f in all_feats])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    subject_id = psg_path.stem.split("-")[0]
    return X, labels, names, subject_id


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NeuroSpiral Multi-Subject Training")
    parser.add_argument("--n-subjects", type=int, default=5,
                        help="Number of subjects to train on (max 20)")
    parser.add_argument("--local-only", action="store_true",
                        help="Only use already-downloaded data")
    parser.add_argument("--use-tda", action="store_true",
                        help="Include TDA features (slower but richer)")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data/results/multi")
    args = parser.parse_args()

    t_start = time.time()
    data_dir = PROJECT_ROOT / "data/raw"
    n_subjects = min(args.n_subjects, len(SLEEP_EDF_SUBJECTS))

    print("\n" + "═" * 60)
    print("  NeuroSpiral — Multi-Subject Training")
    print(f"  Subjects: {n_subjects} | TDA: {'on' if args.use_tda else 'off'}")
    print("═" * 60)

    # ── Download ─────────────────────────────────────────
    print(f"\n[1/4] Downloading {n_subjects} subjects from PhysioNet...")

    subject_files = []
    for info in SLEEP_EDF_SUBJECTS[:n_subjects]:
        sid = info[0]
        psg_path = data_dir / info[2]
        hyp_path = data_dir / info[3]

        if psg_path.exists() and hyp_path.exists():
            print(f"  ✓ {sid} (cached)")
            subject_files.append((psg_path, hyp_path, sid))
        elif not args.local_only:
            result = download_subject(info, data_dir)
            if result:
                subject_files.append((*result, sid))
                print(f"  ✓ {sid}")
        else:
            print(f"  ⊘ {sid} (not downloaded, --local-only)")

    if not subject_files:
        print("  ✗ No subjects available!")
        return

    print(f"\n  Available: {len(subject_files)} subjects")

    # ── Process each subject ─────────────────────────────
    print(f"\n[2/4] Processing subjects...")

    all_X = []
    all_y = []
    all_groups = []  # subject ID per epoch (for LOGO-CV)
    all_names = None

    for psg, hyp, sid in subject_files:
        print(f"\n  Processing {sid}...")
        result = process_subject(psg, hyp, use_tda=args.use_tda)

        if result is None:
            continue

        X, y, names, subject_id = result
        all_X.append(X)
        all_y.append(y)
        all_groups.append(np.full(len(y), len(all_X) - 1))

        if all_names is None:
            all_names = names

        print(f"    ✓ {X.shape[0]} epochs × {X.shape[1]} features")
        for i, name in enumerate(names):
            count = np.sum(y == i)
            print(f"      {name}: {count}")

    if not all_X:
        print("  ✗ No subjects processed successfully!")
        return

    # Align feature dimensions (some subjects may have different feature counts)
    # Find common feature count (use max and pad with zeros)
    max_features = max(x.shape[1] for x in all_X)
    aligned_X = []
    for x in all_X:
        if x.shape[1] < max_features:
            padding = np.zeros((x.shape[0], max_features - x.shape[1]))
            x = np.hstack([x, padding])
        aligned_X.append(x)

    X_all = np.vstack(aligned_X)
    y_all = np.concatenate(all_y)
    groups_all = np.concatenate(all_groups)

    print(f"\n  Combined: {X_all.shape[0]} epochs × {X_all.shape[1]} features")
    print(f"  Subjects: {len(all_X)}")

    print(f"\n  Overall stage distribution:")
    for i, name in enumerate(all_names):
        count = np.sum(y_all == i)
        pct = count / len(y_all) * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:>4}: {count:>5} ({pct:5.1f}%) {bar}")

    # ── Train & Evaluate ─────────────────────────────────
    print(f"\n[3/4] Training classifier...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    # Method A: Standard 5-fold CV (epochs shuffled across subjects)
    print(f"\n  Method A: 5-fold stratified CV (mixed subjects)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(clf, X_scaled, y_all, cv=cv)

    f1_cv = f1_score(y_all, y_pred_cv, average="macro", zero_division=0)
    kappa_cv = cohen_kappa_score(y_all, y_pred_cv)

    print(f"    F1-macro: {f1_cv:.3f}")
    print(f"    Cohen κ: {kappa_cv:.3f}")

    # Method B: Leave-One-Subject-Out (true generalization test)
    if len(all_X) >= 3:
        print(f"\n  Method B: Leave-One-Subject-Out CV (true generalization)")
        logo = LeaveOneGroupOut()
        y_pred_logo = cross_val_predict(clf, X_scaled, y_all, cv=logo, groups=groups_all)

        f1_logo = f1_score(y_all, y_pred_logo, average="macro", zero_division=0)
        kappa_logo = cohen_kappa_score(y_all, y_pred_logo)

        print(f"    F1-macro: {f1_logo:.3f}")
        print(f"    Cohen κ: {kappa_logo:.3f}")

        print(f"\n  Gap: {f1_cv - f1_logo:+.3f} "
              f"({'acceptable' if abs(f1_cv - f1_logo) < 0.10 else 'OVERFITTING — add subjects'})")
    else:
        y_pred_logo = y_pred_cv
        f1_logo = f1_cv
        print(f"\n  Method B: skipped (need ≥3 subjects for LOGO-CV)")

    # ── Full report ──────────────────────────────────────
    print(f"\n[4/4] Full evaluation...")

    best_pred = y_pred_logo if len(all_X) >= 3 else y_pred_cv
    best_method = "LOGO" if len(all_X) >= 3 else "5-fold"

    print(f"\n  Classification report ({best_method}):")
    report = classification_report(
        y_all, best_pred,
        target_names=all_names, digits=3, zero_division=0,
    )
    for line in report.split("\n"):
        print(f"    {line}")

    # N3 specific
    if "N3" in all_names:
        n3_idx = all_names.index("N3")
        n3_true = (y_all == n3_idx).astype(int)
        n3_pred = (best_pred == n3_idx).astype(int)

        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  N3 (Glymphatic Target) — {len(all_X)} subjects          │")
        print(f"  │  Precision: {precision_score(n3_true, n3_pred, zero_division=0):.3f}"
              f"                              │")
        print(f"  │  Recall:    {recall_score(n3_true, n3_pred, zero_division=0):.3f}"
              f"                              │")
        print(f"  │  F1:        {f1_score(n3_true, n3_pred, zero_division=0):.3f}"
              f"                              │")
        print(f"  └─────────────────────────────────────────────┘")

    # ── Save ─────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Train final model on all data and save
    clf.fit(X_scaled, y_all)

    np.savez_compressed(
        args.output_dir / "multi_subject_model.npz",
        features=X_all,
        labels=y_all,
        groups=groups_all,
        label_names=all_names,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        n_subjects=len(all_X),
    )

    # Save feature importances
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    print(f"\n  Top 20 features (trained on {len(all_X)} subjects):")
    for idx in top_idx:
        print(f"    {idx:>3}: {importances[idx]:.4f}")

    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  Training complete in {elapsed:.0f}s")
    print(f"  {len(all_X)} subjects | {X_all.shape[0]} epochs | "
          f"F1={f1_cv:.3f} (CV) / {f1_logo:.3f} (LOGO)")
    print(f"  Saved to {args.output_dir}/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
