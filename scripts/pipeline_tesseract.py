#!/usr/bin/env python3
"""NeuroSpiral — Tesseract-Wasserstein Pipeline.

Integrates the mathematical framework from both research documents:

    EEG signal
      → Takens embedding (1D → ℝ⁴)
      → Clifford torus projection (ℝ⁴ → 𝕋_{√2})
      → Tesseract discretization Q(x) = sgn(x) → {±1}⁴
      → Vertex residence analysis (stability, transitions)
      → Bures-Wasserstein distance to reference states (SPD manifold)
      → Combined feature vector (TDA + spectral + tesseract + BW)
      → RandomForest classification with N3-priority evaluation

Key insight: the tesseract vertices are not arbitrary labels but
the intrinsic discretization of Clifford torus flows. Transitions
between vertices (Hamming distance 1 = edge traversal) correspond
to single-coordinate sign flips — one physiological marker changing
while others remain stable.

The Bures-Wasserstein distance respects the Riemannian geometry
of SPD covariance matrices. For 4×4 matrices, computation is
essentially O(1) via eigendecomposition — viable for real-time
on embedded hardware.

Usage:
    python scripts/pipeline_tesseract.py --download-sample
    python scripts/pipeline_tesseract.py --psg file.edf --hyp hyp.edf
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.edf_loader import load_sleep_edf, extract_epochs_from_annotations
from src.preprocessing.pipeline import preprocess_raw, compute_epoch_quality
from src.features.takens import time_delay_embedding
from src.features.topology import extract_tda_features
from src.geometry.tesseract import (
    VERTICES,
    project_to_clifford_torus,
    extract_tesseract_features,
    analyze_vertex_residence,
    to_torus_angles,
    hamming_distance,
    vertex_code,
)
from src.geometry.wasserstein import (
    trajectory_to_spd,
    compute_reference_spd,
    bures_wasserstein,
    bures_wasserstein_mean_distance,
    extract_distance_features,
)


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

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


def download_sample(output_dir: Path) -> tuple[Path, Path]:
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
        if not p.exists():
            print(f"  ↓ {fname}...")
            urllib.request.urlretrieve(url, p)
        paths[fname] = p
    return paths["SC4001E0-PSG.edf"], paths["SC4001EC-Hypnogram.edf"]


# ──────────────────────────────────────────────────────────────
# Feature extraction per epoch
# ──────────────────────────────────────────────────────────────

def extract_spectral(epoch_1d: np.ndarray, sfreq: float) -> dict[str, float]:
    """Spectral band powers."""
    freqs, psd = scipy_signal.welch(epoch_1d, fs=sfreq, nperseg=min(256, len(epoch_1d)))
    total = np.trapz(psd, freqs)
    feats = {}
    for name, (lo, hi) in SPECTRAL_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        feats[f"spec_{name}"] = float(np.trapz(psd[mask], freqs[mask]) / (total + 1e-10))
    feats["spec_delta_beta"] = feats["spec_delta"] / (feats["spec_beta"] + 1e-10)
    return feats


def extract_full_features(
    epoch_1d: np.ndarray,
    sfreq: float,
    use_tda: bool = True,
    reference_mean: np.ndarray | None = None,
    reference_cov: np.ndarray | None = None,
) -> dict[str, float]:
    """Extract all features: spectral + Takens → TDA + tesseract + BW.

    This is the unified feature extraction combining all three
    mathematical frameworks:
    1. Spectral (classical FFT band powers)
    2. TDA (persistent homology on 4D point cloud)
    3. Tesseract geometry (vertex residence, torus angles, ω-ratio)
    4. Bures-Wasserstein (covariance distance on SPD manifold)
    """
    features = {}

    # 1. Spectral features
    features.update(extract_spectral(epoch_1d, sfreq))

    # 2-4. All require Takens embedding
    try:
        cloud_raw, tau = time_delay_embedding(epoch_1d, dimension=4)
        features["takens_tau"] = float(tau)

        # 2. TDA features (persistent homology)
        if use_tda:
            try:
                tda = extract_tda_features(cloud_raw, max_dim=2, n_subsample=300)
                features.update(tda)
            except Exception:
                pass

        # 3. Tesseract geometry (project to Clifford torus first)
        cloud_torus = project_to_clifford_torus(cloud_raw)
        tess_feats = extract_tesseract_features(cloud_torus)
        for k, v in tess_feats.items():
            features[f"tess_{k}"] = v

        # 4. Bures-Wasserstein distance to reference (if available)
        if reference_mean is not None and reference_cov is not None:
            bw_feats = extract_distance_features(
                cloud_torus, reference_mean, reference_cov
            )
            for k, v in bw_feats.items():
                features[f"bw_{k}"] = v

    except Exception:
        pass

    return features


# ──────────────────────────────────────────────────────────────
# Reference state building
# ──────────────────────────────────────────────────────────────

def build_reference_states(
    epochs: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    sfreq: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build reference SPD covariances for each sleep stage.

    For each stage, embed all epochs via Takens, project to Clifford
    torus, and compute the average covariance (Fréchet mean approximation).

    Returns dict mapping stage_name → (mean, cov).
    """
    references = {}

    for stage_idx, stage_name in enumerate(label_names):
        stage_mask = labels == stage_idx
        if stage_mask.sum() < 5:
            continue

        stage_trajectories = []
        for i in np.where(stage_mask)[0][:100]:  # cap at 100 for speed
            epoch_1d = epochs[i, 0, :]
            try:
                cloud, _ = time_delay_embedding(epoch_1d, dimension=4)
                cloud_torus = project_to_clifford_torus(cloud)
                stage_trajectories.append(cloud_torus)
            except Exception:
                continue

        if len(stage_trajectories) >= 3:
            ref_mean, ref_cov = compute_reference_spd(stage_trajectories)
            references[stage_name] = (ref_mean, ref_cov)
            print(f"    {stage_name}: {len(stage_trajectories)} epochs → "
                  f"ref built (tr(Σ)={np.trace(ref_cov):.2f})")

    return references


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def run_pipeline(
    psg_path: Path,
    hyp_path: Path,
    use_tda: bool = True,
    output_dir: Path | None = None,
):
    t_start = time.time()

    print("\n" + "═" * 60)
    print("  NEUROSPIRAL — Tesseract-Wasserstein Pipeline")
    print("  Clifford torus geometry + Bures-Wasserstein on SPD manifold")
    print("═" * 60)

    # ── Load & preprocess ────────────────────────────────────
    print("\n[1/7] Loading EDF...")
    record = load_sleep_edf(
        psg_path, hyp_path,
        channels=["EEG Fpz-Cz"],
        label_mapping=LABEL_MAPPING,
    )
    result = preprocess_raw(record.raw, 0.5, 30.0, 100.0, {
        "n_components": 10, "method": "fastica",
        "max_iter": 500, "random_state": 42, "eog_threshold": 0.85,
    })
    record.raw = result.raw
    sfreq = result.raw.info["sfreq"]

    epochs, labels, label_names = extract_epochs_from_annotations(record)
    quality = compute_epoch_quality(epochs, sfreq)
    epochs, labels = epochs[quality], labels[quality]

    print(f"  {epochs.shape[0]} epochs × {sfreq} Hz")
    for i, name in enumerate(label_names):
        count = np.sum(labels == i)
        print(f"    {name:>4}: {count:>4} ({count/len(labels)*100:.1f}%)")

    # ── Build reference states ───────────────────────────────
    print("\n[2/7] Building reference SPD covariances per stage...")
    references = build_reference_states(epochs, labels, label_names, sfreq)

    # Use N3 reference for BW distance features (the glymphatic target)
    n3_ref_mean, n3_ref_cov = None, None
    if "N3" in references:
        n3_ref_mean, n3_ref_cov = references["N3"]
        print(f"\n  N3 reference built — this is the glymphatic target")
        print(f"    Mean position: [{', '.join(f'{v:.3f}' for v in n3_ref_mean)}]")
        print(f"    Covariance trace: {np.trace(n3_ref_cov):.4f}")

    # ── Compute inter-stage BW distances ─────────────────────
    if len(references) >= 2:
        print("\n[3/7] Inter-stage Bures-Wasserstein distances...")
        stage_list = sorted(references.keys())
        print(f"\n  {'':>6}", end="")
        for s in stage_list:
            print(f"  {s:>6}", end="")
        print()
        for s1 in stage_list:
            print(f"  {s1:>6}", end="")
            m1, c1 = references[s1]
            for s2 in stage_list:
                m2, c2 = references[s2]
                d = bures_wasserstein(c1, c2)
                print(f"  {d:>6.3f}", end="")
            print()
        print("\n  (Lower = more similar covariance geometry)")
    else:
        print("\n[3/7] Skipping BW distance matrix (insufficient stages)")

    # ── Extract all features ─────────────────────────────────
    print(f"\n[4/7] Extracting features (spectral + TDA + tesseract + BW)...")
    t_feat = time.time()

    all_features = []
    for i in range(epochs.shape[0]):
        if i % 100 == 0:
            elapsed = time.time() - t_feat
            rate = i / (elapsed + 1e-6)
            eta = (epochs.shape[0] - i) / (rate + 1e-6)
            print(f"    {i:>5}/{epochs.shape[0]} ({i/epochs.shape[0]*100:5.1f}%) "
                  f"[{elapsed:.0f}s, ~{eta:.0f}s left]")

        epoch_1d = epochs[i, 0, :]
        feats = extract_full_features(
            epoch_1d, sfreq,
            use_tda=use_tda,
            reference_mean=n3_ref_mean,
            reference_cov=n3_ref_cov,
        )
        all_features.append(feats)

    # Build aligned feature matrix
    all_keys = sorted(set().union(*[f.keys() for f in all_features]))
    X = np.array([[f.get(k, 0.0) for k in all_keys] for f in all_features])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    feat_elapsed = time.time() - t_feat
    print(f"\n  ✓ {X.shape[0]} × {X.shape[1]} features in {feat_elapsed:.1f}s")

    # Feature type breakdown
    n_spec = sum(1 for k in all_keys if k.startswith("spec_"))
    n_tda = sum(1 for k in all_keys if k.startswith("H") and "_" in k)
    n_tess = sum(1 for k in all_keys if k.startswith("tess_"))
    n_bw = sum(1 for k in all_keys if k.startswith("bw_"))
    n_other = X.shape[1] - n_spec - n_tda - n_tess - n_bw
    print(f"    Spectral: {n_spec} | TDA: {n_tda} | Tesseract: {n_tess} | "
          f"Bures-Wasserstein: {n_bw} | Other: {n_other}")

    # ── Vertex analysis ──────────────────────────────────────
    print(f"\n[5/7] Tesseract vertex analysis per stage...")

    for stage_idx, stage_name in enumerate(label_names):
        mask = labels == stage_idx
        if mask.sum() < 5:
            continue

        # Get dominant vertex for this stage
        tess_dom_key = [k for k in all_keys if k == "tess_dominant_vertex"]
        if tess_dom_key:
            col = all_keys.index(tess_dom_key[0])
            dom_vertices = X[mask, col].astype(int)
            unique, counts = np.unique(dom_vertices, return_counts=True)
            top_v = unique[np.argmax(counts)]
            top_pct = counts.max() / mask.sum() * 100
            print(f"    {stage_name:>4} → dominant vertex V{top_v:02d} "
                  f"({top_pct:.0f}% of epochs), "
                  f"code={VERTICES[top_v].astype(int).tolist()}")

    # ── Classification ───────────────────────────────────────
    print(f"\n[6/7] Training classifier (5-fold stratified CV)...")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_s, labels, cv=cv)
    y_proba = cross_val_predict(clf, X_s, labels, cv=cv, method="predict_proba")

    f1_macro = f1_score(labels, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(labels, y_pred)

    print(f"\n  F1-macro: {f1_macro:.3f}")
    print(f"  Cohen's κ: {kappa:.3f}")

    print(f"\n  Classification report:")
    report = classification_report(
        labels, y_pred, target_names=label_names, digits=3, zero_division=0,
    )
    for line in report.split("\n"):
        print(f"    {line}")

    # N3-specific
    if "N3" in label_names:
        n3_idx = label_names.index("N3")
        n3_true = (labels == n3_idx).astype(int)
        n3_pred = (y_pred == n3_idx).astype(int)

        n3_p = precision_score(n3_true, n3_pred, zero_division=0)
        n3_r = recall_score(n3_true, n3_pred, zero_division=0)
        n3_f1 = f1_score(n3_true, n3_pred, zero_division=0)

        print(f"\n  ┌────────────────────────────────────────────┐")
        print(f"  │  N3 (Glymphatic Target)                    │")
        print(f"  │  Precision: {n3_p:.3f}                           │")
        print(f"  │  Recall:    {n3_r:.3f}                           │")
        print(f"  │  F1:        {n3_f1:.3f}                           │")
        print(f"  └────────────────────────────────────────────┘")

    # ── Feature importance & ablation ────────────────────────
    print(f"\n[7/7] Feature importance by module...")

    clf.fit(X_s, labels)
    importances = clf.feature_importances_

    # Group by module
    module_imp = defaultdict(float)
    for i, key in enumerate(all_keys):
        if key.startswith("spec_"):
            module_imp["Spectral"] += importances[i]
        elif key.startswith("H") and "_" in key:
            module_imp["TDA (homology)"] += importances[i]
        elif key.startswith("tess_"):
            module_imp["Tesseract geometry"] += importances[i]
        elif key.startswith("bw_"):
            module_imp["Bures-Wasserstein"] += importances[i]
        else:
            module_imp["Other"] += importances[i]

    total = sum(module_imp.values())
    print(f"\n  {'Module':<25} {'Importance':>12} {'Share':>8}")
    print(f"  {'─'*25} {'─'*12} {'─'*8}")
    for mod, imp in sorted(module_imp.items(), key=lambda x: -x[1]):
        bar = "▓" * int(imp / total * 40)
        print(f"  {mod:<25} {imp:>12.4f} {imp/total*100:>7.1f}%  {bar}")

    # Top 15 individual features
    print(f"\n  Top 15 features:")
    top_idx = np.argsort(importances)[::-1][:15]
    for idx in top_idx:
        print(f"    {all_keys[idx]:<35} {importances[idx]:.4f}")

    # ── Save ─────────────────────────────────────────────────
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save reference states for deployment
        ref_data = {}
        for stage, (m, c) in references.items():
            ref_data[f"{stage}_mean"] = m
            ref_data[f"{stage}_cov"] = c

        np.savez_compressed(
            output_dir / "tesseract_pipeline_results.npz",
            features=X,
            labels=labels,
            predictions=y_pred,
            probabilities=y_proba,
            label_names=label_names,
            feature_names=all_keys,
            **ref_data,
        )
        print(f"\n  💾 Saved to {output_dir}/")

    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  F1-macro={f1_macro:.3f} | κ={kappa:.3f} | "
          f"N3-F1={n3_f1 if 'N3' in label_names else 'N/A':.3f}")
    print(f"{'═'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="NeuroSpiral — Tesseract-Wasserstein Pipeline"
    )
    parser.add_argument("--psg", type=Path)
    parser.add_argument("--hyp", type=Path)
    parser.add_argument("--download-sample", action="store_true")
    parser.add_argument("--no-tda", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data/results/tesseract")
    args = parser.parse_args()

    if args.download_sample:
        print("📥 Downloading Sleep-EDF sample...")
        psg, hyp = download_sample(PROJECT_ROOT / "data/raw")
    elif args.psg and args.hyp:
        psg, hyp = args.psg, args.hyp
    else:
        parser.error("Use --download-sample or provide --psg/--hyp")
        return

    run_pipeline(psg, hyp, use_tda=not args.no_tda, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
