#!/usr/bin/env python3
"""NeuroSpiral — Transfer Validation Pipeline.

The core idea: learn geometric signatures from clinical EEG (ground truth)
and verify they hold in Apple Watch data (noisy consumer sensor).

    STEP 1: Train on PhysioNet EEG
      → Learn which tesseract vertices correspond to which sleep stages
      → Build reference SPD covariances per stage
      → Establish BW distance thresholds

    STEP 2: Load Apple Watch export CSV
      → The CSV already contains torus projections (θ, φ) and vertex
        assignments computed by the Swift SpiralGeometry package
      → Also contains Apple Watch sleep stage labels (W/CORE/DEEP/REM)

    STEP 3: Compare
      → Do the EEG-derived vertex→stage mappings agree with Apple Watch?
      → Are the BW distances between stages preserved?
      → What is the classification accuracy using EEG-trained geometry
        applied to wearable data?

    STEP 4: Report
      → Transfer agreement score
      → Which geometric features generalize, which don't
      → Recommendations for the app

Usage:
    # Full pipeline: train on EEG + validate on Apple Watch
    python scripts/transfer_validate.py \
        --eeg-dir data/raw \
        --watch-csv data/watch/neurospiral_export_30d.csv

    # EEG training only (saves reference geometry)
    python scripts/transfer_validate.py --eeg-dir data/raw --train-only

    # Watch validation only (loads saved reference geometry)
    python scripts/transfer_validate.py \
        --watch-csv data/watch/neurospiral_export_30d.csv \
        --load-reference data/results/transfer/eeg_reference.npz
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.edf_loader import load_sleep_edf, extract_epochs_from_annotations
from src.preprocessing.pipeline import preprocess_raw, compute_epoch_quality
from src.features.takens import time_delay_embedding
from src.geometry.tesseract import (
    VERTICES,
    project_to_clifford_torus,
    extract_tesseract_features,
    nearest_vertex_idx,
    to_torus_angles,
)
from src.geometry.wasserstein import (
    trajectory_to_spd,
    compute_reference_spd,
    bures_wasserstein,
)

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
    "delta": (0.5, 4.0), "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0), "sigma": (12.0, 15.0), "beta": (15.0, 30.0),
}

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"

# Map Apple Watch stages to EEG stages for comparison
WATCH_TO_EEG_MAP = {
    "W": "W",
    "CORE": "N2",      # Apple Watch "core" ≈ N1+N2
    "DEEP": "N3",       # Apple Watch "deep" ≈ N3
    "REM": "REM",
    "UNKNOWN": None,
}


# ══════════════════════════════════════════════════════════════
# STEP 1: Train geometric reference from EEG
# ══════════════════════════════════════════════════════════════

def train_eeg_reference(
    eeg_dir: Path,
    n_subjects: int = 5,
) -> dict:
    """Train geometric reference from PhysioNet EEG data.

    Learns:
    - vertex_stage_map: which vertices belong to which stages
    - stage_spd: reference SPD covariance per stage
    - stage_bw_matrix: BW distances between stages
    - vertex_histogram_per_stage: how often each vertex appears per stage
    """
    import urllib.request

    subjects = [
        ("SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf"),
        ("SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf"),
        ("SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf"),
        ("SC4012E0-PSG.edf", "SC4012EC-Hypnogram.edf"),
        ("SC4021E0-PSG.edf", "SC4021EH-Hypnogram.edf"),
    ][:n_subjects]

    eeg_dir.mkdir(parents=True, exist_ok=True)

    # Download if needed
    for psg, hyp in subjects:
        for fname in [psg, hyp]:
            fpath = eeg_dir / fname
            if not fpath.exists():
                print(f"    ↓ {fname}...")
                try:
                    urllib.request.urlretrieve(f"{BASE_URL}/{fname}", fpath)
                except Exception as e:
                    print(f"    ✗ {fname}: {e}")

    # Process each subject
    all_vertex_assignments = defaultdict(list)  # stage → list of vertex indices
    all_trajectories = defaultdict(list)         # stage → list of trajectory windows
    all_torus_angles = defaultdict(list)         # stage → list of (θ, φ)

    stage_names = None

    for psg_name, hyp_name in subjects:
        psg_path = eeg_dir / psg_name
        hyp_path = eeg_dir / hyp_name
        if not psg_path.exists() or not hyp_path.exists():
            continue

        sid = psg_name.split("-")[0]
        print(f"\n  Processing {sid}...")

        try:
            record = load_sleep_edf(psg_path, hyp_path,
                                     channels=["EEG Fpz-Cz"],
                                     label_mapping=LABEL_MAPPING)
            result = preprocess_raw(record.raw, 0.5, 30.0, 100.0, {
                "n_components": 10, "method": "fastica",
                "max_iter": 500, "random_state": 42, "eog_threshold": 0.85,
            })
            record.raw = result.raw
            sfreq = result.raw.info["sfreq"]
            epochs, labels, names = extract_epochs_from_annotations(record)
            quality = compute_epoch_quality(epochs, sfreq)
            epochs, labels = epochs[quality], labels[quality]

            if stage_names is None:
                stage_names = names

            for i in range(len(epochs)):
                epoch_1d = epochs[i, 0, :]
                stage = names[labels[i]]

                try:
                    cloud, _ = time_delay_embedding(epoch_1d, dimension=4)
                    cloud_torus = project_to_clifford_torus(cloud)

                    # Vertex assignment
                    vidx = nearest_vertex_idx(cloud_torus)
                    # Most common vertex in this epoch
                    dominant = int(Counter(vidx).most_common(1)[0][0])
                    all_vertex_assignments[stage].append(dominant)

                    # Trajectory for SPD computation
                    all_trajectories[stage].append(cloud_torus)

                    # Torus angles (mean per epoch)
                    angles = to_torus_angles(cloud_torus)
                    mean_theta = float(np.mean(angles[:, 0]))
                    mean_phi = float(np.mean(angles[:, 1]))
                    all_torus_angles[stage].append((mean_theta, mean_phi))
                except Exception:
                    continue

            print(f"    ✓ {len(epochs)} epochs")
        except Exception as e:
            print(f"    ✗ {e}")
            continue

    if stage_names is None:
        raise ValueError("No subjects processed successfully")

    # Build reference geometry
    print("\n  Building reference geometry...")

    # 1. Vertex→stage mapping (which vertices dominate each stage)
    vertex_histogram = {}
    vertex_stage_map = {}

    for stage in stage_names:
        assignments = all_vertex_assignments.get(stage, [])
        if not assignments:
            continue

        hist = Counter(assignments)
        vertex_histogram[stage] = dict(hist)

        # Top 3 vertices for this stage
        top_vertices = [v for v, _ in hist.most_common(3)]
        vertex_stage_map[stage] = top_vertices

        total = len(assignments)
        print(f"    {stage}: top vertices = "
              + ", ".join(f"V{v:02d}({hist[v]/total*100:.0f}%)" for v in top_vertices))

    # 2. Reference SPD per stage
    stage_spd = {}
    for stage in stage_names:
        trajs = all_trajectories.get(stage, [])
        if len(trajs) >= 3:
            ref_mean, ref_cov = compute_reference_spd(trajs[:100])
            stage_spd[stage] = (ref_mean, ref_cov)

    # 3. BW distance matrix between stages
    bw_matrix = {}
    stages_with_spd = [s for s in stage_names if s in stage_spd]
    for s1 in stages_with_spd:
        for s2 in stages_with_spd:
            _, c1 = stage_spd[s1]
            _, c2 = stage_spd[s2]
            bw_matrix[(s1, s2)] = bures_wasserstein(c1, c2)

    # Print BW matrix
    if bw_matrix:
        print(f"\n  Bures-Wasserstein distance matrix (EEG reference):")
        print(f"  {'':>6}", end="")
        for s in stages_with_spd:
            print(f"  {s:>6}", end="")
        print()
        for s1 in stages_with_spd:
            print(f"  {s1:>6}", end="")
            for s2 in stages_with_spd:
                print(f"  {bw_matrix.get((s1,s2), 0):>6.3f}", end="")
            print()

    return {
        "stage_names": stage_names,
        "vertex_stage_map": vertex_stage_map,
        "vertex_histogram": vertex_histogram,
        "stage_spd": stage_spd,
        "bw_matrix": bw_matrix,
        "torus_angles": dict(all_torus_angles),
    }


# ══════════════════════════════════════════════════════════════
# STEP 2: Load Apple Watch CSV
# ══════════════════════════════════════════════════════════════

def load_watch_data(csv_path: Path) -> pd.DataFrame:
    """Load Apple Watch export CSV."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    print(f"  Loaded {len(df)} samples from {df['night_id'].nunique()} nights")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Stages: {dict(df['sleep_stage'].value_counts())}")
    return df


# ══════════════════════════════════════════════════════════════
# STEP 3: Transfer validation
# ══════════════════════════════════════════════════════════════

def validate_transfer(
    eeg_ref: dict,
    watch_df: pd.DataFrame,
):
    """Compare EEG-derived geometry with Apple Watch data."""

    print(f"\n{'═'*60}")
    print(f"  Transfer Validation: EEG geometry → Apple Watch")
    print(f"{'═'*60}")

    # ── 3a. Vertex agreement ─────────────────────────────
    print(f"\n[A] Vertex→stage agreement")
    print(f"    Do the same tesseract vertices light up for the same stages?")

    vertex_stage_map = eeg_ref["vertex_stage_map"]
    eeg_stages = eeg_ref["stage_names"]

    # For each Apple Watch stage, check which vertices appear
    watch_vertex_hist = defaultdict(lambda: Counter())
    for _, row in watch_df.iterrows():
        watch_stage = row["sleep_stage"]
        eeg_stage = WATCH_TO_EEG_MAP.get(watch_stage)
        if eeg_stage is None:
            continue
        watch_vertex_hist[eeg_stage][int(row["vertex_index"])] += 1

    agreement_scores = {}
    print(f"\n    {'Stage':<8} {'EEG top vertices':<25} {'Watch top vertices':<25} {'Overlap'}")
    print(f"    {'─'*8} {'─'*25} {'─'*25} {'─'*8}")

    for stage in eeg_stages:
        eeg_top = set(vertex_stage_map.get(stage, []))
        watch_top_counter = watch_vertex_hist.get(stage, Counter())

        if not watch_top_counter:
            continue

        watch_top = set(v for v, _ in watch_top_counter.most_common(3))

        overlap = eeg_top & watch_top
        overlap_pct = len(overlap) / max(len(eeg_top), 1) * 100
        agreement_scores[stage] = overlap_pct

        eeg_str = ", ".join(f"V{v:02d}" for v in sorted(eeg_top))
        watch_str = ", ".join(f"V{v:02d}" for v in sorted(watch_top))
        symbol = "✓" if overlap_pct >= 33 else "✗"

        print(f"    {stage:<8} {eeg_str:<25} {watch_str:<25} {overlap_pct:5.0f}% {symbol}")

    if agreement_scores:
        mean_agreement = np.mean(list(agreement_scores.values()))
        print(f"\n    Mean vertex agreement: {mean_agreement:.0f}%")
        if mean_agreement >= 50:
            print(f"    ✓ Strong transfer — EEG geometry holds in wearable data")
        elif mean_agreement >= 25:
            print(f"    ○ Partial transfer — some vertices generalize")
        else:
            print(f"    ✗ Weak transfer — geometry differs between EEG and wearable")

    # ── 3b. Torus angle distribution comparison ──────────
    print(f"\n[B] Torus angle distributions (θ, φ)")
    print(f"    Are the stages in the same regions of the torus?")

    eeg_angles = eeg_ref.get("torus_angles", {})

    for stage in eeg_stages:
        eeg_a = eeg_angles.get(stage, [])
        watch_mask = watch_df["sleep_stage"].map(WATCH_TO_EEG_MAP) == stage
        watch_rows = watch_df[watch_mask]

        if len(eeg_a) < 5 or len(watch_rows) < 5:
            continue

        eeg_theta = np.array([a[0] for a in eeg_a])
        eeg_phi = np.array([a[1] for a in eeg_a])
        watch_theta = watch_rows["torus_theta"].values
        watch_phi = watch_rows["torus_phi"].values

        # Circular mean comparison
        eeg_mean_theta = np.arctan2(np.mean(np.sin(eeg_theta)), np.mean(np.cos(eeg_theta)))
        watch_mean_theta = np.arctan2(np.mean(np.sin(watch_theta)), np.mean(np.cos(watch_theta)))
        eeg_mean_phi = np.arctan2(np.mean(np.sin(eeg_phi)), np.mean(np.cos(eeg_phi)))
        watch_mean_phi = np.arctan2(np.mean(np.sin(watch_phi)), np.mean(np.cos(watch_phi)))

        delta_theta = abs(np.arctan2(np.sin(eeg_mean_theta - watch_mean_theta),
                                      np.cos(eeg_mean_theta - watch_mean_theta)))
        delta_phi = abs(np.arctan2(np.sin(eeg_mean_phi - watch_mean_phi),
                                    np.cos(eeg_mean_phi - watch_mean_phi)))

        print(f"    {stage}: Δθ = {np.degrees(delta_theta):5.1f}°, "
              f"Δφ = {np.degrees(delta_phi):5.1f}°"
              f" {'(close)' if delta_theta < 0.5 and delta_phi < 0.5 else ''}")

    # ── 3c. BW distances ────────────────────────────────
    print(f"\n[C] Bures-Wasserstein distances")
    print(f"    Is the covariance geometry preserved?")

    eeg_bw = eeg_ref.get("bw_matrix", {})
    stage_spd = eeg_ref.get("stage_spd", {})

    # Compute watch BW matrix
    watch_spd = {}
    for stage in eeg_stages:
        watch_mask = watch_df["sleep_stage"].map(WATCH_TO_EEG_MAP) == stage
        watch_rows = watch_df[watch_mask]
        if len(watch_rows) < 10:
            continue
        theta = watch_rows["torus_theta"].values
        phi = watch_rows["torus_phi"].values
        R = np.sqrt(2)
        points = np.column_stack([
            R * np.cos(theta), R * np.sin(theta),
            R * np.cos(phi), R * np.sin(phi)
        ])
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = (centered.T @ centered) / (len(points) - 1) + 1e-8 * np.eye(4)
        watch_spd[stage] = (mean, cov)

    if len(watch_spd) >= 2:
        print(f"\n    {'Pair':<12} {'EEG BW':>8} {'Watch BW':>9} {'Ratio':>7}")
        print(f"    {'─'*12} {'─'*8} {'─'*9} {'─'*7}")

        ratios = []
        for s1 in eeg_stages:
            for s2 in eeg_stages:
                if s1 >= s2:
                    continue
                eeg_d = eeg_bw.get((s1, s2), None)
                if eeg_d is None or s1 not in watch_spd or s2 not in watch_spd:
                    continue
                _, c1 = watch_spd[s1]
                _, c2 = watch_spd[s2]
                watch_d = bures_wasserstein(c1, c2)
                ratio = watch_d / (eeg_d + 1e-10)
                ratios.append(ratio)
                print(f"    {s1}-{s2:<8} {eeg_d:>8.3f} {watch_d:>9.3f} {ratio:>7.2f}×")

        if ratios:
            mean_ratio = np.mean(ratios)
            print(f"\n    Mean ratio (watch/EEG): {mean_ratio:.2f}×")
            if 0.3 < mean_ratio < 3.0:
                print(f"    ✓ BW geometry scales proportionally — transfer viable")
            else:
                print(f"    ⚠ BW geometry differs significantly — need recalibration")

    # ── 3d. Classification using EEG-trained rules ───────
    print(f"\n[D] Classification: EEG geometry → Watch predictions")

    # Use vertex→stage map from EEG to predict Watch stages
    # For each Watch sample, look up its vertex in the EEG map
    vertex_to_eeg_stage = {}
    for stage, vertices in vertex_stage_map.items():
        for v in vertices:
            if v not in vertex_to_eeg_stage:
                vertex_to_eeg_stage[v] = stage

    valid_mask = watch_df["sleep_stage"].map(WATCH_TO_EEG_MAP).notna()
    valid_df = watch_df[valid_mask].copy()

    if len(valid_df) > 0:
        y_true_eeg = valid_df["sleep_stage"].map(WATCH_TO_EEG_MAP).values
        y_pred_eeg = valid_df["vertex_index"].map(
            lambda v: vertex_to_eeg_stage.get(int(v), "W")
        ).values

        # Filter to stages that exist in both
        common_stages = sorted(set(y_true_eeg) & set(y_pred_eeg))
        if common_stages:
            mask = np.isin(y_true_eeg, common_stages) & np.isin(y_pred_eeg, common_stages)
            y_t = y_true_eeg[mask]
            y_p = y_pred_eeg[mask]

            if len(y_t) > 0:
                print(f"\n    Transfer classification report "
                      f"(EEG geometry applied to Watch data):")
                report = classification_report(
                    y_t, y_p, labels=common_stages, digits=3, zero_division=0,
                )
                for line in report.split("\n"):
                    print(f"      {line}")

                kappa = cohen_kappa_score(y_t, y_p)
                f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
                print(f"\n    Transfer F1-macro: {f1:.3f}")
                print(f"    Transfer κ: {kappa:.3f}")

                # N3/DEEP specific
                n3_true = (y_t == "N3").astype(int)
                n3_pred = (y_p == "N3").astype(int)
                n3_agree = np.mean(n3_true == n3_pred)
                print(f"\n    N3/DEEP agreement: {n3_agree:.1%}")

    # ── Summary ──────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Transfer validation complete")
    print(f"  EEG-trained geometry tested on Apple Watch data")
    print(f"{'═'*60}")

    return agreement_scores


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NeuroSpiral — Transfer Validation (EEG → Apple Watch)"
    )
    parser.add_argument("--eeg-dir", type=Path, default=PROJECT_ROOT / "data/raw",
                        help="Directory with EEG .edf files")
    parser.add_argument("--watch-csv", type=Path,
                        help="Apple Watch export CSV from SpiralGeometry")
    parser.add_argument("--n-subjects", type=int, default=5)
    parser.add_argument("--train-only", action="store_true",
                        help="Only train EEG reference, skip Watch validation")
    parser.add_argument("--load-reference", type=Path,
                        help="Load pre-trained EEG reference instead of training")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data/results/transfer")
    args = parser.parse_args()

    t_start = time.time()

    print("\n" + "═" * 60)
    print("  NeuroSpiral — Transfer Validation")
    print("  EEG clinical data ↔ Apple Watch wearable data")
    print("═" * 60)

    # Step 1: EEG reference
    if args.load_reference:
        print(f"\n[1/3] Loading pre-trained reference from {args.load_reference}...")
        # TODO: implement npz loading
        raise NotImplementedError("Reference loading not yet implemented — use --eeg-dir")
    else:
        print(f"\n[1/3] Training EEG reference ({args.n_subjects} subjects)...")
        eeg_ref = train_eeg_reference(args.eeg_dir, args.n_subjects)

    # Save reference
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Save what we can as npz
    save_dict = {
        "stage_names": eeg_ref["stage_names"],
    }
    for stage, (mean, cov) in eeg_ref.get("stage_spd", {}).items():
        save_dict[f"spd_mean_{stage}"] = mean
        save_dict[f"spd_cov_{stage}"] = cov
    np.savez(args.output_dir / "eeg_reference.npz", **save_dict)
    print(f"\n  ✓ Reference saved to {args.output_dir}/eeg_reference.npz")

    if args.train_only:
        print(f"\n  --train-only: skipping Watch validation")
        print(f"  Next: export Watch data and run with --watch-csv")
        return

    # Step 2: Watch data
    if args.watch_csv is None:
        print(f"\n[2/3] No Apple Watch CSV provided.")
        print(f"  To export from the app:")
        print(f"    1. Open Spiral Journey → DNA Insights → NeuroSpiral 4D")
        print(f"    2. Use the HealthKitExporter to generate CSV")
        print(f"    3. Run: python scripts/transfer_validate.py \\")
        print(f"         --watch-csv path/to/neurospiral_export_30d.csv")
        print(f"\n  Meanwhile, the EEG reference is saved and ready.")
        return

    print(f"\n[2/3] Loading Apple Watch data...")
    watch_df = load_watch_data(args.watch_csv)

    # Step 3: Validate
    print(f"\n[3/3] Running transfer validation...")
    validate_transfer(eeg_ref, watch_df)

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed:.0f}s\n")


if __name__ == "__main__":
    main()
