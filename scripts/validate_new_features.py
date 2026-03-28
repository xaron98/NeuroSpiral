#!/usr/bin/env python3
"""Validate 10 new torus features via conditional mutual information.

Steps:
  0. Download Sleep-EDF data (18 subjects, sleep-cassette) if not present.
  1. Load each subject: PSG (EEG Fpz-Cz), hypnogram, 30s epochs.
  2. Extract delta power + 10 torus v2 features per epoch.
  3. Compute conditional MI: CMI = MI(feature+delta; stage) - MI(delta; stage).
  4. Permutation test (1000 perms) for significance.
  5. Print summary table and save results/new_features_cmi.json.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import mne
from scipy.signal import welch
from sklearn.metrics import mutual_info_score

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.takens import time_delay_embedding
from src.features.torus_features_v2 import extract_torus_features_v2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SFREQ = 100  # Sleep-EDF Fpz-Cz sampling rate
EPOCH_SEC = 30
EPOCH_SAMPLES = SFREQ * EPOCH_SEC
N_PERM = 1000
N_CMI_BINS = 10
RNG = np.random.default_rng(42)

SUBJECTS = [
    "SC4001", "SC4002", "SC4011", "SC4012",
    "SC4021", "SC4022", "SC4031", "SC4041",
    "SC4042", "SC4051", "SC4052", "SC4061",
    "SC4062", "SC4071", "SC4072", "SC4081",
    "SC4091", "SC4092",
]

HYPNOGRAM_NAMES = {
    "SC4001": "SC4001EC", "SC4002": "SC4002EC",
    "SC4011": "SC4011EH", "SC4012": "SC4012EC",
    "SC4021": "SC4021EH", "SC4022": "SC4022EJ",
    "SC4031": "SC4031EC", "SC4041": "SC4041EC",
    "SC4042": "SC4042EC", "SC4051": "SC4051EC",
    "SC4052": "SC4052EC", "SC4061": "SC4061EC",
    "SC4062": "SC4062EC", "SC4071": "SC4071EC",
    "SC4072": "SC4072EH", "SC4081": "SC4081EC",
    "SC4091": "SC4091EC", "SC4092": "SC4092EC",
}

STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": None,
    "Movement time": None,
}

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"

FEATURE_NAMES = [
    "angular_acceleration",
    "geodesic_distance",
    "angular_entropy",
    "theta_harmonic_1",
    "theta_harmonic_2",
    "theta_harmonic_3",
    "theta_harmonic_4",
    "torus_curvature",
    "angular_range",
]

# ---------------------------------------------------------------------------
# Step 0: Download Sleep-EDF data
# ---------------------------------------------------------------------------

def download_sleep_edf(data_dir: Path) -> None:
    """Download 18 subjects from PhysioNet sleep-cassette if not present."""
    data_dir.mkdir(parents=True, exist_ok=True)

    for subj in SUBJECTS:
        psg_remote = f"{subj}E0-PSG.edf"
        psg_local = data_dir / psg_remote

        hyp_base = HYPNOGRAM_NAMES[subj]
        hyp_remote = f"{hyp_base}-Hypnogram.edf"
        hyp_local = data_dir / f"{subj}E0-Hypnogram.edf"  # save with E0 suffix

        for remote_name, local_path in [
            (psg_remote, psg_local),
            (hyp_remote, hyp_local),
        ]:
            if local_path.exists() and local_path.stat().st_size > 0:
                continue
            url = f"{BASE_URL}/{remote_name}"
            print(f"  Downloading {remote_name} -> {local_path.name} ...")
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                print(f"  ERROR downloading {url}: {e}")
                if local_path.exists():
                    local_path.unlink()
                continue

            if not local_path.exists() or local_path.stat().st_size == 0:
                print(f"  WARNING: {local_path.name} is empty or missing after download")
                if local_path.exists():
                    local_path.unlink()

    # Verify
    n_ok = 0
    for subj in SUBJECTS:
        psg = data_dir / f"{subj}E0-PSG.edf"
        hyp = data_dir / f"{subj}E0-Hypnogram.edf"
        if psg.exists() and psg.stat().st_size > 0 and hyp.exists() and hyp.stat().st_size > 0:
            n_ok += 1
    print(f"  {n_ok}/{len(SUBJECTS)} subjects downloaded and verified.\n")


# ---------------------------------------------------------------------------
# Step 1: Load subject data
# ---------------------------------------------------------------------------

def load_subject(data_dir: Path, subj: str) -> tuple[np.ndarray, list[str]] | None:
    """Load PSG + hypnogram, return (epochs_array, stage_labels).

    Returns None if files are missing or loading fails.
    """
    psg_path = data_dir / f"{subj}E0-PSG.edf"
    hyp_path = data_dir / f"{subj}E0-Hypnogram.edf"

    if not psg_path.exists() or not hyp_path.exists():
        print(f"  Skipping {subj}: files not found")
        return None

    # Load PSG
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)

    # Pick EEG Fpz-Cz
    if "EEG Fpz-Cz" not in raw.ch_names:
        print(f"  Skipping {subj}: EEG Fpz-Cz not found (channels: {raw.ch_names})")
        return None
    raw.pick(["EEG Fpz-Cz"])
    raw.filter(0.5, 30.0, verbose=False)

    actual_sfreq = raw.info["sfreq"]
    if actual_sfreq != SFREQ:
        raw.resample(SFREQ, verbose=False)

    signal = raw.get_data()[0]  # shape: (n_samples,)
    total_seconds = len(signal) / SFREQ

    # Load hypnogram annotations
    annots = mne.read_annotations(str(hyp_path))

    # Build epoch labels
    n_epochs = int(total_seconds // EPOCH_SEC)
    epoch_labels: list[str | None] = [None] * n_epochs

    for onset, duration, desc in zip(annots.onset, annots.duration, annots.description):
        # CRITICAL: convert np.str_ to str for dict lookup
        stage = STAGE_MAP.get(str(desc).strip())
        if stage is None:
            continue
        start_epoch = int(onset // EPOCH_SEC)
        n_dur_epochs = int(duration // EPOCH_SEC)
        for e in range(start_epoch, min(start_epoch + n_dur_epochs, n_epochs)):
            epoch_labels[e] = stage

    # Build epoch arrays, reject artifacts, filter out unlabeled epochs
    epochs_list = []
    labels_list = []
    for i in range(n_epochs):
        if epoch_labels[i] is None:
            continue
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        if end > len(signal):
            break
        epoch = signal[start:end]
        # Reject ±500 µV (signal is in V from MNE, threshold = 500e-6)
        if np.max(np.abs(epoch)) > 500e-6:
            continue
        epochs_list.append(epoch)
        labels_list.append(epoch_labels[i])

    if len(epochs_list) == 0:
        print(f"  Skipping {subj}: no valid epochs")
        return None

    return np.array(epochs_list), labels_list


# ---------------------------------------------------------------------------
# Step 2: Feature extraction
# ---------------------------------------------------------------------------

def extract_delta_power(epoch: np.ndarray, sfreq: float = SFREQ) -> float:
    """Compute delta band power (0.5–4 Hz) using Welch PSD."""
    freqs, psd = welch(epoch, fs=sfreq, nperseg=min(256, len(epoch)))
    delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
    if not np.any(delta_mask):
        return 0.0
    return float(np.sum(psd[delta_mask]))


def extract_features_for_epoch(epoch: np.ndarray) -> dict[str, float] | None:
    """Extract delta power + 10 torus v2 features for one epoch."""
    delta = extract_delta_power(epoch)

    try:
        embedding, _ = time_delay_embedding(epoch, dimension=4, tau=25)
    except ValueError:
        return None

    torus_feats = extract_torus_features_v2(embedding)
    torus_feats["delta_power"] = delta
    return torus_feats


# ---------------------------------------------------------------------------
# Step 3 & 4: CMI computation and permutation test
# ---------------------------------------------------------------------------

def discretize(values: np.ndarray, n_bins: int = N_CMI_BINS) -> np.ndarray:
    """Discretize into percentile bins."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    # np.searchsorted can produce bin = n_bins for max value; clip to n_bins-1
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return bins


def mi_score(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Mutual information between two discrete label arrays."""
    return float(mutual_info_score(labels_a, labels_b))


def joint_bins(bin_a: np.ndarray, bin_b: np.ndarray) -> np.ndarray:
    """Create joint bins from two bin arrays via Cantor-like pairing."""
    max_b = int(bin_b.max()) + 1
    return bin_a * max_b + bin_b


def compute_cmi(
    feature_vals: np.ndarray,
    delta_vals: np.ndarray,
    stage_labels: np.ndarray,
) -> float:
    """CMI = MI(feature+delta; stage) - MI(delta; stage)."""
    feat_bins = discretize(feature_vals)
    delta_bins = discretize(delta_vals)
    stage_int = stage_labels  # already integer-encoded

    joint = joint_bins(feat_bins, delta_bins)
    mi_joint = mi_score(joint, stage_int)
    mi_delta = mi_score(delta_bins, stage_int)
    return mi_joint - mi_delta


def permutation_test(
    feature_vals: np.ndarray,
    delta_vals: np.ndarray,
    stage_labels: np.ndarray,
    n_perm: int = N_PERM,
) -> tuple[float, float]:
    """Permutation test for CMI significance.

    Returns (observed_cmi, p_value).
    """
    observed = compute_cmi(feature_vals, delta_vals, stage_labels)

    null_dist = np.zeros(n_perm)
    for i in range(n_perm):
        perm_feat = RNG.permutation(feature_vals)
        null_dist[i] = compute_cmi(perm_feat, delta_vals, stage_labels)

    p_value = float(np.mean(null_dist >= observed))
    return observed, p_value


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "sleep-edf"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Download
    print("=" * 60)
    print("Step 0: Downloading Sleep-EDF data")
    print("=" * 60)
    download_sleep_edf(data_dir)

    # Step 1 & 2: Load subjects and extract features
    print("=" * 60)
    print("Step 1-2: Loading subjects and extracting features")
    print("=" * 60)

    all_features: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}
    all_features["delta_power"] = []
    all_stages: list[int] = []

    stage_to_int = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

    n_loaded = 0
    for subj in SUBJECTS:
        print(f"  Processing {subj} ...")
        result = load_subject(data_dir, subj)
        if result is None:
            continue

        epochs, labels = result
        n_loaded += 1
        n_epochs_ok = 0

        for epoch, label in zip(epochs, labels):
            feats = extract_features_for_epoch(epoch)
            if feats is None:
                continue
            for name in FEATURE_NAMES:
                all_features[name].append(feats[name])
            all_features["delta_power"].append(feats["delta_power"])
            all_stages.append(stage_to_int[label])
            n_epochs_ok += 1

        print(f"    {n_epochs_ok} epochs extracted")

    print(f"\n  Total: {n_loaded} subjects, {len(all_stages)} epochs\n")

    if len(all_stages) < 100:
        print("ERROR: Too few epochs for meaningful analysis. Exiting.")
        sys.exit(1)

    # Convert to arrays
    delta_arr = np.array(all_features["delta_power"])
    stage_arr = np.array(all_stages)

    # Step 3-4: CMI + permutation tests
    print("=" * 60)
    print("Steps 3-4: Conditional MI + permutation tests (1000 perms)")
    print("=" * 60)

    results = {}
    for name in FEATURE_NAMES:
        feat_arr = np.array(all_features[name])
        print(f"  {name:30s} ... ", end="", flush=True)
        cmi, pval = permutation_test(feat_arr, delta_arr, stage_arr, N_PERM)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        print(f"CMI={cmi:.6f}  p={pval:.4f}  {sig}")
        results[name] = {
            "cmi": round(cmi, 6),
            "p_value": round(pval, 4),
            "significant_0.05": pval < 0.05,
            "significant_0.01": pval < 0.01,
            "significant_0.001": pval < 0.001,
        }

    # Step 5: Summary table
    print("\n" + "=" * 60)
    print("Step 5: Summary")
    print("=" * 60)
    print(f"{'Feature':<30s} {'CMI':>10s} {'p-value':>10s} {'Sig':>6s}")
    print("-" * 60)
    for name in FEATURE_NAMES:
        r = results[name]
        sig = "***" if r["significant_0.001"] else "**" if r["significant_0.01"] else "*" if r["significant_0.05"] else "n.s."
        print(f"{name:<30s} {r['cmi']:>10.6f} {r['p_value']:>10.4f} {sig:>6s}")

    n_sig = sum(1 for r in results.values() if r["significant_0.05"])
    print(f"\n  {n_sig}/{len(FEATURE_NAMES)} features significant at p<0.05")
    print(f"  Total epochs: {len(all_stages)}")
    print(f"  Subjects: {n_loaded}")

    # Save JSON
    output = {
        "n_subjects": n_loaded,
        "n_epochs": len(all_stages),
        "n_permutations": N_PERM,
        "n_cmi_bins": N_CMI_BINS,
        "features": results,
    }
    out_path = results_dir / "new_features_cmi.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
