#!/usr/bin/env python3
"""Extract multi-scale bilateral geometric features from HMC dataset.

For each subject, loads both EEG channels (C4-M1, C3-M2), extracts
geometric features at three embedding delays (τ=10, 25, 40) per channel,
plus bilateral inter-channel features per delay, plus spectral features
from channel 1 only.

Feature counts:
  Per-channel geometric: 8 features × 2 channels × 3 τ = 48
  Bilateral per delay:   3 features × 3 τ                =  9
  Total geometric:                                        = 57
  Spectral (ch1 only):                                    =  8

Saves results/hmc_features_multiscale.npz containing:
  features_spectral      : (n_epochs, 8)  float32
  features_geometric     : (n_epochs, 57) float32
  stages                 : (n_epochs,)    int8   (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
  subjects               : (n_epochs,)    int32
  feature_names_spectral : list of str
  feature_names_geometric: list of str

Run:  SKIP_DOWNLOAD=1 python3 scripts/extract_features_multiscale.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import mne
from scipy.signal import welch

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.takens import time_delay_embedding
from src.features.torus_features_v2 import extract_torus_features_v2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SFREQ = 100
EPOCH_SEC = 30
EPOCH_SAMPLES = SFREQ * EPOCH_SEC

TAUS = [10, 25, 40]
CHANNELS = ["c4", "c3"]  # C4-M1, C3-M2

HMC_LABELS = {
    "Sleep stage W": "W",
    "Sleep stage N1": "N1",
    "Sleep stage N2": "N2",
    "Sleep stage N3": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": None,
    "0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "REM", "5": None,
    "W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "REM", "R": "REM",
}

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGES)}

# Per-channel feature names (8 per (channel, τ) combination)
PER_CH_FEATURES = [
    "omega1", "torus_curvature", "angular_acceleration",
    "geodesic_distance", "angular_entropy",
    "phase_diff_std_intra", "phase_coherence_intra", "transition_rate",
]

# Bilateral feature names (3 per τ)
BILATERAL_FEATURES = [
    "phase_diff_std_bilateral", "phase_coherence_bilateral",
    "bilateral_omega_diff",
]

SPECTRAL_NAMES = [
    "delta", "theta", "alpha", "sigma", "beta",
    "delta_beta", "hjorth_activity", "hjorth_mobility",
]


def build_feature_names() -> list[str]:
    """Build ordered list of geometric feature names (57 total)."""
    names: list[str] = []
    # Per-channel: grouped by channel, then tau, then feature
    for ch in CHANNELS:
        for tau in TAUS:
            for feat in PER_CH_FEATURES:
                names.append(f"{feat}_{ch}_t{tau}")
    # Bilateral: grouped by tau, then feature
    for tau in TAUS:
        for feat in BILATERAL_FEATURES:
            names.append(f"{feat}_t{tau}")
    return names


GEOM_NAMES = build_feature_names()
N_SPECTRAL = len(SPECTRAL_NAMES)   # 8
N_GEOM = len(GEOM_NAMES)           # 57


# ===================================================================
# Subject discovery + loading (bilateral)
# ===================================================================
def find_subjects(data_dir: Path) -> list[tuple[str, Path, Path]]:
    pairs = []
    for i in range(1, 500):
        sid = f"SN{i:03d}"
        psg = data_dir / f"{sid}.edf"
        hyp = data_dir / f"{sid}_sleepscoring.edf"
        if psg.exists() and hyp.exists():
            if psg.stat().st_size > 1_000_000 and hyp.stat().st_size > 500:
                pairs.append((sid, psg, hyp))
    return pairs


def _find_channel(raw, candidates: list[str]) -> str | None:
    """Find first matching channel name from candidates."""
    for c in candidates:
        if c in raw.ch_names:
            return c
    return None


def load_subject_bilateral(psg_path: Path, hyp_path: Path):
    """Return (ch1_epochs, ch2_epochs, labels) or None.

    ch1 = C4-M1 (right hemisphere), ch2 = C3-M2 (left hemisphere).
    Rejects epochs where either channel exceeds ±500 µV.
    """
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)

    ch1 = _find_channel(raw, ["EEG C4-M1", "C4-M1", "EEG C4"])
    ch2 = _find_channel(raw, ["EEG C3-M2", "C3-M2", "EEG C3"])

    if ch1 is None or ch2 is None:
        return None

    raw.pick([ch1, ch2])
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)
    raw.filter(0.5, 30.0, verbose=False)

    # Resolve channel order after pick
    ch1_idx = raw.ch_names.index(ch1)
    ch2_idx = raw.ch_names.index(ch2)
    data = raw.get_data()
    signal_ch1 = data[ch1_idx]
    signal_ch2 = data[ch2_idx]

    n_ep = int(len(signal_ch1) / SFREQ // EPOCH_SEC)

    # Read annotations
    annots = mne.read_annotations(str(hyp_path))
    labels: list[str | None] = [None] * n_ep
    for onset, dur, desc in zip(annots.onset, annots.duration, annots.description):
        stage = HMC_LABELS.get(str(desc).strip())
        if stage is None:
            continue
        s = int(onset // EPOCH_SEC)
        nd = max(1, int(dur // EPOCH_SEC))
        for e in range(s, min(s + nd, n_ep)):
            labels[e] = stage

    ch1_epochs, ch2_epochs, labels_out = [], [], []
    for i in range(n_ep):
        if labels[i] is None:
            continue
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        if end > len(signal_ch1):
            break
        ep1 = signal_ch1[start:end]
        ep2 = signal_ch2[start:end]
        # Reject if either channel exceeds ±500 µV
        if np.max(np.abs(ep1)) > 500e-6 or np.max(np.abs(ep2)) > 500e-6:
            continue
        ch1_epochs.append(ep1)
        ch2_epochs.append(ep2)
        labels_out.append(labels[i])

    if not ch1_epochs:
        return None
    return np.array(ch1_epochs), np.array(ch2_epochs), labels_out


# ===================================================================
# Feature extraction
# ===================================================================
def _wrap(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


def _extract_channel_geometric(epoch: np.ndarray, tau: int):
    """Extract 8 geometric features for one (channel, τ) combination.

    Returns (features_8, theta_array, omega1) or None.
    theta_array and omega1 are needed for bilateral computation.
    """
    try:
        emb, _ = time_delay_embedding(epoch, dimension=4, tau=tau)
    except ValueError:
        return None
    if emb.shape[0] < 10 or not np.all(np.isfinite(emb)):
        return None

    theta = np.arctan2(emb[:, 1], emb[:, 0])
    phi = np.arctan2(emb[:, 3], emb[:, 2])
    theta_uw = np.unwrap(theta)
    phi_uw = np.unwrap(phi)
    dtheta = np.diff(theta_uw)

    # omega1
    omega1 = float(np.mean(np.abs(dtheta)))

    # Intra-channel phase difference features (θ vs φ within same channel)
    phase_diff = theta_uw - phi_uw
    phase_diff_std_intra = float(np.std(np.diff(phase_diff)))

    diff_w = _wrap(theta - phi)
    mc = float(np.mean(np.cos(diff_w)))
    ms = float(np.mean(np.sin(diff_w)))
    phase_coherence_intra = float(np.sqrt(mc**2 + ms**2))

    # Transition rate (octant changes in 4D embedding)
    signs = (emb >= 0).astype(int)
    verts = signs[:, 0] * 8 + signs[:, 1] * 4 + signs[:, 2] * 2 + signs[:, 3]
    transition_rate = float(np.sum(np.diff(verts) != 0) / max(len(verts) - 1, 1))

    # Extended torus features from v2
    v2 = extract_torus_features_v2(emb)

    features_8 = np.array([
        omega1,
        v2["torus_curvature"],
        v2["angular_acceleration"],
        v2["geodesic_distance"],
        v2["angular_entropy"],
        phase_diff_std_intra,
        phase_coherence_intra,
        transition_rate,
    ])

    return features_8, theta, omega1


def _compute_bilateral(theta_ch1: np.ndarray, theta_ch2: np.ndarray,
                        omega1_ch1: float, omega1_ch2: float) -> np.ndarray:
    """Compute 3 bilateral features for one delay.

    Returns (3,) array: [phase_diff_std_bilateral, phase_coherence_bilateral,
                          bilateral_omega_diff]
    """
    # Align lengths (should be equal, but be safe)
    n = min(len(theta_ch1), len(theta_ch2))
    t1 = theta_ch1[:n]
    t2 = theta_ch2[:n]

    diff = _wrap(t1 - t2)

    # Mean resultant length (bilateral phase coherence)
    mc = float(np.mean(np.cos(diff)))
    ms = float(np.mean(np.sin(diff)))
    R = np.sqrt(mc**2 + ms**2)

    # Circular std of bilateral phase difference
    circ_std = float(np.sqrt(-2 * np.log(max(R, 1e-10))))

    # Bilateral phase coherence
    phase_coherence_bilateral = float(R)

    # Bilateral omega difference
    omega_diff = float(abs(omega1_ch1 - omega1_ch2))

    return np.array([circ_std, phase_coherence_bilateral, omega_diff])


def extract_spectral(epoch: np.ndarray) -> np.ndarray | None:
    """Extract 8 spectral features from a single-channel epoch."""
    freqs, psd = welch(epoch, fs=SFREQ, nperseg=min(256, len(epoch)))
    total = np.trapz(psd, freqs)
    if total < 1e-20:
        return None

    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
             "sigma": (12, 15), "beta": (15, 30)}
    bp = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        bp[name] = float(np.trapz(psd[mask], freqs[mask]) / total)

    delta_beta = bp["delta"] / (bp["beta"] + 1e-10)
    diff1 = np.diff(epoch)
    activity = float(np.var(epoch))
    mobility = float(np.sqrt(np.var(diff1) / (activity + 1e-10)))

    return np.array([
        bp["delta"], bp["theta"], bp["alpha"], bp["sigma"], bp["beta"],
        delta_beta, activity, mobility,
    ])


def extract_epoch_features(ch1_epoch: np.ndarray, ch2_epoch: np.ndarray):
    """Extract spectral (8) + geometric (57) features for one epoch.

    Returns (spectral_8, geometric_57) or None.
    """
    # --- Spectral from channel 1 only ---
    spectral = extract_spectral(ch1_epoch)
    if spectral is None:
        return None

    # --- Geometric: per-channel features + bilateral ---
    geom_parts: list[np.ndarray] = []

    # Per-delay data for bilateral computation: tau -> {channel: (theta, omega1)}
    bilateral_data: dict[int, dict[str, tuple]] = {}

    for ch_name, ch_epoch in [("c4", ch1_epoch), ("c3", ch2_epoch)]:
        for tau in TAUS:
            result = _extract_channel_geometric(ch_epoch, tau)
            if result is None:
                return None
            features_8, theta_arr, omega1_val = result
            geom_parts.append(features_8)

            if tau not in bilateral_data:
                bilateral_data[tau] = {}
            bilateral_data[tau][ch_name] = (theta_arr, omega1_val)

    # Bilateral features per delay
    for tau in TAUS:
        theta_c4, omega_c4 = bilateral_data[tau]["c4"]
        theta_c3, omega_c3 = bilateral_data[tau]["c3"]
        bilateral = _compute_bilateral(theta_c4, theta_c3, omega_c4, omega_c3)
        geom_parts.append(bilateral)

    geometric = np.concatenate(geom_parts)  # 48 + 9 = 57
    assert len(geometric) == N_GEOM, f"Expected {N_GEOM} geometric, got {len(geometric)}"

    if not np.all(np.isfinite(spectral)) or not np.all(np.isfinite(geometric)):
        return None

    return spectral, geometric


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "hmc"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HMC Multi-Scale Bilateral Feature Extraction")
    print(f"  Channels: C4-M1, C3-M2")
    print(f"  Delays:   τ = {TAUS}")
    print(f"  Features: {N_SPECTRAL} spectral + {N_GEOM} geometric = {N_SPECTRAL + N_GEOM} total")
    print("=" * 70)

    pairs = find_subjects(data_dir)
    print(f"  Found {len(pairs)} subjects\n")

    all_spectral: list[np.ndarray] = []
    all_geometric: list[np.ndarray] = []
    all_stages: list[int] = []
    all_subjects: list[int] = []

    n_loaded = 0
    n_skipped_ch = 0
    t0 = time.time()

    for sid, psg, hyp in pairs:
        result = load_subject_bilateral(psg, hyp)
        if result is None:
            n_skipped_ch += 1
            continue

        ch1_epochs, ch2_epochs, labels = result
        subj_spec, subj_geom, subj_stages = [], [], []

        for ep1, ep2, lab in zip(ch1_epochs, ch2_epochs, labels):
            feats = extract_epoch_features(ep1, ep2)
            if feats is None:
                continue
            spec, geom = feats
            subj_spec.append(spec)
            subj_geom.append(geom)
            subj_stages.append(STAGE_TO_INT[lab])

        if len(subj_spec) < 20:
            continue

        all_spectral.extend(subj_spec)
        all_geometric.extend(subj_geom)
        all_stages.extend(subj_stages)
        all_subjects.extend([n_loaded] * len(subj_spec))
        n_loaded += 1

        if n_loaded % 10 == 0:
            elapsed = time.time() - t0
            rate = elapsed / n_loaded
            remaining = rate * (len(pairs) - n_loaded - n_skipped_ch)
            print(f"    {n_loaded} subjects, {len(all_stages):,} epochs "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    features_spectral = np.array(all_spectral, dtype=np.float32)
    features_geometric = np.array(all_geometric, dtype=np.float32)
    stages = np.array(all_stages, dtype=np.int8)
    subjects = np.array(all_subjects, dtype=np.int32)

    elapsed = time.time() - t0
    print(f"\n  Total: {n_loaded} subjects, {len(stages):,} epochs")
    print(f"  Skipped (missing channel): {n_skipped_ch}")
    print(f"  Spectral matrix:  {features_spectral.shape}")
    print(f"  Geometric matrix: {features_geometric.shape}")
    print(f"  Stage distribution: {dict(zip(*np.unique(stages, return_counts=True)))}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    out_path = results_dir / "hmc_features_multiscale.npz"
    np.savez_compressed(
        out_path,
        features_spectral=features_spectral,
        features_geometric=features_geometric,
        stages=stages,
        subjects=subjects,
        feature_names_spectral=np.array(SPECTRAL_NAMES),
        feature_names_geometric=np.array(GEOM_NAMES),
    )

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved: {out_path}")
    print(f"  File size: {size_mb:.1f} MB")

    # Print feature name summary
    print(f"\n  Spectral features ({N_SPECTRAL}):")
    for name in SPECTRAL_NAMES:
        print(f"    {name}")

    print(f"\n  Geometric features ({N_GEOM}):")
    print(f"    Per-channel (48): {PER_CH_FEATURES[0]}_{CHANNELS[0]}_t{TAUS[0]} ... "
          f"{PER_CH_FEATURES[-1]}_{CHANNELS[-1]}_t{TAUS[-1]}")
    print(f"    Bilateral (9):    {BILATERAL_FEATURES[0]}_t{TAUS[0]} ... "
          f"{BILATERAL_FEATURES[-1]}_t{TAUS[-1]}")
    print(f"\n  Sample names: {GEOM_NAMES[:3]} ... {GEOM_NAMES[-3:]}")


if __name__ == "__main__":
    main()
