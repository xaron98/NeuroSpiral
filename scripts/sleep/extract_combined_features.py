#!/usr/bin/env python3
"""NeuroSpiral — Combined Feature Extraction Pipeline.

Computes BOTH torus geometric features AND spectral features from the
SAME preprocessed data in a SINGLE run, producing ONE aligned output file.

This solves the alignment problem: the two existing feature files
(phase_a_full_features.npz and hmc_features_multiscale.npz) were computed
by DIFFERENT pipelines with different parameters and cannot be combined.

Channels: EEG (C4-M1), ECG, EOG (E1-M2), EMG (chin)
Torus features: 4 channels x 3 tau x 8 features = 96
Coupling features: 6 pairs x 6 features = 36
Spectral features: 4 channels x 8 features = 32

Output: results/combined_features.npz
  torus_individual       : (N, 96)  float32
  coupling               : (N, 36)  float32
  spectral               : (N, 32)  float32
  stages                 : (N,)     int8    (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
  subjects               : (N,)     int16
  feature_names_torus    : (96,)    str
  feature_names_coupling : (36,)    str
  feature_names_spectral : (32,)    str

Usage:
    python scripts/extract_combined_features.py
    python scripts/extract_combined_features.py --data-dir /path/to/hmc
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import mne
from scipy.signal import welch
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SFREQ = 100
EPOCH_SEC = 30
EPOCH_SAMPLES = SFREQ * EPOCH_SEC

SIGNAL_NAMES = ["eeg", "ecg", "eog", "emg"]
PAIRS = list(combinations(SIGNAL_NAMES, 2))  # 6 pairs

# Multi-scale delays per signal (same as phase_a_full_validation.py)
TAU_MULTI = {
    "eeg": [10, 25, 40],
    "ecg": [10, 25, 40],
    "eog": [15, 30, 50],
    "emg": [5, 10, 20],
}

# Channel-specific bandpass filter ranges
FILTER_RANGES = {
    "eeg": (0.5, 30.0),
    "ecg": (0.5, 40.0),
    "eog": (0.3, 35.0),
    "emg": (10.0, 49.0),
}

HMC_LABELS = {
    "Sleep stage W": "W", "Sleep stage N1": "N1", "Sleep stage N2": "N2",
    "Sleep stage N3": "N3", "Sleep stage R": "REM", "Sleep stage ?": None,
    "0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "REM", "5": None,
    "W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "REM", "R": "REM",
}

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGES)}

# 8 torus features per (signal, tau) — DO NOT MODIFY formulas
TORUS_FEAT_NAMES = [
    "omega1", "torus_curvature", "angular_acceleration",
    "geodesic_distance", "angular_entropy",
    "phase_diff_std", "phase_coherence", "transition_rate",
]

# 6 coupling features per pair
COUPLING_FEAT_NAMES = [
    "vel_corr", "vel_coherence", "theta_coherence",
    "mean_phase_angle", "phase_circ_std", "windowed_corr",
]

# 8 spectral features per channel (from CLAUDE.md spec)
SPECTRAL_FEAT_NAMES = [
    "delta", "theta", "alpha", "sigma", "beta",
    "total_power", "sef95", "median_freq",
]

SPECTRAL_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 15.0),
    "beta": (13.0, 30.0),
}


# ─────────────────────────────────────────────────────────────
# Feature name builders
# ─────────────────────────────────────────────────────────────
def build_torus_names() -> list[str]:
    """96 torus feature names: signal_tau_feature."""
    names = []
    for sig in SIGNAL_NAMES:
        for tau in TAU_MULTI[sig]:
            for feat in TORUS_FEAT_NAMES:
                names.append(f"{feat}_{sig}_t{tau}")
    return names


def build_coupling_names() -> list[str]:
    """36 coupling feature names: feature_sigA_sigB."""
    names = []
    for sig_a, sig_b in PAIRS:
        for feat in COUPLING_FEAT_NAMES:
            names.append(f"{feat}_{sig_a}_{sig_b}")
    return names


def build_spectral_names() -> list[str]:
    """32 spectral feature names: feature_signal."""
    names = []
    for sig in SIGNAL_NAMES:
        for feat in SPECTRAL_FEAT_NAMES:
            names.append(f"{feat}_{sig}")
    return names


TORUS_NAMES = build_torus_names()      # 96
COUPLING_NAMES = build_coupling_names()  # 36
SPECTRAL_NAMES = build_spectral_names()  # 32

N_TORUS = len(TORUS_NAMES)       # 96
N_COUPLING = len(COUPLING_NAMES)  # 36
N_SPECTRAL = len(SPECTRAL_NAMES)  # 32


# ─────────────────────────────────────────────────────────────
# Subject discovery and loading
# ─────────────────────────────────────────────────────────────
def find_subjects(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """Find all valid HMC subject pairs (PSG + hypnogram)."""
    pairs = []
    for i in range(1, 500):
        sid = f"SN{i:03d}"
        psg = data_dir / f"{sid}.edf"
        hyp = data_dir / f"{sid}_sleepscoring.edf"
        if psg.exists() and hyp.exists():
            if psg.stat().st_size > 1_000_000 and hyp.stat().st_size > 500:
                pairs.append((sid, psg, hyp))
    return pairs


def load_4_signals(
    psg_path: Path, hyp_path: Path
) -> tuple[dict[str, np.ndarray] | None, list[str | None] | None, int]:
    """Load and preprocess 4 channels from one HMC subject.

    Returns (signals_dict, labels_list, n_epochs) or (None, None, 0).
    All channels are bandpass filtered and resampled to 100 Hz.
    """
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)

    # Find channels
    channels: dict[str, str] = {}
    for name in ["EEG C4-M1", "C4-M1", "EEG C4"]:
        if name in raw.ch_names:
            channels["eeg"] = name
            break
    for name in ["ECG", "ECG II", "EKG"]:
        if name in raw.ch_names:
            channels["ecg"] = name
            break
    if "ecg" not in channels:
        cands = [c for c in raw.ch_names if "ECG" in c.upper()]
        if cands:
            channels["ecg"] = cands[0]
    for name in ["EOG E1-M2", "E1-M2", "EOG E1"]:
        if name in raw.ch_names:
            channels["eog"] = name
            break
    if "eog" not in channels:
        cands = [c for c in raw.ch_names if "EOG" in c.upper()]
        if cands:
            channels["eog"] = cands[0]
    for name in ["EMG chin", "EMG"]:
        if name in raw.ch_names:
            channels["emg"] = name
            break
    if "emg" not in channels:
        cands = [c for c in raw.ch_names if "EMG" in c.upper() or "chin" in c.lower()]
        if cands:
            channels["emg"] = cands[0]

    missing = [k for k in SIGNAL_NAMES if k not in channels]
    if missing:
        return None, None, 0

    # Filter and resample each channel independently
    signals: dict[str, np.ndarray] = {}
    for key, ch_name in channels.items():
        r = raw.copy().pick([ch_name])
        lo, hi = FILTER_RANGES[key]
        r.filter(lo, hi, verbose=False)
        if r.info["sfreq"] != SFREQ:
            r.resample(SFREQ, verbose=False)
        signals[key] = r.get_data()[0]

    # Trim all channels to same length
    min_len = min(len(s) for s in signals.values())
    for key in signals:
        signals[key] = signals[key][:min_len]

    n_ep = int(min_len / SFREQ // EPOCH_SEC)

    # Read hypnogram annotations
    annots = mne.read_annotations(str(hyp_path))
    labels: list[str | None] = [None] * n_ep
    for onset, dur, desc in zip(annots.onset, annots.duration, annots.description):
        stage = HMC_LABELS.get(str(desc).strip())
        if stage is None:
            continue
        s = int(onset // EPOCH_SEC)
        for e in range(s, min(s + max(1, int(dur // EPOCH_SEC)), n_ep)):
            labels[e] = stage

    return signals, labels, n_ep


# ─────────────────────────────────────────────────────────────
# Torus feature extraction (DO NOT MODIFY formulas)
# ─────────────────────────────────────────────────────────────
def _wrap(d: np.ndarray) -> np.ndarray:
    """Wrap angular differences to [-pi, pi]."""
    return (d + np.pi) % (2 * np.pi) - np.pi


def takens_embed(signal: np.ndarray, d: int = 4, tau: int = 25) -> np.ndarray | None:
    """Takens delay embedding: 1D signal -> R^d point cloud."""
    n = len(signal)
    n_emb = n - (d - 1) * tau
    if n_emb < 50:
        return None
    emb = np.zeros((n_emb, d))
    for i in range(d):
        emb[:, i] = signal[i * tau : i * tau + n_emb]
    if np.std(emb) < 1e-15 or not np.all(np.isfinite(emb)):
        return None
    return emb


def torus_features_8(embedding: np.ndarray) -> np.ndarray | None:
    """Extract the 8 canonical torus features from a 4D embedding.

    Returns (8,) array or None on failure.
    Order: omega1, torus_curvature, angular_acceleration,
           geodesic_distance, angular_entropy,
           phase_diff_std, phase_coherence, transition_rate.
    """
    if embedding is None or embedding.shape[0] < 20:
        return None

    theta = np.arctan2(embedding[:, 1], embedding[:, 0])
    phi = np.arctan2(embedding[:, 3], embedding[:, 2])
    dtheta = _wrap(np.diff(theta))
    dphi = _wrap(np.diff(phi))
    N = len(dtheta)
    if N < 5:
        return None

    feats = []

    # 1. omega1: mean absolute angular velocity
    feats.append(float(np.mean(np.abs(dtheta))))

    # 2. torus_curvature: mean |second difference| of theta
    feats.append(float(np.mean(np.abs(np.diff(dtheta)))) if N >= 2 else 0.0)

    # 3. angular_acceleration: variance of angular velocity
    feats.append(float(np.var(dtheta)))

    # 4. geodesic_distance: total arc length on flat torus
    feats.append(float(np.sum(np.sqrt(dtheta**2 + dphi**2))))

    # 5. angular_entropy: Shannon entropy of theta in 16 bins
    counts, _ = np.histogram(theta, bins=16, range=(-np.pi, np.pi))
    c = counts.astype(np.float64)
    total = c.sum()
    if total > 0:
        p = c / total
        p = p[p > 0]
        feats.append(float(-np.sum(p * np.log2(p))))
    else:
        feats.append(0.0)

    # 6. phase_diff_std: circular std of (theta - phi)
    pd = theta - phi
    R_len = np.abs(np.mean(np.exp(1j * pd)))
    feats.append(float(np.sqrt(-2 * np.log(max(R_len, 1e-10)))) if R_len < 1 else 0.0)

    # 7. phase_coherence: mean resultant length R
    feats.append(float(R_len))

    # 8. transition_rate: fraction of timesteps with vertex change
    signs = (embedding >= 0).astype(int)
    verts = signs[:, 0] * 8 + signs[:, 1] * 4 + signs[:, 2] * 2 + signs[:, 3]
    feats.append(float(np.sum(np.diff(verts) != 0) / max(len(verts) - 1, 1)))

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Coupling feature extraction
# ─────────────────────────────────────────────────────────────
def coupling_features_6(
    sig_a: np.ndarray, sig_b: np.ndarray, tau: int = 25
) -> np.ndarray | None:
    """Extract 6 inter-channel coupling features from two signals.

    Returns (6,) array or None on failure.
    """
    emb_a = takens_embed(sig_a, tau=tau)
    emb_b = takens_embed(sig_b, tau=tau)
    if emb_a is None or emb_b is None:
        return None

    theta_a = np.arctan2(emb_a[:, 1], emb_a[:, 0])
    theta_b = np.arctan2(emb_b[:, 1], emb_b[:, 0])
    n = min(len(theta_a), len(theta_b))
    theta_a, theta_b = theta_a[:n], theta_b[:n]
    if n < 50:
        return None

    dtheta_a = _wrap(np.diff(theta_a))
    dtheta_b = _wrap(np.diff(theta_b))

    feats = []

    # 1. vel_corr: correlation of angular velocity magnitudes
    r, _ = pearsonr(np.abs(dtheta_a), np.abs(dtheta_b))
    feats.append(float(r) if np.isfinite(r) else 0.0)

    # 2. vel_coherence: mean resultant length of velocity difference
    R_vel = np.abs(np.mean(np.exp(1j * (dtheta_a - dtheta_b))))
    feats.append(float(R_vel))

    # 3. theta_coherence: phase coherence between channels
    phase_diff = _wrap(theta_a - theta_b)
    R_t = np.abs(np.mean(np.exp(1j * phase_diff)))
    feats.append(float(R_t))

    # 4. mean_phase_angle: mean direction of phase difference
    feats.append(float(np.angle(np.mean(np.exp(1j * phase_diff)))))

    # 5. phase_circ_std: circular std of phase difference
    feats.append(float(np.sqrt(-2 * np.log(max(R_t, 1e-10)))) if R_t < 1 else 0.0)

    # 6. windowed_corr: mean of windowed velocity correlations
    win = 100
    n_wins = len(dtheta_a) // win
    if n_wins > 2:
        wc = []
        for w in range(n_wins):
            s, e = w * win, (w + 1) * win
            if e > len(dtheta_a):
                break
            rw, _ = pearsonr(np.abs(dtheta_a[s:e]), np.abs(dtheta_b[s:e]))
            if np.isfinite(rw):
                wc.append(rw)
        feats.append(float(np.mean(wc)) if wc else 0.0)
    else:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Spectral feature extraction
# ─────────────────────────────────────────────────────────────
def spectral_features_8(epoch: np.ndarray, sfreq: float = SFREQ) -> np.ndarray | None:
    """Compute 8 spectral features from a single-channel epoch.

    Features (from CLAUDE.md spec):
      1. delta: relative power 0.5-4 Hz
      2. theta: relative power 4-8 Hz
      3. alpha: relative power 8-13 Hz
      4. sigma: relative power 12-15 Hz
      5. beta:  relative power 13-30 Hz
      6. total_power: absolute total power (V^2/Hz integrated)
      7. sef95: spectral edge frequency at 95%
      8. median_freq: median frequency (50% cumulative power)

    Returns (8,) array or None if PSD computation fails.
    """
    freqs, psd = welch(epoch, fs=sfreq, nperseg=min(256, len(epoch)))
    total = np.trapz(psd, freqs)
    if total < 1e-30:
        return None

    feats = []

    # Relative band powers
    for band_name in ["delta", "theta", "alpha", "sigma", "beta"]:
        lo, hi = SPECTRAL_BANDS[band_name]
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(float(np.trapz(psd[mask], freqs[mask]) / total))

    # Absolute total power
    feats.append(float(total))

    # Cumulative power for SEF95 and median frequency
    # Use trapezoidal cumulative integration
    df = np.diff(freqs, prepend=freqs[0])
    cum_power = np.cumsum(psd * df)
    cum_total = cum_power[-1]

    if cum_total < 1e-30:
        feats.extend([0.0, 0.0])
    else:
        # SEF95: frequency below which 95% of total power lies
        idx_95 = np.searchsorted(cum_power, 0.95 * cum_total)
        feats.append(float(freqs[min(idx_95, len(freqs) - 1)]))

        # Median frequency: frequency below which 50% of total power lies
        idx_50 = np.searchsorted(cum_power, 0.50 * cum_total)
        feats.append(float(freqs[min(idx_50, len(freqs) - 1)]))

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Per-subject processing
# ─────────────────────────────────────────────────────────────
def process_subject(
    signals: dict[str, np.ndarray],
    labels: list[str | None],
    n_ep: int,
) -> dict | None:
    """Extract ALL features for one subject from the SAME preprocessed data.

    Returns dict with arrays for torus, coupling, spectral, stages,
    or None if too few valid epochs.
    """
    torus_list = []       # -> (n_valid, 96)
    coupling_list = []    # -> (n_valid, 36)
    spectral_list = []    # -> (n_valid, 32)
    stage_list = []

    for i in range(n_ep):
        if labels[i] is None:
            continue

        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES

        # Check bounds
        if end > len(signals["eeg"]):
            break

        # Artifact rejection on EEG (>500 uV)
        if np.max(np.abs(signals["eeg"][start:end])) > 500e-6:
            continue

        # ── Extract epoch signals for all 4 channels ──
        epochs = {key: signals[key][start:end] for key in SIGNAL_NAMES}

        # ── Torus features: 4 signals x multi-tau x 8 = 96 ──
        torus_feats = []
        ok = True
        for key in SIGNAL_NAMES:
            for tau in TAU_MULTI[key]:
                emb = takens_embed(epochs[key], tau=tau)
                f = torus_features_8(emb)
                if f is None:
                    ok = False
                    break
                torus_feats.extend(f)
            if not ok:
                break
        if not ok:
            continue

        # ── Coupling features: 6 pairs x 6 = 36 ──
        coup_feats = []
        for sig_a, sig_b in PAIRS:
            cf = coupling_features_6(epochs[sig_a], epochs[sig_b], tau=25)
            if cf is None:
                ok = False
                break
            coup_feats.extend(cf)
        if not ok:
            continue

        # ── Spectral features: 4 channels x 8 = 32 ──
        spec_feats = []
        for key in SIGNAL_NAMES:
            sf = spectral_features_8(epochs[key], SFREQ)
            if sf is None:
                ok = False
                break
            spec_feats.extend(sf)
        if not ok:
            continue

        # All features computed successfully for this epoch
        torus_list.append(torus_feats)
        coupling_list.append(coup_feats)
        spectral_list.append(spec_feats)
        stage_list.append(STAGE_TO_INT[labels[i]])

    if len(stage_list) < 50:
        return None

    return {
        "torus": np.array(torus_list, dtype=np.float32),
        "coupling": np.array(coupling_list, dtype=np.float32),
        "spectral": np.array(spectral_list, dtype=np.float32),
        "stages": np.array(stage_list, dtype=np.int8),
        "n_valid": len(stage_list),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuroSpiral — Combined Feature Extraction Pipeline"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=PROJECT_ROOT / "data" / "hmc",
        help="Path to HMC dataset directory",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results" / "combined_features.npz",
        help="Output .npz file path",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("  NEUROSPIRAL — Combined Feature Extraction Pipeline")
    print("  Torus + Coupling + Spectral from SAME preprocessed data")
    print("=" * 70)
    print(f"  Data dir:  {data_dir}")
    print(f"  Output:    {output_path}")
    print(f"  Features:  {N_TORUS} torus + {N_COUPLING} coupling + {N_SPECTRAL} spectral"
          f" = {N_TORUS + N_COUPLING + N_SPECTRAL} total")
    print()

    subjects = find_subjects(data_dir)
    print(f"  Found {len(subjects)} subject pairs\n")

    all_torus: list[np.ndarray] = []
    all_coupling: list[np.ndarray] = []
    all_spectral: list[np.ndarray] = []
    all_stages: list[np.ndarray] = []
    all_subjects: list[np.ndarray] = []
    n_ok = 0
    n_skipped_ch = 0
    n_skipped_ep = 0
    t0 = time.time()

    for idx, (sid, psg, hyp) in enumerate(subjects):
        print(f"  [{idx + 1}/{len(subjects)}] {sid}...", end="", flush=True)

        signals, labels, n_ep = load_4_signals(psg, hyp)
        if signals is None:
            n_skipped_ch += 1
            print(" skip (missing channels)")
            continue

        result = process_subject(signals, labels, n_ep)
        if result is None:
            n_skipped_ep += 1
            print(" skip (too few valid epochs)")
            continue

        n_valid = result["n_valid"]
        print(f" {n_valid} epochs")

        all_torus.append(result["torus"])
        all_coupling.append(result["coupling"])
        all_spectral.append(result["spectral"])
        all_stages.append(result["stages"])
        all_subjects.append(np.full(n_valid, n_ok, dtype=np.int16))
        n_ok += 1

        if n_ok % 20 == 0:
            elapsed = time.time() - t0
            total_epochs = sum(len(s) for s in all_stages)
            rate = elapsed / n_ok
            remaining = rate * (len(subjects) - idx - 1)
            print(f"    --- {n_ok} subjects, {total_epochs:,} epochs "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining) ---")

    if n_ok == 0:
        print("\n  ERROR: No subjects processed successfully.")
        sys.exit(1)

    # Concatenate all subjects
    torus_individual = np.vstack(all_torus)
    coupling = np.vstack(all_coupling)
    spectral = np.vstack(all_spectral)
    stages = np.concatenate(all_stages)
    subjects_arr = np.concatenate(all_subjects)

    # Clean NaN/Inf (should not happen, but safety net)
    valid = (
        np.all(np.isfinite(torus_individual), axis=1)
        & np.all(np.isfinite(coupling), axis=1)
        & np.all(np.isfinite(spectral), axis=1)
    )
    n_before = len(stages)
    torus_individual = torus_individual[valid]
    coupling = coupling[valid]
    spectral = spectral[valid]
    stages = stages[valid]
    subjects_arr = subjects_arr[valid]
    n_removed = n_before - len(stages)

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Subjects loaded:        {n_ok}")
    print(f"  Skipped (missing ch):   {n_skipped_ch}")
    print(f"  Skipped (few epochs):   {n_skipped_ep}")
    print(f"  Total epochs:           {len(stages):,}")
    if n_removed > 0:
        print(f"  Removed (NaN/Inf):      {n_removed}")
    print(f"  Torus matrix:           {torus_individual.shape}")
    print(f"  Coupling matrix:        {coupling.shape}")
    print(f"  Spectral matrix:        {spectral.shape}")
    print(f"  Elapsed:                {elapsed:.0f}s ({elapsed / 60:.1f}m)")

    # Stage distribution
    print(f"\n  Stage distribution:")
    for s_int, s_name in enumerate(STAGES):
        count = np.sum(stages == s_int)
        pct = count / len(stages) * 100
        print(f"    {s_name:>4}: {count:>6,} ({pct:5.1f}%)")

    # Save
    np.savez_compressed(
        output_path,
        torus_individual=torus_individual,
        coupling=coupling,
        spectral=spectral,
        stages=stages,
        subjects=subjects_arr,
        feature_names_torus=np.array(TORUS_NAMES),
        feature_names_coupling=np.array(COUPLING_NAMES),
        feature_names_spectral=np.array(SPECTRAL_NAMES),
    )

    size_mb = output_path.stat().st_size / 1e6
    print(f"\n  Saved: {output_path}")
    print(f"  File size: {size_mb:.1f} MB")

    # Verification
    print(f"\n  Verification:")
    print(f"    torus_individual:  {torus_individual.shape} "
          f"(expected (N, {N_TORUS}))")
    print(f"    coupling:          {coupling.shape} "
          f"(expected (N, {N_COUPLING}))")
    print(f"    spectral:          {spectral.shape} "
          f"(expected (N, {N_SPECTRAL}))")
    print(f"    stages:            {stages.shape}")
    print(f"    subjects:          {subjects_arr.shape} "
          f"(unique: {len(np.unique(subjects_arr))})")

    assert torus_individual.shape[1] == N_TORUS, \
        f"Torus shape mismatch: {torus_individual.shape[1]} != {N_TORUS}"
    assert coupling.shape[1] == N_COUPLING, \
        f"Coupling shape mismatch: {coupling.shape[1]} != {N_COUPLING}"
    assert spectral.shape[1] == N_SPECTRAL, \
        f"Spectral shape mismatch: {spectral.shape[1]} != {N_SPECTRAL}"
    assert len(stages) == len(subjects_arr) == torus_individual.shape[0]

    print(f"\n  ALL CHECKS PASSED")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
