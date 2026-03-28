#!/usr/bin/env python3
"""Extract 16 features (8 spectral + 8 geometric) per epoch from HMC dataset.

Saves results/hmc_features.npz containing:
  features : (n_epochs, 16) float32 array
  stages   : (n_epochs,) int8 array (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
  subjects : (n_epochs,) int16 array (subject index for fold assignment)

Run locally before uploading to Colab for temporal_context_colab.py.
"""

from __future__ import annotations

import sys
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


# ===================================================================
# Subject discovery + loading
# ===================================================================
def find_subjects(data_dir: Path) -> list[tuple[str, Path, Path]]:
    pairs = []
    for i in range(1, 200):
        sid = f"SN{i:03d}"
        psg = data_dir / f"{sid}.edf"
        hyp = data_dir / f"{sid}_sleepscoring.edf"
        if psg.exists() and hyp.exists():
            if psg.stat().st_size > 1_000_000 and hyp.stat().st_size > 500:
                pairs.append((sid, psg, hyp))
    return pairs


def load_subject(psg_path: Path, hyp_path: Path):
    """Return (epochs_array, labels) or None."""
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    ch = None
    for c in ["EEG C4-M1", "C4-M1", "EEG C4"]:
        if c in raw.ch_names:
            ch = c
            break
    if ch is None:
        eeg = [c for c in raw.ch_names if "EEG" in c.upper()]
        ch = eeg[0] if eeg else None
    if ch is None:
        return None

    raw.pick([ch])
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)
    raw.filter(0.5, 30.0, verbose=False)

    signal = raw.get_data()[0]
    n_ep = int(len(signal) / SFREQ // EPOCH_SEC)

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

    epochs_out, labels_out = [], []
    for i in range(n_ep):
        if labels[i] is None:
            continue
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        if end > len(signal):
            break
        ep = signal[start:end]
        if np.max(np.abs(ep)) > 500e-6:
            continue
        epochs_out.append(ep)
        labels_out.append(labels[i])

    if not epochs_out:
        return None
    return np.array(epochs_out), labels_out


# ===================================================================
# Feature extraction (16 features per epoch)
# ===================================================================
def _wrap(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


def extract_features(epoch: np.ndarray) -> np.ndarray | None:
    """Return (16,) vector: 8 spectral + 8 geometric, or None."""
    # --- Spectral (8) ---
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

    spec = np.array([
        bp["delta"], bp["theta"], bp["alpha"], bp["sigma"], bp["beta"],
        delta_beta, activity, mobility,
    ])

    # --- Geometric (8) ---
    try:
        emb, _ = time_delay_embedding(epoch, dimension=4, tau=25)
    except ValueError:
        return None
    if emb.shape[0] < 10 or not np.all(np.isfinite(emb)):
        return None

    theta = np.arctan2(emb[:, 1], emb[:, 0])
    phi = np.arctan2(emb[:, 3], emb[:, 2])
    theta_uw = np.unwrap(theta)
    phi_uw = np.unwrap(phi)
    dtheta = np.diff(theta_uw)

    omega1 = float(np.mean(np.abs(dtheta)))

    phase_diff = theta_uw - phi_uw
    phase_diff_std = float(np.std(np.diff(phase_diff)))

    diff_w = _wrap(theta - phi)
    mc = float(np.mean(np.cos(diff_w)))
    ms = float(np.mean(np.sin(diff_w)))
    phase_coherence = float(np.sqrt(mc**2 + ms**2))

    signs = (emb >= 0).astype(int)
    verts = signs[:, 0] * 8 + signs[:, 1] * 4 + signs[:, 2] * 2 + signs[:, 3]
    transition_rate = float(np.sum(np.diff(verts) != 0) / max(len(verts) - 1, 1))

    v2 = extract_torus_features_v2(emb)

    geom = np.array([
        omega1,
        v2["torus_curvature"],
        v2["angular_acceleration"],
        v2["geodesic_distance"],
        v2["angular_entropy"],
        phase_diff_std,
        phase_coherence,
        transition_rate,
    ])

    combined = np.concatenate([spec, geom])
    if not np.all(np.isfinite(combined)):
        return None
    return combined


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "hmc"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("HMC Feature Extraction (16 features per epoch)")
    print("=" * 65)

    pairs = find_subjects(data_dir)
    print(f"  Found {len(pairs)} subjects\n")

    all_features: list[np.ndarray] = []
    all_stages: list[int] = []
    all_subjects: list[int] = []

    n_loaded = 0
    for sid, psg, hyp in pairs:
        result = load_subject(psg, hyp)
        if result is None:
            continue

        epochs, labels = result
        subj_feats = []
        subj_stages = []

        for ep, lab in zip(epochs, labels):
            f = extract_features(ep)
            if f is None:
                continue
            subj_feats.append(f)
            subj_stages.append(STAGE_TO_INT[lab])

        if len(subj_feats) < 20:
            continue

        all_features.extend(subj_feats)
        all_stages.extend(subj_stages)
        all_subjects.extend([n_loaded] * len(subj_feats))
        n_loaded += 1

        if n_loaded % 20 == 0:
            print(f"    {n_loaded} subjects, {len(all_stages)} epochs so far ...")

    features = np.array(all_features, dtype=np.float32)
    stages = np.array(all_stages, dtype=np.int8)
    subjects = np.array(all_subjects, dtype=np.int16)

    print(f"\n  Total: {n_loaded} subjects, {len(stages)} epochs")
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Stage distribution: {dict(zip(*np.unique(stages, return_counts=True)))}")

    out_path = results_dir / "hmc_features.npz"
    np.savez_compressed(out_path, features=features, stages=stages, subjects=subjects)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved: {out_path}")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Epochs: {len(stages)}")
    print(f"  Features per epoch: {features.shape[1]}")


if __name__ == "__main__":
    main()
