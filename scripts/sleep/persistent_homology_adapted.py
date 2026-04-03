#!/usr/bin/env python3
"""NeuroSpiral — Adapted Persistent Homology Analysis.

Gardner et al. (2022) used Neuropixels on individual grid cells.
We cannot replicate that with scalp EEG. Our test is DIFFERENT but valid:

  Does the Takens delay embedding of EEG/ECG/EOG/EMG signals
  produce point clouds with toroidal topology (beta_1 = 2)?

  If yes -> the signal has 2 detectable periodicities
  If no  -> the torus is a useful feature extractor without
            deep topological meaning

Three tests:
  A) PH on raw Takens delay embedding (4D point cloud, per epoch)
  B) PH on combined feature space (128D = 96 torus + 32 spectral)
  C) PH per channel (24D torus features each)

All tests use Z_47 coefficients (same as Gardner) and compare
against shuffled + Gaussian controls with permutation p-values.

Usage:
    python scripts/persistent_homology_adapted.py
    python scripts/persistent_homology_adapted.py --skip-a  # skip raw EDF test
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import mne
import ripser
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_INT = {s: i for i, s in enumerate(STAGES)}

CHANNELS = {
    "EEG": slice(0, 24),
    "ECG": slice(24, 48),
    "EOG": slice(48, 72),
    "EMG": slice(72, 96),
}

HMC_LABELS = {
    "Sleep stage W": "W", "Sleep stage N1": "N1", "Sleep stage N2": "N2",
    "Sleep stage N3": "N3", "Sleep stage R": "REM", "Sleep stage ?": None,
    "0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "REM", "5": None,
}

COEFF = 47  # Z_47, same prime field as Gardner et al.


# ─────────────────────────────────────────────────────────────
# Core PH utilities
# ─────────────────────────────────────────────────────────────
def compute_ph(X, maxdim=2, thresh=None, coeff=COEFF):
    """Compute Vietoris-Rips persistent homology with ripser.

    Returns dict with diagrams, estimated Betti numbers (gap method),
    and per-dimension lifetime arrays.
    """
    if thresh is None:
        from scipy.spatial.distance import pdist
        dists = pdist(X[:min(300, len(X))])
        thresh = float(np.percentile(dists, 60)) * 1.5
        thresh = max(thresh, 0.5)

    result = ripser.ripser(X, maxdim=maxdim, thresh=thresh, coeff=coeff)
    diagrams = result["dgms"]

    betti = []
    lifetimes_all = []
    max_pers = []

    for dim in range(maxdim + 1):
        dgm = diagrams[dim]
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            betti.append(0)
            lifetimes_all.append(np.array([]))
            max_pers.append(0.0)
            continue

        lt = finite[:, 1] - finite[:, 0]
        lt = lt[lt > 1e-10]
        lt = np.sort(lt)[::-1]

        if len(lt) == 0:
            betti.append(0)
            lifetimes_all.append(np.array([]))
            max_pers.append(0.0)
            continue

        # Gap method: find largest ratio between consecutive sorted lifetimes
        n_persistent = _count_persistent_gap(lt)
        betti.append(n_persistent)
        lifetimes_all.append(lt)
        max_pers.append(float(lt[0]))

    return {
        "betti": betti,
        "lifetimes": lifetimes_all,
        "max_persistence": max_pers,
        "thresh": thresh,
    }


def _count_persistent_gap(lifetimes, min_ratio=2.0):
    """Count persistent features using the gap method.

    Sort lifetimes descending. Find the largest ratio gap between
    consecutive lifetimes. Features above the gap are 'persistent'.
    Requires the gap ratio to be at least min_ratio.
    """
    if len(lifetimes) <= 1:
        return len(lifetimes)

    # Compute ratios between consecutive sorted lifetimes
    ratios = lifetimes[:-1] / np.maximum(lifetimes[1:], 1e-15)

    # Find the largest gap
    best_idx = int(np.argmax(ratios))
    best_ratio = ratios[best_idx]

    if best_ratio >= min_ratio:
        return best_idx + 1  # number of features above the gap
    else:
        return 0  # no clear gap -> no persistent features


def subsample_and_scale(X, n_sub, n_pca=0, seed=42):
    """Subsample, scale, and optionally PCA-reduce a point cloud."""
    rng = np.random.default_rng(seed)
    if len(X) > n_sub:
        idx = rng.choice(len(X), size=n_sub, replace=False)
        X = X[idx]
    X = StandardScaler().fit_transform(X)
    if n_pca > 0 and n_pca < X.shape[1]:
        X = PCA(n_components=n_pca, random_state=seed).fit_transform(X)
    return X


# ─────────────────────────────────────────────────────────────
# Takens embedding (local copy, same formulas as pipeline)
# ─────────────────────────────────────────────────────────────
def estimate_tau(signal, max_lag=100, n_bins=64):
    """First local minimum of mutual information."""
    max_lag = min(max_lag, len(signal) // 4)
    mi = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            x = signal
            y = signal.copy()
        else:
            x = signal[:-lag]
            y = signal[lag:]
        hist, _, _ = np.histogram2d(x, y, bins=n_bins)
        pxy = hist / hist.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        val = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if pxy[i, j] > 1e-12 and px[i] > 1e-12 and py[j] > 1e-12:
                    val += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
        mi[lag] = val

    mins = argrelextrema(mi[1:], np.less)[0]
    if len(mins) > 0:
        return int(mins[0] + 1)
    below = np.where(mi[1:] < mi[0] / np.e)[0]
    if len(below) > 0:
        return int(below[0] + 1)
    return max(1, max_lag // 4)


def takens_embed(signal, d=4, tau=25):
    """Takens delay embedding."""
    n_emb = len(signal) - (d - 1) * tau
    if n_emb < 50:
        return None
    emb = np.zeros((n_emb, d))
    for i in range(d):
        emb[:, i] = signal[i * tau: i * tau + n_emb]
    if np.std(emb) < 1e-15 or not np.all(np.isfinite(emb)):
        return None
    return emb


# ─────────────────────────────────────────────────────────────
# TEST A: PH on raw delay embeddings
# ─────────────────────────────────────────────────────────────
def test_a_raw_embedding(data_dir, n_subjects=3, n_epochs_per_stage=20,
                          n_sub=500, seed=42):
    """Compute PH on Takens delay embeddings from raw EEG signals."""
    print(f"\n{'=' * 70}")
    print(f"  TEST A — PH on Raw Takens Delay Embedding (4D)")
    print(f"  {n_subjects} subjects, {n_epochs_per_stage} epochs/stage, "
          f"{n_sub} pts, Z_{COEFF}")
    print(f"{'=' * 70}\n")

    # Find subjects
    subjects = []
    for i in range(1, 200):
        sid = f"SN{i:03d}"
        psg = data_dir / f"{sid}.edf"
        hyp = data_dir / f"{sid}_sleepscoring.edf"
        if psg.exists() and hyp.exists() and psg.stat().st_size > 1_000_000:
            subjects.append((sid, psg, hyp))
        if len(subjects) >= n_subjects:
            break

    if not subjects:
        print("  No EDF files found. Skipping TEST A.")
        return None

    rng = np.random.default_rng(seed)
    stage_betti = {s: [] for s in STAGES}  # stage -> list of (b0, b1, b2)
    stage_maxpers = {s: [] for s in STAGES}
    ctrl_betti_shuf = []
    ctrl_betti_gauss = []

    for sid, psg, hyp in subjects:
        print(f"  {sid}...", end="", flush=True)

        # Load EEG channel
        raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False)
        ch = None
        for name in ["EEG C4-M1", "C4-M1", "EEG C4"]:
            if name in raw.ch_names:
                ch = name
                break
        if ch is None:
            print(" skip (no EEG)")
            continue

        raw.pick([ch])
        sfreq = raw.info["sfreq"]
        if sfreq != 100:
            raw.resample(100, verbose=False)
            sfreq = 100
        raw.filter(0.5, 30.0, verbose=False)
        signal = raw.get_data()[0]

        # Read annotations
        annots = mne.read_annotations(str(hyp))
        n_ep = int(len(signal) / sfreq // 30)
        labels = [None] * n_ep
        for onset, dur, desc in zip(annots.onset, annots.duration, annots.description):
            stage = HMC_LABELS.get(str(desc).strip())
            if stage is None:
                continue
            s = int(onset // 30)
            for e in range(s, min(s + max(1, int(dur // 30)), n_ep)):
                labels[e] = stage

        # Estimate tau from a few epochs
        taus = []
        for i in range(min(5, n_ep)):
            ep = signal[i * 3000:(i + 1) * 3000]
            if len(ep) == 3000 and np.max(np.abs(ep)) < 500e-6:
                taus.append(estimate_tau(ep))
        tau = int(np.median(taus)) if taus else 25
        print(f" tau={tau}", end="", flush=True)

        # Process epochs per stage
        for stage in STAGES:
            stage_epochs = [i for i in range(n_ep)
                            if labels[i] == stage
                            and (i + 1) * 3000 <= len(signal)
                            and np.max(np.abs(signal[i * 3000:(i + 1) * 3000])) < 500e-6]

            if len(stage_epochs) < 5:
                continue

            chosen = rng.choice(stage_epochs,
                                size=min(n_epochs_per_stage, len(stage_epochs)),
                                replace=False)

            for ep_idx in chosen:
                ep_signal = signal[ep_idx * 3000:(ep_idx + 1) * 3000]
                emb = takens_embed(ep_signal, d=4, tau=tau)
                if emb is None:
                    continue

                # Subsample
                if len(emb) > n_sub:
                    idx = rng.choice(len(emb), size=n_sub, replace=False)
                    cloud = emb[idx]
                else:
                    cloud = emb

                cloud = StandardScaler().fit_transform(cloud)

                try:
                    ph = compute_ph(cloud, maxdim=2)
                    stage_betti[stage].append(ph["betti"])
                    stage_maxpers[stage].append(ph["max_persistence"])
                except Exception:
                    continue

            # One shuffled control per stage per subject
            if stage_epochs:
                ep_signal = signal[stage_epochs[0] * 3000:(stage_epochs[0] + 1) * 3000]
                emb = takens_embed(ep_signal, d=4, tau=tau)
                if emb is not None:
                    cloud_s = emb.copy()
                    for col in range(cloud_s.shape[1]):
                        rng.shuffle(cloud_s[:, col])
                    if len(cloud_s) > n_sub:
                        cloud_s = cloud_s[rng.choice(len(cloud_s), n_sub, replace=False)]
                    cloud_s = StandardScaler().fit_transform(cloud_s)
                    try:
                        ph_s = compute_ph(cloud_s, maxdim=2)
                        ctrl_betti_shuf.append(ph_s["betti"])
                    except Exception:
                        pass

        # Gaussian control
        cloud_g = rng.standard_normal((n_sub, 4))
        cloud_g = StandardScaler().fit_transform(cloud_g)
        try:
            ph_g = compute_ph(cloud_g, maxdim=2)
            ctrl_betti_gauss.append(ph_g["betti"])
        except Exception:
            pass

        print(" done")

    # ── Report ────────────────────────────────────────────────
    print(f"\n  {'Stage':<8} {'n':>5} {'beta_0':>8} {'beta_1':>8} {'beta_2':>8}  "
          f"{'H1 max':>8}")
    print(f"  {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*8}  {'-'*8}")

    all_real_b1 = []
    for stage in STAGES:
        arr = np.array(stage_betti[stage]) if stage_betti[stage] else np.zeros((0, 3))
        mp = np.array(stage_maxpers[stage]) if stage_maxpers[stage] else np.zeros((0, 3))
        if len(arr) == 0:
            print(f"  {stage:<8} {'0':>5}")
            continue
        mean_b = arr.mean(axis=0)
        mean_mp = mp.mean(axis=0)
        all_real_b1.extend(arr[:, 1].tolist())
        marker = " <--" if abs(mean_b[1] - 2.0) < 0.5 else ""
        print(f"  {stage:<8} {len(arr):>5} {mean_b[0]:>8.1f} {mean_b[1]:>8.2f} "
              f"{mean_b[2]:>8.2f}  {mean_mp[1]:>8.3f}{marker}")

    # Controls
    ctrl_s = np.array(ctrl_betti_shuf) if ctrl_betti_shuf else np.zeros((0, 3))
    ctrl_g = np.array(ctrl_betti_gauss) if ctrl_betti_gauss else np.zeros((0, 3))

    print(f"\n  Controls:")
    if len(ctrl_s) > 0:
        print(f"  {'Shuffled':<8} {len(ctrl_s):>5} {ctrl_s[:, 0].mean():>8.1f} "
              f"{ctrl_s[:, 1].mean():>8.2f} {ctrl_s[:, 2].mean():>8.2f}")
    if len(ctrl_g) > 0:
        print(f"  {'Gaussian':<8} {len(ctrl_g):>5} {ctrl_g[:, 0].mean():>8.1f} "
              f"{ctrl_g[:, 1].mean():>8.2f} {ctrl_g[:, 2].mean():>8.2f}")

    # Permutation p-value for beta_1 >= 2
    real_b1_mean = np.mean(all_real_b1) if all_real_b1 else 0
    ctrl_b1_all = list(ctrl_s[:, 1]) + list(ctrl_g[:, 1]) if len(ctrl_s) + len(ctrl_g) > 0 else [0]
    p_val = np.mean([c >= real_b1_mean for c in ctrl_b1_all]) if ctrl_b1_all else 1.0

    print(f"\n  mean beta_1 (real): {real_b1_mean:.2f}")
    print(f"  mean beta_1 (ctrl): {np.mean(ctrl_b1_all):.2f}")
    print(f"  p-value (ctrl >= real): {p_val:.3f}")

    return {
        "stage_betti": stage_betti,
        "stage_maxpers": stage_maxpers,
        "ctrl_shuf": ctrl_betti_shuf,
        "ctrl_gauss": ctrl_betti_gauss,
    }


# ─────────────────────────────────────────────────────────────
# TEST B: PH on feature space
# ─────────────────────────────────────────────────────────────
def test_b_feature_space(X_all, stages, n_sub=500, n_pca=15, n_perm=20, seed=42):
    """PH on combined feature space (torus + spectral = 128D)."""
    print(f"\n{'=' * 70}")
    print(f"  TEST B — PH on Feature Space (128D -> PCA {n_pca})")
    print(f"  {n_sub} pts, Z_{COEFF}, {n_perm} permutations")
    print(f"{'=' * 70}\n")

    rng = np.random.default_rng(seed)
    results = {}

    # Per stage
    for s_int, s_name in enumerate(STAGES):
        mask = stages == s_int
        n_avail = mask.sum()
        if n_avail < 100:
            continue

        X_s = subsample_and_scale(X_all[mask], n_sub, n_pca, seed)
        ph = compute_ph(X_s, maxdim=2)

        # Permutation controls
        perm_b1 = []
        for p in range(n_perm):
            X_perm = X_all[mask].copy()
            idx_p = rng.choice(len(X_perm), size=min(n_sub, len(X_perm)), replace=False)
            X_perm = X_perm[idx_p]
            for col in range(X_perm.shape[1]):
                rng.shuffle(X_perm[:, col])
            X_perm = StandardScaler().fit_transform(X_perm)
            if n_pca > 0 and n_pca < X_perm.shape[1]:
                X_perm = PCA(n_components=n_pca, random_state=seed + p).fit_transform(X_perm)
            try:
                ph_p = compute_ph(X_perm, maxdim=2)
                perm_b1.append(ph_p["betti"][1])
            except Exception:
                perm_b1.append(0)

        p_val = np.mean([pb >= ph["betti"][1] for pb in perm_b1])

        results[s_name] = {
            "betti": ph["betti"],
            "max_pers": ph["max_persistence"],
            "lifetimes": ph["lifetimes"],
            "perm_b1_mean": np.mean(perm_b1),
            "perm_b1_std": np.std(perm_b1),
            "p_val": p_val,
        }

        b = ph["betti"]
        lt1 = ph["lifetimes"][1]
        top3 = ", ".join(f"{v:.3f}" for v in lt1[:3]) if len(lt1) >= 3 else str(lt1)
        marker = " <-- beta_1=2!" if b[1] == 2 else ""
        print(f"  {s_name:<6} beta=[{b[0]}, {b[1]}, {b[2]}]  "
              f"H1 top=[{top3}]  "
              f"perm_b1={np.mean(perm_b1):.1f}+/-{np.std(perm_b1):.1f}  "
              f"p={p_val:.3f}{marker}")

    # Gaussian control
    X_gauss = rng.standard_normal((n_sub, X_all.shape[1]))
    X_gauss = subsample_and_scale(X_gauss, n_sub, n_pca, seed + 999)
    ph_g = compute_ph(X_gauss, maxdim=2)
    bg = ph_g["betti"]
    print(f"\n  Gaussian: beta=[{bg[0]}, {bg[1]}, {bg[2]}]  "
          f"H1 max_pers={ph_g['max_persistence'][1]:.3f}")

    results["GAUSSIAN"] = {"betti": bg, "max_pers": ph_g["max_persistence"]}
    return results


# ─────────────────────────────────────────────────────────────
# TEST C: Per-channel PH
# ─────────────────────────────────────────────────────────────
def test_c_per_channel(X_torus, stages, n_sub=500, n_pca=10, n_perm=10, seed=42):
    """PH per channel (24D torus features each)."""
    print(f"\n{'=' * 70}")
    print(f"  TEST C — Per-Channel PH (24D -> PCA {n_pca})")
    print(f"  {n_sub} pts, Z_{COEFF}, {n_perm} permutations")
    print(f"{'=' * 70}\n")

    rng = np.random.default_rng(seed)
    results = {}

    for ch_name, ch_slice in CHANNELS.items():
        print(f"  --- {ch_name} ---")
        results[ch_name] = {}

        for s_int, s_name in enumerate(STAGES):
            mask = stages == s_int
            if mask.sum() < 100:
                continue

            X_ch = X_torus[mask][:, ch_slice]
            X_s = subsample_and_scale(X_ch, n_sub, n_pca, seed)
            ph = compute_ph(X_s, maxdim=2)

            # Permutation controls
            perm_b1 = []
            for p in range(n_perm):
                X_perm = X_ch.copy()
                idx_p = rng.choice(len(X_perm), size=min(n_sub, len(X_perm)), replace=False)
                X_perm = X_perm[idx_p]
                for col in range(X_perm.shape[1]):
                    rng.shuffle(X_perm[:, col])
                X_perm = StandardScaler().fit_transform(X_perm)
                if n_pca > 0 and n_pca < X_perm.shape[1]:
                    X_perm = PCA(n_components=n_pca, random_state=seed + p).fit_transform(X_perm)
                try:
                    ph_p = compute_ph(X_perm, maxdim=2)
                    perm_b1.append(ph_p["betti"][1])
                except Exception:
                    perm_b1.append(0)

            p_val = np.mean([pb >= ph["betti"][1] for pb in perm_b1])
            b = ph["betti"]
            results[ch_name][s_name] = {
                "betti": b, "max_pers": ph["max_persistence"], "p_val": p_val,
            }

            marker = " <--" if b[1] == 2 else ""
            print(f"    {s_name:<6} beta=[{b[0]}, {b[1]}, {b[2]}]  "
                  f"H1 max={ph['max_persistence'][1]:.3f}  "
                  f"p={p_val:.2f}{marker}")
        print()

    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Adapted Persistent Homology for NeuroSpiral"
    )
    parser.add_argument("--data", type=Path,
                        default=PROJECT_ROOT / "results" / "combined_features.npz")
    parser.add_argument("--edf-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "hmc")
    parser.add_argument("--skip-a", action="store_true",
                        help="Skip TEST A (raw EDF loading)")
    parser.add_argument("--n-sub", type=int, default=500)
    parser.add_argument("--n-pca-b", type=int, default=15)
    parser.add_argument("--n-pca-c", type=int, default=10)
    parser.add_argument("--n-perm", type=int, default=20)
    parser.add_argument("--n-subjects-a", type=int, default=3)
    parser.add_argument("--n-epochs-a", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()

    print()
    print("=" * 70)
    print("  NEUROSPIRAL — Adapted Persistent Homology")
    print(f"  Coefficients: Z_{COEFF} (same as Gardner et al. 2022)")
    print(f"  Betti detection: gap method (ratio >= 2.0)")
    print("=" * 70)

    # ── TEST A ────────────────────────────────────────────────
    result_a = None
    if not args.skip_a:
        result_a = test_a_raw_embedding(
            args.edf_dir,
            n_subjects=args.n_subjects_a,
            n_epochs_per_stage=args.n_epochs_a,
            n_sub=args.n_sub,
            seed=args.seed,
        )

    # ── Load feature data for TEST B and C ────────────────────
    print(f"\n  Loading {args.data}...")
    d = np.load(args.data)
    X_torus = d["torus_individual"]   # (N, 96)
    X_spectral = d["spectral"]        # (N, 32)
    stages = d["stages"]
    X_combined = np.hstack([X_torus, X_spectral])  # (N, 128)
    print(f"  Combined features: {X_combined.shape} (96 torus + 32 spectral)")

    # ── TEST B ────────────────────────────────────────────────
    result_b = test_b_feature_space(
        X_combined, stages,
        n_sub=args.n_sub, n_pca=args.n_pca_b, n_perm=args.n_perm, seed=args.seed,
    )

    # ── TEST C ────────────────────────────────────────────────
    result_c = test_c_per_channel(
        X_torus, stages,
        n_sub=args.n_sub, n_pca=args.n_pca_c, n_perm=args.n_perm, seed=args.seed,
    )

    # ── FINAL SUMMARY ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 70}\n")

    # Collect all beta_1 results
    print(f"  {'Test':<30} {'beta_1':>8} {'H1 max':>8} {'p-val':>8}  Note")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")

    # Test A
    if result_a:
        for stage in STAGES:
            arr = result_a["stage_betti"].get(stage, [])
            if arr:
                b1_mean = np.mean([x[1] for x in arr])
                mp_mean = np.mean([x[1] for x in result_a["stage_maxpers"][stage]])
                print(f"  {'A: ' + stage + ' embed (4D)':<30} {b1_mean:>8.2f} "
                      f"{mp_mean:>8.3f} {'':>8}  raw delay")

    # Test B
    for stage in STAGES:
        if stage in result_b:
            r = result_b[stage]
            b1 = r["betti"][1]
            mp = r["max_pers"][1]
            pv = r["p_val"]
            marker = "TORUS!" if b1 == 2 and pv < 0.05 else ""
            print(f"  {'B: ' + stage + ' feat (128D)':<30} {b1:>8} "
                  f"{mp:>8.3f} {pv:>8.3f}  {marker}")

    # Test C
    for ch_name in CHANNELS:
        for stage in ["REM", "N3", "W"]:  # most interesting stages
            if stage in result_c.get(ch_name, {}):
                r = result_c[ch_name][stage]
                b1 = r["betti"][1]
                mp = r["max_pers"][1]
                pv = r["p_val"]
                marker = "TORUS!" if b1 == 2 and pv < 0.05 else ""
                print(f"  {'C: ' + stage + '/' + ch_name + ' (24D)':<30} {b1:>8} "
                      f"{mp:>8.3f} {pv:>8.3f}  {marker}")

    # ── VERDICT ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}\n")

    # Check for beta_1=2 with p < 0.05
    b1_2_cases = []
    for stage in STAGES:
        if stage in result_b and result_b[stage]["betti"][1] == 2:
            if result_b[stage]["p_val"] < 0.05:
                b1_2_cases.append(f"B:{stage}")
    for ch in result_c:
        for stage in result_c[ch]:
            if result_c[ch][stage]["betti"][1] == 2:
                if result_c[ch][stage]["p_val"] < 0.05:
                    b1_2_cases.append(f"C:{stage}/{ch}")

    # Check Test A for mean beta_1 near 2
    a_near_2 = []
    if result_a:
        for stage in STAGES:
            arr = result_a["stage_betti"].get(stage, [])
            if arr:
                b1_mean = np.mean([x[1] for x in arr])
                if 1.5 <= b1_mean <= 2.5:
                    a_near_2.append(f"A:{stage}(mean={b1_mean:.2f})")

    if b1_2_cases:
        print(f"  [+++] beta_1=2 with p<0.05 found in: {', '.join(b1_2_cases)}")
        print(f"        The signal has 2 detectable periodicities.")
        print(f"        Toroidal topology is REAL (not artifact).")
    elif a_near_2:
        print(f"  [++] Raw embeddings show beta_1 near 2: {', '.join(a_near_2)}")
        print(f"       Toroidal topology visible at signal level.")
    else:
        # Check if any beta_1 > controls
        any_sig = any(
            result_b[s]["p_val"] < 0.05
            for s in result_b if s != "GAUSSIAN" and "p_val" in result_b[s]
        )
        if any_sig:
            print(f"  [+] beta_1 significantly above shuffled controls (p<0.05)")
            print(f"      but beta_1 != 2. The torus is a useful feature extractor")
            print(f"      but the topology is more complex than T^2.")
        else:
            print(f"  [---] No significant toroidal topology detected.")
            print(f"        The Clifford torus is a useful COMPUTATIONAL tool")
            print(f"        but the data does not live on a topological torus.")

    print(f"\n  Note: Gardner et al. (2022) used single-neuron recordings")
    print(f"  from Neuropixels in entorhinal cortex. Our EEG/ECG/EOG/EMG")
    print(f"  operates at a fundamentally different scale. Different beta_1")
    print(f"  does NOT invalidate the torus as a feature extraction method.")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
