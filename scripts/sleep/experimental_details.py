#!/usr/bin/env python3
"""NeuroSpiral — Experimental Details for Paper (Reviewer Condition 3).

FNN analysis, tau values, preprocessing docs, RF sensitivity,
per-class metrics, and feature correlation matrix.

Usage:
    python scripts/experimental_details.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import mne
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (cohen_kappa_score, f1_score, confusion_matrix,
                             classification_report, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGES = ["W", "N1", "N2", "N3", "REM"]

HMC_LABELS = {
    "Sleep stage W": "W", "Sleep stage N1": "N1", "Sleep stage N2": "N2",
    "Sleep stage N3": "N3", "Sleep stage R": "REM", "Sleep stage ?": None,
}


# ─────────────────────────────────────────────────────────────
# 1. FALSE NEAREST NEIGHBORS
# ─────────────────────────────────────────────────────────────
def mutual_information(x, y, n_bins=64):
    hist, _, _ = np.histogram2d(x, y, bins=n_bins)
    p = hist / hist.sum()
    px, py = p.sum(1), p.sum(0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p[i,j] > 1e-12 and px[i] > 1e-12 and py[j] > 1e-12:
                mi += p[i,j] * np.log(p[i,j] / (px[i] * py[j]))
    return mi


def estimate_tau(signal, max_lag=100):
    max_lag = min(max_lag, len(signal) // 4)
    mi = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            mi[lag] = mutual_information(signal, signal)
        else:
            mi[lag] = mutual_information(signal[:-lag], signal[lag:])
    mins = argrelextrema(mi[1:], np.less)[0]
    if len(mins) > 0:
        return int(mins[0] + 1)
    below = np.where(mi[1:] < mi[0] / np.e)[0]
    return int(below[0] + 1) if len(below) > 0 else max(1, max_lag // 4)


def takens_embed(signal, d, tau):
    n_emb = len(signal) - (d - 1) * tau
    if n_emb < 50:
        return None
    emb = np.zeros((n_emb, d))
    for i in range(d):
        emb[:, i] = signal[i * tau: i * tau + n_emb]
    return emb


def compute_fnn(signal, tau, d_max=7, threshold=15.0):
    """False Nearest Neighbors for dimensions 2..d_max.

    For each point in d-dimensional embedding, find its nearest neighbor.
    If the distance increases drastically when going to d+1 dimensions,
    the neighbor is "false" — caused by projection, not proximity.

    Returns dict: d -> FNN percentage.
    """
    results = {}
    for d in range(2, d_max + 1):
        emb_d = takens_embed(signal, d, tau)
        emb_d1 = takens_embed(signal, d + 1, tau)
        if emb_d is None or emb_d1 is None:
            results[d] = 100.0
            continue

        n = min(len(emb_d), len(emb_d1))
        emb_d = emb_d[:n]
        emb_d1 = emb_d1[:n]

        # Subsample for speed
        if n > 2000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, 2000, replace=False)
        else:
            idx = np.arange(n)

        n_false = 0
        n_total = 0

        for i in idx:
            # Find nearest neighbor in d dimensions (exclude self)
            dists_d = np.sum((emb_d - emb_d[i]) ** 2, axis=1)
            dists_d[i] = np.inf
            nn = np.argmin(dists_d)
            r_d = np.sqrt(dists_d[nn])

            if r_d < 1e-15:
                continue

            # Check if neighbor is false: |x_{d+1}(i) - x_{d+1}(nn)| / r_d > threshold
            extra_dist = abs(emb_d1[i, -1] - emb_d1[nn, -1])
            if extra_dist / r_d > threshold:
                n_false += 1
            n_total += 1

        results[d] = 100.0 * n_false / max(n_total, 1)
    return results


def main():
    t_start = time.time()
    results_dir = PROJECT_ROOT / "results"

    print()
    print("=" * 70)
    print("  NEUROSPIRAL — Experimental Details for Paper")
    print("  Reviewer Condition 3: justify all methodological choices")
    print("=" * 70)

    # Load combined features
    d = np.load(results_dir / "combined_features.npz")
    X_torus = d["torus_individual"]
    X_spectral = d["spectral"]
    X = np.hstack([X_torus, X_spectral])
    y = d["stages"]
    subjects = d["subjects"]
    fn_t = list(d["feature_names_torus"])
    fn_s = list(d["feature_names_spectral"])

    # ── 1. FALSE NEAREST NEIGHBORS ────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  1. FALSE NEAREST NEIGHBORS — Justifying d=4")
    print(f"{'=' * 70}\n")

    data_dir = PROJECT_ROOT / "data" / "hmc"
    rng = np.random.default_rng(42)

    # Find 10 random subjects
    subj_files = []
    for i in range(1, 200):
        sid = f"SN{i:03d}"
        psg = data_dir / f"{sid}.edf"
        hyp = data_dir / f"{sid}_sleepscoring.edf"
        if psg.exists() and hyp.exists() and psg.stat().st_size > 1_000_000:
            subj_files.append((sid, psg, hyp))
    rng.shuffle(subj_files)
    subj_files = subj_files[:10]

    all_fnn = {d_val: [] for d_val in range(2, 8)}
    tau_values_eeg = []

    print(f"  Analyzing {len(subj_files)} subjects, 5 epochs each...\n")

    for sid, psg, hyp in subj_files:
        raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False)
        ch = None
        for name in ["EEG C4-M1", "C4-M1", "EEG C4"]:
            if name in raw.ch_names:
                ch = name
                break
        if ch is None:
            continue

        raw.pick([ch])
        if raw.info["sfreq"] != 100:
            raw.resample(100, verbose=False)
        raw.filter(0.5, 30.0, verbose=False)
        signal = raw.get_data()[0]

        # Read annotations for stage labels
        annots = mne.read_annotations(str(hyp))
        n_ep = int(len(signal) / 100 // 30)
        labels = [None] * n_ep
        for onset, dur, desc in zip(annots.onset, annots.duration, annots.description):
            stage = HMC_LABELS.get(str(desc).strip())
            if stage is None:
                continue
            s = int(onset // 30)
            for e in range(s, min(s + max(1, int(dur // 30)), n_ep)):
                labels[e] = stage

        # Pick 5 N2 epochs (most common)
        n2_epochs = [i for i in range(n_ep)
                     if labels[i] == "N2"
                     and (i + 1) * 3000 <= len(signal)
                     and np.max(np.abs(signal[i*3000:(i+1)*3000])) < 500e-6]
        if len(n2_epochs) < 5:
            continue

        chosen = rng.choice(n2_epochs, 5, replace=False)

        for ep_idx in chosen:
            ep = signal[ep_idx * 3000:(ep_idx + 1) * 3000]
            tau = estimate_tau(ep)
            tau_values_eeg.append(tau)

            fnn = compute_fnn(ep, tau, d_max=7)
            for d_val, pct in fnn.items():
                all_fnn[d_val].append(pct)

        print(f"  {sid}: tau_median={int(np.median(tau_values_eeg[-5:]))}, "
              f"FNN(d=4)={np.mean([all_fnn[4][-i] for i in range(1, min(6, len(all_fnn[4])+1))]):.1f}%")

    print(f"\n  FNN Results (mean ± std across all epochs):")
    print(f"  {'d':>4} {'FNN %':>10} {'< 1%?':>8}")
    print(f"  {'-'*4} {'-'*10} {'-'*8}")
    for d_val in range(2, 8):
        vals = all_fnn[d_val]
        if vals:
            mean_fnn = np.mean(vals)
            std_fnn = np.std(vals)
            check = "YES" if mean_fnn < 1.0 else "no"
            marker = " <-- d=4" if d_val == 4 else ""
            print(f"  {d_val:>4} {mean_fnn:>8.2f}% ± {std_fnn:.2f}  {check:>6}{marker}")

    if tau_values_eeg:
        print(f"\n  EEG tau distribution: median={np.median(tau_values_eeg):.0f}, "
              f"IQR=[{np.percentile(tau_values_eeg, 25):.0f}, "
              f"{np.percentile(tau_values_eeg, 75):.0f}]")

    # ── 2. TAU VALUES ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  2. TAU VALUES PER CHANNEL (from pipeline code)")
    print(f"{'=' * 70}\n")

    print(f"  Channel   Tau values (multi-scale)")
    print(f"  --------- ------------------------")
    print(f"  EEG       tau = 10, 25, 40")
    print(f"  ECG       tau = 10, 25, 40")
    print(f"  EOG       tau = 15, 30, 50")
    print(f"  EMG       tau = 5, 10, 20")
    print(f"\n  All tau values are in SAMPLES at 100 Hz.")
    print(f"  EEG: 0.10s, 0.25s, 0.40s (delta to alpha timescales)")
    print(f"  ECG: 0.10s, 0.25s, 0.40s (cardiac cycle dynamics)")
    print(f"  EOG: 0.15s, 0.30s, 0.50s (eye movement timescales)")
    print(f"  EMG: 0.05s, 0.10s, 0.20s (faster muscle dynamics)")

    # ── 3. PREPROCESSING ──────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  3. PREPROCESSING DOCUMENTATION")
    print(f"{'=' * 70}\n")

    print(f"  Channel   Bandpass (Hz)   Filter    Resample  Artifact")
    print(f"  --------- -------------- --------- --------- --------")
    print(f"  EEG       0.5 - 30.0     MNE FIR   100 Hz    >500 uV")
    print(f"  ECG       0.5 - 40.0     MNE FIR   100 Hz    -")
    print(f"  EOG       0.3 - 35.0     MNE FIR   100 Hz    -")
    print(f"  EMG       10.0 - 49.0    MNE FIR   100 Hz    -")
    print(f"\n  MNE FIR: firwin design, default order (~6.6x sfreq)")
    print(f"  Epoch length: 30 seconds (3000 samples at 100 Hz)")
    print(f"  Artifact rejection: EEG peak-to-peak > 500 uV")
    print(f"  No ICA applied in the combined pipeline")
    print(f"  No z-scoring per subject (tree-based classifiers handle this)")

    # ── 4. RF SENSITIVITY ANALYSIS ────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  4. RF SENSITIVITY ANALYSIS")
    print(f"{'=' * 70}\n")

    X_s = StandardScaler().fit_transform(X)
    cv = StratifiedGroupKFold(n_splits=5)

    n_trees_list = [50, 100, 200, 500]
    depth_list = [6, 8, 10, 12, 15, None]

    print(f"  {'n_trees':>8}", end="")
    for depth in depth_list:
        d_str = str(depth) if depth else "None"
        print(f"  {d_str:>8}", end="")
    print()
    print(f"  {'-'*8}", end="")
    for _ in depth_list:
        print(f"  {'-'*8}", end="")
    print()

    best_kappa = 0
    best_config = ""
    rf_results = {}

    for n_trees in n_trees_list:
        print(f"  {n_trees:>8}", end="", flush=True)
        for depth in depth_list:
            clf = RandomForestClassifier(
                n_estimators=n_trees, max_depth=depth,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            y_pred = cross_val_predict(clf, X_s, y, groups=subjects, cv=cv)
            kappa = cohen_kappa_score(y, y_pred)
            rf_results[(n_trees, depth)] = kappa
            print(f"  {kappa:>8.3f}", end="", flush=True)
            if kappa > best_kappa:
                best_kappa = kappa
                best_config = f"n_trees={n_trees}, max_depth={depth}"
        print()

    print(f"\n  Best: {best_config} -> kappa={best_kappa:.3f}")

    # Check sensitivity: range across all configs
    all_kappas = list(rf_results.values())
    print(f"  Range: [{min(all_kappas):.3f}, {max(all_kappas):.3f}] "
          f"(spread={max(all_kappas)-min(all_kappas):.3f})")

    # ── 5. PER-CLASS METRICS ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  5. PER-CLASS METRICS (128f, best RF config)")
    print(f"{'=' * 70}\n")

    # Use best config for detailed report
    best_nt, best_md = max(rf_results, key=rf_results.get)
    clf_best = RandomForestClassifier(
        n_estimators=best_nt, max_depth=best_md,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    y_pred_best = cross_val_predict(clf_best, X_s, y, groups=subjects, cv=cv)

    print(classification_report(y, y_pred_best, target_names=STAGES,
                                 digits=3, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred_best)
    print(f"  Confusion matrix:")
    print(f"  {'':>6}", end="")
    for s in STAGES:
        print(f"  {s:>6}", end="")
    print()
    for i, s in enumerate(STAGES):
        print(f"  {s:>6}", end="")
        for j in range(5):
            print(f"  {cm[i,j]:>6}", end="")
        # Biggest confusion
        row = cm[i].copy()
        row[i] = 0
        max_conf_idx = np.argmax(row)
        max_conf_pct = row[max_conf_idx] / cm[i].sum() * 100
        print(f"  -> {STAGES[max_conf_idx]} ({max_conf_pct:.1f}%)")

    # Per-class sensitivity/specificity
    print(f"\n  Per-class sensitivity and specificity:")
    print(f"  {'Stage':<6} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'F1':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for s_int, s_name in enumerate(STAGES):
        tp = cm[s_int, s_int]
        fn = cm[s_int].sum() - tp
        fp = cm[:, s_int].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        f1 = 2 * ppv * sens / max(ppv + sens, 1e-10)
        print(f"  {s_name:<6} {sens:>8.3f} {spec:>8.3f} {ppv:>8.3f} {f1:>8.3f}")

    # ── 6. FEATURE CORRELATION MATRIX ─────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  6. FEATURE CORRELATION MATRIX (8 base features)")
    print(f"{'=' * 70}\n")

    base_feat = ["omega1", "curvature", "accel", "geodesic",
                  "entropy", "phase_std", "coherence", "transition"]

    # Average across all tau and channels: take EEG tau=25 as reference
    # omega1_eeg_t25 = index 8, curvature_eeg_t25 = 9, etc.
    eeg_t25_start = 8  # second set of 8 (t25)
    X_8 = X_torus[:, eeg_t25_start:eeg_t25_start + 8]

    corr = np.corrcoef(X_8.T)

    print(f"  Correlation matrix (EEG, tau=25):")
    print(f"  {'':>12}", end="")
    for fn in base_feat:
        print(f"  {fn[:7]:>7}", end="")
    print()
    for i, fn1 in enumerate(base_feat):
        print(f"  {fn1[:12]:<12}", end="")
        for j in range(8):
            r = corr[i, j]
            marker = "*" if abs(r) > 0.8 and i != j else " "
            print(f"  {r:>6.3f}{marker}", end="")
        print()

    # Identify highly correlated pairs
    high_corr = []
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(corr[i, j]) > 0.8:
                high_corr.append((base_feat[i], base_feat[j], corr[i, j]))

    if high_corr:
        print(f"\n  Highly correlated pairs (|r| > 0.8):")
        for f1, f2, r in high_corr:
            print(f"    {f1} - {f2}: r = {r:.3f}")
    else:
        print(f"\n  No pairs with |r| > 0.8 — features are non-redundant")

    # Also check across channels for omega1
    print(f"\n  Cross-channel omega1 correlation:")
    omega1_cols = [0, 24, 48, 72]  # omega1 at t10 for each channel
    ch_names = ["EEG", "ECG", "EOG", "EMG"]
    omega1_cross = X_torus[:, omega1_cols]
    cc = np.corrcoef(omega1_cross.T)
    print(f"  {'':>6}", end="")
    for cn in ch_names:
        print(f"  {cn:>6}", end="")
    print()
    for i, cn in enumerate(ch_names):
        print(f"  {cn:>6}", end="")
        for j in range(4):
            print(f"  {cc[i,j]:>6.3f}", end="")
        print()

    # ── SAVE ──────────────────────────────────────────────────
    np.savez_compressed(
        results_dir / "experimental_details.npz",
        fnn_results={d_val: np.array(all_fnn[d_val]) for d_val in range(2, 8)},
        tau_eeg_values=np.array(tau_values_eeg) if tau_values_eeg else np.array([]),
        rf_sensitivity=np.array([(nt, md if md else -1, k)
                                  for (nt, md), k in rf_results.items()]),
        confusion_matrix=cm,
        feature_correlation=corr,
        best_kappa=best_kappa,
    )

    elapsed = time.time() - t_start
    print(f"\n  Saved: results/experimental_details.npz")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
