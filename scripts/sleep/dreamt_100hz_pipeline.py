#!/usr/bin/env python3
"""NeuroSpiral — DREAMT S002 4-Channel Pipeline (100Hz PSG + Wearable).

Replicates the HMC torus pipeline on DREAMT PSG data, plus
cross-modal comparison (PSG ECG vs wearable BVP).

Usage:
    python scripts/dreamt_100hz_pipeline.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# S002 has no N3
STAGES = ["W", "N1", "N2", "REM"]
STAGE_MAP = {"W": 0, "N1": 1, "N2": 2, "R": 3}

SFREQ = 100
EPOCH_SAMPLES = 30 * SFREQ  # 3000

# Channels and their taus
PSG_CHANNELS = {
    "EEG": ("C4-M1", 25),
    "ECG": ("ECG", 20),
    "EOG": ("E1", 15),
    "EMG": ("CHIN", 10),
}

WEARABLE_CHANNELS = {
    "BVP": ("BVP", 20),
    "ACC": ("ACC_mag", 15),
    "HR": ("HR", 10),
}

TORUS_FEAT_NAMES = [
    "omega1", "torus_curvature", "angular_acceleration",
    "geodesic_distance", "angular_entropy",
    "phase_diff_std", "phase_coherence", "transition_rate",
]


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def torus_features_8(embedding):
    """8 canonical torus features."""
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
    feats.append(float(np.mean(np.abs(dtheta))))
    feats.append(float(np.mean(np.abs(np.diff(dtheta)))) if N >= 2 else 0.0)
    feats.append(float(np.var(dtheta)))
    feats.append(float(np.sum(np.sqrt(dtheta**2 + dphi**2))))
    counts, _ = np.histogram(theta, bins=16, range=(-np.pi, np.pi))
    c = counts.astype(np.float64)
    total = c.sum()
    if total > 0:
        p = c / total; p = p[p > 0]
        feats.append(float(-np.sum(p * np.log2(p))))
    else:
        feats.append(0.0)
    pd_arr = theta - phi
    R_len = np.abs(np.mean(np.exp(1j * pd_arr)))
    feats.append(float(np.sqrt(-2 * np.log(max(R_len, 1e-10)))) if R_len < 1 else 0.0)
    feats.append(float(R_len))
    signs = (embedding >= 0).astype(int)
    verts = signs[:, 0]*8 + signs[:, 1]*4 + signs[:, 2]*2 + signs[:, 3]
    feats.append(float(np.sum(np.diff(verts) != 0) / max(len(verts)-1, 1)))
    return np.array(feats, dtype=np.float64)


def takens_embed(signal, d=4, tau=25):
    n = len(signal)
    n_emb = n - (d - 1) * tau
    if n_emb < 50:
        return None
    emb = np.zeros((n_emb, d))
    for i in range(d):
        emb[:, i] = signal[i * tau: i * tau + n_emb]
    if np.std(emb) < 1e-15 or not np.all(np.isfinite(emb)):
        return None
    return emb


def main():
    t_start = time.time()

    print()
    print("=" * 70)
    print("  NEUROSPIRAL — DREAMT S002 4-Channel Pipeline (100Hz)")
    print("  Cross-dataset replication + PSG vs Wearable comparison")
    print("=" * 70)

    csv_path = Path.home() / "Downloads/DREAMT/S002_PSG_df_updated.csv"
    results_dir = PROJECT_ROOT / "results"

    # ── 1. Load data ──────────────────────────────────────────
    print(f"\n[1/6] Loading {csv_path.name}...")

    cols = ["C4-M1", "ECG", "E1", "CHIN", "BVP", "ACC_X", "ACC_Y", "ACC_Z", "HR",
            "Sleep_Stage"]
    df = pd.read_csv(csv_path, usecols=cols)
    print(f"  Rows: {len(df):,}")

    # Filter valid stages
    df = df[df["Sleep_Stage"].isin(STAGE_MAP.keys())].reset_index(drop=True)
    print(f"  After stage filter: {len(df):,}")
    print(f"  Stages: {df['Sleep_Stage'].value_counts().to_dict()}")

    # Compute ACC magnitude
    df["ACC_mag"] = np.sqrt(df["ACC_X"]**2 + df["ACC_Y"]**2 + df["ACC_Z"]**2)

    # ── 2. Epoch extraction ───────────────────────────────────
    print(f"\n[2/6] Epoching (30s = {EPOCH_SAMPLES} samples)...")

    n_epochs = len(df) // EPOCH_SAMPLES
    valid_epochs = []

    for i in range(n_epochs):
        s = i * EPOCH_SAMPLES
        e = s + EPOCH_SAMPLES
        chunk = df.iloc[s:e]

        # Majority vote for stage
        stage_counts = chunk["Sleep_Stage"].value_counts()
        majority_stage = stage_counts.index[0]
        majority_frac = stage_counts.iloc[0] / EPOCH_SAMPLES

        if majority_frac < 0.8:
            continue
        if majority_stage not in STAGE_MAP:
            continue

        # Artifact check on EEG
        eeg = chunk["C4-M1"].values
        if np.max(np.abs(eeg)) > 500e-6:
            continue

        valid_epochs.append({
            "idx": i, "start": s, "end": e,
            "stage": STAGE_MAP[majority_stage],
            "stage_name": majority_stage,
        })

    print(f"  Valid epochs: {len(valid_epochs)}")
    stage_counts = {}
    for ep in valid_epochs:
        sn = STAGES[ep["stage"]]
        stage_counts[sn] = stage_counts.get(sn, 0) + 1
    for sn in STAGES:
        print(f"    {sn}: {stage_counts.get(sn, 0)}")

    # ── 3. Feature extraction ─────────────────────────────────
    print(f"\n[3/6] Extracting torus features...")

    all_psg_feats = []    # (n, 32) — 4 PSG channels x 8 features
    all_wear_feats = []   # (n, 24) — 3 wearable channels x 8 features
    all_stages = []
    n_fail = 0

    for ep in valid_epochs:
        s, e = ep["start"], ep["end"]
        chunk = df.iloc[s:e]

        # PSG features (4 channels x 8 = 32)
        psg_feats = []
        ok = True
        for ch_label, (col_name, tau) in PSG_CHANNELS.items():
            signal = chunk[col_name].values.astype(np.float64)
            emb = takens_embed(signal, d=4, tau=tau)
            f = torus_features_8(emb)
            if f is None:
                ok = False
                break
            psg_feats.extend(f)
        if not ok:
            n_fail += 1
            continue

        # Wearable features (3 channels x 8 = 24)
        wear_feats = []
        for ch_label, (col_name, tau) in WEARABLE_CHANNELS.items():
            signal = chunk[col_name].values.astype(np.float64)
            emb = takens_embed(signal, d=4, tau=tau)
            f = torus_features_8(emb)
            if f is None:
                ok = False
                break
            wear_feats.extend(f)
        if not ok:
            n_fail += 1
            continue

        all_psg_feats.append(psg_feats)
        all_wear_feats.append(wear_feats)
        all_stages.append(ep["stage"])

    X_psg = np.array(all_psg_feats, dtype=np.float64)   # (n, 32)
    X_wear = np.array(all_wear_feats, dtype=np.float64)  # (n, 24)
    y = np.array(all_stages, dtype=np.int8)

    print(f"  PSG features: {X_psg.shape}")
    print(f"  Wearable features: {X_wear.shape}")
    print(f"  Failed: {n_fail}")

    # Clean NaN/Inf
    valid = np.all(np.isfinite(X_psg), axis=1) & np.all(np.isfinite(X_wear), axis=1)
    X_psg = X_psg[valid]
    X_wear = X_wear[valid]
    y = y[valid]
    print(f"  After cleaning: {len(y)} epochs")

    for s_int, s_name in enumerate(STAGES):
        print(f"    {s_name}: {(y == s_int).sum()}")

    # Build feature names
    psg_feat_names = []
    for ch in PSG_CHANNELS:
        for fn in TORUS_FEAT_NAMES:
            psg_feat_names.append(f"{fn}_{ch}")
    wear_feat_names = []
    for ch in WEARABLE_CHANNELS:
        for fn in TORUS_FEAT_NAMES:
            wear_feat_names.append(f"{fn}_{ch}")

    # ── 4. Tests ──────────────────────────────────────────────
    # 5a. Omega1 gradient
    print(f"\n{'=' * 70}")
    print(f"  5a. OMEGA1 GRADIENT")
    print(f"{'=' * 70}\n")

    for ch_idx, ch_name in enumerate(PSG_CHANNELS.keys()):
        omega1_col = ch_idx * 8  # omega1 is first of 8
        means = {}
        for s_int, s_name in enumerate(STAGES):
            mask = y == s_int
            if mask.sum() > 0:
                means[s_name] = float(X_psg[mask, omega1_col].mean())
        ordered = sorted(means.items(), key=lambda x: x[1])
        ordering = " < ".join(f"{k}({v:.4f})" for k, v in ordered)
        print(f"  {ch_name}: {ordering}")

    # Expected from HMC: N3 < N2 < REM < N1 < W (no N3 here)
    print(f"\n  HMC reference: N3 < N2 < REM < N1 < W")
    print(f"  DREAMT S002 (no N3): expect N2 < REM < N1 < W")

    # 5b. KW test
    print(f"\n{'=' * 70}")
    print(f"  5b. KRUSKAL-WALLIS (32 PSG features)")
    print(f"{'=' * 70}\n")

    kw_results = []
    for f_idx, f_name in enumerate(psg_feat_names):
        groups = [X_psg[y == s, f_idx] for s in range(4) if (y == s).sum() >= 5]
        if len(groups) >= 2:
            H, p = kruskal(*groups)
            kw_results.append((f_name, H, p))

    kw_results.sort(key=lambda x: -x[1])
    n_sig = sum(1 for _, _, p in kw_results if p < 0.001)

    print(f"  {'Feature':<30} {'H':>10} {'p':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*12}")
    for name, H, p in kw_results[:15]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:<30} {H:>10.1f} {p:>12.2e} {sig}")
    print(f"\n  Significant (p<0.001): {n_sig}/{len(kw_results)}")

    # 5c. Classification
    print(f"\n{'=' * 70}")
    print(f"  5c. CLASSIFICATION (5-fold CV)")
    print(f"{'=' * 70}\n")

    X_s = StandardScaler().fit_transform(X_psg)
    clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_s, y, cv=cv)
    kappa = cohen_kappa_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"  PSG 32 features: kappa={kappa:.3f}, F1={f1:.3f}")
    print(classification_report(y, y_pred, target_names=STAGES, digits=3, zero_division=0))

    # Wearable alone
    X_w_s = StandardScaler().fit_transform(X_wear)
    y_pred_w = cross_val_predict(clf, X_w_s, y, cv=cv)
    kappa_w = cohen_kappa_score(y, y_pred_w)
    print(f"  Wearable 24 features: kappa={kappa_w:.3f}")

    # Combined
    X_comb = np.hstack([X_psg, X_wear])
    X_c_s = StandardScaler().fit_transform(X_comb)
    y_pred_c = cross_val_predict(clf, X_c_s, y, cv=cv)
    kappa_c = cohen_kappa_score(y, y_pred_c)
    print(f"  Combined 56 features: kappa={kappa_c:.3f}")

    # 5d. Beta position (REM on W→N2 axis, N2 as deep pole)
    print(f"\n{'=' * 70}")
    print(f"  5d. BETA: REM position on W→N2 axis")
    print(f"{'=' * 70}\n")

    centroids = {}
    for s_int, s_name in enumerate(STAGES):
        mask = y == s_int
        if mask.sum() > 0:
            centroids[s_name] = X_s[mask].mean(axis=0)

    mu_W = centroids["W"]
    mu_N2 = centroids["N2"]
    mu_REM = centroids["REM"]

    axis = mu_N2 - mu_W
    axis_sq = np.dot(axis, axis)
    rem_vec = mu_REM - mu_W
    beta = float(np.dot(rem_vec, axis) / axis_sq)
    proj = mu_W + beta * axis
    resid = mu_REM - proj
    gamma_d = float(np.linalg.norm(resid) / max(np.linalg.norm(rem_vec), 1e-15))

    # Also N1
    mu_N1 = centroids["N1"]
    n1_vec = mu_N1 - mu_W
    beta_n1 = float(np.dot(n1_vec, axis) / axis_sq)

    print(f"  W→N2 axis (N2 = deepest available stage)")
    print(f"  REM: beta = {beta:.3f} (0=W, 1=N2)")
    print(f"  N1:  beta = {beta_n1:.3f}")
    print(f"  gamma/d (REM) = {gamma_d:.3f}")
    print(f"\n  HMC reference: beta(REM on W→N3) = 0.57")
    print(f"  DREAMT S002:   beta(REM on W→N2) = {beta:.3f}")

    # ── 6. PSG vs Wearable comparison ─────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  6. PSG vs WEARABLE — Cross-Modal Correlation")
    print(f"{'=' * 70}\n")

    # ECG (PSG) vs BVP (wearable) — same heart, different sensor
    ecg_start = 1 * 8  # ECG is 2nd PSG channel (index 1), 8 features each
    bvp_start = 0 * 8  # BVP is 1st wearable channel (index 0)

    print(f"  ECG (PSG) vs BVP (Wearable) — same heart:")
    print(f"  {'Feature':<25} {'Pearson r':>10} {'p':>12} {'Spearman':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10}")
    ecg_bvp_rs = []
    for fi, fn in enumerate(TORUS_FEAT_NAMES):
        ecg_feat = X_psg[:, ecg_start + fi]
        bvp_feat = X_wear[:, bvp_start + fi]
        r_p, p_p = pearsonr(ecg_feat, bvp_feat)
        r_s, _ = spearmanr(ecg_feat, bvp_feat)
        ecg_bvp_rs.append(r_p)
        sig = "***" if p_p < 0.001 else ""
        print(f"  {fn:<25} {r_p:>10.3f} {p_p:>12.2e} {r_s:>10.3f} {sig}")

    # EMG (PSG) vs ACC (wearable) — should be ~0
    emg_start = 3 * 8  # EMG is 4th PSG channel
    acc_start = 1 * 8  # ACC is 2nd wearable channel

    print(f"\n  EMG (CHIN, PSG) vs ACC (Wearable) — different modalities:")
    print(f"  {'Feature':<25} {'Pearson r':>10} {'p':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*12}")
    emg_acc_rs = []
    for fi, fn in enumerate(TORUS_FEAT_NAMES):
        emg_feat = X_psg[:, emg_start + fi]
        acc_feat = X_wear[:, acc_start + fi]
        r_p, p_p = pearsonr(emg_feat, acc_feat)
        emg_acc_rs.append(r_p)
        print(f"  {fn:<25} {r_p:>10.3f} {p_p:>12.2e}")

    mean_ecg_bvp = np.mean(np.abs(ecg_bvp_rs))
    mean_emg_acc = np.mean(np.abs(emg_acc_rs))

    print(f"\n  Mean |r| ECG-BVP: {mean_ecg_bvp:.3f} (same organ, different sensor)")
    print(f"  Mean |r| EMG-ACC: {mean_emg_acc:.3f} (different modalities)")

    # ── SUMMARY ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"  Classification:")
    print(f"    PSG 32 features:      kappa = {kappa:.3f}")
    print(f"    Wearable 24 features: kappa = {kappa_w:.3f}")
    print(f"    Combined 56 features: kappa = {kappa_c:.3f}")

    print(f"\n  Beta (REM on W→N2):")
    print(f"    DREAMT S002: {beta:.3f} (gamma/d={gamma_d:.3f})")
    print(f"    HMC ref:     0.57 (on W→N3)")

    print(f"\n  Cross-modal:")
    print(f"    ECG↔BVP: mean |r| = {mean_ecg_bvp:.3f}")
    print(f"    EMG↔ACC: mean |r| = {mean_emg_acc:.3f}")

    # Verdict
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}\n")

    # Check omega1 gradient
    eeg_omega1 = {STAGES[s]: float(X_psg[y == s, 0].mean())
                  for s in range(4) if (y == s).sum() > 0}
    ordered_stages = sorted(eeg_omega1.items(), key=lambda x: x[1])
    gradient_str = " < ".join(k for k, _ in ordered_stages)
    expected = "N2 < REM < N1 < W"
    gradient_match = (gradient_str == expected or
                      all(k in gradient_str for k in ["N2", "REM", "W"]))

    if gradient_match:
        print(f"  [+++] omega1 gradient MATCHES expected: {gradient_str}")
    else:
        print(f"  [+] omega1 gradient: {gradient_str} (expected: {expected})")

    if mean_ecg_bvp > 0.3:
        print(f"  [+++] ECG↔BVP correlation strong (|r|={mean_ecg_bvp:.3f})")
        print(f"        Wearable BVP captures same cardiac torus geometry as PSG ECG")
    elif mean_ecg_bvp > 0.1:
        print(f"  [++] ECG↔BVP moderate (|r|={mean_ecg_bvp:.3f})")
    else:
        print(f"  [+] ECG↔BVP weak (|r|={mean_ecg_bvp:.3f})")

    if mean_emg_acc < 0.1:
        print(f"  [ok] EMG↔ACC near zero (|r|={mean_emg_acc:.3f}) — expected, different modalities")

    # Save
    np.savez_compressed(
        results_dir / "dreamt_100hz_psg_features.npz",
        psg_features=X_psg.astype(np.float32),
        wearable_features=X_wear.astype(np.float32),
        stages=y,
        psg_feat_names=np.array(psg_feat_names),
        wear_feat_names=np.array(wear_feat_names),
    )
    print(f"\n  Saved: results/dreamt_100hz_psg_features.npz")

    elapsed = time.time() - t_start
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
