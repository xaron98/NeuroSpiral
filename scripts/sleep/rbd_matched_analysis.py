#!/usr/bin/env python3
"""NeuroSpiral — Matched-Control RBD Analysis.

Subject-level beta/gamma comparison between CAP healthy and RBD,
with statistical tests, per-feature analysis, EMG focus, and LOSO classification.

Usage:
    python scripts/rbd_matched_analysis.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGES = ["W", "N1", "N2", "N3", "REM"]
CHANNELS = {"EEG": slice(0, 24), "ECG": slice(24, 48),
            "EOG": slice(48, 72), "EMG": slice(72, 96)}


def compute_subject_beta(X_subj, y_subj, min_epochs=10):
    """Compute beta, gamma/d, ratio for one subject."""
    centroids = {}
    for s_int, s_name in enumerate(STAGES):
        mask = y_subj == s_int
        if s_name in ("W", "N3", "REM") and mask.sum() < min_epochs:
            return None
        if mask.sum() > 0:
            centroids[s_name] = X_subj[mask].mean(axis=0)
    if not all(s in centroids for s in ("W", "N3", "REM")):
        return None
    mu_W, mu_N3, mu_REM = centroids["W"], centroids["N3"], centroids["REM"]
    axis = mu_N3 - mu_W
    axis_sq = np.dot(axis, axis)
    if axis_sq < 1e-15:
        return None
    rem_vec = mu_REM - mu_W
    beta = float(np.dot(rem_vec, axis) / axis_sq)
    proj = mu_W + beta * axis
    resid = mu_REM - proj
    gamma = float(np.linalg.norm(resid))
    d = float(np.linalg.norm(rem_vec))
    gamma_d = gamma / max(d, 1e-15)
    ratio = d / max(float(np.linalg.norm(axis)), 1e-15)
    return beta, gamma_d, ratio


def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    return float((np.mean(a) - np.mean(b)) / max(pooled, 1e-15))


def bootstrap_ci(a, b, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, len(a), replace=True)
        sb = rng.choice(b, len(b), replace=True)
        diffs.append(np.mean(sa) - np.mean(sb))
    return np.percentile(diffs, [2.5, 50, 97.5])


def fdr_correction(pvals):
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_p = np.array(pvals)[sorted_idx]
    adjusted = np.zeros(n)
    for i in range(n-1, -1, -1):
        if i == n-1:
            adjusted[sorted_idx[i]] = sorted_p[i]
        else:
            adjusted[sorted_idx[i]] = min(
                adjusted[sorted_idx[i+1]], sorted_p[i] * n / (i+1))
    return np.clip(adjusted, 0, 1)


def main():
    t_start = time.time()
    print()
    print("=" * 70)
    print("  NEUROSPIRAL — Matched-Control RBD Analysis")
    print("  CAP Healthy (16 subj) vs CAP RBD (18 subj)")
    print("=" * 70)

    # Load
    h = np.load(Path.home() / "Downloads/CAP_Sleep/cap_healthy_features.npz")
    r = np.load(Path.home() / "Downloads/CAP_Sleep/cap_rbd_features.npz")
    X_h, y_h, subj_h = h["features"], h["stages"], h["subjects"]
    X_r, y_r, subj_r = r["features"], r["stages"], r["subjects"]
    print(f"  Healthy: {X_h.shape[0]:,} epochs, {len(np.unique(subj_h))} subjects")
    print(f"  RBD:     {X_r.shape[0]:,} epochs, {len(np.unique(subj_r))} subjects")

    # Normalize jointly
    X_both = np.vstack([X_h, X_r])
    scaler = StandardScaler()
    X_both_s = scaler.fit_transform(X_both)
    X_h_s = X_both_s[:len(X_h)]
    X_r_s = X_both_s[len(X_h):]

    # ── 2. Subject-level beta/gamma ───────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  2. SUBJECT-LEVEL BETA & GAMMA/D")
    print(f"{'=' * 70}\n")

    h_betas, h_gds, h_ratios, h_valid = [], [], [], []
    print(f"  Healthy subjects:")
    for s in np.unique(subj_h):
        mask = subj_h == s
        result = compute_subject_beta(X_h_s[mask], y_h[mask])
        if result is None:
            print(f"    S{s:02d}: insufficient epochs"); continue
        beta, gd, ratio = result
        h_betas.append(beta); h_gds.append(gd); h_ratios.append(ratio); h_valid.append(s)
        n_w = (y_h[mask]==0).sum(); n_n3 = (y_h[mask]==3).sum(); n_rem = (y_h[mask]==4).sum()
        print(f"    S{s:02d}: beta={beta:.3f}  gd={gd:.3f}  (W={n_w}, N3={n_n3}, REM={n_rem})")

    r_betas, r_gds, r_ratios, r_valid = [], [], [], []
    print(f"\n  RBD subjects:")
    for s in np.unique(subj_r):
        mask = subj_r == s
        result = compute_subject_beta(X_r_s[mask], y_r[mask])
        if result is None:
            print(f"    S{s:02d}: insufficient epochs"); continue
        beta, gd, ratio = result
        r_betas.append(beta); r_gds.append(gd); r_ratios.append(ratio); r_valid.append(s)
        n_w = (y_r[mask]==0).sum(); n_n3 = (y_r[mask]==3).sum(); n_rem = (y_r[mask]==4).sum()
        print(f"    S{s:02d}: beta={beta:.3f}  gd={gd:.3f}  (W={n_w}, N3={n_n3}, REM={n_rem})")

    h_betas, h_gds, h_ratios = np.array(h_betas), np.array(h_gds), np.array(h_ratios)
    r_betas, r_gds, r_ratios = np.array(r_betas), np.array(r_gds), np.array(r_ratios)

    # ── 3. Statistical comparison ─────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  3. STATISTICAL COMPARISON")
    print(f"{'=' * 70}\n")

    metrics_list = [("beta", h_betas, r_betas), ("gamma/d", h_gds, r_gds),
                    ("ratio", h_ratios, r_ratios)]

    print(f"  {'Metric':<10} {'Healthy':>14} {'RBD':>14} {'U':>8} {'p':>10} {'d':>8} "
          f"{'95% CI diff':>22}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*8} {'-'*10} {'-'*8} {'-'*22}")
    for name, ha, ra in metrics_list:
        U, p = mannwhitneyu(ha, ra, alternative="two-sided")
        d = cohens_d(ha, ra)
        ci = bootstrap_ci(ha, ra)
        h_str = f"{np.mean(ha):.3f}+/-{np.std(ha):.3f}"
        r_str = f"{np.mean(ra):.3f}+/-{np.std(ra):.3f}"
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        print(f"  {name:<10} {h_str:>14} {r_str:>14} {U:>8.0f} {p:>9.4f}{sig} "
              f"{d:>8.2f} [{ci[0]:+.3f}, {ci[2]:+.3f}]")

    # ROC: beta as predictor
    labels_roc = np.concatenate([np.zeros(len(h_betas)), np.ones(len(r_betas))])
    scores_roc = np.concatenate([h_betas, r_betas])
    auc = roc_auc_score(labels_roc, -scores_roc)  # RBD has lower beta
    rng = np.random.default_rng(42)
    auc_boots = []
    for _ in range(1000):
        idx = rng.choice(len(labels_roc), len(labels_roc), replace=True)
        if len(np.unique(labels_roc[idx])) < 2: continue
        auc_boots.append(roc_auc_score(labels_roc[idx], -scores_roc[idx]))
    auc_lo, auc_hi = np.percentile(auc_boots, [2.5, 97.5])
    print(f"\n  beta as RBD predictor: AUC = {auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]")

    # ── 4. Per-feature analysis ───────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  4. PER-FEATURE ANALYSIS (96 torus features)")
    print(f"{'=' * 70}\n")

    feat_pvals, feat_ds = [], []
    for f in range(X_h.shape[1]):
        h_m = [X_h_s[subj_h==s, f].mean() for s in h_valid]
        r_m = [X_r_s[subj_r==s, f].mean() for s in r_valid]
        U, p = mannwhitneyu(h_m, r_m, alternative="two-sided")
        d = cohens_d(np.array(h_m), np.array(r_m))
        feat_pvals.append(p); feat_ds.append(d)

    feat_pvals = np.array(feat_pvals)
    feat_ds = np.array(feat_ds)
    fdr_p = fdr_correction(feat_pvals)

    try:
        fn = r["feature_names"]
        feat_names = [str(f) for f in fn] if len(fn) == 96 else [f"f{i}" for i in range(96)]
    except Exception:
        feat_names = [f"f{i}" for i in range(96)]

    top_idx = np.argsort(np.abs(feat_ds))[::-1][:15]
    print(f"  Top 15 features by |Cohen's d|:")
    print(f"  {'#':>3} {'Feature':<35} {'d':>8} {'p_raw':>10} {'p_FDR':>10} {'Sig':>5}")
    print(f"  {'-'*3} {'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*5}")
    for rank, idx in enumerate(top_idx):
        sig = "***" if fdr_p[idx]<0.001 else "**" if fdr_p[idx]<0.01 else "*" if fdr_p[idx]<0.05 else ""
        print(f"  {rank+1:>3} {feat_names[idx]:<35} {feat_ds[idx]:>8.3f} "
              f"{feat_pvals[idx]:>10.4f} {fdr_p[idx]:>10.4f} {sig:>5}")
    print(f"\n  Significant: {(feat_pvals<0.05).sum()}/96 raw, {(fdr_p<0.05).sum()}/96 FDR")

    # ── 5. EMG features per stage ─────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  5. EMG FEATURES — Per Sleep Stage")
    print(f"{'=' * 70}\n")

    emg_sl = CHANNELS["EMG"]
    for s_int, s_name in enumerate(STAGES):
        h_mask = y_h == s_int; r_mask = y_r == s_int
        if h_mask.sum() < 10 or r_mask.sum() < 10:
            print(f"  {s_name}: too few epochs"); continue
        emg_p, emg_d = [], []
        for fi in range(emg_sl.start, emg_sl.stop):
            hm = [X_h_s[(subj_h==s)&h_mask, fi].mean() for s in h_valid if ((subj_h==s)&h_mask).sum()>0]
            rm = [X_r_s[(subj_r==s)&r_mask, fi].mean() for s in r_valid if ((subj_r==s)&r_mask).sum()>0]
            if len(hm)<3 or len(rm)<3: emg_p.append(1.0); emg_d.append(0.0); continue
            U, p = mannwhitneyu(hm, rm, alternative="two-sided")
            emg_p.append(p); emg_d.append(cohens_d(np.array(hm), np.array(rm)))
        emg_fdr = fdr_correction(emg_p)
        n_sig = (np.array(emg_fdr) < 0.05).sum()
        max_di = np.argmax(np.abs(emg_d))
        print(f"  {s_name:<4}: FDR<0.05: {n_sig}/24  max|d|={abs(emg_d[max_di]):.3f} "
              f"(feat {emg_sl.start+max_di}: {feat_names[emg_sl.start+max_di] if len(feat_names)>emg_sl.start+max_di else '?'})")

    # ── 6. LOSO Classification ────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  6. LOSO CLASSIFICATION (Healthy vs RBD)")
    print(f"{'=' * 70}\n")

    # Subject-level mean features
    subj_feats, subj_labels, subj_groups = [], [], []
    for s in h_valid:
        subj_feats.append(X_h_s[subj_h==s].mean(axis=0))
        subj_labels.append(0); subj_groups.append(f"H{s}")
    for s in r_valid:
        subj_feats.append(X_r_s[subj_r==s].mean(axis=0))
        subj_labels.append(1); subj_groups.append(f"R{s}")
    X_subj = np.array(subj_feats)
    y_subj = np.array(subj_labels)
    groups_subj = np.array(subj_groups)
    print(f"  Subjects: {len(y_subj)} ({(y_subj==0).sum()} H, {(y_subj==1).sum()} RBD)")

    logo = LeaveOneGroupOut()

    def loso_classify(X, y, groups, label):
        preds, probas, trues = [], [], []
        for tr, te in logo.split(X, y, groups):
            clf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                          class_weight="balanced", random_state=42)
            clf.fit(X[tr], y[tr])
            preds.append(clf.predict(X[te])[0])
            probas.append(clf.predict_proba(X[te])[0, 1])
            trues.append(y[te][0])
        preds, probas, trues = np.array(preds), np.array(probas), np.array(trues)
        acc = (preds == trues).mean()
        kappa = cohen_kappa_score(trues, preds)
        auc_val = roc_auc_score(trues, probas) if len(np.unique(trues)) > 1 else 0
        print(f"  {label:<25} acc={acc:.3f}  kappa={kappa:.3f}  AUC={auc_val:.3f}")
        return acc, kappa, auc_val

    acc_all, kappa_all, auc_all = loso_classify(X_subj, y_subj, groups_subj, "All 96 features")
    acc_emg, kappa_emg, auc_emg = loso_classify(X_subj[:, emg_sl], y_subj, groups_subj, "EMG 24 features")

    X_bg = np.column_stack([np.concatenate([h_betas, r_betas]),
                             np.concatenate([h_gds, r_gds])])
    y_bg = np.concatenate([np.zeros(len(h_betas)), np.ones(len(r_betas))])
    g_bg = np.concatenate([[f"H{s}" for s in h_valid], [f"R{s}" for s in r_valid]])
    acc_bg, kappa_bg, auc_bg = loso_classify(X_bg, y_bg, g_bg, "beta + gamma/d (2f)")

    # ── 7. Summary ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  7. SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"  {'Metric':<10} {'Healthy':>14} {'RBD':>14} {'p':>10} {'Cohen d':>8}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*10} {'-'*8}")
    for name, ha, ra in metrics_list:
        U, p = mannwhitneyu(ha, ra, alternative="two-sided")
        d = cohens_d(ha, ra)
        print(f"  {name:<10} {np.mean(ha):.3f}+/-{np.std(ha):.3f} "
              f"{np.mean(ra):.3f}+/-{np.std(ra):.3f}  {p:>9.4f} {d:>8.2f}")

    print(f"\n  beta AUC for RBD: {auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]")

    # Verdict
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}\n")
    U_b, p_b = mannwhitneyu(h_betas, r_betas, alternative="two-sided")
    d_b = cohens_d(h_betas, r_betas)
    shift = np.mean(h_betas) - np.mean(r_betas)
    if p_b < 0.05 and abs(d_b) > 0.5:
        print(f"  [+++] beta shift SIGNIFICANT (p={p_b:.4f}, d={d_b:.2f})")
        print(f"        Healthy: {np.mean(h_betas):.3f}, RBD: {np.mean(r_betas):.3f}")
        print(f"        Shift = {shift:+.3f} (RBD REM moves {'toward W' if shift>0 else 'toward N3'})")
    elif p_b < 0.05:
        print(f"  [++] beta shift significant (p={p_b:.4f}) small effect (d={d_b:.2f})")
    else:
        print(f"  [+] beta shift not significant (p={p_b:.4f}, d={d_b:.2f}, n={len(h_betas)}+{len(r_betas)})")

    # Save
    np.savez_compressed(
        PROJECT_ROOT / "results" / "rbd_matched_analysis.npz",
        h_betas=h_betas, h_gds=h_gds, h_ratios=h_ratios,
        r_betas=r_betas, r_gds=r_gds, r_ratios=r_ratios,
        h_valid_subj=np.array(h_valid), r_valid_subj=np.array(r_valid),
        feat_pvals=feat_pvals, feat_ds=feat_ds, fdr_p=fdr_p,
        auc_beta=auc, auc_ci=np.array([auc_lo, auc_hi]),
    )
    print(f"\n  Saved: results/rbd_matched_analysis.npz")
    print(f"  Elapsed: {time.time()-t_start:.0f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
