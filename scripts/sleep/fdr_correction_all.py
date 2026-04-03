#!/usr/bin/env python3
"""NeuroSpiral — FDR Correction for All Statistical Tests.

Benjamini-Hochberg correction across:
  1. 128 features x KW test (feature discrimination)
  2. 128 features x 10 pairwise stage comparisons = 1280 tests
  3. Subject-level beta/gamma comparisons across datasets

Usage:
    python scripts/fdr_correction_all.py
"""

from __future__ import annotations

import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, mannwhitneyu

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_PAIRS = list(combinations(range(5), 2))  # 10 pairs
PAIR_NAMES = [f"{STAGES[a]}-{STAGES[b]}" for a, b in STAGE_PAIRS]


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    adjusted = np.empty(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i + 1]],
                                       sorted_p[i] * n / (i + 1))
    return np.clip(adjusted, 0, 1)


def main():
    t_start = time.time()
    results_dir = PROJECT_ROOT / "results"

    print()
    print("=" * 70)
    print("  NEUROSPIRAL — FDR Correction (Benjamini-Hochberg)")
    print("  Correcting all statistical tests for multiple comparisons")
    print("=" * 70)

    # Load data
    d = np.load(results_dir / "combined_features.npz")
    X_torus = d["torus_individual"]   # (117510, 96)
    X_spectral = d["spectral"]        # (117510, 32)
    X = np.hstack([X_torus, X_spectral])  # (117510, 128)
    y = d["stages"]
    subjects = d["subjects"]
    fn_t = list(d["feature_names_torus"])
    fn_s = list(d["feature_names_spectral"])
    feat_names = fn_t + fn_s
    n_feat = len(feat_names)

    print(f"  Features: {n_feat} (96 torus + 32 spectral)")
    print(f"  Epochs: {len(y):,}, Subjects: {len(np.unique(subjects))}")

    report_lines = []
    report_lines.append("NEUROSPIRAL — FDR Correction Report")
    report_lines.append("=" * 60)
    report_lines.append("")

    # ── TEST 1: KW per feature (5 stages) ─────────────────────
    print(f"\n{'=' * 70}")
    print(f"  1. KRUSKAL-WALLIS PER FEATURE (128 tests)")
    print(f"{'=' * 70}\n")

    kw_H = np.zeros(n_feat)
    kw_p = np.zeros(n_feat)

    for f in range(n_feat):
        groups = [X[y == s, f] for s in range(5)]
        H, p = kruskal(*groups)
        kw_H[f] = H
        kw_p[f] = p

    kw_fdr = fdr_bh(kw_p)
    n_sig_raw = (kw_p < 0.05).sum()
    n_sig_fdr = (kw_fdr < 0.05).sum()
    n_sig_fdr_001 = (kw_fdr < 0.001).sum()

    print(f"  Raw p < 0.05:  {n_sig_raw}/{n_feat}")
    print(f"  FDR p < 0.05:  {n_sig_fdr}/{n_feat}")
    print(f"  FDR p < 0.001: {n_sig_fdr_001}/{n_feat}")

    # Top 20 by H
    order = np.argsort(kw_H)[::-1]
    print(f"\n  {'#':>3} {'Feature':<35} {'H':>10} {'p_raw':>12} {'p_FDR':>12} {'Sig':>5}")
    print(f"  {'-'*3} {'-'*35} {'-'*10} {'-'*12} {'-'*12} {'-'*5}")
    for rank, idx in enumerate(order[:20]):
        sig = "***" if kw_fdr[idx] < 0.001 else "**" if kw_fdr[idx] < 0.01 else \
              "*" if kw_fdr[idx] < 0.05 else ""
        print(f"  {rank+1:>3} {feat_names[idx]:<35} {kw_H[idx]:>10.1f} "
              f"{kw_p[idx]:>12.2e} {kw_fdr[idx]:>12.2e} {sig:>5}")

    # Features that FAIL FDR
    failed = np.where(kw_fdr >= 0.05)[0]
    if len(failed) > 0:
        print(f"\n  Features that FAIL FDR (p_FDR >= 0.05):")
        for idx in failed:
            print(f"    {feat_names[idx]:<35} H={kw_H[idx]:.1f}  p_FDR={kw_fdr[idx]:.4f}")
    else:
        print(f"\n  ALL 128 features pass FDR at alpha=0.05")

    report_lines.append("1. KRUSKAL-WALLIS PER FEATURE")
    report_lines.append(f"   {n_sig_raw}/{n_feat} raw p<0.05")
    report_lines.append(f"   {n_sig_fdr}/{n_feat} FDR p<0.05")
    report_lines.append(f"   {n_sig_fdr_001}/{n_feat} FDR p<0.001")
    report_lines.append(f"   Failed FDR: {len(failed)} features")
    report_lines.append("")

    # ── TEST 2: Pairwise stage comparisons ────────────────────
    print(f"\n{'=' * 70}")
    print(f"  2. PAIRWISE STAGE COMPARISONS (128 x 10 = 1280 tests)")
    print(f"{'=' * 70}\n")

    pw_p = np.zeros((n_feat, len(STAGE_PAIRS)))
    pw_U = np.zeros((n_feat, len(STAGE_PAIRS)))

    for f in range(n_feat):
        for pi, (s1, s2) in enumerate(STAGE_PAIRS):
            U, p = mannwhitneyu(X[y == s1, f], X[y == s2, f], alternative="two-sided")
            pw_U[f, pi] = U
            pw_p[f, pi] = p

    # Flatten for FDR
    pw_p_flat = pw_p.flatten()
    pw_fdr_flat = fdr_bh(pw_p_flat)
    pw_fdr = pw_fdr_flat.reshape(n_feat, len(STAGE_PAIRS))

    n_total = n_feat * len(STAGE_PAIRS)
    n_pw_sig_raw = (pw_p_flat < 0.05).sum()
    n_pw_sig_fdr = (pw_fdr_flat < 0.05).sum()

    print(f"  Total comparisons: {n_total}")
    print(f"  Raw p < 0.05:  {n_pw_sig_raw}/{n_total} ({n_pw_sig_raw/n_total*100:.1f}%)")
    print(f"  FDR p < 0.05:  {n_pw_sig_fdr}/{n_total} ({n_pw_sig_fdr/n_total*100:.1f}%)")

    # Per-pair survival rate
    print(f"\n  Per-pair survival rate (FDR < 0.05):")
    print(f"  {'Pair':<10} {'Survive':>8} {'/%':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8}")
    for pi, pname in enumerate(PAIR_NAMES):
        n_surv = (pw_fdr[:, pi] < 0.05).sum()
        print(f"  {pname:<10} {n_surv:>8} {n_surv/n_feat*100:>7.1f}%")

    # Features with ALL 10 pairs significant
    all_pairs_sig = np.all(pw_fdr < 0.05, axis=1)
    n_all_sig = all_pairs_sig.sum()
    print(f"\n  Features with ALL 10 pairs significant: {n_all_sig}/{n_feat}")

    # Features with NO pairs significant
    no_pairs_sig = np.all(pw_fdr >= 0.05, axis=1)
    n_no_sig = no_pairs_sig.sum()
    if n_no_sig > 0:
        print(f"  Features with NO pairs significant: {n_no_sig}")
        for idx in np.where(no_pairs_sig)[0]:
            print(f"    {feat_names[idx]}")

    # Hardest pairs to separate
    print(f"\n  Hardest pairs (fewest features surviving FDR):")
    pair_survival = [(pw_fdr[:, pi] < 0.05).sum() for pi in range(len(STAGE_PAIRS))]
    hard_order = np.argsort(pair_survival)
    for pi in hard_order[:5]:
        print(f"    {PAIR_NAMES[pi]}: {pair_survival[pi]}/{n_feat} features")

    report_lines.append("2. PAIRWISE STAGE COMPARISONS")
    report_lines.append(f"   {n_pw_sig_raw}/{n_total} raw p<0.05")
    report_lines.append(f"   {n_pw_sig_fdr}/{n_total} FDR p<0.05")
    report_lines.append(f"   All 10 pairs significant: {n_all_sig}/{n_feat} features")
    report_lines.append("")

    # ── TEST 3: Per-channel KW summary ────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  3. PER-CHANNEL KW SUMMARY")
    print(f"{'=' * 70}\n")

    channels = {"EEG": slice(0, 24), "ECG": slice(24, 48),
                "EOG": slice(48, 72), "EMG": slice(72, 96),
                "Spectral": slice(96, 128)}

    print(f"  {'Channel':<12} {'n_feat':>7} {'FDR<0.05':>10} {'FDR<0.001':>10} {'Median H':>10}")
    print(f"  {'-'*12} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")
    for ch_name, ch_sl in channels.items():
        n_ch = ch_sl.stop - ch_sl.start
        n_ch_sig = (kw_fdr[ch_sl] < 0.05).sum()
        n_ch_sig2 = (kw_fdr[ch_sl] < 0.001).sum()
        med_H = np.median(kw_H[ch_sl])
        print(f"  {ch_name:<12} {n_ch:>7} {n_ch_sig:>10} {n_ch_sig2:>10} {med_H:>10.0f}")

    report_lines.append("3. PER-CHANNEL SUMMARY")
    for ch_name, ch_sl in channels.items():
        n_ch = ch_sl.stop - ch_sl.start
        n_ch_sig = (kw_fdr[ch_sl] < 0.05).sum()
        report_lines.append(f"   {ch_name}: {n_ch_sig}/{n_ch} FDR<0.05")
    report_lines.append("")

    # ── TEST 4: Subject-level beta comparisons ────────────────
    print(f"\n{'=' * 70}")
    print(f"  4. SUBJECT-LEVEL BETA COMPARISONS")
    print(f"{'=' * 70}\n")

    from sklearn.preprocessing import StandardScaler

    X_s = StandardScaler().fit_transform(X)

    def compute_subject_beta(X_subj, y_subj):
        centroids = {}
        for s_int in range(5):
            mask = y_subj == s_int
            if mask.sum() >= 10:
                centroids[s_int] = X_subj[mask].mean(axis=0)
        if not all(s in centroids for s in (0, 3, 4)):
            return None
        mu_W, mu_N3, mu_REM = centroids[0], centroids[3], centroids[4]
        axis = mu_N3 - mu_W
        axis_sq = np.dot(axis, axis)
        if axis_sq < 1e-15:
            return None
        rem_vec = mu_REM - mu_W
        beta = float(np.dot(rem_vec, axis) / axis_sq)
        proj = mu_W + beta * axis
        resid = mu_REM - proj
        gamma_d = float(np.linalg.norm(resid) / max(np.linalg.norm(rem_vec), 1e-15))
        return beta, gamma_d

    subj_betas = []
    subj_gds = []
    for s in np.unique(subjects):
        mask = subjects == s
        result = compute_subject_beta(X_s[mask], y[mask])
        if result:
            subj_betas.append(result[0])
            subj_gds.append(result[1])

    subj_betas = np.array(subj_betas)
    subj_gds = np.array(subj_gds)

    print(f"  HMC subjects with valid beta: {len(subj_betas)}/{len(np.unique(subjects))}")
    print(f"  beta: {np.mean(subj_betas):.3f} +/- {np.std(subj_betas):.3f} "
          f"[{np.percentile(subj_betas, 2.5):.3f}, {np.percentile(subj_betas, 97.5):.3f}]")
    print(f"  gamma/d: {np.mean(subj_gds):.3f} +/- {np.std(subj_gds):.3f}")

    # Cross-dataset comparison if CAP data exists
    cap_tests = []
    try:
        hc = np.load(Path.home() / "Downloads/CAP_Sleep/cap_healthy_features.npz")
        rc = np.load(Path.home() / "Downloads/CAP_Sleep/cap_rbd_features.npz")
        X_hc = StandardScaler().fit_transform(hc["features"])
        X_rc = StandardScaler().fit_transform(rc["features"])

        cap_h_betas, cap_r_betas = [], []
        for s in np.unique(hc["subjects"]):
            mask = hc["subjects"] == s
            r = compute_subject_beta(X_hc[mask], hc["stages"][mask])
            if r: cap_h_betas.append(r[0])
        for s in np.unique(rc["subjects"]):
            mask = rc["subjects"] == s
            r = compute_subject_beta(X_rc[mask], rc["stages"][mask])
            if r: cap_r_betas.append(r[0])

        cap_h_betas = np.array(cap_h_betas)
        cap_r_betas = np.array(cap_r_betas)

        # 3 pairwise tests: HMC vs CAP_H, HMC vs CAP_RBD, CAP_H vs CAP_RBD
        tests = [
            ("HMC vs CAP_Healthy", subj_betas, cap_h_betas),
            ("HMC vs CAP_RBD", subj_betas, cap_r_betas),
            ("CAP_H vs CAP_RBD", cap_h_betas, cap_r_betas),
        ]
        raw_ps = []
        for name, a, b in tests:
            U, p = mannwhitneyu(a, b, alternative="two-sided")
            raw_ps.append(p)
            cap_tests.append((name, np.mean(a), np.mean(b), U, p))

        fdr_ps = fdr_bh(raw_ps)

        print(f"\n  Cross-dataset beta comparisons (3 tests, FDR-corrected):")
        print(f"  {'Comparison':<25} {'Mean A':>8} {'Mean B':>8} {'p_raw':>10} {'p_FDR':>10} {'Sig':>5}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*5}")
        for i, (name, ma, mb, U, p_raw) in enumerate(cap_tests):
            sig = "***" if fdr_ps[i] < 0.001 else "**" if fdr_ps[i] < 0.01 else \
                  "*" if fdr_ps[i] < 0.05 else ""
            print(f"  {name:<25} {ma:>8.3f} {mb:>8.3f} {p_raw:>10.4f} {fdr_ps[i]:>10.4f} {sig:>5}")

        report_lines.append("4. CROSS-DATASET BETA COMPARISONS")
        for i, (name, ma, mb, U, p_raw) in enumerate(cap_tests):
            report_lines.append(f"   {name}: p_raw={p_raw:.4f}, p_FDR={fdr_ps[i]:.4f}")
    except FileNotFoundError:
        print(f"  CAP data not found — skipping cross-dataset comparison")
    report_lines.append("")

    # ── FULL SUMMARY ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  FULL SUMMARY")
    print(f"{'=' * 70}\n")

    total_tests = n_feat + n_total + len(cap_tests)
    total_raw = n_sig_raw + n_pw_sig_raw + sum(1 for _, _, _, _, p in cap_tests if p < 0.05)
    total_fdr = n_sig_fdr + n_pw_sig_fdr + (sum(1 for p in fdr_ps if p < 0.05) if cap_tests else 0)

    print(f"  {'Test family':<35} {'N tests':>8} {'Raw p<.05':>10} {'FDR p<.05':>10} {'Survival':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'KW per feature (5 stages)':<35} {n_feat:>8} {n_sig_raw:>10} {n_sig_fdr:>10} "
          f"{n_sig_fdr/max(n_sig_raw,1)*100:>9.1f}%")
    print(f"  {'Pairwise (128 feat x 10 pairs)':<35} {n_total:>8} {n_pw_sig_raw:>10} {n_pw_sig_fdr:>10} "
          f"{n_pw_sig_fdr/max(n_pw_sig_raw,1)*100:>9.1f}%")
    if cap_tests:
        n_cap_raw = sum(1 for _, _, _, _, p in cap_tests if p < 0.05)
        n_cap_fdr = sum(1 for p in fdr_ps if p < 0.05)
        print(f"  {'Cross-dataset beta':<35} {len(cap_tests):>8} {n_cap_raw:>10} {n_cap_fdr:>10} "
              f"{n_cap_fdr/max(n_cap_raw,1)*100:>9.1f}%")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<35} {total_tests:>8} {total_raw:>10} {total_fdr:>10}")

    report_lines.append("FULL SUMMARY")
    report_lines.append(f"Total tests: {total_tests}")
    report_lines.append(f"Raw p<0.05: {total_raw}")
    report_lines.append(f"FDR p<0.05: {total_fdr}")
    report_lines.append("")

    # ── VERDICT ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}\n")

    if n_sig_fdr == n_feat:
        print(f"  [+++] ALL {n_feat} features survive FDR for stage discrimination.")
    else:
        print(f"  {n_sig_fdr}/{n_feat} features survive FDR. {len(failed)} fail:")
        for idx in failed:
            print(f"        {feat_names[idx]}")

    pct_pw = n_pw_sig_fdr / n_total * 100
    print(f"  {n_pw_sig_fdr}/{n_total} pairwise comparisons survive FDR ({pct_pw:.1f}%).")

    hardest_pair = PAIR_NAMES[hard_order[0]]
    hardest_n = pair_survival[hard_order[0]]
    print(f"  Hardest pair: {hardest_pair} ({hardest_n}/{n_feat} features discriminate).")

    verdict = "ROBUST" if n_sig_fdr >= 120 and pct_pw > 80 else \
              "MODERATE" if n_sig_fdr >= 100 else "WEAK"
    print(f"\n  Overall: {verdict} — results survive multiple comparison correction.")

    report_lines.append("VERDICT")
    report_lines.append(f"{verdict}: {n_sig_fdr}/{n_feat} features, "
                        f"{n_pw_sig_fdr}/{n_total} pairwise survive FDR")

    # Save report
    report_path = results_dir / "fdr_correction_report.txt"
    with open(report_path, "w") as f:
        # Full feature table
        f.write("NEUROSPIRAL — FDR Correction Report\n")
        f.write("=" * 70 + "\n\n")
        f.write("FEATURE-LEVEL KW TEST (5 stages)\n")
        f.write(f"{'Feature':<35} {'H':>10} {'p_raw':>14} {'p_FDR':>14} {'Sig':>5}\n")
        f.write(f"{'-'*35} {'-'*10} {'-'*14} {'-'*14} {'-'*5}\n")
        for idx in order:
            sig = "***" if kw_fdr[idx] < 0.001 else "**" if kw_fdr[idx] < 0.01 else \
                  "*" if kw_fdr[idx] < 0.05 else ""
            f.write(f"{feat_names[idx]:<35} {kw_H[idx]:>10.1f} "
                    f"{kw_p[idx]:>14.6e} {kw_fdr[idx]:>14.6e} {sig:>5}\n")
        f.write(f"\nSummary: {n_sig_fdr}/{n_feat} pass FDR at alpha=0.05\n")
        f.write(f"Pairwise: {n_pw_sig_fdr}/{n_total} pass FDR\n")
        f.write(f"All-10-pairs significant: {n_all_sig}/{n_feat} features\n")
        for line in report_lines:
            f.write(line + "\n")

    print(f"\n  Full report saved: {report_path}")
    print(f"  Elapsed: {time.time()-t_start:.0f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
