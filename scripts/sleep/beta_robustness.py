#!/usr/bin/env python3
"""NeuroSpiral — Beta Robustness Sweep.

REM decomposition: mu_REM = alpha*mu_W + beta*mu_N3 + gamma*e_perp
Beta = position of REM on the W->N3 axis.

Sweep across feature subsets, distance metrics, normalizations,
and bootstrap CIs to determine if beta is a robust system property.

Usage:
    python scripts/beta_robustness.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGES = ["W", "N1", "N2", "N3", "REM"]


def compute_beta_gamma(centroids, metric="euclidean"):
    """Compute beta (REM position on W->N3 axis) and gamma/d (residual).

    mu_REM = alpha*mu_W + beta*mu_N3 + gamma*e_perp

    Beta is computed as the projection of (mu_REM - mu_W) onto (mu_N3 - mu_W),
    normalized by ||mu_N3 - mu_W||^2. This gives the fractional position:
      beta=0 means REM=W, beta=1 means REM=N3.

    gamma/d = ||residual|| / ||mu_REM - mu_W||
    """
    mu_W = centroids["W"]
    mu_N3 = centroids["N3"]
    mu_REM = centroids["REM"]

    axis = mu_N3 - mu_W
    axis_norm_sq = np.dot(axis, axis)
    if axis_norm_sq < 1e-15:
        return 0.0, 0.0, 0.0

    rem_vec = mu_REM - mu_W
    beta = float(np.dot(rem_vec, axis) / axis_norm_sq)

    projection = mu_W + beta * axis
    residual = mu_REM - projection
    gamma = float(np.linalg.norm(residual))
    d = float(np.linalg.norm(rem_vec))
    gamma_d = gamma / max(d, 1e-15)

    # Also compute ratio = d(W,REM) / d(W,N3)
    ratio = float(np.linalg.norm(rem_vec) / max(np.linalg.norm(axis), 1e-15))

    return beta, gamma_d, ratio


def compute_centroids(X, y, metric="euclidean"):
    """Compute per-stage centroids. For non-Euclidean metrics,
    use medoid (point closest to all others in that metric)."""
    centroids = {}
    if metric == "euclidean":
        for s_int, s_name in enumerate(STAGES):
            mask = y == s_int
            if mask.sum() > 0:
                centroids[s_name] = X[mask].mean(axis=0)
    else:
        for s_int, s_name in enumerate(STAGES):
            mask = y == s_int
            if mask.sum() == 0:
                continue
            X_s = X[mask]
            if len(X_s) > 2000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_s), 2000, replace=False)
                X_s = X_s[idx]
            D = cdist(X_s, X_s, metric=metric)
            medoid_idx = np.argmin(D.sum(axis=1))
            centroids[s_name] = X_s[medoid_idx]
    return centroids


def normalize(X, method, y=None, subjects=None):
    """Apply normalization."""
    if method == "standard":
        return StandardScaler().fit_transform(X)
    elif method == "minmax":
        return MinMaxScaler().fit_transform(X)
    elif method == "robust":
        return RobustScaler().fit_transform(X)
    elif method == "none":
        return X.copy()
    elif method == "per_subject":
        X_out = X.copy()
        for s in np.unique(subjects):
            mask = subjects == s
            if mask.sum() > 1:
                sc = StandardScaler()
                X_out[mask] = sc.fit_transform(X_out[mask])
        return X_out
    return X


def main():
    t_start = time.time()

    print()
    print("=" * 70)
    print("  NEUROSPIRAL — Beta Robustness Sweep")
    print("  Is REM's position on the W->N3 axis a robust property?")
    print("=" * 70)

    # Load data
    d = np.load(PROJECT_ROOT / "results" / "combined_features.npz")
    X_torus = d["torus_individual"]  # (117510, 96)
    X_spectral = d["spectral"]      # (117510, 32)
    X_combined = np.hstack([X_torus, X_spectral])  # (117510, 128)
    y = d["stages"]
    subjects = d["subjects"]

    print(f"  Data: {X_combined.shape}, {len(np.unique(subjects))} subjects")

    # Get top-20 features by RF importance
    print(f"  Computing feature importances for top-20 subset...")
    X_s = StandardScaler().fit_transform(X_combined)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_s, y)
    top20_idx = np.argsort(rf.feature_importances_)[::-1][:20]

    rng = np.random.default_rng(42)
    random32_idx = rng.choice(128, 32, replace=False)

    # ── Feature subsets ───────────────────────────────────────
    feature_configs = {
        "torus_96":     X_torus,
        "spectral_32":  X_spectral,
        "combined_128": X_combined,
        "top_20":       X_combined[:, top20_idx],
        "random_32":    X_combined[:, random32_idx],
    }

    metrics = ["euclidean", "cosine", "correlation", "cityblock"]
    # Note: Mahalanobis requires invertible covariance — use only on scaled data

    norms = ["standard", "minmax", "robust", "none", "per_subject"]

    all_results = []

    # ── 1. Feature subset sweep (Euclidean + StandardScaler) ──
    print(f"\n{'=' * 70}")
    print(f"  1. FEATURE SUBSET SWEEP (Euclidean, StandardScaler)")
    print(f"{'=' * 70}\n")
    print(f"  {'Subset':<20} {'beta':>8} {'gamma/d':>8} {'ratio':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    for name, X in feature_configs.items():
        Xn = StandardScaler().fit_transform(X)
        centroids = compute_centroids(Xn, y, "euclidean")
        if "W" in centroids and "N3" in centroids and "REM" in centroids:
            beta, gd, ratio = compute_beta_gamma(centroids)
            all_results.append(("feat", name, "euclidean", "standard", beta, gd, ratio))
            print(f"  {name:<20} {beta:>8.4f} {gd:>8.4f} {ratio:>8.4f}")

    # ── 2. Distance metric sweep (combined_128 + StandardScaler) ─
    print(f"\n{'=' * 70}")
    print(f"  2. DISTANCE METRIC SWEEP (128 features, StandardScaler)")
    print(f"{'=' * 70}\n")
    print(f"  {'Metric':<20} {'beta':>8} {'gamma/d':>8} {'ratio':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    Xn = StandardScaler().fit_transform(X_combined)
    for metric in metrics:
        centroids = compute_centroids(Xn, y, metric)
        if "W" in centroids and "N3" in centroids and "REM" in centroids:
            beta, gd, ratio = compute_beta_gamma(centroids)
            all_results.append(("metric", metric, metric, "standard", beta, gd, ratio))
            print(f"  {metric:<20} {beta:>8.4f} {gd:>8.4f} {ratio:>8.4f}")

    # Mahalanobis (needs special handling)
    try:
        cov = np.cov(Xn[:5000].T)
        cov_inv = np.linalg.pinv(cov)
        centroids_mah = {}
        for s_int, s_name in enumerate(STAGES):
            mask = y == s_int
            if mask.sum() > 0:
                centroids_mah[s_name] = Xn[mask].mean(axis=0)
        # Use Mahalanobis distances to define projection
        beta_m, gd_m, ratio_m = compute_beta_gamma(centroids_mah)
        all_results.append(("metric", "mahalanobis", "mahalanobis", "standard",
                            beta_m, gd_m, ratio_m))
        print(f"  {'mahalanobis':<20} {beta_m:>8.4f} {gd_m:>8.4f} {ratio_m:>8.4f}")
    except Exception:
        print(f"  {'mahalanobis':<20} {'FAILED':>8}")

    # ── 3. Normalization sweep (combined_128 + Euclidean) ─────
    print(f"\n{'=' * 70}")
    print(f"  3. NORMALIZATION SWEEP (128 features, Euclidean)")
    print(f"{'=' * 70}\n")
    print(f"  {'Normalization':<20} {'beta':>8} {'gamma/d':>8} {'ratio':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    for norm_name in norms:
        Xn = normalize(X_combined, norm_name, y, subjects)
        centroids = compute_centroids(Xn, y, "euclidean")
        if "W" in centroids and "N3" in centroids and "REM" in centroids:
            beta, gd, ratio = compute_beta_gamma(centroids)
            all_results.append(("norm", norm_name, "euclidean", norm_name, beta, gd, ratio))
            print(f"  {norm_name:<20} {beta:>8.4f} {gd:>8.4f} {ratio:>8.4f}")

    # ── 4. Bootstrap CIs (subject-level resampling) ───────────
    print(f"\n{'=' * 70}")
    print(f"  4. BOOTSTRAP CIs (1000 subject-level resamples)")
    print(f"{'=' * 70}\n")

    n_boot = 1000
    unique_subj = np.unique(subjects)
    boot_betas = []
    boot_gds = []
    boot_ratios = []

    Xn = StandardScaler().fit_transform(X_combined)

    for b in range(n_boot):
        boot_subj = rng.choice(unique_subj, size=len(unique_subj), replace=True)
        boot_idx = []
        for s in boot_subj:
            boot_idx.extend(np.where(subjects == s)[0].tolist())
        boot_idx = np.array(boot_idx)

        Xb = Xn[boot_idx]
        yb = y[boot_idx]

        centroids = {}
        for s_int, s_name in enumerate(STAGES):
            mask = yb == s_int
            if mask.sum() > 0:
                centroids[s_name] = Xb[mask].mean(axis=0)

        if "W" in centroids and "N3" in centroids and "REM" in centroids:
            beta, gd, ratio = compute_beta_gamma(centroids)
            boot_betas.append(beta)
            boot_gds.append(gd)
            boot_ratios.append(ratio)

    boot_betas = np.array(boot_betas)
    boot_gds = np.array(boot_gds)
    boot_ratios = np.array(boot_ratios)

    b_lo, b_med, b_hi = np.percentile(boot_betas, [2.5, 50, 97.5])
    g_lo, g_med, g_hi = np.percentile(boot_gds, [2.5, 50, 97.5])
    r_lo, r_med, r_hi = np.percentile(boot_ratios, [2.5, 50, 97.5])

    print(f"  {'Param':<12} {'Median':>10} {'95% CI':>24} {'Width':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*24} {'-'*10}")
    print(f"  {'beta':<12} {b_med:>10.4f} [{b_lo:.4f}, {b_hi:.4f}]{' ':>2} {b_hi-b_lo:>10.4f}")
    print(f"  {'gamma/d':<12} {g_med:>10.4f} [{g_lo:.4f}, {g_hi:.4f}]{' ':>2} {g_hi-g_lo:>10.4f}")
    print(f"  {'ratio':<12} {r_med:>10.4f} [{r_lo:.4f}, {r_hi:.4f}]{' ':>2} {r_hi-r_lo:>10.4f}")

    # ── FULL SUMMARY ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  FULL SUMMARY — All beta values")
    print(f"{'=' * 70}\n")

    all_betas = [r[4] for r in all_results]

    print(f"  {'Sweep':<10} {'Config':<20} {'Metric':<15} {'Norm':<15} "
          f"{'beta':>8} {'gamma/d':>8}")
    print(f"  {'-'*10} {'-'*20} {'-'*15} {'-'*15} {'-'*8} {'-'*8}")
    for sweep, config, metric, norm, beta, gd, ratio in all_results:
        print(f"  {sweep:<10} {config:<20} {metric:<15} {norm:<15} "
              f"{beta:>8.4f} {gd:>8.4f}")

    print(f"\n  Bootstrap 95% CI: beta = [{b_lo:.4f}, {b_hi:.4f}]")

    # ── VERDICT ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}\n")

    beta_range = max(all_betas) - min(all_betas)
    beta_mean = np.mean(all_betas)
    boot_width = b_hi - b_lo

    print(f"  beta range across all configs: [{min(all_betas):.4f}, {max(all_betas):.4f}]")
    print(f"  beta spread: {beta_range:.4f}")
    print(f"  beta mean: {beta_mean:.4f}")
    print(f"  bootstrap 95% CI width: {boot_width:.4f}")

    if beta_range < 0.10 and boot_width < 0.05:
        print(f"\n  [+++] ROBUST. beta varies < 0.10 across ALL configurations")
        print(f"        and bootstrap CI width < 0.05.")
        print(f"        REM's position on the W->N3 axis is a SYSTEM PROPERTY.")
    elif beta_range < 0.20:
        print(f"\n  [++] MODERATELY ROBUST. beta varies < 0.20.")
        print(f"       REM position is consistent but depends somewhat on method.")
    else:
        print(f"\n  [+] FRAGILE. beta varies >= 0.20 across configurations.")
        print(f"     REM position depends significantly on feature/metric choice.")

    in_range = sum(1 for b in all_betas if 0.55 <= b <= 0.70)
    print(f"\n  Configs with beta in [0.55, 0.70]: {in_range}/{len(all_betas)}")
    in_range_wide = sum(1 for b in all_betas if 0.40 <= b <= 0.80)
    print(f"  Configs with beta in [0.40, 0.80]: {in_range_wide}/{len(all_betas)}")

    # Save
    np.savez_compressed(
        PROJECT_ROOT / "results" / "beta_robustness.npz",
        all_results=np.array([(s, c, m, n, b, g, r) for s, c, m, n, b, g, r in all_results],
                             dtype=object),
        boot_betas=boot_betas,
        boot_gds=boot_gds,
        boot_ratios=boot_ratios,
        bootstrap_ci_beta=np.array([b_lo, b_med, b_hi]),
        bootstrap_ci_gd=np.array([g_lo, g_med, g_hi]),
        bootstrap_ci_ratio=np.array([r_lo, r_med, r_hi]),
    )
    print(f"\n  Saved: results/beta_robustness.npz")

    elapsed = time.time() - t_start
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
