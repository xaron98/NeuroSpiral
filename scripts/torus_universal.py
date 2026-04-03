#!/usr/bin/env python3
"""NeuroSpiral — Universal Torus Feature Analysis.

Apply Clifford torus geometry to ANY temporal signal.
"Dame un CSV con senales y etiquetas."

Input:  CSV with signal columns + one label column.
Output: Full diagnostic report — does the torus see structure?

Tests performed:
  1. omega1 gradient across classes + Kruskal-Wallis
  2. 8 torus features × N signals → RF classification (kappa)
  3. Class decomposition: beta, gamma/d (unique component)
  4. Ratio between extremes

Usage:
    python scripts/torus_universal.py data.csv --label-col class
    python scripts/torus_universal.py data.csv --label-col stage --epoch-size 3000
    python scripts/torus_universal.py data.csv --label-col y --signals x1,x2,x3 --tau 25
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Torus feature extraction (DO NOT MODIFY formulas)
# ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "omega1",             # ratio of angular velocities
    "torus_curvature",    # mean geodesic curvature
    "angular_acceleration",  # mean angular acceleration
    "geodesic_distance",  # total distance on torus
    "angular_entropy",    # Shannon entropy of (theta, phi)
    "phase_diff_std",     # phase difference variability
    "phase_coherence",    # mean resultant length R
    "transition_rate",    # rate of large angular jumps
]


def _wrap(d: np.ndarray) -> np.ndarray:
    """Wrap angular differences to [-pi, pi]."""
    return (d + np.pi) % (2 * np.pi) - np.pi


def mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 64) -> float:
    """MI(X;Y) via binned 2D histogram."""
    hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
    p_xy = hist_2d / hist_2d.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > 1e-12 and p_x[i] > 1e-12 and p_y[j] > 1e-12:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi


def estimate_optimal_tau(signal: np.ndarray, max_lag: int = 100) -> int:
    """First local minimum of mutual information (Fraser & Swinney 1986)."""
    max_lag = min(max_lag, len(signal) // 4)
    mi_values = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            mi_values[lag] = mutual_information(signal, signal)
        else:
            mi_values[lag] = mutual_information(signal[:-lag], signal[lag:])
    local_mins = argrelextrema(mi_values[1:], np.less)[0]
    if len(local_mins) > 0:
        return int(local_mins[0] + 1)
    below = np.where(mi_values[1:] < mi_values[0] / np.e)[0]
    if len(below) > 0:
        return int(below[0] + 1)
    return max(1, max_lag // 4)


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

    Returns (8,) array. Order matches FEATURE_NAMES.
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

    return np.array(feats, dtype=np.float64)


# ─────────────────────────────────────────────────────────────
# Data loading and epoching
# ─────────────────────────────────────────────────────────────
def load_csv(path: Path, label_col: str, signal_cols: list[str] | None):
    """Load CSV, return (signals_df, labels_series, signal_names)."""
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Columns: {list(df.columns)}")

    labels = df[label_col]

    if signal_cols:
        missing = [c for c in signal_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Signal columns not found: {missing}")
        signals = df[signal_cols]
    else:
        # Auto-detect: all numeric columns except the label
        signals = df.select_dtypes(include=[np.number]).drop(columns=[label_col], errors="ignore")
        if signals.empty:
            raise ValueError("No numeric signal columns found.")

    signal_names = list(signals.columns)
    return signals.values, labels.values, signal_names


def epoch_fixed(signals: np.ndarray, labels: np.ndarray, epoch_size: int):
    """Split into fixed-size epochs. Label = majority vote per epoch."""
    n_samples, n_signals = signals.shape
    n_epochs = n_samples // epoch_size

    epoch_signals = []  # (n_epochs, n_signals, epoch_size)
    epoch_labels = []

    for i in range(n_epochs):
        s = i * epoch_size
        e = s + epoch_size
        epoch_signals.append(signals[s:e].T)  # (n_signals, epoch_size)

        chunk_labels = labels[s:e]
        unique, counts = np.unique(chunk_labels, return_counts=True)
        epoch_labels.append(unique[np.argmax(counts)])

    return np.array(epoch_signals), np.array(epoch_labels)


def epoch_by_runs(signals: np.ndarray, labels: np.ndarray, min_length: int = 100):
    """Split by contiguous label runs. Discard runs shorter than min_length."""
    n_samples = len(labels)
    epoch_signals = []
    epoch_labels = []

    i = 0
    while i < n_samples:
        current_label = labels[i]
        j = i
        while j < n_samples and labels[j] == current_label:
            j += 1
        run_length = j - i
        if run_length >= min_length:
            epoch_signals.append(signals[i:j].T)  # (n_signals, run_length)
            epoch_labels.append(current_label)
        i = j

    return epoch_signals, np.array(epoch_labels)


# ─────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────
def compute_class_decomposition(centroids: dict[str, np.ndarray]):
    """Generalized decomposition: for each class, how much of its centroid
    is explained by a linear combination of other class centroids?

    Returns dict: class -> {beta: explained_fraction, gamma_d: unique_fraction,
                            coefficients: dict}
    """
    classes = sorted(centroids.keys())
    if len(classes) < 3:
        return {}

    results = {}
    for target_class in classes:
        mu_target = centroids[target_class]
        other_classes = [c for c in classes if c != target_class]
        A = np.column_stack([centroids[c] for c in other_classes])

        # Least squares: A @ x ≈ mu_target
        x, residuals, _, _ = np.linalg.lstsq(A, mu_target, rcond=None)

        projection = A @ x
        residual = mu_target - projection
        norm_target = np.linalg.norm(mu_target)

        if norm_target < 1e-15:
            continue

        beta = float(np.linalg.norm(projection) / norm_target)
        gamma_d = float(np.linalg.norm(residual) / norm_target)

        coefficients = {c: float(x[i]) for i, c in enumerate(other_classes)}

        results[target_class] = {
            "beta": beta,
            "gamma_d": gamma_d,
            "coefficients": coefficients,
        }

    return results


def verdict(kappa: float, n_sig_features: int, n_total_features: int,
            omega1_ratio: float) -> tuple[str, str]:
    """Determine if the torus sees structure. Returns (level, explanation)."""
    sig_pct = n_sig_features / max(n_total_features, 1) * 100

    if kappa >= 0.4 and sig_pct >= 50 and omega1_ratio >= 2.0:
        level = "STRONG"
        explanation = (f"kappa={kappa:.3f}, {sig_pct:.0f}% features significant, "
                       f"omega1 ratio={omega1_ratio:.2f}")
    elif kappa >= 0.2 and sig_pct >= 30:
        level = "MODERATE"
        explanation = (f"kappa={kappa:.3f}, {sig_pct:.0f}% features significant, "
                       f"omega1 ratio={omega1_ratio:.2f}")
    elif kappa >= 0.1 or sig_pct >= 20:
        level = "WEAK"
        explanation = (f"kappa={kappa:.3f}, {sig_pct:.0f}% features significant, "
                       f"omega1 ratio={omega1_ratio:.2f}")
    else:
        level = "NONE"
        explanation = (f"kappa={kappa:.3f}, {sig_pct:.0f}% features significant, "
                       f"omega1 ratio={omega1_ratio:.2f}")

    return level, explanation


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="NeuroSpiral — Universal Torus Feature Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/torus_universal.py sleep_data.csv --label-col stage --epoch-size 3000
  python scripts/torus_universal.py sensor_data.csv --label-col activity --signals acc_x,acc_y,acc_z
  python scripts/torus_universal.py ecg.csv --label-col arrhythmia --tau 20
        """,
    )
    parser.add_argument("csv", type=Path, help="Input CSV file")
    parser.add_argument("--label-col", required=True, help="Name of the label column")
    parser.add_argument("--signals", type=str, default=None,
                        help="Comma-separated signal column names (default: all numeric)")
    parser.add_argument("--epoch-size", type=int, default=None,
                        help="Fixed epoch size in samples (default: auto from label runs)")
    parser.add_argument("--tau", type=int, default=None,
                        help="Takens delay in samples (default: auto-estimate via MI)")
    parser.add_argument("--min-epoch", type=int, default=100,
                        help="Minimum epoch length for label-run mode (default: 100)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save features to .npz file")
    args = parser.parse_args()

    signal_cols = args.signals.split(",") if args.signals else None

    t_start = time.time()

    # ── 1. Load data ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("  NEUROSPIRAL — Universal Torus Feature Analysis")
    print("  Does the Clifford torus see structure in your data?")
    print("=" * 70)

    print(f"\n  Input:     {args.csv}")
    print(f"  Label col: {args.label_col}")

    signals, labels, signal_names = load_csv(args.csv, args.label_col, signal_cols)
    n_samples, n_signals = signals.shape

    print(f"  Signals:   {n_signals} ({', '.join(signal_names)})")
    print(f"  Samples:   {n_samples:,}")

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    print(f"  Classes:   {n_classes} ({', '.join(str(c) for c in unique_labels)})")

    if n_classes < 2:
        print("\n  ERROR: Need at least 2 classes for analysis.")
        sys.exit(1)

    # ── 2. Epoch extraction ──────────────────────────────────
    print(f"\n[1/5] Epoching...")

    if args.epoch_size:
        epoch_data, epoch_labels = epoch_fixed(signals, labels, args.epoch_size)
        print(f"  Fixed epochs: {len(epoch_labels)} x {args.epoch_size} samples")
        fixed_len = True
    else:
        epoch_data, epoch_labels = epoch_by_runs(signals, labels, args.min_epoch)
        print(f"  Label-run epochs: {len(epoch_labels)} (min {args.min_epoch} samples)")
        lengths = [e.shape[1] for e in epoch_data]
        print(f"  Epoch lengths: {min(lengths)}-{max(lengths)} "
              f"(median {int(np.median(lengths))})")
        fixed_len = False

    if len(epoch_labels) < 10:
        print("\n  ERROR: Too few epochs (<10). Check --epoch-size or --min-epoch.")
        sys.exit(1)

    # Class distribution
    for cls in unique_labels:
        count = np.sum(epoch_labels == cls)
        print(f"    {str(cls):>15}: {count:>5} epochs "
              f"({count / len(epoch_labels) * 100:.1f}%)")

    # ── 3. Estimate tau ──────────────────────────────────────
    print(f"\n[2/5] Estimating tau...")

    if args.tau:
        tau_per_signal = {name: args.tau for name in signal_names}
        print(f"  Fixed tau = {args.tau} for all signals")
    else:
        tau_per_signal = {}
        for sig_idx, sig_name in enumerate(signal_names):
            # Estimate from first 5 epochs
            taus = []
            for i in range(min(5, len(epoch_data))):
                if fixed_len:
                    sig = epoch_data[i, sig_idx, :]
                else:
                    sig = epoch_data[i][sig_idx, :]
                if len(sig) > 200:
                    taus.append(estimate_optimal_tau(sig))
            tau_est = int(np.median(taus)) if taus else 25
            tau_per_signal[sig_name] = tau_est
            print(f"  {sig_name}: tau = {tau_est}")

    # ── 4. Extract torus features ────────────────────────────
    print(f"\n[3/5] Extracting 8 torus features per signal per epoch...")

    all_features = []  # (n_valid_epochs, n_signals * 8)
    valid_labels = []
    n_failed = 0

    for i in range(len(epoch_labels)):
        epoch_feats = []
        ok = True

        for sig_idx, sig_name in enumerate(signal_names):
            if fixed_len:
                sig = epoch_data[i, sig_idx, :]
            else:
                sig = epoch_data[i][sig_idx, :]

            tau = tau_per_signal[sig_name]
            emb = takens_embed(sig, d=4, tau=tau)
            f = torus_features_8(emb)
            if f is None:
                ok = False
                break
            epoch_feats.extend(f)

        if ok:
            all_features.append(epoch_feats)
            valid_labels.append(epoch_labels[i])
        else:
            n_failed += 1

    X = np.array(all_features, dtype=np.float64)
    y = np.array(valid_labels)
    n_valid = len(y)

    # Build feature name list
    feat_names = []
    for sig_name in signal_names:
        for fn in FEATURE_NAMES:
            feat_names.append(f"{fn}_{sig_name}")
    n_feat = len(feat_names)

    print(f"  Valid epochs: {n_valid} (failed: {n_failed})")
    print(f"  Feature matrix: ({n_valid}, {n_feat})")

    # Clean NaN/Inf
    valid_mask = np.all(np.isfinite(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    if len(y) < 10:
        print("\n  ERROR: Too few valid epochs after feature extraction.")
        sys.exit(1)

    # Encode labels
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    class_names = list(le.classes_)

    # ── 5. Analysis ──────────────────────────────────────────

    # ─── 5a. omega1 gradient + Kruskal-Wallis ────────────────
    print(f"\n[4/5] Statistical analysis...")
    print(f"\n  {'='*60}")
    print(f"  OMEGA1 GRADIENT")
    print(f"  {'='*60}")

    # omega1 is feature index 0 for each signal
    for sig_idx, sig_name in enumerate(signal_names):
        omega1_col = sig_idx * 8  # omega1 is first of 8 features
        omega1 = X[:, omega1_col]

        class_means = {}
        for cls_idx, cls_name in enumerate(class_names):
            mask = y_int == cls_idx
            if mask.sum() > 0:
                class_means[str(cls_name)] = float(np.mean(omega1[mask]))

        ordered = sorted(class_means.items(), key=lambda x: x[1])
        ordering = " < ".join(f"{name}({val:.4f})" for name, val in ordered)
        ratio = ordered[-1][1] / max(ordered[0][1], 1e-10)

        print(f"\n  {sig_name}:")
        print(f"    Ordering: {ordering}")
        print(f"    Ratio (max/min): {ratio:.2f}")

    # KW test for all features
    print(f"\n  {'='*60}")
    print(f"  KRUSKAL-WALLIS TESTS (8 features x {n_signals} signals)")
    print(f"  {'='*60}\n")

    n_sig = 0
    kw_results = []
    for feat_idx, feat_name in enumerate(feat_names):
        groups = [X[y_int == c, feat_idx] for c in range(len(class_names))
                  if np.sum(y_int == c) >= 5]
        if len(groups) >= 2:
            H, p = kruskal(*groups)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if p < 0.01:
                n_sig += 1
            kw_results.append((feat_name, H, p, sig))

    print(f"  {'Feature':<35} {'H':>10} {'p':>12} {'Sig':>5}")
    print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*5}")
    for name, H, p, sig in sorted(kw_results, key=lambda x: x[2]):
        print(f"  {name:<35} {H:>10.1f} {p:>12.2e} {sig:>5}")

    print(f"\n  Significant (p<0.01): {n_sig}/{n_feat}")

    # ─── 5b. RF Classification ──────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  RANDOM FOREST CLASSIFICATION (5-fold CV)")
    print(f"  {'='*60}\n")

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(clf, X, y_int, cv=cv)
    kappa = cohen_kappa_score(y_int, y_pred)
    f1 = f1_score(y_int, y_pred, average="macro", zero_division=0)

    print(f"  Cohen's kappa: {kappa:.3f}")
    print(f"  F1-macro:      {f1:.3f}")
    print()
    print(classification_report(y_int, y_pred, target_names=[str(c) for c in class_names],
                                 digits=3, zero_division=0))

    # ─── 5c. Class decomposition (beta, gamma/d) ────────────
    print(f"  {'='*60}")
    print(f"  CLASS DECOMPOSITION (beta, gamma/d)")
    print(f"  {'='*60}\n")

    centroids = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_int == cls_idx
        if mask.sum() > 0:
            centroids[str(cls_name)] = np.mean(X[mask], axis=0)

    decomp = compute_class_decomposition(centroids)

    if decomp:
        gamma_d_values = []
        print(f"  {'Class':<15} {'beta':>8} {'gamma/d':>8}   Decomposition")
        print(f"  {'-'*15} {'-'*8} {'-'*8}   {'-'*30}")

        for cls_name in sorted(decomp.keys()):
            d = decomp[cls_name]
            gamma_d_values.append(d["gamma_d"])
            # Top 2 contributors
            top = sorted(d["coefficients"].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            contrib = " + ".join(f"{v:+.2f}*{k}" for k, v in top)
            print(f"  {cls_name:<15} {d['beta']:>8.3f} {d['gamma_d']:>8.3f}   {contrib}")

        if len(gamma_d_values) >= 2:
            gd_arr = np.array(gamma_d_values)
            print(f"\n  gamma/d mean:   {np.mean(gd_arr):.3f}")
            print(f"  gamma/d spread: {np.ptp(gd_arr):.3f} "
                  f"(low spread = per-channel invariant)")
    else:
        print("  Need >= 3 classes for decomposition analysis.")
        gamma_d_values = []

    # ─── 5d. Ratio between extremes ─────────────────────────
    print(f"\n  {'='*60}")
    print(f"  RATIO BETWEEN EXTREMES")
    print(f"  {'='*60}\n")

    # Use first signal's omega1 for the main ratio
    omega1_col = 0
    omega1_all = X[:, omega1_col]
    class_omega1 = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_int == cls_idx
        if mask.sum() > 0:
            class_omega1[str(cls_name)] = float(np.mean(omega1_all[mask]))

    sorted_classes = sorted(class_omega1.items(), key=lambda x: x[1])
    min_cls, min_val = sorted_classes[0]
    max_cls, max_val = sorted_classes[-1]
    omega1_ratio = max_val / max(min_val, 1e-10)

    print(f"  Lowest omega1:  {min_cls} = {min_val:.6f}")
    print(f"  Highest omega1: {max_cls} = {max_val:.6f}")
    print(f"  Ratio:          {omega1_ratio:.2f}")

    if n_signals > 1:
        print(f"\n  Per-signal ratios:")
        per_signal_ratios = []
        for sig_idx, sig_name in enumerate(signal_names):
            col = sig_idx * 8
            means = {}
            for cls_idx, cls_name in enumerate(class_names):
                mask = y_int == cls_idx
                if mask.sum() > 0:
                    means[str(cls_name)] = float(np.mean(X[mask, col]))
            vals = sorted(means.values())
            r = vals[-1] / max(vals[0], 1e-10)
            per_signal_ratios.append(r)
            print(f"    {sig_name:<20}: {r:.2f}")
        print(f"    Mean ratio:          {np.mean(per_signal_ratios):.2f}")

    # ── 6. Verdict ───────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  VERDICT")
    print(f"  {'='*60}\n")

    level, explanation = verdict(kappa, n_sig, n_feat, omega1_ratio)

    symbols = {"STRONG": "+++", "MODERATE": "++", "WEAK": "+", "NONE": "---"}
    messages = {
        "STRONG": "The torus SEES STRONG structure in your data.",
        "MODERATE": "The torus sees MODERATE structure.",
        "WEAK": "The torus sees WEAK structure. Consider more data or different tau.",
        "NONE": "The torus does NOT see structure. Data may lack geometric regularity.",
    }

    print(f"  [{symbols[level]}] {messages[level]}")
    print(f"  Evidence: {explanation}")

    if gamma_d_values:
        mean_gd = np.mean(gamma_d_values)
        if mean_gd < 0.15:
            print(f"  gamma/d = {mean_gd:.3f} -> classes are mostly mixtures of each other")
        elif mean_gd < 0.30:
            print(f"  gamma/d = {mean_gd:.3f} -> classes have some unique geometric content")
        else:
            print(f"  gamma/d = {mean_gd:.3f} -> classes have substantial unique geometry")

    # ── 7. Save ──────────────────────────────────────────────
    if args.output:
        np.savez_compressed(
            args.output,
            features=X.astype(np.float32),
            labels=y,
            labels_int=y_int.astype(np.int8),
            class_names=np.array(class_names),
            feature_names=np.array(feat_names),
            tau_per_signal=np.array([(name, tau_per_signal[name])
                                     for name in signal_names], dtype=object),
        )
        print(f"\n  Saved features to {args.output}")

    elapsed = time.time() - t_start
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
