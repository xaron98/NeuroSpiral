#!/usr/bin/env python3
"""NeuroSpiral — Temporal Context Features to Close the Dimension Gap.

Intrinsic dim ~8-9 but torus captures ~6. Add temporal features
(transitions, context, night position) to capture the missing 2-3 dims.

Usage:
    python scripts/temporal_features.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGES = ["W", "N1", "N2", "N3", "REM"]


def compute_temporal_features(X, y, subjects):
    """Compute temporal context features for each epoch.

    For each epoch i (within the same subject):
      Transition features:
        - delta_omega1: change in omega1 from previous epoch
        - delta_omega1_5: omega1 minus mean of previous 5 epochs
        - delta_norm: L2 norm of feature change from previous epoch
        - delta_norm_5: L2 norm of change from 5-epoch mean
      Temporal context:
        - night_position: epoch_index / total_epochs_in_subject (0-1)
        - time_in_stage: consecutive epochs in current stage / 10 (capped)
      Moving statistics (within ±2 epoch window):
        - ctx_mean_omega1: mean omega1 in ±2 window
        - ctx_std_omega1: std omega1 in ±2 window
        - ctx_mean_entropy: mean angular_entropy in ±2 window
        - ctx_std_entropy: std angular_entropy in ±2 window
        - ctx_mean_coherence: mean phase_coherence in ±2 window
        - ctx_mean_transition: mean transition_rate in ±2 window
      Stage transition features:
        - prev_stage_same: 1 if same stage as previous epoch, 0 otherwise
        - next_stage_same: 1 if same stage as next epoch, 0 otherwise

    Returns (n_epochs, n_temporal_features) array.
    """
    n = len(X)
    n_temp = 14
    T = np.zeros((n, n_temp), dtype=np.float64)
    feat_names = [
        "delta_omega1", "delta_omega1_5", "delta_norm", "delta_norm_5",
        "night_position", "time_in_stage",
        "ctx_mean_omega1", "ctx_std_omega1",
        "ctx_mean_entropy", "ctx_std_entropy",
        "ctx_mean_coherence", "ctx_mean_transition",
        "prev_stage_same", "next_stage_same",
    ]

    # Feature indices in the 96+32=128 feature vector
    # omega1_eeg_t10 = index 0, omega1_eeg_t25 = index 8, etc.
    omega1_idx = 0   # omega1_eeg_t10 (first feature)
    entropy_idx = 4  # angular_entropy_eeg_t10
    coherence_idx = 6  # phase_coherence_eeg_t10
    transition_idx = 7  # transition_rate_eeg_t10

    unique_subj = np.unique(subjects)

    for subj in unique_subj:
        mask = np.where(subjects == subj)[0]
        n_s = len(mask)
        if n_s < 3:
            continue

        X_s = X[mask]
        y_s = y[mask]

        omega1_s = X_s[:, omega1_idx]
        entropy_s = X_s[:, entropy_idx]
        coherence_s = X_s[:, coherence_idx]
        transition_s = X_s[:, transition_idx]

        for local_i in range(n_s):
            global_i = mask[local_i]

            # 1. delta_omega1
            if local_i > 0:
                T[global_i, 0] = omega1_s[local_i] - omega1_s[local_i - 1]
            # 2. delta_omega1_5
            if local_i >= 5:
                T[global_i, 1] = omega1_s[local_i] - np.mean(omega1_s[local_i-5:local_i])
            elif local_i > 0:
                T[global_i, 1] = omega1_s[local_i] - np.mean(omega1_s[:local_i])
            # 3. delta_norm (full feature vector change)
            if local_i > 0:
                T[global_i, 2] = np.linalg.norm(X_s[local_i] - X_s[local_i - 1])
            # 4. delta_norm_5
            if local_i >= 5:
                T[global_i, 3] = np.linalg.norm(X_s[local_i] - np.mean(X_s[local_i-5:local_i], axis=0))
            elif local_i > 0:
                T[global_i, 3] = np.linalg.norm(X_s[local_i] - np.mean(X_s[:local_i], axis=0))

            # 5. night_position
            T[global_i, 4] = local_i / max(n_s - 1, 1)

            # 6. time_in_stage (consecutive epochs in current stage)
            run = 1
            j = local_i - 1
            while j >= 0 and y_s[j] == y_s[local_i]:
                run += 1
                j -= 1
            T[global_i, 5] = min(run / 10.0, 3.0)  # cap at 30 epochs

            # 7-12. Context window ±2
            lo = max(0, local_i - 2)
            hi = min(n_s, local_i + 3)
            window = slice(lo, hi)
            T[global_i, 6] = np.mean(omega1_s[window])
            T[global_i, 7] = np.std(omega1_s[window]) if hi - lo > 1 else 0
            T[global_i, 8] = np.mean(entropy_s[window])
            T[global_i, 9] = np.std(entropy_s[window]) if hi - lo > 1 else 0
            T[global_i, 10] = np.mean(coherence_s[window])
            T[global_i, 11] = np.mean(transition_s[window])

            # 13. prev_stage_same
            if local_i > 0:
                T[global_i, 12] = 1.0 if y_s[local_i] == y_s[local_i - 1] else 0.0
            # 14. next_stage_same
            if local_i < n_s - 1:
                T[global_i, 13] = 1.0 if y_s[local_i] == y_s[local_i + 1] else 0.0

    return T, feat_names


def main():
    t_start = time.time()
    results_dir = PROJECT_ROOT / "results"

    print()
    print("=" * 70)
    print("  NEUROSPIRAL — Temporal Features to Close the Dimension Gap")
    print("  Intrinsic dim ~8-9, torus captures ~6. Adding ~14 temporal features.")
    print("=" * 70)

    # Load
    d = np.load(results_dir / "combined_features.npz")
    X_torus = d["torus_individual"]   # (117510, 96)
    X_spectral = d["spectral"]        # (117510, 32)
    X_base = np.hstack([X_torus, X_spectral])  # (117510, 128)
    y = d["stages"]
    subjects = d["subjects"]
    fn_torus = list(d["feature_names_torus"])
    fn_spectral = list(d["feature_names_spectral"])

    print(f"  Base features: {X_base.shape} (96 torus + 32 spectral)")
    print(f"  Subjects: {len(np.unique(subjects))}")

    # Compute temporal features
    print(f"\n[1/4] Computing 14 temporal features...")
    t0 = time.time()
    X_temp, temp_names = compute_temporal_features(X_base, y, subjects)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Temporal features: {X_temp.shape}")
    print(f"  Names: {temp_names}")

    # Combine
    X_augmented = np.hstack([X_base, X_temp])  # (117510, 142)
    all_names = fn_torus + fn_spectral + temp_names

    print(f"  Augmented: {X_augmented.shape}")

    # Clean NaN/Inf
    valid = np.all(np.isfinite(X_augmented), axis=1)
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        print(f"  Removing {n_invalid} NaN/Inf rows")
        X_augmented = X_augmented[valid]
        X_base_v = X_base[valid]
        y_v = y[valid]
        subjects_v = subjects[valid]
    else:
        X_base_v = X_base
        y_v = y
        subjects_v = subjects

    # ── Classification comparison ─────────────────────────────
    print(f"\n[2/4] Classification comparison (StratifiedGroupKFold, 5 folds)...")

    scaler = StandardScaler()
    cv = StratifiedGroupKFold(n_splits=5)
    clf = RandomForestClassifier(n_estimators=300, max_depth=20,
                                  class_weight="balanced", random_state=42, n_jobs=-1)

    configs = {
        "Torus only (96f)": X_base_v[:, :96],
        "Torus + Spectral (128f)": X_base_v,
        "128f + Temporal (142f)": X_augmented[valid] if n_invalid > 0 else X_augmented,
    }

    results = {}
    for name, X_cfg in configs.items():
        X_s = scaler.fit_transform(X_cfg)
        y_pred = cross_val_predict(clf, X_s, y_v, groups=subjects_v, cv=cv)
        kappa = cohen_kappa_score(y_v, y_pred)
        f1 = f1_score(y_v, y_pred, average="macro", zero_division=0)
        results[name] = {"kappa": kappa, "f1": f1, "n_feat": X_cfg.shape[1]}
        print(f"  {name:<35} kappa={kappa:.3f}  F1={f1:.3f}")

    # Also try: ctx±2 features only (avg of ±2 window for all 128 base features)
    print(f"\n  Computing ctx±2 smoothed features (128f)...")
    X_ctx = np.zeros_like(X_base_v)
    for subj in np.unique(subjects_v):
        mask = np.where(subjects_v == subj)[0]
        X_s_local = X_base_v[mask]
        n_s = len(mask)
        for i in range(n_s):
            lo = max(0, i - 2)
            hi = min(n_s, i + 3)
            X_ctx[mask[i]] = X_s_local[lo:hi].mean(axis=0)

    X_ctx_comb = np.hstack([X_base_v, X_ctx])  # 256f
    X_s = scaler.fit_transform(X_ctx_comb)
    y_pred = cross_val_predict(clf, X_s, y_v, groups=subjects_v, cv=cv)
    kappa_ctx = cohen_kappa_score(y_v, y_pred)
    f1_ctx = f1_score(y_v, y_pred, average="macro", zero_division=0)
    results["128f + ctx±2 (256f)"] = {"kappa": kappa_ctx, "f1": f1_ctx, "n_feat": 256}
    print(f"  {'128f + ctx±2 (256f)':<35} kappa={kappa_ctx:.3f}  F1={f1_ctx:.3f}")

    # Full: 128 base + 14 temporal + 128 ctx = 270
    X_full = np.hstack([X_base_v, X_temp[valid] if n_invalid > 0 else X_temp, X_ctx])
    X_s = scaler.fit_transform(X_full)
    y_pred = cross_val_predict(clf, X_s, y_v, groups=subjects_v, cv=cv)
    kappa_full = cohen_kappa_score(y_v, y_pred)
    f1_full = f1_score(y_v, y_pred, average="macro", zero_division=0)
    results["128f + temporal + ctx (270f)"] = {"kappa": kappa_full, "f1": f1_full, "n_feat": 270}
    print(f"  {'128f + temporal + ctx (270f)':<35} kappa={kappa_full:.3f}  F1={f1_full:.3f}")

    # Best config detailed report
    best_name = max(results, key=lambda k: results[k]["kappa"])
    print(f"\n  Best: {best_name} (kappa={results[best_name]['kappa']:.3f})")

    # Get detailed report for best
    if "270" in best_name:
        X_best = X_full
    elif "256" in best_name:
        X_best = X_ctx_comb
    elif "142" in best_name:
        X_best = X_augmented[valid] if n_invalid > 0 else X_augmented
    else:
        X_best = X_base_v

    X_s = scaler.fit_transform(X_best)
    y_pred_best = cross_val_predict(clf, X_s, y_v, groups=subjects_v, cv=cv)
    print()
    print(classification_report(y_v, y_pred_best, target_names=STAGES, digits=3, zero_division=0))

    # ── Feature importance ────────────────────────────────────
    print(f"[3/4] Feature importance (top 30)...\n")

    # Use 270f (fullest) for importance
    X_s = scaler.fit_transform(X_full)
    clf.fit(X_s, y_v)
    imp = clf.feature_importances_

    # Build names for all 270 features
    full_names = all_names + [f"ctx_{fn}" for fn in fn_torus + fn_spectral]

    top30 = np.argsort(imp)[::-1][:30]
    print(f"  {'#':>3} {'Feature':<40} {'Importance':>12} {'Type':>10}")
    print(f"  {'-'*3} {'-'*40} {'-'*12} {'-'*10}")

    n_temp_in_top20 = 0
    n_ctx_in_top20 = 0
    for rank, idx in enumerate(top30):
        name = full_names[idx] if idx < len(full_names) else f"f{idx}"
        if idx < 96:
            ftype = "torus"
        elif idx < 128:
            ftype = "spectral"
        elif idx < 142:
            ftype = "temporal"
            if rank < 20:
                n_temp_in_top20 += 1
        else:
            ftype = "ctx±2"
            if rank < 20:
                n_ctx_in_top20 += 1
        print(f"  {rank+1:>3} {name:<40} {imp[idx]:>12.4f} {ftype:>10}")

    print(f"\n  Temporal features in top 20: {n_temp_in_top20}")
    print(f"  Context features in top 20: {n_ctx_in_top20}")

    # Per-type importance
    print(f"\n  Importance by feature type:")
    type_imp = {"torus": imp[:96].sum(), "spectral": imp[96:128].sum(),
                "temporal": imp[128:142].sum(), "ctx±2": imp[142:].sum()}
    total_imp = sum(type_imp.values())
    for t, v in sorted(type_imp.items(), key=lambda x: -x[1]):
        print(f"    {t:<12}: {v:.4f} ({v/total_imp*100:.1f}%)")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — Closing the Dimension Gap")
    print(f"{'=' * 70}\n")

    print(f"  {'Config':<40} {'n_feat':>7} {'kappa':>7} {'delta':>8}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*8}")
    baseline_kappa = results["Torus only (96f)"]["kappa"]
    for name in ["Torus only (96f)", "Torus + Spectral (128f)",
                  "128f + Temporal (142f)", "128f + ctx±2 (256f)",
                  "128f + temporal + ctx (270f)"]:
        if name in results:
            r = results[name]
            delta = r["kappa"] - baseline_kappa
            print(f"  {name:<40} {r['n_feat']:>7} {r['kappa']:>7.3f} {delta:>+8.3f}")

    best_kappa = max(r["kappa"] for r in results.values())
    gap_closed = best_kappa - baseline_kappa
    gap_total = 0.80 - baseline_kappa  # vs SOTA
    pct_closed = gap_closed / max(gap_total, 1e-10) * 100

    print(f"\n  Baseline (torus 96f): kappa = {baseline_kappa:.3f}")
    print(f"  Best config:          kappa = {best_kappa:.3f}")
    print(f"  Gap closed:           {gap_closed:+.3f} ({pct_closed:.0f}% of gap to SOTA 0.80)")

    # Verdict
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}\n")

    if best_kappa >= 0.75:
        print(f"  [+++] kappa >= 0.75 — approaching SOTA territory!")
    elif best_kappa >= 0.70:
        print(f"  [++] kappa >= 0.70 — significant improvement over baseline.")
    elif gap_closed > 0.02:
        print(f"  [+] Temporal features contribute +{gap_closed:.3f} kappa.")
    else:
        print(f"  [=] Temporal features provide minimal improvement ({gap_closed:+.3f}).")
        print(f"      The missing dimensions may require fundamentally different")
        print(f"      information (e.g., raw waveform, temporal attention, sequence models).")

    # Save
    np.savez_compressed(
        results_dir / "multiscale_temporal_features.npz",
        X_augmented=X_augmented.astype(np.float32),
        temporal_features=X_temp.astype(np.float32),
        temporal_names=np.array(temp_names),
        stages=y,
        subjects=subjects,
    )
    print(f"\n  Saved: results/multiscale_temporal_features.npz")
    print(f"  Elapsed: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}m)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
