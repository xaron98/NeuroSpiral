#!/usr/bin/env python3
"""NeuroSpiral — Anomaly Detection Prototype (Apple Watch).

Apple Watch can't classify sleep stages (kappa=0.165).
But it CAN detect anomalies by comparing a person to themselves.

For each subject: build personal baseline from first 80% of epochs,
then detect deviations in the remaining 20%. Inject synthetic anomalies
to measure sensitivity/specificity.

Usage:
    python scripts/anomaly_detection_prototype.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGES = ["W", "N1", "N2", "N3", "REM"]


def main():
    t_start = time.time()

    print()
    print("=" * 70)
    print("  NEUROSPIRAL — Anomaly Detection Prototype")
    print("  Personal baseline comparison for wearable sleep monitoring")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────
    d = np.load(Path.home() / "Downloads/Walch/walch_full_features.npz")
    X = d["features"]       # (25182, 18)
    stages = d["stages"]    # (25182,)
    subjects = d["subjects"]  # (25182,)

    n_feat = X.shape[1]
    feat_names = [f"f{i:02d}" for i in range(n_feat)]

    # Try to infer feature types from statistics
    # f12 has mean ~64 and looks like HR → label HR-related features
    col_means = X.mean(axis=0)
    # Heuristic: features with mean in 40-120 range might be HR
    hr_candidates = [i for i in range(n_feat) if 30 < col_means[i] < 130]
    acc_candidates = [i for i in range(n_feat) if i not in hr_candidates]

    print(f"  Data: {X.shape[0]:,} epochs, {len(np.unique(subjects))} subjects, {n_feat} features")
    print(f"  Likely HR features (mean 30-130): {hr_candidates}")
    print(f"  Other features: {acc_candidates[:10]}...")

    unique_subj = np.unique(subjects)
    n_subj = len(unique_subj)

    # ── 1-3. Personal baseline & z-score detection ────────────
    print(f"\n{'=' * 70}")
    print(f"  1-3. PERSONAL BASELINE & Z-SCORE ANOMALY DETECTION")
    print(f"{'=' * 70}\n")

    all_z2_rates = []   # per subject: fraction of test epochs with any |z| > 2
    all_z3_rates = []
    subj_stats = []

    for subj in unique_subj:
        mask = subjects == subj
        X_s = X[mask]
        y_s = stages[mask]
        n_ep = len(X_s)

        if n_ep < 50:
            continue

        # Split: first 80% = baseline, last 20% = test
        split = int(0.8 * n_ep)
        X_base, y_base = X_s[:split], y_s[:split]
        X_test, y_test = X_s[split:], y_s[split:]

        # Build per-stage personal profile
        profile_mean = {}
        profile_std = {}
        for s_int, s_name in enumerate(STAGES):
            base_mask = y_base == s_int
            if base_mask.sum() >= 5:
                profile_mean[s_int] = X_base[base_mask].mean(axis=0)
                profile_std[s_int] = X_base[base_mask].std(axis=0)
                profile_std[s_int][profile_std[s_int] < 1e-10] = 1e-10

        # Z-score test epochs against personal baseline
        n_z2 = 0
        n_z3 = 0
        n_scored = 0

        for i in range(len(X_test)):
            stage = y_test[i]
            if stage not in profile_mean:
                continue
            z = np.abs((X_test[i] - profile_mean[stage]) / profile_std[stage])
            n_scored += 1
            if np.any(z > 2):
                n_z2 += 1
            if np.any(z > 3):
                n_z3 += 1

        if n_scored > 0:
            z2_rate = n_z2 / n_scored
            z3_rate = n_z3 / n_scored
            all_z2_rates.append(z2_rate)
            all_z3_rates.append(z3_rate)
            subj_stats.append({
                "subj": subj, "n_base": split, "n_test": n_ep - split,
                "z2_rate": z2_rate, "z3_rate": z3_rate,
            })

    print(f"  Subjects analyzed: {len(subj_stats)}")
    print(f"\n  Normal deviation rates (no anomalies injected):")
    print(f"    |z| > 2 in any feature: {np.mean(all_z2_rates)*100:.1f}% "
          f"+/- {np.std(all_z2_rates)*100:.1f}% of test epochs")
    print(f"    |z| > 3 in any feature: {np.mean(all_z3_rates)*100:.1f}% "
          f"+/- {np.std(all_z3_rates)*100:.1f}% of test epochs")

    baseline_z2 = np.mean(all_z2_rates)
    baseline_z3 = np.mean(all_z3_rates)

    # ── 4-5. Inject synthetic anomalies ───────────────────────
    print(f"\n{'=' * 70}")
    print(f"  4-5. SYNTHETIC ANOMALY INJECTION & DETECTION")
    print(f"{'=' * 70}")

    # Define anomaly scenarios
    anomaly_configs = []
    # HR-like features (use top candidate or f12)
    hr_feat = hr_candidates[0] if hr_candidates else 12
    for feat_idx in range(n_feat):
        anomaly_configs.append({
            "name": f"{feat_names[feat_idx]}_x2_REM",
            "feat": feat_idx,
            "stage": 4,  # REM
            "multiplier": 2.0,
        })
        anomaly_configs.append({
            "name": f"{feat_names[feat_idx]}_x3_N3",
            "feat": feat_idx,
            "stage": 3,  # N3
            "multiplier": 3.0,
        })

    # Also test specific scenarios from the task description
    key_scenarios = [
        {"name": f"f{hr_feat}_x2_REM (HR-like)", "feat": hr_feat, "stage": 4, "multiplier": 2.0},
        {"name": f"f{hr_feat}_x3_REM (HR-like)", "feat": hr_feat, "stage": 4, "multiplier": 3.0},
    ]
    # Add ACC candidate
    acc_feat = acc_candidates[0] if acc_candidates else 0
    key_scenarios.append(
        {"name": f"f{acc_feat}_x3_N3 (ACC-like)", "feat": acc_feat, "stage": 3, "multiplier": 3.0},
    )

    # Run detection on all subjects with key scenarios first
    print(f"\n  Key scenarios (all subjects):\n")
    print(f"  {'Scenario':<35} {'Sens':>6} {'Spec':>6} {'F1':>6} {'PPV':>6}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    scenario_results = []

    for scenario in key_scenarios:
        all_tp, all_fp, all_fn, all_tn = 0, 0, 0, 0

        for subj in unique_subj:
            mask = subjects == subj
            X_s = X[mask].copy()
            y_s = stages[mask]
            n_ep = len(X_s)
            if n_ep < 50:
                continue

            split = int(0.8 * n_ep)
            X_base, y_base = X_s[:split], y_s[:split]
            X_test, y_test = X_s[split:].copy(), y_s[split:]

            # Build profile
            profile_mean, profile_std = {}, {}
            for s_int in range(5):
                bm = y_base == s_int
                if bm.sum() >= 5:
                    profile_mean[s_int] = X_base[bm].mean(axis=0)
                    profile_std[s_int] = X_base[bm].std(axis=0)
                    profile_std[s_int][profile_std[s_int] < 1e-10] = 1e-10

            # Inject anomaly into target stage epochs in test set
            target_stage = scenario["stage"]
            target_feat = scenario["feat"]
            mult = scenario["multiplier"]

            injected = np.zeros(len(X_test), dtype=bool)
            for i in range(len(X_test)):
                if y_test[i] == target_stage:
                    X_test[i, target_feat] *= mult
                    injected[i] = True

            # Detect with z > 2 threshold
            for i in range(len(X_test)):
                stage = y_test[i]
                if stage not in profile_mean:
                    continue
                z = np.abs((X_test[i] - profile_mean[stage]) / profile_std[stage])
                detected = np.any(z > 2)

                if injected[i] and detected:
                    all_tp += 1
                elif injected[i] and not detected:
                    all_fn += 1
                elif not injected[i] and detected:
                    all_fp += 1
                else:
                    all_tn += 1

        sens = all_tp / max(all_tp + all_fn, 1)
        spec = all_tn / max(all_tn + all_fp, 1)
        ppv = all_tp / max(all_tp + all_fp, 1)
        f1 = 2 * ppv * sens / max(ppv + sens, 1e-10)

        scenario_results.append({
            "name": scenario["name"], "sens": sens, "spec": spec,
            "f1": f1, "ppv": ppv, "tp": all_tp, "fp": all_fp,
            "fn": all_fn, "tn": all_tn,
        })
        print(f"  {scenario['name']:<35} {sens:>5.1%} {spec:>5.1%} {f1:>5.3f} {ppv:>5.1%}")

    # ── Sweep ALL features: which are most sensitive? ─────────
    print(f"\n  Per-feature sensitivity (x2 in REM, z>2 threshold):\n")

    feat_sensitivity = []
    for feat_idx in range(n_feat):
        all_tp, all_fn = 0, 0
        for subj in unique_subj:
            mask = subjects == subj
            X_s = X[mask].copy()
            y_s = stages[mask]
            n_ep = len(X_s)
            if n_ep < 50:
                continue

            split = int(0.8 * n_ep)
            X_base, y_base = X_s[:split], y_s[:split]
            X_test, y_test = X_s[split:].copy(), y_s[split:]

            profile_mean, profile_std = {}, {}
            for s_int in range(5):
                bm = y_base == s_int
                if bm.sum() >= 5:
                    profile_mean[s_int] = X_base[bm].mean(axis=0)
                    profile_std[s_int] = X_base[bm].std(axis=0)
                    profile_std[s_int][profile_std[s_int] < 1e-10] = 1e-10

            # Inject x2 in REM for this feature only
            for i in range(len(X_test)):
                if y_test[i] == 4:  # REM
                    X_test[i, feat_idx] *= 2.0
                    if 4 in profile_mean:
                        z = np.abs((X_test[i] - profile_mean[4]) / profile_std[4])
                        if z[feat_idx] > 2:
                            all_tp += 1
                        else:
                            all_fn += 1

        sens = all_tp / max(all_tp + all_fn, 1)
        feat_sensitivity.append(sens)

    # Rank by sensitivity
    ranked = np.argsort(feat_sensitivity)[::-1]
    print(f"  {'Rank':>4} {'Feature':<10} {'Sens (x2 REM)':>14} {'Type':>8}")
    print(f"  {'-'*4} {'-'*10} {'-'*14} {'-'*8}")
    for rank, idx in enumerate(ranked):
        ftype = "HR?" if idx in hr_candidates else "other"
        print(f"  {rank+1:>4} {feat_names[idx]:<10} {feat_sensitivity[idx]:>13.1%} {ftype:>8}")

    # ── 6. HR vs ACC comparison ───────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  6. HR vs OTHER FEATURES COMPARISON")
    print(f"{'=' * 70}\n")

    hr_sens = [feat_sensitivity[i] for i in hr_candidates] if hr_candidates else []
    other_sens = [feat_sensitivity[i] for i in acc_candidates]

    if hr_sens:
        print(f"  HR-like features (n={len(hr_candidates)}): "
              f"mean sens = {np.mean(hr_sens)*100:.1f}%")
    print(f"  Other features (n={len(acc_candidates)}): "
          f"mean sens = {np.mean(other_sens)*100:.1f}%")

    # Best features overall
    top5 = ranked[:5]
    print(f"\n  Top 5 most sensitive features:")
    for idx in top5:
        ftype = "HR?" if idx in hr_candidates else "other"
        print(f"    {feat_names[idx]}: {feat_sensitivity[idx]*100:.1f}% ({ftype})")

    # ── Multi-feature detection (any z > 2 across all features) ──
    print(f"\n{'=' * 70}")
    print(f"  MULTI-FEATURE DETECTION (z>2 in ANY feature)")
    print(f"{'=' * 70}\n")

    # Test with increasing anomaly magnitudes
    for mult in [1.5, 2.0, 3.0, 5.0]:
        all_tp, all_fp, all_fn, all_tn = 0, 0, 0, 0
        for subj in unique_subj:
            mask = subjects == subj
            X_s = X[mask].copy()
            y_s = stages[mask]
            n_ep = len(X_s)
            if n_ep < 50:
                continue

            split = int(0.8 * n_ep)
            X_base, y_base = X_s[:split], y_s[:split]
            X_test, y_test = X_s[split:].copy(), y_s[split:]

            profile_mean, profile_std = {}, {}
            for s_int in range(5):
                bm = y_base == s_int
                if bm.sum() >= 5:
                    profile_mean[s_int] = X_base[bm].mean(axis=0)
                    profile_std[s_int] = X_base[bm].std(axis=0)
                    profile_std[s_int][profile_std[s_int] < 1e-10] = 1e-10

            # Inject anomaly: multiply the MOST SENSITIVE feature in REM
            best_feat = ranked[0]
            injected = np.zeros(len(X_test), dtype=bool)
            for i in range(len(X_test)):
                if y_test[i] == 4:  # REM
                    X_test[i, best_feat] *= mult
                    injected[i] = True

            for i in range(len(X_test)):
                stage = y_test[i]
                if stage not in profile_mean:
                    continue
                z = np.abs((X_test[i] - profile_mean[stage]) / profile_std[stage])
                detected = np.any(z > 2)
                if injected[i] and detected: all_tp += 1
                elif injected[i] and not detected: all_fn += 1
                elif not injected[i] and detected: all_fp += 1
                else: all_tn += 1

        sens = all_tp / max(all_tp + all_fn, 1)
        spec = all_tn / max(all_tn + all_fp, 1)
        ppv = all_tp / max(all_tp + all_fp, 1)
        f1 = 2 * ppv * sens / max(ppv + sens, 1e-10)
        print(f"  x{mult:.1f} on {feat_names[ranked[0]]}: "
              f"sens={sens:.1%}  spec={spec:.1%}  F1={f1:.3f}  PPV={ppv:.1%}")

    # ── SUMMARY ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}\n")

    mean_base = np.mean([s["n_base"] for s in subj_stats])
    mean_test = np.mean([s["n_test"] for s in subj_stats])

    print(f"  Personal baseline: {mean_base:.0f} epochs (80% of night)")
    print(f"  Test window: {mean_test:.0f} epochs (20% of night)")
    print(f"  Normal false alarm rate (|z|>2): {baseline_z2*100:.1f}%")
    print(f"  Normal false alarm rate (|z|>3): {baseline_z3*100:.1f}%")

    print(f"\n  Detection with x2 anomaly on best feature ({feat_names[ranked[0]]}):")
    # Find the x2 result
    for sr in scenario_results:
        if "x2_REM" in sr["name"] and "HR" in sr["name"]:
            print(f"    Sensitivity: {sr['sens']*100:.1f}%")
            print(f"    Specificity: {sr['spec']*100:.1f}%")
            print(f"    F1: {sr['f1']:.3f}")
            break

    print(f"\n  Top 3 most anomaly-sensitive features:")
    for i, idx in enumerate(ranked[:3]):
        ftype = "HR?" if idx in hr_candidates else "other"
        print(f"    {i+1}. {feat_names[idx]} ({ftype}): {feat_sensitivity[idx]*100:.1f}% "
              f"sensitivity to x2 in REM")

    # Verdict
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}\n")

    best_sens = feat_sensitivity[ranked[0]]
    if best_sens > 0.80:
        print(f"  [+++] STRONG anomaly detection capability.")
    elif best_sens > 0.50:
        print(f"  [++] MODERATE anomaly detection capability.")
    else:
        print(f"  [+] WEAK anomaly detection with x2 perturbation.")

    print(f"  Best feature sensitivity (x2 REM): {best_sens*100:.1f}%")
    print(f"  Normal false alarm rate: {baseline_z2*100:.1f}% (z>2)")
    print(f"")
    print(f"  With personal baseline of ~{mean_base:.0f} epochs, the system")
    print(f"  detects x2 anomalies with {best_sens*100:.0f}% sensitivity")
    print(f"  and {(1-baseline_z2)*100:.0f}% specificity on the best feature.")
    print(f"  The most sensitive features are: "
          f"{', '.join(feat_names[i] for i in ranked[:3])}")

    # Save
    np.savez_compressed(
        PROJECT_ROOT / "results" / "anomaly_detection_prototype.npz",
        feat_sensitivity=np.array(feat_sensitivity),
        feat_names=np.array(feat_names),
        baseline_z2_rate=baseline_z2,
        baseline_z3_rate=baseline_z3,
        ranked_features=ranked,
        subj_stats=np.array([(s["subj"], s["n_base"], s["n_test"],
                               s["z2_rate"], s["z3_rate"]) for s in subj_stats]),
    )
    print(f"\n  Saved: results/anomaly_detection_prototype.npz")
    elapsed = time.time() - t_start
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
