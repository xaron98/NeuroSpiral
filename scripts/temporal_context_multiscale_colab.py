#!/usr/bin/env python3
"""Multi-scale temporal context classification — Colab execution script.

Loads results/hmc_features_multiscale.npz and runs 5-condition comparison
with the expanded geometric feature set (57 geometric features from
multi-scale bilateral extraction).

Conditions:
  1. Spectral only, no context          (8 features)
  2. Spectral + geometric, no context   (65 features)
  3. Spectral only, 5-epoch context     (40 features)
  4. Spectral + geometric, 5-epoch ctx  (325 features)
  5. Geometric only, 5-epoch context    (285 features)

Key comparisons:
  - Wilcoxon cond 1 vs 5: spectral-only (no ctx) vs geometric-only (ctx)
  - Wilcoxon cond 3 vs 5: spectral ctx vs geometric ctx (fair comparison)

Uses XGBoost with tree_method='hist' (CPU).
5-fold subject-stratified CV with same metrics as Paper #1.

Usage (Colab):
  1. Upload hmc_features_multiscale.npz when prompted
  2. Results saved to temporal_context_multiscale.json
  3. Figure saved to temporal_context_multiscale.png
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

# ---------------------------------------------------------------------------
# Install xgboost if needed (Colab)
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAGES = ["W", "N1", "N2", "N3", "REM"]
CONTEXT_HALF = 2  # 2 epochs on each side → 5-epoch window


# ===================================================================
# Data loading
# ===================================================================
def load_features(npz_path: str | None = None) -> dict:
    """Load multi-scale features from .npz file.

    If npz_path is None, tries Colab upload, then local fallback paths.
    """
    if npz_path is not None and os.path.exists(npz_path):
        pass
    else:
        # Try Colab upload
        try:
            from google.colab import files
            print("Upload hmc_features_multiscale.npz:")
            uploaded = files.upload()
            npz_path = list(uploaded.keys())[0]
        except (ImportError, ModuleNotFoundError):
            # Local fallback
            for candidate in [
                "results/hmc_features_multiscale.npz",
                "../results/hmc_features_multiscale.npz",
                "hmc_features_multiscale.npz",
            ]:
                if os.path.exists(candidate):
                    npz_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    "Cannot find hmc_features_multiscale.npz. "
                    "Provide path or run in Colab to upload."
                )

    data = np.load(npz_path)
    print(f"  Loaded: {npz_path}")
    print(f"  Spectral:  {data['features_spectral'].shape}")
    print(f"  Geometric: {data['features_geometric'].shape}")
    print(f"  Epochs:    {len(data['stages'])}")

    return {
        "features_spectral": data["features_spectral"],
        "features_geometric": data["features_geometric"],
        "stages": data["stages"],
        "subjects": data["subjects"],
        "feature_names_spectral": list(data["feature_names_spectral"]),
        "feature_names_geometric": list(data["feature_names_geometric"]),
    }


# ===================================================================
# Temporal context expansion (per-subject)
# ===================================================================
def build_context_per_subject(features: np.ndarray, subjects: np.ndarray,
                               half: int = CONTEXT_HALF) -> np.ndarray:
    """Expand (n, d) -> (n, d*(2*half+1)) with boundary replication per subject.

    Context windows never cross subject boundaries.
    """
    n, d = features.shape
    window = 2 * half + 1
    out = np.zeros((n, d * window), dtype=features.dtype)

    unique_subj = np.unique(subjects)
    for subj in unique_subj:
        mask = np.where(subjects == subj)[0]
        subj_feats = features[mask]
        ns = len(subj_feats)

        for local_i in range(ns):
            parts = []
            for offset in range(-half, half + 1):
                j = max(0, min(ns - 1, local_i + offset))
                parts.append(subj_feats[j])
            out[mask[local_i]] = np.concatenate(parts)

    return out


# ===================================================================
# Main
# ===================================================================
def main(npz_path: str | None = None) -> None:
    t_start = time.time()

    print("=" * 70)
    print("Multi-Scale Temporal Context Classification")
    print("=" * 70)

    # --- Load data ---
    data = load_features(npz_path)
    spec = data["features_spectral"]    # (N, 8)
    geom = data["features_geometric"]   # (N, 57)
    stages = data["stages"]             # (N,)
    subjects = data["subjects"]         # (N,)

    n_spec = spec.shape[1]   # 8
    n_geom = geom.shape[1]   # 57
    N = len(stages)

    print(f"\n  {N:,} epochs, {len(np.unique(subjects))} subjects")
    print(f"  Spectral: {n_spec}  Geometric: {n_geom}")
    print(f"  Stage distribution: {dict(zip(*np.unique(stages, return_counts=True)))}")

    # Combined features (no context)
    combined = np.hstack([spec, geom])  # (N, 65)

    # --- Build temporal context ---
    print("\n  Building temporal context windows ...")
    spec_ctx = build_context_per_subject(spec, subjects, CONTEXT_HALF)
    geom_ctx = build_context_per_subject(geom, subjects, CONTEXT_HALF)
    combined_ctx = build_context_per_subject(combined, subjects, CONTEXT_HALF)

    print(f"  Spectral ctx:  {spec_ctx.shape}")
    print(f"  Geometric ctx: {geom_ctx.shape}")
    print(f"  Combined ctx:  {combined_ctx.shape}")

    # --- Define 5 conditions ---
    conditions = [
        ("Spec only (8)",               spec),
        (f"Spec+Geom ({n_spec + n_geom})",  combined),
        (f"Spec ctx ({spec_ctx.shape[1]})",  spec_ctx),
        (f"Spec+Geom ctx ({combined_ctx.shape[1]})", combined_ctx),
        (f"Geom ctx ({geom_ctx.shape[1]})", geom_ctx),
    ]

    # --- 5-fold subject-stratified CV ---
    print("\n" + "=" * 70)
    print("5-fold subject-stratified cross-validation (XGBoost)")
    print("=" * 70)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(spec, stages, groups=subjects))

    results: dict[str, dict] = {}

    for cond_name, X_cond in conditions:
        print(f"\n  {cond_name} ({X_cond.shape[1]}d) ...")
        t_cond = time.time()

        fold_accs, fold_kappas, fold_f1ms = [], [], []
        fold_f1_per = []

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            X_tr = np.nan_to_num(X_cond[train_idx]).astype(np.float32)
            y_tr = stages[train_idx]
            X_te = np.nan_to_num(X_cond[test_idx]).astype(np.float32)
            y_te = stages[test_idx]

            clf = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=42,
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0,
            )
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)

            fold_accs.append(float(accuracy_score(y_te, pred)))
            fold_kappas.append(float(cohen_kappa_score(y_te, pred)))
            fold_f1ms.append(float(f1_score(y_te, pred, average="macro")))
            fold_f1_per.append(f1_score(y_te, pred, average=None, labels=list(range(5))))

        mean_acc = float(np.mean(fold_accs))
        mean_kappa = float(np.mean(fold_kappas))
        mean_f1m = float(np.mean(fold_f1ms))
        mean_f1_per = np.mean(fold_f1_per, axis=0)

        results[cond_name] = {
            "n_features": int(X_cond.shape[1]),
            "accuracy": {
                "mean": round(mean_acc, 4),
                "std": round(float(np.std(fold_accs)), 4),
                "folds": [round(v, 4) for v in fold_accs],
            },
            "kappa": {
                "mean": round(mean_kappa, 4),
                "std": round(float(np.std(fold_kappas)), 4),
                "folds": [round(v, 4) for v in fold_kappas],
            },
            "f1_macro": {
                "mean": round(mean_f1m, 4),
                "std": round(float(np.std(fold_f1ms)), 4),
                "folds": [round(v, 4) for v in fold_f1ms],
            },
            "f1_per_stage": {
                s: round(float(v), 4) for s, v in zip(STAGES, mean_f1_per)
            },
        }

        per_stage_str = "  ".join(f"{s}={v:.2f}" for s, v in zip(STAGES, mean_f1_per))
        elapsed_cond = time.time() - t_cond
        print(f"    Acc={mean_acc:.3f}  κ={mean_kappa:.3f}  "
              f"F1m={mean_f1m:.3f}  [{per_stage_str}]  ({elapsed_cond:.0f}s)")

    # --- Wilcoxon tests ---
    cond_names = [c[0] for c in conditions]
    cond1_name = cond_names[0]  # Spectral only, no context
    cond3_name = cond_names[2]  # Spectral ctx
    cond5_name = cond_names[4]  # Geometric ctx

    print("\n" + "-" * 70)
    print("Statistical comparisons")
    print("-" * 70)

    comparisons = [
        ("cond1_vs_cond5", cond1_name, cond5_name,
         "Spectral-only (no ctx) vs Geometric-only (ctx)"),
        ("cond3_vs_cond5", cond3_name, cond5_name,
         "Spectral ctx vs Geometric ctx (fair comparison)"),
    ]

    for key, name_a, name_b, desc in comparisons:
        kappa_a = results[name_a]["kappa"]["folds"]
        kappa_b = results[name_b]["kappa"]["folds"]
        f1_a = results[name_a]["f1_macro"]["folds"]
        f1_b = results[name_b]["f1_macro"]["folds"]

        if len(kappa_a) >= 5:
            stat_k, p_k = wilcoxon(kappa_b, kappa_a, alternative="greater")
            stat_f, p_f = wilcoxon(f1_b, f1_a, alternative="greater")
        else:
            stat_k, p_k = 0, 1.0
            stat_f, p_f = 0, 1.0

        delta_kappa = results[name_b]["kappa"]["mean"] - results[name_a]["kappa"]["mean"]
        delta_f1 = results[name_b]["f1_macro"]["mean"] - results[name_a]["f1_macro"]["mean"]

        results[f"wilcoxon_{key}"] = {
            "description": desc,
            "condition_a": name_a,
            "condition_b": name_b,
            "delta_kappa": round(delta_kappa, 4),
            "delta_f1_macro": round(delta_f1, 4),
            "kappa_wilcoxon_stat": float(stat_k),
            "kappa_p_value": round(float(p_k), 6),
            "f1_wilcoxon_stat": float(stat_f),
            "f1_p_value": round(float(p_f), 6),
        }

        print(f"\n  {desc}")
        print(f"    Δκ = {delta_kappa:+.4f}  (p = {p_k:.4f})")
        print(f"    ΔF1m = {delta_f1:+.4f}  (p = {p_f:.4f})")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    header = f"  {'Condition':<30s} {'Dim':>5s} {'Acc':>7s} {'κ':>7s} {'F1m':>7s}"
    for s in STAGES:
        header += f" {s:>5s}"
    print(header)
    print("  " + "-" * (30 + 5 + 7 * 3 + 6 * 5))

    for cond_name in cond_names:
        r = results[cond_name]
        line = (f"  {cond_name:<30s} {r['n_features']:>5d}"
                f" {r['accuracy']['mean']:>7.3f}"
                f" {r['kappa']['mean']:>7.3f}"
                f" {r['f1_macro']['mean']:>7.3f}")
        for s in STAGES:
            line += f" {r['f1_per_stage'][s]:>5.2f}"
        print(line)

    # --- Figure: bar chart ---
    fig, ax = plt.subplots(figsize=(12, 5))
    kappas = [results[c]["kappa"]["mean"] for c in cond_names]
    kappa_stds = [results[c]["kappa"]["std"] for c in cond_names]
    f1ms = [results[c]["f1_macro"]["mean"] for c in cond_names]
    f1_stds = [results[c]["f1_macro"]["std"] for c in cond_names]

    x = np.arange(len(cond_names))
    w = 0.35
    bars1 = ax.bar(x - w / 2, kappas, w, yerr=kappa_stds, capsize=4,
                   label="Cohen's κ", color="#5bffa8", edgecolor="white")
    bars2 = ax.bar(x + w / 2, f1ms, w, yerr=f1_stds, capsize=4,
                   label="Macro F1", color="#5b8bd4", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Multi-scale temporal context: κ and F1 across conditions", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar_set in [bars1, bars2]:
        for bar in bar_set:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig_path = "temporal_context_multiscale.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved {fig_path}")

    # --- Save JSON ---
    json_path = "temporal_context_multiscale.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results -> {json_path}")

    elapsed_total = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)")


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else None
    main(npz)
