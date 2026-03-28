#!/usr/bin/env python3
"""Temporal context test — Colab GPU version.

Run on Google Colab with GPU runtime. Upload hmc_features.npz first.
Uses XGBoost with tree_method='gpu_hist' for fast training.

Tests 5 conditions with 5-fold subject-stratified CV:
  1. Spectral only, no context (8 features)
  2. Spectral + geometric, no context (16 features)
  3. Spectral only, 5-epoch context (40 features)
  4. Spectral + geometric, 5-epoch context (80 features)
  5. Geometric only, 5-epoch context (40 features)

Usage (Colab):
  1. Upload hmc_features.npz when prompted
  2. Script runs all 5 conditions with XGBoost GPU
  3. Results saved to temporal_context.json + bar chart PNG
"""

# --- Install XGBoost if needed ---
import subprocess
import sys
try:
    import xgboost
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    import xgboost

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTEXT_HALF = 2  # epochs on each side -> 5-epoch window

STAGES = ["W", "N1", "N2", "N3", "REM"]

SPECTRAL_NAMES = [
    "delta", "theta", "alpha", "sigma", "beta",
    "delta_beta", "hjorth_activity", "hjorth_mobility",
]
GEOM_NAMES = [
    "omega1", "torus_curvature", "angular_acceleration",
    "geodesic_distance", "angular_entropy", "phase_diff_std",
    "phase_coherence", "transition_rate",
]

N_SPECTRAL = len(SPECTRAL_NAMES)   # 8
N_GEOM = len(GEOM_NAMES)           # 8
N_TOTAL = N_SPECTRAL + N_GEOM      # 16


# ===================================================================
# Load data
# ===================================================================
def load_features(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load hmc_features.npz. Returns (features, stages, subjects)."""
    data = np.load(npz_path)
    return data["features"], data["stages"], data["subjects"]


# ===================================================================
# Temporal context expansion
# ===================================================================
def build_context_per_subject(
    features: np.ndarray,
    subjects: np.ndarray,
    half: int = CONTEXT_HALF,
) -> np.ndarray:
    """Expand (N, d) -> (N, d*(2*half+1)) with boundary replication per subject."""
    N, d = features.shape
    window = 2 * half + 1
    out = np.zeros((N, d * window), dtype=np.float32)

    # Find subject boundaries
    unique_subjs = np.unique(subjects)
    for s in unique_subjs:
        mask = subjects == s
        idxs = np.where(mask)[0]
        n_s = len(idxs)
        subj_feats = features[idxs]

        for local_i in range(n_s):
            parts = []
            for offset in range(-half, half + 1):
                j = max(0, min(n_s - 1, local_i + offset))
                parts.append(subj_feats[j])
            out[idxs[local_i]] = np.concatenate(parts)

    return out


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    # --- Locate or upload the .npz file ---
    npz_path = "hmc_features.npz"

    if not os.path.exists(npz_path):
        # Try results/ subdirectory
        if os.path.exists("results/hmc_features.npz"):
            npz_path = "results/hmc_features.npz"
        else:
            # Colab: prompt upload
            try:
                from google.colab import files
                print("Upload hmc_features.npz:")
                uploaded = files.upload()
                npz_path = list(uploaded.keys())[0]
            except ImportError:
                print("ERROR: hmc_features.npz not found. Run extract_features_hmc.py first.")
                sys.exit(1)

    print("=" * 70)
    print("Temporal Context Test — XGBoost GPU")
    print("=" * 70)

    features, stages, subjects = load_features(npz_path)
    N = len(stages)
    n_subjects = len(np.unique(subjects))
    print(f"  Loaded: {N} epochs, {n_subjects} subjects, {features.shape[1]} features")
    print(f"  Stage distribution: {dict(zip(*np.unique(stages, return_counts=True)))}\n")

    # --- Build temporal context ---
    print("  Building 5-epoch context windows ...")
    ctx = build_context_per_subject(features, subjects, CONTEXT_HALF)
    print(f"  Context shape: {ctx.shape}\n")

    # --- Feature slices ---
    spec_idx = slice(0, N_SPECTRAL)
    spec_ctx_idx = np.concatenate([
        np.arange(i * N_TOTAL, i * N_TOTAL + N_SPECTRAL)
        for i in range(2 * CONTEXT_HALF + 1)
    ])
    geom_ctx_idx = np.concatenate([
        np.arange(i * N_TOTAL + N_SPECTRAL, (i + 1) * N_TOTAL)
        for i in range(2 * CONTEXT_HALF + 1)
    ])

    # --- 5 conditions ---
    conditions = [
        ("Spec only (8)",           features[:, spec_idx]),
        ("Spec+Geom (16)",          features),
        ("Spec ctx (40)",           ctx[:, spec_ctx_idx]),
        ("Spec+Geom ctx (80)",      ctx),
        ("Geom ctx (40)",           ctx[:, geom_ctx_idx]),
    ]

    # --- Detect GPU availability ---
    try:
        tree_method = "gpu_hist"
        # Quick test to see if GPU is available
        test_clf = XGBClassifier(tree_method="gpu_hist", n_estimators=1, device="cuda")
        test_clf.fit(features[:10], stages[:10])
        device = "cuda"
        print("  GPU detected: using tree_method='gpu_hist', device='cuda'\n")
    except Exception:
        try:
            test_clf = XGBClassifier(tree_method="gpu_hist", n_estimators=1)
            test_clf.fit(features[:10], stages[:10])
            device = None
            print("  GPU detected: using tree_method='gpu_hist'\n")
        except Exception:
            tree_method = "hist"
            device = None
            print("  No GPU detected: falling back to tree_method='hist' (CPU)\n")

    # --- 5-fold subject-stratified CV ---
    print("=" * 70)
    print("5-fold subject-stratified cross-validation")
    print("=" * 70)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(features, stages, groups=subjects))

    results: dict[str, dict] = {}

    for cond_name, X_cond in conditions:
        print(f"\n  {cond_name} ({X_cond.shape[1]}d) ...")

        fold_accs, fold_kappas, fold_f1ms = [], [], []
        fold_f1_per = []

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            X_tr = np.nan_to_num(X_cond[train_idx])
            y_tr = stages[train_idx]
            X_te = np.nan_to_num(X_cond[test_idx])
            y_te = stages[test_idx]

            clf_params = dict(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                tree_method=tree_method,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            if device:
                clf_params["device"] = device

            clf = XGBClassifier(**clf_params)
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)

            fold_accs.append(accuracy_score(y_te, pred))
            fold_kappas.append(cohen_kappa_score(y_te, pred))
            fold_f1ms.append(f1_score(y_te, pred, average="macro"))
            fold_f1_per.append(f1_score(y_te, pred, average=None, labels=list(range(5))))

            print(f"    Fold {fold_i+1}/5: Acc={fold_accs[-1]:.3f}  "
                  f"k={fold_kappas[-1]:.3f}  F1m={fold_f1ms[-1]:.3f}")

        mean_acc = float(np.mean(fold_accs))
        mean_kappa = float(np.mean(fold_kappas))
        mean_f1m = float(np.mean(fold_f1ms))
        mean_f1_per = np.mean(fold_f1_per, axis=0)

        results[cond_name] = {
            "n_features": X_cond.shape[1],
            "accuracy": {"mean": round(mean_acc, 4), "std": round(float(np.std(fold_accs)), 4),
                         "folds": [round(v, 4) for v in fold_accs]},
            "kappa": {"mean": round(mean_kappa, 4), "std": round(float(np.std(fold_kappas)), 4),
                      "folds": [round(v, 4) for v in fold_kappas]},
            "f1_macro": {"mean": round(mean_f1m, 4), "std": round(float(np.std(fold_f1ms)), 4),
                         "folds": [round(v, 4) for v in fold_f1ms]},
            "f1_per_stage": {s: round(float(v), 4) for s, v in zip(STAGES, mean_f1_per)},
        }

        per_stage_str = "  ".join(f"{s}={v:.2f}" for s, v in zip(STAGES, mean_f1_per))
        print(f"  => Acc={mean_acc:.3f}  k={mean_kappa:.3f}  F1m={mean_f1m:.3f}  [{per_stage_str}]")

    # --- Wilcoxon test: condition 2 vs 4 ---
    cond2_name = "Spec+Geom (16)"
    cond4_name = "Spec+Geom ctx (80)"
    f1_folds_2 = results[cond2_name]["f1_macro"]["folds"]
    f1_folds_4 = results[cond4_name]["f1_macro"]["folds"]

    if len(f1_folds_2) >= 5:
        stat, p_wilcoxon = wilcoxon(f1_folds_4, f1_folds_2, alternative="greater")
    else:
        stat, p_wilcoxon = 0, 1.0

    delta_f1 = results[cond4_name]["f1_macro"]["mean"] - results[cond2_name]["f1_macro"]["mean"]
    delta_kappa = results[cond4_name]["kappa"]["mean"] - results[cond2_name]["kappa"]["mean"]

    print(f"\n  Wilcoxon (cond 4 > cond 2): stat={stat}, p={p_wilcoxon:.4f}")
    print(f"  Delta F1m: {delta_f1:+.4f}   Delta kappa: {delta_kappa:+.4f}")

    results["wilcoxon_cond4_vs_cond2"] = {
        "statistic": float(stat),
        "p_value": round(float(p_wilcoxon), 6),
        "delta_f1_macro": round(delta_f1, 4),
        "delta_kappa": round(delta_kappa, 4),
    }

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    header = (f"  {'Condition':<28s} {'Dim':>4s} {'Acc':>7s} {'k':>7s} {'F1m':>7s}")
    for s in STAGES:
        header += f" {s:>5s}"
    print(header)
    print("  " + "-" * (28 + 4 + 7 * 3 + 6 * 5))

    for cond_name in [c[0] for c in conditions]:
        r = results[cond_name]
        line = (f"  {cond_name:<28s} {r['n_features']:>4d}"
                f" {r['accuracy']['mean']:>7.3f}"
                f" {r['kappa']['mean']:>7.3f}"
                f" {r['f1_macro']['mean']:>7.3f}")
        for s in STAGES:
            line += f" {r['f1_per_stage'][s]:>5.2f}"
        print(line)

    print(f"\n  Gap closure (cond 2 -> 4):")
    print(f"    Delta kappa: {delta_kappa:+.3f}")
    print(f"    Delta F1m:   {delta_f1:+.3f}")
    print(f"    Wilcoxon p:  {p_wilcoxon:.4f}")

    # --- Figure: bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    cond_labels = [c[0] for c in conditions]
    kappas = [results[c]["kappa"]["mean"] for c in cond_labels]
    kappa_stds = [results[c]["kappa"]["std"] for c in cond_labels]
    f1ms = [results[c]["f1_macro"]["mean"] for c in cond_labels]

    x = np.arange(len(cond_labels))
    w = 0.35
    bars1 = ax.bar(x - w / 2, kappas, w, yerr=kappa_stds, capsize=4,
                   label="Cohen's kappa", color="#5bffa8", edgecolor="white")
    bars2 = ax.bar(x + w / 2, f1ms, w,
                   label="Macro F1", color="#5b8bd4", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Temporal context test: kappa and F1 across conditions (XGBoost GPU)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar_set in [bars1, bars2]:
        for bar in bar_set:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig_path = "temporal_context_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved {fig_path}")

    # --- Save JSON ---
    json_path = "temporal_context.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results -> {json_path}")

    # --- Download results on Colab ---
    try:
        from google.colab import files
        files.download(json_path)
        files.download(fig_path)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
