#!/usr/bin/env python3
"""NeuroSpiral — Clinical Validation with Strict N3 Threshold.

Implements precision-first validation for closed-loop stimulation safety:

    CLINICAL RATIONALE:
    Emitting sound outside N3 can disrupt sleep architecture.
    → Prioritize PRECISION (minimize false stimulation)
    → Accept some missed N3 windows (lower recall is tolerable)

    STRATEGY:
    1. Train RF on TDA+spectral features
    2. Instead of argmax, require P(N3) ≥ threshold to trigger
    3. Sweep thresholds to find optimal precision/recall tradeoff
    4. Generate visual confusion matrix + clinical report
    5. Cross-validate across folds for robust estimates

Usage:
    python scripts/clinical_validate.py --download-sample
    python scripts/clinical_validate.py --psg file.edf --hyp hyp.edf --threshold 0.80
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from scipy import signal as scipy_signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.edf_loader import load_sleep_edf, extract_epochs_from_annotations
from src.preprocessing.pipeline import preprocess_raw, compute_epoch_quality
from src.features.takens import time_delay_embedding
from src.features.topology import extract_tda_features


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

LABEL_MAPPING = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Movement time": "W",
    "Sleep stage ?": None,
}

# Simplified 4-class grouping for clinical interpretation
CLINICAL_GROUPS = {
    "W": 0,       # Vigilia
    "N1": 1,      # NREM Ligero
    "N2": 1,      # NREM Ligero (grouped)
    "N3": 2,      # NREM Profundo (TARGET)
    "REM": 3,     # REM
}
CLINICAL_NAMES = ["Vigilia", "NREM Ligero", "NREM Profundo", "REM"]

SPECTRAL_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 15.0),
    "beta": (15.0, 30.0),
}


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def download_sample(output_dir: Path) -> tuple[Path, Path]:
    import urllib.request
    output_dir.mkdir(parents=True, exist_ok=True)
    base = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
    files = {
        "SC4001E0-PSG.edf": f"{base}/SC4001E0-PSG.edf",
        "SC4001EC-Hypnogram.edf": f"{base}/SC4001EC-Hypnogram.edf",
    }
    paths = {}
    for fname, url in files.items():
        p = output_dir / fname
        if not p.exists():
            print(f"  ↓ {fname}...")
            urllib.request.urlretrieve(url, p)
        paths[fname] = p
    return paths["SC4001E0-PSG.edf"], paths["SC4001EC-Hypnogram.edf"]


def load_and_extract(
    psg_path: Path,
    hyp_path: Path,
    use_tda: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Full pipeline: load → preprocess → extract features."""

    record = load_sleep_edf(
        psg_path, hyp_path,
        channels=["EEG Fpz-Cz"],
        label_mapping=LABEL_MAPPING,
    )
    result = preprocess_raw(
        record.raw, 0.5, 30.0, 100.0,
        {"n_components": 10, "method": "fastica",
         "max_iter": 500, "random_state": 42, "eog_threshold": 0.85},
    )
    record.raw = result.raw
    sfreq = result.raw.info["sfreq"]

    epochs, labels_5class, names_5class = extract_epochs_from_annotations(record)
    quality = compute_epoch_quality(epochs, sfreq)
    epochs, labels_5class = epochs[quality], labels_5class[quality]

    # Map 5-class → 4-class clinical grouping
    labels_4class = np.array([
        CLINICAL_GROUPS[names_5class[l]] for l in labels_5class
    ])

    # Extract features
    n_epochs = epochs.shape[0]
    all_feats = []
    print(f"  Extracting features from {n_epochs} epochs...")

    for i in range(n_epochs):
        if i % 100 == 0:
            print(f"    {i}/{n_epochs} ({i/n_epochs*100:.0f}%)")

        epoch_1d = epochs[i, 0, :]
        feats = {}

        # Spectral
        freqs, psd = scipy_signal.welch(epoch_1d, fs=sfreq, nperseg=min(256, len(epoch_1d)))
        total = np.trapz(psd, freqs)
        for name, (lo, hi) in SPECTRAL_BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            feats[f"spec_{name}"] = float(np.trapz(psd[mask], freqs[mask]) / (total + 1e-10))
        feats["spec_delta_beta"] = feats["spec_delta"] / (feats["spec_beta"] + 1e-10)

        # TDA
        if use_tda:
            try:
                cloud, _ = time_delay_embedding(epoch_1d, dimension=4)
                tda = extract_tda_features(cloud, max_dim=2, n_subsample=300)
                feats.update(tda)
            except Exception:
                pass

        all_feats.append(feats)

    all_keys = sorted(set().union(*[f.keys() for f in all_feats]))
    X = np.array([[f.get(k, 0.0) for k in all_keys] for f in all_feats])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, labels_4class, all_keys


# ──────────────────────────────────────────────────────────────
# Strict threshold prediction
# ──────────────────────────────────────────────────────────────

def predict_with_strict_threshold(
    y_probs: np.ndarray,
    n3_class_idx: int = 2,
    threshold: float = 0.85,
    fallback_class: int = 1,  # NREM Ligero (safe fallback)
) -> np.ndarray:
    """Apply strict confidence threshold for N3 classification.

    If the model predicts N3 but with confidence < threshold,
    reclassify to fallback (NREM Ligero) where no stimulation occurs.

    Parameters
    ----------
    y_probs : (n_samples, n_classes) probability matrix
    n3_class_idx : index of NREM Profundo class
    threshold : minimum confidence to classify as N3
    fallback_class : class to assign when N3 confidence is below threshold
    """
    y_pred = np.argmax(y_probs, axis=1)

    # Penalize uncertain N3 predictions
    uncertain_n3 = (y_pred == n3_class_idx) & (y_probs[:, n3_class_idx] < threshold)
    y_pred[uncertain_n3] = fallback_class

    return y_pred


# ──────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    threshold: float,
    save_path: Path,
):
    """Publication-quality confusion matrix with clinical annotations."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Color map: emphasize the N3 row/column
    cmap = plt.cm.Blues

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    # Annotate cells with count and percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100

            # Highlight N3 diagonal (true positive)
            if i == 2 and j == 2:
                color = "white" if pct > 50 else "#0C447C"
                weight = "bold"
            # Highlight N3 false positives (column 2, not row 2)
            elif j == 2 and i != 2:
                color = "#993C1D" if pct > 10 else ("white" if pct > 50 else "#333")
                weight = "bold" if pct > 10 else "normal"
            else:
                color = "white" if pct > 50 else "#333"
                weight = "normal"

            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=11, fontweight=weight, color=color)

    # Labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicción del algoritmo", fontsize=12, labelpad=10)
    ax.set_ylabel("Fase real (hipnograma médico)", fontsize=12, labelpad=10)
    ax.set_title(
        f"Matriz de confusión clínica — Umbral N3 ≥ {threshold:.0%}\n"
        f"Falsos positivos N3 penalizados",
        fontsize=13, fontweight="bold", pad=15,
    )

    # Highlight N3 row and column
    ax.add_patch(plt.Rectangle((-0.5, 1.5), 4, 1,
                                fill=False, edgecolor="#D85A30",
                                linewidth=2.5, linestyle="--"))
    ax.add_patch(plt.Rectangle((1.5, -0.5), 1, 4,
                                fill=False, edgecolor="#D85A30",
                                linewidth=2.5, linestyle="--"))

    plt.colorbar(im, ax=ax, label="Proporción", shrink=0.8)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  ✓ Confusion matrix saved: {save_path}")
    plt.close()


def plot_threshold_sweep(
    y_true_binary: np.ndarray,
    y_scores: np.ndarray,
    optimal_threshold: float,
    save_path: Path,
):
    """Sweep thresholds showing precision/recall/F1 tradeoff."""
    thresholds = np.arange(0.30, 0.96, 0.01)
    precisions, recalls, f1s = [], [], []

    for thr in thresholds:
        preds = (y_scores >= thr).astype(int)
        p = precision_score(y_true_binary, preds, zero_division=0)
        r = recall_score(y_true_binary, preds, zero_division=0)
        f = f1_score(y_true_binary, preds, zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(thresholds, precisions, color="#534AB7", linewidth=2, label="Precision (falsos positivos ↓)")
    ax.plot(thresholds, recalls, color="#1D9E75", linewidth=2, label="Recall (N3 perdido ↓)")
    ax.plot(thresholds, f1s, color="#D85A30", linewidth=2, linestyle="--", label="F1-Score")

    # Mark optimal
    ax.axvline(optimal_threshold, color="#D85A30", linewidth=1.5, linestyle=":",
               label=f"Umbral óptimo ({optimal_threshold:.2f})")

    # Shade "safe zone" (precision > 0.85)
    safe_mask = np.array(precisions) >= 0.85
    if safe_mask.any():
        first_safe = thresholds[safe_mask][0]
        ax.axvspan(first_safe, thresholds[-1], alpha=0.08, color="#1D9E75")
        ax.text(first_safe + 0.02, 0.95, "Zona segura\n(precision ≥ 85%)",
                fontsize=9, color="#0F6E56", va="top")

    ax.set_xlabel("Umbral de confianza P(N3)", fontsize=12)
    ax.set_ylabel("Métrica", fontsize=12)
    ax.set_title("Barrido de umbral — Tradeoff precision vs recall para N3",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="center left", fontsize=10)
    ax.set_xlim(0.30, 0.95)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  ✓ Threshold sweep saved: {save_path}")
    plt.close()


def plot_hypnogram_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    save_path: Path,
    max_epochs: int = 300,
):
    """Side-by-side hypnogram: real vs predicted (temporal view)."""
    n = min(len(y_true), max_epochs)
    t = np.arange(n) * 30 / 3600  # hours

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 6),
                                         height_ratios=[1, 1, 0.4],
                                         sharex=True)

    # Real hypnogram
    ax1.step(t, y_true[:n], where="post", color="#534AB7", linewidth=1.2)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_ylabel("Real", fontsize=11, fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.2)

    # Predicted hypnogram
    ax2.step(t, y_pred[:n], where="post", color="#1D9E75", linewidth=1.2)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_ylabel("Predicho", fontsize=11, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.2)

    # Error strip
    errors = y_true[:n] != y_pred[:n]
    ax3.fill_between(t, 0, errors.astype(float),
                     step="post", color="#D85A30", alpha=0.6)
    ax3.set_ylabel("Error", fontsize=10)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["✓", "✗"], fontsize=9)
    ax3.set_xlabel("Tiempo (horas)", fontsize=12)
    ax3.set_ylim(-0.1, 1.1)

    fig.suptitle("Hipnograma: Real vs Predicho",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  ✓ Hypnogram comparison saved: {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# Main validation
# ──────────────────────────────────────────────────────────────

def run_clinical_validation(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.85,
    output_dir: Path | None = None,
):
    """Full clinical validation with strict N3 threshold."""

    print("\n" + "═" * 60)
    print("  NEUROSPIRAL — Validación Clínica")
    print("  Prioridad: minimizar falsos positivos en NREM Profundo")
    print("═" * 60)

    N3_IDX = 2  # NREM Profundo in 4-class scheme
    n_folds = 5

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # ── 1. Cross-validated predictions (both default and strict) ──

    print(f"\n[1/5] Cross-validación estratificada ({n_folds} folds)...")

    all_y_true = []
    all_y_pred_default = []
    all_y_pred_strict = []
    all_y_proba = []

    fold_metrics_default = {"precision": [], "recall": [], "f1": [], "kappa": []}
    fold_metrics_strict = {"precision": [], "recall": [], "f1": [], "kappa": []}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_s, y)):
        X_tr, X_te = X_s[train_idx], X_s[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf.fit(X_tr, y_tr)
        y_proba = clf.predict_proba(X_te)
        y_default = np.argmax(y_proba, axis=1)
        y_strict = predict_with_strict_threshold(
            y_proba, n3_class_idx=N3_IDX,
            threshold=threshold, fallback_class=1,
        )

        # N3 binary metrics
        n3_true = (y_te == N3_IDX).astype(int)
        n3_default = (y_default == N3_IDX).astype(int)
        n3_strict = (y_strict == N3_IDX).astype(int)

        fold_metrics_default["precision"].append(precision_score(n3_true, n3_default, zero_division=0))
        fold_metrics_default["recall"].append(recall_score(n3_true, n3_default, zero_division=0))
        fold_metrics_default["f1"].append(f1_score(n3_true, n3_default, zero_division=0))
        fold_metrics_default["kappa"].append(cohen_kappa_score(y_te, y_default))

        fold_metrics_strict["precision"].append(precision_score(n3_true, n3_strict, zero_division=0))
        fold_metrics_strict["recall"].append(recall_score(n3_true, n3_strict, zero_division=0))
        fold_metrics_strict["f1"].append(f1_score(n3_true, n3_strict, zero_division=0))
        fold_metrics_strict["kappa"].append(cohen_kappa_score(y_te, y_strict))

        all_y_true.extend(y_te)
        all_y_pred_default.extend(y_default)
        all_y_pred_strict.extend(y_strict)
        all_y_proba.extend(y_proba)

        print(f"  Fold {fold+1}: "
              f"P(N3)={fold_metrics_strict['precision'][-1]:.3f} "
              f"R(N3)={fold_metrics_strict['recall'][-1]:.3f} "
              f"κ={fold_metrics_strict['kappa'][-1]:.3f}")

    all_y_true = np.array(all_y_true)
    all_y_pred_default = np.array(all_y_pred_default)
    all_y_pred_strict = np.array(all_y_pred_strict)
    all_y_proba = np.array(all_y_proba)

    # ── 2. Compare default vs strict ──

    print(f"\n[2/5] Comparación: umbral estándar (50%) vs estricto ({threshold:.0%})...")

    print(f"\n  {'Métrica N3':<25} {'Default (50%)':>14} {'Estricto ({:.0f}%)':>14}".format(threshold * 100))
    print(f"  {'─'*25} {'─'*14} {'─'*14}")

    for metric in ["precision", "recall", "f1", "kappa"]:
        d_mean = np.mean(fold_metrics_default[metric])
        s_mean = np.mean(fold_metrics_strict[metric])
        delta = s_mean - d_mean
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  {metric:<25} {d_mean:>14.3f} {s_mean:>14.3f}  {arrow} {abs(delta):.3f}")

    # ── 3. Optimal threshold search ──

    print(f"\n[3/5] Búsqueda de umbral óptimo...")

    n3_true_binary = (all_y_true == N3_IDX).astype(int)
    n3_scores = all_y_proba[:, N3_IDX]

    best_threshold = threshold
    best_precision_at_target = 0.0

    print(f"\n  {'Umbral':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FP N3':>8} {'FN N3':>8}")
    print(f"  {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for thr in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        preds = (n3_scores >= thr).astype(int)
        p = precision_score(n3_true_binary, preds, zero_division=0)
        r = recall_score(n3_true_binary, preds, zero_division=0)
        f = f1_score(n3_true_binary, preds, zero_division=0)
        fp = np.sum((preds == 1) & (n3_true_binary == 0))
        fn = np.sum((preds == 0) & (n3_true_binary == 1))

        marker = ""
        if p >= 0.85 and r >= 0.50:  # minimum viable recall
            if p > best_precision_at_target:
                best_precision_at_target = p
                best_threshold = thr
                marker = " ◄"

        print(f"  {thr:>8.2f} {p:>10.3f} {r:>8.3f} {f:>8.3f} {fp:>8d} {fn:>8d}{marker}")

    print(f"\n  Umbral recomendado: {best_threshold:.2f}")

    # ── 4. Clinical report ──

    print(f"\n[4/5] Reporte clínico (umbral = {best_threshold:.2f})...")

    y_final = predict_with_strict_threshold(
        all_y_proba, N3_IDX, best_threshold, fallback_class=1,
    )

    print(f"\n  Classification Report:")
    report = classification_report(
        all_y_true, y_final,
        target_names=CLINICAL_NAMES,
        digits=3, zero_division=0,
    )
    for line in report.split("\n"):
        print(f"    {line}")

    # Safety assessment
    fp_n3 = np.sum((y_final == N3_IDX) & (all_y_true != N3_IDX))
    fn_n3 = np.sum((y_final != N3_IDX) & (all_y_true == N3_IDX))
    tp_n3 = np.sum((y_final == N3_IDX) & (all_y_true == N3_IDX))
    total_stim = tp_n3 + fp_n3

    print(f"\n  ┌──────────────────────────────────────────────────┐")
    print(f"  │  INFORME DE SEGURIDAD CLÍNICA                    │")
    print(f"  ├──────────────────────────────────────────────────┤")
    print(f"  │  Umbral de confianza:    {best_threshold:.0%}                    │")
    print(f"  │                                                  │")
    print(f"  │  Estimulaciones correctas (TP):  {tp_n3:>5}            │")
    print(f"  │  Falsas estimulaciones (FP):     {fp_n3:>5}            │")
    print(f"  │  N3 perdido (FN):                {fn_n3:>5}            │")
    print(f"  │                                                  │")
    if total_stim > 0:
        stim_accuracy = tp_n3 / total_stim
        print(f"  │  Fiabilidad de estimulación:     {stim_accuracy:.1%}           │")
    print(f"  │  Cobertura de N3:                "
          f"{tp_n3/(tp_n3+fn_n3):.1%}           │")
    print(f"  │                                                  │")

    # Final verdict
    n3_prec = precision_score(n3_true_binary, (y_final == N3_IDX).astype(int), zero_division=0)
    if n3_prec >= 0.90:
        print(f"  │  ✅ APTO para estimulación acústica              │")
    elif n3_prec >= 0.80:
        print(f"  │  ⚠️  APTO CON RESERVAS — subir umbral           │")
    else:
        print(f"  │  ❌ NO APTO — precisión insuficiente             │")
    print(f"  └──────────────────────────────────────────────────┘")

    # ── 5. Visualizations ──

    if output_dir:
        print(f"\n[5/5] Generando visualizaciones...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        plot_confusion_matrix(
            all_y_true, y_final, CLINICAL_NAMES,
            best_threshold,
            output_dir / "confusion_matrix_clinical.png",
        )

        # Threshold sweep
        plot_threshold_sweep(
            n3_true_binary, n3_scores, best_threshold,
            output_dir / "threshold_sweep_n3.png",
        )

        # Hypnogram comparison
        plot_hypnogram_comparison(
            all_y_true, y_final, CLINICAL_NAMES,
            output_dir / "hypnogram_comparison.png",
        )

        # Save raw results for further analysis
        np.savez_compressed(
            output_dir / "clinical_validation_results.npz",
            y_true=all_y_true,
            y_pred_default=all_y_pred_default,
            y_pred_strict=y_final,
            y_proba=all_y_proba,
            threshold=best_threshold,
            clinical_names=CLINICAL_NAMES,
        )
        print(f"  ✓ Results saved to {output_dir}/")

    return {
        "threshold": best_threshold,
        "n3_precision": n3_prec,
        "n3_recall": recall_score(n3_true_binary, (y_final == N3_IDX).astype(int), zero_division=0),
        "kappa": cohen_kappa_score(all_y_true, y_final),
    }


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NeuroSpiral — Clinical Validation with Strict N3 Threshold"
    )
    parser.add_argument("--psg", type=Path)
    parser.add_argument("--hyp", type=Path)
    parser.add_argument("--download-sample", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Initial N3 confidence threshold (default: 0.85)")
    parser.add_argument("--no-tda", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data/results/clinical")
    args = parser.parse_args()

    if args.download_sample:
        print("📥 Descargando muestra de Sleep-EDF...")
        psg, hyp = download_sample(PROJECT_ROOT / "data/raw")
    elif args.psg and args.hyp:
        psg, hyp = args.psg, args.hyp
    else:
        parser.error("Usa --download-sample o proporciona --psg y --hyp")
        return

    print(f"\n🧠 Cargando y procesando datos (TDA={'sí' if not args.no_tda else 'no'})...")
    t0 = time.time()
    X, y, feat_names = load_and_extract(psg, hyp, use_tda=not args.no_tda)
    print(f"  ✓ {X.shape[0]} épocas × {X.shape[1]} features en {time.time()-t0:.1f}s")

    print(f"\n  Distribución de fases:")
    for i, name in enumerate(CLINICAL_NAMES):
        count = np.sum(y == i)
        pct = count / len(y) * 100
        print(f"    {name:<15}: {count:>4} ({pct:5.1f}%)")

    results = run_clinical_validation(
        X, y,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )

    print(f"\n{'═'*60}")
    print(f"  Validación clínica completa")
    print(f"  N3 Precision: {results['n3_precision']:.3f} | "
          f"N3 Recall: {results['n3_recall']:.3f} | "
          f"κ: {results['kappa']:.3f}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
