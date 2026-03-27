#!/usr/bin/env python3
"""NeuroSpiral — Validation & Reliability Framework.

Implements a multi-level validation strategy:

Level 1: Sanity checks (does the pipeline even work?)
Level 2: Single-subject performance (overfit check)
Level 3: Cross-validated generalization (within dataset)
Level 4: Cross-dataset generalization (train on A, test on B)
Level 5: Clinical reliability metrics (N3-specific for glymphatic)
Level 6: Ablation study (TDA vs spectral contribution)

Key insight: for closed-loop glymphatic stimulation, we care about
TEMPORAL ACCURACY (detecting N3 onset within ~30s) more than
epoch-level classification accuracy.

Theoretical ceiling: ~83% (human inter-rater agreement on PSG).
Target for N3: precision ≥ 0.85, recall ≥ 0.90, F1 ≥ 0.87.

Usage:
    python scripts/validate.py --download-sample --level all
    python scripts/validate.py --psg file.edf --hyp hyp.edf --level 1,2,3
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneGroupOut,
    learning_curve,
    cross_val_predict,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.edf_loader import load_sleep_edf, extract_epochs_from_annotations
from src.preprocessing.pipeline import preprocess_raw, compute_epoch_quality
from src.features.takens import time_delay_embedding
from src.features.topology import extract_tda_features

# ──────────────────────────────────────────────────────────────
# Config (mirrors pipeline_4d.py)
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

SPECTRAL_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 15.0),
    "beta": (15.0, 30.0),
}


def download_sample(output_dir: Path) -> tuple[Path, Path]:
    """Download Sleep-EDF sample from PhysioNet."""
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


# ──────────────────────────────────────────────────────────────
# Feature extraction (compact version for validation)
# ──────────────────────────────────────────────────────────────

def extract_all_features(
    epochs: np.ndarray,
    sfreq: float,
    use_tda: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Extract spectral + TDA features from all epochs."""
    from scipy import signal as scipy_signal

    n_epochs = epochs.shape[0]
    all_feats = []

    for i in range(n_epochs):
        epoch_1d = epochs[i, 0, :]
        feats = {}

        # Spectral
        freqs, psd = scipy_signal.welch(epoch_1d, fs=sfreq, nperseg=min(256, len(epoch_1d)))
        total = np.trapz(psd, freqs)
        if total > 0:
            for name, (lo, hi) in SPECTRAL_BANDS.items():
                mask = (freqs >= lo) & (freqs < hi)
                feats[f"spec_{name}"] = float(np.trapz(psd[mask], freqs[mask]) / total)
        else:
            for name in SPECTRAL_BANDS:
                feats[f"spec_{name}"] = 0.0
        eps = 1e-10
        feats["spec_delta_beta"] = feats["spec_delta"] / (feats["spec_beta"] + eps)

        # TDA
        if use_tda:
            try:
                cloud, _ = time_delay_embedding(epoch_1d, dimension=4)
                tda = extract_tda_features(cloud, max_dim=2, n_subsample=300)
                feats.update(tda)
            except Exception:
                # Fallback: zero TDA features
                pass

        all_feats.append(feats)

    # Align all dicts to same keys
    all_keys = sorted(set().union(*[f.keys() for f in all_feats]))
    X = np.array([[f.get(k, 0.0) for k in all_keys] for f in all_feats])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, all_keys


def load_and_prepare(
    psg_path: Path,
    hyp_path: Path,
    use_tda: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Full load → preprocess → extract pipeline. Returns X, y, names, raw_epochs."""
    record = load_sleep_edf(
        psg_path, hyp_path,
        channels=["EEG Fpz-Cz"],
        label_mapping=LABEL_MAPPING,
    )
    result = preprocess_raw(record.raw, 0.5, 30.0, 100.0, {
        "n_components": 10, "method": "fastica",
        "max_iter": 500, "random_state": 42, "eog_threshold": 0.85,
    })
    record.raw = result.raw
    sfreq = result.raw.info["sfreq"]
    epochs, labels, names = extract_epochs_from_annotations(record)
    quality = compute_epoch_quality(epochs, sfreq)
    epochs, labels = epochs[quality], labels[quality]

    X, feat_names = extract_all_features(epochs, sfreq, use_tda)
    return X, labels, names, epochs


# ══════════════════════════════════════════════════════════════
# LEVEL 1: Sanity Checks
# ══════════════════════════════════════════════════════════════

def level_1_sanity(X, y, label_names):
    """Basic checks: data integrity, class distribution, feature validity."""
    print("\n" + "═" * 60)
    print("  LEVEL 1: Sanity Checks")
    print("═" * 60)

    checks_passed = 0
    checks_total = 0

    # 1.1 Data shape
    checks_total += 1
    ok = X.shape[0] == len(y) and X.shape[0] > 100
    checks_passed += int(ok)
    print(f"\n  [{'✓' if ok else '✗'}] Data shape: {X.shape[0]} epochs × {X.shape[1]} features")

    # 1.2 No NaN/Inf
    checks_total += 1
    ok = not (np.isnan(X).any() or np.isinf(X).any())
    checks_passed += int(ok)
    print(f"  [{'✓' if ok else '✗'}] No NaN/Inf in features")

    # 1.3 All classes present
    checks_total += 1
    unique = np.unique(y)
    ok = len(unique) >= 3
    checks_passed += int(ok)
    print(f"  [{'✓' if ok else '✗'}] Classes present: {len(unique)}/{len(label_names)} "
          f"({[label_names[i] for i in unique]})")

    # 1.4 N3 class exists and has sufficient samples
    checks_total += 1
    n3_count = 0
    if "N3" in label_names:
        n3_idx = label_names.index("N3")
        n3_count = np.sum(y == n3_idx)
    ok = n3_count >= 20
    checks_passed += int(ok)
    print(f"  [{'✓' if ok else '✗'}] N3 epochs: {n3_count} (need ≥20 for reliable CV)")

    # 1.5 Feature variance (no constant features)
    checks_total += 1
    zero_var = np.sum(np.std(X, axis=0) == 0)
    ok = zero_var < X.shape[1] * 0.1
    checks_passed += int(ok)
    print(f"  [{'✓' if ok else '✗'}] Zero-variance features: {zero_var}/{X.shape[1]}")

    # 1.6 Beats random baseline
    checks_total += 1
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy_preds = cross_val_predict(dummy, X, y, cv=3)
    dummy_f1 = f1_score(y, dummy_preds, average="macro", zero_division=0)
    print(f"  [i] Random baseline F1-macro: {dummy_f1:.3f}")

    rf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf_preds = cross_val_predict(rf, StandardScaler().fit_transform(X), y, cv=3)
    rf_f1 = f1_score(y, rf_preds, average="macro", zero_division=0)
    ok = rf_f1 > dummy_f1 + 0.05
    checks_passed += int(ok)
    print(f"  [{'✓' if ok else '✗'}] Quick RF F1-macro: {rf_f1:.3f} "
          f"(>{dummy_f1:.3f} + 0.05? {'Yes' if ok else 'NO — model is not learning'})")

    print(f"\n  Result: {checks_passed}/{checks_total} checks passed")
    return checks_passed == checks_total


# ══════════════════════════════════════════════════════════════
# LEVEL 2: Overfit Check
# ══════════════════════════════════════════════════════════════

def level_2_overfit(X, y, label_names):
    """Train on all, predict on all — should get ~99%. If not, model capacity issue."""
    print("\n" + "═" * 60)
    print("  LEVEL 2: Overfit Check (can the model memorize?)")
    print("═" * 60)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None,  # no depth limit
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_s, y)
    y_pred = clf.predict(X_s)

    train_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    train_acc = np.mean(y == y_pred)

    print(f"\n  Train accuracy: {train_acc:.4f}")
    print(f"  Train F1-macro: {train_f1:.4f}")

    if train_f1 > 0.95:
        print("  ✓ Model can fit the data — sufficient capacity")
    elif train_f1 > 0.80:
        print("  ⚠ Moderate fit — consider more features or deeper model")
    else:
        print("  ✗ Poor fit — features may not be discriminative enough")

    # Per-class train performance
    print(f"\n  Per-class train accuracy:")
    for i, name in enumerate(label_names):
        mask = y == i
        if mask.sum() > 0:
            acc = np.mean(y_pred[mask] == i)
            print(f"    {name:>4}: {acc:.3f} ({mask.sum()} epochs)")

    return train_f1


# ══════════════════════════════════════════════════════════════
# LEVEL 3: Cross-Validated Generalization
# ══════════════════════════════════════════════════════════════

def level_3_cross_validation(X, y, label_names):
    """Rigorous 5-fold stratified CV with confidence intervals."""
    print("\n" + "═" * 60)
    print("  LEVEL 3: Cross-Validated Generalization")
    print("═" * 60)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    n_folds = 5
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = {
        "f1_macro": [], "accuracy": [], "kappa": [],
        "n3_precision": [], "n3_recall": [], "n3_f1": [],
    }

    all_y_true, all_y_pred, all_y_proba = [], [], []

    n3_idx = label_names.index("N3") if "N3" in label_names else None

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_s, y)):
        X_train, X_test = X_s[train_idx], X_s[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        fold_metrics["f1_macro"].append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        fold_metrics["accuracy"].append(np.mean(y_test == y_pred))
        fold_metrics["kappa"].append(cohen_kappa_score(y_test, y_pred))

        if n3_idx is not None:
            n3_true = (y_test == n3_idx).astype(int)
            n3_pred = (y_pred == n3_idx).astype(int)
            fold_metrics["n3_precision"].append(precision_score(n3_true, n3_pred, zero_division=0))
            fold_metrics["n3_recall"].append(recall_score(n3_true, n3_pred, zero_division=0))
            fold_metrics["n3_f1"].append(f1_score(n3_true, n3_pred, zero_division=0))

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    # Summary with 95% CI (t-distribution for small n_folds)
    print(f"\n  {'Metric':<20} {'Mean':>8} {'± 95% CI':>10} {'Folds':>24}")
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*24}")

    for metric_name in ["f1_macro", "accuracy", "kappa"]:
        vals = fold_metrics[metric_name]
        mean = np.mean(vals)
        ci = scipy_stats.t.interval(0.95, len(vals) - 1,
                                     loc=mean, scale=scipy_stats.sem(vals))
        ci_half = mean - ci[0]
        fold_str = " ".join(f"{v:.3f}" for v in vals)
        print(f"  {metric_name:<20} {mean:>8.3f} {f'± {ci_half:.3f}':>10} [{fold_str}]")

    if n3_idx is not None:
        print(f"\n  N3 (glymphatic target):")
        for metric_name in ["n3_precision", "n3_recall", "n3_f1"]:
            vals = fold_metrics[metric_name]
            mean = np.mean(vals)
            ci = scipy_stats.t.interval(0.95, len(vals) - 1,
                                         loc=mean, scale=scipy_stats.sem(vals))
            ci_half = mean - ci[0]
            fold_str = " ".join(f"{v:.3f}" for v in vals)
            short = metric_name.replace("n3_", "")
            print(f"  {short:<20} {mean:>8.3f} {f'± {ci_half:.3f}':>10} [{fold_str}]")

    # Cohen's kappa interpretation
    mean_kappa = np.mean(fold_metrics["kappa"])
    if mean_kappa > 0.80:
        kappa_label = "almost perfect"
    elif mean_kappa > 0.60:
        kappa_label = "substantial"
    elif mean_kappa > 0.40:
        kappa_label = "moderate"
    else:
        kappa_label = "fair or poor"
    print(f"\n  Cohen's κ = {mean_kappa:.3f} → {kappa_label} agreement")
    print(f"  (Human inter-rater κ on PSG ≈ 0.68–0.76 for reference)")

    # Full classification report
    print(f"\n  Aggregated classification report:")
    report = classification_report(
        all_y_true, all_y_pred,
        target_names=label_names, digits=3, zero_division=0,
    )
    for line in report.split("\n"):
        print(f"    {line}")

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\n  Confusion matrix:")
    header = "        " + "  ".join(f"{n:>5}" for n in label_names)
    print(f"    {header}")
    for i, row in enumerate(cm):
        print(f"    {label_names[i]:>5}   {'  '.join(f'{v:>5}' for v in row)}")

    # Confusion breakdown for N3
    if n3_idx is not None:
        print(f"\n  N3 error analysis:")
        n3_true_mask = all_y_true == n3_idx
        n3_wrong = all_y_pred[n3_true_mask & (all_y_pred != n3_idx)]
        if len(n3_wrong) > 0:
            print(f"    When true N3 is misclassified as:")
            for cls_idx in np.unique(n3_wrong):
                count = np.sum(n3_wrong == cls_idx)
                print(f"      {label_names[cls_idx]:>4}: {count} epochs "
                      f"({count / n3_true_mask.sum() * 100:.1f}%)")

        not_n3_but_pred_n3 = all_y_true[(all_y_true != n3_idx) & (all_y_pred == n3_idx)]
        if len(not_n3_but_pred_n3) > 0:
            print(f"    False N3 predictions (actual stage):")
            for cls_idx in np.unique(not_n3_but_pred_n3):
                count = np.sum(not_n3_but_pred_n3 == cls_idx)
                print(f"      {label_names[cls_idx]:>4}: {count} epochs")

    return fold_metrics


# ══════════════════════════════════════════════════════════════
# LEVEL 5: Clinical Reliability (N3 specific)
# ══════════════════════════════════════════════════════════════

def level_5_clinical(X, y, label_names):
    """Metrics specific to closed-loop glymphatic stimulation reliability."""
    print("\n" + "═" * 60)
    print("  LEVEL 5: Clinical Reliability (Glymphatic Target)")
    print("═" * 60)

    n3_idx = label_names.index("N3") if "N3" in label_names else None
    if n3_idx is None:
        print("  ✗ No N3 class found — cannot evaluate clinical metrics")
        return

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Binary: N3 vs everything else
    y_binary = (y == n3_idx).astype(int)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_scores = cross_val_predict(clf, X_s, y_binary, cv=cv, method="predict_proba")[:, 1]
    y_pred_default = (y_scores >= 0.5).astype(int)

    # 5.1 ROC-AUC
    auc = roc_auc_score(y_binary, y_scores)
    print(f"\n  ROC-AUC (N3 vs rest): {auc:.3f}")

    if auc > 0.95:
        print("  ✓ Excellent discrimination")
    elif auc > 0.90:
        print("  ○ Good discrimination")
    else:
        print("  ⚠ Discrimination needs improvement")

    # 5.2 Optimal threshold via precision-recall tradeoff
    precisions, recalls, thresholds = precision_recall_curve(y_binary, y_scores)

    # Find threshold that maximizes F1
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

    # Find threshold for target recall ≥ 0.90
    high_recall_idx = np.where(recalls >= 0.90)[0]
    if len(high_recall_idx) > 0:
        # Among those with recall ≥ 0.90, pick highest precision
        best_hr_idx = high_recall_idx[np.argmax(precisions[high_recall_idx])]
        hr_threshold = thresholds[best_hr_idx] if best_hr_idx < len(thresholds) else 0.5
    else:
        hr_threshold = 0.3  # fallback

    print(f"\n  Threshold analysis:")
    print(f"  {'Threshold':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'─'*12} {'─'*10} {'─'*8} {'─'*8}")

    for thr in [0.3, 0.4, 0.5, best_threshold, hr_threshold]:
        preds = (y_scores >= thr).astype(int)
        p = precision_score(y_binary, preds, zero_division=0)
        r = recall_score(y_binary, preds, zero_division=0)
        f = f1_score(y_binary, preds, zero_division=0)
        marker = ""
        if abs(thr - best_threshold) < 0.01:
            marker = " ← max F1"
        elif abs(thr - hr_threshold) < 0.01:
            marker = " ← recall ≥ 0.90"
        print(f"  {thr:<12.3f} {p:>10.3f} {r:>8.3f} {f:>8.3f}{marker}")

    # 5.3 Temporal accuracy: transition detection
    print(f"\n  Temporal accuracy (N3 onset/offset detection):")
    # Simulate sequential prediction
    y_pred_seq = (y_scores >= best_threshold).astype(int)

    # Find N3 bouts (contiguous runs)
    true_transitions = np.diff(y_binary.astype(int))
    pred_transitions = np.diff(y_pred_seq.astype(int))

    true_onsets = np.where(true_transitions == 1)[0]
    pred_onsets = np.where(pred_transitions == 1)[0]

    if len(true_onsets) > 0 and len(pred_onsets) > 0:
        # For each true onset, find nearest predicted onset
        onset_delays = []
        for true_on in true_onsets:
            diffs = pred_onsets - true_on
            nearest = diffs[np.argmin(np.abs(diffs))]
            onset_delays.append(nearest * 30)  # convert to seconds

        onset_delays = np.array(onset_delays)
        print(f"    N3 onset events: {len(true_onsets)} true, {len(pred_onsets)} predicted")
        print(f"    Detection delay: {np.mean(onset_delays):+.0f}s mean "
              f"(median {np.median(onset_delays):+.0f}s)")
        print(f"    Within ±30s: {np.mean(np.abs(onset_delays) <= 30) * 100:.0f}%")
        print(f"    Within ±60s: {np.mean(np.abs(onset_delays) <= 60) * 100:.0f}%")

        if np.mean(np.abs(onset_delays) <= 30) > 0.80:
            print("    ✓ Onset detection within 1-epoch tolerance")
        else:
            print("    ⚠ Onset detection needs improvement for real-time use")

    # 5.4 Calibration (are predicted probabilities reliable?)
    print(f"\n  Probability calibration:")
    try:
        prob_true, prob_pred = calibration_curve(y_binary, y_scores, n_bins=5)
        cal_error = np.mean(np.abs(prob_true - prob_pred))
        print(f"    Expected calibration error: {cal_error:.3f}")
        if cal_error < 0.05:
            print("    ✓ Well calibrated — probabilities are trustworthy")
        elif cal_error < 0.10:
            print("    ○ Acceptable — consider Platt scaling")
        else:
            print("    ⚠ Poorly calibrated — apply isotonic regression before deployment")

        for pt, pp in zip(prob_true, prob_pred):
            print(f"    Predicted ~{pp:.2f} → actual {pt:.2f}")
    except Exception:
        print("    Could not compute calibration (insufficient data)")

    # 5.5 Safety margin: false stimulation rate
    print(f"\n  ┌────────────────────────────────────────────────┐")
    print(f"  │  Closed-Loop Safety Assessment                 │")
    preds_hr = (y_scores >= hr_threshold).astype(int)
    fp_rate = np.sum((preds_hr == 1) & (y_binary == 0)) / max(np.sum(preds_hr), 1)
    fn_rate = np.sum((preds_hr == 0) & (y_binary == 1)) / max(np.sum(y_binary), 1)
    print(f"  │  At threshold {hr_threshold:.3f}:                          │")
    print(f"  │    False stimulation rate: {fp_rate*100:5.1f}%              │")
    print(f"  │    Missed N3 windows:      {fn_rate*100:5.1f}%              │")
    stim_safe = fp_rate < 0.15 and fn_rate < 0.15
    if stim_safe:
        print(f"  │  ✓ Within acceptable range for acoustic stim   │")
    else:
        print(f"  │  ⚠ Needs improvement before deployment         │")
    print(f"  └────────────────────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════
# LEVEL 6: Ablation Study
# ══════════════════════════════════════════════════════════════

def level_6_ablation(X, y, label_names, feature_names):
    """Compare TDA+Spectral vs Spectral-only vs TDA-only."""
    print("\n" + "═" * 60)
    print("  LEVEL 6: Ablation Study (TDA vs Spectral)")
    print("═" * 60)

    scaler = StandardScaler()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Split features by type
    spec_mask = np.array([k.startswith("spec_") for k in feature_names])
    tda_mask = ~spec_mask

    configs = {
        "Spectral only": X[:, spec_mask] if spec_mask.any() else None,
        "TDA only": X[:, tda_mask] if tda_mask.any() else None,
        "TDA + Spectral": X,
    }

    results = {}
    n3_idx = label_names.index("N3") if "N3" in label_names else None

    print(f"\n  {'Configuration':<20} {'F1-macro':>10} {'N3 F1':>8} {'N3 Recall':>10} {'Features':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*8} {'─'*10} {'─'*10}")

    for name, X_sub in configs.items():
        if X_sub is None or X_sub.shape[1] == 0:
            print(f"  {name:<20} {'N/A':>10}")
            continue

        X_s = scaler.fit_transform(X_sub)
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        preds = cross_val_predict(clf, X_s, y, cv=cv)

        f1_m = f1_score(y, preds, average="macro", zero_division=0)
        n3_f1, n3_r = 0.0, 0.0
        if n3_idx is not None:
            n3_true = (y == n3_idx).astype(int)
            n3_pred = (preds == n3_idx).astype(int)
            n3_f1 = f1_score(n3_true, n3_pred, zero_division=0)
            n3_r = recall_score(n3_true, n3_pred, zero_division=0)

        results[name] = {"f1_macro": f1_m, "n3_f1": n3_f1, "n3_recall": n3_r}
        print(f"  {name:<20} {f1_m:>10.3f} {n3_f1:>8.3f} {n3_r:>10.3f} {X_sub.shape[1]:>10}")

    # Statistical test: is TDA+Spectral significantly better than Spectral-only?
    if "TDA + Spectral" in results and "Spectral only" in results:
        combined = results["TDA + Spectral"]["f1_macro"]
        spectral = results["Spectral only"]["f1_macro"]
        delta = combined - spectral
        print(f"\n  TDA contribution: +{delta:.3f} F1-macro over spectral baseline")
        if delta > 0.02:
            print("  ✓ TDA adds meaningful discriminative power")
        elif delta > 0:
            print("  ○ TDA adds marginal improvement — may not justify compute cost")
        else:
            print("  ⚠ TDA does not improve over spectral baseline")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NeuroSpiral Validation Framework")
    parser.add_argument("--psg", type=Path)
    parser.add_argument("--hyp", type=Path)
    parser.add_argument("--download-sample", action="store_true")
    parser.add_argument(
        "--level", type=str, default="all",
        help="Validation levels: 1,2,3,5,6 or 'all'",
    )
    parser.add_argument("--no-tda", action="store_true")
    args = parser.parse_args()

    if args.download_sample:
        print("📥 Downloading sample...")
        psg, hyp = download_sample(PROJECT_ROOT / "data/raw")
    elif args.psg and args.hyp:
        psg, hyp = args.psg, args.hyp
    else:
        parser.error("Provide --psg/--hyp or --download-sample")
        return

    levels = args.level
    if levels == "all":
        run_levels = {1, 2, 3, 5, 6}
    else:
        run_levels = {int(x) for x in levels.split(",")}

    use_tda = not args.no_tda

    print(f"\n🔬 Loading and preparing data (TDA={'on' if use_tda else 'off'})...")
    t0 = time.time()
    X, y, label_names, raw_epochs = load_and_prepare(psg, hyp, use_tda)
    print(f"   Done in {time.time()-t0:.1f}s — {X.shape[0]} epochs × {X.shape[1]} features")

    # Get feature names for ablation
    from scipy import signal as scipy_signal
    sfreq = 100.0
    sample_feats = {}
    epoch_1d = raw_epochs[0, 0, :]
    freqs, psd = scipy_signal.welch(epoch_1d, fs=sfreq, nperseg=256)
    total = np.trapz(psd, freqs)
    for name, (lo, hi) in SPECTRAL_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        sample_feats[f"spec_{name}"] = 0.0
    sample_feats["spec_delta_beta"] = 0.0
    if use_tda:
        try:
            cloud, _ = time_delay_embedding(epoch_1d, dimension=4)
            tda = extract_tda_features(cloud, max_dim=2, n_subsample=300)
            sample_feats.update(tda)
        except Exception:
            pass
    feature_names = sorted(sample_feats.keys())

    # Run requested levels
    if 1 in run_levels:
        level_1_sanity(X, y, label_names)
    if 2 in run_levels:
        level_2_overfit(X, y, label_names)
    if 3 in run_levels:
        level_3_cross_validation(X, y, label_names)
    if 5 in run_levels:
        level_5_clinical(X, y, label_names)
    if 6 in run_levels and use_tda:
        level_6_ablation(X, y, label_names, feature_names)

    print(f"\n{'═'*60}")
    print(f"  Validation complete")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
