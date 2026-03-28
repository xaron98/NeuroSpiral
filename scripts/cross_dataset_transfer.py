#!/usr/bin/env python3
"""Cross-dataset transfer: test whether geometric stage boundaries generalize
between Sleep-EDF and HMC without retraining.

Analyses
--------
1. Train Sleep-EDF → test HMC (raw)
2. Train HMC → test Sleep-EDF (raw)
3. Feature distribution shift (Cohen's d)
4. Normalized transfer (z-score per dataset)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from scipy.signal import welch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.takens import time_delay_embedding
from src.features.torus_features_v2 import extract_torus_features_v2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SFREQ = 100
EPOCH_SEC = 30
EPOCH_SAMPLES = SFREQ * EPOCH_SEC

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGES)}

SLEEP_EDF_STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": None,
    "Movement time": None,
}

HMC_STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage N1": "N1",
    "Sleep stage N2": "N2",
    "Sleep stage N3": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": None,
    "0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "REM", "5": None,
    "W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "REM", "R": "REM",
}

SLEEP_EDF_SUBJECTS = [
    "SC4001", "SC4002", "SC4011", "SC4012",
    "SC4021", "SC4022", "SC4031", "SC4041",
    "SC4042", "SC4051", "SC4052", "SC4061",
    "SC4062", "SC4071", "SC4072", "SC4081",
    "SC4091", "SC4092",
]

FEATURE_NAMES = [
    "omega1", "torus_curvature", "angular_acceleration",
    "geodesic_distance", "angular_entropy", "phase_diff_std",
    "phase_coherence", "transition_rate",
]


# ===================================================================
# Loading — Sleep-EDF
# ===================================================================
def load_sleep_edf(data_dir: Path):
    """Return (X, y) across all Sleep-EDF subjects."""
    all_X, all_y = [], []
    n = 0
    for subj in SLEEP_EDF_SUBJECTS:
        psg = data_dir / f"{subj}E0-PSG.edf"
        hyp = data_dir / f"{subj}E0-Hypnogram.edf"
        if not psg.exists() or not hyp.exists():
            continue

        raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False)
        if "EEG Fpz-Cz" not in raw.ch_names:
            continue
        raw.pick(["EEG Fpz-Cz"])
        raw.filter(0.5, 30.0, verbose=False)
        if raw.info["sfreq"] != SFREQ:
            raw.resample(SFREQ, verbose=False)

        signal = raw.get_data()[0]
        n_ep = int(len(signal) / SFREQ // EPOCH_SEC)

        annots = mne.read_annotations(str(hyp))
        labels: list[str | None] = [None] * n_ep
        for onset, dur, desc in zip(annots.onset, annots.duration, annots.description):
            stage = SLEEP_EDF_STAGE_MAP.get(str(desc).strip())
            if stage is None:
                continue
            s = int(onset // EPOCH_SEC)
            for e in range(s, min(s + int(dur // EPOCH_SEC), n_ep)):
                labels[e] = stage

        for i in range(n_ep):
            if labels[i] is None:
                continue
            start = i * EPOCH_SAMPLES
            end = start + EPOCH_SAMPLES
            if end > len(signal):
                break
            epoch = signal[start:end]
            if np.max(np.abs(epoch)) > 500e-6:
                continue
            feats = _extract_features(epoch)
            if feats is None:
                continue
            all_X.append(feats)
            all_y.append(STAGE_TO_INT[labels[i]])

        n += 1
        if n % 5 == 0:
            print(f"    {n} subjects ...")

    return np.array(all_X), np.array(all_y), n


# ===================================================================
# Loading — HMC
# ===================================================================
def load_hmc(data_dir: Path):
    """Return (X, y) across all HMC subjects."""
    all_X, all_y = [], []
    n = 0
    for i in range(1, 200):
        sid = f"SN{i:03d}"
        psg = data_dir / f"{sid}.edf"
        hyp = data_dir / f"{sid}_sleepscoring.edf"
        if not psg.exists() or not hyp.exists():
            continue
        if psg.stat().st_size < 1_000_000:
            continue

        raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False)
        ch = None
        for c in ["EEG C4-M1", "C4-M1", "EEG C4"]:
            if c in raw.ch_names:
                ch = c
                break
        if ch is None:
            eeg = [c for c in raw.ch_names if "EEG" in c.upper()]
            if eeg:
                ch = eeg[0]
            else:
                continue

        raw.pick([ch])
        if raw.info["sfreq"] != SFREQ:
            raw.resample(SFREQ, verbose=False)
        raw.filter(0.5, 30.0, verbose=False)

        signal = raw.get_data()[0]
        n_ep = int(len(signal) / SFREQ // EPOCH_SEC)

        annots = mne.read_annotations(str(hyp))
        labels: list[str | None] = [None] * n_ep
        for onset, dur, desc in zip(annots.onset, annots.duration, annots.description):
            stage = HMC_STAGE_MAP.get(str(desc).strip())
            if stage is None:
                continue
            s = int(onset // EPOCH_SEC)
            nd = max(1, int(dur // EPOCH_SEC))
            for e in range(s, min(s + nd, n_ep)):
                labels[e] = stage

        for j in range(n_ep):
            if labels[j] is None:
                continue
            start = j * EPOCH_SAMPLES
            end = start + EPOCH_SAMPLES
            if end > len(signal):
                break
            epoch = signal[start:end]
            if np.max(np.abs(epoch)) > 500e-6:
                continue
            feats = _extract_features(epoch)
            if feats is None:
                continue
            all_X.append(feats)
            all_y.append(STAGE_TO_INT[labels[j]])

        n += 1
        if n % 20 == 0:
            print(f"    {n} subjects ...")

    return np.array(all_X), np.array(all_y), n


# ===================================================================
# Feature extraction
# ===================================================================
def _wrap(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


def _extract_features(epoch: np.ndarray) -> np.ndarray | None:
    """Return feature vector (8,) or None."""
    try:
        emb, _ = time_delay_embedding(epoch, dimension=4, tau=25)
    except ValueError:
        return None
    if emb.shape[0] < 10 or not np.all(np.isfinite(emb)):
        return None

    theta = np.arctan2(emb[:, 1], emb[:, 0])
    phi = np.arctan2(emb[:, 3], emb[:, 2])
    theta_uw = np.unwrap(theta)
    phi_uw = np.unwrap(phi)
    dtheta = np.diff(theta_uw)

    omega1 = float(np.mean(np.abs(dtheta)))

    # Single-channel phase_diff_std: std of diff(theta_uw - phi_uw)
    phase_diff = theta_uw - phi_uw
    phase_diff_std = float(np.std(np.diff(phase_diff)))

    # Single-channel phase_coherence: mean resultant length of (theta - phi)
    diff_wrapped = _wrap(theta - phi)
    mc = float(np.mean(np.cos(diff_wrapped)))
    ms = float(np.mean(np.sin(diff_wrapped)))
    phase_coherence = float(np.sqrt(mc**2 + ms**2))

    # Transition rate
    signs = (emb >= 0).astype(int)
    verts = signs[:, 0] * 8 + signs[:, 1] * 4 + signs[:, 2] * 2 + signs[:, 3]
    transition_rate = float(np.sum(np.diff(verts) != 0) / max(len(verts) - 1, 1))

    v2 = extract_torus_features_v2(emb)

    vec = np.array([
        omega1,
        v2["torus_curvature"],
        v2["angular_acceleration"],
        v2["geodesic_distance"],
        v2["angular_entropy"],
        phase_diff_std,
        phase_coherence,
        transition_rate,
    ])

    if not np.all(np.isfinite(vec)):
        return None
    return vec


# ===================================================================
# Train & score
# ===================================================================
def train_and_score(X_train, y_train, X_test, y_test, label: str):
    """Train DT + GB, report metrics. Returns dict."""
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    results = {}
    for name, clf in [
        ("DecisionTree", DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)),
    ]:
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        acc = accuracy_score(y_test, pred)
        kappa = cohen_kappa_score(y_test, pred)
        f1m = f1_score(y_test, pred, average="macro")
        f1_per = f1_score(y_test, pred, average=None, labels=list(range(5)))
        cm = confusion_matrix(y_test, pred, labels=list(range(5)))

        results[name] = {
            "accuracy": round(float(acc), 4),
            "kappa": round(float(kappa), 4),
            "f1_macro": round(float(f1m), 4),
            "f1_per_stage": {s: round(float(f), 4) for s, f in zip(STAGES, f1_per)},
            "confusion_matrix": cm.tolist(),
        }
        print(f"    {name:<20s}  Acc={acc:.3f}  k={kappa:.3f}  F1m={f1m:.3f}"
              f"  [{', '.join(f'{s}={f:.2f}' for s, f in zip(STAGES, f1_per))}]")

    return results


# ===================================================================
# Analysis 3 — Distribution shift
# ===================================================================
def compute_cohens_d(X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
    mean_a = np.mean(X_a, axis=0)
    mean_b = np.mean(X_b, axis=0)
    std_a = np.std(X_a, axis=0)
    std_b = np.std(X_b, axis=0)
    pooled = np.sqrt((std_a**2 + std_b**2) / 2.0)
    return np.abs(mean_a - mean_b) / np.maximum(pooled, 1e-15)


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    edf_dir = PROJECT_ROOT / "data" / "sleep-edf"
    hmc_dir = PROJECT_ROOT / "data" / "hmc"
    results_dir = PROJECT_ROOT / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Cross-Dataset Transfer Analysis")
    print("=" * 65)

    # --- Load Sleep-EDF ---
    print("\n  Loading Sleep-EDF (Fpz-Cz) ...")
    X_edf, y_edf, n_edf = load_sleep_edf(edf_dir)
    print(f"  Sleep-EDF: {n_edf} subjects, {len(y_edf)} epochs")

    # --- Load HMC ---
    print("\n  Loading HMC (C4-M1) ...")
    X_hmc, y_hmc, n_hmc = load_hmc(hmc_dir)
    print(f"  HMC: {n_hmc} subjects, {len(y_hmc)} epochs\n")

    # ==================================================================
    # Analysis 1 — Train Sleep-EDF, test HMC (raw)
    # ==================================================================
    print("=" * 65)
    print("Analysis 1: Train Sleep-EDF -> Test HMC (raw)")
    print("=" * 65)
    res_edf_hmc_raw = train_and_score(X_edf, y_edf, X_hmc, y_hmc,
                                       "EDF->HMC raw")

    # ==================================================================
    # Analysis 2 — Train HMC, test Sleep-EDF (raw)
    # ==================================================================
    print("\n" + "=" * 65)
    print("Analysis 2: Train HMC -> Test Sleep-EDF (raw)")
    print("=" * 65)
    res_hmc_edf_raw = train_and_score(X_hmc, y_hmc, X_edf, y_edf,
                                       "HMC->EDF raw")

    # ==================================================================
    # Analysis 3 — Feature distribution shift
    # ==================================================================
    print("\n" + "=" * 65)
    print("Analysis 3: Feature distribution shift (Cohen's d)")
    print("=" * 65)

    d_vals = compute_cohens_d(X_edf, X_hmc)
    shift_results = {}

    print(f"\n  {'Feature':<25s} {'EDF mean':>10s} {'HMC mean':>10s} {'Cohen d':>10s} {'Class':>12s}")
    print("  " + "-" * 70)
    for i, fn in enumerate(FEATURE_NAMES):
        d = float(d_vals[i])
        cls = "invariant" if d < 0.3 else "moderate" if d < 0.8 else "channel-dep"
        print(f"  {fn:<25s} {np.mean(X_edf[:, i]):>10.4f} {np.mean(X_hmc[:, i]):>10.4f}"
              f" {d:>10.3f} {cls:>12s}")
        shift_results[fn] = {
            "edf_mean": round(float(np.mean(X_edf[:, i])), 6),
            "edf_std": round(float(np.std(X_edf[:, i])), 6),
            "hmc_mean": round(float(np.mean(X_hmc[:, i])), 6),
            "hmc_std": round(float(np.std(X_hmc[:, i])), 6),
            "cohens_d": round(d, 4),
            "classification": cls,
        }

    most_invariant = FEATURE_NAMES[int(np.argmin(d_vals))]
    most_shifted = FEATURE_NAMES[int(np.argmax(d_vals))]
    print(f"\n  Most invariant:  {most_invariant} (d={d_vals[np.argmin(d_vals)]:.3f})")
    print(f"  Most shifted:    {most_shifted} (d={d_vals[np.argmax(d_vals)]:.3f})")

    # ==================================================================
    # Analysis 4 — Normalized transfer
    # ==================================================================
    print("\n" + "=" * 65)
    print("Analysis 4: Normalized transfer (z-score per dataset)")
    print("=" * 65)

    edf_mean = np.mean(X_edf, axis=0)
    edf_std = np.std(X_edf, axis=0) + 1e-10
    X_edf_z = (X_edf - edf_mean) / edf_std

    hmc_mean = np.mean(X_hmc, axis=0)
    hmc_std = np.std(X_hmc, axis=0) + 1e-10
    X_hmc_z = (X_hmc - hmc_mean) / hmc_std

    print("\n  Train Sleep-EDF (z) -> Test HMC (z):")
    res_edf_hmc_norm = train_and_score(X_edf_z, y_edf, X_hmc_z, y_hmc,
                                        "EDF->HMC norm")

    print("\n  Train HMC (z) -> Test Sleep-EDF (z):")
    res_hmc_edf_norm = train_and_score(X_hmc_z, y_hmc, X_edf_z, y_edf,
                                        "HMC->EDF norm")

    # ==================================================================
    # Figures
    # ==================================================================
    print("\n  Generating figures ...")

    # --- Figure 1: bar chart comparing kappa and F1 ---
    conditions = [
        "EDF->HMC\nraw",
        "HMC->EDF\nraw",
        "EDF->HMC\nnorm",
        "HMC->EDF\nnorm",
    ]
    gb_kappas = [
        res_edf_hmc_raw["GradientBoosting"]["kappa"],
        res_hmc_edf_raw["GradientBoosting"]["kappa"],
        res_edf_hmc_norm["GradientBoosting"]["kappa"],
        res_hmc_edf_norm["GradientBoosting"]["kappa"],
    ]
    gb_f1s = [
        res_edf_hmc_raw["GradientBoosting"]["f1_macro"],
        res_hmc_edf_raw["GradientBoosting"]["f1_macro"],
        res_edf_hmc_norm["GradientBoosting"]["f1_macro"],
        res_hmc_edf_norm["GradientBoosting"]["f1_macro"],
    ]
    dt_kappas = [
        res_edf_hmc_raw["DecisionTree"]["kappa"],
        res_hmc_edf_raw["DecisionTree"]["kappa"],
        res_edf_hmc_norm["DecisionTree"]["kappa"],
        res_hmc_edf_norm["DecisionTree"]["kappa"],
    ]
    dt_f1s = [
        res_edf_hmc_raw["DecisionTree"]["f1_macro"],
        res_hmc_edf_raw["DecisionTree"]["f1_macro"],
        res_edf_hmc_norm["DecisionTree"]["f1_macro"],
        res_hmc_edf_norm["DecisionTree"]["f1_macro"],
    ]

    x = np.arange(len(conditions))
    w = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(x - 1.5 * w, gb_kappas, w, label="GB kappa", color="#5bffa8")
    ax.bar(x - 0.5 * w, dt_kappas, w, label="DT kappa", color="#5b8bd4")
    ax.bar(x + 0.5 * w, gb_f1s, w, label="GB F1m", color="#f5c842")
    ax.bar(x + 1.5 * w, dt_f1s, w, label="DT F1m", color="#6e3fa0")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Cross-dataset transfer: kappa and macro-F1", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(gb_kappas + [0.01]), max(gb_f1s + [0.01])) * 1.25)

    for bars_x, bars_y in [
        (x - 1.5 * w, gb_kappas),
        (x - 0.5 * w, dt_kappas),
        (x + 0.5 * w, gb_f1s),
        (x + 1.5 * w, dt_f1s),
    ]:
        for xi, vi in zip(bars_x, bars_y):
            ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", va="bottom",
                    fontsize=7)

    # Right: improvement from normalization
    ax = axes[1]
    delta_k_gb = [gb_kappas[2] - gb_kappas[0], gb_kappas[3] - gb_kappas[1]]
    delta_f_gb = [gb_f1s[2] - gb_f1s[0], gb_f1s[3] - gb_f1s[1]]
    labels_delta = ["EDF->HMC", "HMC->EDF"]
    x2 = np.arange(2)
    ax.bar(x2 - 0.15, delta_k_gb, 0.3, label="Delta kappa (GB)", color="#5bffa8")
    ax.bar(x2 + 0.15, delta_f_gb, 0.3, label="Delta F1m (GB)", color="#f5c842")
    ax.set_xticks(x2)
    ax.set_xticklabels(labels_delta, fontsize=11)
    ax.set_ylabel("Improvement from normalization", fontsize=11)
    ax.set_title("Effect of z-score normalization", fontsize=13)
    ax.axhline(0, color="gray", ls="-", lw=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "cross_dataset_transfer.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: Cohen's d bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bar = ["#5bffa8" if d < 0.3 else "#f5c842" if d < 0.8 else "#ff5555"
                  for d in d_vals]
    ax.barh(range(len(FEATURE_NAMES)), d_vals, color=colors_bar,
            edgecolor="white")
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=10)
    ax.set_xlabel("Cohen's d (effect size of dataset shift)", fontsize=11)
    ax.set_title("Feature distribution shift: Sleep-EDF vs HMC", fontsize=13)
    ax.axvline(0.3, ls="--", color="gray", alpha=0.5, label="d=0.3 (small)")
    ax.axvline(0.8, ls="--", color="red", alpha=0.5, label="d=0.8 (large)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    for i, d in enumerate(d_vals):
        ax.text(d + 0.02, i, f"{d:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_shift.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: Confusion matrices (2x2 grid) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = [
        "EDF->HMC raw (GB)",
        "HMC->EDF raw (GB)",
        "EDF->HMC norm (GB)",
        "HMC->EDF norm (GB)",
    ]
    cms = [
        np.array(res_edf_hmc_raw["GradientBoosting"]["confusion_matrix"]),
        np.array(res_hmc_edf_raw["GradientBoosting"]["confusion_matrix"]),
        np.array(res_edf_hmc_norm["GradientBoosting"]["confusion_matrix"]),
        np.array(res_hmc_edf_norm["GradientBoosting"]["confusion_matrix"]),
    ]

    for ax, cm, title in zip(axes.flat, cms, titles):
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_pct = cm / row_sums * 100

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(STAGES, fontsize=9)
        ax.set_yticklabels(STAGES, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(title, fontsize=11)
        for ii in range(5):
            for jj in range(5):
                v = cm_pct[ii, jj]
                c = "white" if v > 50 else "black"
                ax.text(jj, ii, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color=c)

    fig.tight_layout()
    fig.savefig(fig_dir / "transfer_confusion_matrices.png", dpi=150)
    plt.close(fig)

    print("  Saved: cross_dataset_transfer.png")
    print("         feature_shift.png")
    print("         transfer_confusion_matrices.png")

    # ==================================================================
    # Save JSON
    # ==================================================================
    output = {
        "sleep_edf": {"n_subjects": n_edf, "n_epochs": len(y_edf)},
        "hmc": {"n_subjects": n_hmc, "n_epochs": len(y_hmc)},
        "analysis_1_edf_to_hmc_raw": res_edf_hmc_raw,
        "analysis_2_hmc_to_edf_raw": res_hmc_edf_raw,
        "analysis_3_feature_shift": shift_results,
        "analysis_4_edf_to_hmc_norm": res_edf_hmc_norm,
        "analysis_4_hmc_to_edf_norm": res_hmc_edf_norm,
    }

    out_path = results_dir / "cross_dataset_transfer.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 65)
    print("SUMMARY (GradientBoosting)")
    print("=" * 65)

    gb_raw_1 = res_edf_hmc_raw["GradientBoosting"]
    gb_raw_2 = res_hmc_edf_raw["GradientBoosting"]
    gb_nrm_1 = res_edf_hmc_norm["GradientBoosting"]
    gb_nrm_2 = res_hmc_edf_norm["GradientBoosting"]

    print(f"\n  {'Direction':<25s} {'k raw':>8s} {'F1m raw':>8s}"
          f" {'k norm':>8s} {'F1m norm':>8s} {'Dk':>8s}")
    print("  " + "-" * 65)
    print(f"  {'EDF -> HMC':<25s}"
          f" {gb_raw_1['kappa']:>8.3f} {gb_raw_1['f1_macro']:>8.3f}"
          f" {gb_nrm_1['kappa']:>8.3f} {gb_nrm_1['f1_macro']:>8.3f}"
          f" {gb_nrm_1['kappa'] - gb_raw_1['kappa']:>+8.3f}")
    print(f"  {'HMC -> EDF':<25s}"
          f" {gb_raw_2['kappa']:>8.3f} {gb_raw_2['f1_macro']:>8.3f}"
          f" {gb_nrm_2['kappa']:>8.3f} {gb_nrm_2['f1_macro']:>8.3f}"
          f" {gb_nrm_2['kappa'] - gb_raw_2['kappa']:>+8.3f}")

    print(f"\n  Channel-invariant features (d < 0.3):")
    for fn in FEATURE_NAMES:
        if shift_results[fn]["cohens_d"] < 0.3:
            print(f"    {fn} (d={shift_results[fn]['cohens_d']:.3f})")
    print(f"  Channel-dependent features (d > 0.8):")
    for fn in FEATURE_NAMES:
        if shift_results[fn]["cohens_d"] > 0.8:
            print(f"    {fn} (d={shift_results[fn]['cohens_d']:.3f})")

    k_raw_avg = (gb_raw_1["kappa"] + gb_raw_2["kappa"]) / 2
    k_norm_avg = (gb_nrm_1["kappa"] + gb_nrm_2["kappa"]) / 2
    print(f"\n  Mean kappa (raw):  {k_raw_avg:.3f}")
    print(f"  Mean kappa (norm): {k_norm_avg:.3f}")
    if k_norm_avg > k_raw_avg + 0.05:
        print("  -> Normalization substantially helps: geometric boundaries")
        print("     are preserved but absolute scale shifts with montage")
    elif k_norm_avg > k_raw_avg:
        print("  -> Normalization helps modestly")
    else:
        print("  -> Normalization does not help: raw features already transfer")

    print(f"\n  Results -> {out_path}")


if __name__ == "__main__":
    main()
