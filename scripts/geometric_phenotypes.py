#!/usr/bin/env python3
"""Geometric phenotype analysis of Clifford torus features across sleep stages.

Foundation analysis for Paper #2: demonstrates that geometric features are
geometric reformulations of known sleep physiology, not just statistical correlates.

Analyses
--------
1. Sleep stage phenotyping — violin plots + summary statistics
2. Overnight geometric trajectories — 3 representative subjects
3. Process S validation — ω₁ decay across NREM cycles
4. Rule-based classification — explicit thresholds via shallow decision tree (LOO)
5. Correlation heatmap — geometric features vs spectral bands
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from scipy.optimize import curve_fit
from scipy.signal import welch
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.takens import time_delay_embedding
from src.features.torus_features_v2 import extract_torus_features_v2
from src.features.spectral import compute_band_powers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SFREQ = 100
EPOCH_SEC = 30
EPOCH_SAMPLES = SFREQ * EPOCH_SEC

SUBJECTS = [
    "SC4001", "SC4002", "SC4011", "SC4012",
    "SC4021", "SC4022", "SC4031", "SC4041",
    "SC4042", "SC4051", "SC4052", "SC4061",
    "SC4062", "SC4071", "SC4072", "SC4081",
    "SC4091", "SC4092",
]

HYPNOGRAM_NAMES = {
    "SC4001": "SC4001EC", "SC4002": "SC4002EC",
    "SC4011": "SC4011EH", "SC4012": "SC4012EC",
    "SC4021": "SC4021EH", "SC4022": "SC4022EJ",
    "SC4031": "SC4031EC", "SC4041": "SC4041EC",
    "SC4042": "SC4042EC", "SC4051": "SC4051EC",
    "SC4052": "SC4052EC", "SC4061": "SC4061EC",
    "SC4062": "SC4062EC", "SC4071": "SC4071EC",
    "SC4072": "SC4072EH", "SC4081": "SC4081EC",
    "SC4091": "SC4091EC", "SC4092": "SC4092EC",
}

STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": None,
    "Movement time": None,
}

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGES)}
STAGE_COLORS = {
    "W": "#f5c842", "N1": "#90caf9", "N2": "#5b8bd4",
    "N3": "#1a1a6e", "REM": "#6e3fa0",
}

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"

# Paper #1 original 5 geometric features
ORIG_FEATURES = [
    "omega1", "winding_ratio", "phase_diff_std",
    "phase_coherence", "transition_rate",
]

# Paper #2 new torus v2 features (9 scalars from 6 families)
V2_FEATURES = [
    "angular_acceleration", "geodesic_distance", "angular_entropy",
    "theta_harmonic_1", "theta_harmonic_2", "theta_harmonic_3",
    "theta_harmonic_4", "torus_curvature", "angular_range",
]

ALL_GEOM_FEATURES = ORIG_FEATURES + V2_FEATURES
SPECTRAL_BANDS = ["delta", "theta", "alpha", "sigma", "beta"]


# ===================================================================
# Data download
# ===================================================================
def download_sleep_edf(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("SKIP_DOWNLOAD"):
        n_ok = sum(
            1 for s in SUBJECTS
            if (data_dir / f"{s}E0-PSG.edf").exists()
            and (data_dir / f"{s}E0-Hypnogram.edf").exists()
        )
        print(f"  SKIP_DOWNLOAD set. {n_ok}/{len(SUBJECTS)} subjects available.\n")
        return
    for subj in SUBJECTS:
        psg_remote = f"{subj}E0-PSG.edf"
        psg_local = data_dir / psg_remote
        hyp_base = HYPNOGRAM_NAMES[subj]
        hyp_remote = f"{hyp_base}-Hypnogram.edf"
        hyp_local = data_dir / f"{subj}E0-Hypnogram.edf"
        for remote_name, local_path in [
            (psg_remote, psg_local), (hyp_remote, hyp_local),
        ]:
            if local_path.exists() and local_path.stat().st_size > 0:
                continue
            url = f"{BASE_URL}/{remote_name}"
            print(f"  Downloading {remote_name} -> {local_path.name}")
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                print(f"  ERROR: {e}")
                if local_path.exists():
                    local_path.unlink()
            if local_path.exists() and local_path.stat().st_size == 0:
                local_path.unlink()
    n_ok = sum(
        1 for s in SUBJECTS
        if (data_dir / f"{s}E0-PSG.edf").exists()
        and (data_dir / f"{s}E0-Hypnogram.edf").exists()
    )
    print(f"  {n_ok}/{len(SUBJECTS)} subjects ready.\n")


# ===================================================================
# Data loading
# ===================================================================
def load_subject(data_dir: Path, subj: str):
    """Return (epochs_array, labels_list) or None."""
    psg_path = data_dir / f"{subj}E0-PSG.edf"
    hyp_path = data_dir / f"{subj}E0-Hypnogram.edf"
    if not psg_path.exists() or not hyp_path.exists():
        return None

    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    if "EEG Fpz-Cz" not in raw.ch_names:
        return None
    raw.pick(["EEG Fpz-Cz"])
    raw.filter(0.5, 30.0, verbose=False)
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)

    signal = raw.get_data()[0]
    n_epochs = int(len(signal) / SFREQ // EPOCH_SEC)

    annots = mne.read_annotations(str(hyp_path))
    epoch_labels: list[str | None] = [None] * n_epochs
    for onset, duration, desc in zip(
        annots.onset, annots.duration, annots.description
    ):
        stage = STAGE_MAP.get(str(desc).strip())  # CRITICAL: str() for np.str_
        if stage is None:
            continue
        s_ep = int(onset // EPOCH_SEC)
        for e in range(s_ep, min(s_ep + int(duration // EPOCH_SEC), n_epochs)):
            epoch_labels[e] = stage

    epochs_out, labels_out = [], []
    for i in range(n_epochs):
        if epoch_labels[i] is None:
            continue
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        if end > len(signal):
            break
        epoch = signal[start:end]
        if np.max(np.abs(epoch)) > 500e-6:
            continue
        epochs_out.append(epoch)
        labels_out.append(epoch_labels[i])

    if not epochs_out:
        return None
    return np.array(epochs_out), labels_out


# ===================================================================
# Feature extraction
# ===================================================================
def compute_original_5(embedding: np.ndarray) -> dict[str, float]:
    """Compute Paper #1 original 5 geometric features."""
    theta = np.arctan2(embedding[:, 1], embedding[:, 0])
    phi = np.arctan2(embedding[:, 3], embedding[:, 2])

    theta_uw = np.unwrap(theta)
    phi_uw = np.unwrap(phi)
    dtheta = np.diff(theta_uw)
    dphi = np.diff(phi_uw)

    omega1 = float(np.mean(np.abs(dtheta)))
    omega2 = float(np.mean(np.abs(dphi)))
    winding_ratio = omega1 / (omega2 + 1e-10)

    if len(dtheta) > 1:
        cc = np.corrcoef(dtheta, dphi)[0, 1]
        phase_coherence = float(abs(cc)) if np.isfinite(cc) else 0.0
    else:
        phase_coherence = 0.0

    phase_diff = theta_uw - phi_uw
    phase_diff_std = float(np.std(np.diff(phase_diff)))

    # Transition rate via sign-based 4-bit vertex
    signs = (embedding >= 0).astype(int)
    vertices = signs[:, 0] * 8 + signs[:, 1] * 4 + signs[:, 2] * 2 + signs[:, 3]
    n = len(vertices)
    transition_rate = float(np.sum(np.diff(vertices) != 0) / max(n - 1, 1))

    return {
        "omega1": omega1,
        "winding_ratio": winding_ratio,
        "phase_diff_std": phase_diff_std,
        "phase_coherence": phase_coherence,
        "transition_rate": transition_rate,
    }


def extract_all_features(epoch: np.ndarray) -> dict[str, float] | None:
    """Extract original-5 + v2-9 geometric + spectral features for one epoch."""
    try:
        embedding, _ = time_delay_embedding(epoch, dimension=4, tau=25)
    except ValueError:
        return None

    feats: dict[str, float] = {}
    feats.update(compute_original_5(embedding))
    feats.update(extract_torus_features_v2(embedding))

    spectral_rel = compute_band_powers(epoch, SFREQ, relative=True)
    feats.update(spectral_rel)  # delta, theta, alpha, sigma, beta (relative)

    spectral_abs = compute_band_powers(epoch, SFREQ, relative=False)
    feats["delta_power_abs"] = spectral_abs.get("delta", 0.0)

    return feats


# ===================================================================
# Analysis 1 — Geometric phenotype of each sleep stage
# ===================================================================
def analysis_1(pooled_by_stage: dict, fig_dir: Path) -> dict:
    print("\n" + "=" * 70)
    print("Analysis 1: Geometric phenotype of each sleep stage")
    print("=" * 70)

    features_to_plot = ALL_GEOM_FEATURES + ["delta"]
    stats: dict = {}

    for feat in features_to_plot:
        stage_data = {}
        feat_stats = {}
        for s in STAGES:
            vals = np.array(pooled_by_stage[feat][s])
            stage_data[s] = vals
            if len(vals) > 0:
                feat_stats[s] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "p25": float(np.percentile(vals, 25)),
                    "p75": float(np.percentile(vals, 75)),
                    "n": len(vals),
                }
            else:
                feat_stats[s] = {
                    "mean": 0, "std": 0, "median": 0, "p25": 0, "p75": 0, "n": 0,
                }
        stats[feat] = feat_stats

        # --- violin plot ---
        fig, ax = plt.subplots(figsize=(7, 5))
        data_list = [stage_data[s] for s in STAGES]
        colors = [STAGE_COLORS[s] for s in STAGES]

        parts = ax.violinplot(
            data_list, positions=range(len(STAGES)),
            showmeans=True, showmedians=True,
        )
        for idx, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[idx])
            body.set_alpha(0.7)
            body.set_edgecolor("white")
        for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("white")
                parts[key].set_linewidth(1.0)

        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(STAGES, fontsize=12, fontweight="bold")
        ax.set_ylabel(feat, fontsize=12)
        ax.set_title(f"{feat} distribution by sleep stage", fontsize=13)
        ax.set_facecolor("#f5f5f5")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"phenotype_{feat}.png", dpi=150)
        plt.close(fig)

    # Summary table
    header = f"  {'Feature':<25s}"
    for s in STAGES:
        header += f" {s:>10s}"
    print(header)
    print("  " + "-" * (25 + 11 * len(STAGES)))
    for feat in ALL_GEOM_FEATURES:
        line = f"  {feat:<25s}"
        for s in STAGES:
            line += f" {stats[feat][s]['mean']:>10.4f}"
        print(line)

    print(f"\n  Saved {len(features_to_plot)} violin plots to {fig_dir}/")
    return stats


# ===================================================================
# Analysis 2 — Overnight geometric trajectories
# ===================================================================
def analysis_2(per_subject: dict, fig_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("Analysis 2: Overnight geometric trajectories")
    print("=" * 70)

    # Pick 3 subjects with the most valid epochs
    counts = {s: len(d["labels"]) for s, d in per_subject.items()}
    top3 = sorted(counts, key=counts.get, reverse=True)[:3]
    print(f"  Selected: {top3}  (epochs: {[counts[s] for s in top3]})")

    # Features to plot (skip phase_diff_std — single-channel only)
    feats_to_show = [
        ("omega1", "ω₁"),
        ("winding_ratio", "ω₁/ω₂"),
        ("geodesic_distance", "Geodesic distance"),
        ("angular_entropy", "Angular entropy"),
        ("delta", "δ-power (relative)"),
    ]

    for subj in top3:
        data = per_subject[subj]
        labels = data["labels"]
        n = len(labels)
        t_hrs = np.arange(n) * EPOCH_SEC / 3600.0

        n_panels = len(feats_to_show) + 1  # +1 for hypnogram
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(14, 2.4 * n_panels), sharex=True,
        )

        # Panel 0: Hypnogram
        ax = axes[0]
        stage_ints = np.array([STAGE_TO_INT[l] for l in labels])
        ax.step(t_hrs, stage_ints, color="white", linewidth=1.0, where="post")
        for i in range(n - 1):
            ax.axvspan(
                t_hrs[i], t_hrs[i + 1], alpha=0.35,
                color=STAGE_COLORS[labels[i]], linewidth=0,
            )
        ax.set_yticks(range(len(STAGES)))
        ax.set_yticklabels(STAGES, fontsize=9, color="white")
        ax.set_ylabel("Stage", color="white", fontsize=10)
        ax.set_title(
            f"Overnight trajectory — {subj}", fontsize=14, color="white",
        )
        ax.invert_yaxis()
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="gray")

        # Feature panels
        kernel = np.ones(5) / 5.0
        for j, (feat_key, feat_label) in enumerate(feats_to_show):
            ax = axes[j + 1]
            vals = np.array(data["features"][feat_key], dtype=float)
            smoothed = np.convolve(vals, kernel, mode="same") if len(vals) >= 5 else vals

            ax.plot(t_hrs, smoothed, color="#5bffa8", linewidth=0.8, alpha=0.9)
            for i in range(n - 1):
                ax.axvspan(
                    t_hrs[i], t_hrs[i + 1], alpha=0.08,
                    color=STAGE_COLORS[labels[i]], linewidth=0,
                )
            ax.set_ylabel(feat_label, color="white", fontsize=9)
            ax.set_facecolor("#0c0e14")
            ax.tick_params(colors="gray")
            for spine in ax.spines.values():
                spine.set_color("#333")

        axes[-1].set_xlabel("Time (hours)", color="white", fontsize=11)
        fig.tight_layout()
        fig.savefig(
            fig_dir / f"overnight_{subj}.png",
            dpi=150, facecolor="#0c0e14",
        )
        plt.close(fig)

    print(f"  Saved overnight trajectory figures for {top3}.")


# ===================================================================
# Analysis 3 — Process S validation
# ===================================================================
def _identify_nrem_cycles(labels: list[str]) -> list[list[int]]:
    """Return list of NREM cycles, each a list of epoch indices.

    A cycle = contiguous N1/N2/N3 epochs (allowing ≤5 wake epochs)
    terminated by REM or prolonged wake.  Minimum 10 epochs (5 min).
    """
    nrem = {"N1", "N2", "N3"}
    cycles: list[list[int]] = []
    current: list[int] = []
    consecutive_wake = 0

    for i, lab in enumerate(labels):
        if lab in nrem:
            current.append(i)
            consecutive_wake = 0
        elif lab == "W":
            consecutive_wake += 1
            if consecutive_wake <= 5 and current:
                current.append(i)  # allow brief arousals
            else:
                if len(current) >= 10:
                    cycles.append(current)
                current = []
                consecutive_wake = 0
        elif lab == "REM":
            if len(current) >= 10:
                cycles.append(current)
            current = []
            consecutive_wake = 0

    if len(current) >= 10:
        cycles.append(current)
    return cycles


def analysis_3(per_subject: dict, fig_dir: Path) -> dict:
    print("\n" + "=" * 70)
    print("Analysis 3: Process S validation")
    print("=" * 70)

    all_cycle_omega1: dict[int, list[float]] = defaultdict(list)
    all_cycle_delta: dict[int, list[float]] = defaultdict(list)
    per_subj_cycles: dict[str, dict] = {}
    all_cycle_durations_hrs: list[float] = []

    for subj, data in per_subject.items():
        cycles = _identify_nrem_cycles(data["labels"])
        if len(cycles) < 2:
            continue

        omega1_arr = np.array(data["features"]["omega1"])
        delta_arr = np.array(data["features"]["delta_power_abs"])

        subj_o, subj_d = [], []
        for c_num, idxs in enumerate(cycles):
            mo = float(np.mean(omega1_arr[idxs]))
            md = float(np.mean(delta_arr[idxs]))
            subj_o.append(mo)
            subj_d.append(md)
            all_cycle_omega1[c_num].append(mo)
            all_cycle_delta[c_num].append(md)
            all_cycle_durations_hrs.append(len(idxs) * EPOCH_SEC / 3600.0)

        per_subj_cycles[subj] = {
            "omega1": subj_o, "delta": subj_d, "n_cycles": len(cycles),
        }

    if not all_cycle_omega1:
        print("  No subjects with ≥2 NREM cycles found.")
        return {}

    max_c = max(all_cycle_omega1.keys()) + 1
    cycle_x = np.arange(max_c, dtype=float)
    mean_o = np.array([np.mean(all_cycle_omega1[c]) for c in range(max_c)])
    mean_d = np.array([np.mean(all_cycle_delta[c]) for c in range(max_c)])

    # Pooled correlation
    pool_o = [v for s in per_subj_cycles.values() for v in s["omega1"]]
    pool_d = [v for s in per_subj_cycles.values() for v in s["delta"]]
    r_val, p_val = pearsonr(pool_o, pool_d) if len(pool_o) > 2 else (0.0, 1.0)
    print(f"  Pearson r(ω₁, δ) across NREM cycles: {r_val:.4f} (p={p_val:.2e})")

    mean_cycle_dur = float(np.mean(all_cycle_durations_hrs)) if all_cycle_durations_hrs else 1.5
    print(f"  Mean NREM cycle duration: {mean_cycle_dur:.2f} hours")

    result: dict = {
        "pearson_r": float(r_val),
        "pearson_p": float(p_val),
        "n_subjects": len(per_subj_cycles),
        "mean_cycle_duration_hours": mean_cycle_dur,
        "mean_omega1_per_cycle": mean_o.tolist(),
        "mean_delta_per_cycle": mean_d.tolist(),
    }

    # Exponential fit  f(x) = a * exp(-x / tau) + c
    def _exp_decay(x, a, tau, c):
        return a * np.exp(-np.asarray(x, dtype=float) / tau) + c

    fit_curves = {}  # for plotting
    for tag, y_data in [("omega1", mean_o), ("delta", mean_d)]:
        if len(y_data) < 3:
            continue
        try:
            p0 = [y_data[0] - y_data[-1], 2.0, float(y_data[-1])]
            popt, _ = curve_fit(_exp_decay, cycle_x[:len(y_data)], y_data,
                                p0=p0, maxfev=5000)
            tau_cyc = float(popt[1])
            tau_hrs = tau_cyc * mean_cycle_dur
            result[f"tau_{tag}_cycles"] = tau_cyc
            result[f"tau_{tag}_hours"] = tau_hrs
            fit_curves[tag] = popt
            label = "ω₁" if tag == "omega1" else "δ"
            print(f"  Exponential τ for {label}: {tau_cyc:.2f} cycles ≈ {tau_hrs:.1f} hours")
        except (RuntimeError, ValueError) as exc:
            print(f"  Exponential fit failed for {tag}: {exc}")

    print(f"  (Known Process S τ_fall ≈ 4.2 hours)")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: ω₁ vs cycle
    ax = axes[0]
    for info in per_subj_cycles.values():
        ax.plot(range(len(info["omega1"])), info["omega1"],
                "o-", alpha=0.25, ms=4, color="gray")
    ax.plot(cycle_x[:len(mean_o)], mean_o, "s-", color="#5bffa8",
            lw=2.5, ms=8, label="Group mean", zorder=5)
    if "omega1" in fit_curves:
        xf = np.linspace(0, max_c - 1, 200)
        ax.plot(xf, _exp_decay(xf, *fit_curves["omega1"]),
                "--", color="red", lw=1.5,
                label=f"τ = {result['tau_omega1_cycles']:.1f} cyc")
    ax.set_xlabel("NREM cycle #", fontsize=11)
    ax.set_ylabel("Mean ω₁", fontsize=11)
    ax.set_title("ω₁ decay across NREM cycles (Process S)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: scatter ω₁ vs δ
    ax = axes[1]
    ax.scatter(pool_d, pool_o, alpha=0.45, c="#5bffa8", edgecolors="#333", s=28)
    ax.set_xlabel("Mean δ-power (absolute)", fontsize=11)
    ax.set_ylabel("Mean ω₁", fontsize=11)
    ax.set_title(f"ω₁ vs δ-power  (r = {r_val:.3f})", fontsize=13)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "process_s_validation.png", dpi=150)
    plt.close(fig)
    print("  Saved Process S figure.")
    return result


# ===================================================================
# Analysis 4 — Explicit geometric thresholds (LOO)
# ===================================================================
def analysis_4(per_subject: dict, fig_dir: Path) -> dict:
    print("\n" + "=" * 70)
    print("Analysis 4: Rule-based thresholds (leave-one-out)")
    print("=" * 70)

    subjects_list = sorted(per_subject.keys())
    if len(subjects_list) < 3:
        print("  Too few subjects for LOO.")
        return {}

    # Build per-subject feature matrices
    subj_X: dict[str, np.ndarray] = {}
    subj_y: dict[str, np.ndarray] = {}
    for subj in subjects_list:
        data = per_subject[subj]
        n = len(data["labels"])
        cols = []
        for f in ALL_GEOM_FEATURES:
            arr = np.array(data["features"][f], dtype=float)
            cols.append(arr)
        X = np.column_stack(cols)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array([STAGE_TO_INT[l] for l in data["labels"]])
        subj_X[subj] = X
        subj_y[subj] = y

    dt_acc, dt_kap, dt_f1 = [], [], []
    gb_acc, gb_kap, gb_f1 = [], [], []
    rep_tree = None

    for i, test_s in enumerate(subjects_list):
        trn_X = np.vstack([subj_X[s] for s in subjects_list if s != test_s])
        trn_y = np.concatenate([subj_y[s] for s in subjects_list if s != test_s])
        tst_X = subj_X[test_s]
        tst_y = subj_y[test_s]

        # Decision tree — interpretable thresholds
        dt = DecisionTreeClassifier(max_depth=4, random_state=42,
                                    class_weight="balanced")
        dt.fit(trn_X, trn_y)
        dt_pred = dt.predict(tst_X)
        dt_acc.append(accuracy_score(tst_y, dt_pred))
        dt_kap.append(cohen_kappa_score(tst_y, dt_pred))
        dt_f1.append(f1_score(tst_y, dt_pred, average="weighted"))
        if i == 0:
            rep_tree = dt

        # GradientBoosting baseline
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, subsample=0.8, random_state=42,
        )
        gb.fit(trn_X, trn_y)
        gb_pred = gb.predict(tst_X)
        gb_acc.append(accuracy_score(tst_y, gb_pred))
        gb_kap.append(cohen_kappa_score(tst_y, gb_pred))
        gb_f1.append(f1_score(tst_y, gb_pred, average="weighted"))

    # Print tree rules
    if rep_tree is not None:
        print("\n  Decision-tree rules (fold 0):")
        rules = export_text(rep_tree, feature_names=ALL_GEOM_FEATURES, max_depth=4)
        for line in rules.split("\n")[:25]:
            print(f"    {line}")
        if rules.count("\n") > 25:
            print("    ... (truncated)")

    # Feature importances from GradientBoosting (last fold)
    print("\n  Top-5 GradientBoosting feature importances (last fold):")
    imp = gb.feature_importances_
    order = np.argsort(imp)[::-1]
    for rank in range(min(5, len(order))):
        idx = order[rank]
        print(f"    {ALL_GEOM_FEATURES[idx]:<25s}  {imp[idx]:.4f}")

    print(f"\n  {'Method':<25s} {'Accuracy':>10s} {'κ':>10s} {'F1(wt)':>10s}")
    print("  " + "-" * 57)
    print(f"  {'DecisionTree (d=4)':<25s}"
          f" {np.mean(dt_acc):>10.3f} {np.mean(dt_kap):>10.3f} {np.mean(dt_f1):>10.3f}")
    print(f"  {'GradientBoosting':<25s}"
          f" {np.mean(gb_acc):>10.3f} {np.mean(gb_kap):>10.3f} {np.mean(gb_f1):>10.3f}")
    print(f"  (± std acc: DT={np.std(dt_acc):.3f}  GB={np.std(gb_acc):.3f})")

    return {
        "decision_tree": {
            "accuracy": {"mean": float(np.mean(dt_acc)), "std": float(np.std(dt_acc))},
            "kappa": {"mean": float(np.mean(dt_kap)), "std": float(np.std(dt_kap))},
            "f1_weighted": {"mean": float(np.mean(dt_f1)), "std": float(np.std(dt_f1))},
            "per_fold_accuracy": [float(v) for v in dt_acc],
        },
        "gradient_boosting": {
            "accuracy": {"mean": float(np.mean(gb_acc)), "std": float(np.std(gb_acc))},
            "kappa": {"mean": float(np.mean(gb_kap)), "std": float(np.std(gb_kap))},
            "f1_weighted": {"mean": float(np.mean(gb_f1)), "std": float(np.std(gb_f1))},
            "per_fold_accuracy": [float(v) for v in gb_acc],
        },
    }


# ===================================================================
# Analysis 5 — Feature correlation with known physiology
# ===================================================================
def analysis_5(pooled_flat: dict, fig_dir: Path) -> dict:
    print("\n" + "=" * 70)
    print("Analysis 5: Correlation with spectral band powers")
    print("=" * 70)

    ng = len(ALL_GEOM_FEATURES)
    ns = len(SPECTRAL_BANDS)
    pearson_r = np.zeros((ng, ns))
    spearman_r = np.zeros((ng, ns))
    pearson_p = np.zeros((ng, ns))

    for i, gf in enumerate(ALL_GEOM_FEATURES):
        gv = np.array(pooled_flat[gf], dtype=float)
        for j, sb in enumerate(SPECTRAL_BANDS):
            sv = np.array(pooled_flat[sb], dtype=float)
            mask = np.isfinite(gv) & np.isfinite(sv)
            if mask.sum() > 10:
                pearson_r[i, j], pearson_p[i, j] = pearsonr(gv[mask], sv[mask])
                spearman_r[i, j], _ = spearmanr(gv[mask], sv[mask])

    # --- Heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, mat, title in [
        (axes[0], pearson_r, "Pearson r"),
        (axes[1], spearman_r, "Spearman ρ"),
    ]:
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(ns))
        ax.set_xticklabels(SPECTRAL_BANDS, rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(ng))
        ax.set_yticklabels(ALL_GEOM_FEATURES, fontsize=9)
        ax.set_title(title, fontsize=14)
        for ii in range(ng):
            for jj in range(ns):
                v = mat[ii, jj]
                c = "white" if abs(v) > 0.45 else "black"
                ax.text(jj, ii, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=c)
        fig.colorbar(im, ax=ax, shrink=0.55)

    fig.suptitle(
        "Geometric features vs spectral band power", fontsize=15, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Identify orthogonal / redundant features
    max_abs = np.max(np.abs(pearson_r), axis=1)
    rank = np.argsort(max_abs)

    print("\n  Most orthogonal to all spectral bands (novel information):")
    for idx in rank[:5]:
        print(f"    {ALL_GEOM_FEATURES[idx]:<25s}  max|r| = {max_abs[idx]:.3f}")

    print("\n  Most redundant with spectral bands:")
    for idx in rank[-3:][::-1]:
        f = ALL_GEOM_FEATURES[idx]
        best_j = int(np.argmax(np.abs(pearson_r[idx])))
        print(f"    {f:<25s}  max|r| = {max_abs[idx]:.3f}  (with {SPECTRAL_BANDS[best_j]})")

    print("\n  Saved correlation_heatmap.png")

    # Build JSON-friendly result
    corr_result: dict = {}
    for i, gf in enumerate(ALL_GEOM_FEATURES):
        corr_result[gf] = {}
        for j, sb in enumerate(SPECTRAL_BANDS):
            corr_result[gf][sb] = {
                "pearson": float(pearson_r[i, j]),
                "spearman": float(spearman_r[i, j]),
                "pearson_p": float(pearson_p[i, j]),
            }
    return corr_result


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "sleep-edf"
    results_dir = PROJECT_ROOT / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Download ---
    print("=" * 70)
    print("Step 0: Ensure Sleep-EDF data is present")
    print("=" * 70)
    download_sleep_edf(data_dir)

    # --- Load + extract ---
    print("=" * 70)
    print("Loading subjects & extracting features")
    print("=" * 70)

    all_feature_keys = ALL_GEOM_FEATURES + SPECTRAL_BANDS + ["delta_power_abs"]

    per_subject: dict[str, dict] = {}
    pooled_by_stage: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    pooled_flat: dict[str, list] = defaultdict(list)

    n_loaded = 0
    for subj in SUBJECTS:
        print(f"  {subj} ... ", end="", flush=True)
        result = load_subject(data_dir, subj)
        if result is None:
            print("skipped")
            continue

        epochs, labels = result
        n_loaded += 1

        sf: dict[str, list] = {k: [] for k in all_feature_keys}
        sl: list[str] = []

        for epoch, label in zip(epochs, labels):
            feats = extract_all_features(epoch)
            if feats is None:
                continue
            for k in all_feature_keys:
                v = feats.get(k, 0.0)
                sf[k].append(v)
                pooled_by_stage[k][label].append(v)
                pooled_flat[k].append(v)
            sl.append(label)

        per_subject[subj] = {"labels": sl, "features": sf}
        print(f"{len(sl)} epochs")

    total = sum(len(d["labels"]) for d in per_subject.values())
    print(f"\n  {n_loaded} subjects, {total} epochs total\n")
    if total < 100:
        print("ERROR: Too few epochs. Exiting.")
        sys.exit(1)

    # --- Run analyses ---
    stats_1 = analysis_1(pooled_by_stage, fig_dir)
    analysis_2(per_subject, fig_dir)
    proc_s = analysis_3(per_subject, fig_dir)
    thresh = analysis_4(per_subject, fig_dir)
    corrs = analysis_5(pooled_flat, fig_dir)

    # --- Save JSON ---
    output = {
        "n_subjects": n_loaded,
        "n_epochs": total,
        "stage_counts": {
            s: len(pooled_by_stage[ALL_GEOM_FEATURES[0]][s]) for s in STAGES
        },
        "analysis_1_stage_phenotype": stats_1,
        "analysis_3_process_s": proc_s,
        "analysis_4_thresholds": thresh,
        "analysis_5_correlations": corrs,
    }
    out_path = results_dir / "geometric_phenotypes.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)

    # Discriminability ranking (F-ratio proxy)
    print("\n  Feature discriminability (between/within stage variance):")
    f_ratios: dict[str, float] = {}
    for feat in ALL_GEOM_FEATURES:
        stage_means = []
        within_vars = []
        for s in STAGES:
            vals = pooled_by_stage[feat][s]
            if vals:
                stage_means.append(np.mean(vals))
                within_vars.append(np.var(vals))
        if stage_means and within_vars:
            f_ratios[feat] = float(np.var(stage_means) / (np.mean(within_vars) + 1e-15))
        else:
            f_ratios[feat] = 0.0
    for feat, fr in sorted(f_ratios.items(), key=lambda x: -x[1])[:5]:
        print(f"    {feat:<25s}  F-ratio = {fr:.4f}")

    # Physiological predictions
    print("\n  Physiological prediction checks:")
    o1_means = {s: np.mean(pooled_by_stage["omega1"][s]) for s in STAGES}
    print(f"    ω₁ ranking: ", end="")
    for s in sorted(o1_means, key=o1_means.get):
        print(f"{s}({o1_means[s]:.4f}) ", end="")
    print()
    if o1_means.get("N3", 999) < o1_means.get("W", 0):
        print("    ✓ ω₁(N3) < ω₁(W) — matches prediction (slow delta winding)")
    else:
        print("    ✗ ω₁(N3) ≥ ω₁(W) — does NOT match prediction")

    if pooled_by_stage["phase_diff_std"]["N3"]:
        pds_n3 = np.mean(pooled_by_stage["phase_diff_std"]["N3"])
        pds_w = np.mean(pooled_by_stage["phase_diff_std"]["W"])
        if pds_n3 < pds_w:
            print(f"    ✓ phase_diff_std: N3({pds_n3:.4f}) < W({pds_w:.4f}) — "
                  "lower in N3 (coherent delta)")
        else:
            print(f"    ✗ phase_diff_std: N3({pds_n3:.4f}) ≥ W({pds_w:.4f})")

    # Process S
    if proc_s:
        print(f"\n  Process S:")
        print(f"    r(ω₁, δ) = {proc_s['pearson_r']:.3f}  p = {proc_s['pearson_p']:.2e}")
        if "tau_omega1_hours" in proc_s:
            tau = proc_s["tau_omega1_hours"]
            print(f"    ω₁ decay τ = {tau:.1f} h  (expected ≈ 4.2 h)")

    # Classification
    if thresh:
        dt = thresh["decision_tree"]
        gb = thresh["gradient_boosting"]
        print(f"\n  Classification (LOO, geometric features only):")
        print(f"    Rule-based DT:    Acc={dt['accuracy']['mean']:.3f}  "
              f"κ={dt['kappa']['mean']:.3f}  F1={dt['f1_weighted']['mean']:.3f}")
        print(f"    GradientBoosting: Acc={gb['accuracy']['mean']:.3f}  "
              f"κ={gb['kappa']['mean']:.3f}  F1={gb['f1_weighted']['mean']:.3f}")

    print(f"\n  All results  -> {out_path}")
    print(f"  All figures  -> {fig_dir}/")


if __name__ == "__main__":
    main()
