#!/usr/bin/env python3
"""Geometric phenotype analysis on the HMC dataset (dual-channel).

Replicates the Sleep-EDF geometric phenotype analysis for the Haaglanden
Medisch Centrum sleep staging database.  Key differences:
  - Dual-channel: EEG C4-M1 + EEG C3-M2 → true bilateral phase_diff_std
  - AASM native labels (no R&K conversion)
  - 256 Hz resampled to 100 Hz

Analyses
--------
1. Stage phenotyping — violin plots + summary statistics
2. Overnight geometric trajectories — 3 subjects
3. Correlation heatmap — geometric features vs spectral bands
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from scipy.signal import welch
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.takens import time_delay_embedding
from src.features.torus_features_v2 import extract_torus_features_v2
from src.features.spectral import compute_band_powers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SFREQ = 100  # target after resampling
EPOCH_SEC = 30
EPOCH_SAMPLES = SFREQ * EPOCH_SEC

# HMC label mapping (AASM native + numeric variants)
HMC_LABELS = {
    "Sleep stage W": "W",
    "Sleep stage N1": "N1",
    "Sleep stage N2": "N2",
    "Sleep stage N3": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": None,
    "0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "REM", "5": None,
    "W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "REM", "R": "REM",
}

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGES)}
STAGE_COLORS = {
    "W": "#f5c842", "N1": "#90caf9", "N2": "#5b8bd4",
    "N3": "#1a1a6e", "REM": "#6e3fa0",
}

# 8 geometric features for this analysis
GEOM_FEATURES = [
    "omega1", "torus_curvature", "angular_acceleration",
    "geodesic_distance", "angular_entropy", "phase_diff_std",
    "phase_coherence", "transition_rate",
]

SPECTRAL_BANDS = ["delta", "theta", "alpha", "sigma", "beta"]


# ===================================================================
# Subject discovery
# ===================================================================
def find_hmc_subjects(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """Find all valid (PSG, scoring) pairs. Returns [(sid, psg, hyp), ...]."""
    pairs = []
    for i in range(1, 200):
        sid = f"SN{i:03d}"
        psg = data_dir / f"{sid}.edf"
        hyp = data_dir / f"{sid}_sleepscoring.edf"
        if psg.exists() and hyp.exists():
            if psg.stat().st_size > 1_000_000 and hyp.stat().st_size > 500:
                pairs.append((sid, psg, hyp))
    return pairs


# ===================================================================
# Data loading (dual-channel)
# ===================================================================
def load_hmc_subject(
    sid: str, psg_path: Path, hyp_path: Path, print_debug: bool = False
):
    """Load one HMC subject.

    Returns (ch1_epochs, ch2_epochs, labels) or None.
    ch1 = C4-M1, ch2 = C3-M2.  Each is (n_epochs, n_samples).
    """
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)

    if print_debug:
        print(f"    Channel names: {raw.ch_names}")

    # Pick bilateral central channels
    ch1_name, ch2_name = None, None
    ch_pairs = [
        ("EEG C4-M1", "EEG C3-M2"),
        ("C4-M1", "C3-M2"),
        ("EEG C4", "EEG C3"),
        ("C4", "C3"),
    ]
    for c1, c2 in ch_pairs:
        if c1 in raw.ch_names and c2 in raw.ch_names:
            ch1_name, ch2_name = c1, c2
            break

    if ch1_name is None:
        # Fallback: any 2 EEG channels
        eeg_chs = [c for c in raw.ch_names if "EEG" in c.upper() or c.startswith("C")]
        if len(eeg_chs) >= 2:
            ch1_name, ch2_name = eeg_chs[0], eeg_chs[1]
        else:
            return None

    raw.pick([ch1_name, ch2_name])

    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)
    raw.filter(0.5, 30.0, verbose=False)

    ch1_signal = raw.get_data()[0]
    ch2_signal = raw.get_data()[1]
    total_seconds = len(ch1_signal) / SFREQ
    n_epochs = int(total_seconds // EPOCH_SEC)

    # Annotations
    annots = mne.read_annotations(str(hyp_path))

    if print_debug:
        descs = sorted(set(str(d).strip() for d in annots.description))
        print(f"    Annotation descriptions: {descs}")

    epoch_labels: list[str | None] = [None] * n_epochs
    for onset, duration, desc in zip(
        annots.onset, annots.duration, annots.description
    ):
        stage = HMC_LABELS.get(str(desc).strip())
        if stage is None:
            continue
        s_ep = int(onset // EPOCH_SEC)
        n_dur = max(1, int(duration // EPOCH_SEC))
        for e in range(s_ep, min(s_ep + n_dur, n_epochs)):
            epoch_labels[e] = stage

    ch1_out, ch2_out, labels_out = [], [], []
    for i in range(n_epochs):
        if epoch_labels[i] is None:
            continue
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        if end > len(ch1_signal):
            break
        ep1 = ch1_signal[start:end]
        ep2 = ch2_signal[start:end]
        # Reject ±500 µV (MNE returns volts)
        if np.max(np.abs(ep1)) > 500e-6 or np.max(np.abs(ep2)) > 500e-6:
            continue
        ch1_out.append(ep1)
        ch2_out.append(ep2)
        labels_out.append(epoch_labels[i])

    if not ch1_out:
        return None
    return np.array(ch1_out), np.array(ch2_out), labels_out


# ===================================================================
# Feature extraction (dual-channel, 8 geometric + spectral)
# ===================================================================
def _wrap(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2 * np.pi) - np.pi


def extract_features_dual(
    ch1_epoch: np.ndarray, ch2_epoch: np.ndarray
) -> dict[str, float] | None:
    """Extract 8 geometric + spectral features from a dual-channel epoch pair."""
    try:
        emb1, _ = time_delay_embedding(ch1_epoch, dimension=4, tau=25)
        emb2, _ = time_delay_embedding(ch2_epoch, dimension=4, tau=25)
    except ValueError:
        return None

    if emb1.shape[0] < 10 or emb2.shape[0] < 10:
        return None

    # Use the shorter length for bilateral comparisons
    n_min = min(emb1.shape[0], emb2.shape[0])
    emb1 = emb1[:n_min]
    emb2 = emb2[:n_min]

    # --- Per-channel angles ---
    theta_ch1 = np.arctan2(emb1[:, 1], emb1[:, 0])
    theta_ch2 = np.arctan2(emb2[:, 1], emb2[:, 0])

    # omega1 from primary channel (C4-M1)
    theta_uw = np.unwrap(theta_ch1)
    dtheta = np.diff(theta_uw)
    omega1 = float(np.mean(np.abs(dtheta)))

    # Transition rate from primary channel embedding
    signs = (emb1 >= 0).astype(int)
    vertices = signs[:, 0] * 8 + signs[:, 1] * 4 + signs[:, 2] * 2 + signs[:, 3]
    transition_rate = float(np.sum(np.diff(vertices) != 0) / max(len(vertices) - 1, 1))

    # --- Bilateral features ---
    # Phase difference θ_ch1 − θ_ch2 (circular)
    phase_diff = _wrap(theta_ch1 - theta_ch2)

    # phase_diff_std: circular standard deviation
    # circ_std = sqrt(-2 * log(R)) where R = mean resultant length
    mean_cos = float(np.mean(np.cos(phase_diff)))
    mean_sin = float(np.mean(np.sin(phase_diff)))
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    phase_diff_std = float(np.sqrt(-2.0 * np.log(max(R, 1e-10))))

    # phase_coherence: mean resultant length (0=random, 1=perfectly locked)
    phase_coherence = float(R)

    # --- Torus v2 features from primary channel ---
    v2 = extract_torus_features_v2(emb1)

    # --- Spectral (from primary channel, relative) ---
    spectral = compute_band_powers(ch1_epoch, SFREQ, relative=True)
    spectral_abs = compute_band_powers(ch1_epoch, SFREQ, relative=False)

    feats: dict[str, float] = {
        "omega1": omega1,
        "torus_curvature": v2["torus_curvature"],
        "angular_acceleration": v2["angular_acceleration"],
        "geodesic_distance": v2["geodesic_distance"],
        "angular_entropy": v2["angular_entropy"],
        "phase_diff_std": phase_diff_std,
        "phase_coherence": phase_coherence,
        "transition_rate": transition_rate,
    }
    feats.update(spectral)
    feats["delta_power_abs"] = spectral_abs.get("delta", 0.0)
    return feats


# ===================================================================
# Analysis 1 — Stage phenotyping
# ===================================================================
def analysis_1(pooled_by_stage: dict, fig_dir: Path) -> dict:
    print("\n" + "=" * 70)
    print("Analysis 1: HMC stage phenotyping")
    print("=" * 70)

    features_to_plot = GEOM_FEATURES + ["delta"]
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

        # Violin plot
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
        ax.set_title(f"HMC: {feat} by sleep stage", fontsize=13)
        ax.set_facecolor("#f5f5f5")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"hmc_phenotype_{feat}.png", dpi=150)
        plt.close(fig)

    # Summary table
    header = f"  {'Feature':<25s}"
    for s in STAGES:
        header += f" {s:>10s}"
    print(header)
    print("  " + "-" * (25 + 11 * len(STAGES)))
    for feat in GEOM_FEATURES:
        line = f"  {feat:<25s}"
        for s in STAGES:
            line += f" {stats[feat][s]['mean']:>10.4f}"
        print(line)

    print(f"\n  Saved {len(features_to_plot)} violin plots (hmc_phenotype_*.png)")
    return stats


# ===================================================================
# Analysis 2 — Overnight trajectories
# ===================================================================
def analysis_2(per_subject: dict, fig_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("Analysis 2: HMC overnight trajectories")
    print("=" * 70)

    counts = {s: len(d["labels"]) for s, d in per_subject.items()}
    top3 = sorted(counts, key=counts.get, reverse=True)[:3]
    print(f"  Selected: {top3}  (epochs: {[counts[s] for s in top3]})")

    feats_to_show = [
        ("omega1", "ω₁"),
        ("torus_curvature", "Torus curvature"),
        ("geodesic_distance", "Geodesic distance"),
        ("phase_diff_std", "Phase diff std (bilateral)"),
        ("delta", "δ-power (relative)"),
    ]

    for subj in top3:
        data = per_subject[subj]
        labels = data["labels"]
        n = len(labels)
        t_hrs = np.arange(n) * EPOCH_SEC / 3600.0

        n_panels = len(feats_to_show) + 1
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(14, 2.4 * n_panels), sharex=True,
        )

        # Hypnogram
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
            f"HMC overnight — {subj}", fontsize=14, color="white",
        )
        ax.invert_yaxis()
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="gray")

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
            fig_dir / f"hmc_overnight_{subj}.png",
            dpi=150, facecolor="#0c0e14",
        )
        plt.close(fig)

    print(f"  Saved overnight figures for {top3}.")


# ===================================================================
# Analysis 3 — Correlation heatmap
# ===================================================================
def analysis_3(pooled_flat: dict, fig_dir: Path) -> dict:
    print("\n" + "=" * 70)
    print("Analysis 3: HMC correlation with spectral bands")
    print("=" * 70)

    ng = len(GEOM_FEATURES)
    ns = len(SPECTRAL_BANDS)
    pearson_mat = np.zeros((ng, ns))
    spearman_mat = np.zeros((ng, ns))
    pearson_p_mat = np.zeros((ng, ns))

    for i, gf in enumerate(GEOM_FEATURES):
        gv = np.array(pooled_flat[gf], dtype=float)
        for j, sb in enumerate(SPECTRAL_BANDS):
            sv = np.array(pooled_flat[sb], dtype=float)
            mask = np.isfinite(gv) & np.isfinite(sv)
            if mask.sum() > 10:
                pearson_mat[i, j], pearson_p_mat[i, j] = pearsonr(
                    gv[mask], sv[mask]
                )
                spearman_mat[i, j], _ = spearmanr(gv[mask], sv[mask])

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, mat, title in [
        (axes[0], pearson_mat, "Pearson r"),
        (axes[1], spearman_mat, "Spearman ρ"),
    ]:
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(ns))
        ax.set_xticklabels(SPECTRAL_BANDS, rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(ng))
        ax.set_yticklabels(GEOM_FEATURES, fontsize=9)
        ax.set_title(f"HMC: {title}", fontsize=14)
        for ii in range(ng):
            for jj in range(ns):
                v = mat[ii, jj]
                c = "white" if abs(v) > 0.45 else "black"
                ax.text(jj, ii, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=c)
        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle(
        "HMC: Geometric features vs spectral band power", fontsize=15, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "hmc_correlation_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # Report orthogonal / redundant
    max_abs = np.max(np.abs(pearson_mat), axis=1)
    rank = np.argsort(max_abs)

    print("\n  Most orthogonal to spectral bands (novel):")
    for idx in rank[:3]:
        print(f"    {GEOM_FEATURES[idx]:<25s}  max|r| = {max_abs[idx]:.3f}")

    print("\n  Most redundant with spectral bands:")
    for idx in rank[-3:][::-1]:
        f = GEOM_FEATURES[idx]
        best_j = int(np.argmax(np.abs(pearson_mat[idx])))
        print(f"    {f:<25s}  max|r| = {max_abs[idx]:.3f}  (with {SPECTRAL_BANDS[best_j]})")

    print("\n  Saved hmc_correlation_heatmap.png")

    corr_result: dict = {}
    for i, gf in enumerate(GEOM_FEATURES):
        corr_result[gf] = {}
        for j, sb in enumerate(SPECTRAL_BANDS):
            corr_result[gf][sb] = {
                "pearson": float(pearson_mat[i, j]),
                "spearman": float(spearman_mat[i, j]),
                "pearson_p": float(pearson_p_mat[i, j]),
            }
    return corr_result


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "hmc"
    results_dir = PROJECT_ROOT / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Discover subjects
    print("=" * 70)
    print("HMC Geometric Phenotype Analysis (dual-channel)")
    print("=" * 70)

    pairs = find_hmc_subjects(data_dir)
    print(f"  Found {len(pairs)} subjects in {data_dir}/\n")

    if not pairs:
        print("ERROR: No HMC subjects found. Exiting.")
        sys.exit(1)

    # Print channel names + annotation descriptions from first subject
    print("  Probing first subject for channel/annotation info:")
    first_sid, first_psg, first_hyp = pairs[0]

    all_feature_keys = GEOM_FEATURES + SPECTRAL_BANDS + ["delta_power_abs"]

    per_subject: dict[str, dict] = {}
    pooled_by_stage: dict[str, dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    pooled_flat: dict[str, list] = defaultdict(list)

    n_loaded = 0
    for idx, (sid, psg_path, hyp_path) in enumerate(pairs):
        print_debug = idx == 0
        print(f"  {sid} ... ", end="", flush=True)

        try:
            result = load_hmc_subject(sid, psg_path, hyp_path,
                                      print_debug=print_debug)
        except Exception as e:
            print(f"error: {e}")
            continue

        if result is None:
            print("skipped")
            continue

        ch1_epochs, ch2_epochs, labels = result
        n_loaded += 1

        sf: dict[str, list] = {k: [] for k in all_feature_keys}
        sl: list[str] = []

        for ch1_ep, ch2_ep, label in zip(ch1_epochs, ch2_epochs, labels):
            feats = extract_features_dual(ch1_ep, ch2_ep)
            if feats is None:
                continue
            for k in all_feature_keys:
                v = feats.get(k, 0.0)
                sf[k].append(v)
                pooled_by_stage[k][label].append(v)
                pooled_flat[k].append(v)
            sl.append(label)

        per_subject[sid] = {"labels": sl, "features": sf}
        print(f"{len(sl)} epochs")

    total = sum(len(d["labels"]) for d in per_subject.values())
    print(f"\n  {n_loaded} subjects loaded, {total} epochs total")
    stage_counts = {s: len(pooled_by_stage[GEOM_FEATURES[0]][s]) for s in STAGES}
    print(f"  Stage counts: {stage_counts}\n")

    if total < 100:
        print("ERROR: Too few epochs. Exiting.")
        sys.exit(1)

    # --- Run analyses ---
    stats_1 = analysis_1(pooled_by_stage, fig_dir)
    analysis_2(per_subject, fig_dir)
    corrs = analysis_3(pooled_flat, fig_dir)

    # --- Save JSON ---
    output = {
        "dataset": "HMC",
        "n_subjects": n_loaded,
        "n_epochs": total,
        "stage_counts": stage_counts,
        "analysis_1_stage_phenotype": stats_1,
        "analysis_3_correlations": corrs,
    }
    out_path = results_dir / "hmc_geometric_phenotypes.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("HMC SUMMARY")
    print("=" * 70)

    # Physiological checks
    o1_means = {s: np.mean(pooled_by_stage["omega1"][s])
                for s in STAGES if pooled_by_stage["omega1"][s]}
    print("\n  ω₁ by stage:")
    for s in sorted(o1_means, key=o1_means.get):
        print(f"    {s}: {o1_means[s]:.4f}")
    if o1_means.get("N3", 999) < o1_means.get("W", 0):
        print("    ✓ ω₁(N3) < ω₁(W)")
    else:
        print("    ✗ ω₁(N3) ≥ ω₁(W)")

    pds_means = {s: np.mean(pooled_by_stage["phase_diff_std"][s])
                 for s in STAGES if pooled_by_stage["phase_diff_std"][s]}
    print("\n  phase_diff_std (bilateral) by stage:")
    for s in sorted(pds_means, key=pds_means.get):
        print(f"    {s}: {pds_means[s]:.4f}")
    if pds_means.get("N3", 999) < pds_means.get("W", 0):
        print("    ✓ Bilateral synchrony highest in N3 (lowest phase_diff_std)")
    if pds_means.get("REM", 0) > pds_means.get("N2", 999):
        print("    ✓ REM more desynchronized than N2")

    # Discriminability
    print("\n  Feature discriminability (F-ratio):")
    for feat in GEOM_FEATURES:
        stage_means = []
        within_vars = []
        for s in STAGES:
            vals = pooled_by_stage[feat][s]
            if vals:
                stage_means.append(np.mean(vals))
                within_vars.append(np.var(vals))
        if stage_means and within_vars and np.mean(within_vars) > 1e-15:
            fr = np.var(stage_means) / np.mean(within_vars)
        else:
            fr = 0.0
        print(f"    {feat:<25s}  F = {fr:.4f}")

    print(f"\n  Results -> {out_path}")
    print(f"  Figures -> {fig_dir}/hmc_*.png")


if __name__ == "__main__":
    main()
