#!/usr/bin/env python3
"""Sensitivity analysis: effect of embedding parameters d and τ on classification
performance and geometric feature quality.

Tests d ∈ {2, 4, 6, 8} × τ ∈ {10, 15, 20, 25, 30, 40, 50} = 28 combinations.
For each, computes CMI(ω₁ | δ), MI(ω₁, stage), and F-ratio of ω₁ stage separation.
For d=4 only, also evaluates full torus features from torus_features_v2.

Outputs:
  results/figures/sensitivity_cmi_heatmap.png
  results/figures/sensitivity_fratio_heatmap.png
  results/figures/sensitivity_cmi_lines.png
  results/sensitivity_analysis.json
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
from sklearn.metrics import mutual_info_score

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.torus_features_v2 import extract_torus_features_v2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SFREQ = 100
EPOCH_SEC = 30
EPOCH_SAMPLES = SFREQ * EPOCH_SEC

D_VALUES = [2, 4, 6, 8]
TAU_VALUES = [10, 15, 20, 25, 30, 40, 50]
N_CMI_BINS = 10

SUBJECTS = [
    "SC4001", "SC4002", "SC4011", "SC4012",
    "SC4021", "SC4022", "SC4031", "SC4041",
    "SC4042", "SC4051", "SC4052", "SC4061",
    "SC4062", "SC4071", "SC4072", "SC4081",
    "SC4091", "SC4092",
]

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


# ===================================================================
# Data loading (skip download — files must already exist)
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
        stage = STAGE_MAP.get(str(desc).strip())
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
# Embedding & feature helpers
# ===================================================================
def embed(signal: np.ndarray, d: int, tau: int) -> np.ndarray | None:
    """Time-delay embedding. Returns (n_points, d) or None if too short."""
    n_points = len(signal) - (d - 1) * tau
    if n_points < 10:
        return None
    embedded = np.zeros((n_points, d))
    for i in range(d):
        embedded[:, i] = signal[i * tau : i * tau + n_points]
    return embedded


def compute_omega1(embedding: np.ndarray) -> float:
    """ω₁ = mean |dθ/dt| from first two embedding dimensions."""
    theta = np.arctan2(embedding[:, 1], embedding[:, 0])
    theta_uw = np.unwrap(theta)
    dtheta = np.diff(theta_uw)
    return float(np.mean(np.abs(dtheta)))


def compute_delta_power(epoch: np.ndarray) -> float:
    """Delta band power (0.5–4 Hz) via Welch PSD."""
    freqs, psd = welch(epoch, fs=SFREQ, nperseg=min(256, len(epoch)))
    mask = (freqs >= 0.5) & (freqs <= 4.0)
    return float(np.sum(psd[mask])) if np.any(mask) else 0.0


# ===================================================================
# CMI / MI / F-ratio
# ===================================================================
def discretize(values: np.ndarray, n_bins: int = N_CMI_BINS) -> np.ndarray:
    """Discretize into percentile bins."""
    edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    return np.searchsorted(edges[1:-1], values, side="right")


def mi(a: np.ndarray, b: np.ndarray) -> float:
    return float(mutual_info_score(a, b))


def joint_bins(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * (int(b.max()) + 1) + b


def compute_cmi(feat: np.ndarray, delta: np.ndarray, stage: np.ndarray) -> float:
    """CMI = MI(feat+delta; stage) − MI(delta; stage)."""
    fb = discretize(feat)
    db = discretize(delta)
    return mi(joint_bins(fb, db), stage) - mi(db, stage)


def compute_fratio(feat: np.ndarray, stage: np.ndarray) -> float:
    """Between-stage variance / within-stage variance for feat."""
    unique_stages = np.unique(stage)
    stage_means = []
    within_vars = []
    for s in unique_stages:
        vals = feat[stage == s]
        if len(vals) > 1:
            stage_means.append(np.mean(vals))
            within_vars.append(np.var(vals))
    if not within_vars or np.mean(within_vars) < 1e-15:
        return 0.0
    return float(np.var(stage_means) / np.mean(within_vars))


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "sleep-edf"
    results_dir = PROJECT_ROOT / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load all subjects once (raw epochs + labels)
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Loading subjects from data/sleep-edf/")
    print("=" * 65)

    all_epochs: list[np.ndarray] = []
    all_labels: list[int] = []

    n_loaded = 0
    for subj in SUBJECTS:
        result = load_subject(data_dir, subj)
        if result is None:
            continue
        epochs, labels = result
        n_loaded += 1
        for ep, lab in zip(epochs, labels):
            all_epochs.append(ep)
            all_labels.append(STAGE_TO_INT[lab])
        print(f"  {subj}: {len(labels)} epochs")

    all_labels_arr = np.array(all_labels)
    n_total = len(all_epochs)
    print(f"\n  {n_loaded} subjects, {n_total} epochs\n")

    if n_total < 100:
        print("ERROR: Too few epochs. Exiting.")
        sys.exit(1)

    # Pre-compute delta power (independent of d, τ)
    print("Computing delta power for all epochs ...")
    delta_powers = np.array([compute_delta_power(ep) for ep in all_epochs])
    print("  Done.\n")

    # ------------------------------------------------------------------
    # Sweep d × τ
    # ------------------------------------------------------------------
    print("=" * 65)
    print(f"Sensitivity sweep: {len(D_VALUES)} d × {len(TAU_VALUES)} τ"
          f" = {len(D_VALUES) * len(TAU_VALUES)} combinations")
    print("=" * 65)

    results: dict[str, dict] = {}  # key = "d{d}_tau{tau}"
    cmi_grid = np.full((len(D_VALUES), len(TAU_VALUES)), np.nan)
    mi_grid = np.full((len(D_VALUES), len(TAU_VALUES)), np.nan)
    fratio_grid = np.full((len(D_VALUES), len(TAU_VALUES)), np.nan)
    valid_frac_grid = np.full((len(D_VALUES), len(TAU_VALUES)), np.nan)

    for di, d in enumerate(D_VALUES):
        for ti, tau in enumerate(TAU_VALUES):
            key = f"d{d}_tau{tau}"
            print(f"  d={d}, τ={tau:>2d} ... ", end="", flush=True)

            omega1_vals = []
            delta_vals = []
            stage_vals = []
            valid = 0

            for idx in range(n_total):
                emb = embed(all_epochs[idx], d, tau)
                if emb is None:
                    continue
                if not np.all(np.isfinite(emb)):
                    continue
                o1 = compute_omega1(emb)
                if not np.isfinite(o1):
                    continue
                omega1_vals.append(o1)
                delta_vals.append(delta_powers[idx])
                stage_vals.append(all_labels[idx])
                valid += 1

            valid_frac = valid / n_total

            if valid < 100:
                print(f"skipped ({valid} valid epochs)")
                results[key] = {
                    "d": d, "tau": tau,
                    "valid_epochs": valid, "valid_frac": valid_frac,
                    "cmi": None, "mi": None, "fratio": None,
                }
                continue

            o1_arr = np.array(omega1_vals)
            d_arr = np.array(delta_vals)
            s_arr = np.array(stage_vals)

            cmi_val = compute_cmi(o1_arr, d_arr, s_arr)
            mi_val = mi(discretize(o1_arr), s_arr)
            fr_val = compute_fratio(o1_arr, s_arr)

            cmi_grid[di, ti] = cmi_val
            mi_grid[di, ti] = mi_val
            fratio_grid[di, ti] = fr_val
            valid_frac_grid[di, ti] = valid_frac

            entry: dict = {
                "d": d, "tau": tau,
                "valid_epochs": valid, "valid_frac": round(valid_frac, 4),
                "cmi": round(cmi_val, 6),
                "mi": round(mi_val, 6),
                "fratio": round(fr_val, 6),
            }

            # For d=4, also evaluate full torus v2 features
            if d == 4:
                v2_cmis = {}
                # Collect features across epochs
                feat_arrays: dict[str, list[float]] = {}
                for idx_v in range(n_total):
                    emb_v = embed(all_epochs[idx_v], 4, tau)
                    if emb_v is None or not np.all(np.isfinite(emb_v)):
                        continue
                    feats = extract_torus_features_v2(emb_v)
                    for fname, fval in feats.items():
                        feat_arrays.setdefault(fname, []).append(fval)

                # Compute CMI for each v2 feature
                n_v2 = len(next(iter(feat_arrays.values()))) if feat_arrays else 0
                if n_v2 >= 100:
                    # Match delta/stage arrays to valid v2 epochs
                    d_v2 = d_arr[:n_v2]
                    s_v2 = s_arr[:n_v2]
                    for fname, fvals in feat_arrays.items():
                        f_arr = np.array(fvals[:n_v2])
                        if np.std(f_arr) < 1e-15:
                            v2_cmis[fname] = 0.0
                        else:
                            v2_cmis[fname] = round(
                                compute_cmi(f_arr, d_v2, s_v2), 6
                            )
                entry["torus_v2_cmi"] = v2_cmis

            results[key] = entry
            print(f"CMI={cmi_val:.4f}  MI={mi_val:.4f}  F={fr_val:.4f}"
                  f"  ({valid_frac:.0%} valid)")

    # ------------------------------------------------------------------
    # Find optimum
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Results")
    print("=" * 65)

    # Best CMI
    best_cmi_idx = np.nanargmax(cmi_grid)
    best_di, best_ti = np.unravel_index(best_cmi_idx, cmi_grid.shape)
    best_d_cmi = D_VALUES[best_di]
    best_tau_cmi = TAU_VALUES[best_ti]
    best_cmi_val = cmi_grid[best_di, best_ti]

    # Best F-ratio
    best_fr_idx = np.nanargmax(fratio_grid)
    best_di_f, best_ti_f = np.unravel_index(best_fr_idx, fratio_grid.shape)
    best_d_fr = D_VALUES[best_di_f]
    best_tau_fr = TAU_VALUES[best_ti_f]
    best_fr_val = fratio_grid[best_di_f, best_ti_f]

    print(f"\n  Best CMI(ω₁|δ):   d={best_d_cmi}, τ={best_tau_cmi}"
          f"  CMI={best_cmi_val:.6f}")
    print(f"  Best F-ratio(ω₁): d={best_d_fr}, τ={best_tau_fr}"
          f"  F={best_fr_val:.6f}")

    # Compare with d=4, τ=25
    ref_di = D_VALUES.index(4)
    ref_ti = TAU_VALUES.index(25)
    ref_cmi = cmi_grid[ref_di, ref_ti]
    ref_fr = fratio_grid[ref_di, ref_ti]
    ref_mi = mi_grid[ref_di, ref_ti]

    print(f"\n  Reference d=4, τ=25:")
    print(f"    CMI={ref_cmi:.6f}  ({ref_cmi / best_cmi_val * 100:.1f}% of best)")
    print(f"    F  ={ref_fr:.6f}  ({ref_fr / best_fr_val * 100:.1f}% of best)")

    if ref_cmi >= best_cmi_val * 0.90:
        print("    → d=4, τ=25 is NEAR-OPTIMAL for CMI (≥90% of best)")
    else:
        print("    → d=4, τ=25 is SUBOPTIMAL for CMI (<90% of best)")

    if ref_fr >= best_fr_val * 0.90:
        print("    → d=4, τ=25 is NEAR-OPTIMAL for F-ratio (≥90% of best)")
    else:
        print("    → d=4, τ=25 is SUBOPTIMAL for F-ratio (<90% of best)")

    # Effect of d: does d>4 improve CMI?
    print("\n  Effect of additional dimensions (at τ=25):")
    for d in D_VALUES:
        di_d = D_VALUES.index(d)
        c = cmi_grid[di_d, ref_ti]
        f = fratio_grid[di_d, ref_ti]
        label = ""
        if d == 2:
            label = " (baseline, no torus)"
        elif d == 4:
            label = " (Clifford torus)"
        elif d > 4:
            label = " (extra dims)"
        if np.isfinite(c):
            print(f"    d={d}: CMI={c:.6f}  F={f:.6f}{label}")
        else:
            print(f"    d={d}: insufficient valid epochs{label}")

    # d=4 torus v2 features across τ
    print("\n  Torus v2 feature CMI across τ (d=4):")
    print(f"    {'Feature':<25s}", end="")
    for tau in TAU_VALUES:
        print(f" τ={tau:>2d}", end="")
    print()
    print("    " + "-" * (25 + 6 * len(TAU_VALUES)))

    v2_names_printed = False
    for tau in TAU_VALUES:
        k = f"d4_tau{tau}"
        if k in results and "torus_v2_cmi" in results[k]:
            if not v2_names_printed:
                v2_feature_names = list(results[k]["torus_v2_cmi"].keys())
                v2_names_printed = True

    if v2_names_printed:
        for fname in v2_feature_names:
            print(f"    {fname:<25s}", end="")
            for tau in TAU_VALUES:
                k = f"d4_tau{tau}"
                val = results.get(k, {}).get("torus_v2_cmi", {}).get(fname)
                if val is not None:
                    print(f" {val:5.4f}", end="")
                else:
                    print(f"   n/a", end="")
            print()

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print(f"\n  Generating figures ...")

    d_labels = [str(d) for d in D_VALUES]
    tau_labels = [str(t) for t in TAU_VALUES]

    # 1. CMI heatmap
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(cmi_grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(TAU_VALUES)))
    ax.set_xticklabels(tau_labels, fontsize=11)
    ax.set_yticks(range(len(D_VALUES)))
    ax.set_yticklabels(d_labels, fontsize=11)
    ax.set_xlabel("τ (samples at 100 Hz)", fontsize=12)
    ax.set_ylabel("d (embedding dimension)", fontsize=12)
    ax.set_title("CMI(ω₁ | δ-power) across d × τ", fontsize=14)
    for i in range(len(D_VALUES)):
        for j in range(len(TAU_VALUES)):
            v = cmi_grid[i, j]
            if np.isfinite(v):
                c = "white" if v < np.nanmedian(cmi_grid) else "black"
                ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                        fontsize=8, color=c)
    # Mark reference d=4, τ=25
    ax.plot(ref_ti, ref_di, "s", ms=20, mec="red", mfc="none", mew=2)
    # Mark best
    ax.plot(best_ti, best_di, "*", ms=16, color="red")
    fig.colorbar(im, ax=ax, shrink=0.7, label="CMI")
    fig.tight_layout()
    fig.savefig(fig_dir / "sensitivity_cmi_heatmap.png", dpi=150)
    plt.close(fig)

    # 2. F-ratio heatmap
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(fratio_grid, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(TAU_VALUES)))
    ax.set_xticklabels(tau_labels, fontsize=11)
    ax.set_yticks(range(len(D_VALUES)))
    ax.set_yticklabels(d_labels, fontsize=11)
    ax.set_xlabel("τ (samples at 100 Hz)", fontsize=12)
    ax.set_ylabel("d (embedding dimension)", fontsize=12)
    ax.set_title("F-ratio of ω₁ stage separation across d × τ", fontsize=14)
    for i in range(len(D_VALUES)):
        for j in range(len(TAU_VALUES)):
            v = fratio_grid[i, j]
            if np.isfinite(v):
                c = "white" if v < np.nanmedian(fratio_grid) else "black"
                ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                        fontsize=8, color=c)
    ax.plot(ref_ti, ref_di, "s", ms=20, mec="cyan", mfc="none", mew=2)
    ax.plot(best_ti_f, best_di_f, "*", ms=16, color="cyan")
    fig.colorbar(im, ax=ax, shrink=0.7, label="F-ratio")
    fig.tight_layout()
    fig.savefig(fig_dir / "sensitivity_fratio_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. CMI line plot: one line per d
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ["o", "s", "D", "^"]
    colors_d = ["#5bffa8", "#5b8bd4", "#f5c842", "#6e3fa0"]
    for di, d in enumerate(D_VALUES):
        y = cmi_grid[di, :]
        mask = np.isfinite(y)
        x_plot = np.array(TAU_VALUES)[mask]
        y_plot = y[mask]
        ax.plot(x_plot, y_plot, f"-{markers[di]}", color=colors_d[di],
                label=f"d={d}", lw=2, ms=7)

    ax.axvline(25, ls="--", color="gray", alpha=0.5, label="τ=25 (Paper #1)")
    ax.set_xlabel("τ (samples at 100 Hz)", fontsize=12)
    ax.set_ylabel("CMI(ω₁ | δ-power)", fontsize=12)
    ax.set_title("CMI vs τ for each embedding dimension", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "sensitivity_cmi_lines.png", dpi=150)
    plt.close(fig)

    print(f"  Saved: sensitivity_cmi_heatmap.png")
    print(f"         sensitivity_fratio_heatmap.png")
    print(f"         sensitivity_cmi_lines.png")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output = {
        "d_values": D_VALUES,
        "tau_values": TAU_VALUES,
        "n_subjects": n_loaded,
        "n_epochs": n_total,
        "best_cmi": {
            "d": best_d_cmi, "tau": best_tau_cmi,
            "value": round(float(best_cmi_val), 6),
        },
        "best_fratio": {
            "d": best_d_fr, "tau": best_tau_fr,
            "value": round(float(best_fr_val), 6),
        },
        "reference_d4_tau25": {
            "cmi": round(float(ref_cmi), 6),
            "mi": round(float(ref_mi), 6),
            "fratio": round(float(ref_fr), 6),
            "cmi_pct_of_best": round(float(ref_cmi / best_cmi_val * 100), 1),
            "fratio_pct_of_best": round(float(ref_fr / best_fr_val * 100), 1),
        },
        "grid": results,
    }

    out_path = results_dir / "sensitivity_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
