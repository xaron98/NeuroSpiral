#!/usr/bin/env python3
"""Process S validation on the HMC dataset (151 subjects).

Analyses
--------
A. NREM cycle detection and per-cycle feature means
B. Process S exponential decay (ω₁, delta) + cross-correlation
C. 90-minute ultradian cycle detection via Lomb-Scargle on ω₁
D. Epoch-level ω₁ vs delta correlation per subject
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from scipy.optimize import curve_fit
from scipy.signal import welch, lombscargle
from scipy.stats import pearsonr

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


# ===================================================================
# Subject discovery
# ===================================================================
def find_subjects(data_dir: Path) -> list[tuple[str, Path, Path]]:
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
# Loading (single primary channel C4-M1)
# ===================================================================
def load_subject(psg_path: Path, hyp_path: Path):
    """Return (epochs_array, labels) or None.  epochs in volts."""
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)

    ch_name = None
    for c in ["EEG C4-M1", "C4-M1", "EEG C4"]:
        if c in raw.ch_names:
            ch_name = c
            break
    if ch_name is None:
        eeg = [c for c in raw.ch_names if "EEG" in c.upper()]
        if eeg:
            ch_name = eeg[0]
        else:
            return None

    raw.pick([ch_name])
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)
    raw.filter(0.5, 30.0, verbose=False)

    signal = raw.get_data()[0]
    n_epochs = int(len(signal) / SFREQ // EPOCH_SEC)

    annots = mne.read_annotations(str(hyp_path))
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

    epochs_out, labels_out = [], []
    for i in range(n_epochs):
        if epoch_labels[i] is None:
            continue
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        if end > len(signal):
            break
        ep = signal[start:end]
        if np.max(np.abs(ep)) > 500e-6:
            continue
        epochs_out.append(ep)
        labels_out.append(epoch_labels[i])

    if not epochs_out:
        return None
    return np.array(epochs_out), labels_out


# ===================================================================
# Feature extraction
# ===================================================================
def extract_epoch_features(epoch: np.ndarray) -> dict[str, float] | None:
    try:
        emb, _ = time_delay_embedding(epoch, dimension=4, tau=25)
    except ValueError:
        return None
    if emb.shape[0] < 10 or not np.all(np.isfinite(emb)):
        return None

    theta = np.arctan2(emb[:, 1], emb[:, 0])
    dtheta = np.diff(np.unwrap(theta))
    omega1 = float(np.mean(np.abs(dtheta)))

    v2 = extract_torus_features_v2(emb)

    freqs, psd = welch(epoch, fs=SFREQ, nperseg=min(256, len(epoch)))
    delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
    delta_power = float(np.sum(psd[delta_mask])) if np.any(delta_mask) else 0.0

    return {
        "omega1": omega1,
        "torus_curvature": v2["torus_curvature"],
        "geodesic_distance": v2["geodesic_distance"],
        "delta_power": delta_power,
    }


# ===================================================================
# Analysis A — NREM cycle detection
# ===================================================================
def detect_nrem_cycles(labels: list[str]) -> list[list[int]]:
    """Consecutive N1/N2/N3 blocks between REM, ≥10 epochs, ≤5 wake arousals."""
    nrem = {"N1", "N2", "N3"}
    cycles: list[list[int]] = []
    current: list[int] = []
    consec_wake = 0

    for i, lab in enumerate(labels):
        if lab in nrem:
            current.append(i)
            consec_wake = 0
        elif lab == "W":
            consec_wake += 1
            if consec_wake <= 5 and current:
                current.append(i)
            else:
                if len(current) >= 10:
                    cycles.append(current)
                current = []
                consec_wake = 0
        elif lab == "REM":
            if len(current) >= 10:
                cycles.append(current)
            current = []
            consec_wake = 0

    if len(current) >= 10:
        cycles.append(current)
    return cycles


# ===================================================================
# Exponential models
# ===================================================================
def exp_decay(x, a, tau, c):
    return a * np.exp(-np.asarray(x, dtype=float) / tau) + c


def exp_rise(x, a, tau, c):
    return a * np.exp(np.asarray(x, dtype=float) / tau) + c


def fit_exp(x, y, rising: bool = False):
    """Fit exponential. Returns (popt, tau) or (None, None)."""
    fn = exp_rise if rising else exp_decay
    try:
        p0_a = y[-1] - y[0] if rising else y[0] - y[-1]
        p0 = [p0_a, 2.0, float(y[-1]) if not rising else float(y[0])]
        popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=5000)
        return popt, abs(float(popt[1]))
    except (RuntimeError, ValueError):
        return None, None


# ===================================================================
# Analysis C — Lomb-Scargle ultradian periodogram
# ===================================================================
def lomb_scargle_analysis(
    omega1_series: np.ndarray, epoch_sec: float = EPOCH_SEC
) -> tuple[np.ndarray, np.ndarray, float | None, float]:
    """Compute Lomb-Scargle periodogram for overnight ω₁.

    Returns (periods_min, power, dominant_period_min, false_alarm_prob).
    """
    n = len(omega1_series)
    t = np.arange(n) * epoch_sec / 60.0  # time in minutes

    # Frequencies to probe: periods from 30 min to 180 min
    min_period = 30.0   # minutes
    max_period = 180.0
    freqs = np.linspace(1.0 / max_period, 1.0 / min_period, 500)  # cycles/min
    angular_freqs = 2 * np.pi * freqs

    # Normalize signal
    y = omega1_series - np.mean(omega1_series)

    power = lombscargle(t, y, angular_freqs, normalize=True)
    periods = 1.0 / freqs  # in minutes

    # Find peak in 60–120 min range
    mask_ultradian = (periods >= 60) & (periods <= 120)
    if np.any(mask_ultradian):
        peak_idx_local = np.argmax(power[mask_ultradian])
        peak_power = power[mask_ultradian][peak_idx_local]
        peak_period = periods[mask_ultradian][peak_idx_local]

        # Analytic false alarm probability (Baluev 2008 approximation)
        # P(Z > z) ≈ 1 - (1 - exp(-z))^M where M ~ n/2
        M = n / 2.0
        fap = 1.0 - (1.0 - np.exp(-peak_power)) ** M
        fap = max(fap, 0.0)
    else:
        peak_period = None
        fap = 1.0

    return periods, power, peak_period, fap


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "hmc"
    results_dir = PROJECT_ROOT / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Process S Validation — HMC (151 subjects)")
    print("=" * 65)

    pairs = find_subjects(data_dir)
    print(f"  Found {len(pairs)} subjects\n")

    # ------------------------------------------------------------------
    # Load all subjects, extract per-epoch features
    # ------------------------------------------------------------------
    print("Loading subjects & extracting features ...")

    # Per-subject storage
    subj_data: dict[str, dict] = {}
    n_loaded = 0

    for sid, psg, hyp in pairs:
        result = load_subject(psg, hyp)
        if result is None:
            continue

        epochs, labels = result
        n_loaded += 1

        feats_omega1 = []
        feats_curv = []
        feats_geo = []
        feats_delta = []
        valid_labels = []

        for ep, lab in zip(epochs, labels):
            f = extract_epoch_features(ep)
            if f is None:
                continue
            feats_omega1.append(f["omega1"])
            feats_curv.append(f["torus_curvature"])
            feats_geo.append(f["geodesic_distance"])
            feats_delta.append(f["delta_power"])
            valid_labels.append(lab)

        if len(valid_labels) < 20:
            continue

        subj_data[sid] = {
            "labels": valid_labels,
            "omega1": np.array(feats_omega1),
            "torus_curvature": np.array(feats_curv),
            "geodesic_distance": np.array(feats_geo),
            "delta_power": np.array(feats_delta),
        }

        if n_loaded % 20 == 0:
            print(f"  {n_loaded} subjects loaded ...")

    total_epochs = sum(len(d["labels"]) for d in subj_data.values())
    print(f"  {len(subj_data)} subjects, {total_epochs} epochs\n")

    # ------------------------------------------------------------------
    # Analysis A — NREM cycle detection
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Analysis A: NREM cycle detection")
    print("=" * 65)

    all_cycle_omega1: dict[int, list[float]] = defaultdict(list)
    all_cycle_delta: dict[int, list[float]] = defaultdict(list)
    all_cycle_durations_min: list[float] = []
    per_subj_cycles: dict[str, dict] = {}

    pool_cycle_o1: list[float] = []
    pool_cycle_delta: list[float] = []

    for sid, data in subj_data.items():
        cycles = detect_nrem_cycles(data["labels"])
        if len(cycles) < 2:
            continue

        subj_o1, subj_d = [], []
        for c_num, idxs in enumerate(cycles):
            mo = float(np.mean(data["omega1"][idxs]))
            md = float(np.mean(data["delta_power"][idxs]))
            subj_o1.append(mo)
            subj_d.append(md)
            all_cycle_omega1[c_num].append(mo)
            all_cycle_delta[c_num].append(md)
            pool_cycle_o1.append(mo)
            pool_cycle_delta.append(md)
            all_cycle_durations_min.append(len(idxs) * EPOCH_SEC / 60.0)

        per_subj_cycles[sid] = {
            "omega1": subj_o1,
            "delta": subj_d,
            "n_cycles": len(cycles),
        }

    n_with_cycles = len(per_subj_cycles)
    mean_dur = float(np.mean(all_cycle_durations_min)) if all_cycle_durations_min else 0
    print(f"  {n_with_cycles} subjects with ≥2 NREM cycles")
    print(f"  Mean cycle duration: {mean_dur:.1f} min")
    print(f"  Total cycles: {len(pool_cycle_o1)}")

    # ------------------------------------------------------------------
    # Analysis B — Process S decay
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Analysis B: Process S exponential decay")
    print("=" * 65)

    # Group correlation
    r_cycle, p_cycle = pearsonr(pool_cycle_o1, pool_cycle_delta) if len(pool_cycle_o1) > 2 else (0, 1)
    print(f"  Group r(ω₁, δ) at cycle level: {r_cycle:.4f}  p={p_cycle:.2e}")

    # Group-mean curves
    max_c = max(all_cycle_omega1.keys()) + 1 if all_cycle_omega1 else 0
    cycle_x = np.arange(max_c, dtype=float)
    mean_o1 = np.array([np.mean(all_cycle_omega1[c]) for c in range(max_c)])
    mean_delta = np.array([np.mean(all_cycle_delta[c]) for c in range(max_c)])

    # Per-subject exponential fits
    subj_tau_delta: list[float] = []
    subj_tau_omega1: list[float] = []

    for sid, info in per_subj_cycles.items():
        if info["n_cycles"] < 3:
            continue
        x = np.arange(info["n_cycles"], dtype=float)

        # Delta decay
        _, tau_d = fit_exp(x, np.array(info["delta"]), rising=False)
        if tau_d is not None and 0.1 < tau_d < 50:
            subj_tau_delta.append(tau_d)

        # Omega1 rise (inverse of delta)
        _, tau_o = fit_exp(x, np.array(info["omega1"]), rising=True)
        if tau_o is not None and 0.1 < tau_o < 50:
            subj_tau_omega1.append(tau_o)

    mean_cycle_dur_hrs = mean_dur / 60.0

    med_tau_delta_cyc = float(np.median(subj_tau_delta)) if subj_tau_delta else float("nan")
    med_tau_omega1_cyc = float(np.median(subj_tau_omega1)) if subj_tau_omega1 else float("nan")
    med_tau_delta_hrs = med_tau_delta_cyc * mean_cycle_dur_hrs
    med_tau_omega1_hrs = med_tau_omega1_cyc * mean_cycle_dur_hrs

    print(f"  Subjects with ≥3 cycles: {len([s for s in per_subj_cycles.values() if s['n_cycles'] >= 3])}")
    print(f"  Median τ(δ):  {med_tau_delta_cyc:.2f} cycles = {med_tau_delta_hrs:.1f} h"
          f"  (n={len(subj_tau_delta)})")
    print(f"  Median τ(ω₁): {med_tau_omega1_cyc:.2f} cycles = {med_tau_omega1_hrs:.1f} h"
          f"  (n={len(subj_tau_omega1)})")
    print(f"  Known Process S τ_fall ≈ 4.2 h")

    # Group-level exponential fit for plot
    group_popt_delta, group_tau_delta = fit_exp(
        cycle_x[:len(mean_delta)], mean_delta, rising=False
    ) if len(mean_delta) >= 3 else (None, None)
    group_popt_o1, group_tau_o1 = fit_exp(
        cycle_x[:len(mean_o1)], mean_o1, rising=True
    ) if len(mean_o1) >= 3 else (None, None)

    # --- Figure: Process S decay ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: ω₁ per cycle
    ax = axes[0]
    for info in per_subj_cycles.values():
        ax.plot(range(len(info["omega1"])), info["omega1"],
                "o-", alpha=0.07, ms=2, color="gray")
    ax.plot(cycle_x[:len(mean_o1)], mean_o1, "s-", color="#5bffa8",
            lw=2.5, ms=8, label="Group mean", zorder=5)
    if group_popt_o1 is not None:
        xf = np.linspace(0, max_c - 1, 200)
        ax.plot(xf, exp_rise(xf, *group_popt_o1), "--", color="red", lw=1.5,
                label=f"τ = {group_tau_o1:.1f} cyc")
    ax.set_xlabel("NREM cycle #", fontsize=11)
    ax.set_ylabel("Mean ω₁", fontsize=11)
    ax.set_title("ω₁ across NREM cycles (HMC, n=%d)" % n_with_cycles, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: scatter
    ax = axes[1]
    ax.scatter(pool_cycle_delta, pool_cycle_o1, alpha=0.3, c="#5bffa8",
               edgecolors="#333", s=20)
    ax.set_xlabel("Mean δ-power (absolute)", fontsize=11)
    ax.set_ylabel("Mean ω₁", fontsize=11)
    ax.set_title(f"ω₁ vs δ-power  (r = {r_cycle:.3f})", fontsize=13)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "hmc_process_s_decay.png", dpi=150)
    plt.close(fig)
    print("  Saved hmc_process_s_decay.png")

    # ------------------------------------------------------------------
    # Analysis C — Ultradian 90-min cycle from ω₁ periodogram
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Analysis C: Ultradian cycle detection (Lomb-Scargle)")
    print("=" * 65)

    all_periods_at_peak: list[float] = []
    all_faps: list[float] = []
    accumulated_power = None
    n_pgram = 0

    for sid, data in subj_data.items():
        o1 = data["omega1"]
        if len(o1) < 60:  # need ≥30 min
            continue

        periods, power, peak_period, fap = lomb_scargle_analysis(o1)

        if accumulated_power is None:
            accumulated_power = np.zeros_like(power)
        accumulated_power += power
        n_pgram += 1

        if peak_period is not None:
            all_periods_at_peak.append(peak_period)
            all_faps.append(fap)

    if n_pgram > 0:
        group_power = accumulated_power / n_pgram
    else:
        group_power = np.zeros(1)
        periods = np.array([90.0])

    # Significant peaks (FAP < 0.05) in 70–110 min range
    sig_peaks = [p for p, f in zip(all_periods_at_peak, all_faps)
                 if f < 0.05 and 70 <= p <= 110]
    frac_sig = len(sig_peaks) / len(subj_data) if subj_data else 0

    # Group periodogram dominant period in 60–120 range
    mask_ultra = (periods >= 60) & (periods <= 120)
    if np.any(mask_ultra) and n_pgram > 0:
        group_peak_idx = np.argmax(group_power[mask_ultra])
        group_dominant_period = float(periods[mask_ultra][group_peak_idx])
    else:
        group_dominant_period = float("nan")

    print(f"  Subjects with periodograms: {n_pgram}")
    print(f"  Fraction with significant 70–110 min peak (FAP<0.05): "
          f"{len(sig_peaks)}/{len(subj_data)} = {frac_sig:.1%}")
    print(f"  Group-average dominant period: {group_dominant_period:.1f} min")

    if all_periods_at_peak:
        med_peak = float(np.median(all_periods_at_peak))
        print(f"  Median per-subject peak period: {med_peak:.1f} min")
    else:
        med_peak = float("nan")

    # --- Figure: periodogram ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # Top: group average periodogram
    ax = axes[0]
    ax.plot(periods, group_power, color="#5bffa8", lw=1.5)
    ax.axvline(90, ls="--", color="red", alpha=0.7, label="90 min")
    if np.isfinite(group_dominant_period):
        ax.axvline(group_dominant_period, ls=":", color="yellow", alpha=0.7,
                    label=f"Peak: {group_dominant_period:.0f} min")
    ax.set_xlabel("Period (minutes)", fontsize=11)
    ax.set_ylabel("Normalized power", fontsize=11)
    ax.set_title("Group-average Lomb-Scargle periodogram of overnight ω₁",
                 fontsize=13)
    ax.set_xlim(30, 180)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom: histogram of per-subject dominant periods
    ax = axes[1]
    peaks_in_range = [p for p in all_periods_at_peak if 60 <= p <= 120]
    if peaks_in_range:
        ax.hist(peaks_in_range, bins=20, range=(60, 120), color="#5bffa8",
                edgecolor="white", alpha=0.8)
    ax.axvline(90, ls="--", color="red", alpha=0.7, label="90 min")
    ax.set_xlabel("Dominant period in 60–120 min range (minutes)", fontsize=11)
    ax.set_ylabel("# subjects", fontsize=11)
    ax.set_title(f"Per-subject dominant periods (n={len(peaks_in_range)})",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "hmc_ultradian_periodogram.png", dpi=150)
    plt.close(fig)
    print("  Saved hmc_ultradian_periodogram.png")

    # ------------------------------------------------------------------
    # Analysis D — Epoch-level ω₁ vs delta per subject
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Analysis D: Epoch-level ω₁ vs δ correlation")
    print("=" * 65)

    per_subj_r: list[float] = []
    for sid, data in subj_data.items():
        if len(data["omega1"]) < 20:
            continue
        r, _ = pearsonr(data["omega1"], data["delta_power"])
        if np.isfinite(r):
            per_subj_r.append(float(r))

    r_arr = np.array(per_subj_r)
    print(f"  Subjects: {len(r_arr)}")
    print(f"  Mean r:   {np.mean(r_arr):.4f}")
    print(f"  Std:      {np.std(r_arr):.4f}")
    print(f"  Median:   {np.median(r_arr):.4f}")
    print(f"  Range:    [{np.min(r_arr):.4f}, {np.max(r_arr):.4f}]")
    print(f"  Fraction r < 0: {np.mean(r_arr < 0):.1%}")

    # --- Figure: histogram of per-subject r ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(r_arr, bins=40, range=(-1, 1), color="#5bffa8", edgecolor="white",
            alpha=0.8)
    ax.axvline(np.mean(r_arr), ls="--", color="red", lw=2,
               label=f"Mean = {np.mean(r_arr):.3f}")
    ax.axvline(0, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("Pearson r(ω₁, δ-power) per subject", fontsize=12)
    ax.set_ylabel("# subjects", fontsize=12)
    ax.set_title(f"Epoch-level ω₁ vs δ correlation (n={len(r_arr)} subjects)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "hmc_epoch_correlation_hist.png", dpi=150)
    plt.close(fig)
    print("  Saved hmc_epoch_correlation_hist.png")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output = {
        "dataset": "HMC",
        "n_subjects": len(subj_data),
        "n_epochs": total_epochs,
        "analysis_a": {
            "n_subjects_with_cycles": n_with_cycles,
            "total_cycles": len(pool_cycle_o1),
            "mean_cycle_duration_min": round(mean_dur, 1),
        },
        "analysis_b": {
            "cycle_level_pearson_r": round(float(r_cycle), 4),
            "cycle_level_pearson_p": float(p_cycle),
            "median_tau_delta_cycles": round(med_tau_delta_cyc, 2),
            "median_tau_delta_hours": round(med_tau_delta_hrs, 1),
            "n_fits_delta": len(subj_tau_delta),
            "median_tau_omega1_cycles": round(med_tau_omega1_cyc, 2),
            "median_tau_omega1_hours": round(med_tau_omega1_hrs, 1),
            "n_fits_omega1": len(subj_tau_omega1),
            "group_tau_delta_cycles": round(float(group_tau_delta), 2) if group_tau_delta else None,
            "group_tau_omega1_cycles": round(float(group_tau_o1), 2) if group_tau_o1 else None,
        },
        "analysis_c": {
            "n_periodograms": n_pgram,
            "frac_sig_90min_peak": round(frac_sig, 3),
            "n_sig_peaks": len(sig_peaks),
            "group_dominant_period_min": round(group_dominant_period, 1),
            "median_subject_peak_min": round(med_peak, 1) if np.isfinite(med_peak) else None,
        },
        "analysis_d": {
            "n_subjects": len(r_arr),
            "epoch_r_mean": round(float(np.mean(r_arr)), 4),
            "epoch_r_std": round(float(np.std(r_arr)), 4),
            "epoch_r_median": round(float(np.median(r_arr)), 4),
            "epoch_r_min": round(float(np.min(r_arr)), 4),
            "epoch_r_max": round(float(np.max(r_arr)), 4),
            "frac_negative": round(float(np.mean(r_arr < 0)), 3),
        },
    }

    out_path = results_dir / "hmc_process_s.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Group r(ω₁, δ) cycle-level:    {r_cycle:.4f}  (p={p_cycle:.2e})")
    print(f"  Group r(ω₁, δ) epoch-level:    {np.mean(r_arr):.4f}  (mean of {len(r_arr)} subjects)")
    print(f"  Median τ_decay(δ):             {med_tau_delta_hrs:.1f} h  (known ≈ 4.2 h)")
    print(f"  Median τ_decay(ω₁):            {med_tau_omega1_hrs:.1f} h")
    print(f"  Frac. with sig. 90-min peak:   {frac_sig:.1%}  ({len(sig_peaks)}/{len(subj_data)})")
    print(f"  Dominant ultradian period:      {group_dominant_period:.1f} min")
    print(f"\n  Results -> {out_path}")
    print(f"  Figures -> {fig_dir}/hmc_process_s_*.png, hmc_ultradian_*.png, hmc_epoch_*.png")


if __name__ == "__main__":
    main()
