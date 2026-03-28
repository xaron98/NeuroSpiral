#!/usr/bin/env python3
"""Temporal context test: does adding neighboring epochs close the
information-exploitation gap from Paper #1?

Tests 5 conditions with 5-fold subject-stratified CV on HMC:
  1. Spectral only, no context (8 features)
  2. Spectral + geometric, no context (16 features)
  3. Spectral only, 5-epoch context (40 features)
  4. Spectral + geometric, 5-epoch context (80 features)
  5. Geometric only, 5-epoch context (40 features)
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
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

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
CONTEXT_HALF = 2  # epochs on each side → 5-epoch window

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

# 8 spectral feature names
SPECTRAL_NAMES = [
    "delta", "theta", "alpha", "sigma", "beta",
    "delta_beta", "hjorth_activity", "hjorth_mobility",
]

# 8 geometric feature names
GEOM_NAMES = [
    "omega1", "torus_curvature", "angular_acceleration",
    "geodesic_distance", "angular_entropy", "phase_diff_std",
    "phase_coherence", "transition_rate",
]

N_SPECTRAL = len(SPECTRAL_NAMES)   # 8
N_GEOM = len(GEOM_NAMES)           # 8
N_TOTAL = N_SPECTRAL + N_GEOM      # 16


# ===================================================================
# Subject discovery + loading
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


def load_subject(psg_path: Path, hyp_path: Path):
    """Return (epochs_array, labels) or None."""
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    ch = None
    for c in ["EEG C4-M1", "C4-M1", "EEG C4"]:
        if c in raw.ch_names:
            ch = c
            break
    if ch is None:
        eeg = [c for c in raw.ch_names if "EEG" in c.upper()]
        ch = eeg[0] if eeg else None
    if ch is None:
        return None

    raw.pick([ch])
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)
    raw.filter(0.5, 30.0, verbose=False)

    signal = raw.get_data()[0]
    n_ep = int(len(signal) / SFREQ // EPOCH_SEC)

    annots = mne.read_annotations(str(hyp_path))
    labels: list[str | None] = [None] * n_ep
    for onset, dur, desc in zip(annots.onset, annots.duration, annots.description):
        stage = HMC_LABELS.get(str(desc).strip())
        if stage is None:
            continue
        s = int(onset // EPOCH_SEC)
        nd = max(1, int(dur // EPOCH_SEC))
        for e in range(s, min(s + nd, n_ep)):
            labels[e] = stage

    epochs_out, labels_out = [], []
    for i in range(n_ep):
        if labels[i] is None:
            continue
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        if end > len(signal):
            break
        ep = signal[start:end]
        if np.max(np.abs(ep)) > 500e-6:
            continue
        epochs_out.append(ep)
        labels_out.append(labels[i])

    if not epochs_out:
        return None
    return np.array(epochs_out), labels_out


# ===================================================================
# Feature extraction (16 features per epoch)
# ===================================================================
def _wrap(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


def extract_features(epoch: np.ndarray) -> np.ndarray | None:
    """Return (16,) vector: 8 spectral + 8 geometric, or None."""
    # --- Spectral (8) ---
    freqs, psd = welch(epoch, fs=SFREQ, nperseg=min(256, len(epoch)))
    total = np.trapz(psd, freqs)
    if total < 1e-20:
        return None

    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
             "sigma": (12, 15), "beta": (15, 30)}
    bp = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        bp[name] = float(np.trapz(psd[mask], freqs[mask]) / total)

    delta_beta = bp["delta"] / (bp["beta"] + 1e-10)

    diff1 = np.diff(epoch)
    diff2 = np.diff(diff1)
    activity = float(np.var(epoch))
    mobility = float(np.sqrt(np.var(diff1) / (activity + 1e-10)))

    spec = np.array([
        bp["delta"], bp["theta"], bp["alpha"], bp["sigma"], bp["beta"],
        delta_beta, activity, mobility,
    ])

    # --- Geometric (8) ---
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

    phase_diff = theta_uw - phi_uw
    phase_diff_std = float(np.std(np.diff(phase_diff)))

    diff_w = _wrap(theta - phi)
    mc = float(np.mean(np.cos(diff_w)))
    ms = float(np.mean(np.sin(diff_w)))
    phase_coherence = float(np.sqrt(mc**2 + ms**2))

    signs = (emb >= 0).astype(int)
    verts = signs[:, 0] * 8 + signs[:, 1] * 4 + signs[:, 2] * 2 + signs[:, 3]
    transition_rate = float(np.sum(np.diff(verts) != 0) / max(len(verts) - 1, 1))

    v2 = extract_torus_features_v2(emb)

    geom = np.array([
        omega1,
        v2["torus_curvature"],
        v2["angular_acceleration"],
        v2["geodesic_distance"],
        v2["angular_entropy"],
        phase_diff_std,
        phase_coherence,
        transition_rate,
    ])

    combined = np.concatenate([spec, geom])
    if not np.all(np.isfinite(combined)):
        return None
    return combined


# ===================================================================
# Temporal context expansion
# ===================================================================
def build_context(features: np.ndarray, half: int = CONTEXT_HALF) -> np.ndarray:
    """Expand (n, d) -> (n, d*(2*half+1)) using boundary replication."""
    n, d = features.shape
    window = 2 * half + 1
    out = np.zeros((n, d * window))

    for i in range(n):
        parts = []
        for offset in range(-half, half + 1):
            j = max(0, min(n - 1, i + offset))
            parts.append(features[j])
        out[i] = np.concatenate(parts)

    return out


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "hmc"
    results_dir = PROJECT_ROOT / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Temporal Context Test — HMC dataset")
    print("=" * 70)

    # --- Load all subjects ---
    pairs = find_subjects(data_dir)
    print(f"  Found {len(pairs)} subjects\n")

    # Per-subject: features (n_epochs, 16), labels, subject_id
    subj_features: dict[str, np.ndarray] = {}
    subj_labels: dict[str, np.ndarray] = {}

    n_loaded = 0
    for sid, psg, hyp in pairs:
        result = load_subject(psg, hyp)
        if result is None:
            continue

        epochs, labels = result
        feat_list = []
        lab_list = []
        for ep, lab in zip(epochs, labels):
            f = extract_features(ep)
            if f is None:
                continue
            feat_list.append(f)
            lab_list.append(STAGE_TO_INT[lab])

        if len(feat_list) < 20:
            continue

        subj_features[sid] = np.array(feat_list)
        subj_labels[sid] = np.array(lab_list)
        n_loaded += 1
        if n_loaded % 20 == 0:
            print(f"    {n_loaded} subjects loaded ...")

    print(f"  {n_loaded} subjects, "
          f"{sum(len(v) for v in subj_labels.values())} epochs\n")

    # --- Build per-subject context features ---
    print("  Building temporal context windows ...")
    subj_ctx: dict[str, np.ndarray] = {}
    for sid, feats in subj_features.items():
        subj_ctx[sid] = build_context(feats, CONTEXT_HALF)

    # --- Assemble global arrays ---
    sids = sorted(subj_features.keys())

    all_feats = np.vstack([subj_features[s] for s in sids])       # (N, 16)
    all_ctx = np.vstack([subj_ctx[s] for s in sids])              # (N, 80)
    all_y = np.concatenate([subj_labels[s] for s in sids])        # (N,)
    all_groups = np.concatenate([
        np.full(len(subj_labels[s]), i) for i, s in enumerate(sids)
    ])                                                              # (N,) subject id

    N = len(all_y)
    print(f"  Total: {N} epochs, {len(sids)} subjects\n")

    # Feature slices
    spec_idx = slice(0, N_SPECTRAL)               # 0:8
    geom_idx = slice(N_SPECTRAL, N_TOTAL)          # 8:16
    spec_ctx_idx = np.concatenate([
        np.arange(i * N_TOTAL, i * N_TOTAL + N_SPECTRAL)
        for i in range(2 * CONTEXT_HALF + 1)
    ])  # spectral features from all 5 window positions
    geom_ctx_idx = np.concatenate([
        np.arange(i * N_TOTAL + N_SPECTRAL, (i + 1) * N_TOTAL)
        for i in range(2 * CONTEXT_HALF + 1)
    ])  # geometric features from all 5 window positions

    # --- Define 5 conditions ---
    conditions = [
        ("Spec only (8)",           all_feats[:, spec_idx]),
        ("Spec+Geom (16)",          all_feats),
        ("Spec ctx (40)",           all_ctx[:, spec_ctx_idx]),
        ("Spec+Geom ctx (80)",      all_ctx),
        ("Geom ctx (40)",           all_ctx[:, geom_ctx_idx]),
    ]

    # --- 5-fold subject-stratified CV ---
    print("=" * 70)
    print("5-fold subject-stratified cross-validation")
    print("=" * 70)

    # Use StratifiedGroupKFold: stratify by label, group by subject
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(all_feats, all_y, groups=all_groups))

    results: dict[str, dict] = {}

    for cond_name, X_cond in conditions:
        print(f"\n  {cond_name} ({X_cond.shape[1]}d) ...")

        fold_accs, fold_kappas, fold_f1ms = [], [], []
        fold_f1_per = []  # list of arrays

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            X_tr = np.nan_to_num(X_cond[train_idx])
            y_tr = all_y[train_idx]
            X_te = np.nan_to_num(X_cond[test_idx])
            y_te = all_y[test_idx]

            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)

            fold_accs.append(accuracy_score(y_te, pred))
            fold_kappas.append(cohen_kappa_score(y_te, pred))
            fold_f1ms.append(f1_score(y_te, pred, average="macro"))
            fold_f1_per.append(f1_score(y_te, pred, average=None, labels=list(range(5))))

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
        print(f"    Acc={mean_acc:.3f}  k={mean_kappa:.3f}  F1m={mean_f1m:.3f}  [{per_stage_str}]")

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
    ax.set_title("Temporal context test: kappa and F1 across conditions", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar_set in [bars1, bars2]:
        for bar in bar_set:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "temporal_context_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved temporal_context_comparison.png")

    # --- Save JSON ---
    out_path = results_dir / "temporal_context.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results -> {out_path}")


if __name__ == "__main__":
    main()
