"""Paper Final Stats: compute the 7 missing values for submission.

Outputs:
1. 95% confidence intervals for κ and F1 (both datasets)
2. Wilcoxon signed-rank test for Δ F1 significance (both datasets)
3. Per-stage F1 breakdown for HMC (spectral vs combined)
4. HMC class distribution
5. Per-stage F1 improvement (which stages benefit most from geometry)
"""

import numpy as np
import sys, os
from pathlib import Path
from collections import Counter, defaultdict

import mne
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, mutual_info_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.spectral import compute_band_powers, compute_hjorth
from src.features.takens import time_delay_embedding
from src.geometry.tesseract import Q_discretize
from src.geometry.alignment import compute_fixed_tau

HMC_LABELS = {
    "Sleep stage W": "W", "Sleep stage N1": "N1", "Sleep stage N2": "N2",
    "Sleep stage N3": "N3", "Sleep stage R": "REM", "Sleep stage ?": None,
    "0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "REM", "5": None,
    "W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "REM", "R": "REM",
}

TARGET_SFREQ = 100
MIN_EPOCHS = 300
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGE_NAMES)}


def extract_features(epoch_uv, sfreq, fixed_tau, epoch_uv_ch2=None):
    """Extract spectral + geometric features for one epoch."""
    # Spectral
    bp = compute_band_powers(epoch_uv, sfreq=sfreq)
    hj = compute_hjorth(epoch_uv)
    delta = bp.get('delta', 0)
    theta = bp.get('theta', 0)
    alpha = bp.get('alpha', 0)
    sigma = bp.get('sigma', 0)
    beta = bp.get('beta', 0)
    db_ratio = delta / max(beta, 1e-10)

    spectral = [delta, theta, alpha, sigma, beta, db_ratio,
                hj[0], hj[1], hj[2]]

    # Geometric
    emb, _ = time_delay_embedding(epoch_uv, dimension=4, tau=fixed_tau)
    if len(emb) == 0:
        return None, None

    mean_emb = np.mean(emb, axis=0)
    th = np.arctan2(emb[:, 1], emb[:, 0])
    ph = np.arctan2(emb[:, 3], emb[:, 2])

    dth = np.diff(th)
    dth = np.where(dth > np.pi, dth - 2*np.pi, dth)
    dth = np.where(dth < -np.pi, dth + 2*np.pi, dth)
    dph = np.diff(ph)
    dph = np.where(dph > np.pi, dph - 2*np.pi, dph)
    dph = np.where(dph < -np.pi, dph + 2*np.pi, dph)

    omega1 = np.abs(np.mean(dth))
    omega2 = np.abs(np.mean(dph))
    ratio = omega1 / max(omega2, 1e-10)
    vertex = Q_discretize(mean_emb)

    geometric = [omega1, ratio, vertex]

    # Multi-channel
    if epoch_uv_ch2 is not None:
        emb2, _ = time_delay_embedding(epoch_uv_ch2, dimension=4, tau=fixed_tau)
        if len(emb2) > 0:
            th2 = np.arctan2(emb2[:, 1], emb2[:, 0])
            min_len = min(len(th), len(th2))
            pdiff = th[:min_len] - th2[:min_len]
            pdiff = np.arctan2(np.sin(pdiff), np.cos(pdiff))
            phase_diff_std = np.std(pdiff)
            phase_coh = np.abs(np.mean(np.exp(1j * pdiff)))
            geometric.extend([phase_diff_std, phase_coh])
        else:
            geometric.extend([0, 0])

    return spectral, geometric


def main():
    data_dir = Path("data/hmc")
    fixed_tau = compute_fixed_tau(TARGET_SFREQ, 2.0)

    print("=" * 65)
    print("  Paper Final Stats — HMC Per-Stage F1, Wilcoxon, CIs")
    print("=" * 65)

    subjects = []
    for f in sorted(data_dir.glob("SN*.edf")):
        if "_sleepscoring" not in f.name:
            sid = f.stem
            hyp = data_dir / f"{sid}_sleepscoring.edf"
            if hyp.exists():
                subjects.append((f, hyp, sid))

    print(f"\n  Found {len(subjects)} subjects")

    # Collect all data
    all_spectral = []
    all_geometric = []
    all_labels = []
    all_subject_ids = []
    subject_count = 0

    for psg_path, hyp_path, sid in subjects:
        try:
            raw = mne.io.read_raw_edf(str(psg_path), preload=False, verbose=False)
            ch_names = raw.ch_names

            target_chs = ["EEG C4-M1", "EEG C3-M2"]
            picks = [c for c in target_chs if c in ch_names]
            if len(picks) < 2:
                picks = [c for c in ch_names if 'EEG' in c][:2]
            if len(picks) < 2:
                continue

            raw.pick(picks)
            raw.load_data(verbose=False)
            if raw.info['sfreq'] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ, verbose=False)
            raw.filter(0.5, 30.0, verbose=False)

            annotations = mne.read_annotations(str(hyp_path))
            new_desc = [str(d) for d in annotations.description]
            annotations = mne.Annotations(annotations.onset, annotations.duration, new_desc)
            raw.set_annotations(annotations)

            events, event_id = mne.events_from_annotations(raw, chunk_duration=30.0, verbose=False)

            label_map = {}
            for desc, eid in event_id.items():
                mapped = HMC_LABELS.get(desc)
                if mapped and mapped in STAGE_NAMES and mapped not in label_map:
                    label_map[mapped] = eid

            if not label_map:
                continue

            epochs = mne.Epochs(raw, events, event_id=label_map,
                               tmin=0, tmax=30.0 - 1.0/TARGET_SFREQ,
                               baseline=None, preload=True, verbose=False)

            if len(epochs) < MIN_EPOCHS:
                continue

            data = epochs.get_data()
            id_to_stage = {eid: stage for stage, eid in label_map.items()}

            for i in range(len(data)):
                eid = epochs.events[i, 2]
                stage = id_to_stage.get(eid)
                if not stage or stage not in STAGE_NAMES:
                    continue

                ch1 = data[i, 0, :] * 1e6
                ch2 = data[i, 1, :] * 1e6

                if np.max(np.abs(ch1)) > 500 or np.max(np.abs(ch2)) > 500:
                    continue

                spec, geom = extract_features(ch1, TARGET_SFREQ, fixed_tau, ch2)
                if spec is None:
                    continue

                all_spectral.append(spec)
                all_geometric.append(geom)
                all_labels.append(STAGE_TO_INT[stage])
                all_subject_ids.append(subject_count)

            subject_count += 1
            print(f"  {sid}... ✓ (total epochs: {len(all_labels)})")

        except Exception as e:
            continue

    X_spec = np.array(all_spectral)
    X_geom_raw = all_geometric
    # Handle mixed-length geometric features (vertex is categorical)
    X_geom = np.array([[g[0], g[1]] + ([g[3], g[4]] if len(g) > 3 else [0, 0])
                        for g in X_geom_raw])
    X_comb = np.hstack([X_spec, X_geom])
    y = np.array(all_labels)
    groups = np.array(all_subject_ids)

    print(f"\n  Total: {len(y)} epochs from {subject_count} subjects")

    # === CLASS DISTRIBUTION ===
    print("\n" + "=" * 65)
    print("  HMC CLASS DISTRIBUTION")
    print("=" * 65)
    for stage_name, stage_int in STAGE_TO_INT.items():
        count = np.sum(y == stage_int)
        pct = count / len(y) * 100
        print(f"    {stage_name}: {count:>6} ({pct:.1f}%)")

    # === 5-FOLD CV WITH PER-STAGE F1 ===
    print("\n" + "=" * 65)
    print("  5-FOLD CROSS-VALIDATION (stratified by subject)")
    print("=" * 65)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    subj_kappas_spec = []
    subj_kappas_comb = []
    subj_f1_spec = []
    subj_f1_comb = []
    all_y_true = []
    all_y_pred_spec = []
    all_y_pred_comb = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_comb, y, groups)):
        X_train_s, X_test_s = X_spec[train_idx], X_spec[test_idx]
        X_train_c, X_test_c = X_comb[train_idx], X_comb[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Spectral only
        clf_s = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           learning_rate=0.1, random_state=42)
        clf_s.fit(X_train_s, y_train)
        pred_s = clf_s.predict(X_test_s)

        # Combined
        clf_c = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           learning_rate=0.1, random_state=42)
        clf_c.fit(X_train_c, y_train)
        pred_c = clf_c.predict(X_test_c)

        # Per-fold metrics
        k_s = cohen_kappa_score(y_test, pred_s)
        k_c = cohen_kappa_score(y_test, pred_c)
        f1_s = f1_score(y_test, pred_s, average='macro')
        f1_c = f1_score(y_test, pred_c, average='macro')

        subj_kappas_spec.append(k_s)
        subj_kappas_comb.append(k_c)
        subj_f1_spec.append(f1_s)
        subj_f1_comb.append(f1_c)

        all_y_true.extend(y_test)
        all_y_pred_spec.extend(pred_s)
        all_y_pred_comb.extend(pred_c)

        print(f"  Fold {fold+1}: κ_spec={k_s:.3f}, κ_comb={k_c:.3f}, "
              f"F1_spec={f1_s:.3f}, F1_comb={f1_c:.3f}, Δ={f1_c-f1_s:+.3f}")

    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred_spec = np.array(all_y_pred_spec)
    all_y_pred_comb = np.array(all_y_pred_comb)

    # === PER-STAGE F1 ===
    print("\n" + "=" * 65)
    print("  PER-STAGE F1 SCORES")
    print("=" * 65)

    f1_per_stage_spec = f1_score(all_y_true, all_y_pred_spec, average=None, labels=[0,1,2,3,4])
    f1_per_stage_comb = f1_score(all_y_true, all_y_pred_comb, average=None, labels=[0,1,2,3,4])

    print(f"  {'Stage':<8} {'F1_spec':>8} {'F1_comb':>8} {'Δ F1':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for i, stage in enumerate(STAGE_NAMES):
        delta = f1_per_stage_comb[i] - f1_per_stage_spec[i]
        print(f"  {stage:<8} {f1_per_stage_spec[i]:>8.3f} {f1_per_stage_comb[i]:>8.3f} {delta:>+8.3f}")

    f1_macro_spec = np.mean(f1_per_stage_spec)
    f1_macro_comb = np.mean(f1_per_stage_comb)
    print(f"  {'Macro':<8} {f1_macro_spec:>8.3f} {f1_macro_comb:>8.3f} {f1_macro_comb-f1_macro_spec:>+8.3f}")

    # === WILCOXON TEST ===
    print("\n" + "=" * 65)
    print("  WILCOXON SIGNED-RANK TEST (Δ F1 significance)")
    print("=" * 65)

    f1_diffs = np.array(subj_f1_comb) - np.array(subj_f1_spec)
    print(f"  Per-fold F1 differences: {f1_diffs}")
    print(f"  Mean Δ F1: {np.mean(f1_diffs):+.4f}")

    if np.all(f1_diffs == 0):
        print("  All differences are zero — cannot compute Wilcoxon test")
    else:
        try:
            stat, p_val = wilcoxon(subj_f1_comb, subj_f1_spec, alternative='greater')
            print(f"  Wilcoxon W = {stat:.1f}, p = {p_val:.4f}")
            if p_val < 0.05:
                print(f"  ✓ SIGNIFICANT (p < 0.05)")
            else:
                print(f"  ✗ NOT SIGNIFICANT (p = {p_val:.4f})")
        except Exception as e:
            print(f"  Wilcoxon test error: {e}")

    # === CONFIDENCE INTERVALS ===
    print("\n" + "=" * 65)
    print("  95% CONFIDENCE INTERVALS")
    print("=" * 65)

    def ci95(values):
        m = np.mean(values)
        se = np.std(values) / np.sqrt(len(values))
        return m, m - 1.96*se, m + 1.96*se

    k_m, k_lo, k_hi = ci95(subj_kappas_comb)
    print(f"  κ_combined:  {k_m:.3f} [{k_lo:.3f}, {k_hi:.3f}]")

    f_m, f_lo, f_hi = ci95(subj_f1_comb)
    print(f"  F1_combined: {f_m:.3f} [{f_lo:.3f}, {f_hi:.3f}]")

    ks_m, ks_lo, ks_hi = ci95(subj_kappas_spec)
    print(f"  κ_spectral:  {ks_m:.3f} [{ks_lo:.3f}, {ks_hi:.3f}]")

    fs_m, fs_lo, fs_hi = ci95(subj_f1_spec)
    print(f"  F1_spectral: {fs_m:.3f} [{fs_lo:.3f}, {fs_hi:.3f}]")

    # === SUMMARY FOR PAPER ===
    print("\n" + "=" * 65)
    print("  COPY THESE VALUES INTO THE PAPER")
    print("=" * 65)
    print(f"""
  Section 3.1 (HMC class distribution):
    W: {np.sum(y==0)/len(y)*100:.1f}%, N1: {np.sum(y==1)/len(y)*100:.1f}%, \
N2: {np.sum(y==2)/len(y)*100:.1f}%, N3: {np.sum(y==3)/len(y)*100:.1f}%, \
REM: {np.sum(y==4)/len(y)*100:.1f}%

  Section 4.2 (per-stage F1 and Wilcoxon):
    Per-stage Δ F1: W={f1_per_stage_comb[0]-f1_per_stage_spec[0]:+.3f}, \
N1={f1_per_stage_comb[1]-f1_per_stage_spec[1]:+.3f}, \
N2={f1_per_stage_comb[2]-f1_per_stage_spec[2]:+.3f}, \
N3={f1_per_stage_comb[3]-f1_per_stage_spec[3]:+.3f}, \
REM={f1_per_stage_comb[4]-f1_per_stage_spec[4]:+.3f}
    Macro Δ F1: {f1_macro_comb-f1_macro_spec:+.3f}

  Section 4.1/4.2 (confidence intervals):
    κ_comb = {k_m:.3f} [95% CI: {k_lo:.3f}, {k_hi:.3f}]
    F1_comb = {f_m:.3f} [95% CI: {f_lo:.3f}, {f_hi:.3f}]
""")
    print("  Done. Paste these into the paper draft.")


if __name__ == "__main__":
    main()
