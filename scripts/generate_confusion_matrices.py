"""Generate confusion matrices for the paper.
Uses the same pipeline as paper_final_stats.py (with hjorth tuple fix).
Outputs: confusion_matrix_hmc.png (for paper Figure)
"""

import numpy as np
import sys, os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mne
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.spectral import compute_band_powers, compute_hjorth
from src.features.takens import time_delay_embedding
from src.geometry.tesseract import Q_discretize
from src.geometry.alignment import compute_fixed_tau

HMC_LABELS = {
    "Sleep stage W": "W", "Sleep stage N1": "N1", "Sleep stage N2": "N2",
    "Sleep stage N3": "N3", "Sleep stage R": "REM", "Sleep stage ?": None,
}
TARGET_SFREQ = 100
MIN_EPOCHS = 300
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGE_NAMES)}


def extract_features(epoch_uv, sfreq, fixed_tau, epoch_uv_ch2=None):
    bp = compute_band_powers(epoch_uv, sfreq=sfreq)
    hj = compute_hjorth(epoch_uv)
    delta = bp.get('delta', 0)
    beta = bp.get('beta', 0)
    db_ratio = delta / max(beta, 1e-10)
    spectral = [delta, bp.get('theta', 0), bp.get('alpha', 0),
                bp.get('sigma', 0), beta, db_ratio,
                hj[0], hj[1], hj[2]]  # FIXED: tuple not dict

    emb, _ = time_delay_embedding(epoch_uv, dimension=4, tau=fixed_tau)
    if len(emb) == 0:
        return None, None

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
    geometric = [omega1, ratio]

    if epoch_uv_ch2 is not None:
        emb2, _ = time_delay_embedding(epoch_uv_ch2, dimension=4, tau=fixed_tau)
        if len(emb2) > 0:
            th2 = np.arctan2(emb2[:, 1], emb2[:, 0])
            min_len = min(len(th), len(th2))
            pdiff = th[:min_len] - th2[:min_len]
            pdiff = np.arctan2(np.sin(pdiff), np.cos(pdiff))
            geometric.extend([np.std(pdiff), np.abs(np.mean(np.exp(1j * pdiff)))])
        else:
            geometric.extend([0, 0])

    return spectral, geometric


def plot_confusion_matrix(cm, title, filename, normalize=True):
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1 if normalize else None)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(STAGE_NAMES, fontsize=12)
    ax.set_yticklabels(STAGE_NAMES, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('True', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')

    for i in range(5):
        for j in range(5):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = 'white' if val > 0.5 else 'black'
            if normalize:
                ax.text(j, i, f'{val:.2f}\n({count})',
                        ha='center', va='center', color=color, fontsize=9)
            else:
                ax.text(j, i, f'{count}',
                        ha='center', va='center', color=color, fontsize=10)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    data_dir = Path("data/hmc")
    fixed_tau = compute_fixed_tau(TARGET_SFREQ, 2.0)

    print("=" * 60)
    print("  Confusion Matrix Generation — HMC Dataset")
    print("=" * 60)

    subjects = []
    for f in sorted(data_dir.glob("SN*.edf")):
        if "_sleepscoring" not in f.name:
            sid = f.stem
            hyp = data_dir / f"{sid}_sleepscoring.edf"
            if hyp.exists():
                subjects.append((f, hyp, sid))

    print(f"  Found {len(subjects)} subjects")

    all_spectral, all_geometric, all_labels, all_subject_ids = [], [], [], []
    subject_count = 0

    for psg_path, hyp_path, sid in subjects:
        try:
            raw = mne.io.read_raw_edf(str(psg_path), preload=False, verbose=False)
            target_chs = ["EEG C4-M1", "EEG C3-M2"]
            picks = [c for c in target_chs if c in raw.ch_names]
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
                stage = HMC_LABELS.get(str(desc))
                if stage and stage in STAGE_NAMES and stage not in label_map:
                    label_map[stage] = eid
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
                stage = id_to_stage.get(epochs.events[i, 2])
                if not stage:
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
            if subject_count % 20 == 0:
                print(f"  Processed {subject_count} subjects ({len(all_labels)} epochs)")
        except Exception as e:
            continue

    X_spec = np.array(all_spectral)
    X_geom = np.array([[g[0], g[1]] + ([g[2], g[3]] if len(g) > 2 else [0, 0])
                        for g in all_geometric])
    X_comb = np.hstack([X_spec, X_geom])
    y = np.array(all_labels)
    groups = np.array(all_subject_ids)

    print(f"\n  Total: {len(y)} epochs from {subject_count} subjects")

    # 5-fold CV
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    all_true, all_pred_spec, all_pred_comb = [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_comb, y, groups)):
        clf_s = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           learning_rate=0.1, random_state=42)
        clf_s.fit(X_spec[train_idx], y[train_idx])
        pred_s = clf_s.predict(X_spec[test_idx])

        clf_c = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           learning_rate=0.1, random_state=42)
        clf_c.fit(X_comb[train_idx], y[train_idx])
        pred_c = clf_c.predict(X_comb[test_idx])

        all_true.extend(y[test_idx])
        all_pred_spec.extend(pred_s)
        all_pred_comb.extend(pred_c)

        k_s = cohen_kappa_score(y[test_idx], pred_s)
        k_c = cohen_kappa_score(y[test_idx], pred_c)
        print(f"  Fold {fold+1}: κ_spec={k_s:.3f}, κ_comb={k_c:.3f}")

    all_true = np.array(all_true)
    all_pred_spec = np.array(all_pred_spec)
    all_pred_comb = np.array(all_pred_comb)

    # Generate confusion matrices
    out_dir = Path("data/results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_spec = confusion_matrix(all_true, all_pred_spec, labels=[0,1,2,3,4])
    cm_comb = confusion_matrix(all_true, all_pred_comb, labels=[0,1,2,3,4])

    plot_confusion_matrix(cm_spec,
                         "Spectral Features Only (HMC, 5-fold CV)",
                         str(out_dir / "confusion_matrix_spectral.png"))
    plot_confusion_matrix(cm_comb,
                         "Spectral + Geometric Features (HMC, 5-fold CV)",
                         str(out_dir / "confusion_matrix_combined.png"))

    # Side-by-side figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cm, title in [(ax1, cm_spec, "Spectral Only"),
                           (ax2, cm_comb, "Spectral + Geometric")]:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(STAGE_NAMES, fontsize=11)
        ax.set_yticklabels(STAGE_NAMES, fontsize=11)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        for i in range(5):
            for j in range(5):
                val = cm_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=10)

    fig.suptitle('HMC Confusion Matrices (132 subjects, 5-fold CV)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(out_dir / "confusion_matrices_sidebyside.png"),
                dpi=300, bbox_inches='tight')
    plt.savefig(str(out_dir / "confusion_matrices_sidebyside.pdf"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved side-by-side figure")

    # Print raw numbers
    print("\n" + "=" * 60)
    print("  CONFUSION MATRIX (Spectral Only)")
    print("=" * 60)
    print(f"  {'':>6}", end="")
    for s in STAGE_NAMES: print(f"  {s:>6}", end="")
    print()
    for i, s in enumerate(STAGE_NAMES):
        print(f"  {s:>6}", end="")
        for j in range(5):
            print(f"  {cm_spec[i,j]:>6}", end="")
        print()

    print("\n" + "=" * 60)
    print("  CONFUSION MATRIX (Spectral + Geometric)")
    print("=" * 60)
    print(f"  {'':>6}", end="")
    for s in STAGE_NAMES: print(f"  {s:>6}", end="")
    print()
    for i, s in enumerate(STAGE_NAMES):
        print(f"  {s:>6}", end="")
        for j in range(5):
            print(f"  {cm_comb[i,j]:>6}", end="")
        print()

    print("\n  Done.")


if __name__ == "__main__":
    main()
