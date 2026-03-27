"""Phase 1.1: Permutation test for phase_diff_std on HMC.

Computes CMI(phase_diff_std, stage | delta) with 1000 permutations.
This is the strongest geometric feature (MI=0.4269) and needs its own p-value.

Also computes CMI for all enhanced features for the final paper table.
"""

import numpy as np
import sys, os, time
from pathlib import Path
from collections import Counter

import mne
from sklearn.metrics import mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.spectral import compute_band_powers, compute_hjorth
from src.features.takens import time_delay_embedding
from src.geometry.tesseract import Q_discretize, VERTICES
from src.geometry.alignment import compute_fixed_tau

HMC_LABELS = {
    "Sleep stage W": "W", "Sleep stage N1": "N1", "Sleep stage N2": "N2",
    "Sleep stage N3": "N3", "Sleep stage R": "REM", "Sleep stage ?": None,
    "0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "REM", "5": None,
    "W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "REM", "R": "REM",
}

TARGET_SFREQ = 100
MIN_EPOCHS = 500
N_PERMUTATIONS = 1000


def discretize_continuous(values, n_bins=10):
    """Discretize continuous values into bins using percentiles."""
    percentiles = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    return np.digitize(values, percentiles[1:-1])


def compute_cmi(feature, stage, condition):
    """Compute CMI(feature, stage | condition) = MI(feature+condition, stage) - MI(condition, stage)."""
    joint = [f"{f}_{c}" for f, c in zip(feature, condition)]
    mi_joint = mutual_info_score(stage, joint)
    mi_cond = mutual_info_score(stage, condition)
    return mi_joint - mi_cond


def permutation_test_cmi(feature, stage, condition, n_perm=1000):
    """Permutation test for CMI significance."""
    observed = compute_cmi(feature, stage, condition)
    null_dist = []
    for _ in range(n_perm):
        perm = np.random.permutation(feature)
        null_dist.append(compute_cmi(perm, stage, condition))
    null_dist = np.array(null_dist)
    p_value = np.mean(null_dist >= observed)
    return observed, null_dist, p_value


def main():
    data_dir = Path("data/hmc")
    fixed_tau = compute_fixed_tau(TARGET_SFREQ, 2.0)

    print("=" * 65)
    print("  Phase 1.1: Permutation Test for Enhanced Features (HMC)")
    print("  1000 permutations per feature")
    print("=" * 65)

    # Collect all data
    all_stages = []
    all_delta = []
    all_omega1 = []
    all_phase_diff_std = []
    all_phase_coherence = []
    all_bigram_entropy = []
    all_transition_rate = []
    subject_count = 0

    subjects = []
    for f in sorted(data_dir.glob("SN*.edf")):
        if "_sleepscoring" not in f.name:
            sid = f.stem
            hyp = data_dir / f"{sid}_sleepscoring.edf"
            if hyp.exists():
                subjects.append((f, hyp, sid))

    print(f"\n  Processing {len(subjects)} subjects (min {MIN_EPOCHS} epochs)...\n")

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
            new_descriptions = [str(d) for d in annotations.description]
            annotations = mne.Annotations(
                onset=annotations.onset,
                duration=annotations.duration,
                description=new_descriptions
            )
            raw.set_annotations(annotations)

            events, event_id = mne.events_from_annotations(
                raw, chunk_duration=30.0, verbose=False
            )

            stage_names = ["W", "N1", "N2", "N3", "REM"]
            label_map = {}
            for desc, eid in event_id.items():
                mapped = HMC_LABELS.get(desc)
                if mapped and mapped in stage_names:
                    if mapped not in label_map:
                        label_map[mapped] = eid

            if not label_map:
                continue

            epochs = mne.Epochs(
                raw, events, event_id=label_map,
                tmin=0, tmax=30.0 - 1.0 / TARGET_SFREQ,
                baseline=None, preload=True, verbose=False
            )

            if len(epochs) < MIN_EPOCHS:
                continue

            data = epochs.get_data()  # (n_epochs, 2, n_samples)

            id_to_stage = {}
            for desc, eid in event_id.items():
                mapped = HMC_LABELS.get(desc)
                if mapped:
                    id_to_stage[eid] = mapped

            labels = []
            for i in range(len(epochs)):
                eid = epochs.events[i, 2]
                stage = id_to_stage.get(eid)
                if stage and stage in stage_names:
                    labels.append(stage)
                else:
                    labels.append(None)

            # Process epochs — both channels
            prev_vertex = None
            epoch_deltas = []
            epoch_stages = []
            epoch_omega1 = []
            epoch_phase_diff = []
            epoch_phase_coh = []
            epoch_bigram = []
            epoch_trans = []
            transition_count = 0

            for i in range(len(data)):
                if labels[i] is None:
                    continue

                ch1 = data[i, 0, :] * 1e6  # C4
                ch2 = data[i, 1, :] * 1e6  # C3

                if np.max(np.abs(ch1)) > 500 or np.max(np.abs(ch2)) > 500:
                    continue

                # Takens embedding for channel 1
                emb1, _ = time_delay_embedding(ch1, dimension=4, tau=fixed_tau)
                if len(emb1) == 0:
                    continue

                # Takens embedding for channel 2
                emb2, _ = time_delay_embedding(ch2, dimension=4, tau=fixed_tau)
                if len(emb2) == 0:
                    continue

                mean_emb1 = np.mean(emb1, axis=0)
                mean_emb2 = np.mean(emb2, axis=0)

                # Band powers (channel 1)
                bp = compute_band_powers(ch1, sfreq=TARGET_SFREQ)
                delta = bp.get('delta', 0)

                # Winding number (channel 1)
                dtheta = np.diff(np.arctan2(emb1[:, 1], emb1[:, 0]))
                dphi = np.diff(np.arctan2(emb1[:, 3], emb1[:, 2]))
                # Unwrap
                dtheta = np.where(dtheta > np.pi, dtheta - 2*np.pi, dtheta)
                dtheta = np.where(dtheta < -np.pi, dtheta + 2*np.pi, dtheta)
                dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
                dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
                omega1 = np.abs(np.mean(dtheta))

                # Phase difference std between channels
                phase1 = np.arctan2(emb1[:, 1], emb1[:, 0])
                phase2 = np.arctan2(emb2[:, 1], emb2[:, 0])
                phase_diff = phase1[:min(len(phase1), len(phase2))] - phase2[:min(len(phase1), len(phase2))]
                # Circular wrapping
                phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
                phase_diff_std = np.std(phase_diff)

                # Phase coherence
                phase_coh = np.abs(np.mean(np.exp(1j * phase_diff)))

                # Vertex for transition tracking
                vertex = Q_discretize(mean_emb1)
                if prev_vertex is not None and vertex != prev_vertex:
                    transition_count += 1

                # Bigram entropy (rolling window of last 10)
                if len(epoch_stages) >= 10:
                    recent_verts = [Q_discretize(np.mean(time_delay_embedding(
                        data[max(0,i-j), 0, :] * 1e6, dimension=4, tau=fixed_tau)[0], axis=0))
                        for j in range(min(10, i)) if labels[max(0,i-j)] is not None]
                    if len(recent_verts) >= 2:
                        bigrams = [(recent_verts[k], recent_verts[k+1]) 
                                   for k in range(len(recent_verts)-1)]
                        bg_counts = Counter(bigrams)
                        total = sum(bg_counts.values())
                        entropy = -sum((c/total) * np.log2(c/total) 
                                      for c in bg_counts.values() if c > 0)
                    else:
                        entropy = 0
                else:
                    entropy = 0

                trans_rate = transition_count / max(len(epoch_stages) + 1, 1)

                epoch_deltas.append(delta)
                epoch_stages.append(labels[i])
                epoch_omega1.append(omega1)
                epoch_phase_diff.append(phase_diff_std)
                epoch_phase_coh.append(phase_coh)
                epoch_bigram.append(entropy)
                epoch_trans.append(trans_rate)
                prev_vertex = vertex

            if len(epoch_stages) < MIN_EPOCHS * 0.5:
                continue

            all_stages.extend(epoch_stages)
            all_delta.extend(epoch_deltas)
            all_omega1.extend(epoch_omega1)
            all_phase_diff_std.extend(epoch_phase_diff)
            all_phase_coherence.extend(epoch_phase_coh)
            all_bigram_entropy.extend(epoch_bigram)
            all_transition_rate.extend(epoch_trans)

            subject_count += 1
            print(f"  {sid}... ✓ {len(epoch_stages)} epochs")

        except Exception as e:
            continue

    print(f"\n  Total: {len(all_stages)} epochs from {subject_count} subjects")

    # Discretize all features
    stage_labels = all_stages
    delta_bins = discretize_continuous(all_delta).tolist()
    omega1_bins = discretize_continuous(all_omega1).tolist()
    pds_bins = discretize_continuous(all_phase_diff_std).tolist()
    pc_bins = discretize_continuous(all_phase_coherence).tolist()
    be_bins = discretize_continuous(all_bigram_entropy).tolist()
    tr_bins = discretize_continuous(all_transition_rate).tolist()

    print("\n" + "=" * 65)
    print("  PERMUTATION TESTS (1000 permutations each)")
    print("=" * 65)

    features = [
        ("ω₁ (winding number)", omega1_bins),
        ("phase_diff_std", pds_bins),
        ("phase_coherence", pc_bins),
        ("bigram_entropy", be_bins),
        ("transition_rate", tr_bins),
    ]

    print(f"\n  Baseline: MI(delta, stage) = {mutual_info_score(stage_labels, delta_bins):.4f}\n")

    t0 = time.time()
    for name, feat_bins in features:
        print(f"  {name}:")
        mi_feat = mutual_info_score(stage_labels, feat_bins)
        print(f"    MI(feature, stage):     {mi_feat:.4f}")

        observed, null_dist, p_value = permutation_test_cmi(
            feat_bins, stage_labels, delta_bins, n_perm=N_PERMUTATIONS
        )

        print(f"    CMI(feature | delta):   {observed:.4f}")
        print(f"    Null 95th percentile:   {np.percentile(null_dist, 95):.4f}")
        print(f"    p-value:                {p_value:.4f}")
        if p_value < 0.001:
            print(f"    ✓ SIGNIFICANT (p < 0.001)")
        elif p_value < 0.05:
            print(f"    ✓ SIGNIFICANT (p < 0.05)")
        else:
            print(f"    ✗ NOT SIGNIFICANT")
        print()

    elapsed = time.time() - t0
    print(f"  Permutation tests completed in {elapsed:.0f}s")
    print(f"\n  Done. These results go in Table 3 of the paper.")


if __name__ == "__main__":
    main()
