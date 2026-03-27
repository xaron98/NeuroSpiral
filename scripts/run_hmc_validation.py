"""Run NeuroSpiral validation on HMC dataset.

HMC differences from Sleep-EDF:
  - 256 Hz sampling (vs 100 Hz) → resample to 100 Hz for consistent τ=25
  - AASM labels directly (no R&K conversion)
  - Channel names: EEG C4-M1, EEG C3-M2, etc.
  - Naming: SN001.edf + SN001_sleepscoring.edf

Usage:
  python scripts/run_hmc_validation.py --data-dir data/hmc --n-subjects 151
"""

from __future__ import annotations
import sys, os, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mne

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.alignment import compute_fixed_tau

# HMC label mapping (AASM native)
HMC_LABELS = {
    "Sleep stage W": "W",
    "Sleep stage N1": "N1",
    "Sleep stage N2": "N2",
    "Sleep stage N3": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": None,
    # HMC numeric annotations
    "0": "W",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "REM",
    "5": None,
    # Alternative numeric schemes
    "W": "W",
    "N1": "N1",
    "N2": "N2",
    "N3": "N3",
    "REM": "REM",
    "R": "REM",
}

HMC_CHANNELS = ["EEG C4-M1", "EEG C3-M2", "EEG F4-M1", "EEG F3-M2", "EEG O2-M1"]
TARGET_SFREQ = 100  # Resample to match Sleep-EDF and keep τ=25


def list_hmc_subjects(data_dir: Path) -> list[tuple[Path, Path]]:
    """Find all valid HMC subject pairs."""
    pairs = []
    for i in range(1, 155):
        sid = f"SN{i:03d}"
        psg = data_dir / f"{sid}.edf"
        hyp = data_dir / f"{sid}_sleepscoring.edf"
        if psg.exists() and hyp.exists():
            if psg.stat().st_size > 1_000_000 and hyp.stat().st_size > 500:
                pairs.append((psg, hyp))
    return pairs


def process_hmc_subject(
    psg_path: Path,
    hyp_path: Path,
    fixed_tau: int = 25,
) -> tuple[pd.DataFrame, list[str], dict] | None:
    """Process one HMC subject with enhanced multi-channel features."""

    from src.features.spectral import compute_band_powers, compute_hjorth
    from src.features.enhanced import (
        multichannel_takens_embed,
        compute_winding_asymmetric,
        compute_transition_features,
        torus_to_angles,
    )
    from src.features.takens import time_delay_embedding as takens_embed
    from src.geometry.tesseract import Q_discretize, VERTICES

    sid = psg_path.stem

    try:
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)

        # Pick 2 EEG channels for multi-channel embedding
        # Priority: C4-M1 + C3-M2 (bilateral central — best for sleep staging)
        ch_pairs = [
            ("EEG C4-M1", "EEG C3-M2"),   # bilateral central
            ("EEG F4-M1", "EEG F3-M2"),   # bilateral frontal
            ("EEG C4-M1", "EEG F4-M1"),   # ipsilateral central+frontal
        ]
        ch1_name = ch2_name = None
        for c1, c2 in ch_pairs:
            if c1 in raw.ch_names and c2 in raw.ch_names:
                ch1_name, ch2_name = c1, c2
                break

        if ch1_name is None:
            # Fallback: any 2 EEG channels
            eeg_chs = [c for c in raw.ch_names if "EEG" in c.upper()]
            if len(eeg_chs) >= 2:
                ch1_name, ch2_name = eeg_chs[0], eeg_chs[1]
            elif len(eeg_chs) == 1:
                ch1_name = eeg_chs[0]
            else:
                print(f"✗ No EEG channel found")
                return None

        use_multichannel = ch2_name is not None
        if use_multichannel:
            raw.pick([ch1_name, ch2_name])
        else:
            raw.pick([ch1_name])

        # Resample to 100 Hz
        if raw.info["sfreq"] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ)

        # Bandpass filter
        raw.filter(0.5, 30.0, verbose=False)

        # Load annotations — HMC uses integer descriptions, MNE needs strings
        annotations = mne.read_annotations(str(hyp_path))
        new_descriptions = [str(d) for d in annotations.description]
        annotations = mne.Annotations(
            onset=annotations.onset,
            duration=annotations.duration,
            description=new_descriptions
        )
        raw.set_annotations(annotations)

        # Extract 30s epochs
        events, event_id = mne.events_from_annotations(raw, chunk_duration=30.0, verbose=False)

        stage_names_list = ["W", "N1", "N2", "N3", "REM"]
        label_map = {}
        for desc, eid in event_id.items():
            mapped = HMC_LABELS.get(desc)
            if mapped and mapped in stage_names_list:
                if mapped not in label_map:
                    label_map[mapped] = eid

        if not label_map:
            print(f"✗ No valid labels")
            return None

        epochs = mne.Epochs(raw, events, event_id=label_map,
                           tmin=0, tmax=30.0 - 1.0/TARGET_SFREQ,
                           baseline=None, preload=True, verbose=False)

        if len(epochs) == 0:
            print(f"✗ No valid epochs")
            return None

        data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
        labels = [epochs.events[i, 2] for i in range(len(epochs))]

        id_to_stage = {}
        for desc, eid in event_id.items():
            mapped = HMC_LABELS.get(desc)
            if mapped:
                id_to_stage[eid] = mapped

        stage_labels = [id_to_stage.get(l, "?") for l in labels]
        valid_mask = [s in stage_names_list for s in stage_labels]
        data = data[valid_mask]
        stage_labels = [s for s, v in zip(stage_labels, valid_mask) if v]

        if len(data) == 0:
            print(f"✗ No valid epochs after filtering")
            return None

        n_epochs = len(data)
        n_channels = data.shape[1]
        stage_to_int = {s: i for i, s in enumerate(stage_names_list)}
        label_ints = [stage_to_int[s] for s in stage_labels]

        rows = []
        for i in range(n_epochs):
            ch1_data = data[i, 0, :] * 1e6  # V → µV
            ch2_data = data[i, 1, :] * 1e6 if n_channels >= 2 else None

            # Skip artifacts
            if np.max(np.abs(ch1_data)) > 500:
                continue
            if ch2_data is not None and np.max(np.abs(ch2_data)) > 500:
                continue

            # Spectral features (from primary channel)
            bp = compute_band_powers(ch1_data, TARGET_SFREQ)
            hjorth = compute_hjorth(ch1_data)

            # Multi-channel Takens embedding
            if ch2_data is not None:
                embedded = multichannel_takens_embed(ch1_data, ch2_data, fixed_tau)
            else:
                embedded = takens_embed(ch1_data, d=4, tau=fixed_tau)

            # Torus projection
            norms_xy = np.sqrt(embedded[:, 0]**2 + embedded[:, 1]**2 + 1e-10)
            norms_zw = np.sqrt(embedded[:, 2]**2 + embedded[:, 3]**2 + 1e-10)
            R = np.sqrt(2.0)
            projected = embedded.copy()
            projected[:, 0] *= R / norms_xy
            projected[:, 1] *= R / norms_xy
            projected[:, 2] *= R / norms_zw
            projected[:, 3] *= R / norms_zw

            # Vertex (standard Q discretization)
            mean_point = np.mean(projected, axis=0)
            vertex = Q_discretize(mean_point)

            # Asymmetric winding numbers
            winding = compute_winding_asymmetric(projected)

            # Vertex sequence + transition features
            step = max(1, len(projected) // 100)
            vertex_seq = np.array([Q_discretize(p) for p in projected[::step]])
            stability = np.mean(vertex_seq == vertex)
            trans = compute_transition_features(vertex_seq, n_vertices=16)

            # Torus angles for adaptive discretization
            angles = torus_to_angles(projected)
            mean_angle = np.mean(angles, axis=0)

            rows.append({
                "epoch": i,
                "stage": stage_labels[i],
                "label_int": label_ints[i],
                "vertex": vertex,
                # Enhanced winding (asymmetric)
                "omega1": winding["omega1"],
                "omega2": winding["omega2"],
                "winding_ratio": winding["ratio"],
                "phase_coherence": winding["phase_coherence"],
                "phase_diff_std": winding["phase_diff_std"],
                # Transition features
                "stability": stability,
                "transition_rate": trans["transition_rate"],
                "bigram_entropy": trans["bigram_entropy"],
                "self_loop_frac": trans["self_loop_frac"],
                "unique_vertices": trans["unique_vertices"],
                "transitions": trans["n_transitions"],
                # Torus angles
                "torus_theta": mean_angle[0],
                "torus_phi": mean_angle[1],
                # Spectral
                "spec_delta": bp.get("delta", 0),
                "spec_theta": bp.get("theta", 0),
                "spec_alpha": bp.get("alpha", 0),
                "spec_sigma": bp.get("sigma", 0),
                "spec_beta": bp.get("beta", 0),
                "hjorth_activity": hjorth[0],
                "hjorth_mobility": hjorth[1],
                "hjorth_complexity": hjorth[2],
            })

        if not rows:
            print(f"✗ All epochs rejected")
            return None

        df = pd.DataFrame(rows)
        n3_count = sum(1 for s in stage_labels if s == "N3")
        se = sum(1 for s in stage_labels if s != "W") / len(stage_labels) * 100
        mc_str = f"2ch({ch1_name},{ch2_name})" if use_multichannel else f"1ch({ch1_name})"

        traditional = {
            "subject": sid,
            "n_epochs": len(df),
            "n3_count": n3_count,
            "sleep_efficiency": se,
            "channels": mc_str,
        }

        print(f"✓ {len(df)} epochs, N3={n3_count}, SE={se:.0f}% [{mc_str}]")
        return df, stage_names_list, traditional

    except Exception as e:
        print(f"✗ {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="NeuroSpiral — HMC Validation")
    parser.add_argument("--n-subjects", type=int, default=151)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/hmc")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/results/hmc")
    args = parser.parse_args()

    t_start = time.time()

    print("\n" + "═" * 65)
    print("  NeuroSpiral — HMC Cross-Dataset Validation")
    print("  151 PSGs, AASM scoring, Haaglanden Medisch Centrum")
    print("═" * 65)

    pairs = list_hmc_subjects(args.data_dir)
    n = min(args.n_subjects, len(pairs))
    print(f"\n  Found {len(pairs)} subjects, processing {n}")

    fixed_tau = compute_fixed_tau(TARGET_SFREQ, target_freq=2.0)
    print(f"  Fixed τ = {fixed_tau} samples (resampled to {TARGET_SFREQ} Hz)")

    # Import gap analysis functions from publish_validate
    from scripts.publish_validate import (
        gap1_vertex_stage_mapping,
        gap2_inter_subject_consistency,
        gap3_novel_metrics,
    )

    per_subject_data = []
    all_frames = []
    all_traditional = []
    stage_names = None

    for psg_path, hyp_path in pairs[:n]:
        sid = psg_path.stem
        print(f"\n  {sid}...", end=" ")
        result = process_hmc_subject(psg_path, hyp_path, fixed_tau=fixed_tau)
        if result is None:
            continue

        df, names, traditional = result
        if stage_names is None:
            stage_names = names

        df["subject"] = sid
        per_subject_data.append((df, sid))
        all_frames.append(df)
        all_traditional.append(traditional)

    if not all_frames:
        print("\n  ✗ No subjects processed successfully")
        return

    all_data = pd.concat(all_frames, ignore_index=True)
    print(f"\n  Total: {len(all_data)} epochs from {len(per_subject_data)} subjects")

    # Run gap analyses (same as Sleep-EDF)
    print(f"\n[2/4] Gap 1: Vertex ↔ PSG Stage validation...")
    vertex_map, cramers_v, cramers_v_bal, best_v = gap1_vertex_stage_mapping(all_data, stage_names)

    print(f"\n[3/4] Gap 2: Inter-subject consistency...")
    consistency = gap2_inter_subject_consistency(per_subject_data, stage_names)

    print(f"\n[4/4] Gap 3: Novel metric comparison...")
    metrics = gap3_novel_metrics(all_data, all_traditional, per_subject_data)

    elapsed = time.time() - t_start

    print("\n" + "═" * 65)
    print("  HMC CROSS-DATASET VALIDATION SUMMARY")
    print("═" * 65)
    print(f"""
  Dataset:                     HMC (Haaglanden Medisch Centrum)
  Subjects:                    {len(per_subject_data)}
  Total epochs:                {len(all_data)}

  Cramér's V (raw 16×5):       {cramers_v:.3f}
  Cramér's V (balanced):       {cramers_v_bal:.3f}
  Cramér's V (best method):    {best_v:.3f}

  Inter-subject consistency:   {consistency:.0%}
  F1 spectral-only:            {metrics['f1_trad']:.3f}
  F1 combined:                 {metrics['f1_combined']:.3f}
  Δ F1 (geometric adds):       {metrics['delta_f1']:+.3f}
  Time:                        {elapsed:.0f}s

  Cross-dataset check:
    [{'✓' if best_v > 0.2 else '✗'}] V > 0.2 (generalizes across datasets)
    [{'✓' if metrics['delta_f1'] > 0 else '✗'}] Geometric still helps (Δ={metrics['delta_f1']:+.3f})
""")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_data.to_csv(args.output_dir / "hmc_all_epochs.csv", index=False)
    print(f"  Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
