#!/usr/bin/env python3
"""NeuroSpiral — Run preprocessing pipeline on Sleep-EDF data.

Usage:
    python scripts/run_preprocess.py --psg <path.edf> --hyp <hypnogram.edf>
    python scripts/run_preprocess.py --download-sample  # fetch PhysioNet sample

Example:
    python scripts/run_preprocess.py --download-sample
    python scripts/run_preprocess.py \
        --psg data/raw/SC4001E0-PSG.edf \
        --hyp data/raw/SC4001EC-Hypnogram.edf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.edf_loader import load_sleep_edf, extract_epochs_from_annotations
from src.preprocessing.pipeline import preprocess_raw, compute_epoch_quality
from src.features.spectral import extract_features_batch


def download_sample(output_dir: Path) -> tuple[Path, Path]:
    """Download a sample Sleep-EDF file from PhysioNet."""
    import urllib.request

    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
    files = {
        "SC4001E0-PSG.edf": f"{base_url}/SC4001E0-PSG.edf",
        "SC4001EC-Hypnogram.edf": f"{base_url}/SC4001EC-Hypnogram.edf",
    }

    paths = {}
    for fname, url in files.items():
        out_path = output_dir / fname
        if out_path.exists():
            print(f"  ✓ {fname} already exists")
        else:
            print(f"  ↓ Downloading {fname}...")
            urllib.request.urlretrieve(url, out_path)
            print(f"  ✓ Saved to {out_path}")
        paths[fname] = out_path

    return paths["SC4001E0-PSG.edf"], paths["SC4001EC-Hypnogram.edf"]


def plot_epoch_example(
    epoch_data: np.ndarray,
    label: str,
    sfreq: float,
    channel_names: list[str],
    save_path: Path | None = None,
):
    """Plot a single epoch with its power spectrum."""
    import matplotlib.pyplot as plt
    from scipy import signal

    n_channels = epoch_data.shape[0]
    t = np.arange(epoch_data.shape[1]) / sfreq

    fig, axes = plt.subplots(
        n_channels, 2,
        figsize=(14, 3 * n_channels),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    if n_channels == 1:
        axes = axes.reshape(1, 2)

    fig.suptitle(f"Sleep stage: {label} — 30s epoch", fontsize=14, fontweight="bold")

    for ch in range(n_channels):
        data = epoch_data[ch]

        # Time domain
        axes[ch, 0].plot(t, data * 1e6, linewidth=0.5, color="#534AB7")
        axes[ch, 0].set_ylabel(f"{channel_names[ch]}\n(µV)")
        axes[ch, 0].set_xlim(0, t[-1])
        axes[ch, 0].grid(True, alpha=0.3)

        # Frequency domain
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=min(256, len(data)))
        axes[ch, 1].semilogy(freqs, psd, linewidth=1.2, color="#1D9E75")
        axes[ch, 1].set_xlim(0, 30)
        axes[ch, 1].set_ylabel("PSD")
        axes[ch, 1].grid(True, alpha=0.3)

        # Shade delta band
        delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
        axes[ch, 1].fill_between(
            freqs[delta_mask], psd[delta_mask],
            alpha=0.3, color="#1D9E75", label="Delta (0.5-4 Hz)",
        )
        axes[ch, 1].legend(fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="NeuroSpiral preprocessing pipeline")
    parser.add_argument("--psg", type=Path, help="Path to PSG .edf file")
    parser.add_argument("--hyp", type=Path, help="Path to Hypnogram .edf file")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--download-sample", action="store_true", help="Download PhysioNet sample")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--plot", action="store_true", help="Generate epoch visualizations")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Download sample if requested
    if args.download_sample:
        print("\n📥 Downloading Sleep-EDF sample from PhysioNet...")
        raw_dir = PROJECT_ROOT / cfg["data"]["raw_dir"]
        psg_path, hyp_path = download_sample(raw_dir)
    elif args.psg and args.hyp:
        psg_path, hyp_path = args.psg, args.hyp
    else:
        parser.error("Provide --psg and --hyp, or use --download-sample")
        return

    # --- Step 1: Load EDF ---
    print("\n📂 Loading EDF files...")
    record = load_sleep_edf(
        psg_path=psg_path,
        hypnogram_path=hyp_path,
        channels=cfg["preprocessing"]["channels"],
        label_mapping=cfg["data"]["label_mapping"],
    )
    print(f"  Subject: {record.subject_id}")
    print(f"  Duration: {record.duration_hours:.1f}h")
    print(f"  Channels: {record.raw.ch_names}")
    print(f"  Sample rate: {record.sfreq} Hz")

    # --- Step 2: Preprocess ---
    print("\n🔧 Preprocessing...")
    result = preprocess_raw(
        raw=record.raw,
        l_freq=cfg["preprocessing"]["l_freq"],
        h_freq=cfg["preprocessing"]["h_freq"],
        resample_hz=cfg["preprocessing"].get("resample_hz"),
        ica_config=cfg["preprocessing"].get("ica"),
    )
    print(f"  Steps: {' → '.join(result.steps_applied)}")
    print(f"  ICA components excluded: {result.excluded_components}")

    # --- Step 3: Epoch extraction ---
    print("\n✂️  Extracting 30s epochs...")
    # Update record with preprocessed raw
    record.raw = result.raw
    epochs, labels, label_names = extract_epochs_from_annotations(
        record=record,
        epoch_duration=cfg["preprocessing"]["epoch_duration_sec"],
    )
    print(f"  Total epochs: {epochs.shape[0]}")
    print(f"  Shape: {epochs.shape} (epochs × channels × samples)")
    print(f"  Stages: {label_names}")

    # Stage distribution
    print("\n📊 Stage distribution:")
    for i, name in enumerate(label_names):
        count = np.sum(labels == i)
        pct = count / len(labels) * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:>4}: {count:>4} ({pct:5.1f}%) {bar}")

    # --- Step 4: Quality check ---
    print("\n🔍 Quality check...")
    quality_mask = compute_epoch_quality(epochs, result.raw.info["sfreq"])
    n_rejected = np.sum(~quality_mask)
    print(f"  Kept: {np.sum(quality_mask)}, Rejected: {n_rejected} ({n_rejected/len(quality_mask)*100:.1f}%)")

    epochs_clean = epochs[quality_mask]
    labels_clean = labels[quality_mask]

    # --- Step 5: Feature extraction ---
    print("\n⚡ Extracting features...")
    feat_cfg = cfg["features"]["spectral"]
    features = extract_features_batch(
        epochs_clean,
        sfreq=result.raw.info["sfreq"],
        bands=feat_cfg["bands"],
        compute_ratios=feat_cfg.get("compute_ratios", True),
        hjorth=cfg["features"]["temporal"].get("hjorth", True),
        permutation_entropy=cfg["features"]["temporal"].get("permutation_entropy", True),
    )
    print(f"  Feature matrix: {features.shape} (epochs × features)")

    # --- Step 6: Save ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / f"{record.subject_id}_processed.npz",
        features=features,
        labels=labels_clean,
        label_names=label_names,
        sfreq=result.raw.info["sfreq"],
    )
    print(f"\n💾 Saved to {args.output_dir}/{record.subject_id}_processed.npz")

    # --- Optional: Plot N3 epoch ---
    if args.plot:
        print("\n🎨 Generating N3 epoch visualization...")
        n3_idx = label_names.index("N3") if "N3" in label_names else None
        if n3_idx is not None:
            n3_mask = labels_clean == n3_idx
            if np.any(n3_mask):
                first_n3 = np.where(n3_mask)[0][0]
                plot_epoch_example(
                    epochs_clean[first_n3],
                    label="N3 (slow wave sleep)",
                    sfreq=result.raw.info["sfreq"],
                    channel_names=result.raw.ch_names,
                    save_path=args.output_dir / f"{record.subject_id}_N3_epoch.png",
                )

    print("\n✅ Pipeline complete!")
    print(f"  Next: train baseline model with {features.shape[0]} epochs × {features.shape[1] if features.ndim > 1 else 0} features")


if __name__ == "__main__":
    main()
