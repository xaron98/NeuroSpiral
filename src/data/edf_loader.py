"""EDF file loader for polysomnography datasets.

Handles Sleep-EDF (.edf + hypnogram) ingestion with automatic
label mapping from Rechtschaffen & Kales to AASM standard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np


@dataclass
class SleepRecord:
    """Container for a single night's PSG recording."""

    subject_id: str
    raw: mne.io.Raw
    annotations: mne.Annotations
    label_mapping: dict[str, str | None] = field(default_factory=dict)

    @property
    def duration_hours(self) -> float:
        return self.raw.times[-1] / 3600

    @property
    def n_channels(self) -> int:
        return len(self.raw.ch_names)

    @property
    def sfreq(self) -> float:
        return self.raw.info["sfreq"]


def load_sleep_edf(
    psg_path: str | Path,
    hypnogram_path: str | Path,
    channels: list[str] | None = None,
    label_mapping: dict[str, str | None] | None = None,
) -> SleepRecord:
    """Load a Sleep-EDF polysomnography recording.

    Parameters
    ----------
    psg_path : path to the PSG .edf file
    hypnogram_path : path to the hypnogram .edf file
    channels : EEG channels to keep (None = all)
    label_mapping : R&K → AASM label mapping dict.
        Labels mapped to None are discarded.

    Returns
    -------
    SleepRecord with raw data and annotations loaded.
    """
    psg_path = Path(psg_path)
    hypnogram_path = Path(hypnogram_path)

    if not psg_path.exists():
        raise FileNotFoundError(f"PSG file not found: {psg_path}")
    if not hypnogram_path.exists():
        raise FileNotFoundError(f"Hypnogram file not found: {hypnogram_path}")

    # Load raw EEG signal
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

    # Pick specific channels if requested
    if channels:
        available = [ch for ch in channels if ch in raw.ch_names]
        if not available:
            raise ValueError(
                f"None of {channels} found in recording. "
                f"Available: {raw.ch_names}"
            )
        raw.pick(available)

    # Load hypnogram annotations
    annotations = mne.read_annotations(str(hypnogram_path))
    raw.set_annotations(annotations)

    # Extract subject ID from filename (Sleep-EDF convention: SC4001E0-PSG.edf)
    subject_id = psg_path.stem.split("-")[0]

    mapping = label_mapping or {}

    return SleepRecord(
        subject_id=subject_id,
        raw=raw,
        annotations=annotations,
        label_mapping=mapping,
    )


def extract_epochs_from_annotations(
    record: SleepRecord,
    epoch_duration: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Segment continuous recording into labeled epochs.

    Parameters
    ----------
    record : loaded SleepRecord
    epoch_duration : epoch length in seconds (standard: 30s)

    Returns
    -------
    data : array of shape (n_epochs, n_channels, n_samples)
    labels : integer array of shape (n_epochs,)
    label_names : list mapping integer labels to stage names
    """
    raw = record.raw
    sfreq = raw.info["sfreq"]
    n_samples_per_epoch = int(epoch_duration * sfreq)

    # Build events from annotations
    mapping = record.label_mapping
    # Collect valid (mapped) stage names
    valid_stages = sorted({v for v in mapping.values() if v is not None})
    stage_to_int = {stage: i for i, stage in enumerate(valid_stages)}

    epochs_data = []
    epoch_labels = []

    for ann in record.annotations:
        description = ann["description"]

        # Map R&K → AASM
        mapped = mapping.get(description)
        if mapped is None:
            continue  # skip unmapped or explicitly excluded

        onset_sample = int(ann["onset"] * sfreq)
        duration_samples = int(ann["duration"] * sfreq)

        # Some annotations span multiple epochs
        n_sub_epochs = duration_samples // n_samples_per_epoch

        for i in range(n_sub_epochs):
            start = onset_sample + i * n_samples_per_epoch
            stop = start + n_samples_per_epoch

            if stop > raw.n_times:
                break

            segment = raw.get_data(start=start, stop=stop)
            epochs_data.append(segment)
            epoch_labels.append(stage_to_int[mapped])

    if not epochs_data:
        raise ValueError("No valid epochs extracted. Check label_mapping and annotations.")

    data = np.stack(epochs_data, axis=0)
    labels = np.array(epoch_labels, dtype=np.int64)

    return data, labels, valid_stages
