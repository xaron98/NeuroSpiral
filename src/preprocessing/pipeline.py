"""EEG signal preprocessing pipeline.

Implements bandpass filtering, ICA-based artifact removal,
and optional resampling using MNE-Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mne
import numpy as np


@dataclass
class PreprocessingResult:
    """Outcome of the preprocessing pipeline."""

    raw: mne.io.Raw
    ica: mne.preprocessing.ICA | None
    excluded_components: list[int]
    steps_applied: list[str]

    def summary(self) -> dict[str, Any]:
        return {
            "channels": self.raw.ch_names,
            "sfreq": self.raw.info["sfreq"],
            "duration_sec": self.raw.times[-1],
            "n_ica_excluded": len(self.excluded_components),
            "steps": self.steps_applied,
        }


def preprocess_raw(
    raw: mne.io.Raw,
    l_freq: float = 0.5,
    h_freq: float = 30.0,
    resample_hz: float | None = 100.0,
    ica_config: dict | None = None,
) -> PreprocessingResult:
    """Run the full preprocessing pipeline on raw EEG.

    Pipeline order:
    1. Bandpass filter (FIR)
    2. Optional resample
    3. ICA artifact removal (EOG/EMG)

    Parameters
    ----------
    raw : MNE Raw object (must be preloaded)
    l_freq : highpass cutoff in Hz
    h_freq : lowpass cutoff in Hz
    resample_hz : target sample rate (None to skip)
    ica_config : dict with ICA parameters:
        n_components, method, max_iter, random_state, eog_threshold

    Returns
    -------
    PreprocessingResult with cleaned signal and metadata.
    """
    steps = []

    # --- 1. Bandpass filter ---
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        verbose=False,
    )
    steps.append(f"bandpass_{l_freq}-{h_freq}Hz")

    # --- 2. Resample ---
    if resample_hz is not None and raw.info["sfreq"] != resample_hz:
        raw.resample(resample_hz, verbose=False)
        steps.append(f"resample_{resample_hz}Hz")

    # --- 3. ICA artifact removal ---
    ica = None
    excluded = []

    if ica_config is not None:
        n_channels = len(raw.ch_names)
        n_components = min(ica_config.get("n_components", 15), n_channels)
        if n_components < 2:
            steps.append("ica_skipped")
            return PreprocessingResult(raw=raw, ica=None, excluded_components=[], steps_applied=steps)
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=ica_config.get("method", "fastica"),
            max_iter=ica_config.get("max_iter", 500),
            random_state=ica_config.get("random_state", 42),
        )
        ica.fit(raw, verbose=False)

        # Auto-detect EOG-correlated components
        # For Sleep-EDF: use EEG Fpz-Cz as proxy if no dedicated EOG
        eog_threshold = ica_config.get("eog_threshold", 0.85)
        eog_channels = [ch for ch in raw.ch_names if "eog" in ch.lower()]

        if eog_channels:
            eog_indices, eog_scores = ica.find_bads_eog(
                raw,
                ch_name=eog_channels[0],
                threshold=eog_threshold,
                verbose=False,
            )
        else:
            # Fallback: correlate with frontal channel (Fpz-Cz typical for eye artifacts)
            frontal = [ch for ch in raw.ch_names if "fpz" in ch.lower()]
            if frontal:
                eog_indices, eog_scores = ica.find_bads_eog(
                    raw,
                    ch_name=frontal[0],
                    threshold=eog_threshold,
                    verbose=False,
                )
            else:
                eog_indices = []

        excluded = list(eog_indices)
        ica.exclude = excluded
        ica.apply(raw, verbose=False)
        steps.append(f"ica_removed_{len(excluded)}_components")

    return PreprocessingResult(
        raw=raw,
        ica=ica,
        excluded_components=excluded,
        steps_applied=steps,
    )


def compute_epoch_quality(
    epoch_data: np.ndarray,
    sfreq: float,
    amplitude_threshold_uv: float = 200.0,
    flatline_threshold_uv: float = 0.5,
) -> np.ndarray:
    """Flag low-quality epochs for rejection.

    Parameters
    ----------
    epoch_data : shape (n_epochs, n_channels, n_samples)
    sfreq : sampling frequency
    amplitude_threshold_uv : max peak-to-peak amplitude (µV)
    flatline_threshold_uv : min std dev to detect flatlines

    Returns
    -------
    Boolean mask of shape (n_epochs,). True = keep epoch.
    """
    n_epochs = epoch_data.shape[0]
    keep = np.ones(n_epochs, dtype=bool)

    for i in range(n_epochs):
        epoch = epoch_data[i] * 1e6  # convert V → µV

        # Check peak-to-peak amplitude per channel
        ptp = np.ptp(epoch, axis=1)  # µV (assuming data already in µV)
        if np.any(ptp > amplitude_threshold_uv):
            keep[i] = False
            continue

        # Check for flatlines
        std = np.std(epoch, axis=1)
        if np.any(std < flatline_threshold_uv):
            keep[i] = False

    return keep
