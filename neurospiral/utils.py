"""Utility functions for NeuroSpiral.

I/O helpers, preprocessing wrappers, and common operations.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def spectral_features(
    epoch: np.ndarray,
    sfreq: float = 100.0,
) -> np.ndarray | None:
    """Compute 8 spectral features from a single-channel epoch.

    Features:
        1. delta: relative power 0.5-4 Hz
        2. theta: relative power 4-8 Hz
        3. alpha: relative power 8-13 Hz
        4. sigma: relative power 12-15 Hz
        5. beta:  relative power 13-30 Hz
        6. total_power: absolute total power
        7. sef95: spectral edge frequency 95%
        8. median_freq: median frequency

    Returns (8,) array or None.
    """
    freqs, psd = welch(epoch, fs=sfreq, nperseg=min(256, len(epoch)))
    total = np.trapz(psd, freqs)
    if total < 1e-30:
        return None

    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
             "sigma": (12, 15), "beta": (13, 30)}

    feats = []
    for name in ["delta", "theta", "alpha", "sigma", "beta"]:
        lo, hi = bands[name]
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(float(np.trapz(psd[mask], freqs[mask]) / total))

    feats.append(float(total))

    df = np.diff(freqs, prepend=freqs[0])
    cum_power = np.cumsum(psd * df)
    cum_total = cum_power[-1]

    if cum_total < 1e-30:
        feats.extend([0.0, 0.0])
    else:
        idx_95 = np.searchsorted(cum_power, 0.95 * cum_total)
        feats.append(float(freqs[min(idx_95, len(freqs) - 1)]))
        idx_50 = np.searchsorted(cum_power, 0.50 * cum_total)
        feats.append(float(freqs[min(idx_50, len(freqs) - 1)]))

    return np.array(feats, dtype=np.float64)


def fdr_correction(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    adjusted = np.empty(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            adjusted[sorted_idx[i + 1]],
            sorted_p[i] * n / (i + 1),
        )
    return np.clip(adjusted, 0, 1)
