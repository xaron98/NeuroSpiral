"""Takens time-delay embedding.

Reconstructs the phase-space attractor of a 1D time series
by embedding it in d dimensions with delay tau.

References:
    Takens, F. (1981). "Detecting strange attractors in turbulence."
    Fraser, A. & Swinney, H. (1986). "Independent coordinates for strange
    attractors from mutual information."
"""

from __future__ import annotations

import numpy as np
from scipy.signal import argrelextrema


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 64,
) -> float:
    """Compute mutual information MI(X;Y) via binned 2D histogram.

    Used to find the optimal delay tau: the first local minimum of
    MI(x(t), x(t+tau)) indicates sufficient decorrelation without
    total independence (Fraser & Swinney, 1986).
    """
    hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
    p_xy = hist_2d / hist_2d.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > 1e-12 and p_x[i] > 1e-12 and p_y[j] > 1e-12:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi


def estimate_optimal_tau(
    signal: np.ndarray,
    max_lag: int = 100,
    n_bins: int = 64,
) -> int:
    """Find optimal delay tau as first local minimum of mutual information.

    Parameters
    ----------
    signal : 1D time series.
    max_lag : maximum lag to search.
    n_bins : histogram bins for MI estimation.

    Returns
    -------
    Optimal delay in samples.
    """
    max_lag = min(max_lag, len(signal) // 4)
    mi_values = np.zeros(max_lag)

    for lag in range(max_lag):
        if lag == 0:
            mi_values[lag] = mutual_information(signal, signal, n_bins)
        else:
            mi_values[lag] = mutual_information(
                signal[:-lag], signal[lag:], n_bins
            )

    local_mins = argrelextrema(mi_values[1:], np.less)[0]
    if len(local_mins) > 0:
        return int(local_mins[0] + 1)

    threshold = mi_values[0] / np.e
    below = np.where(mi_values[1:] < threshold)[0]
    if len(below) > 0:
        return int(below[0] + 1)

    return max(1, max_lag // 4)


def time_delay_embedding(
    signal: np.ndarray,
    dimension: int = 4,
    tau: int | None = None,
) -> tuple[np.ndarray, int]:
    """Reconstruct phase-space attractor via Takens' embedding.

    Parameters
    ----------
    signal : 1D time series of shape (n_samples,).
    dimension : embedding dimension (d=4 for the Clifford torus).
    tau : delay in samples (None = auto-estimate via MI).

    Returns
    -------
    embedded : array of shape (n_points, dimension).
    tau_used : the delay actually used.

    Raises
    ------
    ValueError : if signal is too short for the given parameters.
    """
    if tau is None:
        tau = estimate_optimal_tau(signal)

    n_points = len(signal) - (dimension - 1) * tau

    if n_points <= 0:
        raise ValueError(
            f"Signal too short ({len(signal)} samples) for "
            f"embedding with d={dimension}, tau={tau}. "
            f"Need at least {(dimension - 1) * tau + 1} samples."
        )

    embedded = np.zeros((n_points, dimension))
    for i in range(dimension):
        embedded[:, i] = signal[i * tau : i * tau + n_points]

    return embedded, tau
