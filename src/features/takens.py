"""Time Delay Embedding based on Takens' Theorem.

Reconstructs the phase-space attractor of a 1D time series
by embedding it in d dimensions with delay τ.

The key insight for sleep EEG: different sleep stages produce
geometrically distinct attractors in the reconstructed space.
N3 (slow-wave) creates smooth, low-dimensional loops;
Wake produces high-dimensional chaotic trajectories.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import argrelextrema


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 64,
) -> float:
    """Compute mutual information between two signals using binned histogram.

    MI(X;Y) = H(X) + H(Y) - H(X,Y)

    Used to find the optimal delay τ: the first local minimum of
    MI(x(t), x(t+τ)) indicates sufficient decorrelation without
    total independence (Fraser & Swinney, 1986).
    """
    eps = 1e-12

    # Joint histogram
    hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
    p_xy = hist_2d / hist_2d.sum()

    # Marginals
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > eps and p_x[i] > eps and p_y[j] > eps:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

    return mi


def estimate_optimal_tau(
    signal: np.ndarray,
    max_lag: int = 100,
    n_bins: int = 64,
) -> int:
    """Find optimal delay τ as first local minimum of mutual information.

    Fraser & Swinney (1986) showed this is superior to autocorrelation
    for nonlinear systems like EEG, because MI captures nonlinear
    dependencies that Pearson correlation misses.

    Parameters
    ----------
    signal : 1D time series
    max_lag : maximum lag to search
    n_bins : histogram bins for MI estimation

    Returns
    -------
    Optimal delay in samples. Falls back to first lag where
    MI drops below 1/e of MI(0) if no local minimum found.
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

    # Find first local minimum
    local_mins = argrelextrema(mi_values[1:], np.less)[0]

    if len(local_mins) > 0:
        return int(local_mins[0] + 1)  # +1 because we searched from index 1

    # Fallback: first crossing below 1/e of MI(0)
    threshold = mi_values[0] / np.e
    below = np.where(mi_values[1:] < threshold)[0]
    if len(below) > 0:
        return int(below[0] + 1)

    # Last resort: use lag = sfreq / dominant_freq (roughly one period of delta)
    return max(1, max_lag // 4)


def time_delay_embedding(
    signal: np.ndarray,
    dimension: int = 4,
    tau: int | None = None,
    n_bins: int = 64,
    max_tau_search: int = 100,
) -> tuple[np.ndarray, int]:
    """Reconstruct phase-space attractor via Takens' embedding.

    Takens' theorem (1981): for a d-dimensional dynamical system,
    an embedding in m ≥ 2d+1 dimensions generically produces a
    diffeomorphic reconstruction of the original attractor.

    For EEG sleep staging, d=4 captures the essential dynamics:
    - Dimensions 1-2: the dominant oscillatory mode (delta/theta)
    - Dimension 3: amplitude modulation (spindles, K-complexes)
    - Dimension 4: nonlinear coupling between scales

    Parameters
    ----------
    signal : 1D time series of shape (n_samples,)
    dimension : embedding dimension (d=4 for the "4D spiral")
    tau : delay in samples (None = auto-estimate via MI)
    n_bins : bins for MI estimation
    max_tau_search : max lag to search for optimal tau

    Returns
    -------
    embedded : array of shape (n_points, dimension) — the point cloud
    tau_used : the delay actually used
    """
    if tau is None:
        tau = estimate_optimal_tau(signal, max_tau_search, n_bins)

    n_points = len(signal) - (dimension - 1) * tau

    if n_points <= 0:
        raise ValueError(
            f"Signal too short ({len(signal)} samples) for "
            f"embedding with d={dimension}, τ={tau}. "
            f"Need at least {(dimension - 1) * tau + 1} samples."
        )

    # Build delay matrix: each row is [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]
    embedded = np.zeros((n_points, dimension))
    for i in range(dimension):
        embedded[:, i] = signal[i * tau : i * tau + n_points]

    return embedded, tau
