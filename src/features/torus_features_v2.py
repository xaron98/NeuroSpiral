"""Extended geometric features for the Clifford torus trajectory (Paper #2).

Builds on Paper #1's five continuous geometric features
(omega1, omega1/omega2, phase_diff_std, phase_coherence, transition_rate)
with six new feature families (10 scalar values total) designed to close the
information-exploitation gap using temporal deep learning.

All features derive from theta(t) and phi(t) angular time series computed via
atan2(emb[:,1], emb[:,0]) and atan2(emb[:,3], emb[:,2]) from the 4D Takens
embedding with d=4, tau=25.

Features
--------
angular_acceleration     : np.var(dtheta) — oscillatory speed variability
geodesic_distance        : total arc length on torus surface per epoch
angular_entropy          : Shannon entropy of theta in 16 bins
theta_harmonic_1..4      : Fourier harmonic magnitudes of theta(t)
torus_curvature          : mean absolute second difference of theta
angular_range            : circular range of theta (smallest covering arc)
"""

from __future__ import annotations

import numpy as np


def _wrap(delta: np.ndarray) -> np.ndarray:
    """Wrap angular differences to [-pi, pi]."""
    return (delta + np.pi) % (2 * np.pi) - np.pi


def _circular_range(angles: np.ndarray) -> float:
    """Compute the circular range: length of the smallest arc covering all angles.

    Algorithm: sort angles on [0, 2pi), compute gaps between consecutive
    angles (including the wrap-around gap), then range = 2pi - max_gap.
    """
    if len(angles) < 2:
        return 0.0
    a = np.mod(angles, 2 * np.pi)
    a_sorted = np.sort(a)
    gaps = np.diff(a_sorted)
    # Include the wrap-around gap
    wrap_gap = 2 * np.pi - a_sorted[-1] + a_sorted[0]
    max_gap = max(float(np.max(gaps)), float(wrap_gap))
    return 2 * np.pi - max_gap


def extract_torus_features_v2(embedding: np.ndarray) -> dict[str, float]:
    """Extract extended geometric features from a 4D Takens embedding.

    Parameters
    ----------
    embedding : np.ndarray of shape (n_points, 4)
        The 4D time-delay embedding.  Columns correspond to
        [x(t), x(t+tau), x(t+2tau), x(t+3tau)].

    Returns
    -------
    dict[str, float]
        Feature name -> scalar value.  Keys:
        angular_acceleration, geodesic_distance, angular_entropy,
        theta_harmonic_1 .. theta_harmonic_4, torus_curvature,
        angular_range.

    Notes
    -----
    Returns zero-valued dict for degenerate inputs (< 10 points, NaN/inf,
    constant signal).
    """
    # --- defaults (returned on degenerate input) ---
    default = {
        "angular_acceleration": 0.0,
        "geodesic_distance": 0.0,
        "angular_entropy": 0.0,
        "theta_harmonic_1": 0.0,
        "theta_harmonic_2": 0.0,
        "theta_harmonic_3": 0.0,
        "theta_harmonic_4": 0.0,
        "torus_curvature": 0.0,
        "angular_range": 0.0,
    }

    # --- guard: shape / size ---
    if embedding.ndim != 2 or embedding.shape[1] < 4 or embedding.shape[0] < 10:
        return dict(default)

    # --- guard: NaN / inf ---
    if not np.all(np.isfinite(embedding)):
        return dict(default)

    # --- extract angles ---
    theta = np.arctan2(embedding[:, 1], embedding[:, 0])  # [-pi, pi]
    phi = np.arctan2(embedding[:, 3], embedding[:, 2])

    # --- guard: constant signal (all identical angles) ---
    if np.ptp(theta) == 0.0 and np.ptp(phi) == 0.0:
        return dict(default)

    # --- circular-wrapped angular differences ---
    dtheta = _wrap(np.diff(theta))
    dphi = _wrap(np.diff(phi))

    features: dict[str, float] = {}

    # 1. Angular acceleration — variance of angular velocity
    features["angular_acceleration"] = float(np.var(dtheta))

    # 2. Geodesic distance — total arc length on flat torus
    features["geodesic_distance"] = float(
        np.sum(np.sqrt(dtheta ** 2 + dphi ** 2))
    )

    # 3. Angular entropy — Shannon entropy of theta in 16 bins
    counts, _ = np.histogram(theta, bins=16, range=(-np.pi, np.pi))
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total > 0:
        probs = counts / total
        probs = probs[probs > 0]
        features["angular_entropy"] = float(-np.sum(probs * np.log2(probs)))
    else:
        features["angular_entropy"] = 0.0

    # 4. Fourier harmonics of theta(t) — magnitudes of harmonics 1-4
    spectrum = np.abs(np.fft.rfft(theta))
    for k in range(1, 5):
        if k < len(spectrum):
            features[f"theta_harmonic_{k}"] = float(spectrum[k])
        else:
            features[f"theta_harmonic_{k}"] = 0.0

    # 5. Torus curvature — mean |second difference| of theta angular series
    if len(dtheta) >= 2:
        features["torus_curvature"] = float(np.mean(np.abs(np.diff(dtheta))))
    else:
        features["torus_curvature"] = 0.0

    # 6. Angular range — circular range of theta
    features["angular_range"] = float(_circular_range(theta))

    return features
