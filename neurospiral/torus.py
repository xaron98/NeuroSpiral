"""Clifford torus projection and the 8 canonical geometric features.

Projects a 4D Takens embedding onto the Clifford torus T^2 = S^1 x S^1
via angle extraction:
    theta = arctan2(v_2, v_1)    (first torus plane)
    phi   = arctan2(v_4, v_3)    (second torus plane)

The 8 features capture rotation speed, curvature, entropy, phase
relationships, and regime transitions of the trajectory on the torus.

References:
    Perea, C.J. (2026). "Toroidal embedding of multimodal polysomnography
    reveals a reproducible geometric gradient of sleep states."
"""

from __future__ import annotations

import numpy as np

from neurospiral.embedding import time_delay_embedding


FEATURE_NAMES = [
    "omega1",              # winding number (angular velocity)
    "torus_curvature",     # mean geodesic curvature
    "angular_acceleration",  # variance of angular velocity
    "geodesic_distance",   # total arc length on torus
    "angular_entropy",     # Shannon entropy of theta
    "phase_diff_std",      # phase difference variability
    "phase_coherence",     # mean resultant length R
    "transition_rate",     # rate of large angular jumps
]


def _wrap(delta: np.ndarray) -> np.ndarray:
    """Wrap angular differences to [-pi, pi]."""
    return (delta + np.pi) % (2 * np.pi) - np.pi


def torus_features(embedding: np.ndarray) -> np.ndarray | None:
    """Extract the 8 canonical torus features from a 4D embedding.

    Parameters
    ----------
    embedding : array of shape (n_points, 4) from Takens embedding.

    Returns
    -------
    Array of shape (8,) with features in order of FEATURE_NAMES,
    or None if the embedding is degenerate.
    """
    if embedding is None or embedding.ndim != 2:
        return None
    if embedding.shape[0] < 20 or embedding.shape[1] < 4:
        return None
    if not np.all(np.isfinite(embedding)):
        return None
    if np.std(embedding) < 1e-15:
        return None

    # Torus angles
    theta = np.arctan2(embedding[:, 1], embedding[:, 0])
    phi = np.arctan2(embedding[:, 3], embedding[:, 2])

    # Wrapped angular differences
    dtheta = _wrap(np.diff(theta))
    dphi = _wrap(np.diff(phi))
    N = len(dtheta)
    if N < 5:
        return None

    feats = []

    # 1. omega1: mean absolute angular velocity
    feats.append(float(np.mean(np.abs(dtheta))))

    # 2. torus_curvature: mean |second difference| of theta
    feats.append(float(np.mean(np.abs(np.diff(dtheta)))))

    # 3. angular_acceleration: variance of angular velocity
    feats.append(float(np.var(dtheta)))

    # 4. geodesic_distance: total arc length on flat torus
    feats.append(float(np.sum(np.sqrt(dtheta**2 + dphi**2))))

    # 5. angular_entropy: Shannon entropy of theta in 16 bins
    counts, _ = np.histogram(theta, bins=16, range=(-np.pi, np.pi))
    c = counts.astype(np.float64)
    total = c.sum()
    if total > 0:
        p = c / total
        p = p[p > 0]
        feats.append(float(-np.sum(p * np.log2(p))))
    else:
        feats.append(0.0)

    # 6. phase_diff_std: circular std of (theta - phi)
    pd = theta - phi
    R_len = np.abs(np.mean(np.exp(1j * pd)))
    feats.append(
        float(np.sqrt(-2 * np.log(max(R_len, 1e-10)))) if R_len < 1 else 0.0
    )

    # 7. phase_coherence: mean resultant length R
    feats.append(float(R_len))

    # 8. transition_rate: fraction of timesteps with vertex change
    signs = (embedding >= 0).astype(int)
    verts = signs[:, 0] * 8 + signs[:, 1] * 4 + signs[:, 2] * 2 + signs[:, 3]
    feats.append(float(np.sum(np.diff(verts) != 0) / max(len(verts) - 1, 1)))

    return np.array(feats, dtype=np.float64)


class TorusEmbedding:
    """High-level API for Clifford torus feature extraction.

    Parameters
    ----------
    d : Embedding dimension (default 4).
    tau : Delay in samples (None = auto-estimate via MI).
    taus : List of delays for multi-scale extraction.
           If provided, overrides ``tau`` and extracts features at each scale.

    Examples
    --------
    >>> torus = TorusEmbedding(d=4, tau=25)
    >>> features = torus.extract_features(signal)
    >>> print(features.shape)  # (8,)

    >>> torus_ms = TorusEmbedding(d=4, taus=[10, 25, 40])
    >>> features_ms = torus_ms.extract_features(signal)
    >>> print(features_ms.shape)  # (24,)
    """

    def __init__(
        self,
        d: int = 4,
        tau: int | None = 25,
        taus: list[int] | None = None,
    ):
        self.d = d
        self.tau = tau
        self.taus = taus or ([tau] if tau else [25])

    def extract_features(self, signal: np.ndarray) -> np.ndarray | None:
        """Extract torus features from a 1D signal.

        Returns array of shape (8 * n_taus,) or None on failure.
        """
        all_feats = []
        for tau in self.taus:
            emb, _ = time_delay_embedding(signal, dimension=self.d, tau=tau)
            f = torus_features(emb)
            if f is None:
                return None
            all_feats.extend(f)
        return np.array(all_feats, dtype=np.float64)

    def extract_features_multichannel(
        self, signals: dict[str, np.ndarray]
    ) -> np.ndarray | None:
        """Extract features from multiple channels (Form A: 1 torus per channel).

        Parameters
        ----------
        signals : dict mapping channel name to 1D signal array.

        Returns array of shape (n_channels * 8 * n_taus,) or None.
        """
        all_feats = []
        for name, signal in signals.items():
            f = self.extract_features(signal)
            if f is None:
                return None
            all_feats.extend(f)
        return np.array(all_feats, dtype=np.float64)

    @property
    def feature_names(self) -> list[str]:
        """Feature names for a single channel."""
        names = []
        for tau in self.taus:
            for fn in FEATURE_NAMES:
                names.append(f"{fn}_t{tau}")
        return names

    def feature_names_multichannel(self, channels: list[str]) -> list[str]:
        """Feature names for multiple channels."""
        names = []
        for ch in channels:
            for tau in self.taus:
                for fn in FEATURE_NAMES:
                    names.append(f"{fn}_{ch}_t{tau}")
        return names
