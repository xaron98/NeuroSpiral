"""Topological Data Analysis (TDA) features from phase-space embeddings.

Computes persistent homology on the 4D point cloud from Takens embedding,
extracting features from persistence diagrams that capture the geometric
"signature" of each sleep stage:

- N3 (slow-wave): dominant H1 loops (delta oscillation creates stable
  closed orbits), low topological entropy (regular, predictable)
- REM: moderate H1 with higher entropy (mixed frequencies)
- Wake: high-dimensional noise, many short-lived features, high entropy
- N1/N2: transitional — intermediate persistence and entropy

Uses Vietoris-Rips filtration on subsampled point clouds for computational
tractability (30s epoch @ 100Hz = 3000 points → subsample to ~300).
"""

from __future__ import annotations

import numpy as np

# Try giotto-tda first, fall back to ripser
try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import (
        PersistenceEntropy,
        Amplitude,
        NumberOfPoints,
        Filtering,
    )

    HAS_GTDA = True
except ImportError:
    HAS_GTDA = False

try:
    import ripser

    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


def subsample_point_cloud(
    points: np.ndarray,
    n_target: int = 300,
    method: str = "stride",
) -> np.ndarray:
    """Reduce point cloud size for tractable homology computation.

    Vietoris-Rips complexity is O(n^3) in the worst case.
    Subsampling to ~300 points keeps computation under ~1s per epoch
    while preserving topological structure.

    Parameters
    ----------
    points : (n_points, d) point cloud
    n_target : target number of points
    method : 'stride' (uniform) or 'random'
    """
    n = points.shape[0]
    if n <= n_target:
        return points

    if method == "stride":
        stride = max(1, n // n_target)
        return points[::stride][:n_target]
    else:
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=n_target, replace=False)
        indices.sort()
        return points[indices]


def _compute_persistence_ripser(
    points: np.ndarray,
    max_dim: int = 2,
    max_edge: float = 2.0,
) -> list[np.ndarray]:
    """Compute persistent homology using ripser.

    Returns list of persistence diagrams, one per homology dimension.
    Each diagram is (n_features, 2) with [birth, death] pairs.
    """
    result = ripser.ripser(
        points,
        maxdim=max_dim,
        thresh=max_edge,
    )
    return result["dgms"]


def _compute_persistence_gtda(
    points: np.ndarray,
    max_dim: int = 2,
    max_edge: float = 2.0,
) -> list[np.ndarray]:
    """Compute persistent homology using giotto-tda.

    Returns list of persistence diagrams per dimension.
    """
    vr = VietorisRipsPersistence(
        homology_dimensions=list(range(max_dim + 1)),
        max_edge_length=max_edge,
        n_jobs=1,
    )
    # gtda expects (n_samples, n_points, n_dims)
    diagrams = vr.fit_transform(points[np.newaxis, :, :])

    # Split into per-dimension diagrams
    result = []
    for dim in range(max_dim + 1):
        mask = diagrams[0, :, 2] == dim
        pairs = diagrams[0, mask, :2]
        # Remove padding (gtda pads with [0, 0])
        valid = ~((pairs[:, 0] == 0) & (pairs[:, 1] == 0))
        result.append(pairs[valid])

    return result


def compute_persistence_diagrams(
    points: np.ndarray,
    max_dim: int = 2,
    max_edge: float = 2.0,
    n_subsample: int = 300,
) -> list[np.ndarray]:
    """Compute persistent homology on a point cloud.

    Automatically selects available backend (giotto-tda or ripser).

    Parameters
    ----------
    points : (n_points, d) point cloud from Takens embedding
    max_dim : maximum homology dimension (0=components, 1=loops, 2=voids)
    max_edge : maximum filtration value (Rips complex edge length)
    n_subsample : subsample to this many points first

    Returns
    -------
    List of persistence diagrams, one per dimension.
    Each is an array of shape (n_features, 2) with [birth, death].
    """
    # Normalize point cloud to unit variance per dimension
    std = points.std(axis=0)
    std[std == 0] = 1.0
    points_norm = (points - points.mean(axis=0)) / std

    # Subsample for tractability
    points_sub = subsample_point_cloud(points_norm, n_subsample)

    if HAS_RIPSER:
        return _compute_persistence_ripser(points_sub, max_dim, max_edge)
    elif HAS_GTDA:
        return _compute_persistence_gtda(points_sub, max_dim, max_edge)
    else:
        raise ImportError(
            "Neither ripser nor giotto-tda is installed. "
            "Install with: pip install ripser  OR  pip install giotto-tda"
        )


# --- Feature extraction from persistence diagrams ---


def persistence_entropy(diagram: np.ndarray) -> float:
    """Shannon entropy of the persistence diagram.

    Measures the "complexity" of topological features:
    low entropy = one dominant feature (regular oscillation)
    high entropy = many features of similar persistence (noise/chaos)
    """
    if len(diagram) == 0:
        return 0.0

    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return 0.0

    probs = lifetimes / lifetimes.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))


def persistence_statistics(diagram: np.ndarray) -> dict[str, float]:
    """Extract statistical features from a persistence diagram.

    Returns mean, std, max, sum of lifetimes, plus birth/death stats.
    """
    if len(diagram) == 0:
        return {
            "n_features": 0,
            "lifetime_mean": 0.0,
            "lifetime_std": 0.0,
            "lifetime_max": 0.0,
            "lifetime_sum": 0.0,
            "birth_mean": 0.0,
            "death_mean": 0.0,
        }

    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return {
            "n_features": 0,
            "lifetime_mean": 0.0,
            "lifetime_std": 0.0,
            "lifetime_max": 0.0,
            "lifetime_sum": 0.0,
            "birth_mean": 0.0,
            "death_mean": 0.0,
        }

    return {
        "n_features": float(len(lifetimes)),
        "lifetime_mean": float(np.mean(lifetimes)),
        "lifetime_std": float(np.std(lifetimes)),
        "lifetime_max": float(np.max(lifetimes)),
        "lifetime_sum": float(np.sum(lifetimes)),
        "birth_mean": float(np.mean(diagram[:, 0])),
        "death_mean": float(np.mean(diagram[:, 1])),
    }


def betti_curve(
    diagram: np.ndarray,
    n_bins: int = 20,
    filtration_range: tuple[float, float] = (0.0, 2.0),
) -> np.ndarray:
    """Compute Betti number as function of filtration parameter.

    β_k(ε) = number of k-dimensional holes alive at scale ε.

    The Betti curve captures *when* topological features appear
    and disappear across scales — N3's stable delta loops produce
    a characteristic bump in β_1 at intermediate scales.

    Returns a 1D vector of length n_bins.
    """
    edges = np.linspace(filtration_range[0], filtration_range[1], n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    curve = np.zeros(n_bins)

    if len(diagram) == 0:
        return curve

    for birth, death in diagram:
        if death <= birth:
            continue
        alive = (centers >= birth) & (centers < death)
        curve[alive] += 1

    return curve


def extract_tda_features(
    points: np.ndarray,
    max_dim: int = 2,
    max_edge: float = 2.0,
    n_subsample: int = 300,
    betti_bins: int = 20,
) -> dict[str, float]:
    """Full TDA feature extraction from a phase-space point cloud.

    Computes persistent homology and extracts:
    - Per-dimension: entropy, lifetime stats, n_features
    - Betti curves (vectorized as features)
    - Cross-dimension ratios (H1/H0 persistence ratio)

    Parameters
    ----------
    points : (n_points, d) from Takens embedding
    max_dim : max homology dimension
    max_edge : Rips complex max edge length
    n_subsample : subsample size
    betti_bins : resolution of Betti curves

    Returns
    -------
    Dict of feature_name → float value.
    """
    diagrams = compute_persistence_diagrams(
        points, max_dim, max_edge, n_subsample
    )

    features: dict[str, float] = {}

    for dim, dgm in enumerate(diagrams):
        prefix = f"H{dim}"

        # Persistence entropy
        features[f"{prefix}_entropy"] = persistence_entropy(dgm)

        # Lifetime statistics
        stats = persistence_statistics(dgm)
        for key, val in stats.items():
            features[f"{prefix}_{key}"] = val

        # Betti curve (flattened)
        curve = betti_curve(dgm, betti_bins, (0.0, max_edge))
        for b, val in enumerate(curve):
            features[f"{prefix}_betti_{b}"] = val

    # Cross-dimension ratios
    eps = 1e-10
    h0_sum = features.get("H0_lifetime_sum", 0.0)
    h1_sum = features.get("H1_lifetime_sum", 0.0)
    features["H1_H0_ratio"] = h1_sum / (h0_sum + eps)

    if max_dim >= 2:
        h2_sum = features.get("H2_lifetime_sum", 0.0)
        features["H2_H1_ratio"] = h2_sum / (h1_sum + eps)

    return features
