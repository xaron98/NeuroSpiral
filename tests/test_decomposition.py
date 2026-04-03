"""Tests for REM decomposition module."""

import numpy as np
from neurospiral.decomposition import compute_beta, compute_class_decomposition


def test_beta_midpoint():
    """Target at midpoint of axis gives beta=0.5."""
    centroids = {
        0: np.array([0.0, 0.0]),  # wake
        3: np.array([2.0, 0.0]),  # N3
        4: np.array([1.0, 0.0]),  # REM at midpoint
    }
    beta, gd, ratio = compute_beta(centroids)
    assert abs(beta - 0.5) < 1e-10
    assert gd < 1e-10  # no orthogonal component


def test_beta_with_residual():
    """Target off-axis produces nonzero gamma/d."""
    centroids = {
        0: np.array([0.0, 0.0]),
        3: np.array([2.0, 0.0]),
        4: np.array([1.0, 1.0]),  # off axis
    }
    beta, gd, ratio = compute_beta(centroids)
    assert abs(beta - 0.5) < 1e-10
    assert gd > 0.3


def test_missing_class():
    """Returns None if required class is missing."""
    centroids = {0: np.array([0.0]), 3: np.array([1.0])}
    assert compute_beta(centroids) is None


def test_class_decomposition():
    """Decomposition produces valid beta and gamma/d."""
    centroids = {
        "W": np.array([1.0, 0.0, 0.0]),
        "N2": np.array([0.0, 1.0, 0.0]),
        "N3": np.array([0.0, 0.0, 1.0]),
    }
    result = compute_class_decomposition(centroids)
    assert len(result) == 3
    for cls in result:
        assert "beta" in result[cls]
        assert "gamma_d" in result[cls]
