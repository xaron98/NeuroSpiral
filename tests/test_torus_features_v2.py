"""Unit tests for torus_features_v2 extended geometric features.

Test signals
------------
1. Pure sine        — single frequency, predictable winding
2. Two-tone         — quasiperiodic, two incommensurate frequencies
3. Constant         — degenerate, all features should be zero
4. White noise      — random trajectory, high entropy / curvature
5. Edge cases       — short embedding, NaN/inf, 1-D input
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.torus_features_v2 import extract_torus_features_v2, _circular_range


# ─── Helpers ────────────────────────────────────────────────────

EXPECTED_KEYS = {
    "angular_acceleration",
    "geodesic_distance",
    "angular_entropy",
    "theta_harmonic_1",
    "theta_harmonic_2",
    "theta_harmonic_3",
    "theta_harmonic_4",
    "torus_curvature",
    "angular_range",
}


def _make_embedding_from_angles(
    theta: np.ndarray,
    phi: np.ndarray,
    R: float = np.sqrt(2),
) -> np.ndarray:
    """Build a 4D torus embedding from angle series."""
    return np.column_stack([
        R * np.cos(theta),
        R * np.sin(theta),
        R * np.cos(phi),
        R * np.sin(phi),
    ])


def _make_sine_embedding(
    n: int = 500,
    freq_theta: float = 1.0,
    freq_phi: float = 1.0,
) -> np.ndarray:
    """Pure sine: both angles are simple sinusoids."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta = freq_theta * t
    phi = freq_phi * t
    return _make_embedding_from_angles(theta, phi)


def _make_twotone_embedding(
    n: int = 500,
    f1: float = 1.0,
    f2: float = np.sqrt(2),
) -> np.ndarray:
    """Two-tone quasiperiodic: incommensurate theta and phi frequencies."""
    t = np.linspace(0, 4 * np.pi, n, endpoint=False)
    theta = f1 * t
    phi = f2 * t
    return _make_embedding_from_angles(theta, phi)


def _make_constant_embedding(n: int = 100) -> np.ndarray:
    """Constant signal: all points identical."""
    R = np.sqrt(2)
    row = np.array([R, 0.0, R, 0.0])
    return np.tile(row, (n, 1))


def _make_noise_embedding(n: int = 500, seed: int = 42) -> np.ndarray:
    """White noise angles."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    return _make_embedding_from_angles(theta, phi)


# ─── Test: return shape and keys ────────────────────────────────

class TestReturnStructure:
    def test_returns_dict_with_all_keys(self):
        emb = _make_sine_embedding()
        feats = extract_torus_features_v2(emb)
        assert isinstance(feats, dict)
        assert set(feats.keys()) == EXPECTED_KEYS

    def test_all_values_are_float(self):
        emb = _make_sine_embedding()
        feats = extract_torus_features_v2(emb)
        for k, v in feats.items():
            assert isinstance(v, float), f"{k} is {type(v)}, expected float"

    def test_all_values_finite(self):
        for emb in [
            _make_sine_embedding(),
            _make_twotone_embedding(),
            _make_noise_embedding(),
        ]:
            feats = extract_torus_features_v2(emb)
            for k, v in feats.items():
                assert math.isfinite(v), f"{k} = {v} is not finite"


# ─── Test: edge cases ───────────────────────────────────────────

class TestEdgeCases:
    def test_short_embedding_returns_zeros(self):
        emb = np.zeros((5, 4))
        feats = extract_torus_features_v2(emb)
        assert set(feats.keys()) == EXPECTED_KEYS
        assert all(v == 0.0 for v in feats.values())

    def test_constant_signal_returns_zeros(self):
        emb = _make_constant_embedding()
        feats = extract_torus_features_v2(emb)
        assert set(feats.keys()) == EXPECTED_KEYS
        assert all(v == 0.0 for v in feats.values())

    def test_nan_input_returns_zeros(self):
        emb = np.full((100, 4), np.nan)
        feats = extract_torus_features_v2(emb)
        assert all(v == 0.0 for v in feats.values())

    def test_inf_input_returns_zeros(self):
        emb = np.full((100, 4), np.inf)
        feats = extract_torus_features_v2(emb)
        assert all(v == 0.0 for v in feats.values())

    def test_wrong_ndim_returns_zeros(self):
        feats = extract_torus_features_v2(np.zeros((100,)))
        assert all(v == 0.0 for v in feats.values())

    def test_too_few_columns_returns_zeros(self):
        feats = extract_torus_features_v2(np.zeros((100, 3)))
        assert all(v == 0.0 for v in feats.values())

    def test_exactly_10_points(self):
        emb = _make_sine_embedding(n=10)
        feats = extract_torus_features_v2(emb)
        assert set(feats.keys()) == EXPECTED_KEYS
        # Should compute without error; geodesic distance > 0 for sine
        assert feats["geodesic_distance"] > 0.0


# ─── Test: pure sine signal ─────────────────────────────────────

class TestPureSine:
    """Pure sine winding: theta = t, phi = t. Predictable geometry."""

    @pytest.fixture
    def feats(self):
        return extract_torus_features_v2(_make_sine_embedding(n=500))

    def test_geodesic_distance_positive(self, feats):
        assert feats["geodesic_distance"] > 0.0

    def test_angular_acceleration_low(self, feats):
        # Constant angular velocity -> near-zero variance
        assert feats["angular_acceleration"] < 1e-4

    def test_angular_entropy_high(self, feats):
        # Uniform-ish coverage of theta over [−π, π]
        # Max entropy for 16 bins = log2(16) = 4.0
        assert feats["angular_entropy"] > 3.0

    def test_harmonics_first_dominant(self, feats):
        # Pure winding should have energy at harmonic 1
        assert feats["theta_harmonic_1"] > 0.0

    def test_torus_curvature_low(self, feats):
        # Nearly constant angular velocity => low curvature
        assert feats["torus_curvature"] < 0.01

    def test_angular_range_near_full(self, feats):
        # Full winding covers most of the circle
        assert feats["angular_range"] > 5.0  # close to 2*pi


# ─── Test: two-tone quasiperiodic ───────────────────────────────

class TestTwoTone:
    """Incommensurate theta and phi: quasiperiodic winding."""

    @pytest.fixture
    def feats(self):
        return extract_torus_features_v2(_make_twotone_embedding(n=500))

    def test_geodesic_distance_larger_than_sine(self, feats):
        sine_feats = extract_torus_features_v2(_make_sine_embedding(n=500))
        # Two-tone with larger traversal in phi => more total arc
        assert feats["geodesic_distance"] > sine_feats["geodesic_distance"] * 0.5

    def test_angular_entropy_high(self, feats):
        assert feats["angular_entropy"] > 2.5

    def test_harmonics_present(self, feats):
        assert feats["theta_harmonic_1"] > 0.0


# ─── Test: white noise ──────────────────────────────────────────

class TestNoise:
    """Random angles: high entropy, high curvature."""

    @pytest.fixture
    def feats(self):
        return extract_torus_features_v2(_make_noise_embedding(n=500))

    def test_high_entropy(self, feats):
        assert feats["angular_entropy"] > 3.5

    def test_high_curvature(self, feats):
        # Random jumps => high second-difference
        assert feats["torus_curvature"] > 0.5

    def test_high_angular_acceleration(self, feats):
        # Random angular velocities => high variance
        assert feats["angular_acceleration"] > 0.5

    def test_angular_range_near_full(self, feats):
        # Noise covers most of the circle
        assert feats["angular_range"] > 5.5


# ─── Test: noise vs sine discrimination ─────────────────────────

class TestDiscrimination:
    """Key discriminative properties between signal types."""

    def test_noise_higher_curvature_than_sine(self):
        sine = extract_torus_features_v2(_make_sine_embedding(n=500))
        noise = extract_torus_features_v2(_make_noise_embedding(n=500))
        assert noise["torus_curvature"] > sine["torus_curvature"] * 10

    def test_noise_higher_acceleration_than_sine(self):
        sine = extract_torus_features_v2(_make_sine_embedding(n=500))
        noise = extract_torus_features_v2(_make_noise_embedding(n=500))
        assert noise["angular_acceleration"] > sine["angular_acceleration"] * 10


# ─── Test: _circular_range helper ───────────────────────────────

class TestCircularRange:
    def test_single_point(self):
        assert _circular_range(np.array([1.0])) == 0.0

    def test_two_opposite_points(self):
        # Points at 0 and pi => range = pi
        r = _circular_range(np.array([0.0, np.pi]))
        assert abs(r - np.pi) < 1e-10

    def test_full_circle(self):
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        r = _circular_range(angles)
        # Should be close to 2*pi * (99/100)
        assert r > 2 * np.pi * 0.95

    def test_clustered_points(self):
        angles = np.array([0.1, 0.2, 0.15, 0.12])
        r = _circular_range(angles)
        assert abs(r - 0.1) < 1e-10

    def test_empty_array(self):
        assert _circular_range(np.array([])) == 0.0
