"""Tests for Clifford torus feature extraction."""

import numpy as np
import pytest
from neurospiral.torus import torus_features, TorusEmbedding, FEATURE_NAMES


def _make_embedding(n=500, freq1=1.0, freq2=0.7):
    """Create a synthetic 4D embedding with known properties."""
    t = np.linspace(0, 10, n)
    return np.column_stack([
        np.cos(2 * np.pi * freq1 * t),
        np.sin(2 * np.pi * freq1 * t),
        np.cos(2 * np.pi * freq2 * t),
        np.sin(2 * np.pi * freq2 * t),
    ])


def test_feature_count():
    """Returns exactly 8 features."""
    emb = _make_embedding()
    feats = torus_features(emb)
    assert feats is not None
    assert len(feats) == 8


def test_feature_names():
    """FEATURE_NAMES has 8 entries."""
    assert len(FEATURE_NAMES) == 8


def test_omega1_positive():
    """omega1 is positive for a rotating trajectory."""
    emb = _make_embedding()
    feats = torus_features(emb)
    assert feats[0] > 0  # omega1


def test_entropy_bounded():
    """Angular entropy is bounded [0, log2(16)]."""
    emb = _make_embedding()
    feats = torus_features(emb)
    assert 0 <= feats[4] <= np.log2(16) + 0.01


def test_coherence_bounded():
    """Phase coherence R in [0, 1]."""
    emb = _make_embedding()
    feats = torus_features(emb)
    assert 0 <= feats[6] <= 1.01


def test_transition_rate_bounded():
    """Transition rate in [0, 1]."""
    emb = _make_embedding()
    feats = torus_features(emb)
    assert 0 <= feats[7] <= 1.01


def test_degenerate_input():
    """Returns None for degenerate inputs."""
    assert torus_features(np.zeros((100, 4))) is None
    assert torus_features(np.ones((5, 4))) is None
    assert torus_features(None) is None


def test_torus_embedding_api():
    """TorusEmbedding class works end-to-end."""
    signal = np.sin(np.linspace(0, 20 * np.pi, 3000)) + 0.1 * np.random.randn(3000)
    torus = TorusEmbedding(d=4, tau=25)
    feats = torus.extract_features(signal)
    assert feats is not None
    assert len(feats) == 8


def test_multiscale():
    """Multi-scale extraction returns n_taus * 8 features."""
    signal = np.sin(np.linspace(0, 20 * np.pi, 3000)) + 0.1 * np.random.randn(3000)
    torus = TorusEmbedding(d=4, taus=[10, 25, 40])
    feats = torus.extract_features(signal)
    assert feats is not None
    assert len(feats) == 24


def test_different_frequencies_different_omega1():
    """Higher frequency signal produces higher omega1."""
    emb_slow = _make_embedding(freq1=0.5)
    emb_fast = _make_embedding(freq1=3.0)
    f_slow = torus_features(emb_slow)
    f_fast = torus_features(emb_fast)
    assert f_fast[0] > f_slow[0]
