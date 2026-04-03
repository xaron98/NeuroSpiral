"""Tests for Takens embedding module."""

import numpy as np
import pytest
from neurospiral.embedding import time_delay_embedding, estimate_optimal_tau


def test_embedding_shape():
    """Embedding output has correct shape."""
    signal = np.sin(np.linspace(0, 10 * np.pi, 1000))
    emb, tau = time_delay_embedding(signal, dimension=4, tau=10)
    assert emb.shape == (1000 - 3 * 10, 4)
    assert tau == 10


def test_embedding_auto_tau():
    """Auto tau estimation returns reasonable value."""
    signal = np.sin(np.linspace(0, 20 * np.pi, 2000)) + 0.1 * np.random.randn(2000)
    tau = estimate_optimal_tau(signal, max_lag=50)
    assert 1 <= tau <= 50


def test_embedding_short_signal():
    """Short signal raises ValueError."""
    signal = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        time_delay_embedding(signal, dimension=4, tau=10)


def test_embedding_delay_structure():
    """Each column is a delayed version of the signal."""
    signal = np.arange(100, dtype=float)
    tau = 5
    emb, _ = time_delay_embedding(signal, dimension=4, tau=tau)
    for i in range(4):
        np.testing.assert_array_equal(emb[:, i], signal[i * tau: i * tau + len(emb)])
