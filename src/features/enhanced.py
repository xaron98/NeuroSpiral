"""Enhanced geometric features for NeuroSpiral.

Four improvements over baseline:
A) Multi-channel Takens embedding (2 channels → ℝ⁴ natural)
B) Asymmetric winding numbers (fix ω₁/ω₂ always = 1.0)
C) Adaptive discretization (k-means on torus, cross-validated)
D) Transition bigram features (vertex_t-1, vertex_t patterns)
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


# ═══════════════════════════════════════════════════════════
# A) Multi-channel Takens embedding
# Instead of d=4 from 1 channel, take d=2 from each of 2 channels
# This captures inter-hemispheric coherence naturally
# ═══════════════════════════════════════════════════════════

def multichannel_takens_embed(
    ch1: np.ndarray,
    ch2: np.ndarray,
    tau: int = 25,
) -> np.ndarray:
    """Embed 2 EEG channels into ℝ⁴ using cross-channel delay embedding.

    ch1, ch2: raw EEG signals (same length)
    tau: time delay in samples

    Returns (N-tau, 4) array where:
      dims 0,1 = ch1(t), ch1(t+τ)   → first torus plane (θ)
      dims 2,3 = ch2(t), ch2(t+τ)   → second torus plane (φ)

    This means ω₁ captures ch1 dynamics and ω₂ captures ch2 dynamics.
    When ch1=C4 (right) and ch2=C3 (left), the ratio ω₁/ω₂ measures
    inter-hemispheric asymmetry — which changes between sleep stages.
    """
    n = len(ch1)
    if len(ch2) != n:
        raise ValueError("Channels must have same length")

    N = n - tau
    if N <= 0:
        raise ValueError(f"Signal too short for tau={tau}")

    embedded = np.zeros((N, 4))
    embedded[:, 0] = ch1[:N]          # ch1(t)
    embedded[:, 1] = ch1[tau:tau+N]   # ch1(t+τ)
    embedded[:, 2] = ch2[:N]          # ch2(t)
    embedded[:, 3] = ch2[tau:tau+N]   # ch2(t+τ)

    return embedded


# ═══════════════════════════════════════════════════════════
# B) Asymmetric winding numbers
# With multi-channel, ω₁ and ω₂ are naturally different
# because they measure rotation in ch1-plane vs ch2-plane
# ═══════════════════════════════════════════════════════════

def compute_winding_asymmetric(projected: np.ndarray) -> dict:
    """Compute winding numbers from torus-projected data.

    projected: (N, 4) array on the Clifford torus

    Returns dict with:
      omega1: rotation speed in ch1 plane (dims 0,1)
      omega2: rotation speed in ch2 plane (dims 2,3)
      ratio: ω₁/ω₂ (asymmetry measure)
      phase_coherence: how synchronized the two planes are
    """
    theta = np.arctan2(projected[:, 1], projected[:, 0])
    phi = np.arctan2(projected[:, 3], projected[:, 2])

    # Unwrap to get continuous phase
    theta_unwrapped = np.unwrap(theta)
    phi_unwrapped = np.unwrap(phi)

    # Instantaneous angular velocities
    dtheta = np.diff(theta_unwrapped)
    dphi = np.diff(phi_unwrapped)

    omega1 = np.mean(np.abs(dtheta))
    omega2 = np.mean(np.abs(dphi))

    ratio = omega1 / (omega2 + 1e-10)

    # Phase coherence: correlation between instantaneous velocities
    # High coherence = channels move together = symmetric brain state
    # Low coherence = channels diverge = asymmetric (e.g., REM lateralization)
    if len(dtheta) > 1:
        coherence = np.abs(np.corrcoef(dtheta, dphi)[0, 1])
    else:
        coherence = 0.0

    # Phase difference variability (low = locked, high = drifting)
    phase_diff = theta_unwrapped - phi_unwrapped
    phase_diff_std = np.std(np.diff(phase_diff))

    return {
        "omega1": omega1,
        "omega2": omega2,
        "ratio": ratio,
        "phase_coherence": coherence,
        "phase_diff_std": phase_diff_std,
    }


# ═══════════════════════════════════════════════════════════
# C) Adaptive discretization
# K-means on torus angles instead of fixed sgn(x) threshold
# Learns optimal boundaries from the data
# ═══════════════════════════════════════════════════════════

def adaptive_discretize_fit(
    torus_angles: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 42,
) -> KMeans:
    """Fit adaptive discretization on training data.

    torus_angles: (N, 2) array of (θ, φ) angles on the torus
    n_clusters: number of discrete states (default 8)

    Returns fitted KMeans model.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(torus_angles)
    return km


def adaptive_discretize_predict(
    torus_angles: np.ndarray,
    km_model: KMeans,
) -> np.ndarray:
    """Assign torus angles to learned clusters.

    torus_angles: (N, 2) array of (θ, φ) angles
    km_model: fitted KMeans from adaptive_discretize_fit

    Returns integer cluster labels.
    """
    return km_model.predict(torus_angles)


def torus_to_angles(projected: np.ndarray) -> np.ndarray:
    """Convert torus-projected points to (θ, φ) angle pairs.

    projected: (N, 4) array on the Clifford torus

    Returns (N, 2) array of angles in [-π, π].
    """
    theta = np.arctan2(projected[:, 1], projected[:, 0])
    phi = np.arctan2(projected[:, 3], projected[:, 2])
    return np.column_stack([theta, phi])


# ═══════════════════════════════════════════════════════════
# D) Transition bigram features
# Captures temporal patterns in vertex sequences
# ═══════════════════════════════════════════════════════════

def compute_transition_features(
    vertex_sequence: np.ndarray,
    n_vertices: int = 16,
) -> dict:
    """Compute transition-based features from a vertex sequence.

    vertex_sequence: sequence of vertex IDs within one epoch
    n_vertices: total number of possible vertices

    Returns dict with:
      n_transitions: number of vertex changes
      transition_rate: fraction of timesteps with vertex change
      dominant_bigram: most common (v_t, v_{t+1}) pair as single int
      bigram_entropy: Shannon entropy of bigram distribution
      self_loop_frac: fraction of time staying at same vertex
      unique_vertices: number of distinct vertices visited
    """
    if len(vertex_sequence) < 2:
        return {
            "n_transitions": 0,
            "transition_rate": 0.0,
            "dominant_bigram": 0,
            "bigram_entropy": 0.0,
            "self_loop_frac": 1.0,
            "unique_vertices": 1,
        }

    # Transitions
    changes = np.diff(vertex_sequence) != 0
    n_transitions = int(np.sum(changes))
    transition_rate = n_transitions / (len(vertex_sequence) - 1)

    # Bigrams
    bigrams = vertex_sequence[:-1] * n_vertices + vertex_sequence[1:]
    unique_bigrams, counts = np.unique(bigrams, return_counts=True)
    probs = counts / counts.sum()

    dominant_bigram = int(unique_bigrams[np.argmax(counts)])

    # Shannon entropy of bigram distribution
    bigram_entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # Self-loop fraction
    self_loops = np.sum(~changes)
    self_loop_frac = self_loops / (len(vertex_sequence) - 1)

    # Unique vertices
    unique_vertices = len(np.unique(vertex_sequence))

    return {
        "n_transitions": n_transitions,
        "transition_rate": transition_rate,
        "dominant_bigram": dominant_bigram,
        "bigram_entropy": bigram_entropy,
        "self_loop_frac": self_loop_frac,
        "unique_vertices": unique_vertices,
    }


# ═══════════════════════════════════════════════════════════
# Combined feature extraction
# ═══════════════════════════════════════════════════════════

def extract_enhanced_features(
    ch1_epoch: np.ndarray,
    ch2_epoch: np.ndarray | None,
    tau: int = 25,
    sfreq: float = 100.0,
) -> dict:
    """Extract all enhanced geometric features from one epoch.

    ch1_epoch: primary EEG channel (e.g., C4-M1), in µV
    ch2_epoch: secondary EEG channel (e.g., C3-M2), in µV. If None, falls back to single-channel.
    tau: Takens delay
    sfreq: sampling frequency

    Returns dict of features ready for DataFrame row.
    """
    from src.geometry.tesseract import Q_discretize, VERTICES
    from src.features.takens import takens_embed

    # Embedding
    if ch2_epoch is not None:
        embedded = multichannel_takens_embed(ch1_epoch, ch2_epoch, tau)
    else:
        embedded = takens_embed(ch1_epoch, d=4, tau=tau)

    # Torus projection (same as baseline)
    norms_xy = np.sqrt(embedded[:, 0]**2 + embedded[:, 1]**2 + 1e-10)
    norms_zw = np.sqrt(embedded[:, 2]**2 + embedded[:, 3]**2 + 1e-10)
    R = np.sqrt(2.0)
    projected = embedded.copy()
    projected[:, 0] *= R / norms_xy
    projected[:, 1] *= R / norms_xy
    projected[:, 2] *= R / norms_zw
    projected[:, 3] *= R / norms_zw

    # Standard vertex (Q discretization)
    mean_point = np.mean(projected, axis=0)
    vertex = Q_discretize(mean_point)

    # Asymmetric winding numbers
    winding = compute_winding_asymmetric(projected)

    # Vertex sequence for transitions (subsample for speed)
    step = max(1, len(projected) // 100)
    vertex_seq = np.array([Q_discretize(p) for p in projected[::step]])

    # Transition features
    trans = compute_transition_features(vertex_seq, n_vertices=16)

    # Vertex stability
    stability = np.mean(vertex_seq == vertex)

    # Torus angles for adaptive discretization
    angles = torus_to_angles(projected)
    mean_angle = np.mean(angles, axis=0)

    return {
        "vertex": vertex,
        "omega1": winding["omega1"],
        "omega2": winding["omega2"],
        "winding_ratio": winding["ratio"],
        "phase_coherence": winding["phase_coherence"],
        "phase_diff_std": winding["phase_diff_std"],
        "stability": stability,
        "transition_rate": trans["transition_rate"],
        "bigram_entropy": trans["bigram_entropy"],
        "self_loop_frac": trans["self_loop_frac"],
        "unique_vertices": trans["unique_vertices"],
        "n_transitions": trans["n_transitions"],
        "torus_theta": mean_angle[0],
        "torus_phi": mean_angle[1],
    }
