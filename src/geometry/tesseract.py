"""Tesseract geometry on the Clifford torus.

Mathematical foundation (proven in both reference documents):

    The 16 vertices V = {±1}⁴ of a tesseract satisfy:
        x² + y² = 2  and  z² + w² = 2
    for all vertices, placing them on a Clifford torus 𝕋_{√2}
    embedded in S³(2) ⊂ ℝ⁴.

    Under double rotation R(ω₁t, ω₂t), vertex orbits remain
    on 𝕋_{√2} for all t — the torus is an invariant manifold.

    The sign-quadrant map Q(x) = (sgn(x), sgn(y), sgn(z), sgn(w))
    provides an intrinsic discretizer: as a trajectory winds on
    the torus, it generates a symbol sequence in {±1}⁴ whenever
    it crosses the separating hyperplanes {xᵢ = 0}.

This module implements:
    - Tesseract vertices and their torus-angle representation
    - Q(x) discretizer mapping ℝ⁴ → {±1}⁴ (nearest vertex)
    - Clifford torus projection (Strategy A from the literature)
    - Double rotation R(α,β) ∈ SO(4)
    - Euclidean and torus-geodesic distances to vertices
    - Hamming distance for transition detection
    - Windowed vertex residence and stability scoring
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


# ──────────────────────────────────────────────────────────────
# Tesseract vertices
# ──────────────────────────────────────────────────────────────

def generate_vertices() -> np.ndarray:
    """Generate the 16 vertices of a unit tesseract.

    Returns array of shape (16, 4) with all sign permutations of (±1, ±1, ±1, ±1).
    """
    return np.array(list(itertools.product([-1, 1], repeat=4)), dtype=np.float64)


def generate_24cell_vertices() -> np.ndarray:
    """Generate the 24 vertices of the 24-cell polytope.

    The 24-cell is the provably optimal arrangement of 24 equal regions on S³
    (Musin 2008, Annals of Mathematics). It contains the tesseract's 16 vertices
    plus 8 axis-aligned vertices (permutations of (±1,0,0,0)).

    The 24-cell has minimum angular separation of 60° vs 52.2° for the tesseract,
    reducing quantization distortion by ~8.7% (Agrell & Eriksson 1998, IEEE Trans IT).

    Returns array of shape (24, 4).
    """
    # 16 tesseract vertices: all (±1, ±1, ±1, ±1) normalized to S³
    tesseract = generate_vertices() / 2.0  # normalize: |v| = 1

    # 8 axis-aligned vertices: (±1,0,0,0), (0,±1,0,0), (0,0,±1,0), (0,0,0,±1)
    axes = np.zeros((8, 4), dtype=np.float64)
    for i in range(4):
        axes[2*i, i] = 1.0
        axes[2*i+1, i] = -1.0

    return np.vstack([tesseract, axes])


# Precompute vertices and their torus angles
VERTICES = generate_vertices()  # (16, 4) — tesseract
VERTICES_24CELL = generate_24cell_vertices()  # (24, 4) — 24-cell

# Each vertex (s1,s2,s3,s4) maps to torus angles:
#   θ₀ = atan2(s2, s1) ∈ {±π/4, ±3π/4}
#   φ₀ = atan2(s4, s3) ∈ {±π/4, ±3π/4}
VERTEX_ANGLES = np.column_stack([
    np.arctan2(VERTICES[:, 1], VERTICES[:, 0]),
    np.arctan2(VERTICES[:, 3], VERTICES[:, 2]),
])  # (16, 2)


# Clinical labels for interpretability (learned mapping, not hardcoded)
# These are placeholders — actual assignment must come from supervised training
VERTEX_LABELS = {i: f"V{i:02d}" for i in range(16)}


# ──────────────────────────────────────────────────────────────
# Sign-quadrant discretizer Q(x)
# ──────────────────────────────────────────────────────────────

def discretize(x: np.ndarray) -> np.ndarray:
    """Sign-quadrant discretizer Q: ℝ⁴ → {±1}⁴.

    Maps a continuous 4D point to the nearest tesseract vertex
    by taking the sign of each coordinate.

    On the torus parameterization (R cos θ, R sin θ, R cos φ, R sin φ),
    this partitions the torus into 4×4 = 16 angular quadrants.

    Parameters
    ----------
    x : array of shape (..., 4)

    Returns
    -------
    Signs array of shape (..., 4) with values in {-1, +1}.
    Zero components get +1 (tie-break convention).
    """
    signs = np.sign(x)
    signs[signs == 0] = 1.0
    return signs


def nearest_vertex_idx(x: np.ndarray, vertices: np.ndarray | None = None) -> np.ndarray:
    """Find index of nearest vertex for each point.

    Parameters
    ----------
    x : array of shape (n, 4) or (4,)
    vertices : vertex set to use (default: VERTICES, 16 tesseract)
               pass VERTICES_24CELL for 24-cell

    Returns
    -------
    Integer indices into vertices, shape (n,) or scalar.
    """
    if vertices is None:
        vertices = VERTICES
    x = np.atleast_2d(x)
    dists = np.linalg.norm(x[:, None, :] - vertices[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def Q_discretize(x: np.ndarray, vertices: np.ndarray | None = None) -> int:
    """Sign-quadrant discretizer for a single 4D point.

    Returns scalar vertex index (int). Convenience wrapper around
    nearest_vertex_idx for use in loops over epochs.

    Parameters
    ----------
    x : array of shape (4,)
    vertices : vertex set (default: VERTICES)

    Returns
    -------
    Integer vertex index.
    """
    if vertices is None:
        vertices = VERTICES
    x = np.atleast_2d(x)
    dists = np.linalg.norm(x - vertices, axis=1)
    return int(np.argmin(dists))


def vertex_code(x: np.ndarray) -> np.ndarray:
    """Convert point(s) to vertex code (the {±1}⁴ sign pattern).

    Equivalent to Q(x) but returns actual vertex coordinates
    (snapped to the nearest of the 16 vertices).
    """
    idx = nearest_vertex_idx(x)
    return VERTICES[idx]


# ──────────────────────────────────────────────────────────────
# Clifford torus projection
# ──────────────────────────────────────────────────────────────

def to_torus_angles(x: np.ndarray) -> np.ndarray:
    """Extract torus angles (θ, φ) from 4D coordinates.

    Given x = (x₁, x₂, x₃, x₄), compute:
        θ = atan2(x₂, x₁)   (angle in xy-plane)
        φ = atan2(x₄, x₃)   (angle in zw-plane)

    Parameters
    ----------
    x : array of shape (..., 4)

    Returns
    -------
    Angles of shape (..., 2) with θ, φ ∈ [-π, π].
    """
    theta = np.arctan2(x[..., 1], x[..., 0])
    phi = np.arctan2(x[..., 3], x[..., 2])
    return np.stack([theta, phi], axis=-1)


def project_to_clifford_torus(x: np.ndarray, R: float = np.sqrt(2)) -> np.ndarray:
    """Project 4D points onto the Clifford torus 𝕋_R.

    Strategy A from the literature:
    Split into (x₁,x₂) and (x₃,x₄), extract angles,
    reconstruct on torus of radius R.

    This discards radial information in each 2D subspace
    but guarantees the output lies exactly on 𝕋_R.

    Parameters
    ----------
    x : array of shape (..., 4) — raw 4D coordinates (e.g., from Takens)
    R : torus radius (default √2 for tesseract vertices)

    Returns
    -------
    Projected points of shape (..., 4) on 𝕋_R.
    """
    angles = to_torus_angles(x)
    theta = angles[..., 0]
    phi = angles[..., 1]

    return np.stack([
        R * np.cos(theta),
        R * np.sin(theta),
        R * np.cos(phi),
        R * np.sin(phi),
    ], axis=-1)


def torus_radii(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the two sub-radii r₁ = √(x²+y²) and r₂ = √(z²+w²).

    On a perfect Clifford torus, both should equal R.
    Deviation from R measures how far the point is from the torus.
    """
    r1 = np.sqrt(x[..., 0]**2 + x[..., 1]**2)
    r2 = np.sqrt(x[..., 2]**2 + x[..., 3]**2)
    return r1, r2


# ──────────────────────────────────────────────────────────────
# Double rotation R(α, β) ∈ SO(4)
# ──────────────────────────────────────────────────────────────

def double_rotation_matrix(alpha: float, beta: float) -> np.ndarray:
    """Construct the SO(4) double rotation matrix.

    R(α,β) rotates by α in the xy-plane and β in the zw-plane.

    When α = ω₁t and β = ω₂t:
        ω₁ ↔ Process S (homeostatic, delta-driven)
        ω₂ ↔ Process C (circadian)
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)

    return np.array([
        [ca, -sa, 0,   0],
        [sa,  ca, 0,   0],
        [0,   0,  cb, -sb],
        [0,   0,  sb,  cb],
    ])


def orbit_point(vertex: np.ndarray, omega1: float, omega2: float, t: float) -> np.ndarray:
    """Compute position of a vertex under double rotation at time t.

    x(t) = R(ω₁t, ω₂t) · vertex

    The result always lies on 𝕋_{√2} ∩ S³(2).
    """
    R = double_rotation_matrix(omega1 * t, omega2 * t)
    return R @ vertex


def generate_orbit(
    vertex: np.ndarray,
    omega1: float,
    omega2: float,
    t_max: float,
    n_points: int = 1000,
) -> np.ndarray:
    """Generate a full orbit trajectory on the Clifford torus.

    Returns array of shape (n_points, 4).
    """
    times = np.linspace(0, t_max, n_points)
    trajectory = np.zeros((n_points, 4))
    for i, t in enumerate(times):
        trajectory[i] = orbit_point(vertex, omega1, omega2, t)
    return trajectory


# ──────────────────────────────────────────────────────────────
# Distance metrics
# ──────────────────────────────────────────────────────────────

def _wrap_angle(delta: np.ndarray) -> np.ndarray:
    """Wrap angular difference to [-π, π]."""
    return (delta + np.pi) % (2 * np.pi) - np.pi


def euclidean_distance_to_vertices(x: np.ndarray) -> np.ndarray:
    """Euclidean distance from point(s) to all 16 vertices.

    Parameters
    ----------
    x : shape (4,) or (n, 4)

    Returns
    -------
    Distances of shape (16,) or (n, 16).
    """
    x = np.atleast_2d(x)
    return np.linalg.norm(x[:, None, :] - VERTICES[None, :, :], axis=2).squeeze()


def torus_geodesic_distance(
    x: np.ndarray,
    vertex_idx: int | None = None,
    R: float = np.sqrt(2),
) -> np.ndarray:
    """Geodesic distance on the flat Clifford torus 𝕋_R.

    The induced metric on 𝕋_R = S¹_R × S¹_R is:
        ds² = R² dθ² + R² dφ²

    So the geodesic distance is:
        d(x, v) = R √(wrap(θ-θ_v)² + wrap(φ-φ_v)²)

    Parameters
    ----------
    x : shape (..., 4) — points in ℝ⁴
    vertex_idx : if given, distance to this specific vertex;
                 if None, returns distances to all 16 vertices
    R : torus radius

    Returns
    -------
    If vertex_idx is given: scalar or shape (...,)
    If vertex_idx is None: shape (..., 16)
    """
    angles = to_torus_angles(x)  # (..., 2)

    if vertex_idx is not None:
        v_angles = VERTEX_ANGLES[vertex_idx]  # (2,)
        dtheta = _wrap_angle(angles[..., 0] - v_angles[0])
        dphi = _wrap_angle(angles[..., 1] - v_angles[1])
        return R * np.sqrt(dtheta**2 + dphi**2)
    else:
        # Distance to all 16 vertices
        dtheta = _wrap_angle(angles[..., 0, np.newaxis] - VERTEX_ANGLES[:, 0])
        dphi = _wrap_angle(angles[..., 1, np.newaxis] - VERTEX_ANGLES[:, 1])
        return R * np.sqrt(dtheta**2 + dphi**2)


def hamming_distance(v1: np.ndarray, v2: np.ndarray) -> int:
    """Hamming distance between two vertex codes.

    Distance of 1 = single sign flip = edge traversal.
    Useful for detecting smooth transitions vs jumps.
    """
    return int(np.sum(v1 != v2))


# ──────────────────────────────────────────────────────────────
# Windowed vertex analysis
# ──────────────────────────────────────────────────────────────

@dataclass
class VertexResidence:
    """Analysis of vertex residence over a trajectory window."""

    dominant_vertex: int          # most visited vertex index
    residence_fraction: float     # fraction of time at dominant vertex
    vertex_histogram: np.ndarray  # (16,) visit counts
    mean_distance: float          # mean distance to dominant vertex
    stability_score: float        # 1 - entropy / max_entropy
    transition_count: int         # number of vertex changes
    transition_sequence: list[int]  # sequence of visited vertices


def analyze_vertex_residence(
    trajectory: np.ndarray,
    metric: str = "euclidean",
) -> VertexResidence:
    """Analyze which vertices a trajectory visits and how stably.

    Parameters
    ----------
    trajectory : shape (n_points, 4) — 4D trajectory
    metric : 'euclidean' or 'torus' for distance computation

    Returns
    -------
    VertexResidence with residence statistics.
    """
    n = trajectory.shape[0]

    # Assign each point to nearest vertex
    assignments = nearest_vertex_idx(trajectory)

    # Histogram
    histogram = np.bincount(assignments, minlength=16).astype(float)
    dominant = int(np.argmax(histogram))
    residence = histogram[dominant] / n

    # Mean distance to dominant vertex
    if metric == "torus":
        dists = torus_geodesic_distance(trajectory, vertex_idx=dominant)
    else:
        dists = np.linalg.norm(trajectory - VERTICES[dominant], axis=1)
    mean_dist = float(np.mean(dists))

    # Stability: 1 - normalized entropy of vertex histogram
    probs = histogram / n
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(16)
    stability = 1.0 - entropy / max_entropy

    # Transition count and sequence
    transitions = np.where(np.diff(assignments) != 0)[0]
    transition_count = len(transitions)

    # Unique sequence (collapse consecutive duplicates)
    sequence = [int(assignments[0])]
    for a in assignments[1:]:
        if a != sequence[-1]:
            sequence.append(int(a))

    return VertexResidence(
        dominant_vertex=dominant,
        residence_fraction=residence,
        vertex_histogram=histogram,
        mean_distance=mean_dist,
        stability_score=stability,
        transition_count=transition_count,
        transition_sequence=sequence,
    )


# ──────────────────────────────────────────────────────────────
# Feature extraction from tesseract geometry
# ──────────────────────────────────────────────────────────────

def extract_tesseract_features(
    trajectory: np.ndarray,
    target_vertex: int | None = None,
) -> dict[str, float]:
    """Extract geometric features from a 4D trajectory using tesseract structure.

    Combines vertex residence analysis with Clifford torus geometry.

    Parameters
    ----------
    trajectory : shape (n_points, 4)
    target_vertex : if given, compute specific distance features to this vertex
                    (e.g., the "glymphatic" vertex, once learned)

    Returns
    -------
    Dict of feature_name → float.
    """
    features: dict[str, float] = {}

    # 1. Project to Clifford torus and measure deviation
    r1, r2 = torus_radii(trajectory)
    features["torus_r1_mean"] = float(np.mean(r1))
    features["torus_r2_mean"] = float(np.mean(r2))
    features["torus_r1_std"] = float(np.std(r1))
    features["torus_r2_std"] = float(np.std(r2))
    # How close to ideal torus (R=√2)?
    features["torus_deviation"] = float(
        np.mean((r1 - np.sqrt(2))**2 + (r2 - np.sqrt(2))**2)
    )

    # 2. Torus angles
    angles = to_torus_angles(trajectory)
    theta, phi = angles[:, 0], angles[:, 1]

    # Angular velocity (proxy for ω₁, ω₂)
    dtheta = np.diff(np.unwrap(theta))
    dphi = np.diff(np.unwrap(phi))
    features["omega1_mean"] = float(np.mean(np.abs(dtheta)))
    features["omega2_mean"] = float(np.mean(np.abs(dphi)))
    features["omega_ratio"] = features["omega1_mean"] / (features["omega2_mean"] + 1e-10)

    # Angular dispersion (circular variance)
    features["theta_dispersion"] = float(1.0 - np.abs(np.mean(np.exp(1j * theta))))
    features["phi_dispersion"] = float(1.0 - np.abs(np.mean(np.exp(1j * phi))))

    # 3. Vertex residence
    residence = analyze_vertex_residence(trajectory)
    features["dominant_vertex"] = float(residence.dominant_vertex)
    features["residence_fraction"] = residence.residence_fraction
    features["vertex_stability"] = residence.stability_score
    features["transition_count"] = float(residence.transition_count)

    # 4. Distance to target vertex (if specified)
    if target_vertex is not None:
        euc_dists = np.linalg.norm(trajectory - VERTICES[target_vertex], axis=1)
        torus_dists = torus_geodesic_distance(trajectory, vertex_idx=target_vertex)

        features["target_euc_mean"] = float(np.mean(euc_dists))
        features["target_euc_min"] = float(np.min(euc_dists))
        features["target_torus_mean"] = float(np.mean(torus_dists))
        features["target_torus_min"] = float(np.min(torus_dists))

        # Windowed Wasserstein-to-vertex (p-th moment, p=2)
        # W₂(μ_window, δ_vertex) = (mean of d²)^{1/2}
        features["target_w2_vertex"] = float(np.sqrt(np.mean(euc_dists**2)))

    return features
