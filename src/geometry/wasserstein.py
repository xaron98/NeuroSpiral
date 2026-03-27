"""Wasserstein distance metrics for neural state comparison.

Two complementary approaches:

1. BURES-WASSERSTEIN (closed-form, for SPD covariance matrices)
   W₂(Σ₁, Σ₂) = [tr(Σ₁) + tr(Σ₂) - 2·tr((Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})]^{1/2}

   This is the mathematically correct distance on the Riemannian
   manifold of symmetric positive definite matrices. For 4×4 SPD
   matrices (our Takens embedding dimension), the eigendecomposition
   is O(1) — viable for real-time on embedded hardware.

   Why not Euclidean on covariances?
   SPD matrices form a curved manifold. Euclidean distance "cuts through"
   the interior of the cone, measuring paths that don't correspond to
   valid covariance matrices. Bures-Wasserstein follows geodesics on
   the manifold — each intermediate point is itself a valid SPD matrix.

2. SLICED-WASSERSTEIN (approximate, for point clouds)
   SW_p(μ, ν) = (∫_{S^{d-1}} W_p(proj_θ μ, proj_θ ν)^p dθ)^{1/p}

   Projects d-dimensional distributions onto random 1D lines where
   W_p has a closed-form solution via sorting. With ~50 projections
   in d=4, gives a good approximation in O(n log n) per projection.

   This is the practical alternative when comparing full point-cloud
   distributions (windowed trajectory vs reference trajectory)
   rather than their covariance summaries.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


# ══════════════════════════════════════════════════════════════
# Bures-Wasserstein distance (closed-form for SPD matrices)
# ══════════════════════════════════════════════════════════════

def _sqrtm_spd(A: np.ndarray) -> np.ndarray:
    """Matrix square root for symmetric positive definite matrices.

    Uses eigendecomposition (guaranteed real for SPD).
    For 4×4 matrices this is essentially O(1).
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    # Clamp small negative eigenvalues from numerical noise
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def bures_wasserstein(
    sigma1: np.ndarray,
    sigma2: np.ndarray,
    regularize: float = 1e-8,
) -> float:
    """Bures-Wasserstein distance between two SPD matrices.

    W₂(Σ₁, Σ₂) = [tr(Σ₁) + tr(Σ₂) - 2·tr((Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})]^{1/2}

    This is the 2-Wasserstein distance between two zero-mean
    Gaussians N(0, Σ₁) and N(0, Σ₂). On the SPD manifold,
    it respects the Riemannian geometry — intermediate points
    along the geodesic are all valid covariance matrices.

    Parameters
    ----------
    sigma1 : (d, d) SPD matrix — current brain state covariance
    sigma2 : (d, d) SPD matrix — target state covariance (e.g., N3)
    regularize : small value added to diagonal for numerical stability

    Returns
    -------
    Non-negative scalar distance.
    """
    d = sigma1.shape[0]

    # Regularize to ensure strict positive definiteness
    sigma1 = sigma1 + regularize * np.eye(d)
    sigma2 = sigma2 + regularize * np.eye(d)

    # Compute Σ₁^{1/2}
    sqrt_s1 = _sqrtm_spd(sigma1)

    # Compute (Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2}
    inner = sqrt_s1 @ sigma2 @ sqrt_s1
    sqrt_inner = _sqrtm_spd(inner)

    # W₂² = tr(Σ₁) + tr(Σ₂) - 2·tr((Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})
    w2_squared = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(sqrt_inner)

    # Numerical safety
    return float(np.sqrt(max(w2_squared, 0.0)))


def bures_wasserstein_mean_distance(
    sigma1: np.ndarray,
    sigma2: np.ndarray,
    mu1: np.ndarray | None = None,
    mu2: np.ndarray | None = None,
    regularize: float = 1e-8,
) -> float:
    """Extended Bures-Wasserstein with mean shift.

    W₂²(N(μ₁,Σ₁), N(μ₂,Σ₂)) = ‖μ₁-μ₂‖² + BW²(Σ₁,Σ₂)

    Includes the contribution from different mean states,
    not just covariance shape.
    """
    bw = bures_wasserstein(sigma1, sigma2, regularize)

    if mu1 is not None and mu2 is not None:
        mean_dist_sq = float(np.sum((mu1 - mu2) ** 2))
        return float(np.sqrt(mean_dist_sq + bw**2))

    return bw


# ══════════════════════════════════════════════════════════════
# SPD matrix construction from trajectory windows
# ══════════════════════════════════════════════════════════════

def trajectory_to_spd(
    trajectory: np.ndarray,
    regularize: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SPD covariance matrix from a trajectory window.

    Parameters
    ----------
    trajectory : shape (n_points, d) — windowed 4D trajectory
    regularize : diagonal regularization

    Returns
    -------
    mean : shape (d,) — centroid of trajectory
    cov : shape (d, d) — regularized covariance matrix (guaranteed SPD)
    """
    mean = np.mean(trajectory, axis=0)
    centered = trajectory - mean
    cov = (centered.T @ centered) / max(trajectory.shape[0] - 1, 1)

    # Ensure strict SPD
    cov = (cov + cov.T) / 2  # symmetrize
    cov += regularize * np.eye(cov.shape[0])

    return mean, cov


def compute_reference_spd(
    trajectories: list[np.ndarray],
    regularize: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reference SPD matrix from multiple trajectory windows.

    Used to build a "template" covariance for a target state
    (e.g., average N3 covariance from labeled training data).

    Parameters
    ----------
    trajectories : list of arrays, each shape (n_points, d)

    Returns
    -------
    mean : shape (d,) — mean of means
    cov : shape (d, d) — Fréchet mean of covariances (approximated
          by arithmetic mean, which is exact for Euclidean, approximate
          for Bures-Wasserstein; acceptable for small d=4).
    """
    means = []
    covs = []
    for traj in trajectories:
        m, c = trajectory_to_spd(traj, regularize)
        means.append(m)
        covs.append(c)

    ref_mean = np.mean(means, axis=0)
    ref_cov = np.mean(covs, axis=0)

    # Re-symmetrize and regularize
    ref_cov = (ref_cov + ref_cov.T) / 2 + regularize * np.eye(ref_cov.shape[0])

    return ref_mean, ref_cov


# ══════════════════════════════════════════════════════════════
# Sliced-Wasserstein distance (approximate, for point clouds)
# ══════════════════════════════════════════════════════════════

def _wasserstein_1d(u: np.ndarray, v: np.ndarray, p: int = 2) -> float:
    """Exact p-Wasserstein distance between 1D empirical distributions.

    W_p(u, v) = (sum |u_sort - v_sort|^p / n)^{1/p}

    For equal-size samples, this is just the Lp distance of sorted arrays.
    For unequal sizes, we interpolate via quantile matching.
    """
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    if len(u_sorted) != len(v_sorted):
        # Quantile matching: interpolate to common grid
        n = max(len(u_sorted), len(v_sorted))
        quantiles = np.linspace(0, 1, n)
        u_interp = np.interp(quantiles, np.linspace(0, 1, len(u_sorted)), u_sorted)
        v_interp = np.interp(quantiles, np.linspace(0, 1, len(v_sorted)), v_sorted)
        u_sorted, v_sorted = u_interp, v_interp

    return float(np.mean(np.abs(u_sorted - v_sorted) ** p) ** (1.0 / p))


def sliced_wasserstein(
    cloud1: np.ndarray,
    cloud2: np.ndarray,
    n_projections: int = 50,
    p: int = 2,
    seed: int = 42,
) -> float:
    """Sliced-Wasserstein distance between two point clouds.

    SW_p(μ, ν) ≈ (1/L Σ W_p(proj_θᵢ μ, proj_θᵢ ν)^p)^{1/p}

    Projects d-dimensional distributions onto random 1D lines,
    computes exact 1D Wasserstein on each, and averages.

    Complexity: O(L · n log n) where L = n_projections, n = cloud size.
    For d=4, L=50, n=300: ~15,000·log(300) ≈ 120K ops — fast.

    Parameters
    ----------
    cloud1 : shape (n1, d) — e.g., current windowed trajectory
    cloud2 : shape (n2, d) — e.g., reference N3 trajectory
    n_projections : number of random 1D projections
    p : Wasserstein order (typically 2)
    seed : random seed for reproducibility

    Returns
    -------
    Approximate p-Wasserstein distance.
    """
    d = cloud1.shape[1]
    rng = np.random.default_rng(seed)

    # Sample random unit directions on S^{d-1}
    directions = rng.standard_normal((n_projections, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    sw_total = 0.0
    for direction in directions:
        # Project both clouds onto this direction
        proj1 = cloud1 @ direction  # (n1,)
        proj2 = cloud2 @ direction  # (n2,)

        # 1D Wasserstein
        w1d = _wasserstein_1d(proj1, proj2, p)
        sw_total += w1d ** p

    return float((sw_total / n_projections) ** (1.0 / p))


# ══════════════════════════════════════════════════════════════
# Combined distance features for the pipeline
# ══════════════════════════════════════════════════════════════

def extract_distance_features(
    current_trajectory: np.ndarray,
    reference_mean: np.ndarray,
    reference_cov: np.ndarray,
    reference_trajectory: np.ndarray | None = None,
) -> dict[str, float]:
    """Extract distance-based features comparing current state to reference.

    Computes three levels of comparison:
    1. Bures-Wasserstein (covariance geometry)
    2. Extended BW with mean shift
    3. Sliced-Wasserstein (full distributional, if reference cloud given)

    Parameters
    ----------
    current_trajectory : shape (n_points, 4) — current window
    reference_mean : shape (4,) — reference state centroid
    reference_cov : shape (4, 4) — reference state SPD covariance
    reference_trajectory : shape (m, 4) — reference point cloud (optional)

    Returns
    -------
    Dict of distance features.
    """
    features: dict[str, float] = {}

    # Current state summary
    curr_mean, curr_cov = trajectory_to_spd(current_trajectory)

    # 1. Bures-Wasserstein on covariances
    bw = bures_wasserstein(curr_cov, reference_cov)
    features["bw_distance"] = bw

    # 2. Extended BW with mean shift
    bw_ext = bures_wasserstein_mean_distance(
        curr_cov, reference_cov, curr_mean, reference_mean
    )
    features["bw_extended"] = bw_ext

    # 3. Euclidean distance between means (for comparison)
    features["mean_euclidean"] = float(np.linalg.norm(curr_mean - reference_mean))

    # 4. Eigenvalue-based features (shape of covariance)
    eigvals_curr = np.sort(np.linalg.eigvalsh(curr_cov))[::-1]
    eigvals_ref = np.sort(np.linalg.eigvalsh(reference_cov))[::-1]

    # Spectral shape similarity
    features["eigenvalue_l2"] = float(np.linalg.norm(eigvals_curr - eigvals_ref))

    # Condition number (dimensionality of the attractor)
    features["condition_number"] = float(eigvals_curr[0] / (eigvals_curr[-1] + 1e-10))

    # 5. Sliced-Wasserstein (if reference cloud available)
    if reference_trajectory is not None:
        sw = sliced_wasserstein(current_trajectory, reference_trajectory)
        features["sw_distance"] = sw

    return features
