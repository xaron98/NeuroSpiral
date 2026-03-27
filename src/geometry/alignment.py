"""Procrustes alignment for inter-subject phase space consistency.

Problem: Takens embedding with different τ per subject produces phase
spaces that are rotated/reflected relative to each other. V12 in subject 1
might correspond to V03 in subject 2 — same geometry, different orientation.

Solution: Orthogonal Procrustes alignment. Choose a reference subject,
then rotate each other subject's torus cloud to minimize ||A - B@R||²
where R is an orthogonal matrix (rotation + reflection in ℝ⁴).

This preserves:
  - Distances between points (isometry)
  - Torus structure (R maps torus to torus)
  - Vertex discretization consistency (same vertex = same state)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes


def align_to_reference(
    target: np.ndarray,
    reference: np.ndarray,
    target_labels: np.ndarray | None = None,
    reference_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Align target point cloud to reference using orthogonal Procrustes.

    If labels are provided, aligns per-stage centroids for better
    correspondence. Otherwise uses random subsampling.

    Parameters
    ----------
    target : (n, 4) array — target subject's torus points
    reference : (m, 4) array — reference subject's torus points
    target_labels : (n,) int array — stage labels for target
    reference_labels : (m,) int array — stage labels for reference

    Returns
    -------
    aligned : (n, 4) — target rotated to match reference
    R : (4, 4) — the rotation matrix applied
    """
    if target_labels is not None and reference_labels is not None:
        # Align using per-stage centroids (more robust)
        R = _procrustes_via_centroids(target, reference, target_labels, reference_labels)
    else:
        # Align using subsampled point clouds
        R = _procrustes_via_subsample(target, reference)

    aligned = target @ R
    return aligned, R


def _procrustes_via_centroids(
    target: np.ndarray,
    reference: np.ndarray,
    target_labels: np.ndarray,
    reference_labels: np.ndarray,
) -> np.ndarray:
    """Find rotation R that aligns stage centroids."""
    # Find common labels
    common = sorted(set(target_labels) & set(reference_labels))
    if len(common) < 2:
        return np.eye(4)

    # Compute centroids per stage
    target_centroids = []
    ref_centroids = []
    for label in common:
        t_mask = target_labels == label
        r_mask = reference_labels == label
        if t_mask.sum() >= 5 and r_mask.sum() >= 5:
            target_centroids.append(target[t_mask].mean(axis=0))
            ref_centroids.append(reference[r_mask].mean(axis=0))

    if len(target_centroids) < 2:
        return np.eye(4)

    A = np.array(ref_centroids)  # (k, 4) — what we want to match
    B = np.array(target_centroids)  # (k, 4) — what we have

    # Center both
    A_mean = A.mean(axis=0)
    B_mean = B.mean(axis=0)
    A_c = A - A_mean
    B_c = B - B_mean

    # Orthogonal Procrustes: find R minimizing ||A_c - B_c @ R||
    R, _ = orthogonal_procrustes(B_c, A_c)
    return R


def _procrustes_via_subsample(
    target: np.ndarray,
    reference: np.ndarray,
    n_samples: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Find rotation R using subsampled point clouds."""
    rng = np.random.default_rng(seed)

    n_t = min(n_samples, len(target))
    n_r = min(n_samples, len(reference))
    n = min(n_t, n_r)

    idx_t = rng.choice(len(target), n, replace=False)
    idx_r = rng.choice(len(reference), n, replace=False)

    A = reference[idx_r]
    B = target[idx_t]

    A_c = A - A.mean(axis=0)
    B_c = B - B.mean(axis=0)

    R, _ = orthogonal_procrustes(B_c, A_c)
    return R


def compute_fixed_tau(sfreq: float, target_freq: float = 2.0) -> int:
    """Compute fixed τ based on delta frequency.

    For delta oscillations (~2 Hz at 100 Hz sampling):
    Quarter period = (1/2Hz) / 4 = 0.125s = 12.5 samples ≈ 13

    However, for Takens d=4, we want the embedding window
    to cover roughly one full cycle: τ*(d-1) ≈ period
    So τ = period / (d-1) = (sfreq/target_freq) / 3 ≈ 17

    We use τ = 25 as a compromise that captures about 1.5 delta cycles
    in the embedding window (25*3 = 75 samples = 0.75s at 100Hz).

    Parameters
    ----------
    sfreq : sampling frequency (Hz)
    target_freq : dominant frequency to align to (Hz), default 2 Hz (delta)

    Returns
    -------
    tau : fixed delay in samples
    """
    period_samples = sfreq / target_freq  # 50 samples for 2Hz@100Hz
    # Embedding window = τ * (d-1) should cover ~1-1.5 periods
    # τ = 1.25 * period / (d-1) = 1.25 * 50 / 3 ≈ 21
    # Round to 25 for robustness
    tau = max(int(round(1.5 * period_samples / 3)), 5)
    return tau
