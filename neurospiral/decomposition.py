"""REM decomposition and geometric position parameters.

Computes beta (REM position on W->N3 axis), gamma/d (unique component),
and related quantities for sleep stage geometry analysis.

The decomposition:
    mu_REM = alpha * mu_W + beta * mu_N3 + gamma * e_perp

where beta measures how far REM is along the Wake-to-N3 axis,
and gamma/d measures the fraction perpendicular to that axis.
"""

from __future__ import annotations

import numpy as np


def compute_beta(
    centroids: dict[int | str, np.ndarray],
    wake_key: int | str = 0,
    deep_key: int | str = 3,
    target_key: int | str = 4,
) -> tuple[float, float, float] | None:
    """Compute beta, gamma/d, and ratio for a target class.

    Parameters
    ----------
    centroids : dict mapping class label to feature centroid.
    wake_key : label for the Wake class.
    deep_key : label for the deep sleep class (N3).
    target_key : label for the target class (REM).

    Returns
    -------
    (beta, gamma_d, ratio) or None if insufficient data.
        beta : position on wake->deep axis (0=wake, 1=deep).
        gamma_d : fraction of centroid orthogonal to axis.
        ratio : ||target - wake|| / ||deep - wake||.
    """
    if not all(k in centroids for k in (wake_key, deep_key, target_key)):
        return None

    mu_w = centroids[wake_key]
    mu_d = centroids[deep_key]
    mu_t = centroids[target_key]

    axis = mu_d - mu_w
    axis_sq = np.dot(axis, axis)
    if axis_sq < 1e-15:
        return None

    target_vec = mu_t - mu_w
    beta = float(np.dot(target_vec, axis) / axis_sq)

    projection = mu_w + beta * axis
    residual = mu_t - projection
    gamma = float(np.linalg.norm(residual))
    d = float(np.linalg.norm(target_vec))
    gamma_d = gamma / max(d, 1e-15)
    ratio = d / max(float(np.linalg.norm(axis)), 1e-15)

    return beta, gamma_d, ratio


def compute_class_decomposition(
    centroids: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Decompose each class centroid as linear combination of others.

    For each class, solve: mu_target = A @ x + residual
    where A = [mu_other1, mu_other2, ...].

    Returns dict: class -> {beta, gamma_d, coefficients}.
    """
    classes = sorted(centroids.keys())
    if len(classes) < 3:
        return {}

    results = {}
    for target in classes:
        mu_target = centroids[target]
        others = [c for c in classes if c != target]
        A = np.column_stack([centroids[c] for c in others])

        x, _, _, _ = np.linalg.lstsq(A, mu_target, rcond=None)

        projection = A @ x
        residual = mu_target - projection
        norm_target = np.linalg.norm(mu_target)

        if norm_target < 1e-15:
            continue

        results[target] = {
            "beta": float(np.linalg.norm(projection) / norm_target),
            "gamma_d": float(np.linalg.norm(residual) / norm_target),
            "coefficients": {c: float(x[i]) for i, c in enumerate(others)},
        }

    return results
