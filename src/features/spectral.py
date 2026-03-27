"""Feature extraction for sleep stage classification.

Computes spectral power, band ratios, Hjorth parameters,
and entropy-based features from EEG epochs.
"""

from __future__ import annotations

import math
import numpy as np
from scipy import signal
from scipy.stats import entropy as scipy_entropy


# --- Spectral Features ---

DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 15.0),
    "beta": (15.0, 30.0),
}


def compute_band_powers(
    epoch: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None,
    n_fft: int = 256,
    relative: bool = True,
) -> dict[str, float]:
    """Compute power spectral density in standard EEG bands.

    Parameters
    ----------
    epoch : 1D array of shape (n_samples,)
    sfreq : sampling frequency
    bands : frequency band definitions
    n_fft : FFT window size for Welch
    relative : if True, return relative (normalized) power

    Returns
    -------
    Dict mapping band names to power values.
    """
    bands = bands or DEFAULT_BANDS

    freqs, psd = signal.welch(
        epoch,
        fs=sfreq,
        nperseg=min(n_fft, len(epoch)),
        noverlap=n_fft // 2,
    )

    total_power = np.trapz(psd, freqs)
    if total_power == 0:
        return {name: 0.0 for name in bands}

    powers = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_power = np.trapz(psd[mask], freqs[mask])
        powers[name] = band_power / total_power if relative else band_power

    return powers


def compute_band_ratios(powers: dict[str, float]) -> dict[str, float]:
    """Compute diagnostically relevant band power ratios.

    Key ratios for sleep staging:
    - delta/beta: high in N3 (slow wave sleep)
    - theta/alpha: elevated during N1 transition
    - sigma/total: spindle activity marker (N2)
    """
    eps = 1e-10  # prevent division by zero

    ratios = {
        "delta_beta": powers.get("delta", 0) / (powers.get("beta", 0) + eps),
        "theta_alpha": powers.get("theta", 0) / (powers.get("alpha", 0) + eps),
        "delta_theta": powers.get("delta", 0) / (powers.get("theta", 0) + eps),
        "sigma_total": powers.get("sigma", 0),  # already relative if normalized
    }
    return ratios


# --- Temporal Features ---

def compute_hjorth_parameters(epoch: np.ndarray) -> dict[str, float]:
    """Compute Hjorth activity, mobility, and complexity.

    These capture the statistical properties of the time-domain signal:
    - Activity: variance (signal power)
    - Mobility: std of first derivative / std of signal
    - Complexity: mobility of first derivative / mobility of signal
    """
    diff1 = np.diff(epoch)
    diff2 = np.diff(diff1)

    activity = np.var(epoch)
    eps = 1e-10

    mobility_signal = np.sqrt(np.var(diff1) / (activity + eps))
    mobility_diff = np.sqrt(np.var(diff2) / (np.var(diff1) + eps))

    return {
        "hjorth_activity": activity,
        "hjorth_mobility": mobility_signal,
        "hjorth_complexity": mobility_diff / (mobility_signal + eps),
    }


def compute_zero_crossing_rate(epoch: np.ndarray) -> float:
    """Zero-crossing rate: frequency of sign changes."""
    signs = np.sign(epoch)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return crossings / len(epoch)


def compute_permutation_entropy(
    epoch: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Permutation entropy for signal complexity estimation.

    Higher values indicate more complex/random signals (wake).
    Lower values indicate more regular/predictable signals (deep sleep).
    """
    n = len(epoch)
    n_perms = 0
    perm_counts: dict[tuple[int, ...], int] = {}

    for i in range(n - (order - 1) * delay):
        indices = list(range(i, i + order * delay, delay))
        values = epoch[indices]
        perm = tuple(np.argsort(values))
        perm_counts[perm] = perm_counts.get(perm, 0) + 1
        n_perms += 1

    if n_perms == 0:
        return 0.0

    probs = np.array(list(perm_counts.values())) / n_perms
    pe = scipy_entropy(probs, base=2)

    if normalize:
        max_entropy = np.log2(math.factorial(order))
        pe = pe / max_entropy if max_entropy > 0 else 0.0

    return pe


# --- Feature Vector Assembly ---

def extract_features_single_epoch(
    epoch: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None,
    compute_ratios: bool = True,
    hjorth: bool = True,
    permutation_entropy: bool = True,
) -> dict[str, float]:
    """Extract full feature vector from a single-channel epoch.

    Parameters
    ----------
    epoch : 1D array of shape (n_samples,)
    sfreq : sampling frequency
    bands : frequency band definitions
    compute_ratios : include band power ratios
    hjorth : include Hjorth parameters
    permutation_entropy : include permutation entropy

    Returns
    -------
    Dict of feature_name → value, ready for sklearn.
    """
    features: dict[str, float] = {}

    # Spectral
    powers = compute_band_powers(epoch, sfreq, bands)
    features.update(powers)

    if compute_ratios:
        ratios = compute_band_ratios(powers)
        features.update(ratios)

    # Temporal
    features["zero_crossing_rate"] = compute_zero_crossing_rate(epoch)

    if hjorth:
        features.update(compute_hjorth_parameters(epoch))

    if permutation_entropy:
        features["perm_entropy"] = compute_permutation_entropy(epoch)

    return features


def extract_features_batch(
    epochs: np.ndarray,
    sfreq: float,
    **kwargs,
) -> np.ndarray:
    """Extract features from all epochs across all channels.

    Parameters
    ----------
    epochs : shape (n_epochs, n_channels, n_samples)
    sfreq : sampling frequency

    Returns
    -------
    Feature matrix of shape (n_epochs, n_features)
    where n_features = n_channels × features_per_channel.
    """
    n_epochs, n_channels, _ = epochs.shape
    all_features = []

    for i in range(n_epochs):
        epoch_feats = []
        for ch in range(n_channels):
            ch_feats = extract_features_single_epoch(epochs[i, ch], sfreq, **kwargs)
            # Prefix feature names with channel index
            epoch_feats.extend(ch_feats.values())
        all_features.append(epoch_feats)

    return np.array(all_features, dtype=np.float64)


def compute_hjorth(epoch, sfreq=100):
    """Compute Hjorth parameters: activity, mobility, complexity."""
    import numpy as np
    dy = np.diff(epoch)
    ddy = np.diff(dy)
    activity = np.var(epoch)
    mobility = np.sqrt(np.var(dy) / (activity + 1e-10))
    complexity = np.sqrt(np.var(ddy) / (np.var(dy) + 1e-10)) / (mobility + 1e-10)
    return activity, mobility, complexity


def compute_hjorth(epoch, sfreq=100):
    """Compute Hjorth parameters: activity, mobility, complexity."""
    import numpy as np
    dy = np.diff(epoch)
    ddy = np.diff(dy)
    activity = np.var(epoch)
    mobility = np.sqrt(np.var(dy) / (activity + 1e-10))
    complexity = np.sqrt(np.var(ddy) / (np.var(dy) + 1e-10)) / (mobility + 1e-10)
    return activity, mobility, complexity
