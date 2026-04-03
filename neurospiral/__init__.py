"""NeuroSpiral: Clifford torus embedding for multi-periodic signal analysis.

Applies Takens delay embedding and Clifford torus geometry to extract
interpretable features from periodic and quasi-periodic signals.

Example:
    >>> from neurospiral import TorusEmbedding
    >>> torus = TorusEmbedding(d=4, tau=25)
    >>> features = torus.extract_features(signal)
"""

__version__ = "0.2.0"

from neurospiral.embedding import time_delay_embedding, estimate_optimal_tau
from neurospiral.torus import torus_features, TorusEmbedding

__all__ = [
    "time_delay_embedding",
    "estimate_optimal_tau",
    "torus_features",
    "TorusEmbedding",
]
