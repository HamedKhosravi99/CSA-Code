"""
Abstract base classes for domain-specific CSA experiments.

Every domain must implement:
  - DomainStream: yields (features, V_t, metadata) per round
  - DomainVerifier: checks correctness of model output vs ground truth
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class RoundData:
    """Data for a single deployment round."""
    item_id: str
    features: np.ndarray       # surrogate input features
    V_t: int                   # verifier output: 1=correct, 0=incorrect
    score_hint: Optional[float] = None  # optional pre-computed logprob-based score
    metadata: Dict = field(default_factory=dict)


class DomainStream(ABC):
    """Abstract base for any domain data stream.

    A stream presents items one at a time in a deployment-like sequence.
    Subclasses load from a pre-computed inference CSV (Phase A output).
    """

    @abstractmethod
    def __len__(self) -> int:
        """Total number of items in the stream."""
        ...

    @abstractmethod
    def get_round(self, t: int) -> RoundData:
        """Return data for round t."""
        ...

    def shuffled_indices(self, seed: int) -> np.ndarray:
        """Return a shuffled index permutation for replay experiments."""
        rng = np.random.RandomState(seed)
        return rng.permutation(len(self))


class DomainVerifier(ABC):
    """Abstract base for domain-specific verification."""

    @abstractmethod
    def verify(self, model_output: Any, ground_truth: Any) -> int:
        """Return 1 if correct, 0 if incorrect."""
        ...
