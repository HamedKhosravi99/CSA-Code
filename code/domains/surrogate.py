"""
Online surrogate model for CSA score computation.

Shared across all domains. Computes S_t = 1 - p_hat_t(features)
where p_hat is a logistic regression trained on the growing buffer D_{t-1}.
Retrained every B rounds. Maintains F_{t-1}-measurability.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class OnlineSurrogate:
    """Logistic regression surrogate for CSA score computation."""

    def __init__(self, retrain_every: int = 25, min_samples: int = 15):
        self.model = None
        self.buffer_X: list = []
        self.buffer_V: list = []
        self.retrain_every = retrain_every
        self.min_samples = min_samples
        self.n_since_train = 0
        self.train_count = 0

    def score(self, features: np.ndarray) -> float:
        """Compute S_t = 1 - p_hat. Uses model from BEFORE this round."""
        if self.model is not None:
            try:
                p_hat = self.model.predict_proba(features.reshape(1, -1))[0, 1]
                return float(np.clip(1.0 - p_hat, 0.005, 0.995))
            except Exception:
                pass
        # Fallback: use first feature as confidence proxy
        if len(features) > 0:
            return float(np.clip(1.0 - features[0], 0.01, 0.99))
        return 0.5

    def observe(self, features: np.ndarray, V_t: int):
        """Add observation AFTER scoring (maintains F_{t-1} measurability)."""
        self.buffer_X.append(features.copy())
        self.buffer_V.append(V_t)
        self.n_since_train += 1

        if (self.n_since_train >= self.retrain_every
                and len(self.buffer_X) >= self.min_samples):
            self._retrain()
            self.n_since_train = 0

    def _retrain(self):
        X = np.array(self.buffer_X)
        y = np.array(self.buffer_V)
        if len(np.unique(y)) < 2:
            return
        try:
            self.model = LogisticRegression(
                max_iter=500, C=1.0, solver='lbfgs')
            self.model.fit(X, y)
            self.train_count += 1
        except Exception:
            pass

    def reset(self):
        """Reset surrogate state for a new replication."""
        self.model = None
        self.buffer_X = []
        self.buffer_V = []
        self.n_since_train = 0
        self.train_count = 0
