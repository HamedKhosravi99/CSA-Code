"""
Passive adapter for A-RCPS (Xu, Karampatziakis, Mineiro, NeurIPS 2024,
"Active, Anytime-Valid Risk Controlling Prediction Sets").

This is a numpy-only reimplementation of the paper's FullyObservedUpperMartingale
(from github.com/neilzxu/active-rcps, file IwUpperMartingale.py).  The purpose
here is an apples-to-apples comparison against CSA on our stream protocol: we
give A-RCPS the full-observation advantage (it sees V_t on every round), which
is the strongest variant of the method under its own assumptions.

The experiment exposes the protocol mismatch between A-RCPS and CSA:
  - A-RCPS bounds the MARGINAL false-positive rate
        R^marg_T := (1/T) * sum_t [released_t AND wrong_t]
  - CSA bounds the SELECTIVE failure rate
        R^act_T := sum_t [released_t AND wrong_t] / N_released
  These are different quantities.  The experiment reports both so the mismatch
  is visible: at any non-trivial accept rate, A-RCPS's selective risk can
  exceed alpha even when its marginal risk is within alpha.
"""
from __future__ import annotations

from math import log
import numpy as np


# =============================================================================
#  Numpy-only port of A-RCPS FullyObservedUpperMartingale.
#  Matches IwUpperMartingale.py:221-239 in neilzxu/active-rcps.
# =============================================================================

class FullyObservedUpperMartingale:
    """
    Grid-based e-process for certifying the smallest beta in [0, 1] such that
    E[rho(X, beta)] <= theta.  Maintains one martingale per beta on a grid,
    advancing a 'certified beta index' whenever the next smaller beta's
    running log-wealth crosses -log(confidence).

    FullyObserved variant: labels are observed on every round; no active
    query budget and no importance weighting.  This is the strongest A-RCPS
    variant under passive labeling.
    """

    def __init__(
        self,
        rho,
        theta: float,
        n_betas: int = 100,
        confidence: float = 0.05,
        n_beta_min: float = 0.0,
    ) -> None:
        assert isinstance(n_betas, int) and n_betas > 1
        assert 0.0 < confidence < 1.0
        self._rho = rho
        self._theta = float(theta)
        self._betas = np.linspace(n_beta_min, 1.0, n_betas)
        self._stats = np.zeros((n_betas, 3))  # [log_wealth, sum_xi, sum_xi^2]
        self._curbetaindex = n_betas - 1  # start: most restrictive beta
        self._curlam = np.zeros(n_betas)
        self._thres = -log(confidence)

    # -- Exposed state --------------------------------------------------------

    @property
    def certified_beta(self) -> float:
        """Current smallest beta proven safe (most permissive certified)."""
        return float(self._betas[self._curbetaindex])

    @property
    def cur_log_wealth(self) -> float:
        """Log-wealth at the current curbetaindex (diagnostic)."""
        return float(self._stats[self._curbetaindex, 0])

    # -- Core update ----------------------------------------------------------

    def addobs(self, x) -> None:
        """Observe x = (p, y) and update martingales."""
        if self._curbetaindex == 0:
            return  # already at most permissive beta; nothing to refine

        # xi(beta) = theta - rho(x, beta)  (FullyObservedUpperMartingale.xi)
        xibetas = np.asarray(self._theta - self._rho(x, self._betas)).reshape(-1)

        # Update running statistics
        self._stats[:, 0] += np.log1p(self._curlam * xibetas)
        self._stats[:, 1] += xibetas
        self._stats[:, 2] += xibetas ** 2

        # Try to advance curbetaindex: find smallest index with log-wealth >= thres
        certified = self._stats[:, 0] >= self._thres
        if certified.any():
            first_certified = int(np.argmax(certified))
            if first_certified < self._curbetaindex:
                self._curbetaindex = first_certified
                # Truncate arrays to active range [0, curbetaindex]
                self._betas = self._betas[: first_certified + 1]
                self._stats = self._stats[: first_certified + 1, :]
                self._curlam = self._curlam[: first_certified + 1]

        # Update bets for next round (follow-the-regularized-leader)
        ftlnum = self._stats[:, 1]
        ftldenom = self._stats[:, 1] + self._stats[:, 2]
        ximin = self._theta - 1.0  # from FullyObservedUpperMartingale.ximin
        max_lam = 0.5 / abs(ximin)
        lams = np.divide(
            ftlnum, ftldenom,
            out=np.zeros_like(ftlnum),
            where=ftldenom != 0,
        )
        self._curlam = np.clip(lams, 0.0, max_lam)


# =============================================================================
#  CSA-framework wrapper so the driver can run it like any other baseline.
# =============================================================================

class ARCPSPassive:
    """
    A-RCPS (Xu, Karampatziakis, Mineiro, NeurIPS 2024) in its FullyObserved
    variant, adapted to the CSA calibrated-score convention.

    CSA's widetilde_S_t in [0, 1] is a confidence score with the convention
    "lower = more confident" (release if widetilde_S_t <= tau).  A-RCPS uses
    p = 1 - widetilde_S so that "higher = more confident" and release-at-beta
    means release if p >= beta.  The two conventions are equivalent under this
    flip.

    A-RCPS's target (theta) is MARGINAL false-positive rate; CSA's target
    (alpha) is SELECTIVE failure rate.  The test harness reports both.
    """

    NAME = "A-RCPS (passive)"

    def __init__(self, alpha: float, n_betas: int = 100,
                 confidence: float = 0.05) -> None:
        self.alpha = float(alpha)

        def rho(x, betas):
            p, y = x
            return (1.0 - y) * (betas <= p).astype(float)

        self.mart = FullyObservedUpperMartingale(
            rho=rho,
            theta=self.alpha,
            n_betas=n_betas,
            confidence=confidence,
        )

    def decide(self, widetilde_s: float, t: int) -> bool:
        """Release item t iff the certified threshold admits its score."""
        p = 1.0 - float(widetilde_s)
        return self.mart.certified_beta <= p

    def update(self, widetilde_s: float, V_t: int, t: int) -> None:
        """Observe V_t (A-RCPS assumes full observation each round)."""
        p = 1.0 - float(widetilde_s)
        self.mart.addobs(x=(p, int(V_t)))
