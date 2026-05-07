"""
Principled baseline implementations for conformal selective acting comparison.

Two established methods adapted for the selective-acting (act/abstain) setting
used in the CSA paper. Each method receives score s_t and verifier V_t at every
round and decides whether to act.

Methods
-------
1. ACI   -- Adaptive Conformal Inference (Gibbs & Candes, 2021)
2. SAOCP -- Strongly Adaptive OCP (Bhatnagar et al., 2023)

Interface (all methods)
-----------------------
    decide(s_t, t) -> bool       # should we act?
    update(s_t, V_t, t) -> None  # observe outcome (V_t always available)
"""

import numpy as np


# ============================================================
# ACI -- Adaptive Conformal Inference (Gibbs & Candes, 2021)
# ============================================================

class ACIMethod:
    """
    Adaptive Conformal Inference adapted for selective acting.

    Maintains a threshold q_t that is adjusted online:

        q_{t+1} = q_t + gamma * (alpha - err_t)

    where err_t = 1 - V_t on rounds where the method acts.  On
    non-acted rounds: a gentle upward push of gamma*alpha/2
    prevents the threshold from getting permanently stuck at zero
    (the "cold start" problem unique to selective prediction).

    Key limitations vs. CSA
    -----------------------
    * No anytime-valid guarantee -- pathwise risk violations are
      possible during adaptation transients.
    * gamma is a tuning parameter; the method is sensitive to it.
    * Oscillates around alpha rather than monotonically accumulating
      safety evidence (no e-process).

    Reference
    ---------
    Gibbs, I. & Candes, E. (2021). Adaptive conformal inference under
    distribution shift. NeurIPS 2021.
    """

    def __init__(self, alpha: float, gamma: float = 0.01,
                 q_init: float = 0.30):
        self.alpha = alpha
        self.gamma = gamma
        self.q = q_init
        self.name = "ACI"

    def decide(self, s_t: float, t: int) -> bool:
        return s_t <= self.q

    def update(self, s_t: float, V_t: int, t: int) -> None:
        if s_t <= self.q:
            # Acted: update based on observed error
            err_t = 1 - V_t
            self.q += self.gamma * (self.alpha - err_t)
        else:
            # Did not act: gentle push toward acting to avoid cold start.
            # Half-rate increase reflects opportunity cost of abstention.
            self.q += self.gamma * self.alpha * 0.5
        self.q = np.clip(self.q, 0.01, 0.99)


# ============================================================
# SAOCP -- Strongly Adaptive Online Conformal Prediction
#          (Bhatnagar et al., 2023)
# ============================================================

class SAOCPMethod:
    """
    Strongly Adaptive OCP via multi-scale expert aggregation.

    Runs K independent ACI experts with geometrically-spaced
    step sizes gamma_k = base * 2^k.  Each expert maintains its
    own threshold q_t^(k) and is updated independently.

    Aggregation:
        q_t^SAOCP = sum_k  w_k * q_t^(k)  /  sum_k w_k

    Weights are updated multiplicatively after each round
    based on each expert's *counterfactual* selective-risk loss:

        w_{t+1}^(k) propto w_t^(k) * exp(-eta * loss_t^(k))

    This achieves "strongly adaptive" regret: the method adapts
    to distribution shifts at multiple timescales simultaneously.

    Key limitation vs. CSA
    ----------------------
    * Better adaptation than single-gamma ACI, but still no
      formal anytime-valid selective-risk guarantee.
    * The aggregation can lag behind rapid shifts.

    Reference
    ---------
    Bhatnagar, A., Wang, H., Xiong, C. & Bai, Y. (2023). Improved
    online conformal prediction via strongly adaptive online learning.
    ICML 2023.
    """

    def __init__(self, alpha: float, K: int = 6,
                 base_gamma: float = 0.002, q_init: float = 0.30):
        self.alpha = alpha
        self.K = K
        gammas = [base_gamma * (2 ** k) for k in range(K)]
        gammas = [min(g, 0.5) for g in gammas]
        self.experts = [ACIMethod(alpha, gamma=g, q_init=q_init)
                        for g in gammas]
        self.weights = np.ones(K, dtype=float) / K
        self.eta = 0.05        # meta learning rate
        self.name = "SAOCP"

    def _combined_threshold(self) -> float:
        qs = np.array([e.q for e in self.experts])
        return float(np.dot(self.weights, qs))

    def decide(self, s_t: float, t: int) -> bool:
        return s_t <= self._combined_threshold()

    def update(self, s_t: float, V_t: int, t: int) -> None:
        # 1) Update each expert's threshold independently
        for expert in self.experts:
            expert.update(s_t, V_t, t)

        # 2) Update meta-weights based on counterfactual loss
        for k, expert in enumerate(self.experts):
            would_act = (s_t <= expert.q)
            if would_act and V_t == 0:
                loss = (1 - self.alpha)   # acted on wrong output
            elif would_act and V_t == 1:
                loss = 0.0                # acted correctly
            else:
                loss = self.alpha * 0.5   # opportunity cost
            self.weights[k] *= np.exp(-self.eta * loss)

        # Renormalise
        total = self.weights.sum()
        if total > 1e-30:
            self.weights /= total
        else:
            self.weights = np.ones(self.K, dtype=float) / self.K


# ============================================================
# LTT -- Learn-then-Test (Angelopoulos et al., 2021)
# ============================================================

class LTTMethod:
    """
    Learn-then-Test applied to streaming selective acting.

    LTT is an OFFLINE risk-control method: given a calibration set of
    (score, verifier) pairs, it certifies the largest threshold tau
    such that the empirical risk among items with score <= tau is
    provably at most alpha with probability 1 - delta.

    Certification uses the Hoeffding-Bentkus (HB) concentration
    inequality over a finite grid of candidate thresholds. We apply
    Bonferroni correction over the grid to control family-wise Type-I
    error, then select the most-permissive threshold (largest tau)
    that is still certified.

    Streaming adaptation
    --------------------
    The method is offline in the Angelopoulos formulation. For a
    streaming comparison to CSA / ACI / SAOCP we use the first
    `cal_size` stream items (= the CSA burn-in period) as the LTT
    calibration set; no acts happen during calibration. After
    calibration, LTT locks its threshold and acts as a fixed-threshold
    method for the rest of the stream. This matches the "warm-start"
    deployment common in practice.

    Key limitations vs. CSA
    -----------------------
    * No anytime-valid guarantee -- LTT's pathwise bound holds only
      in expectation over draws of the calibration set. A single
      stream realisation can still exceed alpha during the
      post-calibration phase.
    * Threshold is locked; no adaptation to distribution shift.
    * Requires a non-trivial calibration prefix.

    Reference
    ---------
    Angelopoulos, A. N., Bates, S., Candes, E. J., Jordan, M. I.,
    Lei, L. (2021). Learn then Test: Calibrating Predictive Algorithms
    to Achieve Risk Control. arXiv:2110.01052.
    """

    def __init__(self, alpha: float, delta: float = 0.10,
                 cal_size: int = 500, n_thresholds: int = 15,
                 score_min: float = 0.01, score_max: float = 0.99):
        self.alpha = alpha
        self.delta = delta
        self.cal_size = cal_size
        self.grid = np.linspace(score_min, score_max, n_thresholds)
        self.cal_scores = []
        self.cal_V = []
        self.tau = None   # set after fitting; None means "refuse"
        self.fitted = False
        self.name = "LTT"

    def decide(self, s_t: float, t: int) -> bool:
        if not self.fitted:
            return False
        if self.tau is None:
            return False  # no certified threshold found
        return s_t <= self.tau

    def update(self, s_t: float, V_t: int, t: int) -> None:
        if self.fitted:
            return
        self.cal_scores.append(float(s_t))
        self.cal_V.append(int(V_t))
        if len(self.cal_scores) >= self.cal_size:
            self._fit()
            self.fitted = True

    def _fit(self):
        """Fit LTT threshold using HB p-value + Bonferroni over grid."""
        scores = np.asarray(self.cal_scores, dtype=float)
        Vs     = np.asarray(self.cal_V,      dtype=int)
        fails  = 1 - Vs  # failure indicator

        adjusted_delta = self.delta / len(self.grid)  # Bonferroni
        valid = []
        for tau in self.grid:
            mask = scores <= tau
            n = int(mask.sum())
            if n < 10:
                continue
            k = int(fails[mask].sum())
            r_hat = k / n
            if r_hat >= self.alpha:
                continue  # empirical already at/above budget: reject
            # Hoeffding p-value
            hoef = float(np.exp(-2.0 * n * (self.alpha - r_hat) ** 2))
            # Bentkus p-value:  e * P[Bin(n, alpha) <= k]
            try:
                from scipy.stats import binom
                bent = float(np.e * binom.cdf(k, n, self.alpha))
            except Exception:
                bent = 1.0
            pval = min(hoef, bent)
            if pval <= adjusted_delta:
                valid.append(tau)

        if valid:
            # Most-permissive certified threshold (largest tau)
            self.tau = float(max(valid))
        else:
            self.tau = None  # No certified threshold; refuse


