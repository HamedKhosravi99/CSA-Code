"""
CSA-RLVR: Conformal Selective Acting (core implementation).

Faithful to Algorithm 1 from the paper draft:
  Score S_t = 1 - p_hat_t(x,y)
  Gate A_t(q) = 1{S_t <= q}
  Increment X_t(q) = A_t(q) * ((1-V_t) - alpha)
  E-process: multiplicative in log-space
  Adaptive lambda from Eq. (9)
  Controller: q_{t+1} = max{q : Cert(q)}
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CSAConfig:
    alpha: float = 0.10
    delta: float = 0.05
    grid_size: int = 50
    grid_min: float = 0.02
    grid_max: float = 0.98
    single_epoch: bool = True
    epoch_length: int = 500


class CSAController:
    """
    CSA-RLVR controller (Algorithm 1).

    Works with an ABSTRACT score s_t in [0,1].
    The caller is responsible for computing s_t and V_t.
    """

    def __init__(self, config: CSAConfig):
        self.cfg = config
        m = config.grid_size
        self.grid = np.linspace(config.grid_min, config.grid_max, m)
        self.m = m

        # Per-threshold state
        self.log_e = np.zeros(m)          # log E_t(q)
        self.sum_x = np.zeros(m)          # running sum of X_s(q)
        self.count = np.zeros(m, dtype=int)  # count of updates
        self.certified = np.zeros(m, dtype=bool)
        self.cert_time = np.full(m, np.nan)  # round when certified

        # Budget: delta_q = delta/(2m) for single-epoch (Theorem 7)
        if config.single_epoch:
            self.delta_q = np.full(m, config.delta / (2 * m))
        else:
            self.delta_q = np.full(m, 6 * config.delta / (np.pi**2 * m))
        self.log_thresh = np.log(1.0 / self.delta_q)

        # Controller state
        self.q_deploy = None
        self.q_deploy_idx = -1

        # Epoch tracking
        self.epoch = 1
        self.epoch_start = 0
        self.t = 0

        # History for analysis
        self.history_acted = []
        self.history_V = []
        self.history_score = []
        self.history_q_deploy = []

    def step(self, s_t: float, V_t: int) -> dict:
        """
        One round of CSA-RLVR.

        Parameters
        ----------
        s_t : float in [0,1], the nonconformity score
        V_t : int in {0,1}, verifier output (1=pass, 0=fail)

        Returns
        -------
        dict with 'acted', 'V_t', 's_t', 'q_deploy'
        """
        alpha = self.cfg.alpha
        grid = self.grid

        # --- ACT OR ABSTAIN ---
        acted = False
        if self.q_deploy is not None and s_t <= self.q_deploy:
            acted = True

        # --- UPDATE E-PROCESSES for all q >= s_t ---
        for k in range(self.m):
            if grid[k] < s_t:
                # A_t(q) = 0 => X_t(q) = 0 => no update
                continue

            # A_t(q) = 1
            X_t_q = (1 - V_t) - alpha  # in {-alpha, 1-alpha}

            # Adaptive lambda (Eq. 9)
            if self.count[k] == 0:
                lam = 0.0
            else:
                mu_hat = self.sum_x[k] / self.count[k]
                lam = np.clip(
                    -mu_hat / (1 - alpha) ** 2,
                    0.0,
                    1.0 / (2 * (1 - alpha))
                )

            # Multiplicative update in log-space
            factor = 1.0 - lam * X_t_q
            if factor > 0:
                self.log_e[k] += np.log(factor)

            # Update running statistics
            self.sum_x[k] += X_t_q
            self.count[k] += 1

            # Certification check
            if not self.certified[k] and self.log_e[k] >= self.log_thresh[k]:
                self.certified[k] = True
                self.cert_time[k] = self.t

        # --- UPDATE CONTROLLER ---
        cert_idx = np.where(self.certified)[0]
        if len(cert_idx) > 0:
            self.q_deploy_idx = cert_idx[-1]
            self.q_deploy = grid[self.q_deploy_idx]
        else:
            self.q_deploy = None
            self.q_deploy_idx = -1

        # --- EPOCH RESTART (CSA-Epoch only) ---
        if not self.cfg.single_epoch:
            if (self.t - self.epoch_start) >= self.cfg.epoch_length:
                self._reset_epoch()

        # --- RECORD ---
        self.history_acted.append(int(acted))
        self.history_V.append(V_t)
        self.history_score.append(s_t)
        self.history_q_deploy.append(
            self.q_deploy if self.q_deploy is not None else np.nan)
        self.t += 1

        return {'acted': acted, 'V_t': V_t, 's_t': s_t,
                'q_deploy': self.q_deploy}

    def _reset_epoch(self):
        self.epoch += 1
        self.epoch_start = self.t
        j = self.epoch
        m = self.m
        self.delta_q = np.full(m, 6*self.cfg.delta / (np.pi**2 * m * j**2))
        self.log_thresh = np.log(1.0 / self.delta_q)
        self.log_e[:] = 0
        self.sum_x[:] = 0
        self.count[:] = 0
        self.certified[:] = False
        self.q_deploy = None
        self.q_deploy_idx = -1

    # ---- Metrics ----

    def selective_risk(self) -> float:
        A = np.array(self.history_acted)
        V = np.array(self.history_V)
        N = A.sum()
        return float((A * (1 - V)).sum() / max(N, 1))

    def action_rate(self) -> float:
        A = np.array(self.history_acted)
        return float(A.mean()) if len(A) > 0 else 0.0

    def cumulative_metrics(self) -> dict:
        A = np.array(self.history_acted, dtype=float)
        V = np.array(self.history_V, dtype=float)
        T = len(A)
        cum_fail = np.cumsum(A * (1 - V))
        cum_act = np.cumsum(A)
        safe_act = np.maximum(cum_act, 1)
        return {
            'risk': cum_fail / safe_act,
            'action_rate': cum_act / np.arange(1, T + 1),
            'cum_actions': cum_act,
            'q_deploy': np.array(self.history_q_deploy),
            'scores': np.array(self.history_score),
        }
