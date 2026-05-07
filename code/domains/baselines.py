"""
Heuristic baselines for CSA comparison.

All baselines share the interface:
    decide(s_t, t) -> bool       # should we act?
    update(s_t, V_t, t) -> None  # observe outcome
"""

import numpy as np


class AlwaysAct:
    """Release every output. No safety layer."""
    def __init__(self):
        self.name = "Always-Act"

    def decide(self, s_t: float, t: int) -> bool:
        return True

    def update(self, s_t: float, V_t: int, t: int):
        pass


class FixedThreshold:
    """Fixed threshold, frozen from initial calibration."""
    def __init__(self, q_fixed: float = 0.5):
        self.q = q_fixed
        self.name = f"Fixed-Threshold"

    def decide(self, s_t: float, t: int) -> bool:
        return s_t <= self.q

    def update(self, s_t: float, V_t: int, t: int):
        pass


class NaiveTuning:
    """Naive online threshold tuning via bisection. No formal guarantee."""
    def __init__(self, alpha: float, q_init: float = 0.5, step: float = 0.02):
        self.alpha = alpha
        self.q = q_init
        self.step = step
        self.cum_fail = 0
        self.cum_act = 0
        self.name = "Naive-Tuning"

    def decide(self, s_t: float, t: int) -> bool:
        return s_t <= self.q

    def update(self, s_t: float, V_t: int, t: int):
        if s_t <= self.q:
            self.cum_act += 1
            if V_t == 0:
                self.cum_fail += 1
        if (t + 1) % 20 == 0 and self.cum_act > 5:
            risk_hat = self.cum_fail / self.cum_act
            if risk_hat > self.alpha + 0.02:
                self.q = max(0.01, self.q - self.step)
            elif risk_hat < self.alpha - 0.02:
                self.q = min(0.99, self.q + self.step)

    def reset(self):
        self.cum_fail = 0
        self.cum_act = 0
        self.q = 0.5
