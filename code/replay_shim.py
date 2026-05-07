import numpy as np
from csa_core import CSAConfig, CSAController
from domains.baselines import AlwaysAct as _AlwaysAct, NaiveTuning as _NaiveTuning


class LiveCSARLVR:
    def __init__(self, alpha, delta=0.10, burn_in_accepts=500, n_thresholds=15):
        self.alpha = alpha
        self.cfg = CSAConfig(alpha=alpha, delta=delta, grid_size=n_thresholds)
        self.ctrl = CSAController(self.cfg)
        self.name = "CSA-RLVR"

    def step(self, s_t, V_t):
        return self.ctrl.step(float(s_t), int(V_t))

    def decide(self, s_t, t):
        q = self.ctrl.q_deploy
        return q is not None and float(s_t) <= q

    def update(self, s_t, V_t, t):
        self.ctrl.step(float(s_t), int(V_t))


class FixedThresholdMethod:
    def __init__(self, alpha, quantile=0.90, burn_in=500):
        self.alpha = alpha
        self.quantile = float(quantile)
        self.burn_in = int(burn_in)
        self.cal_s = []
        self.q = None
        self.name = "Fixed-Threshold"

    def decide(self, s_t, t):
        return self.q is not None and float(s_t) <= self.q

    def update(self, s_t, V_t, t):
        if self.q is None:
            self.cal_s.append(float(s_t))
            if len(self.cal_s) >= self.burn_in:
                self.q = float(np.quantile(self.cal_s, self.quantile))


class NaiveTuningMethod:
    def __init__(self, alpha, eta=0.01, n_thresholds=15):
        self._inner = _NaiveTuning(alpha=alpha, q_init=0.5, step=float(eta))
        self.name = "Naive-Tuning"

    def decide(self, s_t, t):
        return self._inner.decide(float(s_t), int(t))

    def update(self, s_t, V_t, t):
        self._inner.update(float(s_t), int(V_t), int(t))


class AlwaysActMethod:
    def __init__(self):
        self._inner = _AlwaysAct()
        self.name = "Always-Act"

    def decide(self, s_t, t):
        return True

    def update(self, s_t, V_t, t):
        pass
