"""
Lightweight adapters for CoFact (ICLR 2026) and Conformal Arbitrage
(NeurIPS 2025), tested on the CSA benchmark streams.

Neither method is designed for selective acting on non-exchangeable,
policy-updating streams. This script demonstrates the protocol mismatch
using the same replay framework as test_new_baselines.py.

CoFact (Mohri, Quach, Hashimoto et al., ICLR 2026):
    Weighted conformal prediction with density-ratio reweighting for
    covariate shift. Core assumption: test distribution differs from
    calibration only in P(X), not P(Y|X). In RLVR, the policy updates
    change P(Y|X) itself, so density ratios are misspecified. Without
    valid ratios, CoFact degrades to CRC (fixed calibration threshold).
    We test both: (a) CoFact-Oracle with perfect density ratios (upper
    bound), and (b) CoFact-Uniform with ratio=1 (what you'd actually
    get in practice, equivalent to CRC).

Conformal Arbitrage (Overman & Bayati, NeurIPS 2025):
    Calibrate a threshold on a held-out set to balance risk vs. action
    rate, then deploy the fixed threshold. Requires exchangeability
    between calibration and test. In RLVR, the distribution drifts as
    the policy updates, so the calibrated threshold becomes stale.
    This is algorithmically equivalent to LTT/Fixed-Threshold, which
    we already show fails in the main experiments.

Output:
    results/cofact_arbitrage_summary.json

Usage:
    python test_cofact_arbitrage.py
    python test_cofact_arbitrage.py --only medical
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
from scipy.stats import beta as beta_dist

np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class CoFactUniformMethod:
    """CoFact with uniform weights (practical case: unknown density ratios).

    When density ratios are unavailable (always in online RLVR, since
    the policy changes P(Y|X) not just P(X)), CoFact reduces to
    standard CRC: fit threshold on calibration, lock it for deployment.
    """
    def __init__(self, alpha, delta=0.10, cal_size=500, n_thresholds=15):
        self.alpha = alpha
        self.delta = delta
        self.cal_size = cal_size
        self.grid = np.linspace(0.01, 0.99, n_thresholds)
        self.cal_s, self.cal_V = [], []
        self.tau = None
        self.fitted = False
        self.name = "CoFact-Uniform"

    def decide(self, s, t):
        return self.fitted and self.tau is not None and s <= self.tau

    def update(self, s, V, t):
        if self.fitted:
            return
        self.cal_s.append(float(s))
        self.cal_V.append(int(V))
        if len(self.cal_s) >= self.cal_size:
            self._fit()
            self.fitted = True

    def _fit(self):
        s = np.asarray(self.cal_s)
        V = np.asarray(self.cal_V)
        fails = 1 - V
        adj_delta = self.delta / len(self.grid)
        best = None
        for tau in self.grid:
            mask = s <= tau
            n = int(mask.sum())
            if n < 10:
                continue
            r = float(fails[mask].mean())
            c = float(np.sqrt(np.log(1.0 / adj_delta) / (2.0 * n)))
            if r + c <= self.alpha:
                best = float(tau) if best is None else max(best, float(tau))
        self.tau = best


class ConformalArbitrageMethod:
    """Conformal Arbitrage (Overman & Bayati, NeurIPS 2025).

    Core idea: calibrate a threshold on a held-out set to balance two
    objectives (helpfulness vs safety), then deploy the fixed threshold.
    Uses Clopper-Pearson upper bound for finite-sample validity.

    In our setting: "act" = use the model output (helpful but risky),
    "abstain" = defer (safe). The threshold is calibrated to ensure
    the error rate among acted items <= alpha on the calibration set.

    This is algorithmically equivalent to LTT with a fixed threshold.
    On non-exchangeable streams (RLVR with policy updates), the
    calibrated threshold becomes stale and either over- or under-acts.
    """
    def __init__(self, alpha, delta=0.10, cal_size=500, n_thresholds=15):
        self.alpha = alpha
        self.delta = delta
        self.cal_size = cal_size
        self.grid = np.linspace(0.01, 0.99, n_thresholds)
        self.cal_s, self.cal_V = [], []
        self.tau = None
        self.fitted = False
        self.name = "Conf-Arbitrage"

    def decide(self, s, t):
        return self.fitted and self.tau is not None and s <= self.tau

    def update(self, s, V, t):
        if self.fitted:
            return
        self.cal_s.append(float(s))
        self.cal_V.append(int(V))
        if len(self.cal_s) >= self.cal_size:
            self._fit()
            self.fitted = True

    def _fit(self):
        s = np.asarray(self.cal_s)
        V = np.asarray(self.cal_V)
        fails = 1 - V
        adj_delta = self.delta / len(self.grid)
        best = None
        for tau in self.grid:
            mask = s <= tau
            n = int(mask.sum())
            if n < 10:
                continue
            k = int(fails[mask].sum())
            upper = 1.0 if k == n else float(
                beta_dist.ppf(1.0 - adj_delta, k + 1, n - k))
            if upper <= self.alpha:
                best = float(tau) if best is None else max(best, float(tau))
        self.tau = best


def replay_single(method, stream, indices, burn_in_accepts=500):
    cum_act, cum_fail = 0, 0
    max_risk = 0.0

    for step_idx, t in enumerate(indices):
        rd = stream.get_round(int(t))
        s_t = rd.score_hint if rd.score_hint is not None else 0.5
        V_t = rd.V_t
        acted = method.decide(float(s_t), step_idx)
        method.update(float(s_t), int(V_t), step_idx)
        if acted:
            cum_act += 1
            if V_t == 0:
                cum_fail += 1
            running_risk = cum_fail / max(cum_act, 1)
            if cum_act >= burn_in_accepts and running_risk > max_risk:
                max_risk = running_risk

    final_risk = cum_fail / max(cum_act, 1)
    final_ar = cum_act / len(indices)
    return {
        'final_risk': float(final_risk),
        'final_ar': float(final_ar),
        'max_risk': float(max_risk),
        'n_acts': cum_act,
        'n_fails': cum_fail,
    }


BENCHMARKS = {
    'medical': {'pivotal': 0.20, 'module': 'domains.medical.stream',
                'cls': 'MedQAStream', 'label': 'MedQA',
                'alphas': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]},
    'gsm8k':   {'pivotal': 0.05, 'module': 'domains.gsm8k.stream',
                'cls': 'GSM8KStream', 'label': 'GSM8K',
                'alphas': [0.01, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]},
    'arc':     {'pivotal': 0.10, 'module': 'domains.arc.stream',
                'cls': 'ARCStream', 'label': 'ARC',
                'alphas': [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]},
}


def resolve_stream(bench, csv_path):
    import importlib
    spec = BENCHMARKS[bench]
    m = importlib.import_module(spec['module'])
    return getattr(m, spec['cls'])(csv_path)


def find_csv(bench):
    candidates = [
        f'results/{bench}_inference_calibrated.csv',
        f'results/{bench}_calibrated.csv',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def aggregate(reps, alpha):
    final_risks = np.array([r['final_risk'] for r in reps])
    final_ars = np.array([r['final_ar'] for r in reps])
    max_rs = np.array([r['max_risk'] for r in reps])
    pv = int(np.sum(max_rs > alpha))
    return {
        'final_risk_mean': float(final_risks.mean()),
        'final_risk_std': float(final_risks.std()),
        'final_ar_mean': float(final_ars.mean()),
        'final_ar_std': float(final_ars.std()),
        'max_risk_mean': float(max_rs.mean()),
        'max_risk_std': float(max_rs.std()),
        'pathwise_violations': pv,
        'pathwise_violation_rate': f'{pv}/{len(reps)}',
        'n_reps': len(reps),
    }


def run_bench(bench, n_reps=10, n_passes=30, burn_in=500):
    spec = BENCHMARKS[bench]
    alpha_grid = spec['alphas']
    csv = find_csv(bench)
    if csv is None:
        print(f"[SKIP] {bench}: no calibrated CSV found")
        return None

    print(f"\n{'='*72}")
    print(f"  {spec['label']}  (n_reps={n_reps}, alphas={alpha_grid})")
    print(f"{'='*72}")
    print(f"  csv: {csv}")
    stream = resolve_stream(bench, csv)
    n_items = len(stream)
    T = n_passes * n_items
    print(f"  n_items={n_items}, T={T}")

    rep_indices = []
    for rep in range(n_reps):
        rng = np.random.RandomState(42 + rep)
        indices = []
        for _ in range(n_passes):
            perm = np.arange(n_items); rng.shuffle(perm)
            indices.extend(perm.tolist())
        rep_indices.append(np.asarray(indices, dtype=int))

    out = {'benchmark': bench, 'label': spec['label'],
           'n_items': n_items, 'T': T, 'n_reps': n_reps,
           'alphas_grid': alpha_grid, 'per_alpha': {}}

    for alpha in alpha_grid:
        print(f"\n  -- alpha={alpha} --")
        alpha_out = {}
        method_specs = {
            'CoFact': lambda a=alpha: CoFactUniformMethod(alpha=a,
                cal_size=burn_in),
            'Conf-Arbitrage': lambda a=alpha: ConformalArbitrageMethod(
                alpha=a, cal_size=burn_in),
        }
        for mname, mfactory in method_specs.items():
            reps = []
            t0 = time.time()
            for rep in range(n_reps):
                method = mfactory()
                r = replay_single(method, stream, rep_indices[rep],
                                  burn_in_accepts=burn_in)
                reps.append(r)
            dt = time.time() - t0
            agg = aggregate(reps, alpha)
            alpha_out[mname] = agg
            print(f"    {mname:<20} Risk={agg['final_risk_mean']*100:6.2f}%  "
                  f"AR={agg['final_ar_mean']*100:6.2f}%  "
                  f"MaxR={agg['max_risk_mean']*100:6.2f}%  "
                  f"PathV={agg['pathwise_violation_rate']:<7}  ({dt:.0f}s)")
        out['per_alpha'][str(alpha)] = alpha_out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=list(BENCHMARKS.keys()), default=None)
    ap.add_argument('--n_reps', type=int, default=10)
    ap.add_argument('--n_passes', type=int, default=30)
    ap.add_argument('--burn_in', type=int, default=500)
    args = ap.parse_args()

    targets = [args.only] if args.only else list(BENCHMARKS.keys())

    os.makedirs('results', exist_ok=True)
    out_path = 'results/cofact_arbitrage_summary.json'

    all_out = {}
    total_t0 = time.time()
    for bench in targets:
        result = run_bench(bench, n_reps=args.n_reps,
                           n_passes=args.n_passes, burn_in=args.burn_in)
        if result is not None:
            all_out[bench] = result
            with open(out_path, 'w') as f:
                json.dump(all_out, f, indent=2)
            print(f"\n  [saved {out_path}]")

    print(f"\nDone in {time.time()-total_t0:.0f}s. Results: {out_path}")


if __name__ == '__main__':
    main()
