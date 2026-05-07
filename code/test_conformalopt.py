"""
Test conformalopt (Areces, Mohri, Hashimoto, Duchi -- ICML 2025) as a
baseline on 3 benchmarks: MedQA, GSM8K, ARC-Challenge.

Same replay framework as test_new_baselines.py. The conformalopt package
provides online conformal prediction via online optimization; we adapt
it to the selective-acting setting.

Install:
    pip install conformalopt

Usage:
    python test_conformalopt.py                    # all 3 benchmarks
    python test_conformalopt.py --only medical     # one only
    python test_conformalopt.py --n_reps 5         # fewer reps

Output:
    results/conformalopt_summary.json
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
np.seterr(all='ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import conformalopt as oc
    HAS_CONFORMALOPT = True
except ImportError:
    HAS_CONFORMALOPT = False


class ConformalOptMethod:
    """Online Conformal Prediction via Online Optimization (ICML 2025).

    Wraps conformalopt.ConformalPredictor into the decide/update interface
    used by the CSA baseline framework.

    The predictor maintains an online quantile tracker that adapts a
    threshold via gradient descent on the quantile (pinball) loss. We
    map this to selective acting: act when score <= predicted threshold.
    """
    def __init__(self, alpha, lr_type='proportional',
                 quantile_tracker='scalar', cal_scores=None):
        self.alpha = alpha
        self.name = 'ConformalOpt'
        self.cp = oc.ConformalPredictor(
            alpha=alpha,
            lr_type=lr_type,
            quantile_tracker=quantile_tracker,
        )
        if cal_scores is not None and len(cal_scores) > 10:
            self.cp.fit(cal_scores)
            self.fitted = True
        else:
            self.fitted = False
        self.threshold = None

    def decide(self, s, t):
        self.threshold = self.cp.predict()
        return s <= self.threshold

    def update(self, s, V, t):
        if self.threshold is not None:
            score = 1.0 - V
            self.cp.step(self.threshold, score)


class ConformalOptSelectiveMethod:
    """Variant that only updates on acted rounds (selective feedback)."""
    def __init__(self, alpha, lr_type='proportional',
                 quantile_tracker='scalar', cal_scores=None):
        self.alpha = alpha
        self.name = 'ConformalOpt-Sel'
        self.cp = oc.ConformalPredictor(
            alpha=alpha,
            lr_type=lr_type,
            quantile_tracker=quantile_tracker,
        )
        if cal_scores is not None and len(cal_scores) > 10:
            self.cp.fit(cal_scores)
            self.fitted = True
        else:
            self.fitted = False
        self.threshold = None
        self._acted = False

    def decide(self, s, t):
        self.threshold = self.cp.predict()
        self._acted = s <= self.threshold
        return self._acted

    def update(self, s, V, t):
        if self._acted and self.threshold is not None:
            score = 1.0 - V
            self.cp.step(self.threshold, score)


def replay_single(method, stream, indices, burn_in_accepts=500):
    n_acts = 0
    n_fails = 0
    max_risk = 0.0
    cum_act, cum_fail = 0, 0

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
    'medical':  {'pivotal': 0.20, 'module': 'domains.medical.stream',
                 'cls': 'MedQAStream',   'label': 'MedQA',
                 'alphas': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]},
    'gsm8k':    {'pivotal': 0.05, 'module': 'domains.gsm8k.stream',
                 'cls': 'GSM8KStream',   'label': 'GSM8K',
                 'alphas': [0.01, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]},
    'arc':      {'pivotal': 0.10, 'module': 'domains.arc.stream',
                 'cls': 'ARCStream',     'label': 'ARC',
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


def get_cal_scores(stream, n_cal=500, seed=99):
    rng = np.random.RandomState(seed)
    n = len(stream)
    idxs = rng.permutation(n)[:min(n_cal, n)]
    scores = []
    for i in idxs:
        rd = stream.get_round(int(i))
        s = rd.score_hint if rd.score_hint is not None else 0.5
        scores.append(float(s))
    return np.array(scores)


def aggregate(reps, alpha):
    final_risks = np.array([r['final_risk'] for r in reps])
    final_ars   = np.array([r['final_ar']   for r in reps])
    max_rs      = np.array([r['max_risk']   for r in reps])
    pv = int(np.sum(max_rs > alpha))
    return {
        'final_risk_mean': float(final_risks.mean()),
        'final_risk_std':  float(final_risks.std()),
        'final_ar_mean':   float(final_ars.mean()),
        'final_ar_std':    float(final_ars.std()),
        'max_risk_mean':   float(max_rs.mean()),
        'max_risk_std':    float(max_rs.std()),
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

    cal_scores = get_cal_scores(stream, n_cal=burn_in)

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
           'alphas_grid': alpha_grid,
           'per_alpha': {}}

    for alpha in alpha_grid:
        print(f"\n  -- alpha={alpha} --")
        alpha_out = {}
        method_specs = {
            'ConformalOpt': lambda a=alpha: ConformalOptMethod(
                alpha=a, cal_scores=cal_scores),
            'ConformalOpt-Sel': lambda a=alpha: ConformalOptSelectiveMethod(
                alpha=a, cal_scores=cal_scores),
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
    if not HAS_CONFORMALOPT:
        print("ERROR: conformalopt not installed. Run: pip install conformalopt")
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=list(BENCHMARKS.keys()), default=None)
    ap.add_argument('--n_reps', type=int, default=10)
    ap.add_argument('--n_passes', type=int, default=30)
    ap.add_argument('--burn_in', type=int, default=500)
    args = ap.parse_args()

    targets = [args.only] if args.only else list(BENCHMARKS.keys())

    os.makedirs('results', exist_ok=True)
    out_path = 'results/conformalopt_summary.json'

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
