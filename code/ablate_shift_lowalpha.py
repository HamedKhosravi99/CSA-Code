"""
Stress-test distribution-shift ablation at LOW alpha (and 2 harder
scenarios) to demonstrate that every non-CSA method fails.

At α=0.05 on MedQA, the standing IID NEX-Conf numbers (from
`results/new_baselines_summary.json`) already show path-wise violation
on 10/10 streams. This script extends that by also running:
  * three "harder" distribution-shift scenarios
  * the full 10-method comparison (CSA + 5 online baselines + LTT +
    CRC + NEX-Conf + Mohri-Conf) on each scenario
so the paper has a single table that says "here is every competing
method on every shift regime at every alpha, and CSA is the only
survivor".

Scenarios:
  iid          : full shuffle (control)
  easy_hard    : sort by calibrated_score, easy half first, hard half
                 last (matches `ablate_shift.py`)
  quartile_rev : sort by score, split into 4 quartiles, play order
                 Q1 -> Q2 -> Q3 -> Q4 (strongest monotone drift)
  window_outrun: block of easiest items long enough to saturate the
                 calibration window, then switch to hard only (kills
                 offline-calibrated CRC/LTT/Mohri + NEX at low alpha).

Alphas tested: {0.05, 0.10, 0.20}.

Output:
    results/ablation_shift_lowalpha.json

Usage:
    python ablate_shift_lowalpha.py                       # default
    python ablate_shift_lowalpha.py --n_reps 5            # quicker
    python ablate_shift_lowalpha.py --alphas 0.05 0.10    # subset
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
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domains.medical.stream import MedQAStream
from csa_core import CSAConfig, CSAController
from principled_baselines import ACIMethod, SAOCPMethod, LTTMethod
from domains.baselines import AlwaysAct, FixedThreshold, NaiveTuning
from test_new_baselines import (
    CRCMethod, NEXConformalMethod, MohriConformalMethod,
)


# =================================================================
# Scenario builders. Each returns an index array of length
# n_items * n_passes from the calibrated EVAL CSV.
# =================================================================

def build_iid(df, n_passes, seed):
    n = len(df)
    rng = np.random.RandomState(seed)
    idx = []
    for _ in range(n_passes):
        perm = np.arange(n); rng.shuffle(perm)
        idx.extend(perm.tolist())
    return np.asarray(idx, dtype=int)


def build_easy_hard(df, n_passes, seed):
    """Sort asc by score; lower half easy, upper half hard; shuffle within halves."""
    scores = df['calibrated_score'].values
    order = np.argsort(scores)
    mid = len(order) // 2
    easy, hard = order[:mid].copy(), order[mid:].copy()
    rng = np.random.RandomState(seed)
    idx = []
    for _ in range(n_passes):
        e = easy.copy(); rng.shuffle(e)
        h = hard.copy(); rng.shuffle(h)
        idx.extend(e.tolist())
        idx.extend(h.tolist())
    return np.asarray(idx, dtype=int)


def build_quartile_rev(df, n_passes, seed):
    """Sort asc; play Q1 -> Q2 -> Q3 -> Q4 (monotone difficulty drift)."""
    scores = df['calibrated_score'].values
    order = np.argsort(scores)
    n = len(order)
    q = n // 4
    bins = [order[:q], order[q:2*q], order[2*q:3*q], order[3*q:]]
    rng = np.random.RandomState(seed)
    idx = []
    for _ in range(n_passes):
        for b in bins:
            bc = b.copy(); rng.shuffle(bc)
            idx.extend(bc.tolist())
    return np.asarray(idx, dtype=int)


def build_window_outrun(df, n_passes, seed, burn_in=500):
    """Pure easy block long enough to saturate calibration (size > burn_in),
    then pure hard items for the rest. Over n_passes this repeats."""
    scores = df['calibrated_score'].values
    order = np.argsort(scores)
    n = len(order)
    mid = n // 2
    easy, hard = order[:mid].copy(), order[mid:].copy()
    rng = np.random.RandomState(seed)
    idx = []
    # We want the first burn_in + slack items to be purely easy so CRC/Mohri
    # lock onto a permissive tau, then flood with hard.
    for p in range(n_passes):
        e = easy.copy(); rng.shuffle(e)
        h = hard.copy(); rng.shuffle(h)
        if p == 0:
            idx.extend(e.tolist())
            idx.extend(h.tolist())
        else:
            # alternate fairly but biased hard
            idx.extend(h.tolist())
            idx.extend(e.tolist())
    return np.asarray(idx, dtype=int)


SCENARIOS = {
    'iid':           build_iid,
    'easy_hard':     build_easy_hard,
    'quartile_rev':  build_quartile_rev,
    'window_outrun': build_window_outrun,
}


# =================================================================
# Methods factory
# =================================================================

def make_methods(alpha, grid_min, grid_max, burn_in):
    a_max = min(grid_max, max(alpha * 3.0, alpha + 0.05))
    csa_cfg = CSAConfig(alpha=alpha, delta=0.10, grid_size=15,
                        grid_min=grid_min, grid_max=a_max,
                        single_epoch=True)
    return {
        'CSA-RLVR':        ('csa', csa_cfg),
        'Always-Act':      ('baseline', AlwaysAct()),
        'Fixed-Threshold': ('baseline', FixedThreshold(q_fixed=0.5)),
        'Naive-Tuning':    ('baseline', NaiveTuning(alpha=alpha)),
        'ACI':   ('baseline', ACIMethod(alpha=alpha, gamma=0.01, q_init=0.30)),
        'SAOCP': ('baseline', SAOCPMethod(alpha=alpha, K=6, base_gamma=0.002, q_init=0.30)),
        'LTT':   ('baseline', LTTMethod(alpha=alpha, delta=0.10, cal_size=burn_in,
                                        n_thresholds=15, score_min=0.01, score_max=0.99)),
        'CRC':        ('baseline', CRCMethod(alpha=alpha, delta=0.10, cal_size=burn_in)),
        'NEX-Conf':   ('baseline', NEXConformalMethod(alpha=alpha, rho=0.99, window=burn_in)),
        'Mohri-Conf': ('baseline', MohriConformalMethod(alpha=alpha, delta=0.10, cal_size=burn_in)),
    }


def replay_one(kind, method_obj, stream, indices, burn_in):
    """Replay one method over a stream."""
    cum_act = 0; cum_fail = 0; max_risk = 0.0
    if kind == 'csa':
        ctrl = CSAController(method_obj)
    for step_idx, t in enumerate(indices):
        rd = stream.get_round(int(t))
        s_t = float(rd.score_hint) if rd.score_hint is not None else 0.5
        V_t = int(rd.V_t)
        if kind == 'csa':
            res = ctrl.step(s_t, V_t)
            acted = bool(res['acted'])
        else:
            acted = method_obj.decide(s_t, step_idx)
            method_obj.update(s_t, V_t, step_idx)
        if acted:
            cum_act += 1
            if V_t == 0:
                cum_fail += 1
            if cum_act >= burn_in:
                r = cum_fail / cum_act
                if r > max_risk:
                    max_risk = r
    fr = cum_fail / max(cum_act, 1) if cum_act > 0 else 0.0
    fa = cum_act / max(len(indices), 1)
    return {'final_risk': float(fr), 'final_ar': float(fa),
            'max_risk': float(max_risk)}


def aggregate(method_reps, alpha):
    fr = np.array([r['final_risk'] for r in method_reps])
    fa = np.array([r['final_ar']   for r in method_reps])
    mr = np.array([r['max_risk']   for r in method_reps])
    pv = int(np.sum(mr > alpha))
    return {
        'final_risk_mean': float(fr.mean()),
        'final_risk_std':  float(fr.std()),
        'final_ar_mean':   float(fa.mean()),
        'final_ar_std':    float(fa.std()),
        'max_risk_mean':   float(mr.mean()),
        'max_risk_std':    float(mr.std()),
        'pathwise_violations': pv,
        'pathwise_violation_rate': f'{pv}/{len(method_reps)}',
        'n_reps': len(method_reps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',
                    default='results/medical_inference_calibrated.csv')
    ap.add_argument('--meta',
                    default='results/medical_inference_calibrated_meta.json')
    ap.add_argument('--alphas', type=float, nargs='+',
                    default=[0.05, 0.10, 0.20])
    ap.add_argument('--n_reps', type=int, default=10)
    ap.add_argument('--n_passes', type=int, default=30)
    ap.add_argument('--burn_in_accepts', type=int, default=500)
    ap.add_argument('--scenarios', nargs='+',
                    default=list(SCENARIOS.keys()))
    ap.add_argument('--output',
                    default='results/ablation_shift_lowalpha.json')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    with open(args.meta) as f:
        meta = json.load(f)
    grid_min, grid_max = float(meta['grid_min']), float(meta['grid_max'])
    stream = MedQAStream(args.csv)
    n_items = len(df)

    # Sanity scores for the half-split (reported once)
    scores = df['calibrated_score'].values
    order = np.argsort(scores); mid = n_items // 2
    V = df['correct'].values
    err_easy = float(1 - V[order[:mid]].mean())
    err_hard = float(1 - V[order[mid:]].mean())

    print(f"{'='*78}")
    print(f"  SHIFT STRESS-TEST on MedQA ({n_items} items)")
    print(f"  err_easy={err_easy:.3f}  err_hard={err_hard:.3f}  "
          f"ratio={err_hard/max(err_easy,1e-6):.2f}x")
    print(f"  alphas={args.alphas}  scenarios={args.scenarios}  "
          f"reps={args.n_reps}  passes={args.n_passes}")
    print(f"{'='*78}")

    results = {}   # results[alpha][scenario][method] = aggregate dict

    for alpha in args.alphas:
        results[str(alpha)] = {}
        for sc in args.scenarios:
            t0 = time.time()
            per_method = {m: [] for m in make_methods(alpha, grid_min,
                                                     grid_max,
                                                     args.burn_in_accepts)}
            builder = SCENARIOS[sc]
            for rep in range(args.n_reps):
                seed = 42 + rep
                idx = builder(df, args.n_passes, seed) if sc != 'window_outrun' \
                    else builder(df, args.n_passes, seed, args.burn_in_accepts)
                # Fresh instances per rep (state carries across stream)
                methods = make_methods(alpha, grid_min, grid_max,
                                       args.burn_in_accepts)
                for name, (kind, obj) in methods.items():
                    r = replay_one(kind, obj, stream, idx,
                                   args.burn_in_accepts)
                    per_method[name].append(r)
            agg = {m: aggregate(v, alpha) for m, v in per_method.items()}
            results[str(alpha)][sc] = agg
            dt = time.time() - t0
            print(f"\n  alpha={alpha}  scenario={sc}   ({dt:.1f}s)")
            print(f"    {'Method':<18} {'Risk':<9} {'AR':<9} {'MaxR':<9} {'PathV':<8}")
            for m, d in agg.items():
                star = ' *' if d['max_risk_mean'] > alpha else ''
                print(f"    {m:<18} {d['final_risk_mean']*100:<8.2f}% "
                      f"{d['final_ar_mean']*100:<8.2f}% "
                      f"{d['max_risk_mean']*100:<8.2f}% "
                      f"{d['pathwise_violation_rate']:<6}{star}")

    out = {
        'benchmark': 'medical (MedQA)',
        'n_items': n_items,
        'n_reps': args.n_reps,
        'n_passes': args.n_passes,
        'burn_in_accepts': args.burn_in_accepts,
        'grid_min': grid_min, 'grid_max': grid_max,
        'alphas': args.alphas,
        'scenarios': args.scenarios,
        'err_easy': err_easy,
        'err_hard': err_hard,
        'shift_ratio': err_hard / max(err_easy, 1e-6),
        'results': results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
