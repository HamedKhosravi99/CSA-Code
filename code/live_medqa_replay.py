"""
Replay CSA-RLVR + baselines on a stream produced by live_medqa.py.

Reads results_live_medqa/<mode>.json (list of {score, V}) and produces
results_live_medqa/<mode>_replay.json with per-method per-alpha
final_risk, final_ar, pathwise_violations, max_risk, and running curves
over n_shuffles replications.

Usage:
    python live_medqa_replay.py \
        --stream results_live_medqa/pilot.json \
        --out    results_live_medqa/pilot_replay.json \
        --alphas 0.10,0.20,0.30 \
        --n-shuffles 20 --burn-in 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_new_baselines import (
    CRCMethod, MohriConformalMethod, NEXConformalMethod,
)
from principled_baselines import (
    LTTMethod, ACIMethod, SAOCPMethod,
)
from replay_shim import (
    LiveCSARLVR, FixedThresholdMethod, NaiveTuningMethod, AlwaysActMethod,
)


def build_methods(alpha, burn_in):
    return {
        'CSA-RLVR':        (lambda a=alpha, b=burn_in: LiveCSARLVR(
                                alpha=a, delta=0.10, burn_in_accepts=b,
                                n_thresholds=15)),
        'CRC':             (lambda a=alpha, b=burn_in: CRCMethod(
                                alpha=a, delta=0.10, cal_size=b,
                                n_thresholds=15)),
        'NEX-Conf':        (lambda a=alpha, b=burn_in: NEXConformalMethod(
                                alpha=a, rho=0.99, window=b,
                                n_thresholds=15)),
        'Mohri-Conf':      (lambda a=alpha, b=burn_in: MohriConformalMethod(
                                alpha=a, delta=0.10, cal_size=b,
                                n_thresholds=15)),
        'LTT':             (lambda a=alpha, b=burn_in: LTTMethod(
                                alpha=a, delta=0.10, cal_size=b,
                                n_thresholds=15)),
        'ACI':             (lambda a=alpha, b=burn_in: ACIMethod(
                                alpha=a, gamma=0.01)),
        'SAOCP':           (lambda a=alpha, b=burn_in: SAOCPMethod(
                                alpha=a, K=6, base_gamma=0.002)),
        'Fixed-Threshold': (lambda a=alpha, b=burn_in: FixedThresholdMethod(
                                alpha=a, quantile=1.0 - a, burn_in=b)),
        'Naive-Tuning':    (lambda a=alpha, b=burn_in: NaiveTuningMethod(
                                alpha=a, eta=0.01, n_thresholds=15)),
        'Always-Act':      (lambda a=alpha, b=burn_in: AlwaysActMethod()),
    }


def replay_single(scores, Vs, method, burn_in, is_csa):
    T = len(scores)
    cum_act = cum_fail = 0
    max_risk = 0.0
    risk_curve = np.zeros(T)
    ar_curve   = np.zeros(T)
    for t in range(T):
        s_t, V_t = float(scores[t]), int(Vs[t])
        if is_csa:
            res = method.step(s_t, V_t)
            acted = bool(res['acted'])
        else:
            acted = bool(method.decide(s_t, t))
            method.update(s_t, V_t, t)
        if acted:
            cum_act += 1
            if V_t == 0:
                cum_fail += 1
            if cum_act >= burn_in:
                rr = cum_fail / cum_act
                if rr > max_risk:
                    max_risk = rr
        risk_curve[t] = cum_fail / max(cum_act, 1)
        ar_curve[t]   = cum_act / (t + 1)
    return {
        'final_risk': cum_fail / max(cum_act, 1),
        'final_ar':   cum_act / T,
        'max_risk':   float(max_risk),
        'risk_curve': risk_curve.tolist(),
        'ar_curve':   ar_curve.tolist(),
    }


def aggregate(reps, alpha):
    fr = np.array([r['final_risk'] for r in reps])
    fa = np.array([r['final_ar']   for r in reps])
    mx = np.array([r['max_risk']   for r in reps])
    pv = int(np.sum(mx > alpha))
    R  = np.stack([r['risk_curve'] for r in reps])
    A  = np.stack([r['ar_curve']   for r in reps])
    return {
        'final_risk_mean': float(fr.mean()), 'final_risk_std': float(fr.std()),
        'final_ar_mean':   float(fa.mean()), 'final_ar_std':   float(fa.std()),
        'max_risk_mean':   float(mx.mean()), 'max_risk_std':   float(mx.std()),
        'pathwise_violations':     pv,
        'pathwise_violation_rate': f'{pv}/{len(reps)}',
        'risk_curve': R.mean(axis=0).tolist(),
        'ar_curve':   A.mean(axis=0).tolist(),
        'n_reps':     len(reps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stream', required=True)
    ap.add_argument('--out',    required=True)
    ap.add_argument('--alphas', default='0.10,0.20,0.30')
    ap.add_argument('--n-shuffles', type=int, default=20)
    ap.add_argument('--n-passes',   type=int, default=5,
                    help='Shuffle the stream this many times to extend T. '
                         'Set to 1 for no repetition.')
    ap.add_argument('--burn-in',    type=int, default=500)
    ap.add_argument('--seed',       type=int, default=42)
    args = ap.parse_args()

    with open(args.stream) as f:
        data = json.load(f)
    raw_stream = data['stream']
    base_scores = np.array([r['score'] for r in raw_stream])
    base_Vs     = np.array([r['V']     for r in raw_stream])
    print(f'[replay] loaded {len(raw_stream)} genuine rounds, '
          f'n_passes={args.n_passes}, n_shuffles={args.n_shuffles}')

    alphas = [float(x) for x in args.alphas.split(',')]
    rng = np.random.default_rng(args.seed)

    out = {
        'source_stream': args.stream,
        'genuine_rounds': len(raw_stream),
        'n_passes': args.n_passes,
        'n_shuffles': args.n_shuffles,
        'burn_in_accepts': args.burn_in,
        'round_acc': data.get('round_acc', []),
    }

    for alpha in alphas:
        key = f'alpha_{alpha:g}'
        print(f'\n=== {key} ===')
        methods = build_methods(alpha, args.burn_in)
        per_method = {}
        for name, ctor in methods.items():
            reps = []
            for seed in range(args.n_shuffles):
                # Build shuffled stream.
                perm = np.concatenate([
                    rng.permutation(len(raw_stream))
                    for _ in range(args.n_passes)
                ])
                scores = base_scores[perm]
                Vs     = base_Vs[perm]
                reps.append(replay_single(scores, Vs, ctor(),
                                           args.burn_in, is_csa=(name == 'CSA-RLVR')))
            per_method[name] = aggregate(reps, alpha)
            a = per_method[name]
            print(f'  {name:18s}  risk={a["final_risk_mean"]:.3f}  '
                  f'ar={a["final_ar_mean"]:.3f}  '
                  f'PathV={a["pathwise_violation_rate"]}')
        out[key] = per_method

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[replay] wrote {args.out}')


if __name__ == '__main__':
    main()
