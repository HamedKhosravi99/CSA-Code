"""
Distribution-shift ablation for CSA-RLVR.

Anytime-valid risk control is meant to hold under non-stationary streams; the
test here is whether CSA continues to bound risk pathwise at a stream-internal
shift boundary. We construct one benchmark with a planted distribution shift
mid-stream and check the running risk crosses the slack-adjusted bound or not.

Stream construction:
    1. Sort MedQA EVAL items by calibrated_score (low -> high, i.e.
       easiest-first for CSA since low score = high confidence).
    2. Split at the median into an "easy" half (low-risk items) and a
       "hard" half (high-risk items).
    3. For each replication, shuffle WITHIN each half to add stream-level
       stochasticity but preserve the easy->hard shift.
    4. Replay all 7 methods (CSA + 6 baselines) over the shifted stream.

An iid baseline replay (items shuffled fully random) is also run for
comparison; both are done with 10 replications.

The resulting running-risk trajectories show whether each method crosses
the alpha bound at the seam (item ~N/2, where the model's error rate
jumps). CSA should keep PathV=0/10 and smoothly tighten; ACI/SAOCP should
lag and spike; Fixed-Threshold / Always-Act should violate immediately.

Output:
    results/ablation_shift.json     - summary (iid + shift, all methods)
    results/ablation_shift/         - per-rep raw curves for plotting

Usage:
    python ablate_shift.py
    python ablate_shift.py --alpha 0.20 --n_reps 10 --benchmark medical
"""

import argparse
import json
import os
import sys
import time
import warnings

# Suppress sklearn / numpy noise (LogisticRegression inside SAOCP triggers
# divide-by-zero / overflow warnings on degenerate batches; harmless).
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
np.seterr(all='ignore')
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csa_core import CSAConfig
from domains.medical.stream import MedQAStream
from domains.runner import _run_single_method, _create_baselines
from domains.surrogate import OnlineSurrogate


def build_shift_indices(df, n_passes, seed):
    """Build an index sequence that replays the full EVAL set `n_passes` times,
    each pass in easy-first->hard-last order with within-half shuffling.

    Items are sorted by calibrated_score ascending (low score = CSA-confident
    = expected-easy). We split at the median: indices [0..mid) are "easy",
    indices [mid..N) are "hard". Within each half we shuffle per pass.
    """
    scores = df['calibrated_score'].values
    order = np.argsort(scores)  # ascending
    n = len(order)
    mid = n // 2
    easy = order[:mid]
    hard = order[mid:]

    rng = np.random.RandomState(seed)
    indices = []
    for _ in range(n_passes):
        e = easy.copy(); rng.shuffle(e)
        h = hard.copy(); rng.shuffle(h)
        # Concatenate: first all easy items, then all hard items
        indices.extend(e.tolist())
        indices.extend(h.tolist())
    return np.array(indices, dtype=int), mid


def build_iid_indices(df, n_passes, seed):
    """Full-shuffle iid baseline order for comparison."""
    n = len(df)
    rng = np.random.RandomState(seed)
    indices = []
    for _ in range(n_passes):
        perm = np.arange(n)
        rng.shuffle(perm)
        indices.extend(perm.tolist())
    return np.array(indices, dtype=int)


def run_one(stream, indices, alpha, grid_min, grid_max, burn_in):
    """Run CSA + 6 baselines over a given index sequence. Returns dict per method."""
    a_max = min(grid_max, max(alpha * 3.0, alpha + 0.05))
    cfg = CSAConfig(alpha=alpha, delta=0.10, grid_size=15,
                    grid_min=grid_min, grid_max=a_max, single_epoch=True)
    out = {}

    # CSA
    surrogate = OnlineSurrogate(retrain_every=25, min_samples=15)
    r = _run_single_method(stream, indices, method=None, surrogate=surrogate,
                           is_csa=True, csa_config=cfg,
                           burn_in_accepts=burn_in)
    out['CSA-RLVR'] = r

    # Baselines (LTT is included by _create_baselines)
    baselines = _create_baselines(alpha, fixed_q=0.5, ltt_cal_size=burn_in)
    baseline_names = ['Always-Act', 'Fixed-Threshold', 'Naive-Tuning',
                      'ACI', 'SAOCP', 'LTT']
    for name, method in zip(baseline_names, baselines):
        sb = OnlineSurrogate(retrain_every=25, min_samples=15)
        r = _run_single_method(stream, indices, method=method, surrogate=sb,
                               is_csa=False, burn_in_accepts=burn_in)
        out[name] = r
    return out


def summarize(reps_results, alpha, burn_in_accepts=500):
    """Aggregate 10 reps of results into per-method statistics."""
    agg = {}
    method_names = list(reps_results[0].keys())
    for name in method_names:
        final_risks = [r[name]['final_risk'] for r in reps_results]
        final_ars   = [r[name]['final_ar']   for r in reps_results]
        max_risks   = [r[name]['max_risk']   for r in reps_results]
        pv = sum(int(mr > alpha) for mr in max_risks)
        agg[name] = {
            'final_risk_mean': float(np.mean(final_risks)),
            'final_risk_std':  float(np.std(final_risks)),
            'final_ar_mean':   float(np.mean(final_ars)),
            'final_ar_std':    float(np.std(final_ars)),
            'max_risk_mean':   float(np.mean(max_risks)),
            'max_risk_std':    float(np.std(max_risks)),
            'pathwise_violations': pv,
            'n_reps': len(reps_results),
            'pathwise_violation_rate': f'{pv}/{len(reps_results)}',
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.20)
    ap.add_argument('--n_reps', type=int, default=10)
    ap.add_argument('--n_passes', type=int, default=30)
    ap.add_argument('--burn_in_accepts', type=int, default=500)
    ap.add_argument('--csv', default='results/medical_inference_calibrated.csv')
    ap.add_argument('--meta', default='results/medical_inference_calibrated_meta.json')
    ap.add_argument('--out_dir', default='results/ablation_shift')
    args = ap.parse_args()

    # Load data + grid bounds
    df = pd.read_csv(args.csv)
    with open(args.meta) as f:
        meta = json.load(f)
    grid_min, grid_max = float(meta['grid_min']), float(meta['grid_max'])

    print(f"Shift ablation on {args.csv}")
    print(f"  n={len(df)}, alpha={args.alpha}, n_reps={args.n_reps}, "
          f"n_passes={args.n_passes}")
    print(f"  Grid: [{grid_min:.4f}, {grid_max:.4f}]")
    stream = MedQAStream(args.csv)

    # Compute "easy" vs "hard" empirical error (sanity check the shift)
    scores = df['calibrated_score'].values
    order = np.argsort(scores)
    mid = len(df) // 2
    easy_idx, hard_idx = order[:mid], order[mid:]
    V = df['correct'].values
    err_easy = 1 - V[easy_idx].mean()
    err_hard = 1 - V[hard_idx].mean()
    print(f"  Empirical err (easy half) = {err_easy:.3f}")
    print(f"  Empirical err (hard half) = {err_hard:.3f}")
    print(f"  Shift ratio (hard/easy) = {err_hard/max(err_easy,1e-6):.2f}x")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Shift condition -----------------------------------------------------
    print(f"\n=== SHIFT: easy -> hard concatenation, {args.n_reps} reps ===")
    shift_reps = []
    t0 = time.time()
    for rep in range(args.n_reps):
        seed = 42 + rep
        idx, _ = build_shift_indices(df, args.n_passes, seed)
        res = run_one(stream, idx, args.alpha, grid_min, grid_max,
                      args.burn_in_accepts)
        # Save per-rep raw curves (only CSA + 2 main baselines for size)
        for mname in ['CSA-RLVR', 'Always-Act', 'ACI', 'SAOCP']:
            np.save(f'{args.out_dir}/shift_rep{rep}_{mname}_riskcurve.npy',
                    res[mname]['risk_curve'])
        shift_reps.append({m: {k: v for k, v in res[m].items()
                               if k not in ('risk_curve', 'ar_curve')}
                           for m in res})
        print(f"  rep {rep+1}/{args.n_reps} done ({time.time()-t0:.1f}s)")
    shift_summary = summarize(shift_reps, args.alpha, args.burn_in_accepts)

    # --- IID condition (for comparison) --------------------------------------
    print(f"\n=== IID: full shuffle, {args.n_reps} reps (baseline for comparison) ===")
    iid_reps = []
    t0 = time.time()
    for rep in range(args.n_reps):
        seed = 42 + rep
        idx = build_iid_indices(df, args.n_passes, seed)
        res = run_one(stream, idx, args.alpha, grid_min, grid_max,
                      args.burn_in_accepts)
        iid_reps.append({m: {k: v for k, v in res[m].items()
                             if k not in ('risk_curve', 'ar_curve')}
                         for m in res})
        print(f"  rep {rep+1}/{args.n_reps} done ({time.time()-t0:.1f}s)")
    iid_summary = summarize(iid_reps, args.alpha, args.burn_in_accepts)

    # --- Write consolidated output ------------------------------------------
    out = {
        'benchmark': 'medical (MedQA)',
        'alpha': args.alpha,
        'n_reps': args.n_reps,
        'n_passes': args.n_passes,
        'burn_in_accepts': args.burn_in_accepts,
        'n_items': len(df),
        'err_easy': float(err_easy),
        'err_hard': float(err_hard),
        'shift_ratio': float(err_hard / max(err_easy, 1e-6)),
        'iid_summary':   iid_summary,
        'shift_summary': shift_summary,
    }
    out_path = 'results/ablation_shift.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # --- Pretty table --------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"  DISTRIBUTION-SHIFT ABLATION, MedQA @ alpha={args.alpha}, "
          f"{args.n_reps} reps")
    print(f"  easy_err={err_easy:.3f}  hard_err={err_hard:.3f}  "
          f"ratio={err_hard/max(err_easy,1e-6):.2f}x")
    print("=" * 90)
    for cond, summary in [('IID', iid_summary), ('SHIFT (easy->hard)',
                                                  shift_summary)]:
        print(f"\n  --- {cond} ---")
        print(f"  {'Method':<18} {'Risk':<9} {'AR':<9} {'MaxR':<9} "
              f"{'PathV':<10}")
        for name in ['CSA-RLVR', 'Always-Act', 'Fixed-Threshold',
                     'Naive-Tuning', 'ACI', 'SAOCP', 'LTT']:
            m = summary[name]
            star = ' *' if m['final_risk_mean'] > args.alpha else ''
            print(f"  {name:<18} {m['final_risk_mean']*100:<8.2f}% "
                  f"{m['final_ar_mean']*100:<8.2f}% "
                  f"{m['max_risk_mean']*100:<8.2f}% "
                  f"{m['pathwise_violation_rate']:<6}{star}")


if __name__ == '__main__':
    main()
