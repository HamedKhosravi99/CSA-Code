"""
Distribution-shift ablation for the 3 recent SOTA baselines:
    * CRC         -- Conformal Risk Control (Angelopoulos et al., ICLR 2024)
    * NEX-Conf    -- Nonexchangeable Conformal (Barber et al., AoS 2023)
    * Mohri-Conf  -- Conformal Factuality (Mohri & Hashimoto, ICML 2024)

Runs the SAME shift construction as `ablate_shift.py` (so results are
directly comparable to the CSA/LTT shift ablation in the paper):

    1. Sort MedQA EVAL items by calibrated_score (low -> high, i.e.
       easiest-first for CSA since low score = high confidence).
    2. Split at the median into an "easy" half (low-risk items) and a
       "hard" half (high-risk items).
    3. For each replication, shuffle WITHIN each half; concatenate
       easy -> hard; repeat n_passes times.
    4. Replay each of the 3 new methods over the shifted stream.

An iid baseline replay (items fully shuffled) is also run for comparison;
both conditions use 10 replications.

The existing `ablate_shift.py` covers CSA + 6 baselines including LTT.
This file *only* tests the 3 new baselines and writes a SEPARATE
summary JSON, so nothing in the main paper's tables is perturbed.

Output:
    results/ablation_shift_new_baselines.json

Usage:
    python ablate_shift_new_baselines.py
    python ablate_shift_new_baselines.py --n_reps 2      # smoke
    python ablate_shift_new_baselines.py --alpha 0.20    # default
"""

import argparse
import json
import os
import sys
import time
import warnings

# Same noise suppression as ablate_shift.py
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
np.seterr(all='ignore')
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domains.medical.stream import MedQAStream


# ----------------------------------------------------------------------
# Three new baseline classes inlined here (identical to those in
# test_new_baselines.py; duplicated so this script is self-contained).
# ----------------------------------------------------------------------

class CRCMethod:
    """Online CRC (Angelopoulos et al., ICLR 2024).
    Warm-start on first cal_size accepts; certify the largest tau such
    that R_hat(tau) + sqrt(log(1/adj_delta)/(2n)) <= alpha. Threshold
    locked thereafter.
    """
    def __init__(self, alpha, delta=0.10, cal_size=500, n_thresholds=15):
        self.alpha, self.delta, self.cal_size = alpha, delta, cal_size
        self.grid = np.linspace(0.01, 0.99, n_thresholds)
        self.cal_s, self.cal_V = [], []
        self.tau = None; self.fitted = False
        self.name = "CRC"

    def decide(self, s, t):
        return self.fitted and self.tau is not None and s <= self.tau

    def update(self, s, V, t):
        if self.fitted:
            return
        self.cal_s.append(float(s)); self.cal_V.append(int(V))
        if len(self.cal_s) >= self.cal_size:
            self._fit(); self.fitted = True

    def _fit(self):
        s = np.asarray(self.cal_s); V = np.asarray(self.cal_V)
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


class NEXConformalMethod:
    """Nonexchangeable Conformal (Barber et al., AoS 2023).
    Sliding window of recent (s_i, V_i); exponential-recency-weighted
    risk; pick largest tau whose weighted risk <= alpha.
    """
    def __init__(self, alpha, rho=0.99, window=500, n_thresholds=15,
                 warmup=50):
        self.alpha, self.rho, self.window, self.warmup = alpha, rho, window, warmup
        self.grid = np.linspace(0.01, 0.99, n_thresholds)
        self.hs, self.hV = [], []
        self.tau = None
        self.name = "NEX-Conf"

    def decide(self, s, t):
        return self.tau is not None and s <= self.tau

    def update(self, s, V, t):
        self.hs.append(float(s)); self.hV.append(int(V))
        if len(self.hs) > self.window:
            self.hs = self.hs[-self.window:]; self.hV = self.hV[-self.window:]
        if len(self.hs) < self.warmup:
            return
        n = len(self.hs)
        scores = np.asarray(self.hs); fails = 1 - np.asarray(self.hV)
        ages = np.arange(n - 1, -1, -1, dtype=float)
        w = self.rho ** ages; w = w / max(w.sum(), 1e-12)
        best = None
        for tau in sorted(self.grid, reverse=True):
            mask = scores <= tau
            if not mask.any():
                continue
            ws = w[mask].sum()
            if ws <= 1e-12:
                continue
            wr = float((w[mask] * fails[mask]).sum()) / float(ws)
            if wr <= self.alpha:
                best = float(tau); break
        if best is not None:
            self.tau = best


class MohriConformalMethod:
    """Mohri & Hashimoto (ICML 2024), adapted to binary verifier.
    Clopper-Pearson exact upper bound on Bernoulli failure rate; certify
    largest tau such that upper <= alpha.
    """
    def __init__(self, alpha, delta=0.10, cal_size=500, n_thresholds=15):
        self.alpha, self.delta, self.cal_size = alpha, delta, cal_size
        self.grid = np.linspace(0.01, 0.99, n_thresholds)
        self.cal_s, self.cal_V = [], []
        self.tau = None; self.fitted = False
        self.name = "Mohri-Conf"

    def decide(self, s, t):
        return self.fitted and self.tau is not None and s <= self.tau

    def update(self, s, V, t):
        if self.fitted:
            return
        self.cal_s.append(float(s)); self.cal_V.append(int(V))
        if len(self.cal_s) >= self.cal_size:
            self._fit(); self.fitted = True

    def _fit(self):
        from scipy.stats import beta
        s = np.asarray(self.cal_s); V = np.asarray(self.cal_V)
        fails = 1 - V
        adj_delta = self.delta / len(self.grid)
        best = None
        for tau in self.grid:
            mask = s <= tau
            n = int(mask.sum())
            if n < 10:
                continue
            k = int(fails[mask].sum())
            upper = 1.0 if k == n else float(beta.ppf(1.0 - adj_delta, k+1, n-k))
            if upper <= self.alpha:
                best = float(tau) if best is None else max(best, float(tau))
        self.tau = best


# ----------------------------------------------------------------------
# Index constructors (identical to ablate_shift.py)
# ----------------------------------------------------------------------

def build_shift_indices(df, n_passes, seed):
    """Easy-first -> hard-last, shuffled within halves each pass.

    Matches `ablate_shift.py::build_shift_indices` exactly.
    """
    scores = df['calibrated_score'].values
    order = np.argsort(scores)                       # ascending
    n = len(order)
    mid = n // 2
    easy = order[:mid]
    hard = order[mid:]
    rng = np.random.RandomState(seed)
    indices = []
    for _ in range(n_passes):
        e = easy.copy(); rng.shuffle(e)
        h = hard.copy(); rng.shuffle(h)
        indices.extend(e.tolist())
        indices.extend(h.tolist())
    return np.array(indices, dtype=int), mid


def build_iid_indices(df, n_passes, seed):
    """Full-shuffle per pass (iid control)."""
    n = len(df)
    rng = np.random.RandomState(seed)
    indices = []
    for _ in range(n_passes):
        perm = np.arange(n); rng.shuffle(perm)
        indices.extend(perm.tolist())
    return np.array(indices, dtype=int)


# ----------------------------------------------------------------------
# Replay helper: mirrors ablate_shift.run_one() but only iterates over
# the three new methods (no CSA, no other baselines).
# ----------------------------------------------------------------------

def replay_single(method, stream, indices, burn_in_accepts=500):
    """Run one method over one stream pass. Returns final/max metrics.

    Burn-in convention: max_risk only starts being tracked after
    `burn_in_accepts` items have been acted on. Identical semantics to
    `domains/runner.py::_run_single_method`.
    """
    cum_act = 0
    cum_fail = 0
    max_risk = 0.0
    n_steps = len(indices)
    for step_idx in range(n_steps):
        t = int(indices[step_idx])
        rd = stream.get_round(t)
        s_t = rd.score_hint if rd.score_hint is not None else 0.5
        V_t = rd.V_t
        acted = method.decide(float(s_t), step_idx)
        method.update(float(s_t), int(V_t), step_idx)
        if acted:
            cum_act += 1
            if V_t == 0:
                cum_fail += 1
            if cum_act >= burn_in_accepts:
                running_risk = cum_fail / cum_act
                if running_risk > max_risk:
                    max_risk = running_risk
    final_risk = cum_fail / max(cum_act, 1) if cum_act > 0 else 0.0
    final_ar = cum_act / n_steps if n_steps > 0 else 0.0
    return {
        'final_risk': float(final_risk),
        'final_ar':   float(final_ar),
        'max_risk':   float(max_risk),
    }


def run_one(stream, indices, alpha, burn_in):
    """Run all 3 new methods over one stream pass."""
    out = {}
    # Fresh method instances per call (stream state has to be clean)
    methods = {
        'CRC':        CRCMethod(alpha=alpha, delta=0.10, cal_size=burn_in),
        'NEX-Conf':   NEXConformalMethod(alpha=alpha, rho=0.99, window=burn_in),
        'Mohri-Conf': MohriConformalMethod(alpha=alpha, delta=0.10, cal_size=burn_in),
    }
    for name, m in methods.items():
        r = replay_single(m, stream, indices, burn_in_accepts=burn_in)
        out[name] = r
    return out


def summarize(reps_results, alpha):
    """Aggregate reps into per-method statistics (same schema as ablate_shift.py)."""
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


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.20)
    ap.add_argument('--n_reps', type=int, default=10)
    ap.add_argument('--n_passes', type=int, default=30)
    ap.add_argument('--burn_in_accepts', type=int, default=500)
    ap.add_argument('--csv',
                    default='results/medical_inference_calibrated.csv')
    ap.add_argument('--meta',
                    default='results/medical_inference_calibrated_meta.json')
    ap.add_argument('--out_dir',
                    default='results/ablation_shift_new_baselines')
    args = ap.parse_args()

    # Load data + grid bounds (same as ablate_shift.py)
    df = pd.read_csv(args.csv)
    with open(args.meta) as f:
        meta = json.load(f)
    grid_min, grid_max = float(meta['grid_min']), float(meta['grid_max'])

    print(f"Shift ablation (new baselines only) on {args.csv}")
    print(f"  n={len(df)}, alpha={args.alpha}, n_reps={args.n_reps}, "
          f"n_passes={args.n_passes}")
    print(f"  Grid: [{grid_min:.4f}, {grid_max:.4f}]")
    stream = MedQAStream(args.csv)

    # Sanity-check the shift (identical code path as ablate_shift.py)
    scores = df['calibrated_score'].values
    order = np.argsort(scores)
    mid = len(df) // 2
    easy_idx, hard_idx = order[:mid], order[mid:]
    V = df['correct'].values
    err_easy = 1 - V[easy_idx].mean()
    err_hard = 1 - V[hard_idx].mean()
    print(f"  Empirical err (easy half) = {err_easy:.3f}")
    print(f"  Empirical err (hard half) = {err_hard:.3f}")
    print(f"  Shift ratio (hard/easy) = "
          f"{err_hard/max(err_easy,1e-6):.2f}x")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Shift condition ------------------------------------------------
    print(f"\n=== SHIFT: easy -> hard concatenation, {args.n_reps} reps ===")
    shift_reps = []
    t0 = time.time()
    for rep in range(args.n_reps):
        seed = 42 + rep
        idx, _ = build_shift_indices(df, args.n_passes, seed)
        res = run_one(stream, idx, args.alpha, args.burn_in_accepts)
        shift_reps.append(res)
        print(f"  rep {rep+1}/{args.n_reps} done ({time.time()-t0:.1f}s)")
    shift_summary = summarize(shift_reps, args.alpha)

    # --- IID condition --------------------------------------------------
    print(f"\n=== IID: full shuffle, {args.n_reps} reps (comparison) ===")
    iid_reps = []
    t0 = time.time()
    for rep in range(args.n_reps):
        seed = 42 + rep
        idx = build_iid_indices(df, args.n_passes, seed)
        res = run_one(stream, idx, args.alpha, args.burn_in_accepts)
        iid_reps.append(res)
        print(f"  rep {rep+1}/{args.n_reps} done ({time.time()-t0:.1f}s)")
    iid_summary = summarize(iid_reps, args.alpha)

    # --- Write consolidated output (same schema as ablate_shift.json) ---
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
        'methods_tested': ['CRC', 'NEX-Conf', 'Mohri-Conf'],
        'iid_summary':   iid_summary,
        'shift_summary': shift_summary,
    }
    out_path = 'results/ablation_shift_new_baselines.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # --- Pretty table --------------------------------------------------
    print("\n" + "=" * 90)
    print(f"  DISTRIBUTION-SHIFT ABLATION, NEW BASELINES on MedQA @ alpha={args.alpha}, "
          f"{args.n_reps} reps")
    print(f"  easy_err={err_easy:.3f}  hard_err={err_hard:.3f}  "
          f"ratio={err_hard/max(err_easy,1e-6):.2f}x")
    print("=" * 90)
    for cond, summary in [('IID', iid_summary),
                           ('SHIFT (easy->hard)', shift_summary)]:
        print(f"\n  --- {cond} ---")
        print(f"  {'Method':<14} {'Risk':<9} {'AR':<9} {'MaxR':<9} "
              f"{'PathV':<10}")
        for name in ['CRC', 'NEX-Conf', 'Mohri-Conf']:
            m = summary[name]
            star = ' *' if m['final_risk_mean'] > args.alpha else ''
            print(f"  {name:<14} {m['final_risk_mean']*100:<8.2f}% "
                  f"{m['final_ar_mean']*100:<8.2f}% "
                  f"{m['max_risk_mean']*100:<8.2f}% "
                  f"{m['pathwise_violation_rate']:<6}{star}")


if __name__ == '__main__':
    main()
