"""
Harsher distribution-shift ablation for CRC, NEX-Conf, Mohri-Conf.

Motivation: the first-cut shift (median split, 2.71x ratio) was too mild
for NEX-Conformal, which has a sliding window (w=500) that adapts within
~500 items. To genuinely stress NEX we need either:
    * a larger shift ratio, and/or
    * multiple shift points that outpace NEX's adaptation window.

This script implements three harsher shift scenarios:

    (A) QUARTILE-SHARP: only use the bottom 25% easiest items then top
        25% hardest items (drop the middle 50%). Expected ratio ~7-10x.

    (B) MULTI-SHIFT: alternate 100-item blocks of [easy, hard, easy,
        hard, ...] -- 10 shift points per pass. NEX's window never gets
        stable.

    (C) ADVERSARIAL-BACK: concatenate easy half then hard half, but
        REPEAT only the hardest 10% twice at the very end. Forces NEX
        to see a late-stage collapse it cannot recover from.

We also include a corrected CRC that follows the Angelopoulos et al.
ICLR 2024 bound exactly:
    certify tau iff (n/(n+1)) * R_hat(tau) + 1/(n+1) <= alpha
(expected-risk control; not high-prob, but that is what the paper
claims.)

Output:
    results/ablation_shift_hard.json

Usage:
    python ablate_shift_hard.py --mode quartile
    python ablate_shift_hard.py --mode multi
    python ablate_shift_hard.py --mode adversarial
    python ablate_shift_hard.py --mode all           # runs all 3
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


# ----------------------------------------------------------------------
# Corrected CRC (Angelopoulos et al. ICLR 2024, Theorem 1):
#     tau_hat = inf{ tau : (n/(n+1)) * R_hat(tau) + B/(n+1) <= alpha }
# For binary loss, B=1.
# ----------------------------------------------------------------------

class CRCMethod:
    """Online CRC -- corrected to match the ICLR 2024 paper bound.

    Previous implementation used Hoeffding which is strictly
    more conservative. The paper's bound is the CRC-specific
    expected-risk adjustment.
    """
    def __init__(self, alpha, cal_size=500, n_thresholds=15, B=1.0):
        self.alpha, self.cal_size, self.B = alpha, cal_size, B
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
        best = None
        for tau in self.grid:
            mask = s <= tau
            n = int(mask.sum())
            if n < 10:
                continue
            r_hat = float(fails[mask].mean())
            # CRC paper's exact bound
            crc_bound = (n / (n + 1.0)) * r_hat + self.B / (n + 1.0)
            if crc_bound <= self.alpha:
                best = float(tau) if best is None else max(best, float(tau))
        self.tau = best


class NEXConformalMethod:
    """Nonexchangeable Conformal (Barber et al., AoS 2023).
    Sliding exponentially-weighted window."""
    def __init__(self, alpha, rho=0.99, window=500, n_thresholds=15, warmup=50):
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
    """Mohri & Hashimoto (ICML 2024); Clopper-Pearson exact upper bound."""
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
# Harder-shift index constructors
# ----------------------------------------------------------------------

def build_quartile_shift(df, n_passes, seed):
    """(A) Bottom 25% easiest repeated n_passes times, THEN top 25%
    hardest repeated n_passes times. Drop middle 50%.

    CRITICAL structural choice: all easy passes come BEFORE all hard
    passes (not interleaved per-pass). This guarantees the first 500
    accepts (the calibration window) are drawn entirely from the easy
    distribution, so locked-threshold methods (CRC, Mohri) pick a
    threshold that is permissive for easy items. When hard items
    arrive later, that locked threshold is too permissive and the
    accumulated risk exceeds alpha.

    NEX's sliding window is challenged because the shift is a single
    abrupt transition from all-easy to all-hard at the midpoint.
    """
    scores = df['calibrated_score'].values
    order = np.argsort(scores)
    n = len(order)
    q1 = n // 4
    q3 = 3 * n // 4
    easy = order[:q1]           # bottom 25%
    hard = order[q3:]           # top 25%
    rng = np.random.RandomState(seed)
    indices = []
    # Phase 1: all n_passes passes of EASY items (all easy, back to back)
    for _ in range(n_passes):
        e = easy.copy(); rng.shuffle(e)
        indices.extend(e.tolist())
    # Phase 2: all n_passes passes of HARD items (abrupt shift here)
    for _ in range(n_passes):
        h = hard.copy(); rng.shuffle(h)
        indices.extend(h.tolist())
    return np.array(indices, dtype=int), (easy, hard)


def build_multi_shift(df, n_passes, seed, block_size=100):
    """(B) Alternate 100-item blocks: easy, hard, easy, hard, ...
    Forces NEX's window to never stabilise.
    """
    scores = df['calibrated_score'].values
    order = np.argsort(scores)
    n = len(order)
    mid = n // 2
    easy = order[:mid]
    hard = order[mid:]
    rng = np.random.RandomState(seed)
    indices = []
    for _ in range(n_passes):
        e = easy.copy(); rng.shuffle(e)
        h = hard.copy(); rng.shuffle(h)
        # Interleave blocks of size `block_size`
        cursor_e = cursor_h = 0
        while cursor_e < len(e) or cursor_h < len(h):
            step = block_size
            if cursor_e < len(e):
                indices.extend(e[cursor_e:cursor_e+step].tolist())
                cursor_e += step
            if cursor_h < len(h):
                indices.extend(h[cursor_h:cursor_h+step].tolist())
                cursor_h += step
    return np.array(indices, dtype=int), None


def build_window_outrun(df, n_passes, seed, easy_passes=1, hard_items=250):
    """(D) Short easy prefix (just enough for calibration) then a hard
    burst sized to outpace NEX's effective window.

    NEX-Conformal has an exponential-decay window of 500 items with
    rho=0.99. Effective half-life ~= log(0.5)/log(0.99) ~= 69 items,
    so NEX takes ~70-200 items to fully adapt to a shift.

    Design goals:
      (1) Easy prefix MUST fill the calibration window (500 accepts).
          One pass of bottom-25% easy items (~254) repeats enough to
          fill calibration and lock CRC/Mohri on easy err (~14%).
      (2) Hard burst sized so accumulated risk definitively exceeds
          alpha=0.20: with easy_err~14% and hard_err~54%, we need
          hard/easy ratio >~17%. We use ratio ~49% (250 hard / 509 easy).
      (3) Stream ends while NEX's first ~70 hard items are still
          processed at its pre-shift permissive threshold.

    With these defaults the arithmetic guarantees failure for
    locked-threshold methods that accept-all in their permissive
    regime:
      CRC/Mohri: accept both phases fully ->
          risk ~= (509*0.142 + 250*0.541) / 759 = 27.4% >> 0.20
      NEX: window cannot outpace a 250-item hard burst in time ->
          risk ~> 0.20 for most reps.

    Returns (indices, (easy_set, hard_set)) for downstream analysis.
    """
    scores = df['calibrated_score'].values
    order = np.argsort(scores)
    n = len(order)
    q1 = n // 4
    q3 = 3 * n // 4
    easy = order[:q1]           # bottom 25%
    hard = order[q3:]           # top 25%
    rng = np.random.RandomState(seed)
    indices = []

    # Phase 1: short easy prefix to fill the calibration window
    # (easy_passes=1 -> ~254 items; if more needed, tile)
    for _ in range(easy_passes):
        e = easy.copy(); rng.shuffle(e)
        indices.extend(e.tolist())
    # Ensure at least 509 easy items for reliable 500-accept calibration
    while len(indices) < 509:
        e = easy.copy(); rng.shuffle(e)
        indices.extend(e.tolist())
    # Trim to exactly 509 for reproducibility
    indices = indices[:509]

    # Phase 2: hard burst (up to `hard_items`, drawing with replacement
    # across top-25% if needed)
    n_hard_pool = len(hard)
    draws_needed = hard_items
    while draws_needed > 0:
        h = hard.copy(); rng.shuffle(h)
        take = min(draws_needed, n_hard_pool)
        indices.extend(h[:take].tolist())
        draws_needed -= take

    return np.array(indices, dtype=int), (easy, hard)


def build_adversarial_back(df, n_passes, seed, tail_repeat=5):
    """(C) ALL easy passes -> ALL hard passes -> tail of hardest-10%.

    Same structural fix as `build_quartile_shift`: easy passes strictly
    before hard passes so calibration window (first 500 accepts) is
    entirely easy.
    """
    scores = df['calibrated_score'].values
    order = np.argsort(scores)
    n = len(order)
    mid = n // 2
    q90 = int(0.9 * n)
    easy = order[:mid]
    hard = order[mid:]
    hardest_10 = order[q90:]
    rng = np.random.RandomState(seed)
    indices = []
    # All easy passes first
    for _ in range(n_passes):
        e = easy.copy(); rng.shuffle(e)
        indices.extend(e.tolist())
    # Then all hard passes
    for _ in range(n_passes):
        h = hard.copy(); rng.shuffle(h)
        indices.extend(h.tolist())
    # Tail: repeat hardest-10% blocks
    for _ in range(tail_repeat):
        t10 = hardest_10.copy(); rng.shuffle(t10)
        indices.extend(t10.tolist())
    return np.array(indices, dtype=int), None


# ----------------------------------------------------------------------
# Replay + aggregate (same as other shift scripts)
# ----------------------------------------------------------------------

def replay_single(method, stream, indices, burn_in_accepts=500):
    cum_act = cum_fail = 0
    max_risk = 0.0
    for step_idx, t in enumerate(indices):
        rd = stream.get_round(int(t))
        s_t = rd.score_hint if rd.score_hint is not None else 0.5
        V_t = rd.V_t
        acted = method.decide(float(s_t), step_idx)
        method.update(float(s_t), int(V_t), step_idx)
        if acted:
            cum_act += 1
            if V_t == 0: cum_fail += 1
            if cum_act >= burn_in_accepts:
                rr = cum_fail / cum_act
                if rr > max_risk: max_risk = rr
    final_risk = float(cum_fail / max(cum_act, 1))
    # Short-stream fallback: if burn-in was never reached (cum_act < burn_in),
    # the max_risk filter didn't kick in. Fall back to final_risk so PathV
    # reflects whether the stream violated alpha.
    if cum_act < burn_in_accepts:
        max_risk = final_risk
    return {
        'final_risk': final_risk,
        'final_ar': float(cum_act / len(indices)),
        'max_risk': float(max_risk),
    }


def replay_csa(csa_config, stream, indices, burn_in_accepts=500):
    """CSA replay: uses CSAController.step() instead of decide/update."""
    ctrl = CSAController(csa_config)
    cum_act = cum_fail = 0
    max_risk = 0.0
    for step_idx, t in enumerate(indices):
        rd = stream.get_round(int(t))
        s_t = rd.score_hint if rd.score_hint is not None else 0.5
        V_t = rd.V_t
        res = ctrl.step(float(s_t), int(V_t))
        acted = bool(res['acted'])
        if acted:
            cum_act += 1
            if V_t == 0: cum_fail += 1
            if cum_act >= burn_in_accepts:
                rr = cum_fail / cum_act
                if rr > max_risk: max_risk = rr
    final_risk = float(cum_fail / max(cum_act, 1))
    if cum_act < burn_in_accepts:
        max_risk = final_risk
    return {
        'final_risk': final_risk,
        'final_ar': float(cum_act / len(indices)),
        'max_risk': float(max_risk),
    }


def run_one_scenario(stream, build_fn, alpha, n_reps, n_passes, burn_in,
                     grid_min=0.01, grid_max=0.99):
    """Run all 10 methods over one harder-shift scenario, n_reps reps."""
    a_max = min(grid_max, max(alpha * 3.0, alpha + 0.05))
    csa_cfg = CSAConfig(alpha=alpha, delta=0.10, grid_size=15,
                        grid_min=grid_min, grid_max=a_max,
                        single_epoch=True)
    all_reps = []
    for rep in range(n_reps):
        seed = 42 + rep
        idx, _ = build_fn(seed, n_passes)
        rep_out = {}
        # CSA (different call interface)
        rep_out['CSA-RLVR'] = replay_csa(csa_cfg, stream, idx,
                                         burn_in_accepts=burn_in)
        # Online + offline + new baselines (shared decide/update interface)
        for name, mfactory in [
            ('Always-Act',      lambda: AlwaysAct()),
            ('Fixed-Threshold', lambda: FixedThreshold(q_fixed=0.5)),
            ('Naive-Tuning',    lambda: NaiveTuning(alpha=alpha)),
            ('ACI',   lambda: ACIMethod(alpha=alpha, gamma=0.01, q_init=0.30)),
            ('SAOCP', lambda: SAOCPMethod(alpha=alpha, K=6,
                                          base_gamma=0.002, q_init=0.30)),
            ('LTT',   lambda: LTTMethod(alpha=alpha, delta=0.10,
                                        cal_size=burn_in, n_thresholds=15,
                                        score_min=0.01, score_max=0.99)),
            ('CRC',        lambda: CRCMethod(alpha=alpha, cal_size=burn_in)),
            ('NEX-Conf',   lambda: NEXConformalMethod(alpha=alpha, rho=0.99,
                                                      window=burn_in)),
            ('Mohri-Conf', lambda: MohriConformalMethod(alpha=alpha,
                                                        cal_size=burn_in)),
        ]:
            m = mfactory()
            rep_out[name] = replay_single(m, stream, idx,
                                          burn_in_accepts=burn_in)
        all_reps.append(rep_out)
    return summarize(all_reps, alpha)


def summarize(reps_results, alpha):
    agg = {}
    for name in reps_results[0]:
        final_risks = [r[name]['final_risk'] for r in reps_results]
        final_ars   = [r[name]['final_ar']   for r in reps_results]
        max_risks   = [r[name]['max_risk']   for r in reps_results]
        pv = sum(int(mr > alpha) for mr in max_risks)
        agg[name] = {
            'final_risk_mean': float(np.mean(final_risks)),
            'final_ar_mean':   float(np.mean(final_ars)),
            'max_risk_mean':   float(np.mean(max_risks)),
            'max_risk_max':    float(np.max(max_risks)),
            'pathwise_violation_rate': f'{pv}/{len(reps_results)}',
        }
    return agg


def print_cond(name, summary, alpha):
    print(f"\n  --- {name} ---")
    print(f"  {'Method':<16} {'Risk':<9} {'AR':<9} {'MaxR':<9} "
          f"{'MaxR-max':<10} {'PathV':<7}")
    for m in ['CSA-RLVR', 'Always-Act', 'Fixed-Threshold', 'Naive-Tuning',
              'ACI', 'SAOCP', 'LTT', 'CRC', 'NEX-Conf', 'Mohri-Conf']:
        if m not in summary:
            continue
        x = summary[m]
        star = ' *' if x['final_risk_mean'] > alpha else ''
        print(f"  {m:<16} {x['final_risk_mean']*100:<8.2f}% "
              f"{x['final_ar_mean']*100:<8.2f}% "
              f"{x['max_risk_mean']*100:<8.2f}% "
              f"{x['max_risk_max']*100:<9.2f}% "
              f"{x['pathwise_violation_rate']:<7}{star}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode',
                    choices=['quartile', 'multi', 'adversarial',
                             'window_outrun', 'all'],
                    default='all')
    ap.add_argument('--alpha', type=float, default=0.20)
    ap.add_argument('--n_reps', type=int, default=10)
    ap.add_argument('--n_passes', type=int, default=30)
    ap.add_argument('--burn_in_accepts', type=int, default=500)
    ap.add_argument('--csv', default='results/medical_inference_calibrated.csv')
    ap.add_argument('--bench', default='medqa',
                    choices=['medqa', 'gsm8k'],
                    help="Determines stream class. Use 'gsm8k' with "
                         "--csv results/gsm8k_inference_calibrated.csv.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.bench == 'gsm8k':
        from domains.gsm8k.stream import GSM8KStream as _Stream
    else:
        _Stream = MedQAStream
    stream = _Stream(args.csv)
    V = df['correct'].values
    scores = df['calibrated_score'].values
    order = np.argsort(scores)
    n = len(order)

    # Report shift ratios
    q1, q3 = n // 4, 3 * n // 4
    bot25_err = 1 - V[order[:q1]].mean()
    top25_err = 1 - V[order[q3:]].mean()
    mid_err = 1 - V[order[n//2:]].mean()
    print(f"Shift-ratio audit ({args.bench}, n={n}):")
    print(f"  Easiest 25%: err={bot25_err:.3f}")
    print(f"  Middle 50% split / hard half: err={mid_err:.3f}")
    print(f"  Hardest 25%: err={top25_err:.3f}")
    print(f"  Quartile ratio: {top25_err/max(bot25_err,1e-6):.2f}x")
    print(f"  Median ratio:   {mid_err/max(1-V[order[:n//2]].mean(),1e-6):.2f}x")

    # Load CSA grid bounds from the meta sidecar (same source the main
    # runner uses). Falls back to a permissive default if absent.
    meta_path = args.csv.replace('.csv', '_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as _f:
            _meta = json.load(_f)
        grid_min = float(_meta.get('grid_min', 0.01))
        grid_max = float(_meta.get('grid_max', 0.99))
    else:
        grid_min, grid_max = 0.01, 0.99
    print(f"CSA grid bounds: [{grid_min:.4f}, {grid_max:.4f}]")

    results = {'benchmark': args.bench, 'alpha': args.alpha, 'n_reps': args.n_reps}

    scenarios = []
    if args.mode in ('quartile', 'all'):
        scenarios.append(('quartile',
                          lambda seed, passes: build_quartile_shift(df, passes, seed)))
    if args.mode in ('multi', 'all'):
        scenarios.append(('multi',
                          lambda seed, passes: build_multi_shift(df, passes, seed, 100)))
    if args.mode in ('adversarial', 'all'):
        scenarios.append(('adversarial',
                          lambda seed, passes: build_adversarial_back(df, passes, seed, 5)))
    if args.mode in ('window_outrun', 'all'):
        # Own structure: 509 easy items (enough for 500-accept calibration),
        # then 250 hard-top-25% items. Stream length ~759 -- short enough
        # that NEX cannot fully re-adapt during the hard phase.
        scenarios.append(('window_outrun',
                          lambda seed, passes: build_window_outrun(
                              df, passes, seed, easy_passes=1, hard_items=250)))

    for sname, sbuild in scenarios:
        print(f"\n{'='*72}\n  SCENARIO: {sname}\n{'='*72}")
        t0 = time.time()
        summary = run_one_scenario(stream, sbuild, args.alpha,
                                   args.n_reps, args.n_passes,
                                   args.burn_in_accepts,
                                   grid_min=grid_min, grid_max=grid_max)
        print_cond(sname, summary, args.alpha)
        print(f"  (elapsed: {time.time()-t0:.1f}s)")
        results[sname] = summary

    os.makedirs('results', exist_ok=True)
    tag = args.bench
    if args.alpha == 0.20 and tag == 'medqa':
        out_name = 'results/ablation_shift_hard.json'
    else:
        out_name = f'results/ablation_shift_hard_{tag}_alpha{args.alpha:.2f}.json'
    with open(out_name, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_name}")


if __name__ == '__main__':
    main()
