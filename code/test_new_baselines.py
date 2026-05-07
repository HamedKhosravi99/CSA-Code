"""
Standalone test of three recent SOTA risk-control baselines:
    * CRC         -- Conformal Risk Control (Angelopoulos et al., ICLR 2024)
    * NEX-Conf    -- Nonexchangeable Conformal (Barber et al., AoS 2023)
    * Mohri-Conf  -- Conformal Factuality     (Mohri & Hashimoto, ICML 2024)

These baselines are NOT wired into the production runner
(`domains/runner.py`). This script runs them independently against the
released calibrated CSVs and writes a standalone summary JSON, so we
can decide whether to include them in the paper without perturbing any
existing result.

Output:
    results/new_baselines_summary.json

Usage:
    python test_new_baselines.py                       # all 8 benchmarks
    python test_new_baselines.py --only medical        # one only
    python test_new_baselines.py --n_reps 5            # faster / fewer reps
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

# ----------------------------------------------------------------------
# Three new baseline classes (inlined here so we don't touch production
# principled_baselines.py). Interface matches existing baselines:
#     decide(s_t, t) -> bool
#     update(s_t, V_t, t) -> None
# ----------------------------------------------------------------------

class CRCMethod:
    """Online Conformal Risk Control (Angelopoulos et al., ICLR 2024).

    Warm-start with first cal_size accepts; certify the largest tau such
    that R_hat(tau) + sqrt(log(1/adj_delta)/(2n)) <= alpha. Threshold
    locked for rest of stream.
    """
    def __init__(self, alpha, delta=0.10, cal_size=500, n_thresholds=15):
        self.alpha, self.delta, self.cal_size = alpha, delta, cal_size
        self.grid = np.linspace(0.01, 0.99, n_thresholds)
        self.cal_s, self.cal_V = [], []
        self.tau = None
        self.fitted = False
        self.name = "CRC"

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
    """Nonexchangeable (weighted-quantile) Conformal (Barber et al., 2023).

    Maintain a sliding window of recent (s_i, V_i); compute
    exponential-recency-weighted empirical risk for each candidate tau;
    pick largest tau whose weighted risk <= alpha.
    """
    def __init__(self, alpha, rho=0.99, window=500, n_thresholds=15,
                 warmup=50):
        self.alpha, self.rho, self.window = alpha, rho, window
        self.warmup = warmup
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

    Uses Clopper-Pearson exact upper bound on Bernoulli failure rate:
        upper = Beta^{-1}(1 - adj_delta; k+1, n-k)
    Certify largest tau such that upper <= alpha.
    """
    def __init__(self, alpha, delta=0.10, cal_size=500, n_thresholds=15):
        self.alpha, self.delta, self.cal_size = alpha, delta, cal_size
        self.grid = np.linspace(0.01, 0.99, n_thresholds)
        self.cal_s, self.cal_V = [], []
        self.tau = None
        self.fitted = False
        self.name = "Mohri-Conf"

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
# Replay helper: run one method over a shuffled stream
# ----------------------------------------------------------------------

def replay_single(method, stream, indices, burn_in_accepts=500):
    """Run one method over one stream pass. Returns final/max metrics."""
    n_acts = 0
    n_fails = 0
    max_risk = 0.0
    risk_curve_values = []
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
        risk_curve_values.append(cum_fail / max(cum_act, 1))

    final_risk = cum_fail / max(cum_act, 1)
    final_ar = cum_act / len(indices)
    return {
        'final_risk': float(final_risk),
        'final_ar': float(final_ar),
        'max_risk': float(max_risk),
        'n_acts': cum_act,
        'n_fails': cum_fail,
    }


# ----------------------------------------------------------------------
# Benchmark registry: pivotal alpha + stream-class resolver
# ----------------------------------------------------------------------

UNIFORM_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BENCHMARKS = {
    'medical':  {'pivotal': 0.20, 'module': 'domains.medical.stream',   'cls': 'MedQAStream',   'label': 'MedQA',
                 'alphas': UNIFORM_GRID},
    'pubmedqa': {'pivotal': 0.20, 'module': 'domains.pubmedqa.stream',  'cls': 'PubMedQAStream','label': 'PubMedQA',
                 'alphas': UNIFORM_GRID},
    'tatqa':    {'pivotal': 0.20, 'module': 'domains.tatqa.stream',     'cls': 'TATQAStream',   'label': 'TAT-QA',
                 'alphas': UNIFORM_GRID},
    'mednli':   {'pivotal': 0.20, 'module': 'domains.mednli.stream',    'cls': 'MedNLIStream',  'label': 'MedNLI',
                 'alphas': UNIFORM_GRID},
    'gsm8k':    {'pivotal': 0.05, 'module': 'domains.gsm8k.stream',     'cls': 'GSM8KStream',   'label': 'GSM8K',
                 # Keep extra tight-alpha cells (0.01, 0.03, 0.075) + union with uniform grid
                 'alphas': [0.01, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]},
    'headqa':   {'pivotal': 0.20, 'module': 'domains.headqa.stream',    'cls': 'HEADQAStream',  'label': 'HEAD-QA',
                 'alphas': [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]},
    'arc':      {'pivotal': 0.10, 'module': 'domains.arc.stream',       'cls': 'ARCStream',     'label': 'ARC',
                 'alphas': [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]},
    'casehold': {'pivotal': 0.25, 'module': 'domains.casehold.stream',  'cls': 'CaseHOLDStream','label': 'CaseHOLD',
                 'alphas': UNIFORM_GRID},
}


def resolve_stream(bench, csv_path):
    import importlib
    spec = BENCHMARKS[bench]
    m = importlib.import_module(spec['module'])
    return getattr(m, spec['cls'])(csv_path)


def find_csv(bench):
    # MedNLI uses a different file-name convention: `<bench>_calibrated.csv`.
    # Some benchmarks (MMLU-ProfMed, FPB, Financial-logprob variants) also
    # follow that pattern.
    candidates = [
        f'results/{bench}_inference_calibrated.csv',
        f'results/{bench}_calibrated.csv',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def aggregate(reps, alpha):
    import numpy as np
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


def run_bench(bench, n_reps=10, n_passes=30, burn_in=500, alphas=None):
    """Run CRC, NEX-Conf, Mohri-Conf across the full alpha grid for
    a single benchmark."""
    spec = BENCHMARKS[bench]
    # Use the paper-main alpha grid for this benchmark unless overridden
    alpha_grid = alphas if alphas is not None else spec['alphas']
    csv = find_csv(bench)
    if csv is None:
        print(f"[SKIP] {bench}: no calibrated CSV")
        return None

    print(f"\n{'='*72}")
    print(f"  {spec['label']}  (n_reps={n_reps}, alphas={alpha_grid})")
    print(f"{'='*72}")
    print(f"  csv: {csv}")
    stream = resolve_stream(bench, csv)
    n_items = len(stream)
    T = n_passes * n_items
    print(f"  n_items={n_items}, T={T}")

    # Pre-compute IID shuffle indices once per rep, reused across alphas
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
            'CRC':        lambda a=alpha: CRCMethod(alpha=a, delta=0.10, cal_size=burn_in),
            'NEX-Conf':   lambda a=alpha: NEXConformalMethod(alpha=a, rho=0.99, window=burn_in),
            'Mohri-Conf': lambda a=alpha: MohriConformalMethod(alpha=a, delta=0.10, cal_size=burn_in),
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
            print(f"    {mname:<12} Risk={agg['final_risk_mean']*100:6.2f}%  "
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
    ap.add_argument('--burn_in_accepts', type=int, default=500)
    ap.add_argument('--force', action='store_true',
                    help="Re-compute cells that already exist in the "
                         "summary JSON. Default: skip already-computed cells.")
    args = ap.parse_args()

    targets = [args.only] if args.only else list(BENCHMARKS.keys())

    # Load existing summary so we can skip cells that are already done
    # unless --force is passed.
    os.makedirs('results', exist_ok=True)
    out_path = 'results/new_baselines_summary.json'
    if os.path.exists(out_path) and not args.force:
        with open(out_path) as f:
            all_out = json.load(f)
        print(f"Loaded existing {out_path} (will skip already-computed cells)")
    else:
        all_out = {}

    total_t0 = time.time()
    for bench in targets:
        spec = BENCHMARKS[bench]
        # Determine which alphas are missing for this benchmark
        done_alphas = set()
        if bench in all_out and 'per_alpha' in all_out[bench]:
            for k in all_out[bench]['per_alpha']:
                try:
                    done_alphas.add(round(float(k), 3))
                except Exception:
                    pass
        needed_alphas = [a for a in spec['alphas']
                         if round(a, 3) not in done_alphas]
        if args.force:
            needed_alphas = spec['alphas']

        if not needed_alphas:
            print(f"\n[SKIP] {bench}: all alphas already computed "
                  f"({sorted(done_alphas)})")
            continue

        if bench in all_out and not args.force:
            print(f"\n[PARTIAL] {bench}: have {sorted(done_alphas)}, "
                  f"running missing {needed_alphas}")
        # Run only the missing alphas
        r = run_bench(bench, n_reps=args.n_reps, n_passes=args.n_passes,
                      burn_in=args.burn_in_accepts, alphas=needed_alphas)
        if r is not None:
            # Merge into existing bench entry if present
            if bench in all_out:
                all_out[bench]['per_alpha'].update(r['per_alpha'])
                # Refresh alphas_grid as the union
                union = sorted(set(all_out[bench].get('alphas_grid', []))
                               | set(r['alphas_grid']))
                all_out[bench]['alphas_grid'] = union
            else:
                all_out[bench] = r

    with open(out_path, 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total elapsed: {time.time()-total_t0:.1f}s")

    # --- Headline table: all (bench, alpha) cells flattened ----------
    print("\n" + "="*100)
    print(f"  HEADLINE: new SOTA baselines, full alpha grid, {args.n_reps} reps")
    print("="*100)
    print(f"  {'Benchmark':<10} {'alpha':<7}"
          f"{'CRC PV':<9}{'CRC AR':<9}"
          f"{'NEX PV':<9}{'NEX AR':<9}"
          f"{'Mohri PV':<10}{'Mohri AR':<9}")
    print("  " + "-" * 72)

    total_streams = {'CRC': [0, 0], 'NEX-Conf': [0, 0], 'Mohri-Conf': [0, 0]}  # [pv, n_reps]

    for bench, r in all_out.items():
        label = BENCHMARKS[bench]['label']
        for alpha_str, cell in r['per_alpha'].items():
            row = f"  {label:<10} {alpha_str:<7}"
            for m in ['CRC', 'NEX-Conf', 'Mohri-Conf']:
                mm = cell.get(m, {})
                row += (f"{mm.get('pathwise_violation_rate','?'):<9}"
                        f"{mm.get('final_ar_mean',0)*100:<8.1f}%")
                # accumulate
                pv_str = mm.get('pathwise_violation_rate', '0/0')
                if '/' in pv_str:
                    num, den = map(int, pv_str.split('/'))
                    total_streams[m][0] += num
                    total_streams[m][1] += den
            print(row)

    # Grand totals
    print("  " + "-" * 72)
    print("  Grand totals (all (benchmark, alpha) cells):")
    for m in ['CRC', 'NEX-Conf', 'Mohri-Conf']:
        pv, n = total_streams[m]
        rate = pv / max(n, 1) * 100
        print(f"    {m:<12} {pv}/{n}  pathwise violations "
              f"({rate:.1f}% streams violated)")


if __name__ == '__main__':
    main()
