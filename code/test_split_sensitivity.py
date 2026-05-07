"""
Split-sensitivity ablation: re-run CSA with different calibration seeds.

The 80/20 CAL/EVAL split in calibrate_scores.py uses seed=42. This script
re-calibrates with multiple seeds and re-runs CSA at each benchmark's
pivotal alpha, showing that the safety guarantee (PathV=0) holds regardless
of the random partition and that action rates are stable.

Only CSA is tested (baselines are not affected by the calibration split
in a way that changes the paper's conclusions).

Output: results/split_sensitivity_summary.json

Usage:
    python test_split_sensitivity.py
    python test_split_sensitivity.py --only medical
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
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csa_core import CSAConfig, CSAController
from calibrate_scores import detect_raw_score, detect_fail


POD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'Documents', 'GitHub', 'RLVR', 'others',
    'csa_domains', 'pod_backup_20260419', 'results')

BENCHMARKS = {
    'medical': {
        'source_csv': 'medical_inference_sc.csv',
        'stream_module': 'domains.medical.stream',
        'stream_cls': 'MedQAStream',
        'pivotal_alpha': 0.20,
        'label': 'MedQA',
    },
    'gsm8k': {
        'source_csv': 'gsm8k_inference_sc.csv',
        'stream_module': 'domains.gsm8k.stream',
        'stream_cls': 'GSM8KStream',
        'pivotal_alpha': 0.05,
        'label': 'GSM8K',
    },
    'arc': {
        'source_csv': 'arc_inference_sc.csv',
        'stream_module': 'domains.arc.stream',
        'stream_cls': 'ARCStream',
        'pivotal_alpha': 0.10,
        'label': 'ARC',
    },
}


def recalibrate(source_csv, seed, split_ratio=0.2):
    """Re-run isotonic calibration with a different seed.

    Returns (eval_csv_path, grid_min, grid_max, n_cal, n_eval, meta).
    Writes a temporary calibrated CSV.
    """
    df = pd.read_csv(source_csv)
    n = len(df)
    raw = detect_raw_score(df)
    fail = detect_fail(df)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_cal = int(round(n * split_ratio))
    cal_idx = perm[:n_cal]
    eval_idx = perm[n_cal:]

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(raw[cal_idx], fail[cal_idx])

    eval_raw = raw[eval_idx]
    eval_calibrated = iso.predict(eval_raw)

    eval_df = df.iloc[eval_idx].copy().reset_index(drop=True)
    eval_df['raw_score'] = eval_raw
    eval_df['calibrated_score'] = eval_calibrated

    cal_calibrated = iso.predict(raw[cal_idx])
    grid_min = float(max(np.percentile(cal_calibrated, 2), 0.001))
    grid_max = float(min(np.percentile(cal_calibrated, 98), 0.8))

    os.makedirs('results/_split_tmp', exist_ok=True)
    base = os.path.splitext(os.path.basename(source_csv))[0]
    out_csv = f'results/_split_tmp/{base}_seed{seed}.csv'
    eval_df.to_csv(out_csv, index=False)

    meta = {
        'seed': seed, 'n_cal': n_cal, 'n_eval': len(eval_idx),
        'cal_fail_rate': float(fail[cal_idx].mean()),
        'eval_fail_rate': float(fail[eval_idx].mean()),
        'grid_min': grid_min, 'grid_max': grid_max,
    }
    return out_csv, grid_min, grid_max, meta


def load_stream(bench, csv_path):
    import importlib
    spec = BENCHMARKS[bench]
    m = importlib.import_module(spec['stream_module'])
    return getattr(m, spec['stream_cls'])(csv_path)


def run_csa_only(stream, alpha, delta, grid_min, grid_max,
                 n_reps=10, n_passes=30, base_seed=42,
                 burn_in_accepts=500):
    """Run CSA at a single alpha, return per-rep results."""
    n_items = len(stream)
    T = n_items * n_passes
    results = []

    for rep in range(n_reps):
        rng = np.random.RandomState(base_seed + rep)
        indices = np.concatenate([rng.permutation(n_items)
                                  for _ in range(n_passes)])

        a_max = min(grid_max, max(alpha * 3.0, alpha + 0.05))
        config = CSAConfig(
            alpha=alpha, delta=delta, grid_size=15,
            grid_min=grid_min, grid_max=a_max, single_epoch=True)
        controller = CSAController(config)

        cum_act, cum_fail = 0, 0
        max_risk = 0.0

        for step_idx in range(T):
            t = indices[step_idx]
            rd = stream.get_round(int(t))
            s_t = rd.score_hint if rd.score_hint is not None else 0.5
            V_t = rd.V_t

            result = controller.step(float(s_t), int(V_t))
            acted = result['acted']

            if acted:
                cum_act += 1
                if V_t == 0:
                    cum_fail += 1
                running_risk = cum_fail / max(cum_act, 1)
                if cum_act >= burn_in_accepts and running_risk > max_risk:
                    max_risk = running_risk

        final_risk = cum_fail / max(cum_act, 1)
        final_ar = cum_act / T
        results.append({
            'final_risk': float(final_risk),
            'final_ar': float(final_ar),
            'max_risk': float(max_risk),
            'n_acts': cum_act,
        })
    return results


def aggregate(reps, alpha):
    risks = np.array([r['final_risk'] for r in reps])
    ars = np.array([r['final_ar'] for r in reps])
    max_rs = np.array([r['max_risk'] for r in reps])
    pv = int(np.sum(max_rs > alpha))
    return {
        'risk_mean': float(risks.mean()),
        'risk_std': float(risks.std()),
        'ar_mean': float(ars.mean()),
        'ar_std': float(ars.std()),
        'max_risk_mean': float(max_rs.mean()),
        'pathwise_violations': pv,
        'pathwise_violation_rate': f'{pv}/{len(reps)}',
    }


def run_benchmark(bench, seeds, n_reps=10, n_passes=30, delta=0.10):
    spec = BENCHMARKS[bench]
    alpha = spec['pivotal_alpha']
    source_csv = os.path.join(POD_DIR, spec['source_csv'])

    if not os.path.exists(source_csv):
        print(f"[SKIP] {bench}: {source_csv} not found")
        return None

    print(f"\n{'='*70}")
    print(f"  {spec['label']}  alpha={alpha}  seeds={seeds}")
    print(f"{'='*70}")

    per_seed = {}
    for seed in seeds:
        t0 = time.time()
        csv_path, g_min, g_max, cal_meta = recalibrate(source_csv, seed)
        stream = load_stream(bench, csv_path)

        reps = run_csa_only(stream, alpha, delta, g_min, g_max,
                            n_reps=n_reps, n_passes=n_passes)
        agg = aggregate(reps, alpha)
        dt = time.time() - t0

        per_seed[str(seed)] = {
            'calibration': cal_meta,
            'csa_results': agg,
        }

        safe_tag = "SAFE" if agg['pathwise_violations'] == 0 else "VIOL"
        print(f"  seed={seed:>3d}  N_eval={cal_meta['n_eval']}  "
              f"Risk={agg['risk_mean']*100:5.1f}%  "
              f"AR={agg['ar_mean']*100:5.1f}%  "
              f"PathV={agg['pathwise_violation_rate']}  "
              f"[{safe_tag}]  ({dt:.0f}s)")

    risks = [v['csa_results']['risk_mean'] for v in per_seed.values()]
    ars = [v['csa_results']['ar_mean'] for v in per_seed.values()]
    pvs = [v['csa_results']['pathwise_violations'] for v in per_seed.values()]

    summary = {
        'benchmark': bench,
        'label': spec['label'],
        'alpha': alpha,
        'n_seeds': len(seeds),
        'seeds': seeds,
        'all_safe': all(p == 0 for p in pvs),
        'risk_range': [float(min(risks)), float(max(risks))],
        'ar_range': [float(min(ars)), float(max(ars))],
        'risk_cv': float(np.std(risks) / max(np.mean(risks), 1e-9)),
        'ar_cv': float(np.std(ars) / max(np.mean(ars), 1e-9)),
        'per_seed': per_seed,
    }

    print(f"\n  Summary: all_safe={summary['all_safe']}  "
          f"Risk=[{min(risks)*100:.1f}%, {max(risks)*100:.1f}%]  "
          f"AR=[{min(ars)*100:.1f}%, {max(ars)*100:.1f}%]")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=list(BENCHMARKS.keys()), default=None)
    ap.add_argument('--n_reps', type=int, default=10)
    ap.add_argument('--n_passes', type=int, default=30)
    ap.add_argument('--seeds', type=str, default='42,0,1,2,3,4,5,6,7,8,9')
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    targets = [args.only] if args.only else list(BENCHMARKS.keys())

    os.makedirs('results', exist_ok=True)
    out_path = 'results/split_sensitivity_summary.json'

    all_out = {}
    total_t0 = time.time()

    for bench in targets:
        result = run_benchmark(bench, seeds,
                               n_reps=args.n_reps, n_passes=args.n_passes)
        if result is not None:
            all_out[bench] = result
            with open(out_path, 'w') as f:
                json.dump(all_out, f, indent=2)

    print(f"\nDone in {time.time()-total_t0:.0f}s.  Results: {out_path}")

    # Clean up temp CSVs
    import shutil
    tmp_dir = 'results/_split_tmp'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"Cleaned up {tmp_dir}")


if __name__ == '__main__':
    main()
