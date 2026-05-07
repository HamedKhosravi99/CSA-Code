"""
Hyperparameter sensitivity ablation for CSA-RLVR.

CSA has four free hyperparameters (delta, K, burn-in, |T|); this script
sweeps each one-at-a-time around the defaults to quantify the effect on
selective risk and action rate.

Ranges swept (MedQA, alpha=0.20, 10 reps):
    * delta       in {0.05, 0.10, 0.20}    -- anytime-valid confidence
    * burn_in     in {100, 500, 1000}      -- transient-spike exclusion
    * grid_size   in {5, 15, 30}           -- number of thresholds

Each run uses the default value for the non-swept parameters:
    delta=0.10, burn_in=500, grid_size=15

Output: results/ablation_hyperparams.json plus a latex-ready summary
printed to stdout.

Usage:
    python ablate_hyperparams.py
"""

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
from csa_core import CSAConfig
from domains.medical.stream import MedQAStream
from domains.runner import run_domain_experiment


CSV_PATH = 'results/medical_inference_calibrated.csv'
META_PATH = 'results/medical_inference_calibrated_meta.json'
OUT_PATH  = 'results/ablation_hyperparams.json'

ALPHA   = 0.20
N_REPS  = 10
N_PASSES = 30
SEED    = 42

# Default values
DEF_DELTA   = 0.10
DEF_BURN    = 500
DEF_GRID    = 15

# Sweeps
DELTA_SWEEP = [0.05, 0.10, 0.20]
BURN_SWEEP  = [100, 500, 1000]
GRID_SWEEP  = [5, 15, 30]


def load_grid_bounds():
    with open(META_PATH) as f:
        meta = json.load(f)
    return float(meta['grid_min']), float(meta['grid_max'])


def run_one(delta, burn_in, grid_size, gmin, gmax, tag):
    a_max = min(gmax, max(ALPHA * 3.0, ALPHA + 0.05))
    cfg = CSAConfig(
        alpha=ALPHA, delta=delta, grid_size=grid_size,
        grid_min=gmin, grid_max=a_max, single_epoch=True)
    stream = MedQAStream(CSV_PATH)
    t0 = time.time()
    r = run_domain_experiment(
        stream=stream, csa_config=cfg,
        n_reps=N_REPS, n_passes=N_PASSES,
        output_dir='results/ablation_hparam',
        experiment_name=f'ablate_{tag}',
        seed=SEED, burn_in_accepts=burn_in)
    dt = time.time() - t0

    m = r['methods']['CSA-RLVR']
    return {
        'tag': tag,
        'delta': delta,
        'burn_in_accepts': burn_in,
        'grid_size': grid_size,
        'csa_final_risk_mean': float(m['final_risk_mean']),
        'csa_final_risk_std':  float(m['final_risk_std']),
        'csa_final_ar_mean':   float(m['final_ar_mean']),
        'csa_final_ar_std':    float(m['final_ar_std']),
        'csa_max_risk_mean':   float(m['max_risk_mean']),
        'csa_precision_mean':  float(m['precision_mean']),
        'csa_coverage_correct_mean': float(m['coverage_correct_mean']),
        'csa_pathwise_violation_rate': str(m['pathwise_violation_rate']),
        'elapsed_seconds': dt,
    }


def main():
    gmin, gmax = load_grid_bounds()
    print(f"Grid bounds from CAL: [{gmin:.4f}, {gmax:.4f}]")
    print(f"Sweep target: MedQA, alpha={ALPHA}, n_reps={N_REPS}, n_passes={N_PASSES}")

    runs = []
    t_all0 = time.time()

    # Baseline (skip; already matches the existing MedQA 10-rep alpha=0.20 result:
    # CSA AR=39.4%, Risk=11.4%, PathV=0/10)
    runs.append({
        'tag': 'baseline',
        'delta': DEF_DELTA, 'burn_in_accepts': DEF_BURN, 'grid_size': DEF_GRID,
        'csa_final_risk_mean': 0.1144,
        'csa_final_risk_std':  0.0003,
        'csa_final_ar_mean':   0.3944,
        'csa_final_ar_std':    0.0033,
        'csa_max_risk_mean':   0.1241,
        'csa_precision_mean':  0.8856,
        'csa_coverage_correct_mean': 0.5101,
        'csa_pathwise_violation_rate': '0/10',
        'elapsed_seconds': 444.3,
        'note': 'from main MedQA run (results/medical/medical_alpha0.20.json)',
    })

    # Sweep delta (fix burn_in=500, grid_size=15)
    for d in DELTA_SWEEP:
        if d == DEF_DELTA:
            continue
        print(f"\n=== delta={d} ===")
        runs.append(run_one(d, DEF_BURN, DEF_GRID, gmin, gmax, f'delta_{d}'))

    # Sweep burn-in (fix delta=0.10, grid_size=15)
    for b in BURN_SWEEP:
        if b == DEF_BURN:
            continue
        print(f"\n=== burn_in={b} ===")
        runs.append(run_one(DEF_DELTA, b, DEF_GRID, gmin, gmax, f'burn_{b}'))

    # Sweep grid_size (fix delta=0.10, burn_in=500)
    for g in GRID_SWEEP:
        if g == DEF_GRID:
            continue
        print(f"\n=== grid_size={g} ===")
        runs.append(run_one(DEF_DELTA, DEF_BURN, g, gmin, gmax, f'grid_{g}'))

    elapsed = time.time() - t_all0

    out = {
        'benchmark': 'medical (MedQA)',
        'alpha': ALPHA,
        'n_reps': N_REPS,
        'n_passes': N_PASSES,
        'seed': SEED,
        'defaults': {'delta': DEF_DELTA, 'burn_in': DEF_BURN, 'grid_size': DEF_GRID},
        'runs': runs,
        'total_elapsed_seconds': elapsed,
    }
    os.makedirs(os.path.dirname(OUT_PATH) or '.', exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    # Pretty print
    print("\n" + "="*90)
    print(f"  Sensitivity ablation on MedQA @ alpha={ALPHA}, 10 reps")
    print("="*90)
    hdr = f"{'Variant':<20} {'delta':<8} {'burn':<8} {'|T|':<6} {'Risk':<10} {'AR':<10} {'MaxR':<8} {'PathV':<8}"
    print(hdr)
    print('-' * len(hdr))
    for r in runs:
        print(f"{r['tag']:<20} "
              f"{r['delta']:<8} {r['burn_in_accepts']:<8} {r['grid_size']:<6} "
              f"{r['csa_final_risk_mean']*100:<9.2f}% "
              f"{r['csa_final_ar_mean']*100:<9.2f}% "
              f"{r['csa_max_risk_mean']*100:<7.2f}% "
              f"{r['csa_pathwise_violation_rate']}")

    print(f"\nSaved: {OUT_PATH}")
    print(f"Total wall time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
