"""
Master orchestrator for all CSA domain experiments.

Runs CSA replay for all 4 domains from pre-computed inference CSVs.
All replay is CPU-only (numpy/sklearn), runs on any machine.

Usage:
    python run_domains.py --results_dir results/

Expects inference CSVs at:
    results/medical_inference.csv
    results/financial_inference.csv
    results/legal_inference.csv
    results/agents_inference.csv
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import json

from csa_core import CSAConfig
from domains.runner import run_domain_experiment


def run_medical(results_dir: str, n_reps: int, n_passes: int, seed: int,
                burn_in_accepts: int = 500,
                alphas=None, delta: float = 0.10):
    """Run medical domain experiments.

    Prefers `medical_inference_calibrated.csv` (output of calibrate_scores.py)
    which contains only the EVAL split and a `calibrated_score` column.
    Grid range is read from the paired _meta.json (derived on the CAL split),
    so CSA's hyperparameters are chosen on data disjoint from the eval stream.
    """
    calibrated = os.path.join(results_dir, 'medical_inference_calibrated.csv')
    scored_8bit = os.path.join(results_dir, 'medical_inference_scored_8bit.csv')
    scored_path = os.path.join(results_dir, 'medical_inference_scored.csv')
    base_path = os.path.join(results_dir, 'medical_inference.csv')
    for p in [calibrated, scored_8bit, scored_path, base_path]:
        if os.path.exists(p):
            csv_path = p
            break
    else:
        csv_path = base_path
    if not os.path.exists(csv_path):
        print(f"[SKIP] Medical: {csv_path} not found")
        return None

    # Load CAL-derived grid range if available.
    grid_min, grid_max = 0.005, 0.50
    meta_path = os.path.splitext(csv_path)[0] + '_meta.json'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        grid_min = float(meta.get('grid_min', grid_min))
        grid_max = float(meta.get('grid_max', grid_max))
        print(f"Medical: using calibrated CSV, grid=[{grid_min:.4f},{grid_max:.4f}]")
    else:
        print(f"Medical: using {os.path.basename(csv_path)} (no calibration)")

    from domains.medical.stream import MedQAStream
    stream = MedQAStream(csv_path)
    print(f"Medical: {len(stream)} items loaded")

    if alphas is None:
        alphas = [0.10, 0.15, 0.20, 0.30]

    results = {}
    for alpha in alphas:
        # Grid bounds come from CAL split; tighten the upper end to at
        # most ~3x alpha so we don't waste grid points on useless regions.
        a_min = grid_min
        a_max = min(grid_max, max(alpha * 3.0, alpha + 0.05))
        config = CSAConfig(
            alpha=alpha, delta=delta, grid_size=15,
            grid_min=a_min, grid_max=a_max, single_epoch=True)
        r = run_domain_experiment(
            stream=stream, csa_config=config,
            n_reps=n_reps, n_passes=30,
            output_dir=os.path.join(results_dir, 'medical'),
            experiment_name=f"medical_alpha{alpha:.2f}",
            seed=seed,
            burn_in_accepts=burn_in_accepts)
        results[f"alpha_{alpha:.2f}"] = r

    return results


def run_financial(results_dir: str, n_reps: int, n_passes: int, seed: int,
                  burn_in_accepts: int = 500,
                  alphas=None, delta: float = 0.10):
    """Run financial domain experiments.

    Prefers `financial_inference_calibrated.csv` if present (from
    calibrate_scores.py). Grid range comes from the paired _meta.json.
    """
    calibrated = os.path.join(results_dir, 'financial_inference_calibrated.csv')
    base_path = os.path.join(results_dir, 'financial_inference.csv')
    csv_path = calibrated if os.path.exists(calibrated) else base_path
    if not os.path.exists(csv_path):
        print(f"[SKIP] Financial: {csv_path} not found")
        return None

    grid_min, grid_max = 0.02, 0.98
    meta_path = os.path.splitext(csv_path)[0] + '_meta.json'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        grid_min = float(meta.get('grid_min', grid_min))
        grid_max = float(meta.get('grid_max', grid_max))
        print(f"Financial: using calibrated CSV, grid=[{grid_min:.4f},{grid_max:.4f}]")
    else:
        print(f"Financial: using {os.path.basename(csv_path)} (no calibration)")

    from domains.financial.stream import FinQAStream
    stream = FinQAStream(csv_path)
    print(f"Financial: {len(stream)} items loaded")

    if alphas is None:
        alphas = [0.10, 0.15, 0.20, 0.30]

    results = {}
    for alpha in alphas:
        a_max = min(grid_max, max(alpha * 3.0, alpha + 0.05))
        config = CSAConfig(
            alpha=alpha, delta=delta, grid_size=15,
            grid_min=grid_min, grid_max=a_max, single_epoch=True)
        r = run_domain_experiment(
            stream=stream, csa_config=config,
            n_reps=n_reps, n_passes=n_passes,
            output_dir=os.path.join(results_dir, 'financial'),
            experiment_name=f"financial_alpha{alpha:.2f}",
            seed=seed,
            burn_in_accepts=burn_in_accepts)
        results[f"alpha_{alpha:.2f}"] = r

    return results


def run_legal(results_dir: str, n_reps: int, n_passes: int, seed: int,
              burn_in_accepts: int = 500,
              alphas=None, delta: float = 0.10):
    """Run legal domain experiments.

    Prefers `legal_inference_calibrated.csv` if present.
    """
    calibrated = os.path.join(results_dir, 'legal_inference_calibrated.csv')
    base_path = os.path.join(results_dir, 'legal_inference.csv')
    csv_path = calibrated if os.path.exists(calibrated) else base_path
    if not os.path.exists(csv_path):
        print(f"[SKIP] Legal: {csv_path} not found")
        return None

    grid_min, grid_max = 0.02, 0.98
    meta_path = os.path.splitext(csv_path)[0] + '_meta.json'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        grid_min = float(meta.get('grid_min', grid_min))
        grid_max = float(meta.get('grid_max', grid_max))
        print(f"Legal: using calibrated CSV, grid=[{grid_min:.4f},{grid_max:.4f}]")
    else:
        print(f"Legal: using {os.path.basename(csv_path)} (no calibration)")

    from domains.legal.stream import LegalBenchStream
    stream = LegalBenchStream(csv_path)
    print(f"Legal: {len(stream)} items loaded")

    if alphas is None:
        alphas = [0.10, 0.15, 0.20, 0.30]

    results = {}
    for alpha in alphas:
        a_max = min(grid_max, max(alpha * 3.0, alpha + 0.05))
        config = CSAConfig(
            alpha=alpha, delta=delta, grid_size=15,
            grid_min=grid_min, grid_max=a_max, single_epoch=True)
        r = run_domain_experiment(
            stream=stream, csa_config=config,
            n_reps=n_reps, n_passes=n_passes,
            output_dir=os.path.join(results_dir, 'legal'),
            experiment_name=f"legal_alpha{alpha:.2f}",
            seed=seed,
            burn_in_accepts=burn_in_accepts)
        results[f"alpha_{alpha:.2f}"] = r

    return results


def run_agents(results_dir: str, n_reps: int, n_passes: int, seed: int,
               burn_in_accepts: int = 500,
               alphas=None, delta: float = 0.10):
    """Run agents domain experiments.

    Prefers `agents_inference_calibrated.csv` if present.
    """
    calibrated = os.path.join(results_dir, 'agents_inference_calibrated.csv')
    base_path = os.path.join(results_dir, 'agents_inference.csv')
    csv_path = calibrated if os.path.exists(calibrated) else base_path
    if not os.path.exists(csv_path):
        print(f"[SKIP] Agents: {csv_path} not found")
        return None

    grid_min, grid_max = 0.02, 0.98
    meta_path = os.path.splitext(csv_path)[0] + '_meta.json'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        grid_min = float(meta.get('grid_min', grid_min))
        grid_max = float(meta.get('grid_max', grid_max))
        print(f"Agents: using calibrated CSV, grid=[{grid_min:.4f},{grid_max:.4f}]")
    else:
        print(f"Agents: using {os.path.basename(csv_path)} (no calibration)")

    from domains.agents.stream import ALFWorldStream
    stream = ALFWorldStream(csv_path)
    print(f"Agents: {len(stream)} items loaded")

    if alphas is None:
        alphas = [0.10, 0.15, 0.20, 0.30]

    results = {}
    for alpha in alphas:
        a_max = min(grid_max, max(alpha * 3.0, alpha + 0.05))
        config = CSAConfig(
            alpha=alpha, delta=delta, grid_size=15,
            grid_min=grid_min, grid_max=a_max, single_epoch=True)
        r = run_domain_experiment(
            stream=stream, csa_config=config,
            n_reps=n_reps, n_passes=n_passes,
            output_dir=os.path.join(results_dir, 'agents'),
            experiment_name=f"agents_alpha{alpha:.2f}",
            seed=seed,
            burn_in_accepts=burn_in_accepts)
        results[f"alpha_{alpha:.2f}"] = r

    return results


def print_cross_domain_table(all_results: dict, alpha: float = 0.10):
    """Print a cross-domain comparison table for a given alpha."""
    print(f"\n{'='*110}")
    print(f"  CROSS-DOMAIN COMPARISON (alpha={alpha})")
    print(f"{'='*110}")
    print(f"  {'Domain':<15} {'Method':<18} {'Risk':>12} {'AR':>10} "
          f"{'Prec':>8} {'Cov':>8} {'Violations':>12}")
    print(f"  {'-'*85}")

    for domain_name, domain_results in all_results.items():
        key = f"alpha_{alpha:.2f}"
        if domain_results is None or key not in domain_results:
            continue

        summary = domain_results[key]
        methods = summary.get('methods', {})

        for method_name in ['CSA-RLVR', 'ACI', 'SAOCP', 'Always-Act']:
            if method_name not in methods:
                continue
            m = methods[method_name]
            risk = f"{m['final_risk_mean']:.1%}+/-{m['final_risk_std']:.1%}"
            ar = f"{m['final_ar_mean']:.1%}"
            prec = f"{m.get('precision_mean', 1 - m['final_risk_mean']):.1%}"
            cov = f"{m.get('coverage_correct_mean', 0.0):.1%}"
            pv = m['pathwise_violation_rate']
            marker = " **" if m['final_risk_mean'] > alpha else ""
            print(f"  {domain_name:<15} {method_name:<18} {risk:>12} "
                  f"{ar:>10} {prec:>8} {cov:>8} {pv:>12}{marker}")
        print(f"  {'-'*85}")

    print(f"{'='*110}")
    print("  Prec = precision on accepted (1-risk); "
          "Cov = fraction of correct items retained;  ** = exceeds target alpha\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run all CSA domain replay experiments")
    parser.add_argument('--results_dir', default='results',
                        help='Directory containing inference CSVs')
    parser.add_argument('--n_reps', type=int, default=5)
    parser.add_argument('--n_passes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--burn_in_accepts', type=int, default=500,
                        help='Ignore early-warmup spikes for PathV: '
                             'only count risk excursions after this many '
                             'items have been accepted.')
    parser.add_argument('--alphas', nargs='+', type=float, default=None,
                        help='Alpha values to run for the frontier plot. '
                             'Default: [0.10, 0.15, 0.20, 0.30]')
    parser.add_argument('--delta', type=float, default=0.10,
                        help='Miscoverage budget for CSA.')
    parser.add_argument('--domains', nargs='+',
                        default=['medical', 'financial', 'legal', 'agents'],
                        help='Which domains to run')
    args = parser.parse_args()

    t0 = time.time()
    all_results = {}

    domain_runners = {
        'medical': run_medical,
        'financial': run_financial,
        'legal': run_legal,
        'agents': run_agents,
    }

    for domain in args.domains:
        if domain not in domain_runners:
            print(f"Unknown domain: {domain}")
            continue
        print(f"\n{'#'*80}")
        print(f"  DOMAIN: {domain.upper()}")
        print(f"{'#'*80}\n")
        kwargs = {
            'burn_in_accepts': args.burn_in_accepts,
            'delta': args.delta,
        }
        if args.alphas is not None:
            kwargs['alphas'] = args.alphas
        result = domain_runners[domain](
            args.results_dir, args.n_reps, args.n_passes, args.seed, **kwargs)
        all_results[domain] = result

    elapsed = time.time() - t0

    # Cross-domain tables
    for alpha in [0.10, 0.20]:
        print_cross_domain_table(all_results, alpha)

    # Save combined summary
    summary_path = os.path.join(args.results_dir, 'cross_domain_summary.json')
    combined = {}
    for domain, results in all_results.items():
        if results is None:
            continue
        combined[domain] = {}
        for alpha_key, r in results.items():
            # Extract just the summary stats (no curves, for JSON size)
            methods_summary = {}
            for mname, mdata in r.get('methods', {}).items():
                methods_summary[mname] = {
                    k: v for k, v in mdata.items()
                    if k not in ('mean_risk_curve', 'mean_ar_curve')
                }
            combined[domain][alpha_key] = {
                'experiment': r.get('experiment', ''),
                'n_items': r.get('n_items', 0),
                'T': r.get('T', 0),
                'methods': methods_summary,
            }

    os.makedirs(args.results_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nCross-domain summary saved to {summary_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
