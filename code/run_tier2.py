"""
Orchestrator for the *Tier-2 high-stakes* CSA domain experiments.

Runs CSA replay on five high-stakes benchmarks that deepen the paper's
"errors carry real-world consequences" story:
    - casehold   : legal case-holding selection (5-option MCQ)
    - medmcqa    : Indian medical exam (4-option MCQ)
    - tatqa      : hybrid table+text financial QA
    - cybermetric: cybersecurity MCQ
    - ddxplus    : clinical differential diagnosis (4-option MCQ)

Keeps run_domains.py (core 4) and run_extra.py (general-capability 5)
untouched. Run this when you want the full stakes-depth table.

Usage:
    python run_tier2.py --results_dir results/
    python run_tier2.py --results_dir results/ --domains casehold medmcqa
    python run_tier2.py --results_dir results/ --alphas 0.10 0.20 0.30

Each benchmark's runner looks for:
    results/<name>_inference_calibrated.csv  (preferred)
    results/<name>_inference_sc.csv          (self-consistency scored)
    results/<name>_inference.csv             (raw)
Grid bounds come from `results/<name>_inference_calibrated_meta.json`.
"""

import argparse
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from csa_core import CSAConfig
from domains.runner import run_domain_experiment


def _resolve_csv(results_dir: str, stem: str):
    candidates = [
        os.path.join(results_dir, f"{stem}_inference_calibrated.csv"),
        os.path.join(results_dir, f"{stem}_inference_sc.csv"),
        os.path.join(results_dir, f"{stem}_inference.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _load_grid(csv_path: str, default_min=0.005, default_max=0.80):
    meta_path = os.path.splitext(csv_path)[0] + '_meta.json'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return float(meta.get('grid_min', default_min)), \
               float(meta.get('grid_max', default_max))
    return default_min, default_max


def _run_domain(stem: str, stream_cls, results_dir: str, *,
                n_reps: int, n_passes: int, seed: int,
                alphas, delta: float, burn_in_accepts: int):
    csv_path = _resolve_csv(results_dir, stem)
    if csv_path is None:
        print(f"[SKIP] {stem}: no CSV found in {results_dir}")
        return None

    grid_min, grid_max = _load_grid(csv_path)
    label = stem.upper()
    print(f"{label}: using {os.path.basename(csv_path)}, "
          f"grid=[{grid_min:.4f},{grid_max:.4f}]")

    stream = stream_cls(csv_path)
    print(f"{label}: {len(stream)} items loaded")

    out_dir = os.path.join(results_dir, stem)
    results = {}
    for alpha in alphas:
        a_max = min(grid_max, max(alpha * 3.0, alpha + 0.05))
        config = CSAConfig(
            alpha=alpha, delta=delta, grid_size=15,
            grid_min=grid_min, grid_max=a_max, single_epoch=True)
        r = run_domain_experiment(
            stream=stream, csa_config=config,
            n_reps=n_reps, n_passes=n_passes,
            output_dir=out_dir,
            experiment_name=f"{stem}_alpha{alpha:.2f}",
            seed=seed,
            burn_in_accepts=burn_in_accepts)
        results[f"alpha_{alpha:.2f}"] = r
    return results


def run_casehold(results_dir, n_reps, n_passes, seed, **kw):
    from domains.casehold.stream import CaseHOLDStream
    return _run_domain('casehold', CaseHOLDStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_medmcqa(results_dir, n_reps, n_passes, seed, **kw):
    from domains.medmcqa.stream import MedMCQAStream
    return _run_domain('medmcqa', MedMCQAStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_tatqa(results_dir, n_reps, n_passes, seed, **kw):
    from domains.tatqa.stream import TATQAStream
    return _run_domain('tatqa', TATQAStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_cybermetric(results_dir, n_reps, n_passes, seed, **kw):
    from domains.cybermetric.stream import CyberMetricStream
    return _run_domain('cybermetric', CyberMetricStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_ddxplus(results_dir, n_reps, n_passes, seed, **kw):
    from domains.ddxplus.stream import DDXPlusStream
    return _run_domain('ddxplus', DDXPlusStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_lsat_lr(results_dir, n_reps, n_passes, seed, **kw):
    from domains.lsat_lr.stream import LSATLRStream
    return _run_domain('lsat_lr', LSATLRStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_cve(results_dir, n_reps, n_passes, seed, **kw):
    from domains.cve.stream import CVEStream
    return _run_domain('cve', CVEStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_humaneval(results_dir, n_reps, n_passes, seed, **kw):
    from domains.humaneval.stream import HumanEvalStream
    return _run_domain('humaneval', HumanEvalStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_mmlu_profmed(results_dir, n_reps, n_passes, seed, **kw):
    from domains.mmlu_profmed.stream import MMLUProfMedStream
    return _run_domain('mmlu_profmed', MMLUProfMedStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_fomc(results_dir, n_reps, n_passes, seed, **kw):
    from domains.fomc.stream import FOMCStream
    return _run_domain('fomc', FOMCStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_fpb(results_dir, n_reps, n_passes, seed, **kw):
    from domains.fpb.stream import FPBStream
    return _run_domain('fpb', FPBStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_mednli(results_dir, n_reps, n_passes, seed, **kw):
    from domains.mednli.stream import MedNLIStream
    return _run_domain('mednli', MedNLIStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


DOMAIN_RUNNERS = {
    'casehold':     run_casehold,
    'medmcqa':      run_medmcqa,
    'tatqa':        run_tatqa,
    'cybermetric':  run_cybermetric,
    'ddxplus':      run_ddxplus,
    'lsat_lr':      run_lsat_lr,
    'cve':          run_cve,
    'humaneval':    run_humaneval,
    'mmlu_profmed': run_mmlu_profmed,
    'fomc':         run_fomc,
    'fpb':          run_fpb,
    'mednli':       run_mednli,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run CSA replay on Tier-2 high-stakes benchmarks "
                    "(casehold, medmcqa, tatqa, cybermetric, ddxplus).")
    parser.add_argument('--results_dir', default='results',
                        help='Directory containing inference CSVs')
    parser.add_argument('--n_reps', type=int, default=5)
    parser.add_argument('--n_passes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--burn_in_accepts', type=int, default=500,
                        help='Burn-in for PathV (ignore early warmup spikes).')
    parser.add_argument('--alphas', nargs='+', type=float,
                        default=[0.10, 0.15, 0.20, 0.30])
    parser.add_argument('--delta', type=float, default=0.10)
    parser.add_argument('--domains', nargs='+',
                        default=list(DOMAIN_RUNNERS.keys()))
    args = parser.parse_args()

    t0 = time.time()
    all_results = {}

    for dom in args.domains:
        if dom not in DOMAIN_RUNNERS:
            print(f"Unknown tier-2 domain: {dom} (available: "
                  f"{sorted(DOMAIN_RUNNERS)})")
            continue
        print(f"\n{'#'*80}")
        print(f"  TIER-2 DOMAIN: {dom.upper()}")
        print(f"{'#'*80}\n")
        all_results[dom] = DOMAIN_RUNNERS[dom](
            args.results_dir, args.n_reps, args.n_passes, args.seed,
            alphas=args.alphas, delta=args.delta,
            burn_in_accepts=args.burn_in_accepts)

    summary_path = os.path.join(args.results_dir, 'tier2_summary.json')
    combined = {}
    for dom, results in all_results.items():
        if results is None:
            continue
        combined[dom] = {}
        for alpha_key, r in results.items():
            methods_summary = {}
            for mname, mdata in r.get('methods', {}).items():
                methods_summary[mname] = {
                    k: v for k, v in mdata.items()
                    if k not in ('mean_risk_curve', 'mean_ar_curve')
                }
            combined[dom][alpha_key] = {
                'experiment': r.get('experiment', ''),
                'n_items': r.get('n_items', 0),
                'T': r.get('T', 0),
                'model_accuracy': r.get('model_accuracy', 0.0),
                'methods': methods_summary,
            }
    os.makedirs(args.results_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"\nTier-2 summary saved to {summary_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
