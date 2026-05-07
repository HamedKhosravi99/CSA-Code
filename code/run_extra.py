"""
Orchestrator for the *extra* CSA domain experiments.

Runs CSA replay on five additional benchmarks that complement the core
four (medical, legal, financial, agents):
    - gsm8k    : grade-school math
    - math500  : competition math
    - arc      : ARC-Challenge science MCQ
    - mmlu_pro : broad MCQ across 14 subjects
    - pubmedqa : biomedical yes/no/maybe

Keeps `run_domains.py` untouched so you can run the core 4 first and
only invoke this when there's GPU time left for the extras.

Usage:
    python run_extra.py --results_dir results/
    python run_extra.py --results_dir results/ --domains gsm8k arc
    python run_extra.py --results_dir results/ --alphas 0.10 0.20 0.30

Each benchmark's runner looks for:
    results/<name>_inference_calibrated.csv  (preferred; eval split)
    results/<name>_inference_sc.csv          (self-consistency scored)
    results/<name>_inference.csv             (raw inference)
and uses the first that exists. Grid bounds come from
`results/<name>_inference_calibrated_meta.json` when present.
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
    """Return the best-available CSV for a domain (calibrated > sc > raw)."""
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
    """Read grid_min/grid_max from a paired _meta.json if present."""
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
    stem_label = stem.upper()
    using = os.path.basename(csv_path)
    print(f"{stem_label}: using {using}, grid=[{grid_min:.4f},{grid_max:.4f}]")

    stream = stream_cls(csv_path)
    print(f"{stem_label}: {len(stream)} items loaded")

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


def run_gsm8k(results_dir, n_reps, n_passes, seed, **kw):
    from domains.gsm8k.stream import GSM8KStream
    return _run_domain('gsm8k', GSM8KStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_math500(results_dir, n_reps, n_passes, seed, **kw):
    from domains.math500.stream import MATH500Stream
    return _run_domain('math500', MATH500Stream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_arc(results_dir, n_reps, n_passes, seed, **kw):
    from domains.arc.stream import ARCStream
    return _run_domain('arc', ARCStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_mmlu_pro(results_dir, n_reps, n_passes, seed, **kw):
    from domains.mmlu_pro.stream import MMLUProStream
    return _run_domain('mmlu_pro', MMLUProStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_pubmedqa(results_dir, n_reps, n_passes, seed, **kw):
    from domains.pubmedqa.stream import PubMedQAStream
    return _run_domain('pubmedqa', PubMedQAStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_headqa(results_dir, n_reps, n_passes, seed, **kw):
    from domains.headqa.stream import HEADQAStream
    return _run_domain('headqa', HEADQAStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_boolq(results_dir, n_reps, n_passes, seed, **kw):
    from domains.boolq.stream import BoolQStream
    return _run_domain('boolq', BoolQStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_mmlu_pro_bio(results_dir, n_reps, n_passes, seed, **kw):
    from domains.mmlu_pro.stream import MMLUProStream
    return _run_domain('mmlu_pro_bio', MMLUProStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_mmlu_pro_health(results_dir, n_reps, n_passes, seed, **kw):
    from domains.mmlu_pro.stream import MMLUProStream
    return _run_domain('mmlu_pro_health', MMLUProStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_mmlu_pro_law(results_dir, n_reps, n_passes, seed, **kw):
    from domains.mmlu_pro.stream import MMLUProStream
    return _run_domain('mmlu_pro_law', MMLUProStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_csqa(results_dir, n_reps, n_passes, seed, **kw):
    from domains.csqa.stream import CSQAStream
    return _run_domain('csqa', CSQAStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_hellaswag(results_dir, n_reps, n_passes, seed, **kw):
    from domains.hellaswag.stream import HellaSwagStream
    return _run_domain('hellaswag', HellaSwagStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_medcalc(results_dir, n_reps, n_passes, seed, **kw):
    from domains.medcalc.stream import MedCalcStream
    return _run_domain('medcalc', MedCalcStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


def run_winogrande(results_dir, n_reps, n_passes, seed, **kw):
    from domains.winogrande.stream import WinograndeStream
    return _run_domain('winogrande', WinograndeStream, results_dir,
                       n_reps=n_reps, n_passes=n_passes, seed=seed, **kw)


DOMAIN_RUNNERS = {
    'gsm8k':    run_gsm8k,
    'math500':  run_math500,
    'arc':      run_arc,
    'mmlu_pro': run_mmlu_pro,
    'mmlu_pro_bio':    run_mmlu_pro_bio,
    'mmlu_pro_health': run_mmlu_pro_health,
    'mmlu_pro_law':    run_mmlu_pro_law,
    'pubmedqa': run_pubmedqa,
    'headqa':   run_headqa,
    'boolq':    run_boolq,
    'csqa':     run_csqa,
    'hellaswag': run_hellaswag,
    'medcalc':   run_medcalc,
    'winogrande': run_winogrande,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run CSA replay on extra benchmarks (gsm8k, math500, "
                    "arc, mmlu_pro, pubmedqa).")
    parser.add_argument('--results_dir', default='results',
                        help='Directory containing inference CSVs')
    parser.add_argument('--n_reps', type=int, default=5)
    parser.add_argument('--n_passes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--burn_in_accepts', type=int, default=500,
                        help='Burn-in for PathV (ignore early warmup spikes).')
    parser.add_argument('--alphas', nargs='+', type=float,
                        default=[0.10, 0.15, 0.20, 0.30],
                        help='Alpha values for the frontier plot.')
    parser.add_argument('--delta', type=float, default=0.10,
                        help='Miscoverage budget for CSA.')
    parser.add_argument('--domains', nargs='+',
                        default=list(DOMAIN_RUNNERS.keys()),
                        help='Which extra domains to run.')
    args = parser.parse_args()

    t0 = time.time()
    all_results = {}

    for dom in args.domains:
        if dom not in DOMAIN_RUNNERS:
            print(f"Unknown extra domain: {dom} (available: "
                  f"{sorted(DOMAIN_RUNNERS)})")
            continue
        print(f"\n{'#'*80}")
        print(f"  EXTRA DOMAIN: {dom.upper()}")
        print(f"{'#'*80}\n")
        all_results[dom] = DOMAIN_RUNNERS[dom](
            args.results_dir, args.n_reps, args.n_passes, args.seed,
            alphas=args.alphas, delta=args.delta,
            burn_in_accepts=args.burn_in_accepts)

    # Save combined summary (stripped of curves for JSON size)
    summary_path = os.path.join(args.results_dir, 'extra_summary.json')
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

    print(f"\nExtra summary saved to {summary_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
