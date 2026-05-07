"""
Generic CSA experiment runner for any domain.

Runs CSA-RLVR + principled baselines (ACI, SAOCP) + heuristic baselines
over a DomainStream. Handles multi-replication aggregation and JSON output.

Usage:
    results = run_domain_experiment(stream, config, ...)
"""

import json
import os
import time
import numpy as np
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from csa_core import CSAConfig, CSAController
from principled_baselines import ACIMethod, SAOCPMethod, LTTMethod
from domains.surrogate import OnlineSurrogate
from domains.baselines import AlwaysAct, FixedThreshold, NaiveTuning
from domains.base import DomainStream


def _run_single_method(stream, indices, method, surrogate, is_csa=False,
                       csa_config=None, burn_in_accepts=500):
    """Run a single method (CSA or baseline) over one shuffled stream pass.

    `burn_in_accepts` defines how many items must be accepted before
    `max_risk` starts being tracked. Early warmup spikes (when cum_act is
    tiny) are excluded because a single failure then produces a huge
    risk fraction purely as an arithmetic artifact.

    Returns dict with risk_curve, ar_curve, final metrics.
    """
    T = len(indices)

    if is_csa:
        controller = CSAController(csa_config)
    if surrogate is not None:
        surrogate.reset()

    history_acted = []
    history_V = []

    for step_idx in range(T):
        t = indices[step_idx]
        rd = stream.get_round(t)
        features = rd.features
        V_t = rd.V_t

        # Compute score
        if surrogate is not None:
            s_t = surrogate.score(features)
        elif rd.score_hint is not None:
            s_t = rd.score_hint
        else:
            s_t = 0.5

        if is_csa:
            result = controller.step(s_t, V_t)
            acted = result['acted']
        else:
            acted = method.decide(s_t, step_idx)
            method.update(s_t, V_t, step_idx)

        history_acted.append(int(acted))
        history_V.append(V_t)

        # Update surrogate after decision
        if surrogate is not None:
            surrogate.observe(features, V_t)

    A = np.array(history_acted, dtype=float)
    V = np.array(history_V, dtype=float)
    cum_fail = np.cumsum(A * (1 - V))
    cum_act = np.cumsum(A)
    safe_act = np.maximum(cum_act, 1)

    risk_curve = cum_fail / safe_act
    ar_curve = cum_act / np.arange(1, T + 1)

    final_risk = float(risk_curve[-1]) if T > 0 else 0.0
    final_ar = float(ar_curve[-1]) if T > 0 else 0.0

    burn_idx = int(np.searchsorted(cum_act, burn_in_accepts))
    if T > 0 and burn_idx < T:
        max_risk = float(np.max(risk_curve[burn_idx:]))
    else:
        max_risk = final_risk

    return {
        'risk_curve': risk_curve,
        'ar_curve': ar_curve,
        'final_risk': final_risk,
        'final_ar': final_ar,
        'max_risk': max_risk,
    }


def _create_baselines(alpha: float, fixed_q: float = 0.5,
                      ltt_cal_size: int = 500):
    """Create all baseline methods.

    ``ltt_cal_size`` matches the CSA burn-in window so LTT and CSA are
    compared on a level footing (both get the first 500 acts as setup).
    """
    return [
        AlwaysAct(),
        FixedThreshold(q_fixed=fixed_q),
        NaiveTuning(alpha=alpha),
        ACIMethod(alpha=alpha, gamma=0.01, q_init=0.30),
        SAOCPMethod(alpha=alpha, K=6, base_gamma=0.002, q_init=0.30),
        LTTMethod(alpha=alpha, delta=0.10, cal_size=ltt_cal_size,
                  n_thresholds=15, score_min=0.01, score_max=0.99),
    ]


def run_domain_experiment(
    stream: DomainStream,
    csa_config: CSAConfig,
    n_reps: int = 20,
    n_passes: int = 5,
    output_dir: str = 'results',
    experiment_name: str = 'domain',
    seed: int = 42,
    fixed_q: float = 0.5,
    verbose: bool = True,
    burn_in_accepts: int = 500,
    use_surrogate: bool = True,
) -> Dict:
    """Run CSA + all baselines on a domain stream.

    Parameters
    ----------
    stream : DomainStream
        The domain data stream (loaded from inference CSV).
    csa_config : CSAConfig
        CSA algorithm configuration.
    n_reps : int
        Number of replications (different shuffles).
    n_passes : int
        Number of passes through the dataset per replication.
    output_dir : str
        Directory to save results JSON.
    experiment_name : str
        Name for this experiment (used in filenames).
    seed : int
        Base random seed.
    fixed_q : float
        Threshold for FixedThreshold baseline.
    verbose : bool
        Print progress.

    Returns
    -------
    dict : JSON-serializable results with per-method curves and summary.
    """
    alpha = csa_config.alpha
    n_items = len(stream)
    T = n_items * n_passes

    # Model accuracy: fraction of items the base model answers correctly.
    # Used to compute coverage-of-correct (what fraction of answerable items
    # each method actually lets through).
    model_accuracy = float(np.mean([stream.get_round(i).V_t
                                    for i in range(n_items)]))

    method_names = ['CSA-RLVR', 'Always-Act', 'Fixed-Threshold',
                    'Naive-Tuning', 'ACI', 'SAOCP', 'LTT']

    # Accumulators
    all_results = {name: {
        'final_risks': [], 'final_ars': [], 'max_risks': [],
        'risk_curves': [], 'ar_curves': [],
    } for name in method_names}

    t0 = time.time()

    for rep in range(n_reps):
        rep_seed = seed + rep

        # Build shuffled index sequence (n_passes through the data)
        rng = np.random.RandomState(rep_seed)
        indices = np.concatenate([rng.permutation(n_items)
                                  for _ in range(n_passes)])

        # --- CSA-RLVR ---
        # Use score_hint directly when available and use_surrogate=True
        # is the default; pass use_surrogate=False to force no-surrogate
        # even if score_hint is missing (not recommended).
        test_rd = stream.get_round(0)
        needs_surrogate = use_surrogate and test_rd.score_hint is None
        if needs_surrogate:
            csa_surrogate = OnlineSurrogate(retrain_every=25, min_samples=15)
        else:
            csa_surrogate = None
        res = _run_single_method(stream, indices, None, csa_surrogate,
                                 is_csa=True, csa_config=csa_config,
                                 burn_in_accepts=burn_in_accepts)
        all_results['CSA-RLVR']['final_risks'].append(res['final_risk'])
        all_results['CSA-RLVR']['final_ars'].append(res['final_ar'])
        all_results['CSA-RLVR']['max_risks'].append(res['max_risk'])
        all_results['CSA-RLVR']['risk_curves'].append(res['risk_curve'])
        all_results['CSA-RLVR']['ar_curves'].append(res['ar_curve'])

        # --- Baselines ---
        baselines = _create_baselines(alpha, fixed_q=fixed_q,
                                      ltt_cal_size=burn_in_accepts)
        baseline_map = {
            'Always-Act': baselines[0],
            'Fixed-Threshold': baselines[1],
            'Naive-Tuning': baselines[2],
            'ACI': baselines[3],
            'SAOCP': baselines[4],
            'LTT': baselines[5],
        }

        for bname, bmethod in baseline_map.items():
            # Match CSA's choice: only instantiate a surrogate if
            # score_hint is missing from the stream CSV.
            if needs_surrogate:
                surrogate_b = OnlineSurrogate(retrain_every=25, min_samples=15)
            else:
                surrogate_b = None
            res_b = _run_single_method(stream, indices, bmethod, surrogate_b,
                                       is_csa=False,
                                       burn_in_accepts=burn_in_accepts)
            all_results[bname]['final_risks'].append(res_b['final_risk'])
            all_results[bname]['final_ars'].append(res_b['final_ar'])
            all_results[bname]['max_risks'].append(res_b['max_risk'])
            all_results[bname]['risk_curves'].append(res_b['risk_curve'])
            all_results[bname]['ar_curves'].append(res_b['ar_curve'])

        if verbose and (rep + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  Rep {rep+1}/{n_reps} done ({elapsed:.1f}s)")

    elapsed = time.time() - t0

    # --- Aggregate ---
    summary = {
        'experiment': experiment_name,
        'alpha': alpha,
        'delta': csa_config.delta,
        'n_reps': n_reps,
        'n_passes': n_passes,
        'T': T,
        'n_items': n_items,
        'model_accuracy': model_accuracy,
        'elapsed_seconds': elapsed,
        'methods': {},
    }

    for name in method_names:
        risks = np.array(all_results[name]['final_risks'])
        ars = np.array(all_results[name]['final_ars'])
        max_rs = np.array(all_results[name]['max_risks'])

        pathwise_violations = int(np.sum(max_rs > alpha))

        # Average curves
        risk_curves = np.array(all_results[name]['risk_curves'])
        ar_curves = np.array(all_results[name]['ar_curves'])
        mean_risk_curve = risk_curves.mean(axis=0)
        mean_ar_curve = ar_curves.mean(axis=0)

        # Precision on accepted items, and coverage of all correctly-
        # answerable items. Defined per rep then averaged so stds are honest.
        precisions = 1.0 - risks  # per-rep precision
        if model_accuracy > 0:
            coverages = ars * precisions / model_accuracy
        else:
            coverages = np.zeros_like(ars)

        summary['methods'][name] = {
            'final_risk_mean': float(risks.mean()),
            'final_risk_std': float(risks.std()),
            'final_ar_mean': float(ars.mean()),
            'final_ar_std': float(ars.std()),
            'max_risk_mean': float(max_rs.mean()),
            'precision_mean': float(precisions.mean()),
            'precision_std': float(precisions.std()),
            'coverage_correct_mean': float(coverages.mean()),
            'coverage_correct_std': float(coverages.std()),
            'pathwise_violations': pathwise_violations,
            'pathwise_violation_rate': f"{pathwise_violations}/{n_reps}",
            'mean_risk_curve': mean_risk_curve.tolist(),
            'mean_ar_curve': mean_ar_curve.tolist(),
        }

    # --- Print summary table ---
    if verbose:
        print(f"\n{'='*96}")
        print(f"  {experiment_name}  (alpha={alpha}, T={T}, {n_reps} reps, "
              f"model acc={model_accuracy:.1%})")
        print(f"{'='*96}")
        print(f"  {'Method':<18} {'Risk':>10} {'AR':>8} {'MaxR':>8} "
              f"{'Prec':>8} {'Cov':>8} {'PathV':>10}")
        print(f"  {'-'*80}")
        for name in method_names:
            m = summary['methods'][name]
            risk_str = f"{m['final_risk_mean']:.1%}+/-{m['final_risk_std']:.1%}"
            ar_str = f"{m['final_ar_mean']:.1%}"
            maxr_str = f"{m['max_risk_mean']:.1%}"
            prec_str = f"{m['precision_mean']:.1%}"
            cov_str = f"{m['coverage_correct_mean']:.1%}"
            pv_str = m['pathwise_violation_rate']
            marker = " *" if m['final_risk_mean'] > alpha else ""
            print(f"  {name:<18} {risk_str:>10} {ar_str:>8} {maxr_str:>8} "
                  f"{prec_str:>8} {cov_str:>8} {pv_str:>10}{marker}")
        print(f"{'='*96}\n")

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{experiment_name}.json")

    # Convert numpy arrays for JSON serialization
    save_summary = json.loads(json.dumps(summary, default=lambda x:
        x.tolist() if isinstance(x, np.ndarray) else float(x)
        if isinstance(x, (np.float32, np.float64)) else x))

    with open(out_path, 'w') as f:
        json.dump(save_summary, f, indent=2)

    if verbose:
        print(f"  Results saved to {out_path}")

    return summary
