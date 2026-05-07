"""
Metrics computation for CSA on CAP-style benchmarks.
"""

import numpy as np
from typing import Dict


def compute_all_metrics(
    actions: np.ndarray,
    verifier_results: np.ndarray,
    scores: np.ndarray,
    q_deploy_history: np.ndarray,
    alpha: float = 0.10,
    cap_selections: np.ndarray = None,
) -> Dict:
    """
    Compute all metrics for one replication.

    Our method metrics (from our paper):
    - selective_risk: R_T^act over time
    - action_rate: N_T / T over time
    - cum_actions: N_T over time
    - cum_failures: sum of A_t*(1-V_t) over time

    CAP-comparable metrics:
    - fcr_style: identical to selective_risk (our R_T^act = CAP's FCP)
    - average_score_released: mean score among released items
      (analogue of PI length: lower score = more confident = "shorter interval")

    NOT directly comparable to CAP:
    - utility_gap: requires oracle q_t^* (only computable in fully synthetic settings)
    - certification_delay: requires knowing when each threshold was certified
    - average_PI_length: we don't produce intervals
    """
    T = len(actions)
    A = actions.astype(float)
    V = verifier_results.astype(float)

    # Cumulative metrics
    cum_failures = np.cumsum(A * (1 - V))
    cum_actions = np.cumsum(A)
    cum_actions_safe = np.maximum(cum_actions, 1)

    selective_risk = cum_failures / cum_actions_safe
    action_rate = cum_actions / np.arange(1, T + 1)

    # FCR-style metric (matches CAP's FCP exactly under our mapping)
    # FCP(T) = sum_{t} A_t * (1-V_t) / max(sum A_t, 1) = R_T^act
    fcr_style = selective_risk  # identical

    # Average score among released items (closest analogue to PI length)
    # Lower score → more confident → "shorter interval" analogue
    released_scores = []
    avg_score_released = np.full(T, np.nan)
    for t in range(T):
        if A[t] == 1:
            released_scores.append(scores[t])
        if len(released_scores) > 0:
            avg_score_released[t] = np.mean(released_scores)

    # Threshold trajectory
    q_trajectory = np.array(q_deploy_history, dtype=float)

    # If CAP selections provided, compare selection patterns
    cap_comparison = {}
    if cap_selections is not None:
        cap_sel_rate = np.cumsum(cap_selections) / np.arange(1, T + 1)
        # Overlap: fraction of rounds where both CAP and CSA select
        both_select = A * cap_selections
        cap_comparison = {
            'cap_selection_rate': cap_sel_rate,
            'overlap_rate': np.cumsum(both_select) / np.maximum(cum_actions, 1),
        }

    result = {
        # Core CSA metrics
        'selective_risk': selective_risk,
        'action_rate': action_rate,
        'cum_actions': cum_actions,
        'cum_failures': cum_failures,
        'q_trajectory': q_trajectory,

        # CAP-comparable
        'fcr_style': fcr_style,
        'avg_score_released': avg_score_released,

        # Final values
        'final_selective_risk': float(selective_risk[-1]) if T > 0 else 0,
        'final_action_rate': float(action_rate[-1]) if T > 0 else 0,
        'final_cum_actions': int(cum_actions[-1]) if T > 0 else 0,
        'total_rounds': T,
    }
    result.update(cap_comparison)

    return result


def compute_certification_events(log_e_history: np.ndarray,
                                 log_thresh: np.ndarray,
                                 grid: np.ndarray) -> Dict:
    """
    Extract certification events from e-process history.

    Parameters
    ----------
    log_e_history : array of shape (T, m); log e-process at each round
    log_thresh : array of shape (m,); log(1/delta_q) targets
    grid : array of shape (m,); threshold values

    Returns
    -------
    dict with certification_times (per threshold), first_cert_time, etc.
    """
    T, m = log_e_history.shape
    cert_times = {}
    for k in range(m):
        crossed = np.where(log_e_history[:, k] >= log_thresh[k])[0]
        if len(crossed) > 0:
            cert_times[float(grid[k])] = int(crossed[0])

    return {
        'certification_times': cert_times,
        'num_certified': len(cert_times),
        'first_cert_time': min(cert_times.values()) if cert_times else None,
        'last_cert_time': max(cert_times.values()) if cert_times else None,
    }


def make_summary_table(all_experiment_results: Dict[str, Dict]) -> str:
    """
    Create a markdown summary table across all experiments.

    Columns: Experiment | Final Risk | ±std | Action Rate | Runtime
    """
    lines = [
        "| Experiment | α | Final Risk (mean±std) | Action Rate | Runtime (s) |",
        "|-----------|---|----------------------|-------------|-------------|",
    ]
    for name, res in all_experiment_results.items():
        risk_str = f"{res.get('final_risk_mean', 0):.4f}±{res.get('final_risk_std', 0):.4f}"
        ar_str = f"{res.get('final_action_rate_mean', 0):.4f}"
        rt_str = f"{res.get('mean_runtime', 0):.2f}"
        alpha_str = f"{res.get('alpha', 0.10):.2f}"
        lines.append(f"| {name} | {alpha_str} | {risk_str} | {ar_str} | {rt_str} |")

    return "\n".join(lines)
