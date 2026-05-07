"""
Run passive A-RCPS (Xu, Karampatziakis, Mineiro NeurIPS 2024, FullyObserved
variant) on the MedQA EVAL stream, recording BOTH the marginal risk that
A-RCPS actually controls AND the selective risk that CSA controls.  The
comparison exposes the protocol / risk-notion mismatch between the two
methods: A-RCPS's marginal bound is satisfied by design, but its selective
risk (CSA's target metric) typically exceeds alpha.

Usage:
    python run_arcps_medqa.py --alpha 0.20 --n_reps 10

Output:
    results/arcps_medical_alpha{ALPHA}.json
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import pandas as pd

from arcps_adapter import ARCPSPassive


# =============================================================================
#  Per-replication streaming run
# =============================================================================

def run_replication(
    scores: np.ndarray,
    labels: np.ndarray,
    idx: np.ndarray,
    alpha: float,
    n_betas: int = 100,
    confidence: float = 0.05,
) -> dict:
    """
    Stream items in index order `idx`, let A-RCPS decide, collect both risk
    metrics.  Returns summary statistics plus pathwise violation flags.
    """
    arcps = ARCPSPassive(alpha=alpha, n_betas=n_betas, confidence=confidence)
    N = len(idx)
    acts = np.zeros(N, dtype=bool)
    Vs = np.zeros(N, dtype=int)
    for t in range(N):
        j = idx[t]
        s = float(scores[j])
        y = int(labels[j])
        acts[t] = arcps.decide(s, t)
        Vs[t] = y
        # A-RCPS FullyObserved: update on EVERY round regardless of act/abstain.
        arcps.update(s, y, t)

    N_acted = int(acts.sum())
    wrong_and_acted = int(((1 - Vs) * acts.astype(int)).sum())

    # CSA's metric: selective failure rate
    sel_risk = wrong_and_acted / max(N_acted, 1)
    # A-RCPS's metric: marginal failure rate
    marg_risk = wrong_and_acted / N
    AR = N_acted / N

    # Pathwise trajectories
    running_acts = np.cumsum(acts.astype(int))
    running_wrong = np.cumsum((1 - Vs) * acts.astype(int))
    running_acts_safe = np.maximum(running_acts, 1)
    sel_traj = running_wrong / running_acts_safe
    marg_traj = running_wrong / np.arange(1, N + 1)

    # PathV is measured POST-BURN-IN (standard in CSA); we report both
    # unconditional and post-first-accept (burn-in agnostic).
    first_act = int(np.argmax(acts)) if acts.any() else N
    pathV_sel_post = (
        bool((sel_traj[first_act:] > alpha).any()) if first_act < N else False
    )
    pathV_marg_post = (
        bool((marg_traj[first_act:] > alpha).any()) if first_act < N else False
    )

    return {
        "N": int(N),
        "N_acted": N_acted,
        "AR": float(AR),
        "sel_risk": float(sel_risk),
        "marg_risk": float(marg_risk),
        "pathV_sel": pathV_sel_post,
        "pathV_marg": pathV_marg_post,
        "max_sel_traj": float(sel_traj[first_act:].max()) if first_act < N else 0.0,
        "max_marg_traj": float(marg_traj[first_act:].max()) if first_act < N else 0.0,
    }


# =============================================================================
#  Driver
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="results/medical_inference_calibrated.csv",
    )
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--n_reps", type=int, default=10)
    parser.add_argument("--n_passes", type=int, default=3,
                        help="Number of passes over the EVAL set per replication.")
    parser.add_argument("--out",
                        default="results/arcps_medical_alpha0.20.json")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    scores = df["calibrated_score"].to_numpy(dtype=float)
    labels = df["correct"].to_numpy(dtype=int)
    N = len(df)

    print(f"[A-RCPS vs CSA on MedQA]  N={N}, alpha={args.alpha:.2f}, "
          f"n_reps={args.n_reps}, n_passes={args.n_passes}")

    reps = []
    for seed in range(args.n_reps):
        rng = np.random.RandomState(seed)
        idx_all = []
        for _ in range(args.n_passes):
            perm = np.arange(N)
            rng.shuffle(perm)
            idx_all.extend(perm.tolist())
        idx_all = np.asarray(idx_all, dtype=int)
        res = run_replication(scores, labels, idx_all, alpha=args.alpha)
        res["seed"] = seed
        reps.append(res)
        print(f"  seed={seed}: AR={res['AR']:.3f}  sel_risk={res['sel_risk']:.3f}  "
              f"marg_risk={res['marg_risk']:.3f}  "
              f"pathV_sel={int(res['pathV_sel'])}  pathV_marg={int(res['pathV_marg'])}")

    agg = {
        "benchmark": "medical",
        "method": "A-RCPS (FullyObserved, passive adapter)",
        "source": "Xu, Karampatziakis, Mineiro NeurIPS 2024 (github.com/neilzxu/active-rcps)",
        "alpha": float(args.alpha),
        "n_reps": int(args.n_reps),
        "n_passes": int(args.n_passes),
        "N_per_pass": int(N),
        "AR_mean": float(np.mean([r["AR"] for r in reps])),
        "sel_risk_mean": float(np.mean([r["sel_risk"] for r in reps])),
        "marg_risk_mean": float(np.mean([r["marg_risk"] for r in reps])),
        "pathV_sel": int(sum(r["pathV_sel"] for r in reps)),
        "pathV_marg": int(sum(r["pathV_marg"] for r in reps)),
        "max_sel_traj_mean": float(np.mean([r["max_sel_traj"] for r in reps])),
        "max_marg_traj_mean": float(np.mean([r["max_marg_traj"] for r in reps])),
        "replications": reps,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)

    print("\n=== Aggregate results over", args.n_reps, "replications ===")
    print(f"  Accept rate    = {agg['AR_mean']:.3f}")
    print(f"  Selective risk = {agg['sel_risk_mean']:.3f}   "
          f"[CSA target alpha = {args.alpha:.2f}]")
    print(f"  Marginal risk  = {agg['marg_risk_mean']:.3f}   "
          f"[A-RCPS target theta = {args.alpha:.2f}]")
    print(f"  PathV (sel)  = {agg['pathV_sel']}/{args.n_reps}")
    print(f"  PathV (marg) = {agg['pathV_marg']}/{args.n_reps}")
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
