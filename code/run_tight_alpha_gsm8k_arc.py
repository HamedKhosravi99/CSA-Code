"""
Run LTT, CRC, NEX-Conf, ConfFact (Mohri-Conf) at tight alpha values on
GSM8K and ARC so phase_budget / riskandar panels for these two
benchmarks can show the regime where most methods fail.

Tight alphas to cover (chosen to be below the base-model error so that
the headline story changes):
    gsm8k  (base err  5.0%)  -> alpha in {0.01, 0.03, 0.075}
    arc    (base err 10.0%)  -> alpha in {0.05, 0.075}
      (arc 0.05 already has all 10 methods in _verified_numbers.json,
       but the 0.075 cell is new.)

Output: results/tight_alpha_gsm8k_arc.json
Structure matches run_ltt_grid.py output so merge_* can reuse logic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
import numpy as np
np.seterr(all="ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from principled_baselines import LTTMethod
from test_new_baselines import CRCMethod, NEXConformalMethod, MohriConformalMethod


BENCHES = {
    "gsm8k": {
        "csv": "results/gsm8k_inference_calibrated.csv",
        "module": "domains.gsm8k.stream", "cls": "GSM8KStream",
        "alphas": [0.01, 0.03, 0.075],
    },
    "arc": {
        "csv": "results/arc_inference_calibrated.csv",
        "module": "domains.arc.stream", "cls": "ARCStream",
        "alphas": [0.05, 0.075],
    },
}

METHODS = {
    "LTT":        lambda a, bi: LTTMethod(
        alpha=a, delta=0.10, cal_size=bi, n_thresholds=15,
        score_min=0.01, score_max=0.99),
    "CRC":        lambda a, bi: CRCMethod(
        alpha=a, delta=0.10, cal_size=bi, n_thresholds=15),
    "NEX-Conf":   lambda a, bi: NEXConformalMethod(
        alpha=a, rho=0.99, window=bi, n_thresholds=15),
    "Mohri-Conf": lambda a, bi: MohriConformalMethod(
        alpha=a, delta=0.10, cal_size=bi, n_thresholds=15),
}


def make_stream(spec):
    import importlib
    m = importlib.import_module(spec["module"])
    return getattr(m, spec["cls"])(spec["csv"])


def replay(method, stream, indices, burn_in):
    cum_act = cum_fail = 0
    max_risk = 0.0
    for step_idx, t in enumerate(indices):
        rd = stream.get_round(int(t))
        s_t = float(rd.score_hint) if rd.score_hint is not None else 0.5
        V_t = int(rd.V_t)
        acted = method.decide(s_t, step_idx)
        method.update(s_t, V_t, step_idx)
        if acted:
            cum_act += 1
            if V_t == 0:
                cum_fail += 1
            if cum_act >= burn_in:
                r = cum_fail / cum_act
                if r > max_risk:
                    max_risk = r
    final_risk = cum_fail / max(cum_act, 1) if cum_act > 0 else 0.0
    final_ar = cum_act / max(len(indices), 1)
    if cum_act < burn_in:
        max_risk = final_risk
    return {"final_risk": float(final_risk),
            "final_ar": float(final_ar),
            "max_risk": float(max_risk)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_reps", type=int, default=10)
    ap.add_argument("--n_passes", type=int, default=30)
    ap.add_argument("--burn_in_accepts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/tight_alpha_gsm8k_arc.json")
    args = ap.parse_args()

    print(f"Tight-alpha run for missing SOTA-conformal methods on GSM8K/ARC")
    print(f"  {args.n_reps} reps, {args.n_passes} passes, burn-in {args.burn_in_accepts}")
    t_total = time.time()
    results = {}
    for b, spec in BENCHES.items():
        if not os.path.exists(spec["csv"]):
            print(f"[SKIP] {b}: no {spec['csv']}")
            continue
        results[b] = {}
        stream = make_stream(spec)
        n_items = len(stream)
        print(f"  --- {b}  N={n_items} ---")
        for a in spec["alphas"]:
            results[b][f"{a:.3g}"] = {}
            for mname, factory in METHODS.items():
                t0 = time.time()
                reps = []
                for rep in range(args.n_reps):
                    rng = np.random.RandomState(args.seed + rep)
                    idx = np.concatenate(
                        [rng.permutation(n_items) for _ in range(args.n_passes)])
                    m = factory(a, args.burn_in_accepts)
                    reps.append(replay(m, stream, idx, args.burn_in_accepts))
                fr = np.array([r["final_risk"] for r in reps])
                fa = np.array([r["final_ar"]   for r in reps])
                mr = np.array([r["max_risk"]   for r in reps])
                pv = int(np.sum(mr > a))
                dt = time.time() - t0
                results[b][f"{a:.3g}"][mname] = {
                    "alpha":   a,
                    "final_risk_mean": float(fr.mean()),
                    "final_ar_mean":   float(fa.mean()),
                    "max_risk_mean":   float(mr.mean()),
                    "pathwise_violations":     pv,
                    "pathwise_violation_rate": f"{pv}/{args.n_reps}",
                    "n_reps": args.n_reps,
                }
                print(f"    [{b} a={a:.3g} {mname:12s}]  ({dt:5.1f}s) "
                      f"Risk={fr.mean()*100:5.2f}%  AR={fa.mean()*100:5.2f}%  "
                      f"PathV={pv}/{args.n_reps}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTotal wall time: {time.time() - t_total:.1f}s")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
