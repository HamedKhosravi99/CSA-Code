"""
Run LTT on the full alpha grid for every benchmark and emit a summary
JSON that can be merged into paper_tables/_verified_numbers.json.

Why this exists: run_ltt_pivots.py only ran LTT at the per-benchmark
pivotal alpha, so the verified-numbers file has LTT entries only at 1-2
alpha values per benchmark (most cells are missing). Every other method
(CSA, CRC, NEX-Conf, Mohri-Conf, ACI, SAOCP) is fully populated at all
6 alphas, so LTT appears as gaps in phase_budget_allmethods.pdf. This
script fills those gaps.

Usage:
    python run_ltt_grid.py
    python run_ltt_grid.py --n_reps 5          # quicker
    python run_ltt_grid.py --only medical      # single benchmark
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


BENCHES = {
    "medical":  {"csv": "results/medical_inference_calibrated.csv",
                 "module": "domains.medical.stream",  "cls": "MedQAStream"},
    "pubmedqa": {"csv": "results/pubmedqa_inference_calibrated.csv",
                 "module": "domains.pubmedqa.stream", "cls": "PubMedQAStream"},
    "tatqa":    {"csv": "results/tatqa_inference_calibrated.csv",
                 "module": "domains.tatqa.stream",    "cls": "TATQAStream"},
    "mednli":   {"csv": "results/mednli_calibrated.csv",
                 "module": "domains.mednli.stream",   "cls": "MedNLIStream"},
    "gsm8k":    {"csv": "results/gsm8k_inference_calibrated.csv",
                 "module": "domains.gsm8k.stream",    "cls": "GSM8KStream"},
    "headqa":   {"csv": "results/headqa_inference_calibrated.csv",
                 "module": "domains.headqa.stream",   "cls": "HEADQAStream"},
    "arc":      {"csv": "results/arc_inference_calibrated.csv",
                 "module": "domains.arc.stream",      "cls": "ARCStream"},
    "casehold": {"csv": "results/casehold_inference_calibrated.csv",
                 "module": "domains.casehold.stream", "cls": "CaseHOLDStream"},
}

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


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


def run_bench_alpha(bench, spec, alpha, n_reps, n_passes, burn_in, seed):
    stream = make_stream(spec)
    n_items = len(stream)
    t0 = time.time()
    reps = []
    for rep in range(n_reps):
        rng = np.random.RandomState(seed + rep)
        idx = np.concatenate([rng.permutation(n_items) for _ in range(n_passes)])
        ltt = LTTMethod(alpha=alpha, delta=0.10, cal_size=burn_in,
                        n_thresholds=15, score_min=0.01, score_max=0.99)
        reps.append(replay(ltt, stream, idx, burn_in))
    fr = np.array([r["final_risk"] for r in reps])
    fa = np.array([r["final_ar"]   for r in reps])
    mr = np.array([r["max_risk"]   for r in reps])
    pv = int(np.sum(mr > alpha))
    dt = time.time() - t0
    out = {
        "alpha":   alpha,
        "n_items": n_items,
        "final_risk_mean": float(fr.mean()), "final_risk_std": float(fr.std()),
        "final_ar_mean":   float(fa.mean()), "final_ar_std":   float(fa.std()),
        "max_risk_mean":   float(mr.mean()), "max_risk_std":   float(mr.std()),
        "pathwise_violations":     pv,
        "pathwise_violation_rate": f"{pv}/{n_reps}",
        "n_reps": n_reps,
        "seconds": dt,
    }
    print(f"    [{bench} a={alpha:.2f}]  ({dt:5.1f}s) "
          f"Risk={fr.mean()*100:5.2f}%  AR={fa.mean()*100:5.2f}%  "
          f"PathV={pv}/{n_reps}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_reps", type=int, default=10)
    ap.add_argument("--n_passes", type=int, default=30)
    ap.add_argument("--burn_in_accepts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None,
                    help="Run only one benchmark (e.g., --only medical)")
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--out", default="results/ltt_grid_summary.json")
    args = ap.parse_args()

    targets = [args.only] if args.only else list(BENCHES.keys())
    print(f"LTT full grid: {len(targets)} benchmarks x {len(args.alphas)} alphas, "
          f"{args.n_reps} reps, {args.n_passes} passes, burn-in {args.burn_in_accepts}")
    t_total = time.time()
    results = {b: {} for b in targets}
    for b in targets:
        spec = BENCHES[b]
        if not os.path.exists(spec["csv"]):
            print(f"[SKIP] {b}: no {spec['csv']}")
            continue
        print(f"  --- {b} ---")
        for a in args.alphas:
            out = run_bench_alpha(b, spec, a, args.n_reps, args.n_passes,
                                  args.burn_in_accepts, args.seed)
            results[b][f"{a:.2f}"] = out

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTotal wall time: {time.time() - t_total:.1f}s")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
