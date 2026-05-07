"""
Run LTT at every benchmark's pivotal alpha so the violation heatmap
and the per-benchmark tables show real LTT numbers instead of dashes.

LTT is implemented in principled_baselines.py. This script replays it
on each calibrated CSV using the same stream / burn-in / n_reps /
n_passes as every other baseline, then writes a small summary JSON:

    results/ltt_pivotal_summary.json

Usage:
    python run_ltt_pivots.py
    python run_ltt_pivots.py --n_reps 5     # quicker
"""

import argparse, json, os, sys, time, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
import numpy as np
np.seterr(all='ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from principled_baselines import LTTMethod

BENCHES = {
    'medical':  {'pivot': 0.20, 'csv': 'results/medical_inference_calibrated.csv',
                 'module':'domains.medical.stream', 'cls':'MedQAStream'},
    'pubmedqa': {'pivot': 0.20, 'csv': 'results/pubmedqa_inference_calibrated.csv',
                 'module':'domains.pubmedqa.stream', 'cls':'PubMedQAStream'},
    'tatqa':    {'pivot': 0.20, 'csv': 'results/tatqa_inference_calibrated.csv',
                 'module':'domains.tatqa.stream', 'cls':'TATQAStream'},
    'mednli':   {'pivot': 0.20, 'csv': 'results/mednli_calibrated.csv',
                 'module':'domains.mednli.stream', 'cls':'MedNLIStream'},
    'gsm8k':    {'pivot': 0.05, 'csv': 'results/gsm8k_inference_calibrated.csv',
                 'module':'domains.gsm8k.stream', 'cls':'GSM8KStream'},
    'headqa':   {'pivot': 0.20, 'csv': 'results/headqa_inference_calibrated.csv',
                 'module':'domains.headqa.stream', 'cls':'HEADQAStream'},
    'arc':      {'pivot': 0.10, 'csv': 'results/arc_inference_calibrated.csv',
                 'module':'domains.arc.stream', 'cls':'ARCStream'},
    'casehold': {'pivot': 0.25, 'csv': 'results/casehold_inference_calibrated.csv',
                 'module':'domains.casehold.stream', 'cls':'CaseHOLDStream'},
}


def make_stream(spec):
    import importlib
    m = importlib.import_module(spec['module'])
    return getattr(m, spec['cls'])(spec['csv'])


def replay(method, stream, indices, burn_in):
    cum_act = cum_fail = 0; max_risk = 0.0
    for step_idx, t in enumerate(indices):
        rd = stream.get_round(int(t))
        s_t = float(rd.score_hint) if rd.score_hint is not None else 0.5
        V_t = int(rd.V_t)
        acted = method.decide(s_t, step_idx)
        method.update(s_t, V_t, step_idx)
        if acted:
            cum_act += 1
            if V_t == 0: cum_fail += 1
            if cum_act >= burn_in:
                r = cum_fail / cum_act
                if r > max_risk: max_risk = r
    fr = cum_fail / max(cum_act, 1) if cum_act > 0 else 0.0
    fa = cum_act / max(len(indices), 1)
    if cum_act < burn_in:
        max_risk = fr
    return {'final_risk': float(fr), 'final_ar': float(fa),
            'max_risk': float(max_risk)}


def run_bench(bench, spec, n_reps=10, n_passes=30, burn_in=500, seed=42):
    if not os.path.exists(spec['csv']):
        print(f'[SKIP] {bench}: no {spec["csv"]}')
        return None
    stream = make_stream(spec)
    n_items = len(stream)
    alpha = spec['pivot']
    print(f'  [{bench}] alpha={alpha} N={n_items} running LTT...')
    t0 = time.time()
    reps = []
    for rep in range(n_reps):
        rng = np.random.RandomState(seed + rep)
        idx = np.concatenate([rng.permutation(n_items) for _ in range(n_passes)])
        ltt = LTTMethod(alpha=alpha, delta=0.10, cal_size=burn_in,
                        n_thresholds=15, score_min=0.01, score_max=0.99)
        reps.append(replay(ltt, stream, idx, burn_in))
    fr = np.array([r['final_risk'] for r in reps])
    fa = np.array([r['final_ar']   for r in reps])
    mr = np.array([r['max_risk']   for r in reps])
    pv = int(np.sum(mr > alpha))
    dt = time.time() - t0
    out = {
        'alpha':   alpha,
        'n_items': n_items,
        'final_risk_mean': float(fr.mean()), 'final_risk_std': float(fr.std()),
        'final_ar_mean':   float(fa.mean()), 'final_ar_std':   float(fa.std()),
        'max_risk_mean':   float(mr.mean()), 'max_risk_std':   float(mr.std()),
        'pathwise_violations':     pv,
        'pathwise_violation_rate': f'{pv}/{n_reps}',
        'n_reps': n_reps,
    }
    print(f'    ({dt:.1f}s) LTT Risk={fr.mean()*100:.2f}% AR={fa.mean()*100:.2f}% PathV={pv}/{n_reps}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_reps', type=int, default=10)
    ap.add_argument('--n_passes', type=int, default=30)
    ap.add_argument('--burn_in_accepts', type=int, default=500)
    ap.add_argument('--only', default=None)
    args = ap.parse_args()

    results = {}
    targets = [args.only] if args.only else list(BENCHES.keys())
    print(f'LTT @ pivotal-alpha on {len(targets)} benchmarks, {args.n_reps} reps')
    t0 = time.time()
    for b in targets:
        r = run_bench(b, BENCHES[b], n_reps=args.n_reps,
                      n_passes=args.n_passes, burn_in=args.burn_in_accepts)
        if r is not None:
            results[b] = r

    out_path = 'results/ltt_pivotal_summary.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nTotal: {time.time()-t0:.1f}s')
    print(f'Saved: {out_path}')
    print('\nPivotal LTT numbers:')
    print(f'  {"Bench":<12} {"α":<6} {"Risk":<10} {"AR":<10} {"PathV":<8}')
    for b, r in results.items():
        print(f'  {b:<12} {r["alpha"]:<6} '
              f'{r["final_risk_mean"]*100:<9.2f}% '
              f'{r["final_ar_mean"]*100:<9.2f}% '
              f'{r["pathwise_violation_rate"]:<8}')


if __name__ == '__main__':
    main()
