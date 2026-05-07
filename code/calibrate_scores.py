"""
Calibrate confidence scores via isotonic regression on a held-out split.

Domain-agnostic. Splits the scored inference CSV into calibration (CAL)
and evaluation (EVAL) using a fixed seed, fits isotonic regression on
CAL mapping raw confidence to observed failure rate, and writes an EVAL
CSV with a `calibrated_score` column plus a `_meta.json` with the
CAL-derived grid range. Ensures hyperparameter tuning (grid range, etc.)
is chosen on data disjoint from the CSA evaluation stream, preserving
CSA's anytime-valid guarantee.

Raw-signal detection (priority order):
  1. `max_score` (medical; from score_options.py)
  2. `sc_score` (any MCQ domain; from score_selfconsistency.py)
  3. `mean_logprob` (financial, legal)
  4. `mean_action_logprob` (agents)

Failure detection (priority order):
  1. `correct` column (financial, legal)
  2. `success` column (agents)
  3. `scored_answer == gold_answer` (medical)
  4. `sc_answer == gold_answer` (MCQ with self-consistency)

Usage:
    python calibrate_scores.py \
        --input results/medical_inference_scored_8bit.csv \
        --output results/medical_inference_calibrated.csv
"""

import argparse
import re
import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def detect_raw_score(df: pd.DataFrame) -> np.ndarray:
    """Return raw failure-probability score (higher = more likely to fail)
    from whatever confidence signal the CSV carries."""
    if 'max_score' in df.columns:
        return 1.0 - df['max_score'].values.astype(float)
    if 'sc_score' in df.columns:
        return 1.0 - df['sc_score'].values.astype(float)
    if 'mean_logprob' in df.columns:
        lp = df['mean_logprob'].values.astype(float)
        # Normalize to [0,1], higher lp -> lower raw
        return np.clip(1.0 - (lp + 5.0) / 5.0, 0.0, 1.0)
    if 'mean_action_logprob' in df.columns:
        lp = df['mean_action_logprob'].values.astype(float)
        return np.clip(1.0 - (lp + 5.0) / 5.0, 0.0, 1.0)
    raise ValueError(
        "No known confidence column in CSV. Expected one of: "
        "max_score, sc_score, mean_logprob, mean_action_logprob.")


def _parse_gold_list(gold):
    """Parse possibly-list-shaped gold. TAT-QA stores gold as a JSON
    list string like '["$1.9 million"]'. Returns a list of strings."""
    import ast
    if isinstance(gold, (list, tuple)):
        return [str(g) for g in gold]
    s = str(gold).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(g) for g in parsed]
        except (ValueError, SyntaxError):
            pass
    return [s]


_UNIT_WORDS = re.compile(
    r'\b(million|billion|thousand|percent|percentage|dollars?|usd|%)\b',
    re.IGNORECASE)


def _clean_for_numeric(s: str) -> str:
    s = str(s).replace(',', '').replace('$', '').replace('%', '')
    s = _UNIT_WORDS.sub('', s)
    s = s.replace('(', '-').replace(')', '')  # (1.2) -> -1.2 accounting
    s = s.strip()
    m = re.search(r'-?\d+\.?\d*', s)
    return m.group(0) if m else ''


def _numeric_or_string_match(pred, gold, tol: float = 2e-2) -> bool:
    """Return True if pred matches gold. Handles:
      * JSON-list golds (TAT-QA): match against any element.
      * Numeric equality with pct/decimal/1000x-scale tolerance.
      * Unit-bearing golds ('$1.9 million') by stripping unit words.
      * Case-insensitive substring match for free-form text.
    """
    pred_s = str(pred).strip()
    gold_candidates = _parse_gold_list(gold)
    for g in gold_candidates:
        g_s = str(g).strip()
        # Try numeric with unit/decoration stripping
        p_num = _clean_for_numeric(pred_s)
        g_num = _clean_for_numeric(g_s)
        if p_num and g_num:
            try:
                p = float(p_num)
                gv = float(g_num)
                if gv == 0:
                    if abs(p) < tol:
                        return True
                    continue
                for scale in (1.0, 100.0, 0.01):
                    if abs(p * scale - gv) / max(abs(gv), 1e-12) < tol:
                        return True
            except ValueError:
                pass
        # String match fallback (substring, case-insensitive)
        p_low = pred_s.lower().rstrip('.').strip()
        g_low = g_s.lower().rstrip('.').strip()
        if not p_low or not g_low:
            continue
        if p_low == g_low or g_low in p_low or p_low in g_low:
            return True
    return False


def detect_fail(df: pd.DataFrame) -> np.ndarray:
    """Return binary failure per row (1 = model got it wrong).

    Priority: scored_answer/gold_answer > sc_answer/gold_answer >
    correct > success. Scored-answer pairs take priority for MCQ domains
    because the CSA stream uses them as V_t; the `correct` column there
    refers to the free-form generation (which can differ).

    Uses `_numeric_or_string_match` for comparison, which handles both
    MCQ letter/string golds and numeric golds (including percentage-vs-
    decimal equivalence, critical for FinQA where gold stores 0.1446 and
    the model says 14.46).
    """
    if 'scored_answer' in df.columns and 'gold_answer' in df.columns:
        fails = [0 if _numeric_or_string_match(p, g) else 1
                 for p, g in zip(df['scored_answer'], df['gold_answer'])]
        return np.array(fails, dtype=int)
    if 'sc_answer' in df.columns and 'gold_answer' in df.columns:
        fails = [0 if _numeric_or_string_match(p, g) else 1
                 for p, g in zip(df['sc_answer'], df['gold_answer'])]
        return np.array(fails, dtype=int)
    if 'correct' in df.columns:
        return 1 - df['correct'].astype(int).values
    if 'success' in df.columns:
        return 1 - df['success'].astype(int).values
    raise ValueError(
        "No known correctness column in CSV. Expected one of: "
        "(scored_answer,gold_answer), (sc_answer,gold_answer), correct, success.")


def calibrate(input_csv: str, output_csv: str,
              split_ratio: float = 0.2, seed: int = 42):
    df = pd.read_csv(input_csv)
    n = len(df)
    raw = detect_raw_score(df)
    fail = detect_fail(df)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_cal = int(round(n * split_ratio))
    cal_idx = perm[:n_cal]
    eval_idx = perm[n_cal:]

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(raw[cal_idx], fail[cal_idx])

    eval_raw = raw[eval_idx]
    eval_calibrated = iso.predict(eval_raw)

    eval_df = df.iloc[eval_idx].copy().reset_index(drop=True)
    eval_df['raw_score'] = eval_raw
    eval_df['calibrated_score'] = eval_calibrated

    cal_calibrated = iso.predict(raw[cal_idx])
    grid_min = float(max(np.percentile(cal_calibrated, 2), 0.001))
    grid_max = float(min(np.percentile(cal_calibrated, 98), 0.8))

    meta = {
        'input_csv': os.path.basename(input_csv),
        'split_seed': int(seed),
        'split_ratio': float(split_ratio),
        'n_cal': int(n_cal),
        'n_eval': int(len(eval_idx)),
        'cal_fail_rate': float(fail[cal_idx].mean()),
        'eval_fail_rate': float(fail[eval_idx].mean()),
        'grid_min': grid_min,
        'grid_max': grid_max,
        'isotonic_x': iso.X_thresholds_.tolist(),
        'isotonic_y': iso.y_thresholds_.tolist(),
    }

    eval_df.to_csv(output_csv, index=False)
    meta_path = os.path.splitext(output_csv)[0] + '_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"CAL:  {n_cal} items  (fail rate = {meta['cal_fail_rate']:.3f})")
    print(f"EVAL: {meta['n_eval']} items  (fail rate = {meta['eval_fail_rate']:.3f})")
    print(f"Grid range (from CAL): [{grid_min:.4f}, {grid_max:.4f}]")
    print(f"Saved eval CSV:  {output_csv}")
    print(f"Saved metadata:  {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',
                    default='results/medical_inference_scored_8bit.csv')
    ap.add_argument('--output',
                    default='results/medical_inference_calibrated.csv')
    ap.add_argument('--split_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    calibrate(args.input, args.output, args.split_ratio, args.seed)


if __name__ == '__main__':
    main()
