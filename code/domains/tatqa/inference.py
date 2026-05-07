"""
TAT-QA domain inference (hybrid tabular + text financial QA).

Dataset: next-tat/tat-qa (nested schema):
    each item = {table, paragraphs, questions[list]}
    each question = {question, answer(list), answer_type, scale, rel_paragraphs, ...}

We flatten nested items to per-question rows. Gold is
`question['answer']` (list of strings). Scale (`percent`, `million`,
`thousand`, etc.) is applied during verification.

Supports answer types:
  - `span` / `spans`: substring match against paragraphs/table cells
  - `arithmetic` / `counting`: numeric match with pct/decimal/scale tolerance

Usage:
    python domains/tatqa/inference.py \
        --model SUFE-AIFLM-Lab/Fin-R1 \
        --output results/tatqa_inference.csv
"""

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd


TATQA_PROMPT = (
    "You are a financial analyst. Using the table and accompanying text, "
    "answer the question. Think briefly, then end with 'Final answer: "
    "<value>'. For numeric answers give only the number (no units, no "
    "commas, no currency sign). For textual spans, give only the phrase.\n\n"
    "Table:\n{table_md}\n\n"
    "Text:\n{paragraphs}\n\n"
    "Question: {question}\n\n"
    "Response:"
)

SCALE_FACTORS = {
    'percent': 1.0,         # we treat numbers in percent form directly
    'thousand': 1e3,
    'million': 1e6,
    'billion': 1e9,
    '': 1.0,
    None: 1.0,
}


def format_table(table) -> str:
    if table is None:
        return '(no table)'
    if isinstance(table, dict) and 'table' in table:
        table = table['table']
    if not table:
        return '(no table)'
    lines = []
    for row in table[:30]:
        cells = [str(c).replace('\n', ' ').strip()[:60] for c in row]
        lines.append(' | '.join(cells))
    return '\n'.join(lines)


def format_paragraphs(paragraphs) -> str:
    if paragraphs is None:
        return ''
    if isinstance(paragraphs, list):
        parts = []
        for p in paragraphs[:15]:
            if isinstance(p, dict):
                parts.append(str(p.get('text', ''))[:800])
            else:
                parts.append(str(p)[:800])
        return '\n'.join(parts)
    return str(paragraphs)[:3000]


def extract_answer(text: str) -> str:
    if not text:
        return ''
    # 1. Final answer marker
    for pat in [r'[Ff]inal\s+answer\s*:?\s*\$?([^\n]+?)\s*$',
                r'[Tt]he\s+answer\s+is\s*:?\s*\$?([^\n]+?)\s*$',
                r'[Aa]nswer\s*:?\s*\$?([^\n]+?)\s*$']:
        for m in re.finditer(pat, text, re.MULTILINE):
            pass
        ms = list(re.finditer(pat, text, re.MULTILINE))
        if ms:
            return ms[-1].group(1).strip().rstrip('.').rstrip(',').strip()
    # 2. \boxed{} fallback
    idx = text.rfind(r'\boxed{')
    if idx != -1:
        rest = text[idx + len(r'\boxed{'):]
        depth = 1
        out = []
        for c in rest:
            if c == '{':
                depth += 1; out.append(c)
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
                out.append(c)
            else:
                out.append(c)
        return ''.join(out).strip().rstrip('.')
    # 3. Last line
    last = text.strip().split('\n')[-1].strip()
    return last.rstrip('.').strip()


def _normalize_numeric(s) -> float:
    if s is None:
        return None
    s = str(s).strip()
    # Strip currency, percent, commas, parentheses, whitespace
    s = s.replace(',', '').replace('$', '').replace('%', '')
    s = s.replace('(', '-').replace(')', '')  # (12.3) -> -12.3 accounting
    s = s.strip()
    # Extract first numeric token if string contains extra words
    m = re.search(r'-?\d+\.?\d*', s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _numeric_match(pred_val: float, gold_val: float, tol: float = 0.02) -> bool:
    if gold_val == 0:
        return abs(pred_val) < tol
    for scale in (1.0, 100.0, 0.01, 1e3, 1e-3, 1e6, 1e-6):
        if abs(pred_val * scale - gold_val) / max(abs(gold_val), 1e-12) < tol:
            return True
    return False


def _span_match(pred: str, gold: str) -> bool:
    p = str(pred).strip().lower().rstrip('.')
    g = str(gold).strip().lower().rstrip('.')
    if not p or not g:
        return False
    if p == g:
        return True
    if g in p or p in g:
        return True
    # Token overlap for longer phrases
    pt, gt = set(p.split()), set(g.split())
    if gt and gt.issubset(pt):
        return True
    return False


def check_answer(pred_str: str, gold_list, answer_type: str, scale: str) -> int:
    """Return 1 if predicted answer matches gold. Gold is a list of
    strings; we accept a match to any element (or all for 'spans')."""
    if gold_list is None or gold_list == '' or gold_list == []:
        return 0
    if not isinstance(gold_list, (list, tuple)):
        gold_list = [gold_list]

    if answer_type in ('arithmetic', 'counting'):
        p_val = _normalize_numeric(pred_str)
        if p_val is None:
            return 0
        for g in gold_list:
            g_val = _normalize_numeric(g)
            if g_val is None:
                continue
            if _numeric_match(p_val, g_val):
                return 1
        return 0

    # span / spans / other: check if prediction matches any gold span
    # (for multi-span, require all to appear)
    if answer_type == 'spans' and len(gold_list) > 1:
        return int(all(_span_match(pred_str, g) for g in gold_list))
    return int(any(_span_match(pred_str, g) for g in gold_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='SUFE-AIFLM-Lab/Fin-R1')
    parser.add_argument('--output', default='results/tatqa_inference.csv')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--split', default='validation')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of questions (not documents).')
    parser.add_argument('--answer_types', nargs='+', default=None,
                        help="Restrict to specific answer types, e.g. "
                             "'arithmetic' or 'arithmetic count'. Defaults "
                             "to all types.")
    parser.add_argument('--hf_path', default='next-tat/tat-qa')
    args = parser.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    print(f"Loading dataset: {args.hf_path} ({args.split})...")
    ds = load_dataset(args.hf_path, split=args.split)
    print(f"  {len(ds)} documents")

    # Flatten: for each document, enumerate questions
    rows = []
    for doc in ds:
        for q in doc.get('questions', []):
            rows.append({
                'doc_table': doc.get('table'),
                'doc_paragraphs': doc.get('paragraphs'),
                'question_text': q.get('question', ''),
                'gold_answer': q.get('answer', []) or [],
                'answer_type': q.get('answer_type', 'unknown'),
                'scale': q.get('scale', ''),
            })
    print(f"  {len(rows)} questions after flattening")

    if args.answer_types:
        before = len(rows)
        allowed = set(args.answer_types)
        rows = [r for r in rows if r['answer_type'] in allowed]
        print(f"  Filtered to types {sorted(allowed)}: "
              f"{len(rows)} questions (from {before})")

    n = min(len(rows), args.limit) if args.limit > 0 else len(rows)

    print(f"Loading model {args.model}...")
    llm = LLM(model=args.model, dtype='bfloat16',
              gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(n=1, temperature=args.temperature,
                        max_tokens=args.max_new_tokens, logprobs=1)

    prompts = []
    for i in range(n):
        r = rows[i]
        prompts.append(TATQA_PROMPT.format(
            question=r['question_text'],
            table_md=format_table(r['doc_table']),
            paragraphs=format_paragraphs(r['doc_paragraphs'])))

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.1f}s  ({n/elapsed:.1f} items/s)")

    results = []
    for i, out in enumerate(outputs):
        gen = out.outputs[0]
        response = gen.text
        model_answer = extract_answer(response)
        r = rows[i]
        correct = check_answer(model_answer, r['gold_answer'],
                               r['answer_type'], r['scale'])

        if gen.logprobs:
            lps = [list(s.values())[0].logprob for s in gen.logprobs if s]
            mean_lp = float(np.mean(lps)) if lps else -1.0
        else:
            mean_lp = -1.0

        gold_str = json.dumps(r['gold_answer']) if isinstance(r['gold_answer'], list) \
                   else str(r['gold_answer'])

        results.append({
            'item_id': i,
            'question': str(r['question_text'])[:300],
            'response': response[:600],
            'model_answer': str(model_answer)[:200],
            'gold_answer': gold_str[:200],
            'correct': correct,
            'mean_logprob': mean_lp,
            'question_length': len(str(r['question_text'])),
            'response_length': len(response),
            'answer_type': r['answer_type'],
            'scale': r['scale'],
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    acc = df['correct'].mean()
    by_type = df.groupby('answer_type')['correct'].mean().to_dict()
    print(f"Done. n={n}, acc={acc:.1%}")
    print(f"  By answer_type: {by_type}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
