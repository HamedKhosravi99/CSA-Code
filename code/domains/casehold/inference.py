"""
CaseHOLD domain inference (legal case-holding selection).

Dataset: casehold/casehold. Each item is a judicial opinion snippet with
the holding statement removed, plus 5 candidate holdings. The model
must pick the correct holding. Binary MCQ over 5 options (A-E).

Gold comes as an integer `label` in {0..4}.

Usage:
    python domains/casehold/inference.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output results/casehold_inference.csv
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd


CASEHOLD_PROMPT = (
    "[INST] You are a legal expert. Read the judicial opinion snippet "
    "below and select which of the five candidate holdings correctly "
    "completes the <HOLDING> marker. Think briefly, then state your "
    "final answer as 'The answer is (X)' where X is A, B, C, D, or E.\n\n"
    "Opinion:\n{citing_prompt}\n\n"
    "Candidate holdings:\n{options_str} [/INST]"
)

LETTERS = 'ABCDE'


def extract_mcq(text: str, letters: str = LETTERS) -> str:
    if not text:
        return ''
    lower = letters.lower()
    for pat in [rf'[Tt]he\s+answer\s+is\s*[:\s]*\(?([{letters}{lower}])\)?',
                rf'[Aa]nswer\s*[:\s]+\(?([{letters}{lower}])\)?']:
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()
    ms = re.findall(rf'\b([{letters}{lower}])\b', text)
    return ms[-1].upper() if ms else ''


def format_options(item) -> str:
    """Format candidate holdings. Handles both the original
    casehold/casehold schema (`holding_0..4` fields) and the LexGLUE
    schema (`endings` list)."""
    lines = []
    # LexGLUE schema: endings is a list of 5 strings
    if 'endings' in item and item['endings']:
        for i, text in enumerate(item['endings'][:5]):
            if text is not None:
                lines.append(f"{LETTERS[i]}. {str(text).replace(chr(10), ' ').strip()}")
        return '\n'.join(lines)
    # Legacy schema: holding_0 .. holding_4 fields
    for i, letter in enumerate(LETTERS):
        key = f'holding_{i}'
        if key in item and item[key] is not None:
            text = str(item[key]).replace('\n', ' ').strip()
            lines.append(f"{letter}. {text}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--output', default='results/casehold_inference.csv')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--hf_path', default='coastalcph/lex_glue',
                        help='HuggingFace dataset path. Default uses the '
                             'LexGLUE mirror of CaseHOLD (parquet-backed, '
                             'current). Alternative: legacy '
                             'casehold/casehold (dataset script, may fail).')
    parser.add_argument('--hf_config', default='case_hold',
                        help='Config name (LexGLUE requires `case_hold`).')
    parser.add_argument('--split', default='test')
    parser.add_argument('--quantization', default=None,
                        help="vLLM quantization mode: None, 'awq', 'awq_marlin', "
                             "'gptq', 'gptq_marlin', 'fp8'. Required for AWQ "
                             "or GPTQ quantized models (e.g. Qwen2.5-72B-AWQ).")
    parser.add_argument('--max_model_len', type=int, default=None,
                        help="Override vLLM max model length to avoid OOM on "
                             "long contexts with large models.")
    args = parser.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    print(f"Loading dataset: {args.hf_path} (config={args.hf_config}, "
          f"split={args.split})...")
    try:
        if args.hf_config:
            ds = load_dataset(args.hf_path, args.hf_config, split=args.split)
        else:
            ds = load_dataset(args.hf_path, split=args.split)
    except Exception as e:
        raise SystemExit(
            f"Failed to load {args.hf_path}. Alternatives: "
            f"'coastalcph/lex_glue' (config=case_hold), "
            f"'lexlms/lex_glue' (config=case_hold). Error: {e}")
    n = min(len(ds), args.limit) if args.limit > 0 else len(ds)
    print(f"  {n} items")

    print(f"Loading model {args.model} "
          f"(quantization={args.quantization})...")
    llm_kwargs = dict(model=args.model, dtype='bfloat16',
                      gpu_memory_utilization=0.85, trust_remote_code=True)
    if args.quantization:
        llm_kwargs['quantization'] = args.quantization
    if args.max_model_len:
        llm_kwargs['max_model_len'] = args.max_model_len
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(n=1, temperature=args.temperature,
                        max_tokens=args.max_new_tokens, logprobs=1)

    prompts, gold_answers = [], []
    for i in range(n):
        it = ds[i]
        # LexGLUE uses `context`, legacy uses `citing_prompt`; accept both
        cp = str(it.get('context') or it.get('citing_prompt') or '')[:3000]
        prompts.append(CASEHOLD_PROMPT.format(
            citing_prompt=cp, options_str=format_options(it)))
        lbl = int(it.get('label', 0))
        gold_answers.append(LETTERS[lbl] if 0 <= lbl < len(LETTERS) else '')

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.1f}s  ({n/elapsed:.1f} items/s)")

    results = []
    for i, out in enumerate(outputs):
        gen = out.outputs[0]
        response = gen.text
        model_answer = extract_mcq(response)
        gold_answer = gold_answers[i]
        correct = int(model_answer == gold_answer)

        if gen.logprobs:
            lps = [list(s.values())[0].logprob for s in gen.logprobs if s]
            mean_lp = float(np.mean(lps)) if lps else -1.0
        else:
            mean_lp = -1.0

        cp_text = str(ds[i].get('context') or ds[i].get('citing_prompt') or '')
        results.append({
            'item_id': i,
            'question': cp_text[:300],
            'response': response[:500],
            'model_answer': model_answer,
            'gold_answer': gold_answer,
            'correct': correct,
            'mean_logprob': mean_lp,
            'question_length': len(cp_text),
            'response_length': len(response),
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    acc = df['correct'].mean()
    print(f"Done. n={n}, acc={acc:.1%}, saved to {args.output}")


if __name__ == '__main__':
    main()
