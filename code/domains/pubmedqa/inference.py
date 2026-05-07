"""
PubMedQA domain inference (biomedical research QA, yes/no/maybe).

Dataset: qiaojin/PubMedQA (pqa_labeled config), 1000 labeled items.
Each item has a question + context (abstracts) + long_answer + final
decision in {yes, no, maybe}.

Usage:
    python domains/pubmedqa/inference.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output results/pubmedqa_inference.csv
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd


PUBMEDQA_PROMPT = (
    "You are a biomedical research assistant. Given the following research "
    "question and abstract context, decide whether the evidence supports "
    "a 'yes', 'no', or 'maybe' answer. Think briefly, then state your "
    "final answer as 'The answer is: yes' / 'no' / 'maybe'.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)


def extract_yesno_maybe(text: str) -> str:
    if not text:
        return ''
    # Look for final-answer patterns first
    for pat in [r'[Tt]he\s+answer\s+is\s*:?\s*\(?(yes|no|maybe)\)?',
                r'[Aa]nswer\s*:?\s*\(?(yes|no|maybe)\)?',
                r'[Ff]inal\s+answer\s*:?\s*\(?(yes|no|maybe)\)?']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    # Fallback: first standalone yes/no/maybe
    m = re.search(r'\b(yes|no|maybe)\b', text, re.IGNORECASE)
    return m.group(1).lower() if m else ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--output', default='results/pubmedqa_inference.csv')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    print("Loading dataset: qiaojin/PubMedQA (pqa_labeled)...")
    ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
    n = min(len(ds), args.limit) if args.limit > 0 else len(ds)
    print(f"  {n} items")

    print(f"Loading model {args.model}...")
    llm = LLM(model=args.model, dtype='bfloat16',
              gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(n=1, temperature=args.temperature,
                        max_tokens=args.max_new_tokens, logprobs=1)

    prompts = []
    for i in range(n):
        item = ds[i]
        ctx = item.get('context', {})
        if isinstance(ctx, dict):
            ctx_parts = ctx.get('contexts', []) or []
            ctx_str = '\n'.join(str(c) for c in ctx_parts)[:3000]
        else:
            ctx_str = str(ctx)[:3000]
        prompts.append(PUBMEDQA_PROMPT.format(
            question=item['question'], context=ctx_str))

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.1f}s  ({n/elapsed:.1f} items/s)")

    results = []
    for i, out in enumerate(outputs):
        gen = out.outputs[0]
        response = gen.text
        model_answer = extract_yesno_maybe(response)
        gold_answer = str(ds[i].get('final_decision', '')).strip().lower()
        correct = int(model_answer == gold_answer)

        if gen.logprobs:
            lps = [list(s.values())[0].logprob for s in gen.logprobs if s]
            mean_lp = float(np.mean(lps)) if lps else -1.0
        else:
            mean_lp = -1.0

        results.append({
            'item_id': i,
            'question': ds[i]['question'][:300],
            'response': response[:500],
            'model_answer': model_answer,
            'gold_answer': gold_answer,
            'correct': correct,
            'mean_logprob': mean_lp,
            'question_length': len(ds[i]['question']),
            'response_length': len(response),
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    acc = df['correct'].mean()
    print(f"Done. n={n}, acc={acc:.1%}, saved to {args.output}")


if __name__ == '__main__':
    main()
