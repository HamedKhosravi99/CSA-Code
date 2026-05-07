"""
Fast option-scoring pass: adds P(A), P(B), P(C), P(D) to an existing inference CSV.

One forward pass per item (no generation), takes minutes not hours.
Uses the same prompt format as inference.py.
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from mlx_lm import load
import mlx.core as mx
from datasets import load_dataset


def format_medqa_prompt(question: str, options: dict) -> str:
    prompt = (
        "You are a medical expert. Answer the following medical question "
        "by selecting the correct option. Think step by step, then provide "
        "your final answer as 'The answer is (X)' where X is A, B, C, or D.\n\n"
        f"Question: {question}\n\n"
    )
    for key in sorted(options.keys()):
        prompt += f"{key}. {options[key]}\n"
    prompt += "\nAnswer:"
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/medical-4bit')
    parser.add_argument('--csv', default='results/medical_inference.csv')
    parser.add_argument('--output', default='results/medical_inference_scored.csv')
    parser.add_argument('--split', default='test')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})

    option_tokens = {
        'A': tokenizer.encode(' A', add_special_tokens=False)[-1],
        'B': tokenizer.encode(' B', add_special_tokens=False)[-1],
        'C': tokenizer.encode(' C', add_special_tokens=False)[-1],
        'D': tokenizer.encode(' D', add_special_tokens=False)[-1],
    }
    print(f"Option token IDs: {option_tokens}")

    print("Loading dataset...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=args.split)

    df = pd.read_csv(args.csv)
    n = min(len(df), args.limit) if args.limit > 0 else len(df)

    scores_A, scores_B, scores_C, scores_D = [], [], [], []
    max_scores, scored_answers = [], []

    t0 = time.time()
    for idx in range(n):
        item = ds[idx]
        prompt = format_medqa_prompt(item['question'], item['options'])
        tokens = mx.array(tokenizer.encode(prompt))[None]

        logits = model(tokens)
        last_logits = logits[0, -1, :]
        probs = mx.softmax(last_logits)

        p = {}
        for letter, tid in option_tokens.items():
            p[letter] = float(probs[tid])

        scores_A.append(p['A'])
        scores_B.append(p['B'])
        scores_C.append(p['C'])
        scores_D.append(p['D'])

        total = sum(p.values())
        normed = {k: v / total for k, v in p.items()} if total > 0 else p
        best = max(normed, key=normed.get)
        max_scores.append(normed[best])
        scored_answers.append(best)

        mx.eval(probs)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            eta = (n - idx - 1) / speed if speed > 0 else 0
            correct_so_far = sum(1 for i in range(idx+1) if scored_answers[i] == df.iloc[i]['gold_answer'])
            acc = correct_so_far / (idx + 1)
            print(f"  {idx+1}/{n}  acc={acc:.1%}  {speed:.1f} items/s  ETA={eta/60:.0f}min")

    df = df.iloc[:n].copy()
    df['score_A'] = scores_A
    df['score_B'] = scores_B
    df['score_C'] = scores_C
    df['score_D'] = scores_D
    df['max_score'] = max_scores
    df['scored_answer'] = scored_answers

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)

    elapsed = time.time() - t0
    print(f"\nDone. {n} items scored in {elapsed:.1f}s")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
