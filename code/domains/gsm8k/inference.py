"""
GSM8K domain inference (grade-school math word problems).

Dataset: openai/gsm8k (main config, test split), 1319 items.
Answer: single number. Gold comes after "#### " in the answer field.

GPU-only (uses vLLM for throughput). Output CSV mirrors the columns
expected by domains/gsm8k/stream.py and calibrate_scores.py.

Usage:
    python domains/gsm8k/inference.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output results/gsm8k_inference.csv
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd


GSM8K_PROMPT = (
    "Solve the following math word problem step by step. Give your final "
    "answer as a single number after '#### '. Do not include units or "
    "commas in the final answer.\n\n"
    "Problem: {question}\n\n"
    "Solution:"
)


def extract_number(text: str) -> str:
    """Extract the final numeric answer from a response.

    Handles:
      * DeepSeek-R1 style <think>...</think> reasoning (stripped)
      * LaTeX \\boxed{N} answers (R1 distills emit these)
      * GSM8K canonical "#### N"
      * "The answer is N" / "Answer: N"
      * Fallback to last number in the (post-thinking) text
    """
    if not text:
        return ''
    # Strip DeepSeek-R1 thinking block (take what's AFTER </think>)
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    elif '<think>' in text:
        # Unclosed <think> (reasoning cut off); best we can do is use the tail
        text = text.split('<think>', 1)[-1]
    # 0. LAST \boxed{N}
    m = list(re.finditer(r'\\boxed\{\s*(-?[\d,]*\.?\d+)\s*\}', text))
    if m:
        return m[-1].group(1).replace(',', '').rstrip('.')
    # 1. Look for "#### <number>" (GSM8K canonical format)
    m = re.search(r'####\s*(-?[\d,]*\.?\d+)', text)
    if m:
        return m.group(1).replace(',', '').rstrip('.')
    # 2. LAST "The answer is <number>" / "answer: <number>"
    for pat in [r'[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*\$?(-?[\d,]*\.?\d+)',
                r'[Aa]nswer\s*[:\s]+\$?(-?[\d,]*\.?\d+)']:
        ms = list(re.finditer(pat, text))
        if ms:
            return ms[-1].group(1).replace(',', '').rstrip('.')
    # 3. Fallback: last number in the text
    nums = re.findall(r'-?[\d,]*\.?\d+', text)
    if nums:
        return nums[-1].replace(',', '').rstrip('.')
    return ''


def extract_gold(ans_field: str) -> str:
    """Extract the gold number from GSM8K's 'answer' field."""
    m = re.search(r'####\s*(-?[\d,]*\.?\d+)', ans_field)
    if m:
        return m.group(1).replace(',', '').rstrip('.')
    return ans_field.strip().replace(',', '')


def numbers_equal(a: str, b: str) -> bool:
    if not a or not b:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--output', default='results/gsm8k_inference.csv')
    parser.add_argument('--max_new_tokens', type=int, default=4096,
                        help='Default 4096 to accommodate reasoning-distill '
                             'models like DeepSeek-R1-Distill-Qwen-7B.')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--split', default='test')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--no_chat_template', action='store_true')
    args = parser.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading dataset: openai/gsm8k ({args.split})...")
    ds = load_dataset('openai/gsm8k', 'main', split=args.split)
    n = min(len(ds), args.limit) if args.limit > 0 else len(ds)
    print(f"  {n} items")

    print(f"Loading model {args.model}...")
    llm = LLM(model=args.model, dtype='bfloat16',
              gpu_memory_utilization=0.85, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sp = SamplingParams(n=1, temperature=args.temperature,
                        max_tokens=args.max_new_tokens, logprobs=1)

    def format_prompt(question):
        user_msg = GSM8K_PROMPT.format(question=question)
        if args.no_chat_template or tok.chat_template is None:
            return user_msg
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            return user_msg

    prompts = [format_prompt(ds[i]['question']) for i in range(n)]

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.1f}s  ({n/elapsed:.1f} items/s)")

    results = []
    for i, out in enumerate(outputs):
        gen = out.outputs[0]
        response = gen.text
        model_answer = extract_number(response)
        gold_answer = extract_gold(ds[i]['answer'])
        correct = int(numbers_equal(model_answer, gold_answer))

        # Mean logprob over generated tokens (fallback confidence)
        if gen.logprobs:
            lps = []
            for step in gen.logprobs:
                if step:
                    lps.append(list(step.values())[0].logprob)
            mean_lp = float(np.mean(lps)) if lps else -1.0
        else:
            mean_lp = -1.0

        results.append({
            'item_id': i,
            'question': ds[i]['question'][:300],
            'response': response[:600],
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
