"""
ARC-Challenge domain inference (grade-school science MCQ).

Dataset: allenai/ai2_arc (ARC-Challenge config), 1172 test items.
Most questions have 4 choices (A-D), a few have 5 (A-E) or use 1-4.
We normalize gold to a letter in A-E.

Usage:
    python domains/arc/inference.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output results/arc_inference.csv
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd


ARC_PROMPT = (
    "Answer the following science question. Think briefly, then state "
    "your final answer as 'The answer is (X)' where X is one of the "
    "option letters.\n\n"
    "Question: {question}\n\n"
    "{options_str}\n\n"
    "Answer:"
)


def format_choices(choices: dict) -> tuple:
    """Return (options_str, normalized_label_map).
    Normalizes '1'/'2'/'3'/'4' labels to 'A'/'B'/'C'/'D'."""
    labels = choices['label']
    texts = choices['text']
    norm_map = {}
    for lbl in labels:
        if lbl in 'ABCDE':
            norm_map[lbl] = lbl
        elif lbl in '12345':
            norm_map[lbl] = 'ABCDE'[int(lbl) - 1]
        else:
            norm_map[lbl] = lbl.upper()
    lines = [f"{norm_map[lbl]}. {txt}" for lbl, txt in zip(labels, texts)]
    return '\n'.join(lines), norm_map


def extract_mcq_answer(response: str, valid_letters: str = 'ABCDE') -> str:
    if not response:
        return ''
    lower = valid_letters.lower()
    patterns = [
        rf'[Tt]he\s+answer\s+is\s*[:\s]*\(?([{valid_letters}{lower}])\)?',
        rf'[Aa]nswer\s*[:\s]+\(?([{valid_letters}{lower}])\)?',
    ]
    for pat in patterns:
        m = re.search(pat, response)
        if m:
            return m.group(1).upper()
    ms = re.findall(rf'\b([{valid_letters}{lower}])\b', response)
    return ms[-1].upper() if ms else ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--output', default='results/arc_inference.csv')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--no_chat_template', action='store_true')
    args = parser.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("Loading dataset: allenai/ai2_arc (ARC-Challenge, test)...")
    ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
    n = min(len(ds), args.limit) if args.limit > 0 else len(ds)
    print(f"  {n} items")

    print(f"Loading model {args.model}...")
    llm = LLM(model=args.model, dtype='bfloat16',
              gpu_memory_utilization=0.85, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sp = SamplingParams(n=1, temperature=args.temperature,
                        max_tokens=args.max_new_tokens, logprobs=1)

    def _fmt(user_msg):
        if args.no_chat_template or tok.chat_template is None:
            return user_msg
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            return user_msg

    prompts, gold_answers = [], []
    for i in range(n):
        item = ds[i]
        opts_str, norm_map = format_choices(item['choices'])
        raw = ARC_PROMPT.format(
            question=item['question'], options_str=opts_str)
        prompts.append(_fmt(raw))
        gold = str(item.get('answerKey', ''))
        gold_answers.append(norm_map.get(gold, gold.upper()))

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.1f}s  ({n/elapsed:.1f} items/s)")

    results = []
    for i, out in enumerate(outputs):
        gen = out.outputs[0]
        response = gen.text
        model_answer = extract_mcq_answer(response)
        gold_answer = gold_answers[i]
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
