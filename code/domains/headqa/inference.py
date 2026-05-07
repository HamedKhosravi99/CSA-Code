"""
HEAD-QA domain inference (Spanish Ministry of Health licensing exam, EN).

Dataset: EleutherAI/headqa, config 'en'. Test split has 2742 items; we
filter by category to the clinical-decision-support bundle
(pharmacology + medicine + nursery = 1375 items) by default.

Each item has 4 or 5 options. The gold answer is `ra` (1-indexed into
the `answers` list). We normalize to letters A-E.

Usage:
    python domains/headqa/inference.py \
        --model UbiquantAI/Fleming-R1-7B \
        --output results/headqa_inference.csv \
        --categories pharmacology medicine nursery
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd


HEADQA_PROMPT = (
    "You are a clinical expert taking a medical licensing examination. "
    "Read the question and select the correct answer. Think briefly, "
    "then state your final answer as 'The answer is (X)' where X is A, "
    "B, C, D, or E.\n\n"
    "Question: {question}\n\n"
    "{options_str}\n\n"
    "Answer:"
)

LETTERS = 'ABCDE'


def extract_mcq(text: str, letters: str = LETTERS) -> str:
    if not text:
        return ''
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    lower = letters.lower()
    for pat in [rf'[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*\(?([{letters}{lower}])\)?',
                rf'[Ff]inal\s+answer\s*[:\s]*\(?([{letters}{lower}])\)?',
                rf'[Aa]nswer\s*[:\s]+\(?([{letters}{lower}])\)?']:
        ms = list(re.finditer(pat, text))
        if ms:
            return ms[-1].group(1).upper()
    ms = re.findall(rf'\b([{letters}{lower}])\b', text)
    return ms[-1].upper() if ms else ''


def format_options(answers) -> str:
    # answers is a list of {'aid': int, 'atext': str}; aid is 1-indexed
    lines = []
    for ans in answers:
        aid = int(ans.get('aid', 0))
        if 1 <= aid <= len(LETTERS):
            lines.append(f"{LETTERS[aid - 1]}. {ans.get('atext', '')}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='UbiquantAI/Fleming-R1-7B')
    parser.add_argument('--output', default='results/headqa_inference.csv')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--split', default='test')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--categories', nargs='+',
                        default=['pharmacology', 'medicine', 'nursery'],
                        help='Filter to these categories (default: clinical bundle).')
    parser.add_argument('--no_chat_template', action='store_true')
    args = parser.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading dataset: EleutherAI/headqa (en, {args.split})...")
    ds = load_dataset('EleutherAI/headqa', 'en', split=args.split)
    if args.categories:
        cats = set(args.categories)
        ds = ds.filter(lambda x: x['category'] in cats)
        print(f"  filtered to categories {sorted(cats)}: {len(ds)} items")
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

    prompts, gold_answers, categories = [], [], []
    for i in range(n):
        it = ds[i]
        raw = HEADQA_PROMPT.format(
            question=it['qtext'], options_str=format_options(it['answers']))
        prompts.append(_fmt(raw))
        ra = int(it.get('ra', 0))
        gold_answers.append(LETTERS[ra - 1] if 1 <= ra <= len(LETTERS) else '')
        categories.append(it.get('category', ''))

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

        results.append({
            'item_id': i,
            'question': ds[i]['qtext'][:300],
            'response': response[:500],
            'model_answer': model_answer,
            'gold_answer': gold_answer,
            'correct': correct,
            'mean_logprob': mean_lp,
            'question_length': len(ds[i]['qtext']),
            'response_length': len(response),
            'category': categories[i],
            'n_options': len(ds[i]['answers']),
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    acc = df['correct'].mean()
    by_cat = df.groupby('category')['correct'].mean().to_dict()
    print(f"Done. n={n}, acc={acc:.1%}")
    print(f"  By category: {by_cat}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
