"""
MedNLI domain inference (clinical natural-language inference).

Dataset: presencesw/mednli. 1,422 test items. Each item is a
{premise, hypothesis, label} triple where the premise is a sentence
extracted from a MIMIC-III clinical note and the hypothesis is a
physician-authored claim. The model must decide:

    entailment    : premise logically supports the hypothesis
    contradiction : premise contradicts the hypothesis
    neutral       : premise is unrelated or inconclusive

High-stakes: mis-reading a clinical note can cause incorrect treatment
decisions. A "neutral" finding mistaken for "entailment" could lead to
acting on evidence that isn't really present.

Schema:
    sentence1  : premise (clinical note sentence)
    sentence2  : hypothesis
    gold_label : one of {'entailment', 'neutral', 'contradiction'}

Usage:
    python domains/mednli/inference.py \
        --model UbiquantAI/Fleming-R1-7B \
        --output results/mednli_inference.csv
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd


LABELS = ['entailment', 'contradiction', 'neutral']


MEDNLI_PROMPT = (
    "You are a clinical reasoning expert. Given a sentence from a "
    "patient's medical note (Premise) and a physician's claim "
    "(Hypothesis), decide whether the premise logically supports the "
    "hypothesis:\n"
    "  entailment    - the premise supports the hypothesis\n"
    "  contradiction - the premise contradicts the hypothesis\n"
    "  neutral       - the premise neither supports nor contradicts\n\n"
    "Think briefly, then end with exactly 'Final answer: entailment' "
    "OR 'Final answer: contradiction' OR 'Final answer: neutral'.\n\n"
    "Premise: {premise}\n\n"
    "Hypothesis: {hypothesis}\n\n"
    "Response:"
)


def extract_nli(text: str) -> str:
    if not text:
        return ''
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    for pat in [r'[Ff]inal\s+answer\s*:?\s*\(?(entailment|contradiction|neutral)\)?',
                r'[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\(?(entailment|contradiction|neutral)\)?',
                r'[Aa]nswer\s*:?\s*\(?(entailment|contradiction|neutral)\)?']:
        ms = list(re.finditer(pat, text, re.IGNORECASE))
        if ms:
            return ms[-1].group(1).lower()
    ms = list(re.finditer(
        r'\b(entailment|entails|contradiction|contradicts|neutral)\b',
        text, re.IGNORECASE))
    if ms:
        v = ms[-1].group(1).lower()
        if 'entail' in v:
            return 'entailment'
        if 'contradict' in v:
            return 'contradiction'
        return 'neutral'
    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='UbiquantAI/Fleming-R1-7B')
    parser.add_argument('--output', default='results/mednli_inference.csv')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--split', default='test')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--hf_path', default='presencesw/mednli')
    parser.add_argument('--no_chat_template', action='store_true')
    args = parser.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading dataset: {args.hf_path} ({args.split})...")
    ds = load_dataset(args.hf_path, split=args.split)
    n = min(len(ds), args.limit) if args.limit > 0 else len(ds)
    print(f"  {n} items")

    print(f"Loading model {args.model}...")
    llm = LLM(model=args.model, dtype='bfloat16',
              gpu_memory_utilization=0.85, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sp = SamplingParams(n=1, temperature=args.temperature,
                        max_tokens=args.max_new_tokens, logprobs=1)

    def _fmt(raw):
        if args.no_chat_template or tok.chat_template is None:
            return raw
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            return raw

    prompts, gold_answers = [], []
    for i in range(n):
        it = ds[i]
        raw = MEDNLI_PROMPT.format(
            premise=it['sentence1'].strip(),
            hypothesis=it['sentence2'].strip())
        prompts.append(_fmt(raw))
        gold_answers.append(str(it.get('gold_label', '')).lower().strip())

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.1f}s  ({n/elapsed:.1f} items/s)")

    results = []
    for i, out in enumerate(outputs):
        gen = out.outputs[0]
        response = gen.text
        model_answer = extract_nli(response)
        gold_answer = gold_answers[i]
        correct = int(model_answer == gold_answer)
        if gen.logprobs:
            lps = [list(s.values())[0].logprob for s in gen.logprobs if s]
            mean_lp = float(np.mean(lps)) if lps else -1.0
        else:
            mean_lp = -1.0
        item = ds[i]
        results.append({
            'item_id': i,
            'question': (item['sentence1'] + ' || ' + item['sentence2'])[:400],
            'response': response[:500],
            'model_answer': model_answer,
            'gold_answer': gold_answer,
            'correct': correct,
            'mean_logprob': mean_lp,
            'question_length': len(item['sentence1']) + len(item['sentence2']),
            'response_length': len(response),
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    acc = df['correct'].mean()
    by_class = df.groupby('gold_answer')['correct'].mean().to_dict()
    pred_dist = dict(df['model_answer'].value_counts())
    print(f"Done. n={n}, acc={acc:.1%}")
    print(f"  By gold class: {by_class}")
    print(f"  Pred distribution: {pred_dist}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
