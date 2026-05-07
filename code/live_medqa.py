"""
Live RLVR loop on MedQA (Fleming-R1-7B, 4-bit MLX, Apple Silicon).

For k = 0 .. K-1:
    1. Take the next slice of n_per_round MedQA-train problems.
    2. For each problem, sample K_sc=5 CoT-then-MCQ completions at
       temperature 0.7, extract the MCQ letter, majority-vote,
       compute confidence = agreement_fraction, verify against the
       gold answer (V_t = 1 iff majority_answer == gold).
    3. Record (score = 1 - confidence, V_t) into a growing stream.
    4. Write verified-correct (prompt, CoT) pairs to
       adapters/round{k}_train.jsonl.
    5. Run `mlx_lm.lora` for `lora_iters` iterations with 4 LoRA
       layers at learning rate 1e-5 on that jsonl; save adapter to
       adapters/round{k+1}/.
    6. Reload the model with that adapter; continue.

At the end: save stream.json with the full list of
(score, V_t, round_index, item_id) tuples. A separate replay script
(live_medqa_replay.py) runs CSA-RLVR + baselines on this stream.

Usage:
    # Pilot run (K=8 rounds, 100 items/round, K_sc=3, ~3h on M4 10-core):
    python live_medqa.py --mode pilot --out results_live_medqa/pilot.json

    # Medium (K=20, 100, K_sc=5, ~15h):
    python live_medqa.py --mode medium --out results_live_medqa/medium.json

    # Full (K=40, 100, K_sc=5, ~30h):
    python live_medqa.py --mode full --out results_live_medqa/full.json

Base model: models/medical-4bit (pre-quantized Fleming-R1-7B).
MedQA train split: GBaker/MedQA-USMLE-4-options (via HF datasets).

Outputs:
    results_live_medqa/adapters/round{k}/         (LoRA adapter per round)
    results_live_medqa/adapters/round{k}_train.jsonl
    results_live_medqa/<mode>.json                (genuine stream)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlx_backend import load_model, is_mlx_available
assert is_mlx_available(), "MLX required. Run: pip install mlx-lm datasets"

from mlx_lm import generate as mlx_generate
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Mode presets
# ---------------------------------------------------------------------------

MODES = {
    'pilot':  dict(K=8,  n_per_round=100, K_sc=3, lora_iters=30, max_tokens=400),
    'medium': dict(K=20, n_per_round=100, K_sc=5, lora_iters=50, max_tokens=512),
    'full':   dict(K=40, n_per_round=100, K_sc=5, lora_iters=50, max_tokens=512),
}

# ---------------------------------------------------------------------------
# MedQA helpers
# ---------------------------------------------------------------------------

PROMPT_TMPL = (
    "You are a medical expert. Answer the following USMLE question. "
    "Think step by step, then give your final answer as "
    "'The answer is (X)' where X is A, B, C, or D.\n\n"
    "Question: {question}\n\n"
    "{options}\n"
    "Answer:"
)


def format_prompt(item):
    opts_text = '\n'.join(f"{k}. {item['options'][k]}" for k in sorted(item['options']))
    return PROMPT_TMPL.format(question=item['question'], options=opts_text)


def extract_mcq(text: str) -> str:
    if not text:
        return ''
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    for pat in [r'[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?',
                r'[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?',
                r'\\boxed\{\s*\(?([A-Da-d])\)?\s*\}']:
        matches = list(re.finditer(pat, text))
        if matches:
            return matches[-1].group(1).upper()
    matches = re.findall(r'\b([A-Da-d])\b', text)
    return matches[-1].upper() if matches else ''


def gold_letter(item) -> str:
    gold_text = item['answer']
    for k, v in item['options'].items():
        if v == gold_text:
            return k
    return ''


# ---------------------------------------------------------------------------
# Sampling with K-wise self-consistency
# ---------------------------------------------------------------------------

def sample_K(model, tokenizer, prompt: str, K_sc: int, max_tokens: int,
             temperature: float = 0.7) -> list[str]:
    """Generate K_sc completions for one prompt."""
    outs = []
    for _ in range(K_sc):
        txt = mlx_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=None,  # defaults to temperature sampling; pass explicit temp if needed
        )
        outs.append(txt)
    return outs


def sc_score_and_verify(completions: list[str], gold: str) -> tuple[float, int, str, str]:
    """Return (score, V_t, majority_letter, best_completion).

    score = 1 - agreement_fraction (smaller = more confident)
    V_t = 1 iff majority_letter == gold
    best_completion = a completion whose answer equals the majority letter
    (used for SFT on verified-correct outputs when V_t=1)."""
    letters = [extract_mcq(c) for c in completions]
    nonempty = [x for x in letters if x]
    if not nonempty:
        return 1.0, 0, '', ''
    cnt = Counter(nonempty)
    majority, count = cnt.most_common(1)[0]
    agreement = count / len(letters)
    V_t = int(majority == gold)
    best = ''
    for c, l in zip(completions, letters):
        if l == majority:
            best = c
            break
    return float(1.0 - agreement), V_t, majority, best


# ---------------------------------------------------------------------------
# LoRA fine-tuning between rounds
# ---------------------------------------------------------------------------

def write_sft_jsonl(path: str, records: list[dict]):
    """Write prompt+completion JSONL for mlx_lm.lora.

    The format mlx_lm.lora expects is {"text": ...} per line, where each
    "text" entry is the full prompt + expected completion."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for r in records:
            # Format: prompt + '\n' + completion.  We train the model to
            # emit the completion given the prompt.
            full_text = r['prompt'] + '\n' + r['completion']
            f.write(json.dumps({'text': full_text}) + '\n')


def run_lora(model_path: str, sft_jsonl: str, out_adapter_dir: str,
             prior_adapter: str | None, iters: int, num_layers: int = 4,
             lr: float = 1e-5, batch_size: int = 1):
    """Run one round of mlx_lm.lora.  Returns out_adapter_dir on success."""
    # mlx_lm.lora expects a directory with train.jsonl and valid.jsonl.
    # Make a small 90/10 split of the input jsonl.
    data_dir = os.path.join(out_adapter_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    with open(sft_jsonl) as f:
        lines = [ln for ln in f if ln.strip()]
    n_val = max(1, len(lines) // 10)
    np.random.default_rng(0).shuffle(lines)
    with open(os.path.join(data_dir, 'train.jsonl'), 'w') as f:
        f.writelines(lines[n_val:])
    with open(os.path.join(data_dir, 'valid.jsonl'), 'w') as f:
        f.writelines(lines[:n_val])

    cmd = [
        sys.executable, '-m', 'mlx_lm.lora',
        '--model', model_path,
        '--train',
        '--data', data_dir,
        '--fine-tune-type', 'lora',
        '--num-layers', str(num_layers),
        '--batch-size', str(batch_size),
        '--iters', str(iters),
        '--learning-rate', str(lr),
        '--adapter-path', out_adapter_dir,
        '--save-every', str(iters),   # only final adapter
        '--steps-per-report', str(max(1, iters // 4)),
        '--seed', '42',
    ]
    if prior_adapter and os.path.exists(prior_adapter):
        adapter_file = os.path.join(prior_adapter, 'adapters.safetensors')
        if os.path.exists(adapter_file):
            cmd += ['--resume-adapter-file', adapter_file]
    print('  LoRA cmd:', ' '.join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=False)
    dt = time.time() - t0
    print(f'  LoRA done in {dt:.0f}s (returncode={res.returncode})')
    return out_adapter_dir if res.returncode == 0 else None


# ---------------------------------------------------------------------------
# Main live loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pilot', 'medium', 'full'], default='pilot')
    ap.add_argument('--model', default='models/medical-4bit')
    ap.add_argument('--out',   default='results_live_medqa/pilot.json')
    ap.add_argument('--adapters-dir', default='results_live_medqa/adapters')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-lora', action='store_true',
                    help='Generate stream without between-round LoRA updates.')
    args = ap.parse_args()

    cfg = MODES[args.mode]
    print(f'[live_medqa] mode={args.mode}  cfg={cfg}')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    os.makedirs(args.adapters_dir, exist_ok=True)

    # -- Load dataset (train split; reserve eval for the static benchmark) --
    print('[live_medqa] loading MedQA train split ...')
    ds = load_dataset('GBaker/MedQA-USMLE-4-options', split='train')
    n_total_needed = cfg['K'] * cfg['n_per_round']
    if len(ds) < n_total_needed:
        raise ValueError(f'Need {n_total_needed} train items, have {len(ds)}.')
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ds))[:n_total_needed]
    print(f'[live_medqa] selected {n_total_needed} of {len(ds)} train items.')

    # -- Load base model --
    print(f'[live_medqa] loading base model {args.model} ...')
    model, tokenizer = load_model(args.model)

    stream = []          # list of {round,item_id,score,V,majority,gold}
    current_adapter = None
    round_acc = []

    for k in range(cfg['K']):
        t_round = time.time()
        print(f'\n=== Round {k} of {cfg["K"]} ===')
        round_idx = perm[k * cfg['n_per_round']:(k + 1) * cfg['n_per_round']]
        sft_records = []
        correct = 0

        for i, src_idx in enumerate(round_idx):
            item = ds[int(src_idx)]
            gold = gold_letter(item)
            prompt = format_prompt(item)

            completions = sample_K(model, tokenizer, prompt,
                                   K_sc=cfg['K_sc'],
                                   max_tokens=cfg['max_tokens'])
            score, V_t, maj, best = sc_score_and_verify(completions, gold)
            correct += V_t

            stream.append({
                'round':    k,
                'item_id':  int(src_idx),
                'score':    score,
                'V':        V_t,
                'majority': maj,
                'gold':     gold,
            })
            if V_t == 1 and best:
                sft_records.append({'prompt': prompt, 'completion': best})

            if (i + 1) % 10 == 0:
                print(f'  round {k}: {i+1}/{cfg["n_per_round"]} '
                      f'acc={correct/(i+1):.1%}')

        round_accuracy = correct / cfg['n_per_round']
        round_acc.append(round_accuracy)
        dt = time.time() - t_round
        print(f'  round {k} done: acc={round_accuracy:.1%}, '
              f'sft_records={len(sft_records)}, time={dt:.0f}s')

        # Dump stream snapshot after each round so progress is not lost.
        with open(args.out, 'w') as f:
            json.dump({
                'mode':         args.mode,
                'cfg':          cfg,
                'seed':         args.seed,
                'round_acc':    round_acc,
                'stream':       stream,
            }, f, indent=2)
        print(f'  wrote {args.out} ({len(stream)} rounds so far)')

        # LoRA SFT between rounds (skip for last round).
        if args.skip_lora or k == cfg['K'] - 1:
            continue
        if not sft_records:
            print('  no verified-correct records; skipping LoRA')
            continue
        sft_jsonl = os.path.join(args.adapters_dir, f'round{k}_train.jsonl')
        write_sft_jsonl(sft_jsonl, sft_records)
        next_adapter = os.path.join(args.adapters_dir, f'round{k+1}')
        ok = run_lora(args.model, sft_jsonl, next_adapter,
                      prior_adapter=current_adapter,
                      iters=cfg['lora_iters'])
        if ok:
            print(f'  reloading model with adapter {next_adapter}')
            from mlx_lm import load as mlx_load
            model, tokenizer = mlx_load(args.model, adapter_path=next_adapter)
            current_adapter = next_adapter

    print(f'\n[live_medqa] done. total rounds: {len(stream)}')
    print(f'  round accuracies: {[f"{a:.0%}" for a in round_acc]}')
    print(f'  stream written to {args.out}')


if __name__ == '__main__':
    main()
