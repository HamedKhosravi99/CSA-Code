"""
CUDA twin of live_arc_cuda.py for CaseHOLD with Saul-7B-Instruct.

CaseHOLD is a 5-option legal-reasoning MCQ from LexGLUE; the train split has
45000 items, plenty for `medium` (K=20) or `full` (K=25) modes. We reuse the
MCQ extractor / verifier from live_arc_cuda.py with valid_labels = A..E.

Usage:
    python live_casehold_cuda.py --mode medium \
        --model Equall/Saul-7B-Instruct-v1 \
        --out   results_live_casehold/medium.json \
        --adapters-dir results_live_casehold/adapters_cuda
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

if 'HF_HOME' not in os.environ:
    _scratch_cache = os.path.expanduser('~/scratch/.hf_cache')
    if os.path.isdir(_scratch_cache):
        os.environ['HF_HOME'] = _scratch_cache

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig, get_peft_model, PeftModel,
    prepare_model_for_kbit_training,
)

# ---------------------------------------------------------------------------
MODES = {
    'pilot':  dict(K=8,  n_per_round=100, K_sc=3, lora_iters=30, max_new=400),
    'medium': dict(K=20, n_per_round=100, K_sc=5, lora_iters=50, max_new=512),
    'full':   dict(K=25, n_per_round=100, K_sc=5, lora_iters=50, max_new=512),
}

LABELS = ['A', 'B', 'C', 'D', 'E']

SYSTEM_MSG = (
    "You are a legal expert. Read the case excerpt and pick the holding "
    "that best completes it from the five options. Think step by step, then "
    "give your final answer as 'The answer is (X)' where X is one of A, B, "
    "C, D, or E."
)
USER_TMPL = "Case excerpt:\n{context}\n\nWhich holding correctly completes the excerpt?\n{options}"


def _format_options(item):
    endings = item['endings']
    return '\n'.join(f"{LABELS[i]}. {endings[i]}" for i in range(len(endings)))


def format_prompt(item, tokenizer=None):
    opts = _format_options(item)
    user = USER_TMPL.format(context=item['context'], options=opts)
    combined = f"{SYSTEM_MSG}\n\n{user}"
    if tokenizer is not None and getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template(
            [{'role': 'user', 'content': combined}],
            tokenize=False, add_generation_prompt=True,
        )
    return f"{combined}\nAnswer:"


def extract_mcq(text: str, valid_labels: list[str]) -> str:
    if not text:
        return ''
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    label_cls = ''.join(re.escape(l) for l in valid_labels)
    label_cls_lower = label_cls.lower()
    cls = f"[{label_cls}{label_cls_lower}]"
    for pat in [
        rf'[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*\(?({cls})\)?',
        rf'[Aa]nswer\s*[:\s]+\(?({cls})\)?',
        rf'\\boxed\{{\s*\(?({cls})\)?\s*\}}',
    ]:
        m = list(re.finditer(pat, text))
        if m:
            return m[-1].group(1).upper()
    m = re.findall(rf'\b({cls})\b', text)
    return m[-1].upper() if m else ''


def gold_label(item) -> str:
    idx = item.get('label', -1)
    if isinstance(idx, int) and 0 <= idx < len(LABELS):
        return LABELS[idx]
    return ''


# ---------------------------------------------------------------------------
def load_base_model(model_id: str, device: str = 'cuda'):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    return model, tok


def attach_lora(model, num_layers_to_target: int = 4):
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        task_type='CAUSAL_LM',
    )
    return get_peft_model(model, lora_cfg)


# ---------------------------------------------------------------------------
@torch.inference_mode()
def sample_K(model, tokenizer, prompt: str, K_sc: int, max_new: int,
             temperature: float = 0.7) -> list[str]:
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       max_length=3072).to(model.device)
    input_ids = inputs['input_ids'].expand(K_sc, -1)
    attn = inputs['attention_mask'].expand(K_sc, -1)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = inputs['input_ids'].shape[1]
    return [tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True)
            for i in range(K_sc)]


def sc_score_and_verify(completions, gold, valid_labels):
    letters = [extract_mcq(c, valid_labels) for c in completions]
    nonempty = [x for x in letters if x]
    if not nonempty:
        return 1.0, 0, '', ''
    cnt = Counter(nonempty)
    maj, count = cnt.most_common(1)[0]
    agr = count / len(letters)
    V = int(maj == gold)
    best = ''
    for c, l in zip(completions, letters):
        if l == maj:
            best = c
            break
    return float(1 - agr), V, maj, best


# ---------------------------------------------------------------------------
def run_lora_round(model, tokenizer, sft_records, out_dir, iters, lr=1e-5,
                   batch_size=1, max_len=2048, grad_accum=4):
    os.makedirs(out_dir, exist_ok=True)

    def fmt(r):
        return r['prompt'] + '\n' + r['completion'] + tokenizer.eos_token

    texts = [fmt(r) for r in sft_records]
    ds = Dataset.from_dict({'text': texts})

    def tok_fn(ex):
        out = tokenizer(ex['text'], truncation=True, max_length=max_len,
                         padding='max_length')
        out['labels'] = out['input_ids'].copy()
        return out

    ds = ds.map(tok_fn, batched=True, remove_columns=['text'])

    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=iters,
        learning_rate=lr,
        logging_steps=max(1, iters // 4),
        save_strategy='no',
        bf16=True,
        optim='paged_adamw_8bit',
        report_to='none',
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    print(f'  LoRA step done in {dt:.0f}s ({iters} iters)')
    model.save_pretrained(out_dir)
    return out_dir


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pilot', 'medium', 'full'], default='medium')
    ap.add_argument('--model', default='Equall/Saul-7B-Instruct-v1')
    ap.add_argument('--out', default='results_live_casehold/medium.json')
    ap.add_argument('--adapters-dir', default='results_live_casehold/adapters_cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-lora', action='store_true')
    args = ap.parse_args()

    cfg = MODES[args.mode]
    print(f'[live_casehold_cuda] mode={args.mode}  cfg={cfg}')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    os.makedirs(args.adapters_dir, exist_ok=True)
    print(f'[live_casehold_cuda] torch.cuda={torch.cuda.is_available()}; '
          f'device_count={torch.cuda.device_count()}; '
          f'device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')

    print('[live_casehold_cuda] loading CaseHOLD train split ...')
    ds = load_dataset('lex_glue', 'case_hold', split='train')
    need = cfg['K'] * cfg['n_per_round']
    print(f'[live_casehold_cuda] CaseHOLD train size = {len(ds)}; need {need}')
    if len(ds) < need:
        raise ValueError(f'Need {need} items, have {len(ds)}.')
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ds))[:need]

    print(f'[live_casehold_cuda] loading base model {args.model} ...')
    base_model, tokenizer = load_base_model(args.model)
    model = attach_lora(base_model)
    model.print_trainable_parameters()

    valid_labels = LABELS

    stream = []
    round_acc = []
    for k in range(cfg['K']):
        t0 = time.time()
        print(f'\n=== Round {k}/{cfg["K"]} ===')
        idxs = perm[k * cfg['n_per_round']:(k + 1) * cfg['n_per_round']]
        sft_records = []
        correct = 0

        model.eval()
        for i, si in enumerate(idxs):
            item = ds[int(si)]
            gold = gold_label(item)
            prompt = format_prompt(item, tokenizer=tokenizer)
            t_item = time.time()
            cs = sample_K(model, tokenizer, prompt, cfg['K_sc'], cfg['max_new'])
            score, V, maj, best = sc_score_and_verify(cs, gold, valid_labels)
            correct += V
            stream.append({'round': k, 'item_id': int(si),
                           'score': score, 'V': V, 'majority': maj, 'gold': gold})
            if V == 1 and best:
                sft_records.append({'prompt': prompt, 'completion': best})
            dt = time.time() - t_item
            print(f'  r{k} item {i+1}/{cfg["n_per_round"]}: '
                  f'V={V} maj={maj} gold={gold} score={score:.2f} '
                  f'acc={correct/(i+1):.1%} [{dt:.1f}s]',
                  flush=True)

        ra = correct / cfg['n_per_round']
        round_acc.append(ra)
        print(f'  round {k}: acc={ra:.1%}, sft={len(sft_records)}, '
              f't={time.time()-t0:.0f}s')

        with open(args.out, 'w') as f:
            json.dump({'mode': args.mode, 'cfg': cfg, 'seed': args.seed,
                       'round_acc': round_acc, 'stream': stream}, f, indent=2)
        print(f'  wrote {args.out} ({len(stream)} items so far)')

        if args.skip_lora or k == cfg['K'] - 1 or not sft_records:
            continue
        out_adapter = os.path.join(args.adapters_dir, f'round{k+1}')
        model.train()
        run_lora_round(model, tokenizer, sft_records, out_adapter,
                       iters=cfg['lora_iters'])

    print(f'\n[live_casehold_cuda] done. {len(stream)} items, '
          f'round_acc={[f"{a:.0%}" for a in round_acc]}')


if __name__ == '__main__':
    main()
