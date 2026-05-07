"""
CUDA twin of live_medqa.py for NVIDIA GPUs (PACE / H200 / H100 / A100).

Same live RLVR loop on MedQA with Fleming-R1-7B, but using
transformers + PEFT + bitsandbytes 4-bit quantization instead of MLX.

Usage (on PACE Phoenix via SLURM; see run_medqa_pace.sbatch):
    python live_medqa_cuda.py --mode full \
        --model UbiquantAI/Fleming-R1-7B \
        --out   results_live_medqa/full.json \
        --adapters-dir results_live_medqa/adapters
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

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
    'full':   dict(K=40, n_per_round=100, K_sc=5, lora_iters=50, max_new=512),
}

SYSTEM_MSG = (
    "You are a medical expert. Answer the USMLE question. Think step by "
    "step, then give your final answer as 'The answer is (X)' where X is "
    "A, B, C, or D."
)
USER_TMPL = (
    "Question: {question}\n\n{options}"
)


def format_prompt(item, tokenizer=None):
    """Format a MedQA item as a prompt.

    If a Llama-3 / Med42-style tokenizer with a chat template is passed,
    use its chat template (system + user).  Else fall back to a plain
    string concatenation.
    """
    opts = '\n'.join(f"{k}. {item['options'][k]}" for k in sorted(item['options']))
    user = USER_TMPL.format(question=item['question'], options=opts)
    if tokenizer is not None and getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_MSG},
             {'role': 'user',   'content': user}],
            tokenize=False, add_generation_prompt=True,
        )
    return f"{SYSTEM_MSG}\n\n{user}\nAnswer:"


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
def load_base_model(model_id: str, device: str = 'cuda'):
    """Load Fleming-R1-7B in 4-bit (nf4) for memory, bf16 compute."""
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
    """Attach a fresh LoRA adapter (all linear layers, r=16)."""
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
    """Generate K_sc completions from the same prompt in one batched call."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
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


def sc_score_and_verify(completions, gold):
    letters = [extract_mcq(c) for c in completions]
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
                   batch_size=1, max_len=1024, grad_accum=4):
    """One SFT round on verified-correct (prompt, completion) pairs.

    Trains the currently-attached PEFT adapter in-place, saves to out_dir.
    """
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
    ap.add_argument('--mode', choices=['pilot', 'medium', 'full'], default='pilot')
    # Default base model: Med42-v2-8B (Llama-3-based; ~72% on MedQA-USMLE;
    # PACE-compliant, unlike Qwen/DeepSeek which are on Georgia's ban list).
    ap.add_argument('--model', default='m42-health/Llama3-Med42-8B')
    ap.add_argument('--out', default='results_live_medqa/full.json')
    ap.add_argument('--adapters-dir', default='results_live_medqa/adapters_cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-lora', action='store_true')
    args = ap.parse_args()

    cfg = MODES[args.mode]
    print(f'[live_medqa_cuda] mode={args.mode}  cfg={cfg}')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    os.makedirs(args.adapters_dir, exist_ok=True)
    print(f'[live_medqa_cuda] torch.cuda={torch.cuda.is_available()}; '
          f'device_count={torch.cuda.device_count()}; '
          f'device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')

    # Dataset
    print('[live_medqa_cuda] loading MedQA train split ...')
    ds = load_dataset('GBaker/MedQA-USMLE-4-options', split='train')
    need = cfg['K'] * cfg['n_per_round']
    if len(ds) < need:
        raise ValueError(f'Need {need} items, have {len(ds)}.')
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ds))[:need]

    # Model + fresh LoRA
    print(f'[live_medqa_cuda] loading base model {args.model} ...')
    base_model, tokenizer = load_base_model(args.model)
    model = attach_lora(base_model)
    model.print_trainable_parameters()

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
            gold = gold_letter(item)
            prompt = format_prompt(item, tokenizer=tokenizer)
            t_item = time.time()
            cs = sample_K(model, tokenizer, prompt, cfg['K_sc'], cfg['max_new'])
            score, V, maj, best = sc_score_and_verify(cs, gold)
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
        print(f'  wrote {args.out} ({len(stream)} rounds)')

        if args.skip_lora or k == cfg['K'] - 1 or not sft_records:
            continue
        out_adapter = os.path.join(args.adapters_dir, f'round{k+1}')
        model.train()
        run_lora_round(model, tokenizer, sft_records, out_adapter,
                       iters=cfg['lora_iters'])

    print(f'\n[live_medqa_cuda] done. {len(stream)} rounds, '
          f'round_acc={[f"{a:.0%}" for a in round_acc]}')


if __name__ == '__main__':
    main()
