"""
Medical domain inference: Fleming-R1-7B on MedQA-USMLE.

Runs on Apple Silicon (MLX, 4-bit) or cloud GPU (CUDA).
Auto-detects the best available backend.

Usage (Apple M4):
    python setup_mlx_models.py --domain medical
    python domains/medical/inference.py \
        --model models/medical-4bit \
        --output results/medical_inference.csv

Usage (Cloud GPU):
    python domains/medical/inference.py \
        --model UbiquantAI/Fleming-R1-7B \
        --output results/medical_inference.csv
"""

import argparse
import os
import re
import sys
import time
import numpy as np
import pandas as pd

# Backend detection
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from mlx_backend import is_mlx_available, load_model, generate_with_logprobs
    HAS_MLX = is_mlx_available()
except ImportError:
    HAS_MLX = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def format_medqa_prompt(question: str, options: dict) -> str:
    """Format a MedQA question as a prompt for the model."""
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


def extract_mcq_answer(response: str) -> str:
    """Extract A/B/C/D from model response. Handles DeepSeek-R1 style
    <think>...</think> blocks by extracting from AFTER the thinking."""
    if not response:
        return ''
    # Strip DeepSeek-R1 thinking (take what's after </think>)
    if '</think>' in response:
        response = response.split('</think>', 1)[1]
    # Priority: LAST explicit answer marker (post-CoT correction wins)
    for pat in [r'[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?',
                r'[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?',
                r'\\boxed\{\s*\(?([A-Da-d])\)?\s*\}']:
        matches = list(re.finditer(pat, response))
        if matches:
            return matches[-1].group(1).upper()
    # Fallback: last standalone A-D
    matches = re.findall(r'\b([A-Da-d])\b', response)
    if matches:
        return matches[-1].upper()
    return ''


def run_inference_mlx(model_name: str, output_path: str,
                      max_new_tokens: int = 1024, split: str = 'test',
                      limit: int = 0):
    """Run inference using MLX backend (Apple Silicon)."""
    assert HAS_DATASETS, "datasets not installed. Run: pip install datasets"

    print(f"[MLX] Loading dataset: GBaker/MedQA-USMLE-4-options ({split})...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)

    print(f"[MLX] Loading model: {model_name}...")
    model, tokenizer = load_model(model_name)

    results = []
    n = min(len(ds), limit) if limit > 0 else len(ds)
    t0 = time.time()

    for idx in range(n):
        item = ds[idx]
        question = item['question']
        options = item['options']
        gold_text = item['answer']
        # Convert gold answer text to letter (A/B/C/D)
        gold_answer = ''
        for key in options:
            if options[key] == gold_text:
                gold_answer = key
                break

        prompt = format_medqa_prompt(question, options)

        try:
            response, mean_logprob = generate_with_logprobs(
                model, tokenizer, prompt,
                max_tokens=max_new_tokens, temp=0.3, top_p=0.95)
        except Exception as e:
            print(f"  Item {idx} generation error: {e}")
            response, mean_logprob = '', -1.0

        model_answer = extract_mcq_answer(response)
        correct = int(model_answer == gold_answer)

        meta = item.get('meta_info', 'other')
        subject = meta if meta else 'other'

        results.append({
            'item_id': idx,
            'question': question[:200],
            'response': response[:500],
            'model_answer': model_answer,
            'gold_answer': gold_answer,
            'correct': correct,
            'mean_logprob': mean_logprob,
            'question_length': len(question),
            'response_length': len(response),
            'subject': subject,
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = np.mean([r['correct'] for r in results])
            speed = (idx + 1) / elapsed
            eta = (n - idx - 1) / speed if speed > 0 else 0
            print(f"  {idx+1}/{n}  acc={acc:.1%}  "
                  f"{speed:.1f} items/s  ETA={eta/60:.0f}min")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)

    accuracy = df['correct'].mean()
    elapsed = time.time() - t0
    print(f"\nDone. {n} items, accuracy={accuracy:.1%}, time={elapsed:.1f}s")
    print(f"Saved to {output_path}")
    return df


def run_inference_cuda(model_name: str, output_path: str, batch_size: int = 8,
                       max_new_tokens: int = 1024, split: str = 'test',
                       limit: int = 0, use_chat_template: bool = True,
                       quantization: str = None, max_model_len: int = None,
                       gpu_memory_utilization: float = 0.85,
                       max_num_seqs: int = None):
    """Run inference using CUDA backend via vLLM (cloud GPU, fast).

    `use_chat_template=True` applies the tokenizer's chat template to
    each prompt. Required for reasoning-distilled / chat-tuned models
    (DeepSeek-R1 distills, Saul, etc.); harmless no-op for base models
    that lack a template.

    `quantization` enables vLLM quantization for large-model ablations
    (e.g. AWQ for Qwen2.5-72B-Instruct-AWQ). Accepted values: None,
    'awq', 'awq_marlin', 'gptq', 'gptq_marlin', 'fp8'.
    """
    assert HAS_DATASETS
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[CUDA/vLLM] Loading dataset: GBaker/MedQA-USMLE-4-options ({split})...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    n = min(len(ds), limit) if limit > 0 else len(ds)
    print(f"  {n} items")

    print(f"[CUDA/vLLM] Loading model: {model_name} "
          f"(quantization={quantization})...")
    llm_kwargs = dict(model=model_name, dtype='bfloat16',
                      gpu_memory_utilization=gpu_memory_utilization,
                      trust_remote_code=True)
    if quantization:
        llm_kwargs['quantization'] = quantization
    if max_model_len:
        llm_kwargs['max_model_len'] = max_model_len
    if max_num_seqs:
        llm_kwargs['max_num_seqs'] = max_num_seqs
    llm = LLM(**llm_kwargs)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    sp = SamplingParams(n=1, temperature=0.3, top_p=0.95,
                        max_tokens=max_new_tokens, logprobs=1)

    def _fmt(raw):
        user_msg = format_medqa_prompt(raw['question'], raw['options'])
        if not use_chat_template or tok.chat_template is None:
            return user_msg
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            return user_msg

    prompts = [_fmt(ds[i]) for i in range(n)]

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.1f}s  ({n/elapsed:.1f} items/s)")

    results = []
    for i, out in enumerate(outputs):
        gen = out.outputs[0]
        response = gen.text

        if gen.logprobs:
            lps = [list(s.values())[0].logprob for s in gen.logprobs if s]
            mean_lp = float(np.mean(lps)) if lps else -1.0
        else:
            mean_lp = -1.0

        model_answer = extract_mcq_answer(response)
        # In GBaker/MedQA-USMLE-4-options, `answer` is the option TEXT and
        # `answer_idx` is the letter. Prefer the letter; fall back to
        # resolving text against the options dict.
        gold_answer = str(ds[i].get('answer_idx', '')).strip().upper()
        if not gold_answer:
            gold_text = ds[i].get('answer', '')
            for key, val in (ds[i].get('options') or {}).items():
                if val == gold_text:
                    gold_answer = key
                    break
        correct = int(model_answer == gold_answer)
        subject = ds[i].get('meta_info') or 'other'

        results.append({
            'item_id': i,
            'question': ds[i]['question'][:200],
            'response': response[:500],
            'model_answer': model_answer,
            'gold_answer': gold_answer,
            'correct': correct,
            'mean_logprob': mean_lp,
            'question_length': len(ds[i]['question']),
            'response_length': len(response),
            'subject': subject,
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)

    accuracy = df['correct'].mean()
    print(f"\nDone. n={n}, accuracy={accuracy:.1%}, time={elapsed:.1f}s")
    print(f"Saved to {output_path}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Medical domain inference")
    parser.add_argument('--model', default='UbiquantAI/Fleming-R1-7B')
    parser.add_argument('--output', default='results/medical_inference.csv')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--split', default='test')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of items (0=all)')
    parser.add_argument('--backend', choices=['mlx', 'cuda', 'auto'],
                        default='auto', help='Inference backend')
    parser.add_argument('--quantization', default=None,
                        help="vLLM quantization mode (e.g. 'awq_marlin'). "
                             "Required for AWQ/GPTQ quantized models.")
    parser.add_argument('--max_model_len', type=int, default=None,
                        help="Override vLLM max model length to avoid OOM on "
                             "long contexts with large models.")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help="Fraction of GPU memory for vLLM engine. Raise "
                             "to 0.92-0.95 for 72B-AWQ on 48GB GPUs.")
    parser.add_argument('--max_num_seqs', type=int, default=None,
                        help="Max concurrent sequences (lowers KV cache "
                             "pressure; use 16-32 for large models).")
    args = parser.parse_args()

    # Auto-detect backend
    backend = args.backend
    if backend == 'auto':
        backend = 'mlx' if HAS_MLX else 'cuda'
    print(f"Using backend: {backend}")

    if backend == 'mlx':
        assert HAS_MLX, "MLX not available. Install: pip install mlx-lm"
        run_inference_mlx(args.model, args.output,
                          args.max_new_tokens, args.split, args.limit)
    else:
        run_inference_cuda(args.model, args.output, args.batch_size,
                           args.max_new_tokens, args.split, args.limit,
                           quantization=args.quantization,
                           max_model_len=args.max_model_len,
                           gpu_memory_utilization=args.gpu_memory_utilization,
                           max_num_seqs=args.max_num_seqs)
