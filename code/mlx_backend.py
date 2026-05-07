"""
MLX backend for Apple Silicon inference.

Provides model loading (4-bit quantized) and generation with logprobs
for running 7B models on Apple M4/M-series Macs with 16GB unified memory.

Requirements:
    pip install mlx-lm
"""

import os

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def is_mlx_available() -> bool:
    """Check if MLX backend is available (Apple Silicon)."""
    return HAS_MLX


def load_model(model_path: str):
    """Load an MLX model and tokenizer.

    Args:
        model_path: Path to local MLX model directory (e.g., 'models/medical-4bit')
                    or a HuggingFace model ID (will load at full precision).

    Returns:
        (model, tokenizer) tuple.
    """
    assert HAS_MLX, "mlx-lm not installed. Run: pip install mlx-lm"
    model, tokenizer = mlx_load(
        model_path,
        tokenizer_config={"trust_remote_code": True},
    )
    return model, tokenizer


def generate_with_logprobs(model, tokenizer, prompt: str,
                           max_tokens: int = 1024, temp: float = 0.3,
                           top_p: float = 0.95):
    """Generate text with per-token logprobs using MLX.

    Args:
        model: MLX model from load_model().
        tokenizer: Tokenizer from load_model().
        prompt: Input prompt string.
        max_tokens: Maximum tokens to generate.
        temp: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        (response_text, mean_logprob) tuple.
    """
    logprobs_list = []
    full_text = ""
    sampler = make_sampler(temp=temp, top_p=top_p)

    for resp in stream_generate(
        model, tokenizer, prompt,
        max_tokens=max_tokens, sampler=sampler,
    ):
        full_text += resp.text
        # resp.logprobs contains the logprob for the generated token
        if resp.logprobs is not None:
            try:
                lp = resp.logprobs
                if lp.ndim == 0:
                    logprobs_list.append(float(lp.item()))
                elif lp.ndim == 1:
                    logprobs_list.append(float(lp[resp.token].item()))
                else:
                    logprobs_list.append(float(lp.item()))
            except (IndexError, AttributeError):
                pass

    mean_logprob = (sum(logprobs_list) / len(logprobs_list)
                    if logprobs_list else -1.0)

    return full_text, mean_logprob


def generate_simple(model, tokenizer, prompt: str,
                    max_tokens: int = 1024, temp: float = 0.3):
    """Fallback: generate text without logprobs (uses mlx_lm.generate)."""
    from mlx_lm import generate
    text = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temp,
        verbose=False,
    )
    return text, -1.0
