"""
K-sample self-consistency scorer (GPU-only, vLLM-based).

Supports MCQ, yes/no, yes/no/maybe, and open-ended numeric / boxed-math
domains. For each item, samples K completions at temperature > 0,
extracts the final answer, and writes:
    sc_answer  : majority-vote answer (string)
    sc_score   : max_count / K       (agreement ratio = confidence)
    max_score  : alias of sc_score   (so calibrate_scores.py picks it up)
    sc_count_X : per-option counts   (only for discrete-option domains)

Sampling agreement is a far better confidence signal than quantized-LLM
logprobs and is what most RLVR / selective-prediction papers use.

Domains supported:
    medical (A-D), legal (Yes/No),
    gsm8k (numeric), math500 (boxed),
    arc (A-E), mmlu_pro (A-J), pubmedqa (yes/no/maybe)

Usage:
    python score_selfconsistency.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --csv    results/gsm8k_inference.csv \
        --domain gsm8k --k 5 \
        --output results/gsm8k_inference_sc.csv
"""

import argparse
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd


# ---------- Extractors ---------------------------------------------------

def extract_mcq(text: str, letters: str) -> str:
    if not text:
        return ''
    lower = letters.lower()
    for pat in [rf'[Tt]he\s+answer\s+is\s*[:\s]*\(?([{letters}{lower}])\)?',
                rf'[Aa]nswer\s*[:\s]+\(?([{letters}{lower}])\)?']:
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()
    ms = re.findall(rf'\b([{letters}{lower}])\b', text)
    return ms[-1].upper() if ms else ''


def extract_yesno(text: str) -> str:
    """Robust yes/no extractor. Prefers LAST explicit 'Final answer:' /
    'The answer is:' marker over first-word match, because CoT responses
    often emit a provisional token and revise later."""
    if not text:
        return ''
    # Priority 1: LAST explicit final-answer marker
    for pat in [r'[Ff]inal\s+answer\s*:?\s*\(?(yes|no)\)?',
                r'[Tt]he\s+answer\s+is\s*:?\s*\(?(yes|no)\)?',
                r'[Aa]nswer\s*:?\s*\(?(yes|no)\)?']:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).capitalize()
    # Priority 2: first-word (common terse-answer case)
    head = text.strip().lower().rstrip('.')[:40]
    if re.search(r'\byes\b', head):
        return 'Yes'
    if re.search(r'\bno\b', head):
        return 'No'
    # Priority 3: last yes/no anywhere
    matches = list(re.finditer(r'\b(yes|no)\b', text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).capitalize()
    return ''


def extract_yesno_maybe(text: str) -> str:
    if not text:
        return ''
    for pat in [r'[Tt]he\s+answer\s+is\s*:?\s*\(?(yes|no|maybe)\)?',
                r'[Aa]nswer\s*:?\s*\(?(yes|no|maybe)\)?']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    m = re.search(r'\b(yes|no|maybe)\b', text, re.IGNORECASE)
    return m.group(1).lower() if m else ''


def extract_number(text: str) -> str:
    if not text:
        return ''
    m = re.search(r'####\s*(-?[\d,]*\.?\d+)', text)
    if m:
        return m.group(1).replace(',', '').rstrip('.')
    for pat in [r'[Tt]he\s+answer\s+is\s*[:\s]*\$?(-?[\d,]*\.?\d+)',
                r'[Aa]nswer\s*[:\s]+\$?(-?[\d,]*\.?\d+)']:
        m = re.search(pat, text)
        if m:
            return m.group(1).replace(',', '').rstrip('.')
    nums = re.findall(r'-?[\d,]*\.?\d+', text)
    return nums[-1].replace(',', '').rstrip('.') if nums else ''


def extract_boxed(text: str) -> str:
    """Last \\boxed{...} content, handling nested braces."""
    if not text:
        return ''
    idx = text.rfind(r'\boxed{')
    if idx == -1:
        idx = text.rfind(r'\boxed {')
        if idx == -1:
            return ''
        start = idx + len(r'\boxed {')
    else:
        start = idx + len(r'\boxed{')
    depth, out = 1, []
    for c in text[start:]:
        if c == '{':
            depth += 1
            out.append(c)
        elif c == '}':
            depth -= 1
            if depth == 0:
                break
            out.append(c)
        else:
            out.append(c)
    s = ''.join(out).strip().replace(' ', '').rstrip('.')
    return s


def _extract_financial(text: str) -> str:
    """Extract + canonically normalize a financial numeric answer.

    Handles two kinds of equivalence in FinQA:
      (a) unit/format: '14.46', '14.46%', '14.460', '$14.46' -> same vote
      (b) pct-vs-decimal: '14.46' (i.e. 14.46%) and '0.1446' (same ratio
          expressed as decimal) -> same vote. We canonicalize by
          rescaling |v|<1 values to their x100 (percentage) form, which
          is the more common convention in CoT responses. Then we round
          to 3 significant figures so small rounding differences
          between samples cluster together.
    """
    n = extract_number(text)
    if not n:
        return ''
    try:
        v = float(n.replace(',', '').replace('%', '').replace('$', ''))
        if abs(v) < 1 and v != 0:
            v = v * 100.0  # canonicalize to pct form
        return f"{v:.3g}"  # 3 sig figs absorb minor rounding differences
    except ValueError:
        return n.strip()


EXTRACTORS = {
    'medical':     lambda s: extract_mcq(s, 'ABCD'),
    'legal':       extract_yesno,
    'arc':         lambda s: extract_mcq(s, 'ABCDE'),
    'mmlu_pro':    lambda s: extract_mcq(s, 'ABCDEFGHIJ'),
    'pubmedqa':    extract_yesno_maybe,
    'gsm8k':       extract_number,
    'math500':     extract_boxed,
    'financial':   _extract_financial,
    # Tier-2 high-stakes adds
    'casehold':    lambda s: extract_mcq(s, 'ABCDE'),
    'medmcqa':     lambda s: extract_mcq(s, 'ABCD'),
    'tatqa':       lambda s: _extract_tatqa(s),
    'cybermetric': lambda s: extract_mcq(s, 'ABCDE'),
    'ddxplus':     lambda s: extract_mcq(s, 'ABCD'),
    'lsat_lr':        lambda s: extract_mcq(s, 'ABCDE'),
    'cve':            extract_yesno,
    'mmlu_profmed':   lambda s: extract_mcq(s, 'ABCD'),
    'fomc':           lambda s: _extract_fomc(s),
    'fpb':            lambda s: _extract_fpb(s),
    'mednli':         lambda s: _extract_mednli(s),
    'headqa':         lambda s: extract_mcq(s, 'ABCDE'),
    'boolq':          lambda s: _extract_boolq_yesno(s),
    'mmlu_pro_bio':    lambda s: extract_mcq(s, 'ABCDEFGHIJ'),
    'mmlu_pro_health': lambda s: extract_mcq(s, 'ABCDEFGHIJ'),
    'mmlu_pro_law':    lambda s: extract_mcq(s, 'ABCDEFGHIJ'),
    'csqa':            lambda s: extract_mcq(s, 'ABCDE'),
    'hellaswag':       lambda s: extract_mcq(s, 'ABCD'),
    'medcalc':         extract_number,
    'winogrande':      lambda s: extract_mcq(s, 'AB'),
}


def _extract_boolq_yesno(text: str) -> str:
    if not text:
        return ''
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    for pat in [r'[Ff]inal\s+answer\s*:?\s*\(?(yes|no)\)?',
                r'[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\(?(yes|no)\)?',
                r'[Aa]nswer\s*:?\s*\(?(yes|no)\)?']:
        ms = list(re.finditer(pat, text, re.IGNORECASE))
        if ms:
            return ms[-1].group(1).lower()
    ms = list(re.finditer(r'\b(yes|no|true|false)\b', text, re.IGNORECASE))
    if ms:
        v = ms[-1].group(1).lower()
        return 'yes' if v in ('yes', 'true') else 'no'
    return ''


def _extract_mednli(text: str) -> str:
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


def _extract_fpb(text: str) -> str:
    if not text:
        return ''
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    for pat in [r'[Ff]inal\s+answer\s*:?\s*\(?(positive|negative|neutral)\)?',
                r'[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\(?(positive|negative|neutral)\)?',
                r'[Aa]nswer\s*:?\s*\(?(positive|negative|neutral)\)?']:
        ms = list(re.finditer(pat, text, re.IGNORECASE))
        if ms:
            return ms[-1].group(1).lower()
    ms = list(re.finditer(r'\b(positive|negative|neutral)\b', text, re.IGNORECASE))
    if ms:
        return ms[-1].group(1).lower()
    return ''


def _extract_fomc(text: str) -> str:
    if not text:
        return ''
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    for pat in [r'[Ff]inal\s+answer\s*:?\s*\(?(dovish|hawkish|neutral)\)?',
                r'[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\(?(dovish|hawkish|neutral)\)?',
                r'[Aa]nswer\s*:?\s*\(?(dovish|hawkish|neutral)\)?']:
        ms = list(re.finditer(pat, text, re.IGNORECASE))
        if ms:
            return ms[-1].group(1).lower()
    ms = list(re.finditer(r'\b(dovish|hawkish|neutral)\b', text, re.IGNORECASE))
    if ms:
        return ms[-1].group(1).lower()
    return ''

# For discrete-option domains, we record per-option counts.
# Open-ended (gsm8k, math500, tatqa) omit per-option columns.
DOMAIN_OPTIONS = {
    'medical':     list('ABCD'),
    'legal':       ['Yes', 'No'],
    'arc':         list('ABCDE'),
    'mmlu_pro':    list('ABCDEFGHIJ'),
    'pubmedqa':    ['yes', 'no', 'maybe'],
    'casehold':    list('ABCDE'),
    'medmcqa':     list('ABCD'),
    'cybermetric': list('ABCDE'),
    'ddxplus':     list('ABCD'),
    'lsat_lr':        list('ABCDE'),
    'cve':            ['Yes', 'No'],
    'mmlu_profmed':   list('ABCD'),
    'fomc':           ['dovish', 'hawkish', 'neutral'],
    'fpb':            ['positive', 'negative', 'neutral'],
    'mednli':         ['entailment', 'contradiction', 'neutral'],
}


def _extract_tatqa(text: str) -> str:
    """Extract either a number or a free-form span after 'Answer:'."""
    if not text:
        return ''
    for pat in [r'[Aa]nswer\s*:?\s*(.+?)(?:\n|$)',
                r'[Ff]inal\s+answer\s*:?\s*(.+?)(?:\n|$)']:
        m = re.search(pat, text)
        if m:
            v = m.group(1).strip().rstrip('.').strip()
            return v.replace(',', '').replace('$', '').replace('%', '')
    tail = text.strip().split('\n')[-1].rstrip('.').strip()
    return tail.replace(',', '').replace('$', '').replace('%', '')


# ---------- Prompt builders ----------------------------------------------

def _build_medical(n):
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split='test')
    prompts = []
    for i in range(n):
        it = ds[i]
        opts = '\n'.join(f"{k}. {it['options'][k]}"
                         for k in sorted(it['options'].keys()))
        prompts.append(
            "You are a medical expert. Answer the following medical question "
            "by selecting the correct option. Think step by step, then provide "
            "your final answer as 'The answer is (X)' where X is A, B, C, or D.\n\n"
            f"Question: {it['question']}\n\n{opts}\n\nAnswer:")
    return prompts


def _build_legal(df, n):
    if 'input_text' not in df.columns:
        raise ValueError("Legal CSV must contain 'input_text' column")
    return df['input_text'].iloc[:n].astype(str).tolist()


def _build_gsm8k(n):
    from datasets import load_dataset
    from domains.gsm8k.inference import GSM8K_PROMPT
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    return [GSM8K_PROMPT.format(question=ds[i]['question']) for i in range(n)]


def _build_math500(n):
    from datasets import load_dataset
    from domains.math500.inference import MATH_PROMPT
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    return [MATH_PROMPT.format(question=ds[i]['problem']) for i in range(n)]


def _build_arc(n):
    from datasets import load_dataset
    from domains.arc.inference import ARC_PROMPT, format_choices
    ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
    prompts = []
    for i in range(n):
        opts_str, _ = format_choices(ds[i]['choices'])
        prompts.append(ARC_PROMPT.format(
            question=ds[i]['question'], options_str=opts_str))
    return prompts


def _build_mmlu_pro(df, n, categories=None):
    from datasets import load_dataset
    from domains.mmlu_pro.inference import MMLU_PROMPT, format_options
    ds = load_dataset('TIGER-Lab/MMLU-Pro', split='test')
    if categories is None and df is not None and 'subject' in df.columns:
        categories = sorted(set(str(c) for c in df['subject']))
    if categories:
        cats = set(categories)
        ds = ds.filter(lambda x: x['category'] in cats)
    return [MMLU_PROMPT.format(
        question=ds[i]['question'],
        options_str=format_options(ds[i]['options']))
        for i in range(min(n, len(ds)))]


def _build_headqa(df, n, categories=None):
    from datasets import load_dataset
    from domains.headqa.inference import HEADQA_PROMPT, format_options
    ds = load_dataset('EleutherAI/headqa', 'en', split='test')
    if categories is None and df is not None and 'category' in df.columns:
        categories = sorted(set(str(c) for c in df['category']))
    if categories:
        cats = set(categories)
        ds = ds.filter(lambda x: x['category'] in cats)
    prompts = []
    for i in range(min(n, len(ds))):
        prompts.append(HEADQA_PROMPT.format(
            question=ds[i]['qtext'],
            options_str=format_options(ds[i]['answers'])))
    return prompts


def _build_boolq(n, split='validation'):
    from datasets import load_dataset
    from domains.boolq.inference import BOOLQ_PROMPT
    ds = load_dataset('google/boolq', split=split)
    prompts = []
    for i in range(min(n, len(ds))):
        prompts.append(BOOLQ_PROMPT.format(
            passage=str(ds[i]['passage'])[:2500],
            question=ds[i]['question']))
    return prompts


def _build_mmlu_pro_filtered(n, categories):
    from datasets import load_dataset
    from domains.mmlu_pro.inference import MMLU_PROMPT, format_options
    ds = load_dataset('TIGER-Lab/MMLU-Pro', split='test')
    cats = set(categories)
    ds = ds.filter(lambda x: x['category'] in cats)
    return [MMLU_PROMPT.format(
        question=ds[i]['question'],
        options_str=format_options(ds[i]['options']))
        for i in range(min(n, len(ds)))]


def _build_csqa(n, split='validation'):
    from datasets import load_dataset
    from domains.csqa.inference import CSQA_PROMPT, format_choices
    ds = load_dataset('tau/commonsense_qa', split=split)
    return [CSQA_PROMPT.format(
        question=ds[i]['question'],
        options_str=format_choices(ds[i]['choices']))
        for i in range(min(n, len(ds)))]


def _build_hellaswag(n, split='validation'):
    from datasets import load_dataset
    from domains.hellaswag.inference import HELLASWAG_PROMPT, format_endings
    ds = load_dataset('Rowan/hellaswag', split=split)
    return [HELLASWAG_PROMPT.format(
        context=ds[i]['ctx'],
        options_str=format_endings(ds[i]['endings']))
        for i in range(min(n, len(ds)))]


def _build_medcalc(n, split='test', exclude_dates=True):
    from datasets import load_dataset
    from domains.medcalc.inference import MEDCALC_PROMPT
    ds = load_dataset('ncbi/MedCalc-Bench-v1.0', split=split)
    if exclude_dates:
        ds = ds.filter(lambda x: x.get('Output Type', '').lower() != 'date')
    return [MEDCALC_PROMPT.format(
        note=str(ds[i]['Patient Note'])[:4000],
        question=ds[i]['Question'])
        for i in range(min(n, len(ds)))]


def _build_winogrande(n, split='validation', config='winogrande_xl'):
    from datasets import load_dataset
    from domains.winogrande.inference import WG_PROMPT
    ds = load_dataset('allenai/winogrande', config, split=split,
                      trust_remote_code=False)
    return [WG_PROMPT.format(
        sentence=ds[i]['sentence'],
        option1=ds[i]['option1'],
        option2=ds[i]['option2'])
        for i in range(min(n, len(ds)))]


def _build_pubmedqa(n):
    from datasets import load_dataset
    from domains.pubmedqa.inference import PUBMEDQA_PROMPT
    ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
    prompts = []
    for i in range(n):
        ctx = ds[i].get('context', {})
        if isinstance(ctx, dict):
            ctx_parts = ctx.get('contexts', []) or []
            ctx_str = '\n'.join(str(c) for c in ctx_parts)[:3000]
        else:
            ctx_str = str(ctx)[:3000]
        prompts.append(PUBMEDQA_PROMPT.format(
            question=ds[i]['question'], context=ctx_str))
    return prompts


def _build_financial(df: pd.DataFrame, n: int,
                     hf_path: str = 'ChanceFocus/flare-finqa',
                     split: str = 'test'):
    """Build financial prompts, same schema-agnostic handler used in
    domains/financial/inference.py (supports both original FinQA and
    flare-finqa schemas)."""
    from datasets import load_dataset
    from domains.financial.inference import format_finqa_prompt
    ds = load_dataset(hf_path, split=split)
    prompts = []
    for i in range(n):
        it = ds[i]
        if 'query' in it and 'text' in it:  # flare-finqa schema
            q = it.get('query', '')
            ctx = str(it.get('text', ''))[:4000]
            prompts.append(
                "You are a financial analyst. Read the following financial "
                "context (including any table) and answer the question. "
                "Show your reasoning step by step, then end with 'Final "
                f"answer: <number>'.\n\nContext:\n{ctx}\n\nQuestion: {q}\n\nResponse:")
        else:  # original FinQA
            qa = it.get('qa', it)
            question = qa.get('question', it.get('question', ''))
            table = it.get('table', [])
            pre_text = ' '.join(it.get('pre_text', []))
            post_text = ' '.join(it.get('post_text', []))
            prompts.append(format_finqa_prompt(question, table, pre_text, post_text))
    return prompts


def _build_mmlu_profmed(n, split='test'):
    from datasets import load_dataset
    from domains.mmlu_profmed.inference import MMLU_PROFMED_PROMPT, format_options
    ds = load_dataset('cais/mmlu', 'professional_medicine', split=split)
    prompts = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        prompts.append(MMLU_PROFMED_PROMPT.format(
            question=it['question'], options=format_options(it['choices'])))
    return prompts


def _build_fomc(n, split='test', hf_path='gtfintechlab/fomc_communication'):
    from datasets import load_dataset
    from domains.fomc.inference import FOMC_PROMPT
    ds = load_dataset(hf_path, split=split)
    prompts = []
    for i in range(min(n, len(ds))):
        prompts.append(FOMC_PROMPT.format(sentence=ds[i]['sentence']))
    return prompts


def _build_fpb(n, split='test', hf_path='ChanceFocus/flare-fpb'):
    from datasets import load_dataset
    from domains.fpb.inference import FPB_PROMPT
    ds = load_dataset(hf_path, split=split)
    prompts = []
    for i in range(min(n, len(ds))):
        prompts.append(FPB_PROMPT.format(sentence=ds[i]['text']))
    return prompts


def _build_mednli(n, split='test', hf_path='presencesw/mednli'):
    from datasets import load_dataset
    from domains.mednli.inference import MEDNLI_PROMPT
    ds = load_dataset(hf_path, split=split)
    prompts = []
    for i in range(min(n, len(ds))):
        prompts.append(MEDNLI_PROMPT.format(
            premise=ds[i]['sentence1'].strip(),
            hypothesis=ds[i]['sentence2'].strip()))
    return prompts


def _build_cve(df, n, hf_path='CyberNative/Code_Vulnerability_Security_DPO',
                split='train'):
    """Rebuild CVE prompts by re-doing the DPO->binary expansion so that
    CSV row order matches our expansion order exactly."""
    from datasets import load_dataset
    from domains.cve.inference import CVE_PROMPT
    ds = load_dataset(hf_path, split=split)
    rows = []
    for it in ds:
        for variant, code_key in [('chosen', 'chosen'), ('rejected', 'rejected')]:
            rows.append({
                'lang': str(it.get('lang', '')),
                'vulnerability': str(it.get('vulnerability', '')),
                'question': str(it.get('question', '')),
                'code': str(it.get(code_key, '')),
            })
    prompts = []
    for i in range(min(n, len(rows))):
        r = rows[i]
        prompts.append(CVE_PROMPT.format(
            vulnerability=r['vulnerability'][:500],
            question=r['question'][:500],
            lang=r['lang'], code=r['code'][:3000]))
    return prompts


def _build_lsat_lr(n, hf_path='hails/agieval-lsat-lr', split='test'):
    import re as _re
    from datasets import load_dataset
    from domains.lsat_lr.inference import LSAT_PROMPT
    ds = load_dataset(hf_path, split=split)
    prompts = []
    for i in range(min(n, len(ds))):
        q = str(ds[i].get('query', ''))
        q = _re.sub(r'A:\s*Among\s+A\s+through\s+E,?\s*the\s+answer\s+is\s*$',
                    '', q, flags=_re.IGNORECASE).rstrip()
        prompts.append(LSAT_PROMPT.format(query=q))
    return prompts


def _build_casehold(n, hf_path='coastalcph/lex_glue', hf_config='case_hold',
                     split='test'):
    from datasets import load_dataset
    from domains.casehold.inference import CASEHOLD_PROMPT, format_options
    ds = load_dataset(hf_path, hf_config, split=split)
    prompts = []
    for i in range(n):
        it = ds[i]
        cp = str(it.get('context') or it.get('citing_prompt') or '')[:3000]
        prompts.append(CASEHOLD_PROMPT.format(
            citing_prompt=cp, options_str=format_options(it)))
    return prompts


def _build_medmcqa(n):
    from datasets import load_dataset
    from domains.medmcqa.inference import MEDMCQA_PROMPT
    ds = load_dataset('openlifescienceai/medmcqa', split='validation')
    prompts = []
    for i in range(n):
        it = ds[i]
        prompts.append(MEDMCQA_PROMPT.format(
            question=it['question'],
            opa=it.get('opa', ''), opb=it.get('opb', ''),
            opc=it.get('opc', ''), opd=it.get('opd', '')))
    return prompts


def _build_tatqa(df: pd.DataFrame, n: int, hf_path='next-tat/tat-qa',
                  answer_types=None):
    """Rebuild TAT-QA prompts by flattening the nested document schema.
    The CSV row order (produced by domains/tatqa/inference.py) must
    exactly match our flatten order, so we use the same flatten logic
    here."""
    from datasets import load_dataset
    from domains.tatqa.inference import (TATQA_PROMPT, format_table,
                                          format_paragraphs)
    ds = load_dataset(hf_path, split='validation')
    rows = []
    for doc in ds:
        for q in doc.get('questions', []):
            rows.append({
                'table': doc.get('table'),
                'paragraphs': doc.get('paragraphs'),
                'question': q.get('question', ''),
                'answer_type': q.get('answer_type', 'unknown'),
            })
    if answer_types:
        allowed = set(answer_types)
        rows = [r for r in rows if r['answer_type'] in allowed]
    prompts = []
    for i in range(min(n, len(rows))):
        r = rows[i]
        prompts.append(TATQA_PROMPT.format(
            question=r['question'],
            table_md=format_table(r['table']),
            paragraphs=format_paragraphs(r['paragraphs'])))
    return prompts


def _build_cybermetric(df: pd.DataFrame, n: int,
                        hf_path='PeterChenNIST/CyberMetric'):
    from datasets import load_dataset
    from domains.cybermetric.inference import CYBER_PROMPT, extract_gold
    ds = load_dataset(hf_path, split='test')
    prompts = []
    for i in range(n):
        it = ds[i]
        opts_str, _ = extract_gold(it)
        q = it.get('question') or it.get('query') or ''
        prompts.append(CYBER_PROMPT.format(question=q, options_str=opts_str))
    return prompts


def _build_ddxplus(df: pd.DataFrame, n: int,
                    hf_path='aai530-group6/ddxplus', seed=42):
    """Rebuild the MCQ prompt for each DDXPlus item using deterministic
    per-item seed (same scheme as inference.py)."""
    import random
    from datasets import load_dataset
    from domains.ddxplus.inference import (DDX_PROMPT, safe_str,
                                            format_evidences, LETTERS)
    ds = load_dataset(hf_path, split='test')
    all_paths = sorted(set(str(ds[i].get('PATHOLOGY', '')).strip()
                           for i in range(len(ds))
                           if ds[i].get('PATHOLOGY')))
    all_paths = [p for p in all_paths if p]

    prompts = []
    for i in range(n):
        it = ds[i]
        gold = str(it.get('PATHOLOGY', '')).strip()
        if not gold:
            prompts.append('')
            continue
        distractors = [p for p in all_paths if p != gold]
        rng_i = random.Random(seed + i)
        sampled = rng_i.sample(distractors, k=min(3, len(distractors)))
        opts = sampled + [gold]
        rng_i.shuffle(opts)
        opts_str = '\n'.join(f"{LETTERS[j]}. {p}" for j, p in enumerate(opts))
        prompts.append(DDX_PROMPT.format(
            age=safe_str(it.get('AGE', '')),
            sex=safe_str(it.get('SEX', '')),
            initial_evidence=safe_str(it.get('INITIAL_EVIDENCE', '')),
            evidences=format_evidences(it.get('EVIDENCES', [])),
            options_str=opts_str))
    return prompts


def build_prompts(domain: str, df: pd.DataFrame, n: int,
                  tatqa_answer_types=None):
    if domain == 'medical':     return _build_medical(n)
    if domain == 'legal':       return _build_legal(df, n)
    if domain == 'gsm8k':       return _build_gsm8k(n)
    if domain == 'math500':     return _build_math500(n)
    if domain == 'arc':         return _build_arc(n)
    if domain == 'mmlu_pro':    return _build_mmlu_pro(df, n)
    if domain == 'mmlu_pro_bio':    return _build_mmlu_pro_filtered(n, ['biology'])
    if domain == 'mmlu_pro_health': return _build_mmlu_pro_filtered(n, ['health'])
    if domain == 'mmlu_pro_law':    return _build_mmlu_pro_filtered(n, ['law'])
    if domain == 'csqa':            return _build_csqa(n)
    if domain == 'hellaswag':       return _build_hellaswag(n)
    if domain == 'medcalc':         return _build_medcalc(n)
    if domain == 'winogrande':      return _build_winogrande(n)
    if domain == 'headqa':      return _build_headqa(df, n)
    if domain == 'boolq':       return _build_boolq(n)
    if domain == 'pubmedqa':    return _build_pubmedqa(n)
    if domain == 'financial':   return _build_financial(df, n)
    if domain == 'casehold':    return _build_casehold(n)
    if domain == 'medmcqa':     return _build_medmcqa(n)
    if domain == 'tatqa':       return _build_tatqa(df, n, answer_types=tatqa_answer_types)
    if domain == 'cybermetric': return _build_cybermetric(df, n)
    if domain == 'ddxplus':     return _build_ddxplus(df, n)
    if domain == 'lsat_lr':     return _build_lsat_lr(n)
    if domain == 'cve':         return _build_cve(df, n)
    if domain == 'mmlu_profmed': return _build_mmlu_profmed(n)
    if domain == 'fomc':        return _build_fomc(n)
    if domain == 'fpb':         return _build_fpb(n)
    if domain == 'mednli':      return _build_mednli(n)
    raise ValueError(f"Unsupported domain: {domain}")


# ---------- Main ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--csv', required=True,
                    help='Existing inference CSV to append columns to')
    ap.add_argument('--domain', required=True, choices=list(EXTRACTORS.keys()))
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--max_tokens', type=int, default=512)
    ap.add_argument('--output', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--quantization', default=None,
                    help="vLLM quantization mode ('awq', 'awq_marlin', "
                         "'gptq', 'fp8', ...). Required for AWQ/GPTQ models.")
    ap.add_argument('--max_model_len', type=int, default=None,
                    help="Override vLLM max model length (use for large/AWQ).")
    ap.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                    help="Raise to 0.92-0.95 for 72B-AWQ on 48GB GPUs.")
    ap.add_argument('--max_num_seqs', type=int, default=None,
                    help="Max concurrent sequences for KV cache; 16-32 for 72B.")
    ap.add_argument('--tatqa_answer_types', nargs='+', default=None,
                    help='For --domain tatqa: restrict to these answer types '
                         "(must match what inference.py was run with).")
    args = ap.parse_args()

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise SystemExit("vllm not installed. `pip install vllm` on GPU instance.")

    df = pd.read_csv(args.csv)
    n = min(len(df), args.limit) if args.limit > 0 else len(df)

    print(f"Loading {args.model} (quantization={args.quantization})...")
    llm_kwargs = dict(model=args.model, dtype='bfloat16',
                      gpu_memory_utilization=args.gpu_memory_utilization,
                      trust_remote_code=True)
    if args.quantization:
        llm_kwargs['quantization'] = args.quantization
    if args.max_model_len:
        llm_kwargs['max_model_len'] = args.max_model_len
    if args.max_num_seqs:
        llm_kwargs['max_num_seqs'] = args.max_num_seqs
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(n=args.k, temperature=args.temperature,
                        max_tokens=args.max_tokens)

    prompts = build_prompts(args.domain, df, n,
                            tatqa_answer_types=args.tatqa_answer_types)
    assert len(prompts) == n

    print(f"Sampling K={args.k} for {n} items...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({n*args.k/max(elapsed,1e-6):.0f} compl/s)")

    extract = EXTRACTORS[args.domain]
    discrete_opts = DOMAIN_OPTIONS.get(args.domain)  # None for open-ended

    sc_answers, sc_scores = [], []
    per_opt_counts = ({o: [] for o in discrete_opts}
                      if discrete_opts is not None else None)

    for out in outputs:
        votes = [extract(g.text) for g in out.outputs]
        if discrete_opts is not None:
            votes = [v for v in votes if v in discrete_opts]
        else:
            votes = [v for v in votes if v]
        ctr = Counter(votes)
        if not ctr:
            sc_answers.append('')
            sc_scores.append(0.0)
            if per_opt_counts is not None:
                for o in discrete_opts:
                    per_opt_counts[o].append(0)
            continue
        best, best_count = ctr.most_common(1)[0]
        sc_answers.append(best)
        sc_scores.append(best_count / args.k)
        if per_opt_counts is not None:
            for o in discrete_opts:
                per_opt_counts[o].append(ctr.get(o, 0))

    df = df.iloc[:n].copy()
    if per_opt_counts is not None:
        for o, counts in per_opt_counts.items():
            df[f'sc_count_{o}'] = counts
    df['sc_answer'] = sc_answers
    df['sc_score'] = sc_scores
    # Alias so calibrate_scores.py picks it up via max_score path.
    df['max_score'] = sc_scores
    if 'scored_answer' not in df.columns:
        df['scored_answer'] = sc_answers

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
