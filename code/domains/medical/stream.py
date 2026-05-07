"""
Medical domain stream: MedQA-USMLE-4-options.

Loads pre-computed inference CSV (from inference.py) and presents
MCQ items as a deployment stream for CSA replay experiments.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from domains.base import DomainStream, DomainVerifier, RoundData


# Subject categories for feature encoding
MEDQA_SUBJECTS = [
    'anatomy', 'biochemistry', 'behavioral_science', 'cell_biology',
    'genetics', 'immunology', 'microbiology', 'pathology',
    'pharmacology', 'physiology', 'biostatistics', 'other',
]


class MedQAVerifier(DomainVerifier):
    """Exact-match MCQ verifier for MedQA."""

    def verify(self, model_answer: str, gold_answer: str) -> int:
        if not model_answer or not gold_answer:
            return 0
        return int(model_answer.strip().upper() == gold_answer.strip().upper())


def extract_mcq_answer(response: str) -> str:
    """Extract A/B/C/D answer from model response.

    Tries multiple patterns:
    1. "The answer is (X)" or "Answer: X"
    2. Last standalone letter A-D in the response
    3. First letter if response is very short
    """
    if not response:
        return ''

    # Pattern 1: explicit answer statement
    patterns = [
        r'[Tt]he\s+answer\s+is\s*[:\s]*\(?([A-Da-d])\)?',
        r'[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?',
        r'\b([A-Da-d])\s*[\.\)]\s*$',
    ]
    for pat in patterns:
        match = re.search(pat, response)
        if match:
            return match.group(1).upper()

    # Pattern 2: last standalone A-D
    matches = re.findall(r'\b([A-Da-d])\b', response)
    if matches:
        return matches[-1].upper()

    # Pattern 3: single character response
    response_stripped = response.strip()
    if len(response_stripped) == 1 and response_stripped.upper() in 'ABCD':
        return response_stripped.upper()

    return ''


class MedQAStream(DomainStream):
    """Stream of MedQA items from pre-computed inference CSV.

    Expected CSV columns:
        item_id, question, response, model_answer, gold_answer,
        correct, mean_logprob, question_length, response_length, subject
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.verifier = MedQAVerifier()
        self.has_calibrated = 'calibrated_score' in self.df.columns
        self._precompute_features()

    def _precompute_features(self):
        """Build feature vectors for all items."""
        df = self.df
        self.has_scores = 'max_score' in df.columns
        self.features_array = []

        for _, row in df.iterrows():
            if self.has_scores:
                feat_conf = row.get('max_score', 0.25)
                scores = [row.get(f'score_{l}', 0.25) for l in 'ABCD']
                total = sum(scores)
                normed = [s / total for s in scores] if total > 0 else scores
                feat_entropy = -sum(p * np.log(p + 1e-10) for p in normed) / np.log(4)
                feat_margin = sorted(normed)[-1] - sorted(normed)[-2] if len(normed) > 1 else 0
            else:
                logprob = row.get('mean_logprob', -1.0)
                feat_conf = np.clip((logprob + 5.0) / 5.0, 0, 1)
                feat_entropy = 0.5
                feat_margin = 0.0

            q_len = row.get('question_length', 200)
            feat_qlen = np.clip(q_len / 1000.0, 0, 1)

            subject = str(row.get('subject', 'other')).lower()
            subj_idx = MEDQA_SUBJECTS.index(subject) if subject in MEDQA_SUBJECTS else len(MEDQA_SUBJECTS) - 1
            feat_subj = subj_idx / len(MEDQA_SUBJECTS)

            self.features_array.append(
                np.array([feat_conf, feat_entropy, feat_margin, feat_qlen, feat_subj]))

        self.features_array = np.array(self.features_array)

    def __len__(self) -> int:
        return len(self.df)

    def get_round(self, t: int) -> RoundData:
        row = self.df.iloc[t]
        if self.has_calibrated:
            correct = int(row.get('scored_answer', '') == row.get('gold_answer', ''))
            score_hint = float(np.clip(row['calibrated_score'], 0.001, 0.999))
        elif self.has_scores:
            correct = int(row.get('scored_answer', '') == row.get('gold_answer', ''))
            score_hint = float(np.clip(1.0 - row.get('max_score', 0.25), 0.005, 0.995))
        else:
            correct = int(row['correct'])
            score_hint = None
        return RoundData(
            item_id=str(row.get('item_id', t)),
            features=self.features_array[t],
            V_t=correct,
            score_hint=score_hint,
            metadata={
                'subject': row.get('subject', ''),
                'gold_answer': row.get('gold_answer', ''),
            },
        )
