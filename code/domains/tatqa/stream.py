"""
TAT-QA domain stream: hybrid table + text financial QA.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from domains.base import DomainStream, RoundData


ANSWER_TYPES = ['span', 'spans', 'arithmetic', 'counting', 'unknown']


class TATQAStream(DomainStream):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.has_calibrated = 'calibrated_score' in self.df.columns
        self._precompute_features()

    def _precompute_features(self):
        df = self.df
        self.features_array = []
        for _, row in df.iterrows():
            logprob = float(row.get('mean_logprob', -1.0))
            feat_conf = np.clip((logprob + 5.0) / 5.0, 0, 1)
            q_len = float(row.get('question_length', 200))
            feat_qlen = np.clip(q_len / 500.0, 0, 1)
            r_len = float(row.get('response_length', 200))
            feat_rlen = np.clip(r_len / 1000.0, 0, 1)
            at = str(row.get('answer_type', 'unknown')).lower()
            at_idx = ANSWER_TYPES.index(at) if at in ANSWER_TYPES else len(ANSWER_TYPES) - 1
            feat_at = at_idx / max(len(ANSWER_TYPES), 1)
            self.features_array.append(
                np.array([feat_conf, feat_qlen, feat_rlen, feat_at]))
        self.features_array = np.array(self.features_array)

    def __len__(self) -> int:
        return len(self.df)

    def get_round(self, t: int) -> RoundData:
        row = self.df.iloc[t]
        if self.has_calibrated:
            score_hint = float(np.clip(row['calibrated_score'], 0.001, 0.999))
        else:
            logprob = float(row.get('mean_logprob', -1.0))
            score_hint = float(np.clip(1.0 - (logprob + 5.0) / 5.0, 0.005, 0.995))
        return RoundData(
            item_id=str(row.get('item_id', t)),
            features=self.features_array[t],
            V_t=int(row['correct']),
            score_hint=score_hint,
            metadata={
                'gold_answer': row.get('gold_answer', ''),
                'answer_type': row.get('answer_type', ''),
            },
        )
