"""
Shared data loading + styling for the three Risk+AR figure variants:
    build_riskar_splitpanel.py  (stacked Risk on top, AR on bottom per bench)
    build_riskar_pareto.py      (Risk vs AR phase trajectory per bench)
    build_riskar_twinaxis.py    (twin y-axis: Risk solid, AR dashed)
"""
from __future__ import annotations

import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VFILE = os.path.join(HERE, "paper_tables/_verified_numbers.json")
LIVEFILE = os.path.join(HERE, "..", "results_live_v2",
                        "live_replay_extended_baselines.json")
LIVEFILE_A05 = os.path.join(HERE, "..", "results_live_v2",
                             "live_replay_extended_baselines_alpha05.json")

with open(VFILE) as f:
    V = json.load(f)
with open(LIVEFILE) as f:
    LIVE = json.load(f)
with open(LIVEFILE_A05) as f:
    LIVE_A05 = json.load(f)

BENCH = ["medical", "pubmedqa", "tatqa", "mednli",
         "gsm8k", "headqa", "arc", "casehold"]
BENCH_DISP = {
    "medical": "MedQA", "pubmedqa": "PubMedQA",
    "tatqa": "TAT-QA", "mednli": "MedNLI",
    "gsm8k": "GSM8K", "headqa": "HEAD-QA",
    "arc": "ARC-Challenge", "casehold": "CaseHOLD",
}
PIVOT = {"medical": 0.20, "pubmedqa": 0.20, "tatqa": 0.20, "mednli": 0.20,
         "gsm8k": 0.05, "headqa": 0.20, "arc": 0.10, "casehold": 0.25}
ERR = {"medical": 31.5, "pubmedqa": 23.9, "tatqa": 25.7, "mednli": 21.0,
       "gsm8k": 5.0, "headqa": 26.0, "arc": 10.0, "casehold": 34.0}

METH = ["CSA-RLVR", "Always-Act", "Fixed-Threshold", "Naive-Tuning",
        "ACI", "SAOCP", "LTT", "CRC", "NEX-Conf", "Mohri-Conf"]
DISP = {"CSA-RLVR": "CSA (ours)", "Always-Act": "Always-Act",
        "Fixed-Threshold": "Fixed-Thr.", "Naive-Tuning": "Naive-Tun.",
        "ACI": "ACI", "SAOCP": "SAOCP",
        "LTT": "LTT", "CRC": "CRC", "NEX-Conf": "NEX-Conf",
        "Mohri-Conf": "ConfFact"}

# CSA: gold stars, black edge, thickest.  All others: thinner colored.
STYLE = {
    "CSA-RLVR":        dict(marker="*", color="#FFD700", lw=2.4, ms=12, edgecolor="black", lwmk=1.0, zorder=6),
    "Always-Act":      dict(marker="o", color="#d62728", lw=1.2, ms=5,  edgecolor=None,    lwmk=0.0, zorder=3),
    "Fixed-Threshold": dict(marker="s", color="#ff7f0e", lw=1.2, ms=5,  edgecolor=None,    lwmk=0.0, zorder=3),
    "Naive-Tuning":    dict(marker="^", color="#9467bd", lw=1.2, ms=5,  edgecolor=None,    lwmk=0.0, zorder=3),
    "ACI":             dict(marker="D", color="#2ca02c", lw=1.2, ms=5,  edgecolor=None,    lwmk=0.0, zorder=3),
    "SAOCP":           dict(marker="v", color="#8c8c18", lw=1.2, ms=5,  edgecolor=None,    lwmk=0.0, zorder=3),
    "LTT":             dict(marker="P", color="#8b0000", lw=1.2, ms=6,  edgecolor=None,    lwmk=0.0, zorder=3),
    "CRC":             dict(marker="X", color="#17becf", lw=1.2, ms=6,  edgecolor=None,    lwmk=0.0, zorder=3),
    "NEX-Conf":        dict(marker="p", color="#e377c2", lw=1.2, ms=6,  edgecolor=None,    lwmk=0.0, zorder=3),
    "Mohri-Conf":      dict(marker="h", color="#708090", lw=1.2, ms=6,  edgecolor=None,    lwmk=0.0, zorder=3),
}


def collect(bench: str, method: str):
    """Return (alphas, risks, ars) for one benchmark/method."""
    alphas, risks, ars = [], [], []
    for a_key in sorted(V["combined_main_plus_new"][bench].keys(), key=float):
        a_f = float(a_key)
        if a_f > 0.32:
            continue
        cell = V["combined_main_plus_new"][bench][a_key].get(method)
        if cell is None:
            continue
        alphas.append(a_f)
        risks.append(cell["risk"])
        ars.append(cell["ar"])
    return np.array(alphas), np.array(risks), np.array(ars)


def collect_live(method: str):
    """Return (alphas, risks, ars) for the live RLVR setting."""
    alphas = [0.05, 0.20, 0.40]
    keys = ["alpha_0.05", "alpha_0.2", "alpha_0.4"]
    srcs = [LIVE_A05, LIVE, LIVE]
    risks, ars = [], []
    for key, src in zip(keys, srcs):
        cell = src[key].get(method)
        if cell is None:
            return np.array([]), np.array([]), np.array([])
        risks.append(cell["final_risk_mean"])
        ars.append(cell["final_ar_mean"])
    return np.array(alphas), np.array(risks), np.array(ars)
