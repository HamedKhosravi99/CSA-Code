"""
Shared helpers for the five 5-method visualization variants
(build_option1_*.py through build_option5_*.py).

Selected methods (5 per panel instead of all 10):
    CSA-RLVR           -- our method
    SAOCP              -- best online baseline (lower risk than ACI on average)
    Mohri-Conf (ConfFact) -- best offline conformal (fewest refuse cells)
    NEX-Conf           -- non-exchangeable conformal
    Fixed-Threshold    -- best heuristic (highest AR among the simple ones)
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib.pyplot as plt
from build_riskar_common import (
    HERE, BENCH, BENCH_DISP, PIVOT, ERR, STYLE, collect,
)

METH_SEL = ["CSA-RLVR", "SAOCP", "Mohri-Conf", "NEX-Conf", "Fixed-Threshold"]
DISP_SEL = {"CSA-RLVR": "CSA (ours)", "SAOCP": "SAOCP",
            "Mohri-Conf": "ConfFact",  "NEX-Conf": "NEX-Conf",
            "Fixed-Threshold": "Fixed-Thr."}

ALPHA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
X_MIN, X_MAX = 0.02, 0.32

# Base-accuracy per benchmark for Coverage calculation.
# ERR in build_riskar_common is stored in PERCENT (e.g., 31.5 for MedQA),
# so divide by 100 before subtracting from 1.
ACC = {b: 1.0 - ERR[b] / 100.0 for b in BENCH}

# Shortened benchmark names for tight panel titles.
SHORT = {"medical": "MedQA", "pubmedqa": "PubMedQA",
         "tatqa": "TAT-QA", "mednli": "MedNLI",
         "gsm8k": "GSM8K", "headqa": "HEAD-QA",
         "arc": "ARC-Chal.", "casehold": "CaseHOLD"}


def collect_with_cov(bench: str, method: str):
    """(alphas, risks, ars, covs) for a (benchmark, method) on the canonical
    6-alpha grid.  Cov = AR*(1-Risk)/(1-Err), clipped to [0,1]."""
    alphas, risks, ars = collect(bench, method)
    covs = np.array([
        float(np.clip(a * (1 - r) / (ACC[bench] if ACC[bench] > 0 else 1.0),
                      0.0, 1.0))
        for a, r in zip(ars, risks)
    ]) if len(alphas) > 0 else np.array([])
    return alphas, risks, ars, covs


def apply_rc_defaults():
    plt.rcParams.update({
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.7,
        "xtick.color":    "#444444",
        "ytick.color":    "#444444",
        "font.family":    "DejaVu Sans",
    })


def method_handles(markersize_csa=7, markersize_base=5):
    return [
        plt.Line2D(
            [0], [0],
            color=STYLE[m]["color"],
            marker=STYLE[m]["marker"],
            linestyle="-",
            linewidth=1.8 if m == "CSA-RLVR" else 1.0,
            markerfacecolor=STYLE[m]["color"],
            markeredgecolor="black" if m == "CSA-RLVR" else (STYLE[m].get("edgecolor") or STYLE[m]["color"]),
            markeredgewidth=0.7 if m == "CSA-RLVR" else 0.3,
            markersize=markersize_csa if m == "CSA-RLVR" else markersize_base,
            label=DISP_SEL[m],
        )
        for m in METH_SEL
    ]


def line_kwargs(m, is_primary=True):
    st = STYLE[m]
    is_csa = (m == "CSA-RLVR")
    return dict(
        color=st["color"],
        linewidth=1.8 if is_csa else 0.9,
        marker=st["marker"],
        markersize=5 if is_csa else 3,
        markeredgecolor="black" if is_csa else (st.get("edgecolor") or st["color"]),
        markeredgewidth=0.5 if is_csa else 0.3,
        alpha=0.9 if is_csa else 0.85,
        zorder=6 if is_csa else 3,
    )
