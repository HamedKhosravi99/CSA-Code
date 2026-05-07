"""
Per-benchmark phase-budget plot for all 10 methods.

2x4 grid (8 benchmark panels). Each panel plots, as a function of the
target risk level alpha:

  - empirical Risk of every method (one line per method)
  - y = alpha reference line (black dotted)
  - the alpha-violation zone (above y=alpha, shaded red)
  - the pivotal alpha for the benchmark (gray vertical line)

Numbers come from paper_tables/_verified_numbers.json, the same source
the headline table pulls from.  No fabrication.
"""
from __future__ import annotations

import json
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VFILE = os.path.join(HERE, "paper_tables/_verified_numbers.json")
LIVEFILE = os.path.join(HERE, "..", "results_live_v2",
                        "live_replay_extended_baselines.json")
LIVEFILE_A05 = os.path.join(HERE, "..", "results_live_v2",
                             "live_replay_extended_baselines_alpha05.json")
OUT_PDF = os.path.join(HERE, "figures/phase_budget_allmethods.pdf")
OUT_PNG = os.path.join(HERE, "figures/phase_budget_allmethods.png")

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

# Methods + their display names (ConfFact per naming convention).
METH = ["CSA-RLVR", "Always-Act", "Fixed-Threshold", "Naive-Tuning",
        "ACI", "SAOCP", "LTT", "CRC", "NEX-Conf", "Mohri-Conf"]
DISP = {"CSA-RLVR": "CSA (ours)", "Always-Act": "Always-Act",
        "Fixed-Threshold": "Fixed-Thr.", "Naive-Tuning": "Naive-Tun.",
        "ACI": "ACI", "SAOCP": "SAOCP",
        "LTT": "LTT", "CRC": "CRC", "NEX-Conf": "NEX-Conf",
        "Mohri-Conf": "ConfFact"}
STYLE = {
    "CSA-RLVR":        dict(marker="*", color="#FFD700", lw=2.6, ms=13,
                             markeredgecolor="black", markeredgewidth=1.0,
                             zorder=6),
    "Always-Act":      dict(marker="o", color="#d62728", lw=1.3, ms=5,  zorder=3),
    "Fixed-Threshold": dict(marker="s", color="#ff7f0e", lw=1.3, ms=5,  zorder=3),
    "Naive-Tuning":    dict(marker="^", color="#9467bd", lw=1.3, ms=5,  zorder=3),
    "ACI":             dict(marker="D", color="#2ca02c", lw=1.3, ms=5,  zorder=3),
    "SAOCP":           dict(marker="v", color="#8c8c18", lw=1.3, ms=5,  zorder=3),
    "LTT":             dict(marker="P", color="#8b0000", lw=1.3, ms=6,  zorder=3),
    "CRC":             dict(marker="X", color="#17becf", lw=1.3, ms=6,  zorder=3),
    "NEX-Conf":        dict(marker="p", color="#e377c2", lw=1.3, ms=6,  zorder=3),
    "Mohri-Conf":      dict(marker="h", color="#708090", lw=1.3, ms=6,  zorder=3),
}


def collect(bench: str, method: str):
    """Return (alphas, risks) arrays for a given benchmark + method."""
    alphas, risks = [], []
    for a_key in sorted(V["combined_main_plus_new"][bench].keys(), key=float):
        a_f = float(a_key)
        if a_f > 0.32:
            continue
        cell = V["combined_main_plus_new"][bench][a_key].get(method)
        if cell is None:
            continue
        alphas.append(a_f)
        risks.append(cell["risk"])
    return np.array(alphas), np.array(risks)


def collect_live(method: str):
    """Return (alphas, risks) arrays for the live-RLVR + LoRA setting, all
    three alpha runs concatenated (0.05 from the alpha05 JSON, 0.20 and
    0.40 from the original extended JSON)."""
    alphas = [0.05, 0.20, 0.40]
    keys = ["alpha_0.05", "alpha_0.2", "alpha_0.4"]
    srcs = [LIVE_A05, LIVE, LIVE]
    risks = []
    for key, src in zip(keys, srcs):
        cell = src[key].get(method)
        if cell is None:
            return np.array([]), np.array([])
        risks.append(cell["final_risk_mean"])
    return np.array(alphas), np.array(risks)


# 3 x 3 = 9 panels: 8 benchmarks (top-left 2x4 block reorganized) + 1 live-RLVR.
fig, axes = plt.subplots(3, 3, figsize=(13.5, 11), sharey=True)
axes = axes.flatten()

for idx, b in enumerate(BENCH):
    ax = axes[idx]

    # Violation region (above y = alpha) shaded red
    xs = np.linspace(0, 0.32, 200)
    ax.fill_between(xs, xs, 1.0, color="red", alpha=0.05, zorder=0)

    # y = alpha reference line
    ax.plot(xs, xs, ":", color="black", linewidth=1.3, alpha=0.7,
            label=r"$y=\alpha$ budget" if idx == 0 else None, zorder=2)

    # Pivotal alpha
    a_piv = PIVOT[b]
    ax.axvline(a_piv, color="gray", linestyle="-", linewidth=0.8, alpha=0.45,
               zorder=1)

    # Each method's Risk curve
    for m in METH:
        alphas, risks = collect(b, m)
        if len(alphas) == 0:
            continue
        st = STYLE[m]
        ax.plot(alphas, risks, linestyle="-", **st,
                label=DISP[m] if idx == 0 else None)

    ax.set_title(f"{BENCH_DISP[b]}  (Err={ERR[b]:.1f}\\%,  "
                 rf"$\alpha^{{\star}}={a_piv:.2f}$)", fontsize=10)
    ax.set_xlim(0, 0.32)
    ax.set_ylim(-0.02, 0.85)
    ax.set_xlabel(r"Target risk budget $\alpha$", fontsize=9)
    if idx % 3 == 0:
        ax.set_ylabel("Empirical Risk", fontsize=10)
    ax.grid(alpha=0.25)

# -- Panel 8: live RLVR + online LoRA (all 10 methods) ---------------------
ax = axes[8]
xs_live = np.linspace(0, 0.50, 200)
ax.fill_between(xs_live, xs_live, 1.0, color="red", alpha=0.05, zorder=0)
ax.plot(xs_live, xs_live, ":", color="black", linewidth=1.3, alpha=0.7, zorder=2)
# Pivotal alpha for live RLVR = 0.40 per user.
ax.axvline(0.40, color="gray", linestyle="-", linewidth=0.8, alpha=0.45, zorder=1)
for m in METH:
    alphas, risks = collect_live(m)
    if len(alphas) == 0:
        continue
    st = STYLE[m]
    ax.plot(alphas, risks, linestyle="-", **st)
ax.set_title("Live RLVR + online LoRA\n"
             r"(Qwen2.5-Math-7B, Err$\approx$48\%, $\alpha^{\star}=0.40$)",
             fontsize=10)
ax.set_xlim(0, 0.50)
ax.set_ylim(-0.02, 0.85)
ax.set_xlabel(r"Target risk budget $\alpha$", fontsize=9)
ax.grid(alpha=0.25)

# Shared legend above, one row, with the y=alpha line entry appended.
handles = [
    plt.Line2D([0], [0], marker=STYLE[m]["marker"], color=STYLE[m]["color"],
               linestyle="-", linewidth=STYLE[m]["lw"],
               markersize=STYLE[m]["ms"], label=DISP[m])
    for m in METH
]
handles.append(
    plt.Line2D([0], [0], color="black", linestyle=":", linewidth=1.3,
               label=r"$y=\alpha$ budget")
)
fig.legend(handles=handles, ncol=6, loc="upper center",
           bbox_to_anchor=(0.5, 1.04), fontsize=9.5, frameon=False)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Wrote: {OUT_PDF}")
print(f"Wrote: {OUT_PNG}")
