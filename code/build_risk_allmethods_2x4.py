"""
2 x 4 version of risk_allmethods_3x3.py.  8 benchmark panels, all 10
methods, legend sits above the grid (no dedicated legend panel).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from build_riskar_common import (
    BENCH, PIVOT, STYLE, DISP, collect,
)
from build_option_common import HERE, SHORT, apply_rc_defaults

OUT_PDF = os.path.join(HERE, "figures/risk_allmethods_2x4.pdf")
OUT_PNG = os.path.join(HERE, "figures/risk_allmethods_2x4.png")

apply_rc_defaults()

METH = ["CSA-RLVR",
        "LTT", "CRC", "Mohri-Conf", "NEX-Conf",
        "ACI", "SAOCP",
        "Fixed-Threshold", "Naive-Tuning", "Always-Act"]

X_MAX = 0.32
Y_MAX = 0.42


def line_kwargs(m):
    st = STYLE[m]
    is_csa = (m == "CSA-RLVR")
    return dict(
        color=st["color"],
        linewidth=2.0 if is_csa else 0.9,
        marker=st["marker"],
        markersize=6 if is_csa else 3.5,
        markerfacecolor=st["color"],
        markeredgecolor="black" if is_csa else st["color"],
        markeredgewidth=0.6 if is_csa else 0.0,
        alpha=0.95 if is_csa else 0.85,
        zorder=6 if is_csa else 3,
    )


fig, axes = plt.subplots(2, 4, figsize=(16, 6.8), sharey=True)
axes = axes.flatten()

for idx, b in enumerate(BENCH):
    ax = axes[idx]
    a_piv = PIVOT[b]

    xs = np.linspace(0, X_MAX, 200)
    ax.fill_between(xs, xs, Y_MAX, color="#d62728", alpha=0.05, zorder=0)
    ax.plot(xs, xs, linestyle=":", color="black", linewidth=1.0,
            alpha=0.7, zorder=2)
    ax.axvline(a_piv, color="#888888", linewidth=0.6, alpha=0.5, zorder=1)

    for m in METH:
        alphas, risks, ars = collect(b, m)
        if len(alphas) == 0:
            continue
        ax.plot(alphas, risks, **line_kwargs(m))

    ax.set_title(SHORT[b] + rf"  ($\alpha^{{\star}}{{=}}{a_piv:.2f}$)",
                 fontsize=13.5, pad=5, color="#000000", fontweight="bold")
    ax.set_xlim(0, X_MAX)
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_xticklabels(["0.05", "0.10", "0.15", "0.20", "0.25", "0.30"],
                       fontsize=11.5, color="#000000")
    ax.set_ylim(-0.02, Y_MAX)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.tick_params(axis="x", labelsize=11.5, labelcolor="#000000")
    ax.tick_params(axis="y", labelsize=12, labelcolor="#000000")
    ax.grid(alpha=0.22, linewidth=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color("#000000")
        ax.spines[sp].set_linewidth(0.9)

    if idx % 4 == 0:
        ax.set_ylabel("Risk", fontsize=14.5, color="#000000", fontweight="bold")
    if idx >= 4:
        ax.set_xlabel(r"$\alpha$", fontsize=14.5, color="#000000", fontweight="bold")

handles = []
for m in METH:
    st = STYLE[m]
    is_csa = (m == "CSA-RLVR")
    handles.append(plt.Line2D(
        [0], [0],
        color=st["color"],
        marker=st["marker"],
        linestyle="-",
        linewidth=2.0 if is_csa else 1.0,
        markerfacecolor=st["color"],
        markeredgecolor="black" if is_csa else st["color"],
        markeredgewidth=0.6 if is_csa else 0.0,
        markersize=9 if is_csa else 5,
        label=DISP[m],
    ))
handles.append(plt.Line2D([0], [0], color="black", linestyle=":",
                          linewidth=1.0, label=r"$y=\alpha$"))
handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#d62728",
                             alpha=0.15, edgecolor="none",
                             label=r"violation zone"))

leg = fig.legend(handles=handles, ncol=6, loc="upper center",
                 bbox_to_anchor=(0.5, 1.02), fontsize=13, frameon=False,
                 handlelength=2.2, columnspacing=1.5, handletextpad=0.7,
                 labelcolor="#000000")

plt.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Wrote: {OUT_PDF}")
print(f"Wrote: {OUT_PNG}")
