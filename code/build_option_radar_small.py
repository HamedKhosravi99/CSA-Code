"""
Small-multiples radar, 2 x 3 grid: one mini-radar per *non-trivial*
method (the 5 methods that have at least one safe/covering cell).
The sixth panel is a text/legend panel.

Vertex_b = Cov(m, b, alpha^*)   if Risk(m, b, alpha^*) <= alpha^*
         = 0                     otherwise

The 5 methods whose polygon is identically zero (SAOCP, ACI,
Fixed-Threshold, Naive-Tuning, Always-Act) are called out by name in
the legend panel; they do not get their own subplot.
"""
from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from build_option_common import (
    HERE, BENCH, PIVOT, STYLE, SHORT, ACC, apply_rc_defaults,
)

OUT_PDF = os.path.join(HERE, "figures/option_radar_small.pdf")
OUT_PNG = os.path.join(HERE, "figures/option_radar_small.png")

V = json.load(open(os.path.join(HERE, "paper_tables/_verified_numbers.json")))

# Only the five methods with at least one nonzero Safe-Coverage vertex.
# Ordered CSA first, then the other four by descending mean SafeCov.
METH = ["CSA-RLVR", "NEX-Conf", "Mohri-Conf", "LTT", "CRC"]
DISP = {"CSA-RLVR": "CSA (ours)",
        "NEX-Conf":  "NEX-Conf",
        "Mohri-Conf": "ConfFact",
        "LTT":       "LTT",
        "CRC":       "CRC"}

# Methods that are omitted because their Safe-Coverage polygon is
# identically zero (pathwise-violating on every benchmark).
ZERO_METH = ["SAOCP", "ACI", "Fixed-Threshold", "Naive-Tuning", "Always-Act"]


def pvcount(s):
    try: return int(s.split("/")[0])
    except: return 0


def a_key(a: float) -> str:
    return {0.05: "0.05", 0.1: "0.1", 0.15: "0.15",
            0.2:  "0.2",  0.25: "0.25", 0.3:  "0.3"}[round(a, 2)]


def safe_cov_vals(m):
    vals = []
    for b in BENCH:
        a_piv = a_key(PIVOT[b])
        d = V["combined_main_plus_new"][b].get(a_piv, {}).get(m)
        if d is None:
            vals.append(0.0); continue
        ar = d.get("ar", 0)
        risk = d.get("risk", 0)
        pv = pvcount(d.get("pv", "0/10"))
        cov = min(1.0, ar * (1 - risk) / ACC[b]) if ACC[b] > 0 else 0.0
        vals.append(0.0 if pv > 0 else float(cov))
    return vals


apply_rc_defaults()

N = len(BENCH)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles_closed = np.concatenate([angles, [angles[0]]])

# 2 x 3 figure: 5 polar panels, 6th panel is a text/legend panel.
fig = plt.figure(figsize=(14, 8.0))

positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

means = {m: float(np.mean(safe_cov_vals(m))) for m in METH}

for idx, m in enumerate(METH):
    r, c = positions[idx]
    ax = fig.add_subplot(2, 3, r * 3 + c + 1, projection="polar")
    vals = safe_cov_vals(m)
    vals_closed = vals + [vals[0]]
    st = STYLE[m]
    is_csa = (m == "CSA-RLVR")

    # Radial grid.
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([".25", ".50", ".75", "1.0"],
                       fontsize=7.5, color="#555555")
    ax.set_rlabel_position(72)

    # Perimeter labels.
    ax.set_xticks(angles)
    ax.set_xticklabels([SHORT[b] for b in BENCH],
                       fontsize=10, color="#000000")
    ax.tick_params(axis="x", pad=6, colors="#000000")

    # Reference ring at 0.5.
    ref = [0.5] * N + [0.5]
    ax.plot(angles_closed, ref, color="#cccccc", linewidth=0.5,
            linestyle="--", alpha=0.6, zorder=1)

    ax.grid(alpha=0.3, linewidth=0.4)
    ax.spines["polar"].set_color("#888888")
    ax.spines["polar"].set_linewidth(0.5)

    # Polygon.
    ax.fill(angles_closed, vals_closed, color=st["color"],
            alpha=0.35 if is_csa else 0.20, zorder=3)
    ax.plot(angles_closed, vals_closed, color=st["color"],
            linewidth=2.4 if is_csa else 1.6,
            marker=st["marker"],
            markersize=9 if is_csa else 6,
            markerfacecolor=st["color"],
            markeredgecolor="black",
            markeredgewidth=0.8 if is_csa else 0.4,
            alpha=0.95, zorder=5)

    title_color = "#000000"
    title_weight = "bold" if is_csa else "semibold"
    ax.set_title(f"{DISP[m]}   mean SafeCov = {means[m]:.2f}",
                 fontsize=11.5, pad=12,
                 color=title_color, fontweight=title_weight)


# Sixth panel: caption only (each mini-radar's title already labels the
# method, so no separate method legend is needed).
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis("off")

header = "Safe-Coverage radar per method"
body = (
    r"Vertex at benchmark $b$ = Cov at $\alpha^{\star}_b$ if safe, else 0."
    "\n"
    r"Cov = AR$\cdot(1-\mathrm{Risk})/(1-\mathrm{Err})$."
    "\n\n"
    "CSA (ours) is the only method with a nonzero vertex on\n"
    "every benchmark (full octagon).  The other four methods\n"
    "have partial coverage: zero-vertices mark benchmarks where\n"
    "they refuse or pathwise-violate at " r"$\alpha^{\star}_b$."
)
omitted = (
    "Omitted (identically-zero Safe-Coverage polygon):\n"
    "   SAOCP,  ACI,  Fixed-Threshold,  Naive-Tuning,  Always-Act"
)

ax6.text(0.03, 0.96, header, transform=ax6.transAxes,
         ha="left", va="top", fontsize=12.5, color="#000000",
         fontweight="bold")
ax6.text(0.03, 0.84, body, transform=ax6.transAxes,
         ha="left", va="top", fontsize=10.5, color="#000000",
         linespacing=1.5)
ax6.text(0.03, 0.22, omitted, transform=ax6.transAxes,
         ha="left", va="top", fontsize=10, color="#333333",
         linespacing=1.5, style="italic")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()

print(f"Wrote: {OUT_PDF}")
print(f"Wrote: {OUT_PNG}")
print()
print("Mean Safe-Coverage per plotted method:")
for m in METH:
    print(f"  {DISP[m]:14s}  {means[m]:.3f}")
print()
print("Omitted (all zero): " + ", ".join(ZERO_METH))
