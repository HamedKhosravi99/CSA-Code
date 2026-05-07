"""
Live-RLVR trajectory with action rate (AR).  2 x 2 grid:

    Top row:    running selective risk R^act_t over t   (ALL methods)
    Bottom row: running action rate  AR_t over t        (only methods
                whose running risk never exceeds alpha; i.e. PathV=0)

One column per alpha in {0.20, 0.40}.  Colours and markers match the
rest of the paper's figures; CSA is gold with thicker line.

We use the term *action rate* throughout (not "accept rate"): AR_t is
the fraction of rounds in which the controller decided to act,
AR_t = N_t^{act} / t.
"""
from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from build_riskar_common import HERE, METH, DISP, STYLE

LIVE = json.load(open(os.path.join(HERE, "..", "results_live_v2",
                                     "live_replay_extended_baselines.json")))
OUT_PDF = os.path.join(HERE, "figures/live_trajectory_with_ar.pdf")
OUT_PNG = os.path.join(HERE, "figures/live_trajectory_with_ar.png")

plt.rcParams.update({
    "axes.edgecolor":  "#222222", "axes.linewidth": 0.9,
    "xtick.color":     "#111111", "ytick.color":    "#111111",
    "axes.labelcolor": "#111111",
    "xtick.labelsize": 10.5,      "ytick.labelsize": 10.5,
    "font.family":     "DejaVu Sans",
})

# Which methods are "safe" (pathwise-valid) at each alpha?  A method is
# safe at alpha if its running-risk trajectory never exceeded alpha on
# any of the 20 replications (PathV = 0/20 post-burn-in).
def safe_methods(alpha_key):
    return [m for m in METH
            if LIVE[alpha_key][m]["pathwise_violations"] == 0]

fig, axes = plt.subplots(2, 2, figsize=(14, 7.0), sharex="col")
ALPHAS = [("alpha_0.2", 0.20), ("alpha_0.4", 0.40)]

# Marker-every spacing and per-method marker sizes for trajectory curves.
# Trajectories are 4000 points long; showing a marker every 400 points
# gives ~10 markers per curve, enough for visual identification without
# clutter.  CSA markers are offset by MARK_OFFSET so gold stars do not
# sit exactly on top of the other markers.
MARK_EVERY = 400
MARK_OFFSET = 200

def marker_kwargs(m, is_csa=False):
    """Marker styling for a method's trajectory, matching riskandar.pdf."""
    st = STYLE[m]
    return dict(
        marker=st["marker"],
        markersize=10 if is_csa else 6,
        markerfacecolor=st["color"],
        markeredgecolor=st["edgecolor"] or st["color"],
        markeredgewidth=1.0 if is_csa else 0.6,
    )


for col, (akey, alpha) in enumerate(ALPHAS):
    ax_r = axes[0, col]    # risk trajectory (top)
    ax_a = axes[1, col]    # action-rate trajectory (bottom)
    safe_set = set(safe_methods(akey))

    # -- Top: risk R^act_t -----------------------------------------------
    ax_r.axhspan(alpha, 1.0, color="#d62728", alpha=0.055, zorder=0)
    ax_r.axhline(alpha, color="black", linestyle="--", linewidth=1.2,
                 alpha=0.7, zorder=2)
    for i, m in enumerate(METH):
        cell = LIVE[akey][m]
        rc = np.asarray(cell.get("risk_curve", []))
        if len(rc) == 0: continue
        st = STYLE[m]
        is_csa = (m == "CSA-RLVR")
        # Offset markers per method so they don't pile up on the same x.
        mo = MARK_OFFSET if is_csa else int((i * 47) % MARK_EVERY)
        ax_r.plot(np.arange(1, len(rc) + 1), rc,
                  color=st["color"],
                  linewidth=(st["lw"] + 0.4) if is_csa else st["lw"],
                  alpha=0.9,
                  zorder=6 if is_csa else 3,
                  markevery=(mo, MARK_EVERY),
                  **marker_kwargs(m, is_csa=is_csa))
    ax_r.set_title(rf"$\alpha = {alpha:.2f}$", fontsize=13, pad=5,
                   color="#000000", fontweight="bold")
    ax_r.set_ylim(-0.01, max(0.58, alpha * 1.4))
    ax_r.set_xlim(0, 4000)
    if col == 0:
        ax_r.set_ylabel(r"Running selective risk $R^{\mathrm{act}}_t$",
                        fontsize=12, color="#000000")
    ax_r.grid(alpha=0.18, linewidth=0.5)
    for sp in ("top", "right"): ax_r.spines[sp].set_visible(False)

    # -- Bottom: action rate AR_t, ONLY safe methods ---------------------
    for i, m in enumerate(METH):
        if m not in safe_set:
            continue
        cell = LIVE[akey][m]
        ac = np.asarray(cell.get("ar_curve", []))
        if len(ac) == 0: continue
        st = STYLE[m]
        is_csa = (m == "CSA-RLVR")
        mo = MARK_OFFSET if is_csa else int((i * 47) % MARK_EVERY)
        ax_a.plot(np.arange(1, len(ac) + 1), ac,
                  color=st["color"],
                  linewidth=(st["lw"] + 0.4) if is_csa else st["lw"],
                  alpha=0.9,
                  zorder=6 if is_csa else 3,
                  markevery=(mo, MARK_EVERY),
                  **marker_kwargs(m, is_csa=is_csa))
    ax_a.set_xlim(0, 4000)
    ax_a.set_ylim(-0.02, 0.60)
    ax_a.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    if col == 0:
        ax_a.set_ylabel(r"Running action rate $\mathrm{AR}_t$",
                        fontsize=12, color="#000000")
    ax_a.set_xlabel(r"Stream step $t$", fontsize=12, color="#000000")
    ax_a.grid(alpha=0.18, linewidth=0.5)
    for sp in ("top", "right"): ax_a.spines[sp].set_visible(False)

    # Inset annotation: put it in the top-left where it will never sit
    # on a curve (all action-rate trajectories start at 0 and grow
    # monotonically to at most ~0.55).
    ax_a.text(0.02, 0.96,
              r"only pathwise-valid methods  (PathV$=0/20$) shown",
              transform=ax_a.transAxes, ha="left", va="top",
              fontsize=9.5, style="italic", color="#111111",
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                        boxstyle="round,pad=0.20"))

# Shared legend: each swatch shows the method's colour AND marker so
# the reader can match curves to legend entries by shape as well as
# hue (matches the riskandar.pdf convention).
handles = [
    plt.Line2D([0], [0],
               color=STYLE[m]["color"],
               marker=STYLE[m]["marker"],
               linestyle="-",
               linewidth=STYLE[m]["lw"] + (0.4 if m == "CSA-RLVR" else 0),
               markerfacecolor=STYLE[m]["color"],
               markeredgecolor=STYLE[m]["edgecolor"] or STYLE[m]["color"],
               markeredgewidth=1.0 if m == "CSA-RLVR" else 0.6,
               markersize=10 if m == "CSA-RLVR" else 7,
               label=DISP[m])
    for m in METH
]
handles.append(
    plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.2,
               label=r"$y = \alpha$ (budget)"))
fig.legend(handles=handles, ncol=6, loc="upper center",
           bbox_to_anchor=(0.5, 1.02), fontsize=11, frameon=False,
           handlelength=1.8, columnspacing=1.2, labelcolor="#000000")

fig.suptitle("Live RLVR + online LoRA on Qwen2.5-Math-7B "
             "($T=4000$ steps, 20 reps):  "
             "running risk (top) and action rate for pathwise-valid "
             "methods (bottom)",
             y=1.07, fontsize=12.5, color="#000000")
plt.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()

# Print which methods went into each AR panel
print(f"Wrote: {OUT_PDF}")
print(f"Wrote: {OUT_PNG}")
print()
for akey, alpha in ALPHAS:
    print(f"alpha={alpha:.2f} :: AR panel shows {len(safe_methods(akey))} "
          f"pathwise-valid methods: {safe_methods(akey)}")
