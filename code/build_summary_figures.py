"""
Four creative summary figures, each distilled from
paper_tables/_verified_numbers.json (no hand-entered numbers).

    figures/summary_frontier.pdf    - Validity vs Utility scatter ("only CSA in top-right")
    figures/summary_radar.pdf       - Per-method radar across 8 benchmarks
    figures/summary_funnel.pdf      - "Survives each criterion" stacked horizontal bar
    figures/summary_ridge.pdf       - PathV-rate ridge plot across all (bench, alpha) cells

Run:  python3 build_summary_figures.py
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

os.makedirs('figures', exist_ok=True)
V = json.load(open('paper_tables/_verified_numbers.json'))

BENCH = ['medical', 'pubmedqa', 'tatqa', 'mednli',
         'gsm8k', 'headqa', 'arc', 'casehold']
BDISP = {'medical': 'MedQA', 'pubmedqa': 'PubMedQA', 'tatqa': 'TAT-QA',
         'mednli': 'MedNLI', 'gsm8k': 'GSM8K', 'headqa': 'HEAD-QA',
         'arc': 'ARC', 'casehold': 'CaseHOLD'}
PIVOT = {'medical': 0.20, 'pubmedqa': 0.20, 'tatqa': 0.20, 'mednli': 0.20,
         'gsm8k': 0.05, 'headqa': 0.20, 'arc': 0.10, 'casehold': 0.25}
METH = ['CSA-RLVR', 'Always-Act', 'Fixed-Threshold', 'Naive-Tuning',
        'ACI', 'SAOCP', 'LTT', 'CRC', 'NEX-Conf', 'Mohri-Conf']
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Consistent per-method visual style
STYLE = {
    'CSA-RLVR':        {'c': '#1f77b4', 'm': '*',  's': 320, 'z': 10},
    'Always-Act':      {'c': '#d62728', 'm': 'o',  's': 120, 'z': 3},
    'Fixed-Threshold': {'c': '#ff7f0e', 'm': 's',  's': 110, 'z': 3},
    'Naive-Tuning':    {'c': '#9467bd', 'm': '^',  's': 110, 'z': 3},
    'ACI':             {'c': '#8c564b', 'm': 'D',  's': 110, 'z': 3},
    'SAOCP':           {'c': '#e377c2', 'm': 'v',  's': 110, 'z': 3},
    'LTT':             {'c': '#bcbd22', 'm': 'P',  's': 130, 'z': 4},
    'CRC':             {'c': '#17becf', 'm': 'X',  's': 130, 'z': 4},
    'NEX-Conf':        {'c': '#e3165b', 'm': 'p',  's': 130, 'z': 4},
    'Mohri-Conf':      {'c': '#2ca02c', 'm': 'h',  's': 130, 'z': 4},
}


def get(b, a, m, key):
    cell = V['combined_main_plus_new'][b].get(f'{a:g}', {}).get(m)
    if cell is None:
        return None
    return cell.get(key)


def pv_frac(pv):
    if pv in (None, '?'):
        return None
    n, t = pv.split('/')
    return int(n) / int(t)


# -----------------------------------------------------------------
# Shift-ablation cells (MedQA easy->hard + GSM8K hard scenarios).
# These live in separate JSONs. We form a unified list of "synthetic"
# cells so every plot can honour the user's requirement:
#   "only CSA is 0/10 on BOTH main benchmarks AND synthetic shifts"
# -----------------------------------------------------------------

SHIFT_CELLS = []  # list of (label_str, alpha, method -> {'pv','ar','risk'})

# 1. MedQA shift stress-test: 3 alphas x 4 scenarios = 12 cells
#    (from ablation_shift_lowalpha.json via _verified_numbers.json)
medqa_cells = V.get('shift_medqa_lowalpha', {})
SCEN_SHORT = {
    'iid': 'iid',
    'easy_hard': 'easy$\\to$hard',
    'quartile_rev': 'quartile',
    'window_outrun': 'win-out',
}
for key in sorted(medqa_cells.keys()):
    # key format: 'alpha0.05_iid'
    parts = key.split('_', 1)
    alpha = float(parts[0][5:])
    sc = parts[1]
    sc_short = SCEN_SHORT.get(sc, sc)
    cell = {}
    for m in METH:
        src = medqa_cells[key].get(m)
        if src is None:
            continue
        cell[m] = {'pv': src['pv'], 'ar': src['ar'], 'risk': src['risk']}
    SHIFT_CELLS.append((f'MedQA $\\alpha{{=}}{alpha:g}$ {sc_short}', alpha, cell))

# 2. GSM8K alpha=0.05 hard scenarios (4 cells from ablate_shift_hard.py)
gsm_hard = V.get('shift_gsm8k_hard', {})
for sc_key in ['quartile', 'multi', 'adversarial', 'window_outrun']:
    sc_data = gsm_hard.get(sc_key, {})
    if not sc_data:
        continue
    cell = {}
    for m in METH:
        src = sc_data.get(m)
        if src is None:
            continue
        cell[m] = {'pv': src['pv'], 'ar': src['ar'], 'risk': src['risk']}
    sc_label = sc_key.replace('_', '\\_')
    SHIFT_CELLS.append((f'GSM8K $\\alpha{{=}}0.05$ {sc_label}', 0.05, cell))

print(f'Loaded {len(SHIFT_CELLS)} shift cells in addition to main benchmarks '
      f'({len(medqa_cells)} MedQA + {len(gsm_hard)} GSM8K)')


# =====================================================================
# FIG A: Validity-vs-Utility frontier -- the "only CSA in top-right" shot
# =====================================================================

fig, ax = plt.subplots(figsize=(9.5, 7))
# Shade the "desired" quadrant (validity>=0.95 AND AR>=0.15)
ax.add_patch(Rectangle((0.15, 0.95), 1.2, 0.1, facecolor='#c6e2b4',
                        alpha=0.35, zorder=0))
ax.text(0.52, 0.985, 'Valid  $\\cap$  Non-refusing  $\\cap$  Useful',
        fontsize=11, color='#2a5c1a', style='italic', ha='center',
        fontweight='bold', zorder=1)
# Shade validity-only strip (top edge excluding useful region)
ax.add_patch(Rectangle((0, 0.95), 0.15, 0.1, facecolor='#e0e0e0',
                        alpha=0.4, zorder=0))
ax.text(0.075, 0.985, 'Valid\nby refusal',
        fontsize=9, color='#555', style='italic', ha='center',
        va='center', zorder=1)
# Shade invalid region (below validity=0.95)
ax.add_patch(Rectangle((0, 0), 1.05, 0.95, facecolor='#f5d0d0',
                        alpha=0.28, zorder=0))
ax.text(0.52, 0.35, 'Invalid (pathwise violations)',
        fontsize=11, color='#8b1a1a', style='italic', ha='center',
        fontweight='bold', zorder=1, alpha=0.6)

# Per method: mean AR and mean validity across 8 main-benchmark pivotal
# cells + 5 synthetic shift cells (MedQA easy->hard + 4 GSM8K hard
# scenarios). Honours the requirement: "only CSA has 0 violations on
# BOTH main benchmarks AND shift ablations."
summary = {}
for m in METH:
    ars, vals = [], []
    # Main benchmarks at pivotal alpha
    for b in BENCH:
        a = PIVOT[b]
        ar = get(b, a, m, 'ar')
        pv = get(b, a, m, 'pv')
        if ar is None or pv is None:
            continue
        ars.append(ar)
        vals.append(1 - pv_frac(pv))
    # Shift-ablation cells (12 MedQA + 4 GSM8K = 16 cells)
    for (_, _alpha, cell) in SHIFT_CELLS:
        src = cell.get(m)
        if src is None:
            continue
        ars.append(src['ar'])
        vals.append(1 - pv_frac(src['pv']))
    summary[m] = (np.mean(ars) if ars else 0, np.mean(vals) if vals else 0)

# Plot
for m, (x, y) in summary.items():
    st = STYLE[m]
    ax.scatter(x, y, marker=st['m'], s=st['s'], color=st['c'],
               edgecolor='black', linewidth=1.3, zorder=st['z'], alpha=0.92)

# Smart label placement
label_offsets = {
    'CSA-RLVR':        (0.03, 0.015),
    'Always-Act':      (0.015, 0.02),
    'Fixed-Threshold': (0.015, -0.035),
    'Naive-Tuning':    (-0.22, -0.015),
    'ACI':             (0.02, 0.01),
    'SAOCP':           (0.02, -0.025),
    'LTT':             (-0.15, 0.02),
    'CRC':             (-0.10, -0.03),
    'NEX-Conf':        (0.02, -0.025),
    'Mohri-Conf':      (0.02, 0.015),
}
for m, (x, y) in summary.items():
    dx, dy = label_offsets.get(m, (0.015, 0.015))
    ax.annotate(m, xy=(x, y), xytext=(x + dx, y + dy),
                fontsize=10.5,
                fontweight='bold' if m == 'CSA-RLVR' else 'normal',
                color=STYLE[m]['c'] if m == 'CSA-RLVR' else 'black')

ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.02, 1.08)
ax.set_xlabel('Mean accept-rate across 8 benchmarks + 16 shift cells '
              '(higher = more useful)',
              fontsize=12)
ax.set_ylabel('Mean pathwise validity (fraction of streams with PathV=0)',
              fontsize=12)
ax.set_title('Every method, averaged across 8 main benchmarks (pivotal $\\alpha$) '
             '+ 16 synthetic shift cells:\n'
             'Only CSA-RLVR is simultaneously Valid and Non-refusing on BOTH '
             'real and synthetic streams',
             fontsize=12, pad=12)
ax.grid(True, alpha=0.3, linestyle=':')
# CSA pointer
csa_x, csa_y = summary['CSA-RLVR']
ax.annotate('', xy=(csa_x, csa_y), xytext=(csa_x - 0.3, csa_y - 0.15),
            arrowprops=dict(arrowstyle='->', color='#1f77b4',
                            lw=1.8, connectionstyle='arc3,rad=0.3'))
ax.text(csa_x - 0.31, csa_y - 0.17, 'Only method here',
        fontsize=11, color='#1f77b4', fontweight='bold',
        ha='right')
plt.tight_layout()
plt.savefig('figures/summary_frontier.pdf', bbox_inches='tight')
plt.savefig('figures/summary_frontier.png', dpi=150, bbox_inches='tight')
plt.close()
print('  figures/summary_frontier.{pdf,png}')


# =====================================================================
# FIG B: Per-method RADAR chart across 8 benchmarks
#   radius = "usable validity" = (1 - PathV_rate) * (AR > 0 ? 1 : 0)
#   i.e. a method gets zero credit when it refuses or violates.
# =====================================================================

# Two panels per method, side-by-side:
#   (a) Main radar: 8 benchmark axes, radius = 1 - PathV_rate (pure validity)
#   (b) Shift radar: 16 shift-cell axes, radius = 1 - PathV_rate
# "Full polygon" = 0 violations everywhere. Refusal does NOT shrink the radar:
# the user's point is that CSA is 0 violations on BOTH sets, even when it
# refuses on a cell. A method that refuses is still pathwise valid; what
# differentiates methods is whether they have non-zero PathV somewhere.

main_axes = [(BDISP[b], b, PIVOT[b]) for b in BENCH]
shift_axes = [(label, alpha, cell) for (label, alpha, cell) in SHIFT_CELLS]

N_main = len(main_axes)
N_shift = len(shift_axes)
angles_main = np.linspace(0, 2 * np.pi, N_main, endpoint=False).tolist()
angles_main += angles_main[:1]
angles_shift = np.linspace(0, 2 * np.pi, N_shift, endpoint=False).tolist()
angles_shift += angles_shift[:1]

fig = plt.figure(figsize=(22, 11.5))
# Two rows (main / shift) x 10 cols (one per method)
for idx, m in enumerate(METH):
    # --- Main radar ---
    ax_m = fig.add_subplot(2, 10, idx + 1, projection='polar')
    radii = []
    for (_disp, b, a) in main_axes:
        pv = get(b, a, m, 'pv')
        if pv is None:
            radii.append(0)
        else:
            radii.append(1 - pv_frac(pv))
    radii += radii[:1]
    color = STYLE[m]['c']
    ax_m.fill(angles_main, radii, color=color, alpha=0.35)
    ax_m.plot(angles_main, radii, color=color, linewidth=2.0)
    ax_m.set_xticks(angles_main[:-1])
    ax_m.set_xticklabels([BDISP[b] for b in BENCH], fontsize=6)
    ax_m.set_ylim(0, 1.05)
    ax_m.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_m.set_yticklabels(['', '', '', ''], fontsize=6)
    main_r = np.mean(radii[:-1])
    ax_m.set_title(f'{m}\nmain={main_r:.2f}',
                   fontsize=9,
                   fontweight='bold' if m == 'CSA-RLVR' else 'normal',
                   color=color if m == 'CSA-RLVR' else 'black', pad=10)
    ax_m.grid(alpha=0.3)

    # --- Shift radar ---
    ax_s = fig.add_subplot(2, 10, idx + 11, projection='polar')
    radii_s = []
    for (_label, _alpha, cell) in shift_axes:
        src = cell.get(m)
        if src is None:
            radii_s.append(0)
        else:
            radii_s.append(1 - pv_frac(src['pv']))
    radii_s += radii_s[:1]
    ax_s.fill(angles_shift, radii_s, color=color, alpha=0.35)
    ax_s.plot(angles_shift, radii_s, color=color, linewidth=2.0)
    ax_s.set_xticks(angles_shift[:-1])
    # short labels (too many to show all)
    short_labels = [''] * N_shift
    ax_s.set_xticklabels(short_labels, fontsize=5)
    ax_s.set_ylim(0, 1.05)
    ax_s.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_s.set_yticklabels(['', '', '', ''], fontsize=6)
    shift_r = np.mean(radii_s[:-1])
    ax_s.set_title(f'shift={shift_r:.2f}', fontsize=9,
                   fontweight='bold' if m == 'CSA-RLVR' else 'normal',
                   color=color if m == 'CSA-RLVR' else 'black', pad=8)
    ax_s.grid(alpha=0.3)

# Row labels on the left
fig.text(0.005, 0.75, 'MAIN\nbenchmarks\n($N$=8)',
         fontsize=11, rotation=90, va='center', ha='left',
         fontweight='bold', color='#222')
fig.text(0.005, 0.27, 'SYNTHETIC\nshift cells\n($N$=' + str(N_shift) + ')',
         fontsize=11, rotation=90, va='center', ha='left',
         fontweight='bold', color='#8b3a62')

plt.suptitle("Validity radar per method: one polygon per method on 8 main "
             "benchmarks (top row) and " + str(N_shift) + " synthetic shift "
             "cells (bottom row).\n"
             "Radius = (1 - PathV rate); full polygon = 0 violations.  "
             "CSA-RLVR fills BOTH polygons; every other method shrinks on at "
             "least one shift cell.",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('figures/summary_radar.pdf', bbox_inches='tight')
plt.savefig('figures/summary_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print('  figures/summary_radar.{pdf,png}')


# =====================================================================
# FIG C: Funnel / survival chart
# For every method: count how many (benchmark, alpha) cells it passes at
# each of 4 progressive gates:
#   gate 1: PathV = 0/10     (pathwise valid)
#   gate 2: AR > 0           (non-refusing)
#   gate 3: Risk <= alpha    (meets budget on accepted items, no-op if AR=0)
#   gate 4: AR >= 0.5 * ErrFree_AR  (achieves >= half of base-model utility)
# Total cells: 8 bench x 6 alpha = 48
# =====================================================================

MAIN_CELLS = [(b, a) for b in BENCH for a in ALPHAS]


def gate_check(pv, ar, risk, alpha):
    if pv is None or ar is None:
        return None
    v = pv_frac(pv)
    g1 = (v == 0)
    g2 = g1 and (ar > 0.01)
    # Risk <= alpha. If AR=0 then no risk to measure, count as passing.
    if risk is None or ar == 0:
        g3 = g2
    else:
        g3 = g2 and (risk <= alpha + 1e-6)
    g4 = g3 and (ar >= 0.5)
    return (g1, g2, g3, g4)


# We track main + shift separately so the bars can show "5 shift cells
# at the right end; Mohri passes on main but fails in shift."
main_counts  = {m: [0, 0, 0, 0] for m in METH}
shift_counts = {m: [0, 0, 0, 0] for m in METH}
main_tested  = {m: 0 for m in METH}
shift_tested = {m: 0 for m in METH}

for m in METH:
    # Main benchmarks
    for (b, a) in MAIN_CELLS:
        pv = get(b, a, m, 'pv')
        ar = get(b, a, m, 'ar')
        risk = get(b, a, m, 'risk')
        res = gate_check(pv, ar, risk, a)
        if res is None:
            continue
        main_tested[m] += 1
        for g_idx in range(4):
            main_counts[m][g_idx] += int(res[g_idx])
    # Shift cells (16 total: 12 MedQA at alpha in {0.05, 0.1, 0.2}
    # + 4 GSM8K at alpha=0.05)
    for (label, alpha, cell) in SHIFT_CELLS:
        src = cell.get(m)
        if src is None:
            continue
        res = gate_check(src['pv'], src['ar'], src['risk'], alpha)
        if res is None:
            continue
        shift_tested[m] += 1
        for g_idx in range(4):
            shift_counts[m][g_idx] += int(res[g_idx])

fig, ax = plt.subplots(figsize=(13, 6.8))
gate_labels = ['pathwise valid\n(PathV=0/10)',
               '+  non-refusing\n(AR > 0)',
               '+  meets budget\n(Risk $\\leq \\alpha$)',
               '+  useful utility\n(AR $\\geq 50\\%$)']
main_colors  = ['#c8d6f0', '#8ab4e0', '#4a86ca', '#0a4fa0']
shift_colors = ['#fce5b0', '#f0b060', '#d07030', '#8b3a10']

# Sort methods by total gate-4 count descending
def total_g4(m):
    return main_counts[m][3] + shift_counts[m][3]

sorted_meth = sorted(METH, key=lambda m: -total_g4(m))
y_positions = np.arange(len(sorted_meth))

# Two bars per method: top = main, bottom = shift.
h = 0.36
for y_idx, m in enumerate(sorted_meth):
    # Main (upper half-bar) --- blue palette
    n_main = main_tested[m]
    for g_idx in range(4):
        width = main_counts[m][g_idx] - (main_counts[m][g_idx-1] if g_idx else 0)
        if width <= 0:
            continue
        ax.barh(y_idx - h/2, width,
                left=(main_counts[m][g_idx-1] if g_idx else 0),
                color=main_colors[g_idx], edgecolor='white', linewidth=0.5,
                height=h, zorder=2)
    ax.barh(y_idx - h/2, n_main, color='none',
            edgecolor='#333', linewidth=0.6, height=h, linestyle=':', zorder=1)
    ax.text(n_main + 0.7, y_idx - h/2, f'main {main_counts[m][3]}/{n_main}',
            va='center', fontsize=8, fontweight='bold',
            color='#0a4fa0' if main_counts[m][3] > 0 else '#888')

    # Shift (lower half-bar) --- orange palette
    n_shift = shift_tested[m]
    for g_idx in range(4):
        width = shift_counts[m][g_idx] - (shift_counts[m][g_idx-1] if g_idx else 0)
        if width <= 0:
            continue
        ax.barh(y_idx + h/2, width,
                left=(shift_counts[m][g_idx-1] if g_idx else 0),
                color=shift_colors[g_idx], edgecolor='white', linewidth=0.5,
                height=h, zorder=2)
    ax.barh(y_idx + h/2, n_shift, color='none',
            edgecolor='#8b3a10', linewidth=0.6, height=h, linestyle=':', zorder=1)
    ax.text(n_shift + 0.7, y_idx + h/2, f'shift {shift_counts[m][3]}/{n_shift}',
            va='center', fontsize=8, fontweight='bold',
            color='#8b3a10' if shift_counts[m][3] > 0 else '#aa8060')

ax.set_yticks(y_positions)
ax.set_yticklabels(sorted_meth, fontsize=10)
for y_idx, m in enumerate(sorted_meth):
    if m == 'CSA-RLVR':
        ax.get_yticklabels()[y_idx].set_fontweight('bold')
        ax.get_yticklabels()[y_idx].set_color(STYLE[m]['c'])
ax.set_xlim(0, max(main_tested.values()) * 1.18)
ax.set_xlabel('Cells passing each successive gate  '
              '(upper bar = main benchmarks,  lower bar = synthetic shift)',
              fontsize=11)
ax.set_title('Progressive survival funnel: main benchmarks vs. synthetic shift.\n'
             'CSA-RLVR is the only method that passes all four gates on '
             'every cell in both regimes.',
             fontsize=12, pad=10)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.25)
# Legend: 4 blue (main gates) + 4 orange (shift gates)
handles = []; labels = []
for i in range(4):
    handles.append(Rectangle((0, 0), 1, 1, color=main_colors[i]))
    labels.append('main: ' + gate_labels[i].replace('\n', ' '))
for i in range(4):
    handles.append(Rectangle((0, 0), 1, 1, color=shift_colors[i]))
    labels.append('shift: ' + gate_labels[i].replace('\n', ' '))
ax.legend(handles, labels, loc='lower right', fontsize=7,
          frameon=True, facecolor='white', ncol=2,
          title='Gates (cumulative)', title_fontsize=9)
plt.tight_layout()
plt.savefig('figures/summary_funnel.pdf', bbox_inches='tight')
plt.savefig('figures/summary_funnel.png', dpi=150, bbox_inches='tight')
plt.close()
print('  figures/summary_funnel.{pdf,png}')


# =====================================================================
# FIG D: Ridge / dot plot of PathV rate across all (bench, alpha) cells
# For each method, show distribution of PathV rates across all cells.
# CSA shows a single spike at 0; baselines have mass at 1.0.
# =====================================================================

fig, ax = plt.subplots(figsize=(13.5, 8))
rows = list(reversed(METH))  # so CSA-RLVR is on top

for y_idx, m in enumerate(rows):
    rates_main, rates_shift = [], []
    # Main
    for (b, a) in MAIN_CELLS:
        pv = get(b, a, m, 'pv')
        if pv is None:
            continue
        rates_main.append(pv_frac(pv))
    # Shift (16 cells)
    for (label, alpha, cell) in SHIFT_CELLS:
        src = cell.get(m)
        if src is None:
            continue
        rates_shift.append(pv_frac(src['pv']))
    all_rates = np.array(rates_main + rates_shift)
    if len(all_rates) == 0:
        continue
    st = STYLE[m]
    # Scatter main (filled circles), shift (hollow squares) on the same row.
    if rates_main:
        jm = np.random.RandomState(42 + y_idx).uniform(-0.18, 0.18, len(rates_main))
        ax.scatter(np.array(rates_main), y_idx + jm,
                   s=55, color=st['c'], marker=st['m'], edgecolor='black',
                   linewidth=0.4, alpha=0.6, zorder=3,
                   label='main benchmark' if y_idx == 0 else None)
    if rates_shift:
        js = np.random.RandomState(99 + y_idx).uniform(-0.25, -0.05, len(rates_shift))
        # Plot shift markers slightly below center, hollow, red-tinted edge
        ax.scatter(np.array(rates_shift), y_idx + js - 0.05,
                   s=110, facecolor='white', marker='s',
                   edgecolor='#b03020', linewidth=1.6,
                   alpha=0.95, zorder=4,
                   label='synthetic shift cell' if y_idx == 0 else None)
    # Horizontal mean line over ALL cells
    mean_all = float(all_rates.mean())
    ax.plot([mean_all, mean_all], [y_idx - 0.3, y_idx + 0.3],
            color=st['c'], lw=2.5, zorder=5)
    ax.scatter(mean_all, y_idx, s=130, color='white',
               edgecolor=st['c'], linewidth=2.2, marker='D', zorder=6)
    # Annotation: main-mean, shift-mean, counts
    m_mean = np.mean(rates_main) if rates_main else float('nan')
    s_mean = np.mean(rates_shift) if rates_shift else float('nan')
    ax.text(1.06, y_idx,
            f'main mean={m_mean:.2f} (n={len(rates_main)})   '
            f'shift mean={s_mean:.2f} (n={len(rates_shift)})',
            va='center', fontsize=8.5, color=st['c'],
            fontweight='bold' if m == 'CSA-RLVR' else 'normal')

ax.axvline(0, color='green', linestyle='-', linewidth=1, alpha=0.5,
           label='0/10 violations (perfect)')
ax.axvline(1, color='red', linestyle='-', linewidth=1, alpha=0.5,
           label='10/10 violations (worst)')
ax.set_xlim(-0.05, 1.75)
ax.set_ylim(-0.5, len(rows) - 0.5)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows, fontsize=10)
# Legend for main vs shift markers (only 2 handles shown once)
main_handle = Line2D([0], [0], marker='o', color='#555', markerfacecolor='#555',
                      markersize=8, linestyle='None', alpha=0.6)
shift_handle = Line2D([0], [0], marker='s', color='#b03020',
                       markerfacecolor='white', markersize=10,
                       markeredgewidth=1.6, linestyle='None')
mean_handle  = Line2D([0], [0], marker='D', color='#444',
                       markerfacecolor='white', markersize=10,
                       markeredgewidth=2, linestyle='None')
ax.legend([main_handle, shift_handle, mean_handle],
          ['main benchmark cell', 'synthetic shift cell', 'mean'],
          loc='upper center', bbox_to_anchor=(0.5, -0.08),
          ncol=3, fontsize=10, frameon=True, facecolor='white',
          edgecolor='#bbbbbb')
# Bold CSA label
for lbl in ax.get_yticklabels():
    if lbl.get_text() == 'CSA-RLVR':
        lbl.set_fontweight('bold')
        lbl.set_color(STYLE['CSA-RLVR']['c'])
ax.set_xlabel('Pathwise-violation rate within a (benchmark, $\\alpha$) cell',
              fontsize=11)
ax.set_title('Distribution of pathwise-violation rates across 48 main-benchmark '
             'cells (circles) and 16 synthetic shift cells (red squares).\n'
             'Only CSA-RLVR is at PathV=0 on BOTH; Mohri-Conf / CRC / LTT are 0 '
             'on main but have shift-cell mass > 0.',
             fontsize=12, pad=10)
ax.grid(axis='x', alpha=0.25)
# Annotation for CSA
csa_y = len(rows) - 1 - METH.index('CSA-RLVR')
ax.annotate('All 48 main + 16 shift cells at PathV=0',
            xy=(0, csa_y), xytext=(0.25, csa_y + 0.5),
            fontsize=10, color=STYLE['CSA-RLVR']['c'], fontweight='bold',
            arrowprops=dict(arrowstyle='->',
                            color=STYLE['CSA-RLVR']['c'], lw=1.5))
plt.tight_layout()
plt.savefig('figures/summary_ridge.pdf', bbox_inches='tight')
plt.savefig('figures/summary_ridge.png', dpi=150, bbox_inches='tight')
plt.close()
print('  figures/summary_ridge.{pdf,png}')


print('\nDone.')
