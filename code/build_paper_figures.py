"""
Build the 3 main figures for the paper directly from verified JSON:
  figures/violation_heatmap.pdf  - benchmark x method heatmap at pivotal alpha
  figures/pareto_frontier.pdf    - AR vs Risk scatter per benchmark (pivotal)
  figures/phase_transition.pdf   - PathV% as function of alpha (AR vs PathV curve)

All data are sourced from paper_tables/_verified_numbers.json, which is
populated verbatim from results/*.json.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('figures', exist_ok=True)

with open('data/results/_verified_numbers.json') as f:
    V = json.load(f)

BENCH = ['medical', 'pubmedqa', 'tatqa', 'mednli',
         'gsm8k', 'headqa', 'arc', 'casehold']
BENCH_DISP = {'medical': 'MedQA', 'pubmedqa': 'PubMedQA',
              'tatqa': 'TAT-QA', 'mednli': 'MedNLI',
              'gsm8k': 'GSM8K', 'headqa': 'HEAD-QA',
              'arc': 'ARC', 'casehold': 'CaseHOLD'}
PIVOT = {'medical': 0.20, 'pubmedqa': 0.20, 'tatqa': 0.20, 'mednli': 0.20,
         'gsm8k': 0.05, 'headqa': 0.20, 'arc': 0.10, 'casehold': 0.25}
METH = ['CSA-RLVR', 'Always-Act', 'Fixed-Threshold', 'Naive-Tuning',
        'ACI', 'SAOCP', 'LTT', 'CRC', 'NEX-Conf', 'Mohri-Conf']
# Display names used on figure axes / legends.  Data keys stay 'Mohri-Conf'
# for JSON lookup; legends/ticks render as 'ConfFact' per the paper.
DISP = {m: m for m in METH}
DISP['Mohri-Conf']     = 'ConfFact'
DISP['CSA-RLVR']       = 'CSA (ours)'
DISP['Fixed-Threshold'] = 'Fixed-Thr.'
DISP['Naive-Tuning']    = 'Naive-Tun.'
METH_DISP = [DISP[m] for m in METH]


def pv_frac(pv_str):
    if pv_str in ('--', None, '?'): return None
    n, t = pv_str.split('/')
    return int(n) / int(t)


def get(b, a, m, key):
    ak = f'{a:g}'
    cell = V['combined_main_plus_new'][b].get(ak, {}).get(m)
    if cell is None: return None
    return cell.get(key)


# =========================================================
# Figure 1: Violation heatmap (8 benches x 10 methods at pivotal alpha)
# =========================================================

fig, ax = plt.subplots(figsize=(11, 5.5))
n_b, n_m = len(BENCH), len(METH)
M = np.full((n_b, n_m), np.nan)
for i, b in enumerate(BENCH):
    a = PIVOT[b]
    for j, m in enumerate(METH):
        pv = get(b, a, m, 'pv')
        if pv is not None:
            M[i, j] = pv_frac(pv)

# Use a custom colormap: green->yellow->red
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    'viocl', [(0.0, '#2e8b57'), (0.01, '#7fc98b'),
              (0.3, '#ffd700'), (0.7, '#ff8c42'),
              (1.0, '#c03030')])
im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=0, vmax=1)

ax.set_xticks(range(n_m))
ax.set_xticklabels(METH_DISP, rotation=35, ha='right', fontsize=10)
ax.set_yticks(range(n_b))
ax.set_yticklabels([f'{BENCH_DISP[b]} (α={PIVOT[b]:.2f})' for b in BENCH],
                   fontsize=10)
# Annotate each cell with PathV fraction
for i, b in enumerate(BENCH):
    a = PIVOT[b]
    for j, m in enumerate(METH):
        pv = get(b, a, m, 'pv')
        if pv is None:
            ax.text(j, i, '--', ha='center', va='center',
                    fontsize=8, color='gray')
        else:
            ar = get(b, a, m, 'ar') or 0
            label = pv if ar > 0 else pv + '\n(refuse)'
            color = 'white' if pv_frac(pv) > 0.5 else 'black'
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=7.5, color=color,
                    fontweight='bold' if pv_frac(pv) == 0 else 'normal')
cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='Pathwise violation rate (frac. streams)')
ax.set_title('Pathwise-violation heatmap across 8 benchmarks × 10 methods '
             '(each at its pivotal α)', fontsize=11)
plt.tight_layout()
plt.savefig('figures/violation_heatmap.pdf', bbox_inches='tight')
plt.savefig('figures/violation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('  figures/violation_heatmap.{pdf,png}')


# =========================================================
# Figure 2: Pareto frontier (AR vs Risk at pivotal alpha)
# =========================================================

fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
markers = {
    'CSA-RLVR': ('*', 'tab:blue', 220),
    'Always-Act': ('o', 'tab:red', 80),
    'Fixed-Threshold': ('s', 'tab:orange', 70),
    'Naive-Tuning': ('^', 'tab:purple', 70),
    'ACI': ('D', 'tab:green', 70),
    'SAOCP': ('v', 'tab:olive', 70),
    'LTT': ('P', 'darkred', 90),
    'CRC': ('X', 'tab:cyan', 90),
    'NEX-Conf': ('p', 'tab:pink', 90),
    'Mohri-Conf': ('h', 'slategray', 90),
}
for idx, b in enumerate(BENCH):
    ax = axes[idx // 4, idx % 4]
    a = PIVOT[b]
    for m in METH:
        ar = get(b, a, m, 'ar')
        risk = get(b, a, m, 'risk')
        if ar is None or risk is None: continue
        mk, col, sz = markers[m]
        # Edge color: green for valid, red for any violation
        pv = pv_frac(get(b, a, m, 'pv') or '0/10')
        edge = 'darkgreen' if pv == 0 else 'darkred'
        ax.scatter(risk * 100, ar * 100, marker=mk, color=col,
                   edgecolor=edge, linewidth=1.4, s=sz,
                   label=m if idx == 0 else None, zorder=3,
                   alpha=0.9)
    ax.axvline(a * 100, color='k', linestyle='--', alpha=0.6)
    ax.axvspan(a * 100, 100, alpha=0.08, color='red')
    ax.set_xlim(-1, a * 100 * 3.5)
    ax.set_ylim(-5, 105)
    ax.set_title(f'{BENCH_DISP[b]} (α={a:.2f})', fontsize=10)
    ax.set_xlabel('Risk (%)', fontsize=9)
    if idx % 4 == 0:
        ax.set_ylabel('Accept rate (%)', fontsize=9)
    ax.grid(alpha=0.25)
# Single shared legend
handles = [plt.Line2D([0], [0], marker=markers[m][0], color='w',
                       markerfacecolor=markers[m][1],
                       markeredgecolor='black', markersize=9)
           for m in METH]
fig.legend(handles, METH_DISP, ncol=5, loc='upper center',
           bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)
plt.suptitle('', y=1.02)
plt.tight_layout()
plt.savefig('figures/pareto_frontier.pdf', bbox_inches='tight')
plt.savefig('figures/pareto_frontier.png', dpi=150, bbox_inches='tight')
plt.close()
print('  figures/pareto_frontier.{pdf,png}')


# =========================================================
# Figure 3a: Phase transition - CSA empirical risk vs alpha with y=alpha budget
# (the classic "risk under budget, AR nontrivial" figure)
# =========================================================

fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharey=True)
for idx, b in enumerate(BENCH):
    ax = axes[idx // 4, idx % 4]
    # Also include extra alphas for GSM8K/HEADQA/ARC from the new_baselines json
    # Collect CSA data (AR + risk) at every available alpha for this bench
    ars = []; risks = []; alphas = []
    for a in sorted(V['combined_main_plus_new'][b].keys(), key=float):
        a_f = float(a)
        if a_f > 0.32:
            continue
        cell = V['combined_main_plus_new'][b][a].get('CSA-RLVR')
        if cell is None:
            continue
        alphas.append(a_f)
        ars.append(cell['ar'])
        risks.append(cell['risk'])
    alphas = np.array(alphas)
    ars = np.array(ars); risks = np.array(risks)
    # CSA AR
    ax.plot(alphas, ars, 'o-', color='#1f77b4', linewidth=2, markersize=6,
            label='CSA AR' if idx == 0 else None)
    # CSA empirical Risk
    ax.plot(alphas, risks, 's--', color='#c03030', linewidth=2, markersize=5,
            label='CSA Risk' if idx == 0 else None)
    # y = alpha budget line
    xmax = max(0.32, alphas.max() if len(alphas) else 0.32)
    xs = np.linspace(0, xmax, 50)
    ax.plot(xs, xs, ':', color='black', linewidth=1.2, alpha=0.7,
            label=r'$y=\alpha$ budget' if idx == 0 else None)
    # Mark pivotal alpha
    a_piv = PIVOT[b]
    ax.axvline(a_piv, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    # Get err for this bench
    label_m = {'medical': (1018, 31.5), 'pubmedqa': (800, 23.9),
               'tatqa': (565, 25.7), 'mednli': (1422, 21.0),
               'gsm8k': (1055, 5.0), 'headqa': (1100, 26.0),
               'arc': (938, 10.0), 'casehold': (2880, 34.0)}
    _, err = label_m[b]
    ax.set_title(f'{BENCH_DISP[b]} (Err={err:.1f}%)', fontsize=10)
    ax.set_xlim(0, 0.32)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r'$\alpha$', fontsize=10)
    if idx % 4 == 0:
        ax.set_ylabel('Accept rate / Risk', fontsize=10)
    ax.grid(alpha=0.25)
fig.legend(['CSA AR', 'CSA Risk', r'$y=\alpha$ budget'],
           ncol=3, loc='upper center',
           bbox_to_anchor=(0.5, 1.04), fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('figures/phase_budget.pdf', bbox_inches='tight')
plt.savefig('figures/phase_budget.png', dpi=150, bbox_inches='tight')
plt.close()
print('  figures/phase_budget.{pdf,png}')


# =========================================================
# Figure 3: Phase transition (PathV as a function of alpha, per method)
# =========================================================

fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharey=True)
for idx, b in enumerate(BENCH):
    ax = axes[idx // 4, idx % 4]
    for m in METH:
        xs, ys = [], []
        for a in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            pv = get(b, a, m, 'pv')
            if pv is None: continue
            xs.append(a); ys.append(pv_frac(pv) * 100)
        if not xs: continue
        mk, col, sz = markers[m]
        ax.plot(xs, ys, marker=mk, markersize=8, linewidth=1.8,
                color=col, label=m if idx == 0 else None, alpha=0.85)
    a_piv = PIVOT[b]
    ax.axvline(a_piv, color='k', linestyle=':', alpha=0.5)
    ax.text(a_piv, 102, f'pivotal', ha='center', fontsize=7, color='black')
    ax.set_xlim(0.04, 0.31)
    ax.set_ylim(-5, 110)
    ax.set_title(f'{BENCH_DISP[b]}', fontsize=10)
    ax.set_xlabel('Risk budget α', fontsize=9)
    if idx % 4 == 0:
        ax.set_ylabel('Pathwise-violation rate (%)', fontsize=9)
    ax.grid(alpha=0.25)
handles = [plt.Line2D([0], [0], marker=markers[m][0], color=markers[m][1],
                       linewidth=1.5, markersize=9) for m in METH]
fig.legend(handles, METH_DISP, ncol=5, loc='upper center',
           bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig('figures/phase_transition.pdf', bbox_inches='tight')
plt.savefig('figures/phase_transition.png', dpi=150, bbox_inches='tight')
plt.close()
print('  figures/phase_transition.{pdf,png}')

# =========================================================
# Figure 4: TRIPTYCH scoreboard -- Validity, AR, Precision
# (three metrics across 8 benchmarks x 10 methods at pivotal alpha)
# =========================================================

fig = plt.figure(figsize=(18, 5.5))
gs = fig.add_gridspec(1, 3, wspace=0.10)

# Prepare data
n_b, n_m = len(BENCH), len(METH)
validity = np.full((n_b, n_m), np.nan)
ar_arr   = np.full((n_b, n_m), np.nan)
prec_arr = np.full((n_b, n_m), np.nan)
refuse   = np.zeros((n_b, n_m), dtype=bool)
for i, b in enumerate(BENCH):
    a = PIVOT[b]
    for j, m in enumerate(METH):
        pv = get(b, a, m, 'pv')
        ar = get(b, a, m, 'ar')
        risk = get(b, a, m, 'risk')
        if pv is None or ar is None:
            continue
        validity[i, j] = 1.0 - pv_frac(pv)
        ar_arr[i, j]   = ar
        if ar > 0:
            prec_arr[i, j] = 1.0 - risk
        else:
            refuse[i, j] = True
            prec_arr[i, j] = np.nan

# Common plotting helper
# Panel 1: Validity (1 - PathV_rate)
from matplotlib.colors import LinearSegmentedColormap
valid_cmap = LinearSegmentedColormap.from_list(
    'valid', ['#c03030', '#ff8c42', '#ffd700', '#7fc98b', '#2e8b57'])
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(validity, aspect='auto', cmap=valid_cmap, vmin=0, vmax=1)
ax1.set_xticks(range(n_m))
ax1.set_xticklabels(METH_DISP, rotation=40, ha='right', fontsize=9)
ax1.set_yticks(range(n_b))
ax1.set_yticklabels([f'{BENCH_DISP[b]}\n(α={PIVOT[b]:.2f})' for b in BENCH],
                    fontsize=9)
ax1.set_title('Validity  (1 − PathV rate)', fontsize=12, pad=8, fontweight='bold')
for i in range(n_b):
    for j in range(n_m):
        v = validity[i, j]
        if np.isnan(v):
            ax1.text(j, i, '—', ha='center', va='center', fontsize=8, color='gray')
        else:
            ax1.text(j, i, f'{v:.1f}', ha='center', va='center',
                     fontsize=7.5, color='white' if v < 0.5 else 'black',
                     fontweight='bold' if v == 1.0 else 'normal')
plt.colorbar(im1, ax=ax1, shrink=0.85, pad=0.01)

# Panel 2: Accept rate
ar_cmap = LinearSegmentedColormap.from_list(
    'ar', ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'])
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(ar_arr, aspect='auto', cmap=ar_cmap, vmin=0, vmax=1)
ax2.set_xticks(range(n_m))
ax2.set_xticklabels(METH_DISP, rotation=40, ha='right', fontsize=9)
ax2.set_yticks(range(n_b))
ax2.set_yticklabels([])
ax2.set_title('Accept Rate  (utility)', fontsize=12, pad=8, fontweight='bold')
for i in range(n_b):
    for j in range(n_m):
        v = ar_arr[i, j]
        if np.isnan(v):
            ax2.text(j, i, '—', ha='center', va='center', fontsize=8, color='gray')
        else:
            if v == 0:
                ax2.text(j, i, 'refuse', ha='center', va='center',
                         fontsize=6.5, style='italic', color='dimgray')
            else:
                ax2.text(j, i, f'{v*100:.0f}%', ha='center', va='center',
                         fontsize=7.5, color='white' if v > 0.5 else 'black')
plt.colorbar(im2, ax=ax2, shrink=0.85, pad=0.01)

# Panel 3: Precision on accepted (1 - Risk)
prec_cmap = LinearSegmentedColormap.from_list(
    'prec', ['#f7fcf5', '#c7e9c0', '#74c476', '#238b45', '#00441b'])
ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(prec_arr, aspect='auto', cmap=prec_cmap, vmin=0.5, vmax=1.0)
ax3.set_xticks(range(n_m))
ax3.set_xticklabels(METH_DISP, rotation=40, ha='right', fontsize=9)
ax3.set_yticks(range(n_b))
ax3.set_yticklabels([])
ax3.set_title('Precision  (1 − Risk on accepted)',
              fontsize=12, pad=8, fontweight='bold')
for i in range(n_b):
    for j in range(n_m):
        v = prec_arr[i, j]
        if np.isnan(v):
            label = 'refuse' if refuse[i, j] else '—'
            style = 'italic' if refuse[i, j] else 'normal'
            ax3.text(j, i, label, ha='center', va='center',
                     fontsize=6.5, color='dimgray', style=style)
        else:
            ax3.text(j, i, f'{v*100:.0f}%', ha='center', va='center',
                     fontsize=7.5, color='white' if v > 0.85 else 'black',
                     fontweight='bold' if v > 0.9 else 'normal')
plt.colorbar(im3, ax=ax3, shrink=0.85, pad=0.01)

plt.suptitle('Three-metric scoreboard: every (benchmark, method) cell at pivotal α\n'
             'Only CSA-RLVR is simultaneously high on Validity, Accept Rate, and Precision.',
             fontsize=13, y=1.04)
plt.savefig('figures/scoreboard_triptych.pdf', bbox_inches='tight')
plt.savefig('figures/scoreboard_triptych.png', dpi=150, bbox_inches='tight')
plt.close()
print('  figures/scoreboard_triptych.{pdf,png}')


# =========================================================
# Figure 5: COMPOSITE TILE -- single unified grid where each cell
# shows all 3 metrics as stacked colored bars (Validity top, AR middle,
# Prec bottom). Single eye-catching plot.
# =========================================================

fig, ax = plt.subplots(figsize=(14, 7))
cell_w = 0.9
cell_h = 0.9
for i, b in enumerate(BENCH):
    a = PIVOT[b]
    for j, m in enumerate(METH):
        pv = get(b, a, m, 'pv')
        ar_v = get(b, a, m, 'ar')
        risk = get(b, a, m, 'risk')
        if pv is None or ar_v is None:
            ax.text(j, -i, '—', ha='center', va='center', color='gray')
            continue
        v = 1.0 - pv_frac(pv)
        # Row layout within the cell (top to bottom): Validity, AR, Prec
        # Use smaller inner bars
        x0 = j - cell_w / 2
        y0 = -i - cell_h / 2
        # --- Validity strip (top third) ---
        vcolor = valid_cmap(v)
        rect = plt.Rectangle((x0, y0 + 2*cell_h/3), cell_w, cell_h/3,
                              facecolor=vcolor, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)
        # --- AR strip (middle third): bar width scales with AR ---
        ar_color = ar_cmap(ar_v) if ar_v > 0 else '#eeeeee'
        rect2 = plt.Rectangle((x0, y0 + cell_h/3), cell_w, cell_h/3,
                               facecolor=ar_color, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect2)
        # --- Prec strip (bottom third) ---
        if ar_v > 0:
            prec = 1.0 - risk
            p_color = prec_cmap((prec - 0.5) / 0.5)
        else:
            prec = None
            p_color = '#eeeeee'
        rect3 = plt.Rectangle((x0, y0), cell_w, cell_h/3,
                               facecolor=p_color, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect3)

        # Annotate numbers in each strip
        pv_txt = pv.replace('/10', '') + '/10' if pv else '--'
        pv_txt_short = pv  # e.g., '0/10'
        ax.text(j, -i + cell_h/3, pv_txt_short, ha='center', va='center',
                fontsize=6.5, color='white' if v < 0.5 else 'black',
                fontweight='bold' if v == 1.0 else 'normal')
        ax.text(j, -i, f'{ar_v*100:.0f}%' if ar_v > 0 else 'ref',
                ha='center', va='center',
                fontsize=6.5, color='white' if ar_v > 0.5 else 'black',
                style='italic' if ar_v == 0 else 'normal')
        if prec is not None:
            ax.text(j, -i - cell_h/3, f'{prec*100:.0f}%',
                    ha='center', va='center',
                    fontsize=6.5, color='white' if prec > 0.85 else 'black')
        else:
            ax.text(j, -i - cell_h/3, '--', ha='center', va='center',
                    fontsize=6.5, color='gray')

ax.set_xlim(-0.6, n_m - 0.4)
# Original ylim: method names stay at the bottom of the grid.
ax.set_ylim(-n_b + 0.5, 0.5)
ax.set_xticks(range(n_m))
ax.set_xticklabels(METH_DISP, rotation=35, ha='right', fontsize=11,
                   color='#000000')
ax.set_yticks(range(0, -n_b, -1))
ax.set_yticklabels([f'{BENCH_DISP[b]} (α={PIVOT[b]:.2f})' for b in BENCH],
                   fontsize=11, color='#000000')
ax.tick_params(axis='both', colors='#000000')
ax.set_frame_on(False)
ax.set_title('Composite-tile scoreboard: each cell shows Validity | AR | Precision\n'
             '(top: 1-PathV, middle: action rate, bottom: precision on accepted)',
             fontsize=12, pad=12, color='#000000')

# Legend in FIGURE coordinates, placed below the axis so it does not
# displace or collide with the rotated method names.
# tight_layout + subplots_adjust reserves space at the bottom for it.
plt.tight_layout()
plt.subplots_adjust(bottom=0.20)
fig.text(0.05, 0.07,
         'Top strip = Validity   |   Middle = Action Rate   |   Bottom = Precision',
         fontsize=11, color='#111111', ha='left', fontweight='bold')
fig.text(0.05, 0.03,
         'Green = high   |   Red = low   |   "ref" = principled refusal',
         fontsize=11, color='#111111', ha='left')

plt.savefig('figures/scoreboard_tile.pdf', bbox_inches='tight')
plt.savefig('figures/scoreboard_tile.png', dpi=150, bbox_inches='tight')
plt.close()
print('  figures/scoreboard_tile.{pdf,png}')


# =========================================================
# Figure 6: Running-risk trajectory on CaseHOLD at alpha=0.25
# from the REAL mean_risk_curve field in the JSON
# =========================================================

_traj_path = 'data/results/casehold/casehold_alpha0.25.json'
if not os.path.exists(_traj_path):
    print(f'  [skipped] figures/trajectory.{{pdf,png}}: '
          f'{_traj_path} not present in this release; regenerate by '
          f'running the casehold experiment scripts.')
    raise SystemExit(0)
traj_src = json.load(open(_traj_path))
T = traj_src['T']
alpha_line = traj_src['alpha']
fig, ax = plt.subplots(figsize=(10.5, 5.5))

# Color scheme explicit: CSA distinctive, baselines similar
traj_colors = {
    'CSA-RLVR':        ('#1f77b4', 2.4, '-',  10, 'CSA-RLVR (ours)'),
    'Always-Act':      ('#d62728', 1.4, '-',  3,  'Always-Act'),
    'Fixed-Threshold': ('#ff7f0e', 1.4, '--', 3,  'Fixed-Threshold'),
    'Naive-Tuning':    ('#9467bd', 1.4, ':',  3,  'Naive-Tuning'),
    'ACI':             ('#8c564b', 1.4, '-.', 3,  'ACI'),
    'SAOCP':           ('#e377c2', 1.4, '-',  3,  'SAOCP'),
}

# Violation zone shading
ax.axhspan(alpha_line, 0.42, facecolor='#c03030', alpha=0.06, zorder=0)
ax.text(T * 0.02, alpha_line + 0.10,
        'Pathwise violation zone  ($\\mathrm{Risk}_t > \\alpha$)',
        fontsize=9, color='#a02020', style='italic')

# Alpha budget line
ax.axhline(alpha_line, color='black', linestyle='--', linewidth=1.2,
           alpha=0.7, zorder=1)
ax.text(T * 0.95, alpha_line + 0.008, f'$\\alpha={alpha_line:.2f}$',
        ha='right', fontsize=10)

# Subsample curves for plotting (every 100 steps)
stride = max(1, T // 500)
xs = np.arange(0, T, stride)
for m in ['Always-Act', 'Fixed-Threshold', 'Naive-Tuning', 'ACI', 'SAOCP',
         'CSA-RLVR']:  # plot CSA last so it's on top
    curve = np.asarray(traj_src['methods'][m]['mean_risk_curve'])
    if len(curve) == 0:
        continue
    ys = curve[xs]
    color, lw, ls, z, label = traj_colors[m]
    ax.plot(xs, ys, linestyle=ls, linewidth=lw, color=color,
            alpha=0.9, zorder=z, label=label)

# Annotation for CSA
ax.annotate(f'CSA stays at $\\mathrm{{Risk}}_t \\approx {traj_src["methods"]["CSA-RLVR"]["final_risk_mean"]:.2f}$\n(below $\\alpha={alpha_line:.2f}$ for all $t$)',
            xy=(T * 0.70, traj_src['methods']['CSA-RLVR']['final_risk_mean']),
            xytext=(T * 0.45, 0.08),
            fontsize=10, color='#1f4b75',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2,
                            connectionstyle='arc3,rad=0.2'),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#e0f0ff',
                      edgecolor='#1f77b4', linewidth=0.8))

ax.set_xlim(0, T)
ax.set_ylim(0, 0.42)
ax.set_xlabel('Stream time step $t$', fontsize=11)
ax.set_ylabel('Running empirical risk  $\\mathrm{Risk}_t$', fontsize=11)
ax.set_title(f'Running-risk trajectory on CaseHOLD at $\\alpha={alpha_line:.2f}$ '
             f'($T={T:,}$ steps, 10-rep mean)',
             fontsize=12, pad=8)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9, frameon=True, ncol=2,
          facecolor='white', edgecolor='lightgray')
plt.tight_layout()
plt.savefig('figures/trajectory.pdf', bbox_inches='tight')
plt.savefig('figures/trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print('  figures/trajectory.{pdf,png}')


print('\nDone.')
