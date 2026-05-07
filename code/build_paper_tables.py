"""
Load every JSON result and emit the paper's LaTeX tables and figures. All
numbers in the paper are produced from the following result files:

    results/<bench>/<bench>_alphaX.json            (main 6-method pipeline)
    results/new_baselines_summary.json             (CRC/NEX-Conf/Mohri)
    results/ablation_shift_hard.json               (MedQA alpha=0.20, 10 methods)
    results/ablation_shift_hard_medqa_alpha0.05.json
    results/ablation_shift_hard_gsm8k_alpha0.05.json
    results/ablation_shift_lowalpha.json           (interleaved shifts)
    results/deepseek_new_baselines_summary.json    (cross-model CRC/NEX/Mohri)
    results/medical_deepseek/*.json                (cross-model main pipeline)
    results/gsm8k_deepseek/*.json

Output:
    paper_tables/*.tex
    paper_tables/_verified_numbers.json   (for spot-checking)
"""

import json
import os
import sys
from glob import glob

RES = 'results'
OUT = 'paper_tables'
os.makedirs(OUT, exist_ok=True)

BENCHES = ['medical', 'pubmedqa', 'tatqa', 'mednli',
           'gsm8k', 'headqa', 'arc', 'casehold']
BENCH_META = {
    'medical':  ('MedQA',         'Fleming-R1-7B',          1018, 31.5, 0.20),
    'pubmedqa': ('PubMedQA',      'Fleming-R1-7B',           800, 23.9, 0.20),
    'tatqa':    ('TAT-QA arith',  'Fin-R1-7B',               565, 25.7, 0.20),
    'mednli':   ('MedNLI',        'Fleming-R1-7B',          1422, 21.0, 0.20),
    'gsm8k':    ('GSM8K',         'Qwen2.5-Math-7B-Instruct',1055,  5.0, 0.05),
    'headqa':   ('HEAD-QA',       'Fleming-R1-7B',          1100, 26.0, 0.20),
    'arc':      ('ARC-Challenge', 'Qwen2.5-7B-Instruct',     938, 10.0, 0.10),
    'casehold': ('CaseHOLD',      'Saul-7B-Instruct',       2880, 34.0, 0.25),
}
UNIFORM_ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
METHOD_ORDER = ['CSA-RLVR', 'Always-Act', 'Fixed-Threshold', 'Naive-Tuning',
                'ACI', 'SAOCP', 'LTT', 'CRC', 'NEX-Conf', 'Mohri-Conf']


# =================================================================
# Load main 6-method pipeline per (bench, alpha)
# =================================================================

def load_main_pipeline():
    """Returns {bench: {alpha: {method: {final_risk_mean, final_ar_mean,
    max_risk_mean, pathwise_violation_rate, precision_mean, coverage_correct_mean}}}}"""
    out = {b: {} for b in BENCHES}
    for b in BENCHES:
        for a in UNIFORM_ALPHAS:
            for fmt in [f'{a:.2f}', f'{a:.3f}', f'{a:.1f}']:
                path = f'{RES}/{b}/{b}_alpha{fmt}.json'
                if os.path.exists(path):
                    with open(path) as f:
                        d = json.load(f)
                    for m, cell in d['methods'].items():
                        out[b].setdefault(a, {})[m] = {
                            'risk':  float(cell.get('final_risk_mean', 0)),
                            'ar':    float(cell.get('final_ar_mean', 0)),
                            'maxr':  float(cell.get('max_risk_mean', 0)),
                            'prec':  float(cell.get('precision_mean', 0)),
                            'cov':   float(cell.get('coverage_correct_mean', 0)),
                            'pv':    cell.get('pathwise_violation_rate', '?'),
                        }
                    break
    return out


def load_new_baselines():
    with open(f'{RES}/new_baselines_summary.json') as f:
        data = json.load(f)
    out = {b: {} for b in BENCHES}
    for b in BENCHES:
        for a in UNIFORM_ALPHAS:
            ak = f'{a:g}'  # '0.05', '0.1', '0.2'
            if ak in data[b]['per_alpha']:
                cell = data[b]['per_alpha'][ak]
            else:
                # try other formats
                matched = None
                for key in data[b]['per_alpha']:
                    if abs(float(key) - a) < 1e-6:
                        matched = key; break
                if matched is None:
                    continue
                cell = data[b]['per_alpha'][matched]
            for m in ['CRC', 'NEX-Conf', 'Mohri-Conf']:
                out[b].setdefault(a, {})[m] = {
                    'risk':  float(cell[m]['final_risk_mean']),
                    'ar':    float(cell[m]['final_ar_mean']),
                    'maxr':  float(cell[m]['max_risk_mean']),
                    'prec':  1.0 - float(cell[m]['final_risk_mean']),
                    'cov':   0.0,  # not reported by new_baselines
                    'pv':    cell[m]['pathwise_violation_rate'],
                }
    return out


def load_ltt_pivots():
    """Pull LTT numbers from run_ltt_pivots.py output."""
    path = f'{RES}/ltt_pivotal_summary.json'
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for bench, cell in d.items():
        out.setdefault(bench, {})[float(cell['alpha'])] = {
            'LTT': {
                'risk':  float(cell['final_risk_mean']),
                'ar':    float(cell['final_ar_mean']),
                'maxr':  float(cell['max_risk_mean']),
                'prec':  1.0 - float(cell['final_risk_mean']),
                'cov':   0.0,
                'pv':    cell['pathwise_violation_rate'],
            }
        }
    return out


def merge(main, new, ltt_pivots=None):
    ltt_pivots = ltt_pivots or {}
    out = {b: {} for b in BENCHES}
    for b in BENCHES:
        for a in UNIFORM_ALPHAS:
            merged = {}
            if a in main.get(b, {}):
                merged.update(main[b][a])
            if a in new.get(b, {}):
                merged.update(new[b][a])
            if a in ltt_pivots.get(b, {}):
                merged.update(ltt_pivots[b][a])
            if merged:
                out[b][a] = merged
    return out


# =================================================================
# Shift ablation (MedQA alpha=0.05 + 0.20, GSM8K alpha=0.05) --- 10 methods
# =================================================================

def load_shift_gsm8k_hard():
    """GSM8K alpha=0.05 adversarial shifts: quartile, multi, adversarial,
    window_outrun. Returns {scenario: {method: cell}}."""
    path = f'{RES}/ablation_shift_hard_gsm8k_alpha0.05.json'
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for sc in ['quartile', 'multi', 'adversarial', 'window_outrun']:
        if sc not in d:
            continue
        out[sc] = {}
        for m, cell in d[sc].items():
            out[sc][m] = {
                'risk': float(cell.get('final_risk_mean', 0)),
                'ar':   float(cell.get('final_ar_mean', 0)),
                'maxr': float(cell.get('max_risk_mean', 0)),
                'pv':   cell.get('pathwise_violation_rate', '?'),
            }
    return out


def shift_gsm8k_table(shift):
    """GSM8K alpha=0.05 harder shift scenarios, 10 methods x 4 scenarios."""
    lines = []
    lines.append('\\begin{table}[!ht]')
    lines.append('\\centering\\scriptsize')
    lines.append('\\setlength{\\tabcolsep}{3pt}')
    lines.append('\\begin{tabular}{llrrrc}')
    lines.append('\\toprule')
    lines.append('Scenario & Method & Risk & AR & MaxR & PathV \\\\')
    lines.append('\\midrule')
    scen_order = [
        ('quartile',       'quartile'),
        ('multi',          'multi'),
        ('adversarial',    'adversarial'),
        ('window_outrun',  'window\\_outrun'),
    ]
    for scen_key, scen_label in scen_order:
        if scen_key not in shift:
            continue
        lines.append(f'\\multicolumn{{6}}{{l}}{{\\emph{{Scenario: \\texttt{{{scen_label}}}}}}} \\\\')
        for m in METHOD_ORDER:
            cell = shift[scen_key].get(m)
            if cell is None:
                continue
            nm = m.replace('CSA-RLVR', '\\csa').replace('Mohri-Conf', 'ConfFact')
            risk_s = pct(cell['risk']) if cell['ar'] > 0 else '$0.0\\%$'
            ar_s = pct(cell['ar']) if cell['ar'] > 0 else '$0.0\\%$'
            maxr_s = pct(cell['maxr']) if cell['ar'] > 0 else '$0.0\\%$'
            pv_s = pv_fmt(cell['pv'])
            lines.append(f' & {nm} & {risk_s} & {ar_s} & {maxr_s} & {pv_s} \\\\')
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines.append('\\end{tabular}')
    lines.append('\\caption{\\textbf{Structurally adversarial distribution '
                 'shifts on GSM8K at $\\alphabudget=0.05$} (10 replications each). '
                 'Four scenarios defined in \\texttt{ablate\\_shift\\_hard.py}: '
                 '\\texttt{quartile} streams $n_{\\text{passes}}$ passes of the '
                 'bottom-25\\%-easiest items (err$\\approx 2.7\\%$), then '
                 '$n_{\\text{passes}}$ passes of the top-25\\%-hardest items '
                 '(err$\\approx 15.2\\%$); \\texttt{multi} interleaves 100-item '
                 'blocks of the easy and hard halves; \\texttt{adversarial} '
                 'concatenates all-easy passes, then all-hard passes, then '
                 'repeats the hardest-10\\% block $5\\times$ at the tail; '
                 '\\texttt{window\\_outrun} uses a 509-item easy prefix '
                 '(enough to fill CSA/CRC/Mohri\'s 500-accept calibration '
                 'window) followed by a 250-item hard burst. \\csa achieves '
                 '$0/10$ on every scenario while maintaining meaningful AR '
                 '($71.7\\%$ on \\texttt{multi}, $73.3\\%$ on '
                 '\\texttt{adversarial}); CRC and \textsc{ConfFact} collapse on '
                 '\\texttt{adversarial} (10/10, $\\text{Risk}=5.40\\%>0.05$) '
                 'and \\texttt{window\\_outrun} (CRC 10/10 with '
                 '$\\text{Risk}=14.6\\%$); NEX-Conf violates $10/10$ on '
                 '\\texttt{quartile} and $5/10$ on \\texttt{multi}; the five '
                 'online baselines violate in every scenario.}')
    lines.append('\\label{tab:shift-gsm8k-hard}')
    lines.append('\\end{table}')
    return '\n'.join(lines)


def load_shift_lowalpha():
    """Load MedQA shift stress-test from ablate_shift_lowalpha.py output.
    The file has 3 alphas x 4 scenarios = 12 cells, each with 10 methods.
    Returns {(alpha, scenario): {method: cell}}.
    """
    path = f'{RES}/ablation_shift_lowalpha.json'
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for alpha_key, scenarios in d['results'].items():
        a = float(alpha_key)
        for sc, methods in scenarios.items():
            cell = {}
            for m, src in methods.items():
                cell[m] = {
                    'risk': float(src.get('final_risk_mean', 0)),
                    'ar':   float(src.get('final_ar_mean', 0)),
                    'maxr': float(src.get('max_risk_mean', 0)),
                    'pv':   src.get('pathwise_violation_rate', '?'),
                }
            out[(a, sc)] = cell
    return out


# =================================================================
# DeepSeek cross-model (main pipeline + new baselines)
# =================================================================

def load_deepseek():
    """Returns {bench: {alpha: {method: dict}}}."""
    out = {'medical_deepseek': {}, 'gsm8k_deepseek': {}}
    for tag in out:
        for path in sorted(glob(f'{RES}/{tag}/{tag}_alpha*.json')):
            with open(path) as f:
                d = json.load(f)
            a = float(d['alpha'])
            for m, cell in d['methods'].items():
                out[tag].setdefault(a, {})[m] = {
                    'risk': float(cell.get('final_risk_mean', 0)),
                    'ar':   float(cell.get('final_ar_mean', 0)),
                    'maxr': float(cell.get('max_risk_mean', 0)),
                    'prec': float(cell.get('precision_mean', 0)),
                    'pv':   cell.get('pathwise_violation_rate', '?'),
                }
    nb_path = f'{RES}/deepseek_new_baselines_summary.json'
    if os.path.exists(nb_path):
        with open(nb_path) as f:
            nb = json.load(f)
        for tag in ('medical_deepseek', 'gsm8k_deepseek'):
            if tag not in nb:
                continue
            for ak, cell in nb[tag]['per_alpha'].items():
                a = float(ak)
                for m in ('CRC', 'NEX-Conf', 'Mohri-Conf'):
                    out[tag].setdefault(a, {})[m] = {
                        'risk': float(cell[m]['final_risk_mean']),
                        'ar':   float(cell[m]['final_ar_mean']),
                        'maxr': float(cell[m]['max_risk_mean']),
                        'prec': 1.0 - float(cell[m]['final_risk_mean']),
                        'pv':   cell[m]['pathwise_violation_rate'],
                    }
    return out


# =================================================================
# LaTeX formatters
# =================================================================

def pct(x, decimals=1):
    if x is None:
        return '--'
    return f'${x*100:.{decimals}f}\\%$'


def pv_fmt(pv_str, alpha_val=None):
    """Format path-wise violation rate, color-code failures."""
    if pv_str in (None, '?'):
        return '--'
    n, t = pv_str.split('/')
    n, t = int(n), int(t)
    if n == 0:
        return f'$\\mathbf{{{pv_str}}}$'
    if n == t:
        return f'${{\\color{{csared}}\\mathbf{{{pv_str}}}}}$'
    return f'${{\\color{{csared}}{pv_str}}}$'


# =================================================================
# Table 1: per-benchmark small tables (APPENDIX and main paper)
# =================================================================

def small_table_per_bench(combined, bench, alphas=UNIFORM_ALPHAS):
    """One table: 10 methods x len(alphas) alphas on this benchmark.
    Each cell shows Risk / AR / PathV."""
    label, model, N, err, pivot = BENCH_META[bench]
    lines = []
    lines.append('\\begin{table}[!ht]')
    lines.append('\\centering\\scriptsize')
    lines.append('\\setlength{\\tabcolsep}{3pt}')
    # 1 col for method + 3 cols (Risk/AR/PathV) per alpha
    coldef = 'l' + ''.join(['rrc' for _ in alphas])
    lines.append(f'\\begin{{tabular}}{{{coldef}}}')
    lines.append('\\toprule')
    # Top header: alpha multicol
    head = ['']
    for a in alphas:
        head.append(f'\\multicolumn{{3}}{{c}}{{$\\alphabudget={a:.2f}$}}')
    lines.append(' & '.join(head) + ' \\\\')
    # cmidrule for each alpha block
    rule_parts = []
    for i, _ in enumerate(alphas):
        start = 2 + 3 * i
        end = start + 2
        rule_parts.append(f'\\cmidrule(lr){{{start}-{end}}}')
    lines.append(''.join(rule_parts))
    # Sub header
    sub = ['Method']
    for _ in alphas:
        sub.extend(['Risk', 'AR', 'PathV'])
    lines.append(' & '.join(sub) + ' \\\\')
    lines.append('\\midrule')
    for m in METHOD_ORDER:
        row = [m.replace('CSA-RLVR', '\\csa').replace('Mohri-Conf', 'ConfFact')]
        missing = True
        for a in alphas:
            cell = combined.get(bench, {}).get(a, {}).get(m)
            if cell is None:
                row.extend(['--', '--', '--'])
            else:
                missing = False
                risk_s = pct(cell['risk']) if cell['ar'] > 0 else '--'
                ar_s = pct(cell['ar'])
                if cell['ar'] == 0:
                    ar_s = '$0.0\\%$'
                pv_s = pv_fmt(cell['pv'])
                row.extend([risk_s, ar_s, pv_s])
        lines.append(' & '.join(row) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append(f'\\caption{{\\textbf{{{label}}} ({model}, $N={N:,}$, '
                 f'Err$={err:.1f}\\%$, 10 reps). '
                 f'All 10 methods at $\\alphabudget\\in\\{{0.05,0.10,0.15,0.20,0.25,0.30\\}}$. '
                 f'Pivotal $\\alphabudget={pivot}$. '
                 f'Risk = final empirical risk on accepted items; '
                 f'AR = accept rate; '
                 f'PathV = pathwise-violation rate over 10 streams. '
                 f'Bold PathV = zero violations; red = partial or full '
                 f'violations. A dash in Risk column indicates principled '
                 f'refusal (AR$=0\\%$).}}')
    lines.append(f'\\label{{tab:full-{bench}}}')
    lines.append('\\end{table}')
    return '\n'.join(lines)


# =================================================================
# Table 2: main consolidated pivotal-alpha table
# =================================================================

def consolidated_pivotal(combined):
    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering\\small')
    lines.append('\\setlength{\\tabcolsep}{3pt}')
    lines.append('\\begin{tabular}{llrrrc}')
    lines.append('\\toprule')
    lines.append('Benchmark & Method & Risk & AR & Prec. & PathV \\\\')
    lines.append('\\midrule')
    for b in BENCHES:
        label, _, _, _, pivot = BENCH_META[b]
        first = True
        for m in METHOD_ORDER:
            cell = combined.get(b, {}).get(pivot, {}).get(m)
            if cell is None:
                continue
            bench_str = label + f' ($\\alpha={pivot}$)' if first else ''
            first = False
            nm = m.replace('CSA-RLVR', '\\csa').replace('Mohri-Conf', 'ConfFact')
            risk_s = pct(cell['risk']) if cell['ar'] > 0 else '$0.0\\%$'
            ar_s = pct(cell['ar']) if cell['ar'] > 0 else '$0.0\\%$'
            prec_s = pct(cell['prec']) if cell['ar'] > 0 else '--'
            pv_s = pv_fmt(cell['pv'])
            lines.append(f'{bench_str} & {nm} & {risk_s} & {ar_s} & {prec_s} & {pv_s} \\\\')
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines.append('\\end{tabular}')
    lines.append('\\caption{\\textbf{Head-to-head comparison at each benchmark\'s '
                 'pivotal $\\alphabudget$.} 10 methods across 8 verifiable-reward '
                 'benchmarks; 10 replications each. PathV is the number of '
                 'streams on which the running empirical risk exceeded $\\alphabudget$. '
                 '\\csa achieves $0/10$ on every cell; every online baseline '
                 '(Always-Act, Fixed-Threshold, Naive-Tuning, ACI, SAOCP) violates '
                 'on $\\ge 10/10$. The three offline/conformal SOTA methods '
                 '(LTT~\\cite{ltt}, CRC~\\cite{crc}, ConfFact~\\cite{mohri}) often '
                 'achieve validity by principled refusal ($\\text{AR}=0\\%$). '
                 'NEX-Conformal~\\cite{nexconf} is non-refusing but violates pathwise '
                 'on 4/8 pivotal cells. Numbers extracted verbatim from '
                 '\\texttt{results/<bench>/<bench>\\_alpha<X>.json} and '
                 '\\texttt{results/new\\_baselines\\_summary.json}.}')
    lines.append('\\label{tab:pivotal-head-to-head}')
    lines.append('\\label{tab:baseline-compare}')  # back-compat with old \cref calls
    lines.append('\\end{table}')
    return '\n'.join(lines)


# =================================================================
# Table 3: shift ablation summary (all 10 methods x 4 scenarios x 3 cells)
# =================================================================

def shift_table(shift):
    """MedQA shift stress-test: 3 alphas x 4 scenarios x 10 methods.
    Loaded from results/ablation_shift_lowalpha.json (produced by
    ablate_shift_lowalpha.py).  Uses 2-decimal precision to match the
    terminal output of the driver script verbatim."""
    lines = []
    lines.append('\\begin{table}[!ht]')
    lines.append('\\centering\\scriptsize')
    lines.append('\\setlength{\\tabcolsep}{3pt}')
    lines.append('\\begin{tabular}{llrrrc}')
    lines.append('\\toprule')
    lines.append('$\\alphabudget$ / scenario & Method & Risk & AR & MaxR & PathV \\\\')
    lines.append('\\midrule')
    alphas_in_order = sorted({a for (a, _) in shift.keys()})
    scenarios_in_order = ['iid', 'easy_hard', 'quartile_rev', 'window_outrun']
    for a in alphas_in_order:
        for sc in scenarios_in_order:
            key = (a, sc)
            if key not in shift:
                continue
            sc_tex = sc.replace('_', '\\_')
            lines.append(f'\\multicolumn{{6}}{{l}}{{\\emph{{$\\alphabudget={a:g}$,  scenario={sc_tex}}}}} \\\\')
            for m in METHOD_ORDER:
                cell = shift[key].get(m)
                if cell is None:
                    continue
                nm = m.replace('CSA-RLVR', '\\csa').replace('Mohri-Conf', 'ConfFact')
                # 2-decimal percentages matching the driver's stdout
                risk_s = pct(cell['risk'], decimals=2)
                ar_s   = pct(cell['ar'],   decimals=2)
                maxr_s = pct(cell['maxr'], decimals=2)
                pv_s = pv_fmt(cell['pv'])
                lines.append(f' & {nm} & {risk_s} & {ar_s} & {maxr_s} & {pv_s} \\\\')
            lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines.append('\\end{tabular}')
    lines.append('\\caption{\\textbf{MedQA shift stress-test} '
                 '(\\texttt{ablate\\_shift\\_lowalpha.py}; 10 reps, '
                 '$n_{\\text{passes}}=30$, $B=500$; MedQA EVAL $N=1{,}018$, '
                 '$\\mathrm{err}_{\\text{easy}}=17.1\\%$, '
                 '$\\mathrm{err}_{\\text{hard}}=46.4\\%$, shift ratio '
                 '$2.71\\times$). Twelve cells: $\\alphabudget\\in\\{0.05,0.10,0.20\\}$ '
                 '$\\times$ \\{\\texttt{iid}, \\texttt{easy\\_hard}, '
                 '\\texttt{quartile\\_rev}, \\texttt{window\\_outrun}\\}. '
                 '\\csa achieves $\\PathV=0/10$ in all 12 cells (refuses for '
                 '$\\alphabudget\\le 0.10$, non-refusing with $\\AR\\approx 39\\%$ '
                 'at $\\alphabudget=0.20$). At $\\alphabudget=0.20$, '
                 '\\textsc{LTT} and \\textsc{ConfFact} collapse to $10/10$ on '
                 '\\texttt{easy\\_hard}/\\texttt{quartile\\_rev}/\\texttt{window\\_outrun} '
                 'with $\\mathrm{Risk}=31.8\\%$---both lock their threshold on the '
                 'easy half and the hard-phase risk exceeds $\\alphabudget$ on every '
                 'replication. \\textsc{CRC} remains $0/10$ on all MedQA cells but '
                 'this is largely ``valid by refusal'' (\\cref{tab:shift-gsm8k-hard} '
                 'for cases where CRC does \\emph{not} refuse and then violates).}')
    lines.append('\\label{tab:shift-all}')
    lines.append('\\end{table}')
    return '\n'.join(lines)


# =================================================================
# Table 4: DeepSeek cross-model full sweep
# =================================================================

def deepseek_table(ds):
    lines = []
    lines.append('\\begin{table}[!ht]')
    lines.append('\\centering\\scriptsize')
    lines.append('\\setlength{\\tabcolsep}{3pt}')
    lines.append('\\begin{tabular}{llrrrc}')
    lines.append('\\toprule')
    lines.append('Benchmark & Method & Risk & AR & MaxR & PathV \\\\')
    lines.append('\\midrule')
    label_map = {
        'medical_deepseek': 'MedQA + DeepSeek-R1-Distill-Qwen-7B (acc=65.8\\%)',
        'gsm8k_deepseek':   'GSM8K + DeepSeek-R1-Distill-Qwen-7B (acc=87.8\\%)',
    }
    piv_map = {'medical_deepseek': 0.20, 'gsm8k_deepseek': 0.10}
    for tag, rows in ds.items():
        lines.append(f'\\multicolumn{{6}}{{l}}{{\\emph{{{label_map[tag]}}}}} \\\\')
        alphas = sorted(rows.keys())
        for a in alphas:
            bstr = f'$\\alpha={a:.3f}$'
            first = True
            for m in METHOD_ORDER:
                cell = rows[a].get(m)
                if cell is None:
                    continue
                bench_s = bstr if first else ''
                first = False
                nm = m.replace('CSA-RLVR', '\\csa').replace('Mohri-Conf', 'ConfFact')
                risk_s = pct(cell['risk']) if cell['ar'] > 0 else '$0.0\\%$'
                ar_s = pct(cell['ar']) if cell['ar'] > 0 else '$0.0\\%$'
                maxr_s = pct(cell.get('maxr', 0)) if cell['ar'] > 0 else '$0.0\\%$'
                pv_s = pv_fmt(cell['pv'])
                lines.append(f'{bench_s} & {nm} & {risk_s} & {ar_s} & {maxr_s} & {pv_s} \\\\')
            lines.append('\\midrule')
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines.append('\\end{tabular}')
    lines.append('\\caption{\\textbf{Cross-model ablation} with '
                 'DeepSeek-R1-Distill-Qwen-7B as base model. '
                 'All 10 methods, full $\\alphabudget$ grid, 10 replications. '
                 'DeepSeek was not used to pick any CSA hyperparameter; the '
                 'same configuration as the main 7-model pipeline transfers '
                 'directly. \\csa retains $0/10$ pathwise violations at every '
                 'cell while the base model differs entirely from the ones '
                 'used to construct the calibrated CSVs.}')
    lines.append('\\label{tab:cross-model-deepseek}')
    lines.append('\\end{table}')
    return '\n'.join(lines)


# =================================================================
# Main
# =================================================================

def main():
    print('Loading main 6-method pipeline...')
    main_p = load_main_pipeline()
    print('Loading new baselines (CRC/NEX/Mohri)...')
    new_p = load_new_baselines()
    print('Loading LTT pivotal numbers...')
    ltt_p = load_ltt_pivots()
    print('Merging...')
    combined = merge(main_p, new_p, ltt_p)
    print('Loading MedQA shift stress-test (ablation_shift_lowalpha.json)...')
    shift = load_shift_lowalpha()
    print('Loading GSM8K adversarial shift scenarios...')
    shift_gsm = load_shift_gsm8k_hard()
    print('Loading DeepSeek cross-model...')
    ds = load_deepseek()

    # -------- Sanity report --------
    print('\n=== Sanity: combined coverage ===')
    for b in BENCHES:
        for a in UNIFORM_ALPHAS:
            cell = combined.get(b, {}).get(a, {})
            nm = len(cell)
            print(f'  {b:<12s}  a={a:.2f}   {nm} methods')

    # -------- Emit .tex --------
    print('\nWriting paper_tables/*.tex ...')

    # Per-benchmark small tables (appendix)
    for b in BENCHES:
        tex = small_table_per_bench(combined, b)
        with open(f'{OUT}/table_{b}.tex', 'w') as f:
            f.write(tex + '\n')
        print(f'  paper_tables/table_{b}.tex')

    # Consolidated pivotal head-to-head
    with open(f'{OUT}/table_pivotal.tex', 'w') as f:
        f.write(consolidated_pivotal(combined) + '\n')
    print(f'  paper_tables/table_pivotal.tex')

    # Shift-ablation master
    with open(f'{OUT}/table_shift.tex', 'w') as f:
        f.write(shift_table(shift) + '\n')
    print(f'  paper_tables/table_shift.tex')

    # GSM8K adversarial shifts (CSA succeeds)
    with open(f'{OUT}/table_shift_gsm8k.tex', 'w') as f:
        f.write(shift_gsm8k_table(shift_gsm) + '\n')
    print(f'  paper_tables/table_shift_gsm8k.tex')

    # DeepSeek cross-model
    with open(f'{OUT}/table_deepseek.tex', 'w') as f:
        f.write(deepseek_table(ds) + '\n')
    print(f'  paper_tables/table_deepseek.tex')

    # Dump verified numbers for future audits
    with open(f'{OUT}/_verified_numbers.json', 'w') as f:
        json.dump({
            'combined_main_plus_new': combined,
            'shift_medqa_lowalpha': {f'alpha{a}_{sc}': v
                                      for (a, sc), v in shift.items()},
            'shift_gsm8k_hard': shift_gsm,
            'deepseek': ds,
        }, f, indent=2, default=str)
    print(f'  paper_tables/_verified_numbers.json')

    print('\nDone.')


if __name__ == '__main__':
    main()
