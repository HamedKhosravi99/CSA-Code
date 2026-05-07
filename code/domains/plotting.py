"""
Cross-domain plotting for CSA experiments.

Generates publication-quality figures:
  1. Per-domain risk curves (4 panels)
  2. Per-domain action rate curves (4 panels)
  3. Cross-domain summary bar chart
  4. Per-domain certification dynamics

Usage:
    python domains/plotting.py --results_dir results/
"""

import argparse
import json
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Consistent styling
DOMAIN_LABELS = {
    'medical': 'Medical (MedQA)',
    'financial': 'Financial (FinQA)',
    'legal': 'Legal (LegalBench)',
    'agents': 'Agents (ALFWorld)',
}

METHOD_COLORS = {
    'CSA-RLVR': '#2176AE',
    'ACI': '#E8871E',
    'SAOCP': '#57B894',
    'Always-Act': '#D64045',
    'Fixed-Threshold': '#999999',
    'Naive-Tuning': '#B07CC6',
}

METHOD_STYLES = {
    'CSA-RLVR': {'linewidth': 2.5, 'linestyle': '-'},
    'ACI': {'linewidth': 1.8, 'linestyle': '--'},
    'SAOCP': {'linewidth': 1.8, 'linestyle': '-.'},
    'Always-Act': {'linewidth': 1.2, 'linestyle': ':'},
    'Fixed-Threshold': {'linewidth': 1.2, 'linestyle': ':'},
    'Naive-Tuning': {'linewidth': 1.2, 'linestyle': '--'},
}


def load_domain_results(results_dir: str, domain: str, alpha: float):
    """Load experiment JSON for a domain and alpha."""
    json_path = os.path.join(results_dir, domain,
                             f"{domain}_alpha{alpha:.2f}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def plot_risk_curves_panel(results_dir: str, alpha: float = 0.10,
                           output_path: str = None):
    """Generate 4-panel risk curve figure (one per domain)."""
    assert HAS_MPL, "matplotlib required"

    domains = ['medical', 'financial', 'legal', 'agents']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    show_methods = ['CSA-RLVR', 'ACI', 'SAOCP', 'Always-Act']

    for i, domain in enumerate(domains):
        ax = axes[i]
        data = load_domain_results(results_dir, domain, alpha)

        if data is None:
            ax.text(0.5, 0.5, f'{domain}\n(no data)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='gray')
            ax.set_title(DOMAIN_LABELS.get(domain, domain))
            continue

        methods = data.get('methods', {})
        T = data.get('T', 0)
        x = np.arange(1, T + 1)

        for method in show_methods:
            if method not in methods:
                continue
            curve = methods[method].get('mean_risk_curve', [])
            if not curve:
                continue
            y = np.array(curve)
            ax.plot(x[:len(y)], y,
                    color=METHOD_COLORS.get(method, 'gray'),
                    label=method,
                    **METHOD_STYLES.get(method, {}))

        # Alpha line
        ax.axhline(y=alpha, color='red', linewidth=1.0, linestyle='--',
                    alpha=0.7, label=f'$\\alpha={alpha}$')

        ax.set_title(DOMAIN_LABELS.get(domain, domain), fontsize=13)
        ax.set_xlabel('Round $t$')
        ax.set_ylabel('Selective Risk')
        ax.set_ylim(-0.02, min(0.5, alpha * 4))
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=9, loc='upper right')

    fig.suptitle(f'Selective Risk Curves ($\\alpha={alpha}$)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path is None:
        output_path = os.path.join(results_dir,
                                   f'risk_curves_alpha{alpha:.2f}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved risk curves: {output_path}")


def plot_action_rate_panel(results_dir: str, alpha: float = 0.10,
                           output_path: str = None):
    """Generate 4-panel action rate figure."""
    assert HAS_MPL, "matplotlib required"

    domains = ['medical', 'financial', 'legal', 'agents']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    show_methods = ['CSA-RLVR', 'ACI', 'SAOCP']

    for i, domain in enumerate(domains):
        ax = axes[i]
        data = load_domain_results(results_dir, domain, alpha)

        if data is None:
            ax.text(0.5, 0.5, f'{domain}\n(no data)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='gray')
            ax.set_title(DOMAIN_LABELS.get(domain, domain))
            continue

        methods = data.get('methods', {})
        T = data.get('T', 0)
        x = np.arange(1, T + 1)

        for method in show_methods:
            if method not in methods:
                continue
            curve = methods[method].get('mean_ar_curve', [])
            if not curve:
                continue
            y = np.array(curve)
            ax.plot(x[:len(y)], y,
                    color=METHOD_COLORS.get(method, 'gray'),
                    label=method,
                    **METHOD_STYLES.get(method, {}))

        ax.set_title(DOMAIN_LABELS.get(domain, domain), fontsize=13)
        ax.set_xlabel('Round $t$')
        ax.set_ylabel('Action Rate')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=9, loc='lower right')

    fig.suptitle(f'Action Rate Curves ($\\alpha={alpha}$)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path is None:
        output_path = os.path.join(results_dir,
                                   f'action_rates_alpha{alpha:.2f}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved action rates: {output_path}")


def plot_cross_domain_bar(results_dir: str, alpha: float = 0.10,
                          output_path: str = None):
    """Generate cross-domain bar chart comparing final risk and AR."""
    assert HAS_MPL, "matplotlib required"

    domains = ['medical', 'financial', 'legal', 'agents']
    show_methods = ['CSA-RLVR', 'ACI', 'SAOCP', 'Always-Act']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    domain_labels = [DOMAIN_LABELS.get(d, d) for d in domains]
    x = np.arange(len(domains))
    width = 0.18

    for method_idx, method in enumerate(show_methods):
        risks = []
        ars = []
        risk_stds = []

        for domain in domains:
            data = load_domain_results(results_dir, domain, alpha)
            if data and method in data.get('methods', {}):
                m = data['methods'][method]
                risks.append(m['final_risk_mean'])
                risk_stds.append(m['final_risk_std'])
                ars.append(m['final_ar_mean'])
            else:
                risks.append(0)
                risk_stds.append(0)
                ars.append(0)

        offset = (method_idx - len(show_methods) / 2 + 0.5) * width
        color = METHOD_COLORS.get(method, 'gray')

        ax1.bar(x + offset, risks, width, yerr=risk_stds,
                label=method, color=color, alpha=0.85,
                capsize=3, error_kw={'linewidth': 1})
        ax2.bar(x + offset, ars, width,
                label=method, color=color, alpha=0.85)

    ax1.axhline(y=alpha, color='red', linewidth=1.5, linestyle='--',
                alpha=0.8, label=f'$\\alpha={alpha}$')
    ax1.set_ylabel('Final Selective Risk')
    ax1.set_xticks(x)
    ax1.set_xticklabels(domain_labels, fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis='y', alpha=0.3)

    ax2.set_ylabel('Final Action Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels(domain_labels, fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'Cross-Domain Comparison ($\\alpha={alpha}$)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if output_path is None:
        output_path = os.path.join(results_dir,
                                   f'cross_domain_bar_alpha{alpha:.2f}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-domain bar: {output_path}")


def generate_all_figures(results_dir: str):
    """Generate all publication figures."""
    assert HAS_MPL, "matplotlib not installed. Run: pip install matplotlib"

    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    for alpha in [0.10, 0.20]:
        plot_risk_curves_panel(
            results_dir, alpha,
            os.path.join(figures_dir, f'risk_curves_alpha{alpha:.2f}.pdf'))
        plot_action_rate_panel(
            results_dir, alpha,
            os.path.join(figures_dir, f'action_rates_alpha{alpha:.2f}.pdf'))
        plot_cross_domain_bar(
            results_dir, alpha,
            os.path.join(figures_dir, f'cross_domain_bar_alpha{alpha:.2f}.pdf'))

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate CSA figures")
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()
    generate_all_figures(args.results_dir)
