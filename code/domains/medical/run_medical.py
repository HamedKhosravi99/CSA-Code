"""
Medical domain CSA replay experiment.

Loads pre-computed inference CSV and runs CSA + baselines.
Runs locally on CPU (no GPU needed).

Usage:
    python domains/medical/run_medical.py \
        --csv results/medical_inference.csv \
        --alpha 0.10 \
        --n_reps 20
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from csa_core import CSAConfig
from domains.runner import run_domain_experiment
from domains.medical.stream import MedQAStream


def main():
    parser = argparse.ArgumentParser(description="Medical CSA replay")
    parser.add_argument('--csv', required=True,
                        help='Path to medical inference CSV')
    parser.add_argument('--alpha', type=float, default=0.10)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--n_reps', type=int, default=20)
    parser.add_argument('--n_passes', type=int, default=5)
    parser.add_argument('--output_dir', default='results/medical')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Loading medical stream from {args.csv}...")
    stream = MedQAStream(args.csv)
    print(f"  {len(stream)} items loaded")

    config = CSAConfig(
        alpha=args.alpha,
        delta=args.delta,
        grid_size=50,
        grid_min=0.02,
        grid_max=0.98,
        single_epoch=True,
    )

    experiment_name = f"medical_alpha{args.alpha:.2f}"

    results = run_domain_experiment(
        stream=stream,
        csa_config=config,
        n_reps=args.n_reps,
        n_passes=args.n_passes,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        seed=args.seed,
    )

    # Also run with alpha=0.20 if primary is 0.10
    if args.alpha == 0.10:
        print("\n--- Running secondary alpha=0.20 ---")
        config_20 = CSAConfig(
            alpha=0.20, delta=args.delta, grid_size=50,
            grid_min=0.02, grid_max=0.98, single_epoch=True)
        run_domain_experiment(
            stream=stream, csa_config=config_20,
            n_reps=args.n_reps, n_passes=args.n_passes,
            output_dir=args.output_dir,
            experiment_name="medical_alpha0.20",
            seed=args.seed)


if __name__ == '__main__':
    main()
