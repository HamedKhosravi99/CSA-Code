# Code for: Conformal Selective Acting: Anytime-Valid Risk Control for RLVR-Trained LLMs

## Structure

### Core algorithm
- `csa_core.py` -- CSA controller (Algorithm 1): e-process per threshold on a Bonferroni grid
- `calibrate_scores.py` -- Isotonic-regression calibration for surrogate confidence scores
- `score_options.py` -- Fast option-scoring pass for MCQ domains
- `score_selfconsistency.py` -- K-sample self-consistency scorer (GPU, vLLM)
- `metrics.py` -- Metrics computation for CAP-style benchmarks
- `mlx_backend.py` -- Apple-Silicon MLX backend for 4-bit quantized 7B inference

### Domain framework (`domains/`)
- `domains/base.py` -- Abstract base classes (DomainStream, DomainVerifier)
- `domains/runner.py` -- Generic CSA experiment runner
- `domains/surrogate.py` -- Online surrogate (logistic regression)
- `domains/baselines.py` -- Heuristic baselines (Always-Act, Fixed-Threshold, Naive-Tuning)
- `domains/plotting.py` -- Cross-domain plotting utilities

Eight benchmark domains (each has `stream.py` + `inference.py`):
`medical/`, `pubmedqa/`, `tatqa/`, `mednli/`, `gsm8k/`, `headqa/`, `arc/`, `casehold/`

### Experiment orchestrators
- `run_domains.py` -- Main orchestrator for all 8-benchmark CSA experiments
- `run_extra.py` -- Orchestrator for GSM8K, ARC, PubMedQA runs
- `run_tier2.py` -- Tier-2 benchmark experiments

### Baselines
- `principled_baselines.py` -- ACI and SAOCP adapted for selective acting
- `test_new_baselines.py` -- CRC, NEX-Conf, ConfFact (post-hoc conformal SOTA)
- `arcps_adapter.py` -- Passive A-RCPS adapter (Xu et al., NeurIPS 2024)
- `run_arcps_medqa.py` -- A-RCPS comparison on MedQA
- `active-rcps/` -- A-RCPS reference implementation

### Ablation studies
- `ablate_hyperparams.py` -- Sensitivity to delta, burn-in, grid size
- `ablate_shift.py` -- Distribution-shift ablation (mild)
- `ablate_shift_hard.py` -- Harsh distribution shift (16 adversarial cells)
- `ablate_shift_lowalpha.py` -- Low-alpha interleaved shift
- `ablate_shift_new_baselines.py` -- Shift ablation for CRC/NEX-Conf/ConfFact

### LTT comparison
- `run_ltt_pivots.py` -- LTT at per-benchmark pivotal alpha
- `run_ltt_grid.py` -- LTT on full alpha grid
- `merge_ltt_grid.py` -- Merge LTT grid results
- `run_tight_alpha_gsm8k_arc.py` -- Tight alpha runs for GSM8K/ARC
- `merge_tight_alpha.py` -- Merge tight alpha results

### Live RLVR loop
- `live_medqa.py` -- Live RLVR on MedQA with online LoRA (Apple Silicon / MLX)
- `live_medqa_cuda.py` -- CUDA version
- `live_medqa_replay.py` -- Replay from saved checkpoints

### Figure and table generation
- `build_paper_figures.py` -- violation_heatmap, pareto_frontier, scoreboard figures
- `build_paper_tables.py` -- All LaTeX benchmark tables from verified JSON
- `build_risk_allmethods_2x4.py` -- 2x4 risk grid (all methods, all benchmarks)
- `build_option_radar_small.py` -- Compact radar chart
- `build_live_trajectory_with_ar.py` -- Live RLVR trajectory with action rate
- `build_phase_budget_allmethods.py` -- Phase-budget figure (all methods)
- `build_summary_figures.py` -- Cross-domain summary figures
- `build_riskar_common.py` -- Shared utilities for risk/AR plots
- `build_option_common.py` -- Shared utilities for option plots

## Dependencies

See `requirements.txt`. Key packages: numpy, matplotlib, scikit-learn, torch, transformers.

## Reproducing results

1. Run domain experiments: `python run_domains.py`
2. Run baselines: `python principled_baselines.py && python test_new_baselines.py`
3. Run ablations: `python ablate_shift_hard.py`
4. Generate tables: `python build_paper_tables.py`
5. Generate figures: `python build_paper_figures.py && python build_risk_allmethods_2x4.py`

All results are written to `../data/results/` as JSON; table/figure builders read from there.
