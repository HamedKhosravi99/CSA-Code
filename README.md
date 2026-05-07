# Conformal Selective Acting (CSA)

> **Anytime-valid, pathwise selective-risk control for RLVR-trained LLMs.**
> Anonymous code release accompanying a double-blind submission.

## What this paper proposes

A regulated operator deploys a local specialist LLM (medical Q&A, legal-citation
review, financial reporting, …) under a per-deployment error budget α on
released outputs, evaluated against a verifier on **this** deployment's stream,
**simultaneously at every wall-clock round** — no pooling, no waiting for a
long-run average.

Existing risk-control wrappers cannot deliver this guarantee on adaptive,
online-updated RLVR streams: offline conformal-risk methods require
exchangeability; online-conformal methods bound only long-run averages;
non-exchangeable extensions are marginally valid; the closest anytime wrapper
controls marginal rather than selective risk.

**Conformal Selective Acting (CSA)** fills the empty cell of the framework: it
maintains a Ville-type e-process per candidate score threshold on a Bonferroni
grid, evaluated against the RLVR filtration. We prove

1. an **anytime-pathwise selective-risk bound** R<sub>T</sub><sup>act</sup> ≤ α + O(N<sub>T</sub><sup>−1/2</sup>),
2. **rate-optimal certification** matching Θ(η̄<sup>−2</sup> log(1/δ)),
3. a **horizon-independent release-rate gap**.

Across **eight specialist benchmarks**, **sixteen adversarial
distribution-shift cells**, and **five live RLVR cells with online LoRA**
across four base models in three architecture families, CSA is the only
method among the ten directly compared that satisfies both pathwise validity
and non-refusing deployment on every cell.

## Repository layout

```
.
├── code/                              source (NumPy-only controller, ~300 lines)
│   ├── csa_core.py                    CSA controller (Algorithm 1)
│   ├── domains/                       per-benchmark stream and verifier
│   │   ├── medical/  pubmedqa/  tatqa/  mednli/
│   │   ├── gsm8k/    headqa/    arc/   casehold/
│   │   ├── runner.py                  generic experiment runner
│   │   ├── surrogate.py               isotonic-calibrated logistic surrogate
│   │   └── baselines.py               heuristic baselines (Always/Fixed/Naive)
│   ├── active-rcps/                   third-party A-RCPS reference port
│   ├── live_<bench>_cuda.py           live RLVR loop with online LoRA
│   ├── ablate_*.py                    hyperparameter / shift / split ablations
│   ├── run_*.py                       experiment orchestrators
│   ├── principled_baselines.py        ACI / SAOCP for selective acting
│   ├── test_new_baselines.py          CRC / NEX-Conf / Mohri implementations
│   ├── build_paper_figures.py         regenerates the figures
│   ├── build_paper_tables.py          regenerates the LaTeX tables
│   ├── README.md                      module-level documentation
│   └── requirements.txt               Python dependencies
│
├── data/results/                      aggregate JSON summaries
│   ├── _paper_data.json               headline numbers for every cell
│   ├── _verified_numbers.json         spot-check snapshot consumed by figures
│   ├── cross_domain_summary.json      cross-benchmark summary
│   ├── shift/                         16 adversarial-ordering cells (App. F.5)
│   │   ├── ablation_shift.json
│   │   ├── ablation_shift_hard.json   MedQA at α=0.20
│   │   ├── ablation_shift_hard_alpha0.05.json
│   │   ├── ablation_shift_hard_gsm8k_alpha0.05.json
│   │   ├── ablation_shift_lowalpha.json    interleaved low-α variants
│   │   └── ablation_shift_new_baselines.json
│   ├── ablations/                     hyperparameter / split / GPU sensitivity
│   │   ├── ablation_hyperparams.json
│   │   ├── ablation_gpu_summary.json
│   │   └── split_sensitivity_summary.json
│   ├── baselines/                     extended baselines and cross-model
│   │   ├── new_baselines_summary.json     CRC / NEX-Conf / Mohri
│   │   ├── deepseek_new_baselines_summary.json   cross-model on DeepSeek-R1
│   │   └── arcps_medical_alpha0.20.json   A-RCPS port on MedQA
│   └── ltt/                           LTT comparison runs
│       ├── ltt_grid_summary.json
│       ├── ltt_pivotal_summary.json
│       └── tight_alpha_gsm8k_arc.json
│
├── figures/                           PDF + PNG output of build_paper_figures.py
└── paper_tables/                      LaTeX output of build_paper_tables.py
```

## Quick start

```bash
pip install -r code/requirements.txt
python code/build_paper_figures.py    # → figures/*.pdf, figures/*.png
python code/build_paper_tables.py     # → paper_tables/*.tex
```

Both scripts run on CPU in seconds and consume only the aggregate JSONs in
`data/results/`. No GPU required for regeneration.

## Reproducing the live RLVR experiments

The live RLVR cells (Section 6.2 of the paper) require a single GPU and run
in 4-bit NF4 with vLLM + PEFT:

| Cell      | Script                      | GPU             | Wall-clock |
|-----------|-----------------------------|-----------------|-----------:|
| MATH      | `code/live_medqa_cuda.py`   | A100-80GB       | ~4 h       |
| MedQA     | `code/live_medqa_cuda.py`   | A100-80GB       | ~30.5 h    |
| HEAD-QA   | (live HEAD-QA pipeline)     | H200-141GB      | ~30.5 h    |
| ARC-C     | `code/live_arc_cuda.py`     | H200-141GB      | ~6 h       |
| CaseHOLD  | `code/live_casehold_cuda.py`| H200-141GB      | ~8 h       |

Hyperparameters are pinned in `code/csa_core.py` and reused across all
benchmarks; only the stream length scales with the per-benchmark evaluation
set size. No per-benchmark grid search or δ-tuning is performed.

## Module documentation

`code/README.md` documents every script and module in detail.

## License

MIT (see `LICENSE`).
