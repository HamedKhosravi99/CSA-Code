# Conformal Selective Acting (CSA)

Anonymous code and data release accompanying a double-blind submission. This
repository contains the controller, surrogate, baselines, ablation scripts, and
per-replication JSON outputs used to produce every table and figure in the
paper.

## Layout

```
.
├── code/              source for the controller, baselines, and experiments
│   ├── csa_core.py        CSA controller (Algorithm 1): per-threshold
│   │                      e-process on a Bonferroni grid
│   ├── domains/           per-benchmark stream + verifier
│   │                      (medical, pubmedqa, tatqa, mednli, gsm8k,
│   │                       headqa, arc, casehold)
│   ├── active-rcps/       reference port of A-RCPS used in App. F.8
│   ├── live_<bench>_cuda.py   live RLVR loop with online LoRA per cell
│   ├── ablate_*.py            hyperparameter and distribution-shift ablations
│   ├── run_*.py               experiment orchestrators
│   ├── build_paper_*.py       regenerates the LaTeX tables and PDF figures
│   └── README.md              detailed module-level documentation
├── data/
│   └── results/        per-(benchmark, alpha, method, seed) JSON outputs
│                       (28 subfolders, ~750 MB total)
└── docs/               reproducibility notes
```

## Reproducing the paper

All numbers in the paper are produced from `data/results/` via deterministic
post-processing scripts. To regenerate them:

```bash
pip install -r code/requirements.txt
python code/build_paper_tables.py    # regenerates per-benchmark LaTeX tables
python code/build_paper_figures.py   # regenerates the figures
```

The full experiment grid (eight specialist benchmarks × six α values × ten
methods × ten replications) takes a few hours on CPU once the JSON outputs are
loaded. The live RLVR cells (Section 6.2 in the paper) require a single GPU:

| Cell      | Script                  | GPU             | Wall-clock |
|-----------|-------------------------|-----------------|-----------:|
| MATH      | `live_medqa_cuda.py`    | A100-80GB       | 4 h        |
| MedQA     | `live_medqa_cuda.py`    | A100-80GB       | 30.5 h     |
| HEAD-QA   | (live HEAD-QA pipeline) | H200-141GB      | 30.5 h     |
| ARC-C     | `live_arc_cuda.py`      | H200-141GB      | 6 h        |
| CaseHOLD  | `live_casehold_cuda.py` | H200-141GB      | 8 h        |

All cells run in 4-bit NF4 on a single GPU; inference uses vLLM and LoRA
training uses PEFT. Hyperparameters are pinned in `code/csa_core.py` and reused
across benchmarks; only the stream length scales with the per-benchmark
evaluation set size.

## Module documentation

`code/README.md` documents every script and module in detail (CSA controller,
surrogate, baselines, ablations, figure-generation pipeline).

## License

MIT (see `LICENSE`).
