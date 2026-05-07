# Conformal Selective Acting (CSA)

Anonymous code release accompanying a double-blind submission. This
repository contains the controller, surrogate, baselines, ablation scripts,
and the aggregate JSON summaries used to produce every table and figure in
the paper.

## Layout

```
.
├── code/              source for the controller, baselines, and experiments
│   ├── csa_core.py        CSA controller (Algorithm 1): per-threshold
│   │                      e-process on a Bonferroni grid
│   ├── domains/           per-benchmark stream and verifier
│   │                      (medical, pubmedqa, tatqa, mednli, gsm8k,
│   │                       headqa, arc, casehold)
│   ├── active-rcps/       reference port of A-RCPS used in App. F.8
│   ├── live_<bench>_cuda.py    live RLVR loop with online LoRA per cell
│   ├── ablate_*.py             hyperparameter and distribution-shift ablations
│   ├── run_*.py                experiment orchestrators
│   ├── build_paper_*.py        regenerates the LaTeX tables and PDF figures
│   ├── README.md               module-level documentation
│   └── requirements.txt        Python dependencies
└── data/results/      aggregate JSON summaries used by the build scripts
                       (17 files: paper-data, verified-numbers,
                        per-ablation summaries, cross-domain summary,
                        new-baselines summary, LTT grid summaries)
```

The aggregate JSONs in `data/results/` are produced by the experiment
scripts in `code/run_*.py` and `code/live_*_cuda.py`. Per-replication raw
outputs (one JSON per benchmark, alpha, and seed) are not included to keep
the release small; they are regenerated deterministically by re-running the
corresponding scripts.

## Reproducing the paper

```bash
pip install -r code/requirements.txt
python code/build_paper_figures.py   # regenerates the figures from
                                     # data/results/_verified_numbers.json
```

`code/build_paper_tables.py` regenerates the LaTeX tables; it expects the
per-benchmark per-alpha JSONs (re-derive by running the experiment scripts
below).

The live RLVR cells (Section 6.2) require a single GPU:

| Cell      | Script                  | GPU             | Wall-clock |
|-----------|-------------------------|-----------------|-----------:|
| MATH      | `live_medqa_cuda.py`    | A100-80GB       | 4 h        |
| MedQA     | `live_medqa_cuda.py`    | A100-80GB       | 30.5 h     |
| HEAD-QA   | (live HEAD-QA pipeline) | H200-141GB      | 30.5 h     |
| ARC-C     | `live_arc_cuda.py`      | H200-141GB      | 6 h        |
| CaseHOLD  | `live_casehold_cuda.py` | H200-141GB      | 8 h        |

All cells run in 4-bit NF4 on a single GPU; inference uses vLLM and LoRA
training uses PEFT. Hyperparameters are pinned in `code/csa_core.py` and
reused across benchmarks; only the stream length scales with the
per-benchmark evaluation set size.

## Module documentation

`code/README.md` documents every script and module in detail (CSA
controller, surrogate, baselines, ablations, figure-generation pipeline).

## License

MIT (see `LICENSE`).
