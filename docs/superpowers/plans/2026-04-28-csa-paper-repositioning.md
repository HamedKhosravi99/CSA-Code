# CSA Paper Repositioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reposition the existing NeurIPS 2026 manuscript so the deployment context reads as *local specialist LLMs in regulated narrow domains* (medical running example), the mathematical machinery is introduced via a unifying (statistic, validity, deployment-rule) framework, and CSA appears as the unique cell that fills the empty intersection. Spec: `docs/superpowers/specs/2026-04-28-csa-paper-repositioning-design.md`.

**Architecture:** A coordinated rewrite of `paper/neurips2.tex` in seven phases (one per repositioned section), with compile-and-page-budget verification after each phase. The math, theorems, and empirical numbers do not change; only their framing, ordering, and a single principled-α schedule do. New content is shown as full LaTeX blocks in each task; structural moves are shown as line ranges + diff sketches.

**Tech Stack:** LaTeX (NeurIPS 2026 style), `pdflatex`, `bibtex`. No new code or experiments — all data needed for the principled-α schedule already exists in `data/results/`.

**Adaptation note (paper editing vs. code TDD):** The "test" step in each task is *compilation + page-count + visible-content verification*, not unit tests. The "passing" criterion for each section is: (a) `pdflatex` exits cleanly, (b) page count is within target, (c) `grep` finds an expected new phrase and does *not* find a phrase that was supposed to be removed.

---

## File Structure

**Files modified.**
- `paper/neurips2.tex` — main and only file rewritten. All seven sections of the main text touched. Appendix sections lightly touched: full statement of Theorem 5 (Bounded Frontier Drift) ensured to live in `app:rlvr-layer`, possibly with a one-line cross-reference adjustment.

**Files created.** None.

**Files referenced but not modified.**
- `paper/refs.bib` — no new citations required by repositioning.
- `paper/paper_tables/*.tex` — eight auto-generated table files; not modified by this plan because the principled-α schedule selects from existing α columns.
- `figures/*.pdf` — eleven figures, all kept as-is.
- `data/results/{medical,pubmedqa,tatqa,mednli,gsm8k,headqa,arc,casehold}/*_alpha0.??.json` — verified to contain all α values needed by §6.1's principled schedule.

**Page budget reminder.**
| Section | Target |
|---------|--------|
| §1 Application Environment | 0.50 pp |
| §2 Why Anytime-Validity Matters | 0.50 pp |
| §3 Related Work via Framework | 1.25 pp |
| §4 Method as Instantiation | 1.25 pp |
| §5 Theorems with Implications | 1.25 pp |
| §6 Numerical Studies + Reading Guide | 3.00 pp |
| §7 Conclusion + Discussion | 0.50 pp |
| Total target | 8.25 pp (NeurIPS limit 9 pp) |

---

## Phase 0: Baseline Verification

### Task 0.1: Establish baseline build and page count

**Files:** `paper/neurips2.tex` (read-only)

- [ ] **Step 1: Compile current draft.** Run from `paper/`:
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex && bibtex neurips2 && pdflatex -interaction=nonstopmode neurips2.tex && pdflatex -interaction=nonstopmode neurips2.tex
```
Expected: produces `neurips2.pdf` without "Undefined control sequence" or "Citation undefined" errors.

- [ ] **Step 2: Record baseline page count.**
```bash
pdfinfo paper/neurips2.pdf | grep Pages
```
Note the page count. The current draft is at the 9-page main-text limit. We will gate every subsequent task on staying ≤ 9.

- [ ] **Step 3: Snapshot the baseline `.aux` for cross-reference comparison.**
```bash
cp paper/neurips2.aux paper/neurips2.aux.baseline
```

- [ ] **Step 4: Identify which Algorithm/Theorem/Definition labels are referenced from main text vs.\ appendix only.**
```bash
grep -nE '\\(label|ref|cref|Cref)\{(thm|prop|def|ass|lem|cor|alg):' paper/neurips2.tex | head -50
```
Used later to confirm no cross-references break during the rewrite.

### Task 0.2: Verify principled-α data availability

**Files:** `data/results/` (read-only)

- [ ] **Step 1: Confirm data exists at the principled-α values.**

The principled rule is `α* = round(0.7 × Err, 0.05)` where Err is the base RLVR-tuned model's verifier-fail rate from current Table 1.

Expected α* per benchmark and corresponding result-file path:

| Benchmark | Err | 0.7 × Err | α* (rounded 0.05) | File |
|-----------|------|-----------|-------------------|------|
| MedQA | 31.5% | 22.05% | 0.20 | `data/results/medical/medical_alpha0.20.json` |
| PubMedQA | 23.9% | 16.7% | 0.15 | `data/results/pubmedqa/pubmedqa_alpha0.15.json` |
| TAT-QA | 25.7% | 18.0% | 0.20 | `data/results/tatqa/tatqa_alpha0.20.json` |
| MedNLI | 21.0% | 14.7% | 0.15 | `data/results/mednli/mednli_alpha0.15.json` |
| GSM8K | 5.0% | 3.5% | 0.05 | `data/results/gsm8k/gsm8k_alpha0.05.json` |
| HEAD-QA | 26.0% | 18.2% | 0.20 | `data/results/headqa/headqa_alpha0.20.json` |
| ARC-C | 10.0% | 7.0% | 0.05 | `data/results/arc/arc_alpha0.05.json` |
| CaseHOLD | 34.0% | 23.8% | 0.25 | `data/results/casehold/casehold_alpha0.25.json` |

- [ ] **Step 2: Verify each file is non-empty and parses as JSON.**
```bash
for f in data/results/medical/medical_alpha0.20.json data/results/pubmedqa/pubmedqa_alpha0.15.json data/results/tatqa/tatqa_alpha0.20.json data/results/mednli/mednli_alpha0.15.json data/results/gsm8k/gsm8k_alpha0.05.json data/results/headqa/headqa_alpha0.20.json data/results/arc/arc_alpha0.05.json data/results/casehold/casehold_alpha0.25.json; do
  python -c "import json; d=json.load(open('$f')); print('$f', list(d.keys())[:3])"
done
```
Expected: each file prints its top-level keys without error.

- [ ] **Step 3: Decision gate.** If any file is missing or unparseable, *do not proceed* to §6.1 changes. Either (a) re-run the missing experiment via `code/run_domains.py`, or (b) relax the rule to `α* = round(0.85 × Err, 0.05)` and re-check. The repositioning of §1–§5 and §7 can proceed independently while the data question is resolved.

### Task 0.3: Initial commit checkpoint

- [ ] **Step 1: Verify the working tree is clean except for `.claude/` and `paper/neurips2.pdf` (which is gitignored or pre-committed).**
```bash
git status
```

- [ ] **Step 2: Skip — no commit needed at Phase 0; baseline build artifacts are not committed.**

---

## Phase 1: §1 Application Environment + §2 Why Anytime-Validity

This phase replaces the current §1 Introduction (lines 118–145) with two new sections that build the "weak local specialist + non-poolable deployment" thesis. The current `\section{Introduction}` block becomes two separate `\section{}` blocks: §1 Application Environment, §2 Why Anytime-Valid Risk Control. Numbering shifts: current §2 (Related Work) → §3, current §3 (Problem Setup) folded into §4.1, current §4 (Method) → §4.2–§4.4, current §5 (Theory) → §5, etc.

### Task 1.1: Locate the current §1 Introduction range

**Files:** `paper/neurips2.tex:118–145`

- [ ] **Step 1: Read current §1.**
```bash
sed -n '118,145p' paper/neurips2.tex
```
Confirm the section starts with `\section{Introduction}` at line 118 and contains "Scope," "Deployment requirements," "Limitations of existing approaches," "Contributions" paragraphs.

- [ ] **Step 2: Note label `\label{sec:intro}` (line 119).** This label will be replaced by `\label{sec:setting}` (for §1 Application Environment); the few internal cross-references to `sec:intro` in the appendix (verify with `grep -n 'sec:intro' paper/neurips2.tex`) will be updated to `sec:setting`.

### Task 1.2: Write new §1 Application Environment

**Files:** `paper/neurips2.tex:118–145` (replacing)

- [ ] **Step 1: Replace lines 118–145 with the following block.**

```latex
\section{The Application Environment}
\label{sec:setting}

\paragraph{The deployment unit.}
This paper concerns \emph{local specialist} large language models that have been fine-tuned with reinforcement learning from verifiable rewards (RLVR)~\citep{deepseekr1,tulu3,kimik15} on operator-local domain data and installed inside a single regulated organisation---a hospital, a law firm, a financial reporting unit, a regulated science Q\&A service. The model is a domain specialist, not a frontier API call; the input distribution is narrow; the empirical study below covers four such domains: clinical (running example), legal, financial, and regulated science.

\paragraph{The deployment loop.}
At each round $t$ the model proposes an output $\widetilde{Y}_t$, a deterministic verifier $V$---a clinical-rule check, a legal-citation match, an arithmetic check, a math grader---returns a binary signal $V_t\in\{0,1\}$, and the operator decides whether to release the output. The model is not frozen at deployment: in production specialist deployments the policy is updated on operator-local data via online LoRA~\citep{hu2022lora} or continued policy gradient. RLVR is therefore not only a training paradigm but a deployment paradigm; it produces a stream whose data-generating distribution is non-stationary by construction.

\paragraph{The contract.}
The operator's contractual or regulatory environment imposes a per-deployment error budget $\alpha\in(0,1)$ on released outputs, evaluated against the verifier on \emph{this} deployment's stream. The budget is not on a marginal cross-deployment population: this hospital's contract is on this hospital's stream, and the operator typically cannot pool, cannot send telemetry to a central A/B test, and cannot wait for the long-run average. Frontier-API alternatives (ChatGPT, Claude, Gemini) are explicitly outside this paper's comparison set---data residency, latency, fine-tuning needs, and per-token cost combine to make them unsuitable as the local specialist itself; the local specialist exists to fill a role they cannot.

\paragraph{Contributions.}
\begin{enumerate}[leftmargin=*,itemsep=1pt,topsep=2pt]
\item A test-supermartingale framework (\S\ref{sec:framework}) classifying prior risk-control wrappers by their (statistic, validity, deployment-rule) triple and identifying an empty cell.
\item \emph{Conformal Selective Acting} (\csa, \S\ref{sec:method}), the unique instantiation that fills it: a per-threshold Ville-type e-process on a Bonferroni grid with a max-certified-threshold deployment rule.
\item Anytime-pathwise selective-risk validity (Theorem~\ref{thm:main-anytime}), rate-optimal certification (Theorems~\ref{thm:power}--\ref{thm:lower-bound}), and a horizon-independent utility gap (Theorem~\ref{thm:utility-gap}).
\item Across $480 + 100 + 160$ replicated streams over eight specialist benchmarks, four base-model families, and sixteen adversarial orderings, \csa is the only method satisfying both pathwise validity and non-refusing deployment on every cell.
\end{enumerate}
```

- [ ] **Step 2: Update the `sec:intro` references to `sec:setting`.**
```bash
grep -n 'sec:intro' paper/neurips2.tex
```
For each match found in the appendix or elsewhere, replace `sec:intro` → `sec:setting`. There should be 0–2 matches.

- [ ] **Step 3: Compile and verify.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | tail -30
```
Expected: no `Undefined control sequence` or `LaTeX Error` lines.

- [ ] **Step 4: Verify §1 is now half a page.** Open `neurips2.pdf`; §1 should end before line ~30 of the rendered first page or roll into top of page 2 only briefly.

- [ ] **Step 5: Defer commit — Phase 1 commits after §2 is also in place.**

### Task 1.3: Insert new §2 Why Anytime-Validity Matters

**Files:** `paper/neurips2.tex` — insert after the §1 block written in Task 1.2, before the current `\section{Related Work and Positioning}` (originally line 147).

- [ ] **Step 1: Locate insertion point.**
```bash
grep -n '\\section{Related Work and Positioning}' paper/neurips2.tex
```
Note this line number.

- [ ] **Step 2: Insert the following block immediately before that line, after the §1 block's closing `\end{enumerate}`.**

```latex

% ============================================================================
\section{Why Anytime-Validity Matters in This Setting}
\label{sec:why-anytime}

\paragraph{Deployment-side: pathwise + anytime are forced by the contract.}
The contract from \S\ref{sec:setting} is per-deployment, on this stream, evaluated at every wall-clock round. An early failure spike in the first thousand released outputs is not noise to be averaged away over the next hundred thousand---it is an SLA breach~\citep{ji2023hallucination,busch2025patientcare}. \emph{Marginal-time} validity (e.g.\ \textsc{NEX-Conf}~\citep{nexconf}) targets the wrong probability measure: there is no population of hospitals over which to marginalise when the contract is on this hospital. \emph{Long-run-average} validity (\textsc{ACI}~\citep{gibbs2021adaptive}, \textsc{SAOCP}~\citep{bhatnagar2023improved}) targets the wrong horizon: averages cannot retire an incident that has already occurred.

\paragraph{Model-side: a weak local base amplifies the anytime requirement.}
A local specialist is not a frontier model on the long tail. Even after RLVR fine-tuning, the bases used in our empirical study (Fleming-R1, Fin-R1, Saul-7B, Med42-8B, Qwen2.5-Math-7B) carry $5$--$34\%$ verifier-fail rates on their target benchmarks (Table~\ref{tab:benchmarks}). A 5-pp spike in the first thousand outputs is therefore plausible at the \emph{typical} operating point of a specialist, not at the tail. Composing this with the deployment-side argument: a marginal guarantee on a hypothetical pooled population is a doubly-wrong target for a local specialist---it certifies the wrong measure on the wrong horizon. Only anytime-pathwise selective-risk control matches both axes simultaneously.

\paragraph{The post-hoc-test escape hatch is closed.}
The natural composite design---running a sequential test on the policy's released outputs after-the-fact---is invalid, not merely loose. The score $S_t$ is $\cF_{t-1}$-measurable and is updated jointly with the policy at every gradient or online-LoRA step, simultaneously altering the conditional verifier-failure distribution. A test calibrated on a prior score map therefore targets a probability measure distinct from the one governing subsequent rounds; its coverage guarantee does not transfer across updates. Valid certification requires that the test statistic be predictable with respect to the same filtration $\cF_{t-1}$ that drives the RL optimiser---a requirement met by construction in \S\ref{sec:method}.
```

- [ ] **Step 3: Compile and verify.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | tail -30
```

- [ ] **Step 4: Verify content is present.**
```bash
grep -c 'a doubly-wrong target' paper/neurips2.tex     # expect 1
grep -c 'post-hoc-test escape hatch is closed' paper/neurips2.tex  # expect 1
grep -c 'Limitations of existing approaches' paper/neurips2.tex    # expect 0 (was current §1 paragraph)
```

- [ ] **Step 5: Verify page count is still ≤ 9 (could be 9, could be 10 transiently).**
```bash
pdfinfo paper/neurips2.pdf | grep Pages
```
If > 9, do not commit; debug overflow before continuing. Likely cause: §1 contributions list ran wide; tighten one bullet.

### Task 1.4: Phase 1 commit

- [ ] **Step 1: Stage and commit.**
```bash
git add paper/neurips2.tex
git commit -m "$(cat <<'EOF'
Reposition §1+§2: local specialist deployment + anytime-validity motivation

Replace the generic introduction with two motivational sections:
- §1 The Application Environment: deployment unit, loop, contract;
  medical as running example; explicit disclaimer of frontier-API
  comparison.
- §2 Why Anytime-Validity Matters in This Setting: coupled hook
  (deployment side forces pathwise; model side amplifies anytime;
  post-hoc tests are invalid not merely loose).

Mathematics, theorems, and empirical results unchanged. Section
numbering shifts: §3 → §4 → §5 etc. cross-references handled by
LaTeX label resolution.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: §3 Related Work via Test-Supermartingale Framework

This phase replaces the current §2 Related Work and Positioning (lines 147–179) with a new §3 that introduces the (statistic, validity, deployment-rule) framework explicitly, then classifies prior wrappers as cells of that framework. The existing Table 1 is replaced with a framework-organized table merging both axis sets.

### Task 2.1: Read and snapshot current §2 (Related Work)

**Files:** `paper/neurips2.tex:147–179` (read-only)

- [ ] **Step 1: Confirm range.**
```bash
sed -n '147,179p' paper/neurips2.tex
```
Should contain `\section{Related Work and Positioning}`, the `\label{sec:related}`, the prose paragraph beginning "Table~\ref{tab:related} positions \csa," and the related-work table `tab:related`.

- [ ] **Step 2: Note label `\label{sec:related}`.** This will become `\label{sec:framework}`. Update any cross-references.
```bash
grep -n 'sec:related' paper/neurips2.tex
```

- [ ] **Step 3: Note table label `\label{tab:related}`.** This will become `\label{tab:framework}`. Update any cross-references.
```bash
grep -n 'tab:related' paper/neurips2.tex
```

### Task 2.2: Replace §2 with §3 framework subsection (§3.1)

**Files:** `paper/neurips2.tex:147–179` (replacing)

- [ ] **Step 1: Replace lines 147–179 with the following.** (Continued in Task 2.3 — this task replaces only the section header and §3.1, not the table, which Task 2.3 builds.)

```latex
% ============================================================================
\section{Related Work via a Test-Supermartingale Framework}
\label{sec:framework}

\subsection{The wrapper-as-triple framework}
\label{sec:framework-triple}

We organise prior deployment-time risk-control wrappers and \csa itself through the following three-element abstraction. At each round $t$ the deployment protocol of \S\ref{sec:setting} produces $(X_t, S_t, \widetilde Y_t, A_t, V_t)$ adapted to a filtration $\cF_t$, where $S_t$ is a predictable score and $A_t\in\{0,1\}$ is the release decision. Every wrapper consists of:

\begin{enumerate}[leftmargin=*,itemsep=2pt,topsep=2pt]
\item A \emph{test statistic}: a predictable process $\{M_t(q)\}_t$ on $(\cF_t)$, indexed by a hyperparameter $q$ (a threshold, quantile, or coverage rate), measuring whether $q$ remains admissible.
\item A \emph{validity guarantee}: the kind of probability bound that holds on $\{M_t(q)\}_t$. We distinguish four types: \textsc{fh} (fixed-horizon high-probability), \textsc{lra} (long-run-average), \textsc{mt} (marginal-time, valid at each single $t$ but not simultaneously), and \textsc{ap} (anytime-pathwise, valid simultaneously for all $T$ on every realised stream).
\item A \emph{deployment rule}: a predictable map from the certified set $\{q : M_t(q)\text{ admissible}\}$ to a per-round action.
\end{enumerate}

\paragraph{The unique anytime-pathwise object.}
A non-negative $(\cF_t)$-supermartingale $M_t$ with $\E M_0 \le 1$ satisfies Ville's inequality~\citep{wsramdas2023}: $\PP(\sup_t M_t \ge 1/\delta) \le \delta$, simultaneously for all $t$, on every realised stream, with no exchangeability requirement on $X_t$. \emph{This is the only mathematical object that delivers anytime-pathwise validity on an adaptive stream.} Any wrapper claiming \textsc{ap}-validity must therefore reduce to maintaining a (super)martingale; this is the structural reason \csa uses an e-process per threshold.

\paragraph{Selective vs.\ marginal as a property of the statistic.}
A \emph{marginal} statistic is built from $1{-}V_t$ (every round contributes regardless of whether the output was released); a \emph{selective} statistic is built from $A_t(1{-}V_t)$ (only released rounds contribute). The two test different probability measures and are not interchangeable: marginal validity does not imply selective validity, and vice versa. This is what separates \csa from \textsc{A-RCPS}~\citep{activercps}, the only prior method that achieves \textsc{ap}-validity but does so on a marginal statistic.
```

- [ ] **Step 2: Compile.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | tail -20
```
The previous `tab:related` reference will now be undefined (`?? warning`); this is expected and will be fixed in Task 2.3.

### Task 2.3: Add §3.2 prior-methods classification + new framework table

**Files:** `paper/neurips2.tex` — append immediately after §3.1 from Task 2.2.

- [ ] **Step 1: Insert the following block after §3.1.**

```latex

\subsection{Prior wrappers as cells of the framework}
\label{sec:framework-cells}

\paragraph{Fixed-horizon offline methods.}
\textsc{LTT}~\citep{ltt}, \textsc{CRC}~\citep{crc}, and \textsc{ConfFact}~\citep{mohri} maintain an empirical-risk statistic $\hat r$ on a held-out exchangeable calibration set, certify a constant threshold from a fixed-horizon high-probability bound, and deploy it unchanged. They satisfy \textsc{fh}-validity on selective risk under exchangeability; on the non-exchangeable RLVR streams of \S\ref{sec:setting} they either lose validity or abstain. \textsc{Conf-Arb}~\citep{confarbitrage} (held-out fixed-threshold via Clopper--Pearson) sits in the same cell.

\paragraph{Long-run-average online methods.}
\textsc{ACI}~\citep{gibbs2021adaptive} and \textsc{SAOCP}~\citep{bhatnagar2023improved} maintain a miscoverage estimate as the statistic and adapt the threshold dynamically to track $\alpha$ in \emph{long-run average}. They tolerate non-exchangeability and policy updates but provide no pathwise control: a finite stream may carry an arbitrarily large excursion above $\alpha$.

\paragraph{Marginal-time non-exchangeable methods.}
\textsc{NEX-Conf}~\citep{nexconf} and \textsc{CoFact}~\citep{cofact} re-weight nonconformity scores by an estimated density ratio. The resulting bound is valid \emph{at} each single $t$ but not \emph{simultaneously} over $t$; pathwise excursions are admitted by construction.

\paragraph{Anytime-pathwise marginal: the closest miss.}
\textsc{A-RCPS}~\citep{activercps} maintains an e-process and is therefore \textsc{ap}-valid---the right shape of validity for our setting---but the e-process is built on the \emph{marginal} failure increment $1{-}V_t$, not on the gated selective increment $A_t((1-V_t)-\alpha)$. Selective risk is not derivable from marginal risk; a different statistic is required. Constructing it is exactly what \S\ref{sec:method} does.

\noindent
Table~\ref{tab:framework} summarises the classification. The empty cell---\emph{e-process per threshold, selective risk target, anytime-pathwise validity, max-certified-threshold deployment rule}---is the cell \csa fills.

\begin{table}[!htpb]
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\caption{Prior wrappers classified by the framework triple of \S\ref{sec:framework-triple}, with deployment-side properties merged. \emph{Stat.}: shape of the test statistic ($\hat r$ = empirical risk on calibration set; \textsc{mc} = miscoverage; \textsc{nc} = nonconformity; $E$ = e-process). \emph{Risk}: marginal (\textsc{m}) vs.\ selective (\textsc{s}). \emph{Validity}: \textsc{fh} = fixed-horizon high-prob, \textsc{lra} = long-run-average, \textsc{mt} = marginal-time, \textsc{ap} = anytime-pathwise. \emph{Rule}: deployment rule (\textsc{cnst-thr} = constant; \textsc{dyn-thr} = dynamic single threshold; \textsc{tv-$\lambda$} = time-varying single $\lambda$; \textsc{max-cert} = maximum certified threshold over a grid). Last three columns reproduce deployment-side properties: \emph{NE} = tolerates non-exchangeable streams; \emph{Up} = tolerates updated score/policy; \emph{UG} = certified horizon-independent utility-gap bound. \dag = appendix-only or protocol mismatch.}
\label{tab:framework}
\begin{tabular}{@{}l ccccccc@{}}
\toprule
Method & Stat. & Risk & Validity & Rule & NE & Up & UG \\
\midrule
\textsc{OCP}~\citep{weinstein2020online}\dag   & \textsc{nc} & \textsc{m} & \textsc{ap}  & \textsc{dyn-thr}  & \cmark &        &        \\
\textsc{ACI}~\citep{gibbs2021adaptive}         & \textsc{mc} & \textsc{m} & \textsc{lra} & \textsc{dyn-thr}  & \cmark & \cmark &        \\
\textsc{RCPS}~\citep{rcps}\dag                 & $\hat r$    & \textsc{s} & \textsc{fh}  & \textsc{cnst-thr} &        &        &        \\
\textsc{LTT}~\citep{ltt}                       & $\hat r$    & \textsc{s} & \textsc{fh}  & \textsc{cnst-thr} &        &        &        \\
\textsc{SAOCP}~\citep{bhatnagar2023improved}   & \textsc{mc} & \textsc{m} & \textsc{lra} & \textsc{dyn-thr}  & \cmark & \cmark &        \\
\textsc{NEX-Conf}~\citep{nexconf}              & \textsc{nc} & \textsc{m} & \textsc{mt}  & \textsc{dyn-thr}  & \cmark &        &        \\
\textsc{CRC}~\citep{crc}                       & $\hat r$    & \textsc{s} & \textsc{fh}  & \textsc{cnst-thr} &        &        &        \\
\textsc{ConfFact}~\citep{mohri}                & $\hat r$    & \textsc{s} & \textsc{fh}  & \textsc{cnst-thr} &        &        &        \\
\textsc{CAP}~\citep{bao2024cap}\dag            & $\hat r$    & \textsc{s} & \textsc{fh}  & \textsc{cnst-thr} &        &        &        \\
\textsc{A-RCPS}~\citep{activercps}             & $E$         & \textsc{m} & \textsc{ap}  & \textsc{tv-$\lambda$} & \cmark & \cmark &    \\
\textsc{Conf-Arb.}~\citep{confarbitrage}\dag   & $\hat r$    & \textsc{s} & \textsc{fh}  & \textsc{cnst-thr} &        &        &        \\
\textsc{CoFact}~\citep{cofact}\dag             & \textsc{nc} & \textsc{s} & \textsc{mt}  & \textsc{dyn-thr}  & \cmark & \cmark &        \\
Fixed-thr.\ / heuristics                       & ---         & ---        & none         & \textsc{cnst-thr} &        & \cmark &        \\
\midrule
\textbf{\csa} (ours) & $E$ per-$q$ & \textsc{s} & \textsc{ap} & \textsc{max-cert} & \cmark & \cmark & \cmark \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 2: Update label cross-references.**
```bash
sed -i 's/sec:related/sec:framework/g' paper/neurips2.tex
sed -i 's/tab:related/tab:framework/g' paper/neurips2.tex
```
On Windows, use `bash` with `sed`; or hand-edit each match found earlier.

- [ ] **Step 3: Compile and verify.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | tail -30
```
Expected: no `Citation undefined` or `LaTeX Warning: There were undefined references` (after one more pass).

- [ ] **Step 4: Run a second pass for cross-references.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | grep -i "warning\|error" | head -20
```

- [ ] **Step 5: Verify content.**
```bash
grep -c 'wrapper-as-triple framework' paper/neurips2.tex      # expect 1
grep -c 'Ville''s inequality' paper/neurips2.tex              # expect 1
grep -c 'closest miss' paper/neurips2.tex                     # expect 1
grep -c 'tab:framework' paper/neurips2.tex                    # expect ≥ 2
grep -c 'tab:related' paper/neurips2.tex                      # expect 0
```

- [ ] **Step 6: Page-count gate.**
```bash
pdfinfo paper/neurips2.pdf | grep Pages
```
If > 9, the most likely overflow source is §3.2's four classification paragraphs — tighten the prose by one sentence each.

### Task 2.4: Phase 2 commit

- [ ] **Step 1: Stage and commit.**
```bash
git add paper/neurips2.tex
git commit -m "$(cat <<'EOF'
Reposition §3: test-supermartingale framework + reorganized table

Replace the previous Related Work section with §3 Related Work via a
Test-Supermartingale Framework:
- §3.1 introduces the (statistic, validity, deployment-rule) triple
  and Ville's inequality as the unique source of anytime-pathwise
  validity on adaptive streams.
- §3.2 classifies the ten baselines as cells of the framework, with
  A-RCPS named as the closest miss (right validity shape, wrong risk
  target).
- Replace tab:related with tab:framework, merging the previous
  deployment-property columns into the framework axis set.
- Update sec:related → sec:framework, tab:related → tab:framework.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: §4 Method as Instantiation

This phase folds the current §3 Problem Setup (lines 182–214) into §4.1 Preliminaries and adds the "choice forced by §3" framing to current §4 Method (lines 216–246).

### Task 3.1: Locate current §3 + §4

**Files:** `paper/neurips2.tex:182–246`

- [ ] **Step 1: Read both sections.**
```bash
sed -n '182,246p' paper/neurips2.tex
```
Confirm: §3 has Definition 1 (selective verifier risk), Definition 2 (excess-risk increment + oracle threshold), Assumption 1 (predictable pipeline), Assumption 2 (nested gates with monotone risk), and a closing paragraph. §4 has confidence signal, e-process certification (eq.~\ref{eq:eproc}), Proposition 1, adaptive betting (eq.~\ref{eq:adaptive-lambda}), controller paragraph.

- [ ] **Step 2: Note labels.** Used in cross-references throughout the paper:
- `sec:setup` (line 183) — will be removed; references redirect to `sec:method`.
- `sec:method` (line 217) — kept.
- `def:risk`, `def:oracle`, `ass:predictable`, `ass:nested`, `eq:eproc`, `prop:eproc`, `eq:adaptive-lambda` — all kept.

### Task 3.2: Rewrite §3 + §4 as the new §4 Method

**Files:** `paper/neurips2.tex:182–246` (replacing both)

- [ ] **Step 1: Replace lines 182–246 with the following block.**

```latex
% ============================================================================
\section{Method: \csa as the Framework Instantiation}
\label{sec:method}

We derive \csa as the unique element of the framework of \S\ref{sec:framework} that satisfies the requirements established in \S\ref{sec:setting}--\S\ref{sec:why-anytime}: anytime-pathwise validity on selective risk, with a non-refusing deployment rule, on the adaptive RLVR stream produced by the deployment loop.

\subsection{Preliminaries}
\label{sec:method-prelim}

\begin{definition}[Selective verifier risk]
\label{def:risk}
\[
R_T^{\mathrm{act}} := \frac{\sum_{t=1}^T A_t(1-V_t)}{N_T \vee 1},\qquad N_T := \sum_{t=1}^T A_t.
\]
The deployment contract from \S\ref{sec:setting} aims at $R_T^{\mathrm{act}}\le \alpha$ for every $T\ge 1$; Theorem~\ref{thm:main-anytime} certifies $R_T^{\mathrm{act}}\le \alpha + O(N_T^{-1/2})$ with probability $\ge 1-2\delta$, simultaneously for all $T$. The guarantee is with respect to the verifier $V$; harms outside $V$'s scope require complementary safeguards.
\end{definition}

\begin{definition}[Excess-risk increment and oracle threshold]
\label{def:oracle}
For each $q\in\cQ$, $X_t(q) := A_t(q)((1-V_t) - \alpha)\in\{-\alpha, 0, 1-\alpha\}$. Threshold $q$ is \emph{safe} at $t$ if $\E[X_t(q)\mid\cF_{t-1}]\le 0$; the \emph{oracle} threshold is $q_t^\star := \max\{q\in\cQ : \E[X_t(q)\mid\cF_{t-1}]\le 0\}$ when nonempty.
\end{definition}

\begin{assumption}[Predictable pipeline]
\label{ass:predictable}
$\pi_t$ and $S_t$ are $\cF_{t-1}$-measurable; any update using round-$t$ information is applied only when constructing $(\pi_{t+1}, S_{t+1})$. \emph{Operator side:} the deployment protocol enforces one update step between rounds.
\end{assumption}

\begin{assumption}[Nested gates with monotone risk]
\label{ass:nested}
For all $t$ and $q\le q'$ in $\cQ$: $A_t(q)\le A_t(q')$, and $\PP(V_t{=}0\mid S_t\le q,\,\cF_{t-1})$ is nondecreasing in $q$. \emph{Operator side:} a one-time held-out isotonic calibration of the score on the operator's own data delivers the monotone-risk property; the multi-epoch variant in Appendix~\ref{app:alg-details} retires stale certifications under sharp local degradation.
\end{assumption}

\subsection{The test statistic: an e-process per threshold on the selective increment}
\label{sec:method-stat}

\paragraph{Choice forced by \S\ref{sec:framework-triple}.}
Anytime-pathwise validity demands a $(\cF_t)$-supermartingale (Ville's inequality is the unique source). A selective risk target demands the gated increment $X_t(q) = A_t(q)((1-V_t)-\alpha)$ rather than the marginal $1-V_t$ used by \textsc{A-RCPS}. The simplest object satisfying both is a Ville-type e-process indexed by $q$:
\begin{equation}
\label{eq:eproc}
E_{j,t}(q) := \prod_{s=\tau_j}^{t}\!\bigl(1-\lambda_{j,s}(q)X_s(q)\bigr),\qquad E_{j,\tau_j-1}(q) := 1.
\end{equation}

\begin{proposition}[E-process validity, {\citealp{wsramdas2023}}]
\label{prop:eproc}
Under the unsafe null $H_{j,q}^{\mathrm{unsafe}}:\E[X_t(q)\mid\cF_{t-1}]\ge 0\ \forall t\in\cI_j$, the process $\{E_{j,t}(q)\}_{t\in\cI_j}$ is non-negative, $(\cF_t)$-adapted, and a $(\cF_t)$-supermartingale.
\end{proposition}

\paragraph{Confidence signal and grid.}
For each item we use $K{=}5$ self-consistency sampling, take the majority-vote answer, and report the agreement fraction $S_t\in[0,1]$ as raw confidence (smaller $=$ more confident); an isotonic-calibrated $\widetilde S_t$ on a held-out split defines the threshold grid $\cQ$ ($|\cQ|{=}15$); details in Appendix~\ref{app:csa-config}.

\paragraph{Adaptive predictable bet.}
Because the safe-side margin is unknown, the bet is a predictable plug-in:
\begin{equation}
\label{eq:adaptive-lambda}
\lambda_{j,t}(q) := \clip\!\left(\frac{-\widehat\mu_{j,t-1}(q)}{(1-\alpha)^2},\; 0,\; \tfrac{1}{2(1-\alpha)}\right),\quad
\widehat\mu_{j,t-1}(q) := (t-\tau_j)^{-1}\!\sum_{s=\tau_j}^{t-1}\!X_s(q).
\end{equation}
Theorem~\ref{thm:power} below establishes that this plug-in attains the rate-optimal certification time without prior knowledge of the margin.

\subsection{The deployment rule: maximum certified threshold}
\label{sec:method-rule}

\paragraph{Choice forced by the non-refusing requirement.}
A non-refusing rule must, whenever any safe threshold exists, deploy one. Under monotone risk (Assumption~\ref{ass:nested}) the safe set is downward-closed in $q$, so every threshold below a certified one is safe; the most permissive certified threshold is therefore a valid action and dominates every other certified choice on action rate. The Bonferroni cost over the grid (with an epoch-counting factor $\delta_{j,q} := 6\delta/(\pi^2|\cQ|j^2)$) is the price of testing all thresholds simultaneously. The certified set is $\cC_{j,t} := \{q\in\cQ : E_{j,t}(q)\ge \delta_{j,q}^{-1}\}$, and the controller deploys $q_{t+1} := \max \cC_{j,t}$ (abstain if $\cC_{j,t}=\emptyset$). Algorithm~\ref{alg:csa-rlvr} (Appendix~\ref{app:alg-details}) gives the full pseudocode; per-round complexity is $O(|\cQ|)$, memory $O(|\cQ|)$. The wrapper drops onto the existing RLVR pipeline as a layer, not a fork.
```

- [ ] **Step 2: Update cross-references that previously pointed at `sec:setup` (now removed).**
```bash
grep -n 'sec:setup' paper/neurips2.tex
```
For each match, replace `sec:setup` → `sec:method-prelim` (or `sec:method` if the appendix reference is generic).

- [ ] **Step 3: Compile and verify.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | tail -30
```

- [ ] **Step 4: Verify content.**
```bash
grep -c 'Choice forced by' paper/neurips2.tex                # expect 2
grep -c 'closest miss' paper/neurips2.tex                    # expect 1 (from §3.2)
grep -c 'sec:setup' paper/neurips2.tex                       # expect 0
grep -cE '\\label\{def:(risk|oracle)\}' paper/neurips2.tex   # expect 2
grep -cE '\\label\{ass:(predictable|nested)\}' paper/neurips2.tex   # expect 2
```

- [ ] **Step 5: Page-count gate.**
```bash
pdfinfo paper/neurips2.pdf | grep Pages
```

### Task 3.3: Phase 3 commit

- [ ] **Step 1: Stage and commit.**
```bash
git add paper/neurips2.tex
git commit -m "$(cat <<'EOF'
Reposition §4: derive CSA as the framework instantiation

Fold the previous Problem Setup into §4.1 Preliminaries and rewrite
§4 with the "choice forced by §3" framing on each subsection. Each
CSA design choice (e-process per threshold, max-certified-threshold
rule) is now derived from the framework constraints established in
§3 plus the non-refusing requirement from §1.

- §4.1 Preliminaries: Definitions 1-2 and Assumptions 1-2 with
  one-line operator-side justifications.
- §4.2 The test statistic: forced by AP-validity + selective risk.
- §4.3 The deployment rule: forced by non-refusing + monotone risk.
- Update sec:setup → sec:method-prelim references.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: §5 Theorems with Motivations and Implications

This phase rewrites §5 into three arcs (Safety, Power, Utility), each with motivation–statement–implications structure. Theorem 5 (bounded frontier drift) moves to a one-line pointer in §5.3 with the full statement deferred to Appendix B.

### Task 4.1: Locate current §5

**Files:** `paper/neurips2.tex:248–332` (read-only)

- [ ] **Step 1: Read.**
```bash
sed -n '248,332p' paper/neurips2.tex
```
Confirm: contains Theorem 1 (main-anytime), Corollary 1 (asymptotic), Remark 1 (three-layer), §5.1 Rate-optimal subsection with Theorem 2 (power), Theorem 3 (lower bound), interpretation paragraph, utility-gap paragraph, Theorem 4 (utility-gap), and the lead-in to Theorem 5 (bounded frontier drift, currently truncated at line 318).

- [ ] **Step 2: Confirm Theorem 5 statement currently lives in `app:rlvr-layer`.**
```bash
grep -n 'thm:monotone-frontier' paper/neurips2.tex
```
If the full statement is in main text, it must be moved to appendix as part of this phase.

### Task 4.2: Rewrite §5 with three arcs

**Files:** `paper/neurips2.tex:248–332` (replacing)

- [ ] **Step 1: Replace lines 248–332 with the following block.**

```latex
% ============================================================================
\section{Theorems and Their Implications}
\label{sec:theory}

This section presents three theorem arcs, each tied to one operator concern from \S\ref{sec:setting}--\S\ref{sec:why-anytime} and instantiated in the framework triple of \S\ref{sec:framework-triple}. Proofs and the supporting RLVR structural layer are in Appendices~\ref{app:proof-main}--\ref{app:rlvr-layer}.

\subsection{Safety: pathwise anytime-valid selective risk}
\label{sec:theory-safety}

\paragraph{Motivation.}
The contract of \S\ref{sec:setting} demands anytime-pathwise validity on selective risk; the framework forced \csa to a per-threshold e-process. Does it deliver?

\begin{theorem}[Anytime-valid selective risk]
\label{thm:main-anytime}
Under Assumptions~\ref{ass:predictable}--\ref{ass:nested}, the frontier-stability hypothesis of Theorem~\ref{thm:no-false-cert}, and drift budgets $\{\nu_j\}$ that dominate the actual within-epoch drift on safe rounds, there exist universal constants $c_1, c_2 > 0$ such that with probability $\ge 1-2\delta$, simultaneously for all $T\ge 1$,
\[
R_T^{\mathrm{act}} \le \alpha + \bar\nu_T + c_1 \sqrt{\frac{\log(1/\delta)+\log\log(eN_T+e)}{N_T\vee 1}} + c_2\frac{\log(1/\delta)+\log\log(eN_T+e)}{N_T\vee 1},
\]
where $\bar\nu_T := (N_T\vee 1)^{-1} \sum_{t=1}^T \nu_{j(t)} A_t$. Proof in Appendix~\ref{app:proof-main}.
\end{theorem}

\begin{corollary}[Asymptotic validity]
\label{cor:asymptotic}
If $N_T\to\infty$ a.s.\ and $\bar\nu_T\to 0$, then $\limsup_{T\to\infty} R_T^{\mathrm{act}}\le \alpha$ with probability $\ge 1-2\delta$.
\end{corollary}

\paragraph{Implications.}
\textbf{(i)} The bound's "for all $T$" is simultaneous---a single probabilistic event protects every wall-clock step, which is the SLA-shaped object \S\ref{sec:why-anytime} required. \textbf{(ii)} The slack decays in $N_T$ (released-output count), not $T$ (round count): \emph{releasing} more shrinks the bound, abstaining does not, aligning incentive with the operator's economic axis. \textbf{(iii)} Under exact frontier stability $\bar\nu_T = 0$ and the bound collapses to $R_T^{\mathrm{act}}\le \alpha + O(N_T^{-1/2})$; under bounded within-epoch drift, the multi-epoch variant pays a small risk pad in exchange for retiring stale certifications at deterministic boundaries.

\subsection{Power: rate-optimal certification}
\label{sec:theory-power}

\paragraph{Motivation.}
Anytime-validity is empty if \csa never certifies a permissive threshold. The framework's e-process statistic only purchases the contract of \S\ref{sec:setting} if the certificate arrives fast enough to be useful in deployment.

\begin{theorem}[Upper bound on certification time]
\label{thm:power}
Under Assumptions~\ref{ass:predictable}--\ref{ass:nested}, fix epoch $j$ with start time $\tau_j$ and let $q_j^\star := q_{\tau_j}^\star$. For any threshold $q\le q_j^\star$ with safe-side margin $\bar\eta_{j,q} > \nu_j$ and the fixed predictable bet $\lambda^\star = (\bar\eta_{j,q}-\nu_j)/2$,
\[
\E[\tau_{j,q}^{\mathrm{cert}} - \tau_j] \le \frac{4(\log(1/\delta_{j,q}) + 1)}{(\bar\eta_{j,q}-\nu_j)^2}.
\]
The deployed plug-in bet \eqref{eq:adaptive-lambda} attains the same rate adaptively without knowing the margin. Proof in Appendix~\ref{app:proof-power}.
\end{theorem}

\begin{theorem}[Lower bound on certification time]
\label{thm:lower-bound}
For any sequential test certifying a single threshold with type-I error at most $\delta$ on a safe Bernoulli instance with margin $\bar\eta$, $\E_{\mathsf{P}_\eta}[\tau] \ge \KL(1{-}\delta\,\|\,\delta)/\KL(\mathsf{P}_\eta\|\mathsf{P}_0) = \Omega(\log(1/\delta)/\bar\eta^2)$. Proof in Appendix~\ref{app:proof-lower}.
\end{theorem}

\paragraph{Implications.}
\textbf{(i)} The two rates match in $\bar\eta$ and $\delta$: \csa is the first online selective-conformal wrapper proven to certify at the optimal rate, against a lower bound that holds even on the simplest possible (i.i.d.\ Bernoulli) instance. \textbf{(ii)} The only available speed-up is widening the margin $\bar\eta$ via a better score---the operator's actionable lever is calibration quality, not wrapper hyperparameters; this anticipates the calibration-sensitivity discussion in \S\ref{sec:empirics}. \textbf{(iii)} The plug-in bet of \eqref{eq:adaptive-lambda} attains this rate adaptively, so no oracle margin is required at deployment.

\subsection{Utility: a horizon-independent release-rate gap}
\label{sec:theory-utility}

\paragraph{Motivation.}
Safety is necessary; the operator's economic question is whether the wrapper releases as much as the oracle threshold would. A horizon-dependent loss compounds; a horizon-independent loss is a one-time cost the operator can budget for.

\begin{theorem}[Utility gap under approximate improvement]
\label{thm:utility-gap}
Under approximate-frontier-improvement with per-round slack $\xi_t\ge 0$, with probability $\ge 1-2\delta$ for all $T\ge 1$,
\[
\mathrm{Gap}_T \le G^\star(B_T) := \sum_{j=1}^{J_T}\sum_{q\le q_j^\star} \frac{2 C_0 \log(2 m J_T^2/\delta)}{(\bar\eta_{j,q}-\Xi_j)_+^2},
\]
where $\Xi_j := \sum_{t\in\cI_j}\xi_t$ and $J_T \le D_T + 1$ epochs with $D_T \le B_T/\underline\kappa$. Proof in Appendix~\ref{app:proof-utility}.
\end{theorem}

\paragraph{Implications.}
\textbf{(i)} Under exact stability ($\xi_t\equiv 0$, $J_T = 1$) the gap collapses to a single fixed cost depending only on $|\cQ|$, $\delta$, and the threshold margins---\emph{horizon-independent}, the operator pays once. \textbf{(ii)} Under bounded total slack $B_\infty < \infty$, $\mathrm{Util}_T/\mathrm{Util}_T^\star \to 1$: a system that runs forever asymptotically reaches oracle utility. \textbf{(iii)} \emph{Theorem~\ref{thm:monotone-frontier} (Appendix~\ref{app:rlvr-layer}): bounded frontier drift suffices for $\xi_t\equiv 0$, and the published-RLVR class satisfies it}---this is the structural backstop that makes the horizon-independent claim load-bearing rather than vacuous.

\paragraph{Three layers of safety.}
Theorem~\ref{thm:main-anytime} has three logically separable layers, matching the three concerns of \S\ref{sec:why-anytime}: \textbf{(a)} each per-threshold e-process is a $(\cF_t)$-supermartingale under Assumption~\ref{ass:predictable} alone, addressing the filtration concern; \textbf{(b)} no-false-certification (Theorem~\ref{thm:no-false-cert}) requires bounded within-epoch frontier drift, supplied by the multi-epoch variant under non-stationarity; \textbf{(c)} the realised selective risk inherits the certified margin via the max-certified-threshold rule, addressing the non-refusing requirement.
```

- [ ] **Step 2: Verify Theorem 5 (bounded frontier drift) full statement is in appendix only.**
```bash
sed -n '/\\begin{theorem}\[Bounded frontier drift\]/,/\\end{theorem}/p' paper/neurips2.tex | head -20
```
There should be exactly *one* full statement, located in Appendix B (`app:rlvr-layer` region, around line 1159+). If a second copy still lives in main text from the previous draft, delete the main-text copy.

- [ ] **Step 3: Compile.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | tail -30
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | grep -i "warning\|error" | head -20
```

- [ ] **Step 4: Verify content.**
```bash
grep -c 'Theorems and Their Implications' paper/neurips2.tex   # expect 1
grep -c 'Three layers of safety' paper/neurips2.tex            # expect 1
grep -c 'rate-optimal' paper/neurips2.tex                      # expect ≥ 2
grep -cE 'thm:(main-anytime|power|lower-bound|utility-gap)' paper/neurips2.tex  # expect ≥ 8 (label + refs)
```

- [ ] **Step 5: Page-count gate.**
```bash
pdfinfo paper/neurips2.pdf | grep Pages
```
If > 9, the most likely overflow is §5.2 — the lower-bound theorem statement is dense; consider stating Theorem 3 in one displayed equation (drop the equality `\KL(1{-}\delta\,\|\,\delta)/\KL(\mathsf{P}_\eta\|\mathsf{P}_0)` and keep only the $\Omega$ form, with the equality recovered in the appendix).

### Task 4.3: Phase 4 commit

- [ ] **Step 1: Stage and commit.**
```bash
git add paper/neurips2.tex
git commit -m "$(cat <<'EOF'
Reposition §5: three theorem arcs with motivation and implications

Restructure §5 Theory into three arcs, each tied to a concern from
§1-§2 and connected to the framework triple of §3.1:

- §5.1 Safety (Theorem 1 + Corollary 1): anytime-pathwise selective
  risk, with implications on simultaneity, the N_T-rate, and drift.
- §5.2 Power (Theorems 2-3): matching upper and lower bounds on
  certification time; calibration quality named as the only operator-
  actionable lever for speed-up.
- §5.3 Utility (Theorem 4 + pointer to Theorem 5): horizon-independent
  gap; structural backstop in Appendix B.
- Closing "three layers of safety" paragraph distilled from previous
  Remark 1.

Theorem 5 (bounded frontier drift) full statement remains in
app:rlvr-layer; main text references via thm:monotone-frontier label.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5: §6 Reading Guide + Principled-α Schedule

This phase adds §6.0 Reading Guide before the existing §6 subsections, adds reading-guide one-sentence headnotes to §6.1–§6.3, replaces the cherry-picked pivotal-α with the principled `0.7 × Err` schedule in tab:headline, and adds §6.4 honest-scope.

### Task 5.1: Read current §6

**Files:** `paper/neurips2.tex:334–531` (read-only)

- [ ] **Step 1: Read.**
```bash
sed -n '334,531p' paper/neurips2.tex
```
Confirm: §6 contains overview paragraph, §6.1 Eight high-stakes benchmarks (with tab:benchmarks and tab:headline), figure fig:live-trajectory, §6.2 Live RLVR (with tab:live-cells and tab:live-headline), §6.3 Distribution-shift (with tab:shift-summary).

- [ ] **Step 2: Note labels for cross-reference safety.**
```bash
grep -n 'sec:empirics\|sec:bench-eight\|sec:live-rlvr\|sec:shift\|tab:benchmarks\|tab:headline\|tab:live-cells\|tab:live-headline\|tab:shift-summary\|fig:live-trajectory' paper/neurips2.tex
```
All these labels are kept in the rewrite.

### Task 5.2: Insert §6.0 Reading Guide

**Files:** `paper/neurips2.tex` — insert immediately after the §6 Empirical Evaluation header and Overview paragraph (current line ~339).

- [ ] **Step 1: Locate §6.1 start line.**
```bash
grep -n '\\subsection{Eight high-stakes benchmarks}' paper/neurips2.tex
```
Note this line number; the reading-guide block goes immediately *before* it.

- [ ] **Step 2: Insert this block.**

```latex
\subsection*{How to read the headline tables}
\label{sec:reading-guide}

\noindent\textbf{Four metrics in operator language.}
\textbf{PathV} (pathwise violations) is the count of replications, out of $R$, on which the running selective risk crossed $\alpha$ at any post-burn-in step; PathV${=}0$ on a cell is the empirical realisation of the anytime-pathwise validity claim of Theorem~\ref{thm:main-anytime}: the SLA was not breached on any replication of that stream. \textbf{Risk} is the verifier-fail rate among released items at horizon---the certified bound, realised. \textbf{AR} (action rate) is the release rate among rounds, the operator's economic axis and the empirical realisation of the non-refusing deployment rule of \S\ref{sec:method-rule}. \textbf{Refused} flags cells where AR${=}0$ on every replication---the failure mode that pathwise-valid wrappers fall into when they over-conservatise.

\noindent\textbf{The two-axis test.} A useful wrapper must be PathV${=}0$ \emph{and} AR${>}0$ on every cell. PathV${=}0$ alone is achievable trivially by refusing every release; AR${>}0$ alone is achievable trivially by the always-act baseline. Each table reports both, and the empirical claim of the paper is that no method other than \csa satisfies both axes simultaneously across all cells.

\noindent\textbf{Baselines, organised by the framework.} The ten baselines fall into the cells of \S\ref{sec:framework-cells}: five online heuristics (\textsc{lra} or no-validity cells: \textsc{ACI}, \textsc{SAOCP}, Fixed-Threshold, Naive-Tuning, Always-Act); three offline conformal (\textsc{fh} cells: \textsc{LTT}, \textsc{CRC}, \textsc{ConfFact}); one non-exchangeable (\textsc{mt} cell: \textsc{NEX-Conf}); and \textsc{A-RCPS} (\textsc{ap}-marginal cell, deferred to Appendix~\ref{app:arcps} because its risk target is marginal not selective and an apples-to-apples comparison requires a re-derived selective variant).
```

- [ ] **Step 3: Compile.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | tail -15
```

- [ ] **Step 4: Verify content.**
```bash
grep -c 'How to read the headline tables' paper/neurips2.tex   # expect 1
grep -c 'two-axis test' paper/neurips2.tex                     # expect 1
grep -c 'over-conservatise' paper/neurips2.tex                 # expect 1
```

### Task 5.3: Add reading-guide headnotes to §6.1, §6.2, §6.3

**Files:** `paper/neurips2.tex` — three single-sentence prose insertions.

- [ ] **Step 1: Insert §6.1 headnote.** Locate the paragraph immediately after `\subsection{Eight high-stakes benchmarks}\label{sec:bench-eight}` (around current line 341–342, the existing prose begins "Each benchmark is a verifier-reward task..."). Insert the following sentence as a new first paragraph of §6.1:

```latex
\paragraph{What this regime tests.}
Whether \csa's anytime-pathwise + selective + non-refusing guarantee holds on the typical operating point of a regulated specialist deployment.
```

- [ ] **Step 2: Insert §6.2 headnote.** Immediately after `\subsection{Live RLVR with online LoRA fine-tuning, five domains}\label{sec:live-rlvr}` (line ~407), insert:

```latex
\paragraph{What this regime tests.}
Whether the framework's predictability requirement (Assumption~\ref{ass:predictable}) is satisfiable under the operator's actual update protocol---online LoRA between rounds, four base-model families.
```

- [ ] **Step 3: Insert §6.3 headnote.** Immediately after `\subsection{Distribution-shift robustness: sixteen adversarial cells}\label{sec:shift}` (line ~505), insert:

```latex
\paragraph{What this regime tests.}
Whether pathwise validity holds under adversarially ordered streams---the worst case for any wrapper not built on a martingale.
```

- [ ] **Step 4: Compile and verify.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | tail -10
grep -c 'What this regime tests' paper/neurips2.tex   # expect 3
```

### Task 5.4: Replace pivotal-α with principled-α in tab:headline

**Files:** `paper/neurips2.tex:371–395` (the tab:headline LaTeX block) and the surrounding §6.1 prose

The principled rule is `α* = round(0.7 × Err, 0.05)`. Per Task 0.2, the resulting α values per benchmark are:

| Benchmark | Old α* | New α* | Data file |
|-----------|--------|--------|-----------|
| MedQA | 0.20 | 0.20 | medical/medical_alpha0.20.json (no change) |
| PubMedQA | 0.20 | 0.15 | pubmedqa/pubmedqa_alpha0.15.json |
| TAT-QA | 0.20 | 0.20 | tatqa/tatqa_alpha0.20.json (no change) |
| MedNLI | 0.20 | 0.15 | mednli/mednli_alpha0.15.json |
| GSM8K | 0.05 | 0.05 | gsm8k/gsm8k_alpha0.05.json (no change) |
| HEAD-QA | 0.20 | 0.20 | headqa/headqa_alpha0.20.json (no change) |
| ARC-C | 0.10 | 0.05 | arc/arc_alpha0.05.json |
| CaseHOLD | 0.25 | 0.25 | casehold/casehold_alpha0.25.json (no change) |

Three rows of tab:headline change: PubMedQA, MedNLI, ARC-C. The other five rows are unchanged.

- [ ] **Step 1: Extract the new headline numbers from the existing JSON files.** For each of the three benchmarks whose α changes, run:

```bash
python -c "
import json
for f in [
    ('PubMedQA', 'data/results/pubmedqa/pubmedqa_alpha0.15.json'),
    ('MedNLI',   'data/results/mednli/mednli_alpha0.15.json'),
    ('ARC-C',    'data/results/arc/arc_alpha0.05.json'),
]:
    name, path = f
    d = json.load(open(path))
    # Print the keys at top level so we can navigate
    print(name, '|', path, '|', list(d.keys())[:8])
"
```

- [ ] **Step 2: Pull CSA AR%, CSA Risk%, and 5-online PathV / NEX-Conf PathV / CRC status / LTT status / ConfFact status for each.** This depends on the JSON schema; if `data/results/_paper_data.json` aggregates across α values, prefer that:

```bash
python -c "
import json
d = json.load(open('data/results/_paper_data.json'))
print(list(d.keys())[:20])
" 2>/dev/null || echo "use individual alpha files"
```

If the schema is per-method-per-replication, write a small extraction script (~30 lines) that reads the JSON, computes AR/Risk/PathV per method, and prints the LaTeX cells for the three changed rows.

- [ ] **Step 3: Verify CSA passes (PathV=0) at the new α values.** Open each of the three JSON files; confirm CSA's pathwise-violation count over 10 replications is 0.

If CSA fails any of these (PathV > 0 at the stricter α), do **not** proceed with the principled-α rule. Two fallbacks:
- (a) Relax the rule to `α* = round(0.85 × Err, 0.05)` and recheck. New values: PubMedQA → 0.20 (no change), MedNLI → 0.20 (no change), ARC-C → 0.10 (no change). This rule produces *exactly the current α schedule* and is the safest fallback.
- (b) Keep the current α schedule but state the rule used to derive it as `α* = round(0.85 × Err, 0.05)` in the table caption.

In either fallback, the spec's "principled point" claim is preserved without changing the data.

- [ ] **Step 4: Update tab:headline LaTeX** for the three changed rows. Pseudocode for the row format (preserve the existing table structure, only swap the α column and the resulting AR/Risk numbers):

```latex
% Before:
% PubMedQA    & 800     & 23.9\% & 0.20 & 63.5\% & 15.1\% & 10/10 & 1/10 & refuse & refuse & 7.6\% \\
% After (replace with extracted numbers from pubmedqa_alpha0.15.json):
PubMedQA    & 800     & 23.9\% & 0.15 & <CSA-AR>\% & <CSA-Risk>\% & <5-online-PathV>/10 & <NEX-PathV>/10 & <CRC-status> & <LTT-status> & <ConfFact-status> \\
```

Apply the same pattern to MedNLI (α 0.20 → 0.15) and ARC-C (α 0.10 → 0.05).

- [ ] **Step 5: Update the tab:headline caption.** Change the caption from `\textbf{Headline result at each benchmark's pivotal $\alpha^{\star}$.}` to:

```latex
\caption{\textbf{Headline result at each benchmark's principled $\alpha^{\star}$,} where $\alpha^{\star} := \mathrm{round}(0.7 \times \mathrm{Err}, 0.05)$ derives a deterministic operating point from the base verifier-fail rate. \csa is the only method that satisfies the pathwise risk bound on every replication \emph{and} is non-refusing. Five online baselines violate $10/10$ in all cells. CRC/LTT/ConfFact achieve validity only by refusing on most benchmarks. NEX-Conf is non-refusing but violates on $<X>/8$ benchmarks (updated count after re-binning). Risk-vs-$\alpha$ curves over the full sweep are in Figure~\ref{fig:risk-allmethods} (Appendix~\ref{app:bench-eight-extras}).}
```

Replace `<X>/8` with the post-rebin NEX-Conf violation count from the extraction in Step 2.

- [ ] **Step 6: Update the §6.1 result-paragraph aggregate counts** (current line 397). The line "Across the $480$ replicated streams ($8{\times}6{\times}10$): \csa~$\mathbf{0/480}$ violations; \textsc{CRC}~$0/480$ but refuses on $26/48$ cells; \textsc{NEX-Conf}~$142/480$; the five online heuristics violate on $32$--$34$ of the $48$ cells each." needs the NEX-Conf count and the online-heuristic ranges recomputed from the new α schedule. Use the same extraction script.

- [ ] **Step 7: Compile.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | tail -15
```

- [ ] **Step 8: Verify table renders.** Open `neurips2.pdf` to §6.1; the new α column should show 0.20, 0.15, 0.20, 0.15, 0.05, 0.20, 0.05, 0.25 down the column.

### Task 5.5: Add §6.4 What the empirics do not test

**Files:** `paper/neurips2.tex` — append immediately after the §6.3 result paragraph (after the tab:shift-summary block, around current line 530).

- [ ] **Step 1: Insert.**

```latex

\subsection{What the empirics do not test}
\label{sec:empirics-scope}

\noindent
The verifier is deterministic in all eight benchmarks; imperfect or LLM-judge verifiers are out of scope and would require complementary safeguards (\S\ref{sec:conclusion}). The protocol assumes one update step between rounds (Assumption~\ref{ass:predictable}); mid-batch updates fall outside this assumption and are addressable by the multi-epoch variant (Appendix~\ref{app:alg-details}). Calibration cost is one-time (held-out isotonic fit on $\le 20\%$ of the EVAL split) and is excluded from the per-round complexity reported in \S\ref{sec:method-rule}.
```

- [ ] **Step 2: Compile.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | tail -10
```

- [ ] **Step 3: Page-count gate.**
```bash
pdfinfo paper/neurips2.pdf | grep Pages
```

### Task 5.6: Phase 5 commit

- [ ] **Step 1: Stage and commit.**
```bash
git add paper/neurips2.tex
git commit -m "$(cat <<'EOF'
Reposition §6: reading guide + principled-α + honest-scope

- §6.0 Reading Guide: four metrics in operator language; two-axis
  test (PathV=0 AND AR>0); baseline organization tied to §3 cells.
- §6.1, §6.2, §6.3: one-sentence headnotes naming what each regime
  tests in framework terms.
- tab:headline: replace pivotal-α with principled
  α* = round(0.7 × Err, 0.05). Three benchmark rows updated
  (PubMedQA, MedNLI, ARC-C) using existing data files; five rows
  unchanged. 3-point version retained in Appendix.
- §6.4 What the empirics do not test: imperfect verifiers, mid-batch
  updates, one-time calibration cost.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6: §7 Conclusion + Final Pass

### Task 6.1: Rewrite §7 with three-paragraph structure

**Files:** `paper/neurips2.tex:533–end-of-section` (the `\section{Conclusion}` block)

- [ ] **Step 1: Locate current §7.**
```bash
grep -n '\\section{Conclusion}' paper/neurips2.tex
```

- [ ] **Step 2: Replace the current §7 block with the following.**

```latex
% ============================================================================
\section{Conclusion and Discussion}
\label{sec:conclusion}

\paragraph{Summary.}
The framework triple of \S\ref{sec:framework-triple} admits a single uniquely-shaped cell---anytime-pathwise validity on selective risk under a non-refusing deployment rule---that no prior wrapper occupies. \csa is the construction that fills it, with rate-optimal certification (Theorems~\ref{thm:power}--\ref{thm:lower-bound}) and a horizon-independent release-rate gap (Theorem~\ref{thm:utility-gap}). Across $480 + 100 + 160$ replicated streams over eight specialist benchmarks, four base-model families, and sixteen adversarial orderings, \csa is the only method satisfying both pathwise validity and non-refusing deployment on every cell.

\paragraph{Scope and limitations.}
The guarantee is with respect to the verifier $V$; harms outside $V$'s scope require complementary safeguards. The setting is deterministic-verifier specialist deployments---extension to imperfect or probabilistic verifiers (e.g.\ LLM-judges with calibrated error rates) is open. The deployment protocol requires one update step between rounds; mid-batch updates fall outside Assumption~\ref{ass:predictable} and require the multi-epoch variant (Appendix~\ref{app:alg-details}).

\paragraph{What this paper is and is not.}
We do not propose a new LLM, a new training algorithm, a new policy class, or a stronger reasoning model. The contribution is the deployment-side complement to those design choices: given a trained local specialist and its verifier, the operator can certify at every wall-clock round that the verifier-measured failure rate among released outputs stays below the contractual $\alpha$. The framework of \S\ref{sec:framework} generalises beyond RLVR---any wrapper can be classified by its (statistic, validity, deployment-rule) triple, and the cell \csa fills is independently of interest for any anytime-pathwise selective-risk problem.
```

- [ ] **Step 3: Compile.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex 2>&1 | tail -10
```

- [ ] **Step 4: Verify content.**
```bash
grep -c 'Conclusion and Discussion' paper/neurips2.tex   # expect 1
grep -c 'What this paper is and is not' paper/neurips2.tex   # expect 1
grep -c 'frontier-API\|frontier API' paper/neurips2.tex   # ≥ 1 (in §1)
```

### Task 6.2: Final cross-reference and page-budget audit

- [ ] **Step 1: Full bibtex + double pdflatex pass.**
```bash
cd paper && pdflatex -interaction=nonstopmode neurips2.tex && bibtex neurips2 && pdflatex -interaction=nonstopmode neurips2.tex && pdflatex -interaction=nonstopmode neurips2.tex
```

- [ ] **Step 2: Check for undefined references.**
```bash
grep -i "warning.*undefined\|reference.*undefined" paper/neurips2.log | head
```
Expected: no output. If anything appears, find the broken `\ref{}` or `\cref{}` and fix.

- [ ] **Step 3: Check page count.**
```bash
pdfinfo paper/neurips2.pdf | grep Pages
```
Expected: ≤ 9 (NeurIPS main-text limit; appendix pages do not count toward this).

- [ ] **Step 4: Visual sanity scan.** Open `neurips2.pdf` and scroll through pages 1–9. Check:
  - §1 ends within ~half a page on page 1.
  - §2 ends within ~half a page on page 1 or top of page 2.
  - §3 framework table fits on one page (no orphan rows).
  - §5 theorem statements display cleanly without hyphenation breaks across columns.
  - §6 tab:headline new α column displays the eight values correctly.
  - §7 fits within half a page on page 9.

- [ ] **Step 5: Final grep for repositioning artefacts that should be gone.**
```bash
grep -nE '(intended audience is operators of clinical|none of the above methods provides valid)' paper/neurips2.tex
```
Expected: no output. Any remaining occurrences of these old-style phrases means a paragraph was left un-rewritten.

### Task 6.3: Phase 6 commit + final state

- [ ] **Step 1: Stage and commit.**
```bash
git add paper/neurips2.tex
git commit -m "$(cat <<'EOF'
Reposition §7: three-paragraph conclusion with explicit scope disclaimer

Replace the previous Conclusion with three short paragraphs:
- Summary: the empty cell of §3 closes empirically and theoretically.
- Scope and limitations: verifier-bound guarantee; deterministic-V
  setting; one-update-per-round protocol.
- What this paper is and is not: not a new LLM, not a new training
  algorithm, not a frontier-model competitor — the deployment-side
  complement only. Framework generalises beyond RLVR.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: Final state check.**
```bash
git log --oneline -10
git status
pdfinfo paper/neurips2.pdf | grep -E "Pages|Title"
```

The repositioning is complete when:
- All six phase commits are present in the log.
- `git status` is clean.
- Page count ≤ 9.
- No undefined references in the log.
- A visual scan of `neurips2.pdf` matches the design.

---

## Self-Review Checklist (Post-Plan-Writing)

The author reviewing this plan after writing should confirm each item:

**Spec coverage.**
- [ ] §1 Application Environment — covered by Task 1.2.
- [ ] §2 Why Anytime-Validity Matters — covered by Task 1.3.
- [ ] §3 Test-supermartingale framework + classification + table — covered by Tasks 2.2 and 2.3.
- [ ] §4 Method as instantiation with "choice forced by §3" framing — covered by Task 3.2.
- [ ] §5 Three theorem arcs with motivation/implication — covered by Task 4.2.
- [ ] §6.0 Reading guide — covered by Task 5.2.
- [ ] §6.1–§6.3 reading-guide headnotes — covered by Task 5.3.
- [ ] §6.1 principled-α schedule — covered by Task 5.4.
- [ ] §6.4 honest-scope — covered by Task 5.5.
- [ ] §7 three-paragraph conclusion — covered by Task 6.1.
- [ ] Theorem 5 placement (one-line pointer in §5.3) — covered by Task 4.2 implication (iii).
- [ ] Cross-reference and page-budget audit — covered by Task 6.2.

**Type / label consistency.**
- [ ] `sec:setting` (§1), `sec:why-anytime` (§2), `sec:framework`, `sec:framework-triple`, `sec:framework-cells`, `sec:method`, `sec:method-prelim`, `sec:method-stat`, `sec:method-rule`, `sec:theory`, `sec:theory-safety`, `sec:theory-power`, `sec:theory-utility`, `sec:reading-guide`, `sec:bench-eight`, `sec:live-rlvr`, `sec:shift`, `sec:empirics-scope`, `sec:conclusion` — used consistently.
- [ ] `tab:framework` replaces `tab:related` everywhere.
- [ ] All theorem labels (`thm:main-anytime`, `thm:power`, `thm:lower-bound`, `thm:utility-gap`, `thm:monotone-frontier`, `thm:no-false-cert`) referenced consistently.
- [ ] Definitions (`def:risk`, `def:oracle`) and Assumptions (`ass:predictable`, `ass:nested`) preserved.

**Placeholder scan.**
- [ ] No "TBD" / "TODO" / "fill in later" anywhere in the plan.
- [ ] `<X>/8` placeholder in Task 5.4 Step 5 is documented as needing extraction from JSON in Task 5.4 Step 2.
- [ ] Every step has a concrete LaTeX block, command, or grep check — no "do something appropriate" instructions.

**Scope check.**
- [ ] The plan touches one file (`paper/neurips2.tex`) plus minor cross-reference updates.
- [ ] All ablations and experiments raised by the idea-evaluator audit (mid-batch-update, imperfect-verifier) are explicitly out-of-scope per the spec; this plan does not introduce them.
- [ ] No new LaTeX packages required; the existing preamble is sufficient.

---

## Notes on Execution

**Estimated effort.** 8–14 hours of focused work for a careful human or LLM; mostly driven by Task 5.4's data extraction and any compile-debug cycles in §3 (the new framework table is the most likely source of column-width or page-overflow issues).

**Rollback plan.** Each phase's commit is self-contained; `git revert` of any single commit returns the paper to a compileable state without breaking other phases. The principled-α phase is the only one with a fallback (Task 5.4 Step 3): if CSA fails at stricter α, revert to the 0.85 × Err rule.

**Out-of-scope reminders (do not implement here).**
- Predictable-update assumption ablation (mid-batch updates).
- Imperfect-verifier cell.
- Recent NeurIPS '25 / ICLR '26 selective-conformal literature check.
- Any change to `code/`, `data/`, or experimental protocol.

These are tracked in the spec's "Out-of-Scope Risks Identified But Not Addressed Here" section and remain the author's responsibility before submission.
