---
title: CSA NeurIPS 2026 Paper Repositioning Design
date: 2026-04-28
status: Draft awaiting user review
target: paper/neurips2.tex
---

# Repositioning Design: *Conformal Selective Acting* for the Local Specialist LLM Setting

## 1. Goal

Reposition the existing manuscript so that:

1. The deployment context is a **local specialist LLM** RLVR-fine-tuned and installed inside a regulated organization, *not* a frontier API model.
2. The paper does not compete with ChatGPT / Claude / Gemini; it provides the deployment-side complement that lets a local specialist meet a per-deployment SLA.
3. The mathematical machinery (Ville-type e-processes, supermartingales, predictable bets) is introduced through a **unifying framework** that classifies prior wrappers, so CSA appears as the unique cell that satisfies the requirements derived from the deployment context.
4. Empirical results are read through the framework — every metric maps to a property of the (statistic, validity, deployment-rule) triple.

The mathematics, theorems, and empirical results are unchanged. What changes is the framing of §1, §2, §3, and the reading guide of §6, plus targeted additions to §5 and §7.

## 2. Decisions Taken

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| Q1 | Application scope | **Narrow-domain class** — local specialist LLMs in regulated narrow domains (clinical, legal, financial, regulated science Q&A); medical as running example | Matches the existing 8-benchmark empirical breadth; lets empirics do the generalization work without diluting the local-specialist thesis |
| Q2 | Mathematical framework | **Test-supermartingale + deployment-rule** triple: (test statistic $M_t(q)$ on $(\mathcal F_t)$, validity guarantee type, deployment rule) | The framework's machinery is literally what CSA uses — strongest internal coherence between §3 and §4 |
| Q3 | Why-anytime hook | **Coupled hook (C)** — weak base + non-poolable deployment compose multiplicatively | Closes both reviewer escape hatches ("use a stronger model" and "averages are fine if the model is good enough") |
| A | Architecture | **Architecture 1** — standalone motivational duo (§1+§2), theorem section earns its space via motivation/implication framing | Matches user-specified 7-section flow exactly; keeps theorem statements in main text as the abstract advertises |
| T1 | Table 1 in §3 | **Replace with framework-organized table** — columns: Statistic, Risk type, Validity, Rule, Non-exch, Updt, Util-gap | Both axis sets (framework + deployment properties) in one object |
| T5 | Theorem 5 placement | **One-line pointer in §5.3 Implication (iii)**, full statement in Appendix | §5 stays compact and three-arc-clean; structural backstop visible in main text |
| α | Pivotal-α schedule | **Single principled point ($0.7\,\mathrm{Err}$ rounded to 0.05)** in headline; 3-point version in Appendix | Defends against cherry-pick objection without restructuring tab:headline |
| L | Limitations placement | **§6.4 in §6** (empirical limits next to data) | Limitations of the regimes are empirical not philosophical; honest to keep them next to the data |

## 3. Page Budget (9-page NeurIPS Main Text)

| Section | Pages | Function |
|---------|-------|----------|
| §1 Application Environment | 0.50 | Deployment unit + loop + contract; medical running example pinned |
| §2 Why Anytime-Validity Matters | 0.50 | Coupled hook (C); marginal/long-run-average/post-hoc all eliminated |
| §3 Related Work via Framework | 1.25 | (Statistic, Validity, Rule) triple introduced + 10 baselines classified |
| §4 Method as Instantiation | 1.25 | CSA derived from framework + algorithm |
| §5 Theorems with Implications | 1.25 | Three arcs: Safety, Power, Utility |
| §6 Numerical Studies + Reading Guide | 3.00 | §6.0 reading guide + 3 regimes + §6.4 honest scope |
| §7 Conclusion + Discussion | 0.50 | What was shown / scope / what we are not contributing |
| **Total** | **8.25** | Leaves ~0.75 pp buffer for figure captions, table rules, spacing |

§1 and §2 must be tightly written. If either grows past ~0.6 pages during drafting, that is the signal to either compress or fall back to Architecture 2 (move some theorem detail to appendix).

## 4. Section-by-Section Design

### §1 Application Environment (~0.5 pages)

**Argumentative job.** By the end of §1 the reader can answer: *what is the deployment unit this paper protects?*

**Three paragraphs, each grounded in a concrete artifact:**

1. **The deployment unit.** A local specialist LLM, RLVR-fine-tuned on domain data, installed inside a regulated organization (hospital, law firm, financial reporting unit, regulated science Q&A). Specialist, not frontier API; narrow input distribution. Domain examples — medical (running example), legal, financial, regulated science — match the empirical study.
2. **The deployment loop.** Each round: model proposes an output; deterministic verifier (clinical rule, legal-citation match, arithmetic check, math grader) returns binary; operator decides whether to release. The model is updated on local data via online LoRA or continued policy gradient — RLVR at deployment, not only at training.
3. **The contract.** Per-deployment error budget $\alpha$ on released outputs, evaluated against the verifier on this deployment's stream. The operator cannot pool across hospitals, cannot send telemetry to a central A/B test, cannot wait for the long-run average. Frontier-API alternatives are explicitly disclaimed (data residency, latency, fine-tuning needs, cost) — they are not the comparison set.

**Material that moves in.** Current §1 "intended operating context" paragraph (lines 124–125) and parts of "scope" (lines 121–122), reframed around the local-specialist axis.

**New material.** Explicit "deployment unit + loop + contract" decomposition; medical-as-running-example pinned in para 1; explicit disclaimer of frontier-API comparison.

**Material that moves out.** Limitations-of-existing-approaches paragraphs (current lines 127–136) → §3.

**Risk.** 3 paragraphs in 0.5 page = ~5 sentences each, no padding. If para 1 wants to grow into a list of regulated industries, cut back to the four already covered empirically.

**Contributions list.** Hold the existing 4-bullet contributions list (current lines 138–144) at the *end* of §1 in 4 lines. NeurIPS readers expect it; cutting it loses signal. Trim to one line per contribution.

### §2 Why Anytime-Validity Matters in This Setting (~0.5 pages)

**Argumentative job.** Establish that *only* pathwise + anytime + selective + non-exchangeable matches §1's contract.

**Three paragraphs executing the (C) coupled hook:**

1. **The deployment side forces pathwise + anytime.** The contract is per-deployment, on this stream, evaluated at every wall-clock step. An early failure spike is an SLA breach, not a tail of the average. Marginal-time validity targets the wrong probability measure — there is no population to marginalize over when the contract is on this hospital's stream. Long-run-average validity targets the wrong horizon — averages cannot retire an incident that has already occurred.
2. **The model side forces anytime.** The specialist still has 5–34% verifier failure on its target benchmark (Table 1, current draft). On a weak base, a 5-pp spike in the first thousand outputs is plausible at the *typical* operating point, not just at the tail. The two arguments compose multiplicatively: a marginal guarantee on a pooled population is a doubly-wrong target for a local specialist deployment.
3. **The post-hoc-test escape hatch is closed.** The natural composite — sequential test on the policy's released outputs after-the-fact — is *invalid*, not merely loose. The score $S_t$ and policy $\pi_t$ are updated jointly; a test calibrated on prior $S$ targets a probability measure distinct from the one governing subsequent rounds. Certification must be predictable with respect to the same filtration that drives the optimizer.

**Material that moves in.** Current §1 "Limitations of existing approaches" + post-hoc-invalidity paragraph (lines 127–136).

**New material.** The (C)-style explicit decomposition: deployment-side / model-side / joint claim, named separately so a reviewer's objection can be located.

**Material that moves out.** Bullet-list-of-five-prior-methods → §3.

**Risk.** Para 2 must do real work — the multiplicative composition is the paper's central motivational sentence. If para 2 is one sentence the section fails. Aim for 4–5 sentences in para 2.

### §3 Related Work via Test-Supermartingale Framework (~1.25 pages)

**Argumentative job.** Set up the framework's mathematical language *and* show that every prior wrapper occupies a particular cell, with one cell empty — the cell §4 fills.

#### §3.1 The framework (~0.6 pages)

- Define deployment protocol formally but compactly: at each $t$ operator observes $X_t$, policy emits $\widetilde Y_t$, predictable score $S_t(X_t,\widetilde Y_t)$ computed, gate $A_t(q):=\1\{S_t\le q\}$ produces an action, verifier emits $V_t\in\{0,1\}$. Filtration $\mathcal F_t$. (Folds in necessary half of current §3 Problem Setup.)
- Introduce the **wrapper-as-triple** abstraction:
  1. *Test statistic.* A predictable process $\{M_t(q)\}_{t}$ on $(\mathcal F_t)$, indexed by hyperparameter $q$, measuring whether $q$ is "still good enough."
  2. *Validity guarantee.* What probability bound holds on $\{M_t(q)\}$: fixed-horizon high-prob, long-run-average, marginal-time, or anytime-pathwise.
  3. *Deployment rule.* A predictable map from the certified set $\{q : M_t(q)\text{ admissible}\}$ to a per-round action.
- State Ville's inequality once, in operator language: a non-negative $(\mathcal F_t)$-supermartingale starting at $\le 1$ satisfies $\mathbb P(\sup_t M_t\ge 1/\delta)\le\delta$ — anytime, pathwise, no exchangeability required. **Conclusion: the only mathematical object delivering anytime-pathwise validity on an adaptive stream is a (super)martingale.**
- Distinguish **selective** from **marginal** risk *as a property of the statistic*: marginal statistic uses $1{-}V_t$ (every round counts); selective statistic uses $A_t(1{-}V_t)$ (only released rounds). They test different probability measures and are not interchangeable.

#### §3.2 Prior methods as cells of the framework (~0.65 pages)

- One paragraph each (3–4 sentences) classifying baselines, organized by validity type:
  - **Fixed-horizon offline** (LTT, CRC, ConfFact): empirical-risk statistic on exchangeable calibration set, constant-threshold rule.
  - **Long-run-average online** (ACI, SAOCP): adaptive miscoverage estimate, dynamic-threshold rule, no pathwise control.
  - **Marginal-time non-exchangeable** (NEX-Conf, CoFact): density-ratio-reweighted nonconformity, valid at each single $t$ but not simultaneously.
  - **Anytime-pathwise marginal** (A-RCPS): e-process *on marginal failure*, anytime-pathwise but on the wrong risk target.
- **Replace existing Table 1** with framework-organized table. Rows: methods. Columns: Statistic | Risk target | Validity | Rule | Non-exch | Updt | Util-gap. Empty cell — *e-process per threshold, selective risk, anytime-pathwise, max-certified-threshold rule* — is exactly where CSA lands in §4.
- **Two sentences explicitly closing the A-RCPS gap.** "A-RCPS uses the right kind of statistic (e-process) and gets the right kind of validity (anytime-pathwise), but on the marginal-failure measure. The selective-risk measure is not derivable from the marginal one; a different statistic is required, which is what §4 constructs."

**Material that moves in.** Current §2 Related Work and Positioning + Table 1 (lines 147–179); post-hoc-invalidity argument's predictability requirement; fragments of current §3 Problem Setup needed to define the protocol.

**New material.** The (statistic, validity, deployment-rule) triple as explicit organizing axis; Ville-inequality "single sentence" naming supermartingales as the unique anytime-pathwise object; selective-vs-marginal-as-property-of-the-statistic distinction.

**Material that moves out.** Longer assumption commentary (current lines 209–213) → appendix. Per-method failure narratives → existing head-to-head appendices.

**Risk.** §3.1 must not become a conformal-prediction tutorial. Cut anything not classification-load-bearing. Framework table must fit column width without `\resizebox`.

### §4 Method: CSA as Instantiation (~1.25 pages)

**Argumentative job.** Derive CSA mechanically from §3 — pick the test statistic, pick the deployment rule, observe what validity falls out. Reader should leave §4 thinking *of course this is the wrapper; it is what the framework forces once I commit to anytime-pathwise + selective + non-refusing*.

#### §4.1 Preliminaries (~0.25 pages)

- Selective verifier risk $R_T^{\mathrm{act}}$ and oracle threshold $q_t^\star$ (current Definitions 1–2 kept).
- Two assumptions, each with one-line operator-side justification:
  - **Predictable pipeline** (Assumption 1): $\pi_t, S_t$ are $\mathcal F_{t-1}$-measurable. *Operator side*: enforced by deployment protocol — one update step between rounds.
  - **Nested gates with monotone risk** (Assumption 2): held-out isotonic calibration of the score makes $\mathbb P(V_t{=}0\mid S_t\le q)$ nondecreasing in $q$. *Operator side*: one-time calibration step on operator's own data.

#### §4.2 The CSA test statistic (~0.4 pages)

- *Choice forced by §3.* Anytime-pathwise validity → must be a supermartingale under the unsafe null. Selective risk → must be built from the gated excess-risk increment $X_t(q):=A_t(q)((1{-}V_t){-}\alpha)$, not from $1{-}V_t$.
- Ville-type e-process per threshold:
  $$E_{j,t}(q) := \prod_{s=\tau_j}^{t}\!\bigl(1-\lambda_{j,s}(q)X_s(q)\bigr),\qquad E_{j,\tau_j-1}(q):=1.$$
- Proposition (E-process validity, current Prop 1): under unsafe null, $\{E_{j,t}(q)\}$ is non-negative $(\mathcal F_t)$-supermartingale. *One-sentence framing*: this is the framework's anytime-pathwise statistic, built on the framework's selective-risk increment.
- Adaptive predictable bet $\lambda_{j,t}(q)$ from running mean (current eq. 4); one-sentence "predictable plug-in for unknown margin" + appendix pointer.

#### §4.3 The CSA deployment rule (~0.3 pages)

- *Choice forced by §3 + non-refusing requirement.* Deploy the most permissive certified threshold. Under monotone risk, if $q$ is certified then every $q'\le q$ is implicitly safe, so $\max\mathcal C_{j,t}$ is a valid action.
- Bonferroni over threshold grid with epoch-counting factor $\delta_{j,q}=6\delta/(\pi^2|\mathcal Q|j^2)$. *One-sentence framing*: Bonferroni cost = price of testing all thresholds; $j^2$ factor = price of revoking certifications across epochs.
- Certified set $\mathcal C_{j,t}=\{q : E_{j,t}(q)\ge \delta_{j,q}^{-1}\}$; action $q_{t+1}=\max\mathcal C_{j,t}$; abstain if empty.

#### §4.4 Algorithm and complexity (~0.3 pages)

- Algorithm box (current Algorithm 1, kept verbatim).
- Per-round complexity $O(|\mathcal Q|)=O(15)$, $O(1)$ memory per threshold, no held-out calibration store after $t=0$, no telemetry. *Operator side*: drops onto existing RLVR pipeline as wrapper, not fork.
- One paragraph pointing at multi-epoch variant in appendix as protection against monotone-risk degradation under sharp shift.

**Material that moves in.** Current §3 Problem Setup (compressed) + current §4 Method (kept).

**New material.** The "choice forced by §3" framing on each subsection — every CSA design choice presented as the unique element-of-the-framework satisfying constraints from §1–§3.

**Material that moves out.** Implementation-choices-not-fixed-by-theory, computational/memory analysis, verifier-evaluation-on-abstained-rounds → existing appendices.

**Risk.** §4.2 is the densest subsection. If anything overflows it is this one. Mitigation: cut the adaptive-bet derivation prose; just state the bet, point at appendix, rely on §3.1's "predictable plug-in" framing.

### §5 Theorems with Motivations and Implications (~1.25 pages)

**Argumentative job.** Three theorem arcs, each tied to one operator concern from §1–§2. Every theorem statement bracketed by *one motivation sentence* and *2–3 implication sentences*. Framework triple from §3 is the recurring connective.

#### §5.1 Safety: pathwise anytime-valid selective risk (~0.4 pages)

- *Motivation.* §2's contract demands anytime-pathwise validity on selective risk. The framework forced us to a per-threshold e-process; does it deliver?
- *Theorem 1 (Anytime-valid selective risk).* For all $T\ge 1$, with probability $\ge 1{-}2\delta$:
  $$R_T^{\mathrm{act}}\le \alpha + \bar\nu_T + c_1\sqrt{\tfrac{\log(1/\delta)+\log\log(eN_T+e)}{N_T\vee 1}}+c_2\tfrac{\log(1/\delta)+\log\log(eN_T+e)}{N_T\vee 1}.$$
- *Implications.*
  1. "For all $T$" is simultaneous — same probabilistic event protects every wall-clock step. SLA-shaped object §2 needed.
  2. Slack decays in $N_T$ (released-output count), not $T$ (round count) — releasing shrinks the bound, abstaining does not. Right operator incentive.
  3. Under exact stability $\bar\nu_T=0$; under bounded drift $\bar\nu_T$ small and multi-epoch variant retires stale certificates at deterministic boundaries.
- *Corollary 1 (asymptotic).* One line: $\limsup R_T^{\mathrm{act}}\le\alpha$ a.s.

#### §5.2 Power: rate-optimal certification (~0.45 pages)

- *Motivation.* Anytime-validity is empty if CSA never certifies a permissive threshold. Framework triple only buys §1's contract if certificate arrives fast enough.
- *Theorem 2 (upper bound).* For safe threshold $q\le q_j^\star$ with margin $\bar\eta_{j,q}>\nu_j$ and deployed plug-in bet, $\mathbb E[\tau^{\mathrm{cert}}_{j,q}-\tau_j]\le 4(\log(1/\delta_{j,q})+1)/(\bar\eta_{j,q}-\nu_j)^2$.
- *Theorem 3 (lower bound).* Any sequential test certifying a single threshold at type-I level $\delta$ on a safe Bernoulli instance with margin $\bar\eta$ has $\mathbb E[\tau]\ge\Omega(\log(1/\delta)/\bar\eta^2)$.
- *Implications.*
  1. Rates match in $\bar\eta$ and $\delta$ — CSA is the first online selective-conformal wrapper proven to certify at the optimal rate.
  2. Only available speed-up is widening margin $\bar\eta$ via better score — operator-actionable lever is calibration quality, not wrapper hyperparameters.
  3. Under predictable updates, adaptive plug-in bet attains this rate without knowing $\bar\eta$ in advance.

#### §5.3 Utility: horizon-independent release-rate gap (~0.4 pages)

- *Motivation.* Safety is necessary; the operator's economic question is whether the wrapper releases as much as the oracle threshold would. Horizon-dependent loss compounds forever; horizon-independent loss is one-time.
- *Theorem 4 (utility gap).* Under approximate frontier improvement with per-round slack $\xi_t\ge 0$, with probability $\ge 1{-}2\delta$ for all $T$:
  $$\mathrm{Gap}_T\le G^\star(B_T):=\sum_{j=1}^{J_T}\sum_{q\le q_j^\star}\frac{2C_0\log(2m J_T^2/\delta)}{(\bar\eta_{j,q}-\Xi_j)_+^2}.$$
- *Implications.*
  1. Under exact stability ($\xi_t\equiv 0$, $J_T{=}1$), gap collapses to single fixed cost depending only on $|\mathcal Q|$, $\delta$, and threshold margins — *horizon-independent*, operator pays once.
  2. Under bounded total slack $B_\infty<\infty$, $\mathrm{Util}_T/\mathrm{Util}_T^\star\to 1$ — system that runs forever asymptotically reaches oracle utility.
  3. *Theorem 5 (Appendix): bounded frontier drift suffices for $\xi_t\equiv 0$, and the published-RLVR class satisfies it.*

**Closing paragraph (~3 sentences).** Three layers of safety distilled from current Remark 1: (a) per-threshold e-process is supermartingale under Assumption 1 alone, (b) no-false-certification adds bounded drift via multi-epoch variant, (c) realized selective risk inherits certified margin via max-certified-threshold rule.

**Material that moves in.** Current §5 Theory in full.

**New material.** Motivation–statement–implication triplet around each theorem; framework-triple connective in each motivation sentence; "operator-actionable lever" framing on Theorem 2 (transition into §6's calibration-sensitivity discussion).

**Material that moves out.** Theorem 5 full statement → Appendix; longer "Interpretation" paragraph after Theorem 3 (lines 297–298) → collapses into Implication (i) of §5.2.

**Risk.** §5.2 is the longest subsection at 0.45 pages. If formal statements eat the budget, the implication sentences are what gets cut — and they are the new repositioning content. Mitigation: state each theorem on a single line of display math, drop verbose conditions to "under Assumption~X" with appendix cross-ref. Theorem 1's full bound considered as $R_T^{\mathrm{act}}\le\alpha+O(N_T^{-1/2})$ headline + footnote with full slack expression.

### §6 Numerical Studies and How to Read Them (~3 pages)

**Argumentative job.** Each empirical regime answers one operator concern from §1–§2. The reading guide maps the four headline metrics back to §3's framework triple.

#### §6.0 Reading guide (~0.3 pages)

- **Four metrics in operator language.** **PathV** = replications where running selective risk crossed $\alpha$ at any post-burn-in step → empirical evidence for anytime-pathwise validity (§5.1). **Risk** = realized verifier-fail rate among released items at horizon → certified bound, realized. **AR** = release rate → empirical evidence for non-refusing deployment rule (§4.3). **Refused** = cells where AR $=0$ across all replications → failure mode of the (anytime, refusing) trade-off.
- **The two-axis test.** A useful wrapper must be PathV $=0$ *and* AR $>0$ on every cell. PathV $=0$ alone is achievable trivially by refusing; AR $>0$ alone is achievable trivially by always-acting. Each table reports both; the empirical claim is that no method other than CSA satisfies both axes simultaneously across all cells.
- **Baseline organization tied to §3 cells.** Five online heuristics (long-run-average or no-validity cells); three offline conformal (fixed-horizon cells); one non-exchangeable (marginal-time cell); A-RCPS (anytime-marginal cell, deferred to Appendix).

#### §6.1 Eight high-stakes specialist benchmarks (~0.9 pages)

- *Reading-guide headnote (1 sentence).* This regime answers whether CSA's anytime-pathwise + selective + non-refusing guarantee holds on the typical operating point of a regulated specialist deployment.
- Table tab:benchmarks (8 specialist bases × verifier-fail rates) — kept.
- Table tab:headline (CSA vs all baselines at each benchmark's principled $\alpha^\star=0.7\,\mathrm{Err}$ rounded to nearest $0.05$) — *single principled point per benchmark*; 3-point version in Appendix. Drop Always-Act row to fit.
- Result paragraph (4 sentences): aggregate CSA 0/480 violations; CRC 0/480 but refuses 26/48 cells; NEX-Conf 142/480; online heuristics 32–34/48 cells violated. *Read this as*: only CSA simultaneously satisfies the contract and is non-refusing on every benchmark — empty cell of §3 closes empirically.
- Pointer to Appendix for risk-vs-$\alpha$ curves, per-method-per-benchmark tables, split-seed sensitivity.

#### §6.2 Live RLVR with online LoRA (~1.0 pages)

- *Reading-guide headnote (1 sentence).* This regime tests the framework's predictability requirement (Assumption 1) under the operator's actual update protocol — online LoRA between rounds, four base-model families.
- Table tab:live-cells (5 cells, $T\in[800,4000]$ genuine rounds) — kept.
- Table tab:live-headline (10 methods × 4 cells; AR + Risk + PathV) — kept; consider dropping Always-Act row to fit without `\resizebox`.
- Figure fig:live-trajectory (Qwen2.5-Math-7B, top: running risk, bottom: running AR among PathV=0 methods) — kept.
- Result paragraph (4 sentences): CSA 0/100 violations with AR $\ge 50\%$ on every cell; next-best pathwise-valid (LTT/ConfFact) 1.4–5.5× lower AR; ACI/SAOCP/Naive-Tuning $\ge 7/20$ violations on every cell. *Read this as*: predictability requirement is satisfiable under realistic protocol; heuristics' violation rates would breach any per-deployment SLA from §1.
- Pointer to Appendix for hyperparameter sensitivity, sparse-verifier ablation, per-cell HEAD-QA/MedQA detail.

#### §6.3 Distribution-shift robustness (~0.55 pages)

- *Reading-guide headnote (1 sentence).* This regime tests pathwise validity under adversarially ordered streams — the worst case for any wrapper not built on a martingale.
- Table tab:shift-summary (16 cells × 10 reps each) — kept.
- Result paragraph (4 sentences): CSA 0/16 violated cells (mean Risk 3.2%); CRC 4/16; LTT 4/16; ACI/SAOCP/Fixed-Threshold/Always-Act 14–16/16 violated. *Read this as*: anytime-pathwise validity is reordering-invariant by construction (supermartingale property does not depend on stream order); long-run-average baselines fail here precisely because their averaging — source of efficiency on i.i.d. streams — is what an adversarial ordering attacks.

#### §6.4 What the empirics do not test (~0.15 pages — three sentences)

- Verifier is deterministic in all eight benchmarks; imperfect or LLM-judge verifiers are out of scope.
- Protocol assumes one update step between rounds; mid-batch updates fall outside Assumption 1 and are addressable by the multi-epoch variant (Appendix).
- Calibration cost is one-time (held-out isotonic fit on $\le 20\%$ of EVAL split) and is excluded from per-round complexity.

**Material that moves in.** Current §6 in full.

**New material.** §6.0 Reading Guide (central repositioning move); reading-guide headnotes on §6.1–§6.3; principled-$\alpha$ schedule kills cherry-pick objection; §6.4 honest-scope paragraph.

**Material that moves out.** Per-method-per-benchmark tables, risk-vs-$\alpha$ curves, hyperparameter sensitivity, sparse-verifier ablation, CoFact/Conf-Arb head-to-head → existing appendices.

**Risk.** Tables eat ~1.5 of 3-page budget. Reading-guide prose + four result paragraphs + three headnotes + §6.4 = ~1.3 pages. If §6.0 grows past 0.3 pages it is repeating §3 — cut. If a table does not fit at full font, drop Always-Act row.

### §7 Conclusion and Discussion (~0.5 pages)

**Argumentative job.** Three short paragraphs. Para 3 is repositioning-critical — disclaims comparisons the title invites that the paper is not making.

1. **What was shown (~3–4 sentences).** Framework triple of §3 admits a single uniquely-shaped cell — anytime-pathwise + selective-risk + non-refusing — that no prior wrapper occupies. CSA fills it, with rate-optimal certification (Theorems 2–3) and horizon-independent release-rate gap (Theorem 4). Across $480 + 100 + 160$ replicated streams over eight specialist benchmarks, four base-model families, and sixteen adversarial orderings, CSA is the only method satisfying both pathwise validity and non-refusing deployment on every cell.
2. **Scope and limitations (~3–4 sentences).** Guarantee is with respect to verifier $V$; harms outside $V$'s scope require complementary safeguards. Setting is deterministic-verifier specialist deployments — extension to imperfect/probabilistic verifiers is open. Protocol requires one update step between rounds; mid-batch updates fall outside Assumption 1 and require multi-epoch variant.
3. **What we are and are not contributing (~3–4 sentences).** This paper does not propose a new LLM, training algorithm, policy class, or stronger reasoning model. The contribution is the deployment-side complement to those design choices: given the trained local specialist and its verifier, the operator can certify at every wall-clock round that verifier-measured failure rate among released outputs stays below contractual $\alpha$. The framework of §3 generalizes beyond RLVR — any wrapper can be classified by its (statistic, validity, deployment-rule) triple, and the cell CSA fills is independently of interest for any anytime-pathwise selective-risk problem.

**Material that moves in.** Current §7 Conclusion compactly.

**New material.** Para 3's explicit "we are not competing with frontier models / are not proposing a new training algorithm" disclaimer; generalization-beyond-RLVR closing sentence repositioning the framework as the longer-term contribution.

## 5. Out of Scope for This Repositioning

The following are *not* changing in this repositioning and are left as-is:

- All theorem proofs (Appendices C–G).
- The RLVR structural layer (Appendix B: bounded frontier drift, single-epoch sufficiency, drift-aware extension).
- The CSA-Epoch multi-epoch algorithm (Appendix A.3).
- All experimental code (`code/`).
- All numerical results (`data/results/`).
- Bibliography (`refs.bib`) — no new citations required by repositioning.
- The 11 figure PDFs (`figures/`) — keep all.
- The NeurIPS broader-impact statement (post-§7, NeurIPS-mandated).

## 6. Out-of-Scope Risks Identified But Not Addressed Here

These were raised by the idea-evaluator audit and are tracked, but are not included in this repositioning:

- **Predictable-update assumption ablation** (mid-batch update protocol). Recommended addition to Appendix; the spec for *that* ablation is a separate design.
- **Imperfect-verifier cell** (LLM-judge with calibrated error rate). Would require new experiment; out of scope for this repositioning.
- **Recent NeurIPS '25 / ICLR '26 selective-conformal literature check.** Manual review by the author before submission, not a repositioning task.

## 7. Transition to Implementation Plan

Once this spec is approved, the next step is a writing plan that decomposes the LaTeX edits into ordered, independently-verifiable tasks (e.g., "rewrite §1 to the three-paragraph deployment-unit/loop/contract structure," "build new framework Table 1 with Statistic/Risk/Validity/Rule/Deployment-property columns," "add reading-guide subsection §6.0," etc.). The writing plan will live alongside this spec and is what the implementation phase consumes.
