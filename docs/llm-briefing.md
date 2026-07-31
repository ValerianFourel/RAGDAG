# RAGDAG — briefing for an LLM

Written 2026-07-31. Self-contained. Assumes no prior context on this project.

Purpose: hand this to a model so it can reason about the project without
re-deriving anything. Every number here was measured from the published
artefacts, not copied from prior prose. Claims are tagged **[FACT]**
(computed from data), **[HYPOTHESIS]** (consistent with data, not established),
or **[PROPOSED]** (not yet done).

Companion documents: `audit.md` (pre-refactor code audit), `session-log.md`
(build history and defect list). Where this document and `session-log.md`
disagree, this one is later and was computed from the published dataset.

---

## 1. The system under study

A three-stage retrieval pipeline, treated as a structural causal model.

```
Q ──► M1 = BM25(Q)     ┐
  └─► M2 = Dense(Q)    ├─► C = TopK(M1) ∪ TopK(M2) ──► R = CE(Q,C) ──► Y = rank(d)
                       ┘
```

- `Q` — query text
- `M1`, `M2` — two first-stage retrievers, each returning its top `K=50`
- `C` — candidate pool, the **union** of the two top-50 lists (~85 distinct docs)
- `R` — cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) rescoring the pool
- `Y` — rank of a chosen target document `d`, censored at `MISSING_RANK = K+1 = 51`

Models: dense `BAAI/bge-small-en-v1.5`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`.
Collections: six BEIR datasets. Hardware: 4×A100, query-sharded.

**The methodological core.** The pipeline is a deterministic function the
authors own. So the cross-world quantity `Y(q1, M(q0))` — the outcome when one
stage sees the old query and the rest see the new one — is **evaluated by
running the pipeline**, not estimated from data. There is no
sequential-ignorability assumption to defend. This is the framework's actual
contribution and nothing in this document challenges it.

**The intervention.** `do(Q := Q + t)`: append one word `t` to the query.
- *Treatment arm*: `t` is drawn TF-IDF-weighted from the target document `d`.
- *Control arm*: `t` is a corpus word absent from both query and document.

**The path decomposition.** Each treated pair is evaluated in five
counterfactual worlds (baseline, full, and three stage-freezes), giving four
signed path effects that sum exactly to the total:

```
reranker = r(baseline)          - r(freeze_candidates)
lexical  = r(freeze_candidates) - r(freeze_dense)
dense    = r(freeze_candidates) - r(freeze_bm25)
residual = the 2x2 lexical x dense interaction contrast
```

`residual` is misnamed: it is a **mediator–mediator interaction**
(VanderWeele & Vansteelandt), not regression noise. `reranker` is the natural
(pure) direct effect; the first-stage total is the natural (pure) indirect effect.

Two first-stage configurations are run for every pair: `union` (BM25 + dense)
and `bm25_only`. Comparing them is how "credit assignment is architectural" is
tested.

---

## 2. The three criteria — exact definitions

These are mechanical pass/fail gates in `report.py`. Understanding their exact
form matters, because two of the three failures below are threshold artefacts
rather than substantive results.

### C1 — INTERVENTION EFFECTS EXIST
`report.py:85`

Cluster-bootstrapped difference in mean **|Δrank|** between treatment and
control arms, resampling whole queries (1000 resamples, `interventions.py:438`).
**PASS iff the 95% CI lower bound > 0.**

Question it answers: does injecting a word from the target document move that
document more than injecting an unrelated word of the same form? I.e. is there
any causal signal at all, above the mechanical effect of lengthening the query.

### C2 — MEDIATION IS NON-DEGENERATE
`report.py:109`

Two legs, **both** required:
- **degeneracy leg**: largest single-path share ≤ **90%**
- **shift leg**: largest share change between the two first-stage configs ≥ **10pp**

Share is defined at `mediation.py:215` as a ratio of mean **absolute** path
magnitudes:

```
share_p = mean|path_p| / Σ_q mean|path_q|      over q ∈ {reranker, lexical, dense, residual}
```

Question it answers: is credit genuinely divided between paths, and does the
division depend on architecture?

**Both thresholds (90%, 10pp) are hardcoded with no derivation.**

### C3 — CONFOUNDING IS LIVE
`report.py:150`

Pick up to 3 concept terms with corpus document frequency in [5%, 40%]
that also appear in a hardcoded `MEDICAL_LEXICON` (`dml_analysis.py:109`).
For each, compare a naive OLS estimate of the concept's effect on cross-encoder
score against a `DoubleMLPLR` estimate with LightGBM nuisances.
**PASS iff ≥1 concept's naive point estimate falls outside the DML 95% CI.**

Question it answers: would an observational analyst, without adjustment, reach a
measurably wrong conclusion about this pipeline?

Overall verdict: 3 PASS → `YES`; ≤1 PASS and no INCONCLUSIVE → `NO`; else `MIXED`.

---

## 3. How the evidence was gathered

Method, for reproducibility. Total elapsed: minutes, not hours.

1. **Downloaded the published artefacts.** `ValerianFourel/ragdag-results` on
   HuggingFace — 219 files, 55 MB, 8 top-level directories.
2. **Built a file-presence matrix** across all directories. Two datasets were
   missing an identical, contiguous suffix of artefacts. That bracketed the
   crash to two source lines without reading a single log file.
3. **Reproduced the crash locally** by calling the suspect function on an empty
   DataFrame. One line of Python, exact exception reproduced.
4. **Recomputed C1 and C2 from the surviving parquets** using the project's own
   estimators (`interventions.cluster_bootstrap_diff`, `mediation.mediation_ratio`),
   recovering verdicts for the two datasets that never produced a report.
5. **Stratified the mediation data** by pair-activity class and by baseline pool
   depth to find *why* C2 fails, rather than just that it fails.
6. **Cross-checked invariants** the dataset card claims (control-arm
   `delta_bm25 == 0`, path additivity) across all six collections.

Key technique: **the shape of what is missing localises a crash more precisely
than logs do**, when artefacts are written in a known sequence.

---

## 4. What was observed — complete results

### 4.1 Baseline retrieval quality [FACT]

| dataset | queries | BM25 | dense | full | CE gain | reranker helps? |
|---|---|---|---|---|---|---|
| quora | 10,000 | 0.8056 | 0.8863 | 0.8280 | **−0.0583** | NO |
| scidocs | 1,000 | 0.1578 | 0.2052 | 0.1696 | **−0.0356** | NO |
| fiqa | 648 | 0.2514 | 0.4035 | 0.3698 | **−0.0337** | NO |
| scifact | 300 | 0.6863 | 0.7127 | 0.6876 | **−0.0252** | NO |
| trec-covid | 50 | 0.5994 | 0.7383 | 0.7478 | +0.0095 | yes |
| nfcorpus | 323 | 0.3233 | 0.3455 | 0.3568 | +0.0113 | yes |

nDCG@10. **The cross-encoder degrades retrieval on 4 of 6 collections.**
`config.KNOWN_RERANKER_HARMFUL` lists only `beir/scifact/test`, so this was
under-recognised. The report *does* emit a warning whenever
`full ≤ best_single` (`report.py:316`), so all four carry it — but
`session-log.md` still records fiqa as a clean `YES` replication.

### 4.2 The three criteria, all six collections [FACT]

C1/C2 recomputed from parquets; C3 read from `dml_comparison.csv`.

| dataset | C1 contrast (95% CI) | C1 | max share | shift | C2 | C3 | verdict |
|---|---|---|---|---|---|---|---|
| trec-covid | +6.092 [+4.642, +7.415] | PASS | 75.4% | 67.0 | PASS | 2/3 | YES |
| nfcorpus | +6.979 [+6.018, +7.932] | PASS | 81.0% | 54.3 | PASS | 3/3 | YES |
| fiqa | +3.967 [+3.450, +4.520] | PASS | 84.2% | 76.8 | PASS | 1/1 | YES |
| scidocs | +5.839 [+5.447, +6.230] | PASS | **90.2%** | 58.3 | **FAIL** | — | MIXED |
| scifact | +1.222 [+0.761, +1.741] | PASS | **92.8%** | 65.2 | **FAIL** | 2/3 | MIXED |
| quora | +2.817 [+2.605, +3.032] | PASS | **95.2%** | 68.8 | **FAIL** | — | MIXED |

- **C1 passes on all six.** The most robust result in the project.
- **C2 fails on three of six**, always on the degeneracy leg. The shift leg
  passes everywhere at 54–77pp against a 10pp bar — a 5–7× margin.
- **C3 is INCONCLUSIVE on scidocs and quora** because `MEDICAL_LEXICON` does not
  intersect a CS-paper or general-question corpus. fiqa's C3 rests on a single
  concept (`risk`, df 7.0%) and is weak evidence.
- **scidocs fails C2 by 0.18pp** (90.18% vs a 90.0% threshold). Dropping inert
  pairs moves it to 90.21%. The verdict is not robust to trivial analysis choices.

### 4.3 Why C2 fails — the diagnosis [FACT]

The share is a ratio of mean absolute path magnitudes. `lexical` and `dense` are
**membership** effects: they are nonzero only when the injected word changes
*which documents enter the top-K union*. If the pool does not churn, both freeze
worlds produce an identical pool and both paths are **exactly zero**.

Pair composition of the `union` config, as % of pairs:

| dataset | inert | reranker only | first-stage only | **both act** | fs magnitude when active | share |
|---|---|---|---|---|---|---|
| trec-covid | 15.7 | 35.7 | 1.6 | **47.0** | 5.88 | 75.4% |
| nfcorpus | 16.3 | 54.8 | 0.6 | **28.4** | 7.69 | 81.0% |
| fiqa | 29.8 | 49.9 | 0.8 | **19.6** | 4.40 | 84.2% |
| scidocs | 23.4 | 53.8 | 0.8 | **22.1** | 3.05 | 90.2% |
| scifact | 64.7 | 27.4 | 0.6 | **7.2** | 1.96 | 92.8% |
| quora | 46.4 | 47.6 | 0.3 | **5.8** | 2.59 | 95.2% |

Correlations with the reranker share (n=6):

| against | Pearson | Spearman |
|---|---|---|
| % pairs where first stage is exactly zero | **+0.939** | **+0.943** |
| baseline censoring rate | −0.899 | −0.943 |
| CE gain (reranker usefulness) | −0.857 | −0.771 |

**The decisive number: conditional on the first stage acting at all, the
reranker share is 65.4 / 66.3 / 70.7 / 78.3 / 84.7 / 81.8% — every collection
under the 90% threshold.** No collection is degenerate among pairs where the
decomposition has something to decompose.

Fully-inert pairs are *not* the cause: they scale all four means proportionally,
so removing them changes the share by ≤0.03pp.

Share stratified by where the target starts in the pool (sentinel 51 = in pool
but reranked below position 50):

| dataset | r≤10 | 11–25 | 26–50 | 51 (censored) |
|---|---|---|---|---|
| quora | 98.7% | 96.6% | 94.3% | 88.0% |
| scifact | 97.8% | 95.5% | 91.5% | 87.0% |
| scidocs | 96.7% | 91.3% | 90.7% | 86.3% |
| nfcorpus | 89.9% | 80.6% | 83.7% | 77.1% |
| fiqa | 81.1% | 87.2% | 85.7% | 81.4% |
| trec-covid | 58.7% | 79.6% | 76.3% | 75.1% |

Monotone decreasing on four of six. trec-covid inverts — it has ~500 judged
relevant documents per topic, so the pool is saturated with relevant competitors.

**Interpretation.** C2 is not measuring whether path attribution carries
information. It is measuring **how often a one-word query edit changes pool
membership**. On easy collections (quora sits at nDCG 0.83) the target is
already buried deep inside a stable pool and the first stage has nothing to do.

### 4.4 The sampling frame is the root cause [FACT]

`interventions.py:326`:

```python
eligible = sorted(set(run.candidates) & relevant)
```

**Every target is a relevant document already in the baseline candidate pool.**
Pool membership is determined by `M1`/`M2` — the mediators.

Important correction to a natural misreading: `base_censored` does **not** mean
out-of-pool. `MISSING_RANK = K+1 = 51` while the union pool holds ~85 documents,
so a censored target is *in the pool but reranked below position 50*. **The
dataset contains zero out-of-pool targets. The recall channel is unmeasured, not
under-measured.**

Causal status: conditioning on `M(q0)` — the *baseline* mediator value — is
conditioning on a pre-treatment function of `q0`. It creates **no collider bias
and no post-treatment bias**. The estimand is well-defined. It is simply
*conditional*, and the conditioning event is precisely the event that neuters
the mechanism being measured.

### 4.5 A finding the reports do not contain [FACT]

`provenance_base`, `provenance_new` and `in_candidates` are already logged in
every `interventions.parquet`. Ejection from the pool
(`provenance_new == 'none'`) is directly observable:

| dataset | treatment | control | ratio | ejection by baseline admission channel (control arm) |
|---|---|---|---|---|
| nfcorpus | 1.71% | 14.89% | 8.7× | bm25 23.3% · **both 1.5%** · dense 31.8% |
| trec-covid | 1.48% | 15.93% | 10.8× | bm25 7.2% · **both 1.8%** · dense 27.3% |
| scidocs | 0.37% | 5.07% | 13.8× | bm25 5.4% · **both 0.1%** · dense 14.5% |
| fiqa | 0.55% | 4.95% | 9.0× | bm25 5.6% · **both 0.0%** · dense 13.3% |
| quora | 0.01% | 0.61% | 46.5× | bm25 2.5% · **both 0.0%** · dense 8.4% |
| scifact | 0.00% | 0.82% | — | bm25 8.3% · **both 0.0%** · dense 6.2% |

Three consequences:

1. **The treatment/control contrast is far cleaner on the membership margin than
   on rank** (9–46× vs C1's +1.2 to +7.0). Censoring is currently flattening the
   strongest signal in the experiment into a sentinel value.
2. **Channel redundancy is causally protective.** Documents admitted by *both*
   channels are essentially never displaced (0.0–1.8%); singly-admitted
   documents are displaced 2.5–32% of the time.
3. **Dense-only admission is the fragile one** in 5 of 6 collections. This
   independently corroborates the stability result (BM25 > full > dense under
   paraphrase) via a completely different probe — pool ejection under noise
   injection.

Standalone: adding one irrelevant word to a query ejects a relevant document
from the pool 15% of the time on NFCorpus.

---

## 5. What worked

### Verified invariants [FACT]

| invariant | result |
|---|---|
| control-arm `delta_bm25 == 0` | **100.00%** on all 5 post-fix collections |
| path additivity: Σ signed paths − total | **exactly 0.00e+00**, all 6 collections, ~63,000 pairs |
| control-arm `select_prob` flat | 1 distinct value everywhere |
| all 4 shards present | every dataset — no silent truncation |
| dataset-scoped results | 6 clean directories, no overwrites |

The additivity check is the strongest validation in the project: the
decomposition **reconstructs** the total effect at machine precision rather
than approximating it. On NFCorpus,
`12.7549 − 0.8898 − 1.0791 + 0.7451 = 11.5304`, exactly the measured Δrank.

### The stem-leak fix, visible in the data [FACT]

`_archive_prefix` is a pre-fix run and still contains 2 leaked control terms in
4,512 — `having` and `lasting`, the latter moving its target −23 ranks while
labelled "control". Post-fix NFCorpus: **zero**.

### Engineering that held

- **[FACT]** Sharding: 4 workers × 6 datasets, every shard present in the
  published repo — no silent truncation.
- **[FACT]** Offline/air-gapped execution worked for all six collections.
- **[FACT]** Compute, from the published `timings_*of4.json`: quora ~31
  min/worker (10,000 queries), scidocs ~9 min/worker (1,000 queries).
- **[reported in `session-log.md`, not re-verified here]** sharded output
  bit-identical to a single-process run on the CPU/ONNX path; 4×A100 at ~800 CE
  pairs/s/GPU; 323 queries full design ≈ 9 min.

---

## 6. What failed

### 6.1 The crash — scidocs and quora produced no report [FACT]

**Root cause chain:**

1. `select_concepts` (`dml_analysis.py:109`) filters on `MEDICAL_LEXICON`,
   NFCorpus-specific. On CS papers and general questions the intersection with
   the df band [5%, 40%] is empty. It returns `[]` — and prints a warning saying
   "Module 4 will be reported as INCONCLUSIVE, not as a null result."
2. `build_panel` still runs at full cost (quora: 829,581 rows) and writes
   `dml_panel.parquet` with no `has_*` columns.
3. `analyse(panel, [])` returns an empty DataFrame → `dml_comparison.csv` is
   written at **1 byte** (no header, no rows).
4. `print_comparison` guards `res.empty` (`dml_analysis.py:292`) and returns
   cleanly. **Safe.**
5. `plot_dml` has **no such guard**. `dml_analysis.py:332` does
   `res["naive_coef"]` → **`KeyError: 'naive_coef'`**. Reproduced directly.
6. `run_all.py:316` calls it unconditionally → the merge process dies.
7. Stage 5 (stability) and stage 6 (report) never run; `run_meta.json` and
   `timings.json` never written.

**Cost:** 450 stability rows per dataset were computed on the GPUs and
discarded. All four workers on both datasets completed every stage (quora
~31 min/worker, scidocs ~9 min/worker). The run was thrown away by a plotting
call for a stage that had already correctly declared itself inconclusive.

**Fix:** an emptiness guard at the top of `plot_dml`, matching the one
`print_comparison` already has. **Not yet applied.**

Two hypotheses that were checked and **disproved**: it was not a missing
prefetch (both corpora clearly loaded — quora produced 59,303 interventions),
and it was not walltime (workers finished well inside the allocation).

### 6.2 Publication defects [FACT]

- **Empty and stray directories are published as datasets.** `result_dirs()`
  (`publish.py:142`) returns every subdirectory with no filter, and
  `config.py:244` runs `RESULTS_DIR.mkdir(parents=True, exist_ok=True)` at
  import — so `publish.py` *creates* a phantom directory merely by importing
  config. Published repo contains `_archive_prefix/` (a stale pre-provenance
  run) and `shards/` (the empty `results/shards` created by
  `setup_login.sh:27`), both indistinguishable from real datasets in the listing.

- **`code_fingerprint` in MANIFEST.json is publish-time, not run-time.**
  `write_manifest` calls `config.code_fingerprint()`, which hashes the *current
  working tree*. All 8 directories therefore claim `b5c5b50258862d0e`. But
  `_archive_prefix` is demonstrably older code: it lacks all 12 provenance
  columns added in commit 2abe189 (`select_prob`, `term_idf`, `term_in_title`,
  `term_bm25_form`, …), has no `origin_documents.parquet`, and its mediation
  numbers differ from NFCorpus in the 5th significant digit (81.02548% vs
  81.02627%). **Different code, identical claimed fingerprint.** This is exactly
  the hazard the dataset card warns about, and the field designed to catch it
  cannot. The trustworthy field is `run_meta.json` → `code`, which
  `_archive_prefix`, `shards`, quora and scidocs all lack.

- All 8 manifests are `git_dirty: true` at SHA 6068634. Probably benign (the
  dirt was likely `docs/session-log.md`, which is not in `_CODE_MODULES`) — but
  the manifest cannot demonstrate that.

### 6.3 Methodological weaknesses [FACT unless noted]

- **The absolute-value share is not a decomposition.** Signed paths sum exactly
  to the total; taking `|·|` breaks additivity — which is precisely what
  `nonadditivity_pct_of_total` (0.98%–8.7%) measures. So "81% of the effect" is
  a share of `Σ mean|path|`, a quantity that **is not the total effect**. It is a
  heuristic importance measure presented as an exact decomposition.
- **The 90% and 10pp thresholds are arbitrary** and hardcoded, and one verdict
  turns on 0.18pp.
- **The reranker is a confound.** The share anti-correlates with reranker
  usefulness at −0.86, and 4 of 6 collections run a cross-encoder that actively
  degrades nDCG. **[HYPOTHESIS]** a badly-transferred reranker rescores nearly
  independently of first-stage evidence, so it both destroys nDCG and absorbs
  attribution. Untested.
- **Control terms are not frequency-matched** (median df 0.36% vs treatment
  1.27%, p≈3e-56), which likely inflates C1.
- **DML is not cluster-robust.** SEs measured understated 1.55–1.81×; a
  synthetic replica fired the C3 test on 48.7% of *null* datasets with i.i.d.
  SEs vs 5.3% cluster-robust. C3 survives on real data (2/3 concepts) but must
  be refit with `DoubleMLClusterData`. Also `query_len` is constant within a
  query, so i.i.d. folds let LightGBM fit a per-query intercept and manufacture
  precision.
- **DML sensitivity: RV ≈ |t|/√n** gives 1.51% / 1.34% / 0.18% for the three
  NFCorpus concepts — all below DoubleML's 3% default. So *"unadjusted
  attribution is biased"* is defensible; *"the adjusted +0.09 is the true
  effect"* is not.
- **GPU determinism untested.** `use_deterministic_algorithms(False)`, and CE
  batch composition depends on cache state.

---

## 7. What is proposed

### 7.1 Reframe the headline [PROPOSED]

**Do not lead with "the reranker gets 81% of the credit."** Lead with:

1. **The signed decomposition**, which is exact and defensible. The surprising
   part is the *negative* first-stage signs: raising a document's BM25 score
   *hurts* it, because it pulls competitors into a fixed-size pool. That needs
   no threshold to be interesting.
2. **The two-regime result**, which replicates on all six collections at
   54–77pp. "Credit assignment is architectural" does not depend on the absolute
   share at all.
3. **The share as opportunity × conditional importance.** Marginal share =
   P(first stage acts) × importance when it acts. Both are already computable
   from existing parquets — opportunity 6–49%, conditional share 65–85%. This
   *explains* the 75–95% spread instead of apologising for it.

### 7.2 The recall-regime experiment — and why the obvious version is wrong [PROPOSED]

**The trap.** Sampling out-of-pool targets and re-running the same code produces
the **exact mirror image** of the current artefact. For `d ∉ C`, `CE(q,d)` is
never evaluated, so `reranker = r(baseline) − r(freeze_candidates)` is
mechanically 0 — not "the reranker did nothing" but "the reranker was never
asked". Reranker share would collapse toward 0 by construction and look like a
spectacular refutation. It would be just as wrong as the current inflation.

**The fix: split the margins.** Rank is only defined given admission, so factor
the effect the way the pipeline factors:

```
extensive margin:  S = 1{d ∈ C}      admission   — the first stages decide this
intensive margin:  rank | S = 1      position    — the reranker decides this
```

The current experiment measures the intensive margin *only*, because it
conditions on `S(q0) = 1`. These are different estimands; both are needed.

**The extensive margin has clean structure.** `S = A ∨ B` where
`A = 1{d ∈ TopK(BM25)}`, `B = 1{d ∈ TopK(dense)}`. Binary, exactly computable,
no sentinel. The OR gate makes masking — "already in the room via dense, so
BM25 changes nothing" — a **structural interaction to measure** rather than a
nuisance that deflates the denominator.

**Principal strata are directly observable, and this is a selling point.**
Every target falls into one cell of `(S(q0), S(q1))`:

| | stays out | comes in |
|---|---|---|
| **was out** | never-in (inert) | **rescued** ← the recall effect |
| **was in** | **displaced** ← the harm effect | always-in ← what is measured now |

In a stochastic setting principal stratification needs monotonicity assumptions
and yields bounds. Because the pipeline is deterministic, **every unit's stratum
is observed by construction.** This is the same determinism argument that powers
the cross-world claim, applied to a second problem — it extends the framework
rather than patching it. Note `displaced` is the mechanism behind the negative
first-stage effects already found, so the typology explains the existing result.

**Better outcome than the binary: admission margin.**
`margin_A(d) = bm25(d) − bm25(K-th ranked doc)`, same for dense. Continuous,
**uncensored, defined for every document whether admitted or not**. Kills the
sentinel problem and gives dose-response instead of a binary contrast — and
`term_idf`, `term_tfidf_weight`, `select_prob` are already logged, so injected
BM25 mass is a measurable dose.

### 7.3 Concrete changes [PROPOSED]

| # | change | where | cost |
|---|---|---|---|
| 1 | emptiness guard in `plot_dml` | `dml_analysis.py:325` | 2 lines |
| 2 | filter empty dirs in `result_dirs` | `publish.py:142` | 2 lines |
| 3 | read `code_fingerprint` from `run_meta.json`, don't recompute | `publish.py:158` | small |
| 4 | report share as opportunity × conditional importance | re-analysis | no new compute |
| 5 | replace C2's 90% gate with the stratified curve | `report.py:119` | no new compute |
| 6 | stratified target sampling by admission margin | `interventions.py:326` | new run, **cheaper** |
| 7 | Shapley over the four paths instead of `|·|` shares | `mediation.py:215` | moderate |
| 8 | refit DML with `DoubleMLClusterData` | `dml_analysis.py` | moderate |
| 9 | re-run scifact/scidocs with a domain-appropriate reranker | config | new run |
| 10 | replace `MEDICAL_LEXICON` with frequency-stratified vocabulary sample | `dml_analysis.py:109` | moderate |

On #6: this is **cheaper** than the current design, not more expensive. The
current run spends GPU time on pairs that are 46–65% inert on scifact and quora.
Stratifying by margin spends the budget where the response surface has gradient.
Sample all judged-relevant documents in ~5 bins from deep-in through marginal to
deep-out, equal per bin, reweighted to the population for marginal quantities.

Report per stratum: extensive `ΔP(admitted)` decomposed over the OR gate
(BM25-path, dense-path, masking interaction); intensive rank decomposition
restricted to `always-in` where it is well defined; and the **stratum
proportions themselves**, which are a result rather than a diagnostic.

### 7.4 Falsification tests [PROPOSED]

- **Placebo**: control terms must leave `margin_A` exactly unchanged.
  `delta_bm25 == 0` already holds at 100.00% post-fix — carry the invariant over.
- **Masking prediction**: for documents already admitted by dense, BM25-side
  injections must have *exactly* zero effect on `S`. Sharp and falsifiable.
- **Monotonicity**: injecting a term present in `d` must weakly increase
  `bm25(d)`. Any violation is a tokeniser bug, not a finding.
- **Sentinel sensitivity**: rerun the intensive decomposition at two values of
  `K`. Anything that moves is a units artefact — `K` sets both the conditioning
  event and the outcome scale (`MISSING_RANK = K+1`).

### 7.5 The claim the paper should make [PROPOSED]

Not *"the reranker gets 81% of the credit."*

Instead: *in a deterministic retrieval SCM, path-specific effects can be
computed exactly rather than estimated; doing so shows credit assignment is
architectural — first-stage effects are negative because raising a document's
first-stage score displaces it by pulling in competitors, and the reranker's
apparent dominance is governed by how often a query edit changes pool
membership, which we measure directly and which varies 6–49% across collections.*

---

## 8. Summary ledger

**Does the C2 failure invalidate the causal framework? No.**

| component | status |
|---|---|
| Identification (determinism → cross-world quantities computed, not estimated) | **intact** |
| Exact additivity of the signed decomposition | **verified**, 0.00e+00 across 63k pairs |
| Two-regime result (credit assignment is architectural) | **replicates**, all 6 collections |
| C1 (intervention effects exist) | **passes**, all 6 collections |
| Per-pair path decomposition | **valid** |
| The marginal `|·|` share as a general claim | **damaged** — aggregation artefact + not a true decomposition |
| External validity of "81%" | **damaged** — conditional on a frame that neuters the mechanism |
| The recall channel | **unmeasured** |
| C3 on non-biomedical collections | **inconclusive by construction** |

The framework is sound. One summary statistic and the generality claim resting
on it are not. Items 1–5 in §7.3 require no new compute — they are re-analyses
of parquets that already exist.

---

## 9. Reproduction

```bash
# fetch the published artefacts (public dataset repo)
python scripts/publish.py --repo ValerianFourel/ragdag-results --download

# recompute C1/C2 for any dataset, including the two with no REPORT.md
python - <<'EOF'
import pandas as pd, interventions as I, mediation as M
inter = pd.read_parquet("<dir>/interventions.parquet")
print(I.cluster_bootstrap_diff(inter, "delta_rank", absolute=True))
print(M.mediation_ratio(pd.read_parquet("<dir>/mediation.parquet")))
EOF

# reproduce the crash
python -c "
import pandas as pd, dml_analysis as D
D.print_comparison(pd.DataFrame())   # safe
D.plot_dml(pd.DataFrame())           # KeyError: 'naive_coef'
"
```

Published dataset: `https://huggingface.co/datasets/ValerianFourel/ragdag-results`
(219 files, 55 MB, public). Directories `_archive_prefix/` and `shards/` are
artefacts of the publishing defect in §6.2 and should not be treated as results.
