# Handover — the admission experiment, both channels, verified at full scale

Branch: `admission-model`, head `801068a`. Written 2026-08-02. Covers the work
from "admission validated on one collection at smoke scale" to "both retrieval
channels measured and verified on six BEIR collections at three pool depths".

Companions: `admission-model.md` (why the lexical module exists),
`session-log.md` (the MVP build and the rank-side results), `audit.md` (the
pre-refactor code audit).

Read this if you are picking the project up. It states what exists, what the
numbers are, what was checked and how, what broke, and what is still open.

---

## 1. What the experiment is, in one page

A three-stage retrieval pipeline — BM25, a dense dual encoder, and a
cross-encoder reranker over the union of their top-K lists — is a
**deterministic function we own**. Counterfactuals are therefore *evaluated*,
not estimated.

The intervention is one word appended to the query, `do(Q := Q ⊕ t)`. The unit
is a triple (query `q`, relevant target document `d`, injected term `t`).
Targets are drawn from the baseline union pool, so every effect measured here
is **displacement or rescue within the pool the pipeline already produces**.

The outcome is **admission** — membership in a channel's top-K — not rank. Two
things move under the intervention and both matter:

* the target's own score (**direct effect**);
* every other document containing `t`, which raises the K-th-largest score, the
  bar the target must clear (**interference**).

Each trial is evaluated in four counterfactual worlds built from the *same*
score deltas, differing only in who receives them:

```
Y00  nobody moves            Y10  target's delta only
Y01  competitors' deltas     Y11  everything (the real intervention)

direct       = Y10 − Y00      interference = Y01 − Y00
total        = Y11 − Y00      interaction  = Y11 − Y10 − Y01 + Y00
```

This within-word decomposition is what makes the two channels comparable, and
it is the only clean decomposition available in the dense channel (see §4).

---

## 2. The formulas as implemented

### BM25 (`admission.py`, k₁ = 1.5, b = 0.75)

The index is **bm25s' `lucene` variant**, which omits the textbook `(k1+1)`
numerator factor:

```
IDF(t)    = log(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
Δs(t, d′) = IDF(t) · tf(t,d′) / ( tf(t,d′) + k₁·(1 − b + b·|d′|/avgdl) )
```

applied to exactly the `df(t)` documents containing `t`, and to no others —
that sparsity is why the whole-corpus response is cheap.

One consequence is load-bearing: the saturation ratio is strictly below 1, so
**Δs < IDF(t)** for every document. A target whose margin above the cut exceeds
`IDF(t)` therefore *cannot* be displaced by any competitor gaining `t`. That is
the immunity certificate, and it is what `scripts/check_admission.py` tests.

Postings and document lengths are built with the **index's own tokenizer**
(`bm25s.tokenize`, 33-word stoplist, Snowball stems). Using the project's
covariate tokenizer instead — sklearn's 318-word stoplist — puts a few-percent
error in every document length and breaks the prediction. This has bitten the
project twice; do not "simplify" it.

### Dense (`pipeline.py`, `dense_admission.py`)

```
e_d(j)  = raw_j / ‖raw_j‖             unit rows, fp32, cached .npz
e_q(q)  = encode(BGE_QUERY_PREFIX + q, normalize=True)
s_j(q)  = e_d(j) · e_q(q)             exhaustive dot product, no ANN
Δq_t    = e_q(q ⊕ t) − e_q(q)         two independent encodes, both normalised
Δs_j(t) = e_d(j) · Δq_t               exact: scoring is linear in q
|Δs_j|  ≤ ‖e_d(j)‖ · ‖Δq‖             Cauchy–Schwarz
```

No encoder additivity is assumed anywhere: `Δq` is a finite difference of two
separately normalised vectors. There is **no closed form for `Δq`** — the
encoder must run — but *given* `Δq` the corpus response is exact linear
algebra, which is enough for the four worlds.

### Tie rules — a deliberate asymmetry

* **Lexical** (`admission.admitted`): `score ≥ K-th largest value`. Ties all
  admit, so the set can exceed K. This is the published stage-7 semantics and
  is locked by a regression test.
* **Dense** (`dense_admission.beat_count`): the *production* rule — j beats the
  target iff `s_j > s_t`, or `s_j == s_t` and `j < target index`; admitted iff
  fewer than K beat it. Exactly K documents, matching
  `pipeline._topk_ids`'s `lexsort((idx, −score))`.

The dense module follows production because it joins to the union pool. Do not
"unify" these without re-running everything.

---

## 3. The code

| file | role |
|---|---|
| `admission.py` | stage 7 — lexical admission: postings, closed form, four worlds, threat count, support/lift coordinates, DoubleML arm contrast |
| `dense_admission.py` | stage 8 — dense admission: fp64 reference scorer, delta audit, four worlds, mechanistic coordinates, immunity certificates, union gate, cluster-bootstrap tables, grouped-CV models |
| `interventions.py` | the do() sampler: support × lift stratification, bin-matched controls, logged `select_prob` |
| `scripts/test_worlds.py` | unit tests for both channels' membership rules |
| `scripts/check_admission.py` | one-command post-run gate; exits non-zero on any failure |
| `scripts/show_results.py` | prints every result table; `--sweep` for the K arms |
| `scripts/admission_k.sbatch` | the job: stages 7+8 at one pinned K |

Artefacts per results directory: `admission_{panel.parquet, audit.json,
by_support.csv, cate.csv, surgical.csv}`, `dense_admission_{panel.parquet,
audit.json, by_bin.csv, models.csv}`, `union_gate.csv`. All match `publish.py`'s
existing include globs.

### Running it

```bash
source scripts/env.sh

# primary arm (K must be pinned - see §6)
sbatch --export=ALL,RAGDAG_WS="$RAGDAG_WS",K_CANDIDATES=50 scripts/admission_k.sbatch

# sensitivity arms, only after the primary arm has finished
sbatch --export=ALL,RAGDAG_WS="$RAGDAG_WS",K_CANDIDATES=20,RESULTS_SUFFIX=_k20  scripts/admission_k.sbatch
sbatch --export=ALL,RAGDAG_WS="$RAGDAG_WS",K_CANDIDATES=100,RESULTS_SUFFIX=_k100 scripts/admission_k.sbatch

# verify, then read
python scripts/check_admission.py          # must print ALL CHECKS PASSED
python scripts/show_results.py --sweep
python scripts/publish.py
```

Cost: one A100, six collections, both stages — under 5 minutes per arm. The
world evaluations run in fp64 on GPU (`torch.topk`) and fall back to numpy
automatically; fp64 makes the GPU path bit-identical to the numpy one, so the
device changes wall-clock and nothing else.

---

## 4. What was verified, and how

Scale: **56,220 trials per K arm** — 6,930 queries, 12,931 (query, document)
pairs, 28,110 treatment / 28,110 control (exactly 1:1), 43,284 baseline-admitted
and 12,936 baseline-excluded. Across three K arms and two channels that is
≈1.35 M counterfactual admission evaluations, plus 7,200 audited injections.

### BM25 closed form vs the live index (200 injections per arm)

| collection | max abs error | tolerance | max score | ulps | agreement |
|---|---|---|---|---|---|
| NFCorpus | 9.54e-07 | 1.25e-05 | 13.12 | 1 | 1.0000 |
| TREC-COVID | 1.91e-06 | 1.37e-05 | 14.32 | 2 | 1.0000 |
| SciDocs | 1.91e-06 | 1.53e-05 | 16.05 | 1 | 1.0000 |
| Quora | 1.91e-06 | 1.73e-05 | 18.15 | 1 | 1.0000 |
| FiQA | 1.91e-06 | 1.74e-05 | 18.23 | 1 | 1.0000 |
| SciFact | 3.81e-06 | 3.16e-05 | 33.12 | 1 | 1.0000 |

Every observed error is exactly **one or two float32 ulps of the largest score
in play** — correct to the last representable bit. The `exact` flag uses a
*relative* tolerance (8 ulps of the max score); an absolute 1e-6 flags ordinary
fp32 rounding on collections with large scores and produced a false alarm once.

### Dense identity, Cauchy–Schwarz, and production replication

| collection | max rel identity error | × fp64 eps | CS violations |
|---|---|---|---|
| NFCorpus | 5.47e-16 | 2.5 | 0 |
| SciFact | 5.85e-16 | 2.6 | 0 |
| SciDocs | 7.08e-16 | 3.2 | 0 |
| Quora | 8.85e-16 | 4.0 | 0 |
| TREC-COVID | 1.14e-15 | 5.1 | 0 |
| FiQA | 1.18e-15 | 5.3 | 0 |

Production replication, all 18 arms: fp64-reference top-K vs the pipeline's own
fp32 path — **exact set match 1.000**, Jaccard 1.000, target-admission
agreement 1.000. No ANN anywhere (exhaustive dot product). Batch-vs-single
encoding drift measured at 1.34e-07 – 2.09e-07 and recorded in the audit JSON,
so fp32 noise cannot be mistaken for a modelling error.

`dense_admission` **aborts the panel** if its audit fails, rather than emitting
numbers that inherit the error.

### Immunity certificate

| K arm | certified immune | violations |
|---|---|---|
| 20 | 4,577 | **0** |
| 50 | 6,526 | **0** |
| 100 | 8,151 | **0** |
| total | **19,254** | **0** |

At-risk documents were ejected at 4–30% over the same runs. A falsifiable
prediction tested 19,254 times without a single failure.

### Unit tests (`scripts/test_worlds.py`, run by the job before touching data)

500 randomised four-world trials against a brute-force sort **with deliberate
ties and negative deltas**; 200 trials confirming torch and numpy agree exactly;
hand-checked cases for `beat_count`'s stale-slot exclusion; the CS bound on
random embeddings; a union-gate truth table; and 300 trials locking
`admission.admitted` to the ≥K-th-value rule so a future edit cannot silently
change published BM25 semantics.

---

## 5. Results

All figures at K=50 unless stated. Lexical tables are point estimates; dense
tables carry query-cluster bootstrap 95% CIs.

### Lexical displacement — one irrelevant word vs an admitted target

Worst support bin per collection, with the discriminating covariate:

| collection | worst interference | median support | mean threat | mean co-treated |
|---|---|---|---|---|
| TREC-COVID | **−0.3040** | 0.245 | 17.8 | 45,621 |
| NFCorpus | **−0.2380** | 0.031 | 20.9 | 115 |
| FiQA | −0.0868 | 0.095 | 6.3 | 5,873 |
| SciDocs | −0.0616 | 0.304 | 6.5 | 7,759 |
| SciFact | −0.0204 | 0.291 | 1.3 | 1,654 |
| Quora | −0.0081 | 0.013 | 0.6 | 7,610 |

**Footprint is not threat.** NFCorpus's worst bin touches 115 documents and
ejects 23.8%; Quora's touches 7,610 and ejects 0.8%. `threat_count` — the
co-treated competitors whose boost overtakes the target's baseline score —
tracks the effect; the raw co-treated count does not.

### Lexical rescue — own-lift admission of an excluded target

`Y10 − Y00`, rarest bin → most common bin:

| collection | rarest | most common |
|---|---|---|
| NFCorpus | **0.948** | 0.317 |
| SciFact | 0.929 | 0.214 |
| Quora | 0.899 | 0.592 |
| SciDocs | 0.863 | 0.271 |
| TREC-COVID | 0.837 | 0.308 |
| FiQA | 0.802 | 0.301 |

Monotone in all six collections, and remarkably tight at the rare end
(0.80–0.95) across corpora that differ by two orders of magnitude in size. Net
rescue (`Y11`) sits below pure rescue everywhere, and the gap widens with
support: the rescuing word energises competitors too, costing up to 21 points
on NFCorpus.

### Dense — the control arm's structural zero does not transfer

`surg_direct`, control arm, dense-admitted, rarest bin:

| collection | dense direct effect | mean target_delta |
|---|---|---|
| TREC-COVID | **−0.4424** | −0.0190 |
| NFCorpus | **−0.3200** | −0.0249 |
| FiQA | −0.1650 | −0.0190 |
| SciDocs | −0.1603 | −0.0124 |
| Quora | −0.0749 | −0.0402 |
| SciFact | −0.0061 | −0.0036 |

In BM25 this quantity is exactly 0 by construction. In embedding space a
lexically absent word still moves the target's own score, mostly downward,
reaching −0.44. **"Control = pure interference" is a fact about BM25's
sparsity, not about retrieval** — which is precisely why the same-word four
worlds were needed. Dense rescue in the excluded stratum is positive and
support-decaying as well (Quora 0.83 → 0.31, NFCorpus 0.46 → 0.42).

### The union gate — channel effects are mostly masked

Treatment arm, dense-admitted stratum:

| collection | BM25 changes | → union changes | masked |
|---|---|---|---|
| Quora | 1,152 | **2** | 99.8% |
| SciDocs | 936 | 37 | 96.0% |
| NFCorpus | 714 | 98 | 86.3% |
| FiQA | 560 | 21 | 96.3% |
| TREC-COVID | 363 | 24 | 93.4% |
| SciFact | 47 | 2 | 95.7% |
| **total** | **3,772** | **184** | **95.1%** |

Symmetrically, **634 of 641 (98.9%)** dense rescues of BM25-admitted targets are
redundant. A hybrid retriever is roughly 20× more robust to query perturbation
than either channel alone. **Any channel-level headline number must be reported
next to its pool-level counterpart**, or the paper overclaims.

### Mechanistic prediction (leave-queries-out CV, out-of-fold)

Rescue-task ROC-AUC — margin alone → + `target_delta` → full model:

| collection | margin | +target_delta | full |
|---|---|---|---|
| Quora | 0.656 | 0.946 | 0.968 |
| NFCorpus | 0.752 | 0.888 | 0.980 |
| FiQA | 0.755 | 0.935 | 0.975 |
| TREC-COVID | 0.761 | 0.925 | 0.986 |
| SciDocs | 0.770 | 0.950 | 0.977 |

Adding **lexical support** to margin moves ejection AUC by −0.015 to +0.001 —
never materially positive. Target movement is the dominant covariate for both
tasks; lexical support carries no information about dense admission. These are
predictive, not causal, statements.

### K sensitivity (same trials, different bar)

| collection | interference 20/50/100 | rescue 20/50/100 |
|---|---|---|
| TREC-COVID | −0.215 / −0.182 / −0.065 | 0.599 / 0.606 / 0.674 |
| NFCorpus | −0.187 / −0.179 / −0.110 | 0.630 / 0.647 / 0.693 |
| FiQA | −0.056 / −0.042 / −0.043 | 0.531 / 0.579 / 0.615 |
| SciDocs | −0.041 / −0.039 / −0.018 | 0.496 / 0.574 / 0.632 |
| SciFact | −0.014 / −0.009 / −0.004 | 0.538 / 0.630 / 0.630 |
| Quora | −0.007 / −0.007 / −0.004 | 0.746 / 0.793 / 0.806 |

Interference weakens monotonically with depth, rescue strengthens, the
collection ordering is preserved at every K, and immunity holds at every K.
Qualitatively invariant, quantitatively K-dependent.

### Where the original theory was wrong

The support-based theory predicted an interference **peak** at intermediate word
frequency (rare words boost hard but touch too few documents; common words
touch everything but boost too weakly). NFCorpus shows that peak; **TREC-COVID
rises monotonically instead**, because its targets sit *on* the cut (median
margin 0.000), so the "too weak to matter" force never engages. Support governs
how many competitors move; **margin decides whether it matters**. Report the
refinement, not the original claim.

---

## 6. Incidents, and the guards that now prevent them

Every one of these produced plausible output rather than an error. The guards
matter more than the fixes.

| incident | cause | guard now in place |
|---|---|---|
| 972 NFCorpus trials appeared to be outside *both* channels' top-K — impossible, since targets come from the union pool | a cancelled CPU job rewrote stage-7 panels at **K=20** (the default is 50 on GPU, 20 on CPU) while stage 8 ran at K=50; the join mixed them | both audits stamp `k_candidates`; the union join **refuses** on mismatch; `scripts/admission_k.sbatch` refuses to start without an explicit `K_CANDIDATES` |
| a K-sweep would have overwritten the primary results | K-variant runs shared the per-dataset directory | `RESULTS_SUFFIX` scopes them (`results/<tag>_k20/…`); `publish.py` picks them up as separate datasets |
| a job "ran" but refreshed nothing | HoreKa exports `DATASETS=/hkfs/home/dataset/datasets` site-wide, so `${DATASETS:-<six collections>}` never fell back | the loop variable is `RAGDAG_DATASETS`, and the script rejects anything that is not a dataset id or that resolves to a filesystem path |
| the report crashed on SciDocs and Quora, twice | a zero-concept module 4 wrote a **headerless** CSV, which `pd.read_csv` cannot parse | `analyse()` returns a schema-stable empty frame; both readers catch `EmptyDataError`; the report emits an explicit INCONCLUSIVE section |
| the predictor audit reported `exact: false` on four collections | an absolute 1e-6 tolerance flags ordinary fp32 rounding when scores are large | relative tolerance: 8 float32 ulps of the largest score, with `max_abs_score` and `tolerance` recorded |
| a Quora run was destroyed mid-flight | `mv results …` ran while the job was still writing | none automatable — **never let two jobs write `results/` at once**. Both data-loss incidents in this project came from concurrent writers, not from bad code |

Operational rule of thumb: `squeue --me` must be empty before you submit
anything that writes to `results/`, and every submission pins `K_CANDIDATES`.

---

## 7. Open items

**A known defect, small but real.** `admission.py` computes the base query's
BM25 scores from `bm25_terms(q0)`, which returns a **set** — deduplicated —
while production `_bm25_array` scores the raw **token stream**. A query with a
repeated stem therefore gets a slightly different score vector, moving
borderline documents across the K line. This is the residual 0.02–1% of trials
that `check_admission.py` still flags on FiQA, Quora, SciDocs and TREC-COVID at
K=50; NFCorpus and SciFact (no repeated stems) show exactly 0. It affects
`base_admitted` labelling only — never a within-trial comparison, since all four
worlds share the same base vector. **Fix before publication**: score the token
stream, then re-run stages 7+8.

**Statistics.** The dense tables carry query-cluster bootstrap 95% CIs; the
lexical tables are point estimates. Attach the same intervals before any
lexical number goes in print. Trials are nested within pairs within queries —
never use trial-level i.i.d. standard errors.

**Scope.** Targets are drawn from the union pool, so everything here is
displacement and rescue *within* documents the pipeline already finds. The
unconditional recall experiment — targeting relevant documents outside the
whole pool — remains the decisive missing piece and requires changing
`select_targets`.

**Dense limits.** No closed form for `Δq` (the encoder must run); the
Cauchy–Schwarz certificate is valid but vacuous at the observed perturbation
sizes (‖Δq‖ ≈ 0.11–0.48, i.e. one appended word rotates the query embedding
substantially), so it certifies nothing in practice; and the OR-gate *path*
decomposition (lexical / dense / masking shares per frozen world) is still not
implemented in `mediation.py`.

**Inherited from the rank-side work.** `residual` is still not renamed
`mediator_interaction`; the rank-side DoubleML is still i.i.d. rather than
cluster-robust; GPU determinism is untested; control terms are not
frequency-matched on the rank side.

---

## 8. What to say, and what not to say

Defensible as written:

* the closed form and the dense identity reproduce the production system to the
  last representable bit (18/18 arms, agreement 1.000);
* a single irrelevant word ejects an already-retrieved document from a
  single channel's top-50 up to 30% of the time, concentrated entirely at the
  boundary;
* a rare, document-characteristic word admits a previously-missed document
  80–95% of the time, decaying monotonically with corpus frequency, on all six
  collections;
* the immunity bound held 19,254 times with zero violations;
* ~95% of channel-level admission changes are masked at the hybrid gate.

Not defensible without more work:

* any unconditional statement about **recall** (scope is the union pool);
* dense **interference** as an identified quantity (control words move the
  target directly — use the four worlds, not the arm contrast);
* cross-K comparisons of effect *magnitudes* as behavioural claims (K is a unit
  as well as a condition);
* the Cauchy–Schwarz certificate as a practical dense safety check (valid,
  vacuous at these ‖Δq‖).
