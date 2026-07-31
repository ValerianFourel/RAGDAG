# Admission model — does the document get retrieved at all?

Branch: `admission-model`. Written 2026-07-31. Status: validated on NFCorpus at
smoke scale, not yet run at full scale on any collection.

Companion to `llm-briefing.md`, which documents why the rank-based analysis
needed this.

---

## 1. Why

Modules 2–3 measure **where a document lands given it is already in the
candidate pool**. That conditioning is what made C2 degenerate: targets are
drawn from `set(run.candidates) & relevant` (`interventions.py:326`), so the
first stages are structurally silent on 77–94% of pairs and the reranker
absorbs the attribution by default.

For RAG the prior question is the one that matters. No amount of reranking
recovers a document the first stage never returned. So the outcome here is
membership, not rank:

```
A = 1{d ∈ TopK(BM25)}          the outcome
S = A ∨ B                      full admission, via the OR of both channels
```

The reranker does not appear. It cannot — it only reorders what the union hands
it. The degeneracy that broke C2 is impossible by construction.

---

## 2. What makes this checkable rather than fitted

Scope is deliberately the **lexical channel**, because BM25 has a closed form:

```
Δscore(t, d) = IDF(t) · tf(t,d) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
```

Appending `t` to a query changes the score of **exactly** the `df(t)` documents
containing `t`, by that amount, and of **no others**. So the whole-corpus
response to a `do()` operation is computable *before running the pipeline*, and
the prediction can be checked against what the pipeline actually did.

That check is `audit_predictor`, and it is the load-bearing test of the module:

```
predictor audit: 100 injections, max |score error| = 9.537e-07,
                 admission agreement = 1.0000
```

Float32 precision. This is arithmetic agreement, not a goodness-of-fit.

**It failed the first time, at 6.77.** Two defects, both caught by the audit:

| defect | consequence |
|---|---|
| bm25s' `lucene` method **omits the `(k1+1)` numerator factor** | constant multiplier, cannot change a ranking, so nobody notices — but scales every score by 2.5 at k1=1.5 |
| `corpus.doc_len` is **not** the BM25 document length | it comes from `content_tokens`, which applies scikit-learn's 318-word stoplist while the index applies bm25s' 33-word one — a few-percent error in every length |

The second is the same defect class as the `having` control-term leak: deciding
term identity or text length with anything other than the index's own
tokeniser. Fixing it also dropped the postings build from 16.6s to 2.3s.

Everything downstream — support, lift, document lengths — is now computed in
**BM25 space**, so the model's coordinates are the scoring function's own
rather than a differently-tokenised proxy.

---

## 3. The causal structure

Admission is **competitive**:

```
A = 1{ s_d > s_(K) }
```

Both sides move under an intervention. `s_d` rises by **lift** (how much *this*
document gains). `s_(K)` — the bar — rises because the `df(t)` co-treated
competitors gain too, which is governed by **support**.

That second term is interference. It is why the target's own score rising does
not guarantee admission, and **no independent-scoring retrieval model predicts
it** — query likelihood, DFR and BM25 all score documents in isolation.

This splits the two arms cleanly, and the split is exact rather than assumed:

- **control** terms are absent from the target by construction, so their direct
  effect on `s_d` is **exactly zero**. Any admission change they cause is
  **pure interference** — competitors displacing the target.
- **treatment** terms move both. The contrast therefore identifies the direct
  effect net of interference, and the control arm alone identifies the
  interference effect.

---

## 4. Results (NFCorpus, 40-query smoke, support/lift sampler)

438 interventions, 30 queries. **Mechanism demonstration, not a result.**

| support | co-treated docs | keep (treat) | keep (control) | interference | direct effect |
|---|---|---|---|---|---|
| 0.4% | 13 | 1.000 | 0.871 | −0.129 | +0.129 |
| 1.2% | 48 | 1.000 | 0.913 | −0.087 | +0.087 |
| 3.2% | 117 | 0.955 | 0.857 | −0.143 | +0.097 |
| **6.7%** | **245** | 0.964 | **0.720** | **−0.280** | **+0.244** |
| 14.8% | 558 | 0.957 | 0.867 | −0.133 | +0.090 |
| 28.4% | 1196 | 0.968 | 0.864 | −0.136 | +0.104 |

**Interference peaks near 5–10% support** — matching the non-monotone churn
the closed form predicted before any of this was run. Two forces oppose:

- rare terms give a huge per-document boost but touch too few documents to fill
  a pool (entrants are bounded by `df(t)`);
- common terms touch everything, but the IDF penalty makes each boost too weak
  to clear the bar.

At the peak, **a single irrelevant word ejects the target from the BM25 top-50
28% of the time**, purely by pulling competitors in.

DoubleML arm contrast, cluster-robust by query:

```
+0.126  [0.045, 0.208]   p = 0.002   n = 318 over 29 queries
```

---

## 5. The positivity trap

The first CATE fit returned `coef = 135.2, se = 608.4`.

That is a **positivity violation, not a numerical accident**. Control lift is
exactly 0 by construction; treatment lift is always > 0. So `lift` predicts the
arm perfectly. The partially linear score divides by `Var(D − E[D|X])`, and a
near-deterministic propensity detonates it.

**Lift *is* the treatment, not a covariate.** `ARM_CONTRAST_X` now excludes it,
and `_overlap_ok` cross-fits the propensity and refuses to report if it exceeds
0.95. Currently 0.593.

Support is excluded from the adjustment set too, for the opposite reason: the
sampler matches the arms bin-for-bin on support, so it is balanced by design.
It belongs on the dose-response axis, not in `X`.

The propensity here is **known** — the sampler assigned the arm 1:1 within each
support bin, so `E[D|X] = 0.5` rather than something to estimate. The machine
learner is responsible only for the outcome regression, which is where it helps
(variance, not bias). Very little applied DML can say that; it is a direct
payoff from having logged `select_prob`.

---

## 6. What this does not yet do

- **Lexical channel only.** The dense channel has no closed form. Its exact
  delta needs one encoder pass per (query, term) plus a matrix-vector product
  against the cached document embeddings — cheap, but not free, and not built.
- **All targets are in-pool**, so this measures **displacement only**.
  `rescued` — a relevant document that was not retrieved becoming retrieved —
  is still structurally impossible until `select_targets` changes. That remains
  the single highest-value edit in the project.
- **The OR-gate path decomposition is not implemented.** `mediation.decompose`
  still records only ranks; it needs membership and admitting channel per
  frozen world to give lexical / dense / masking terms for admission.
- **30 queries.** Nothing here is a finding yet.

---

## 7. Running it

Wired as **stage 7** of `run_all`, in the merge pass (CPU-bound: BM25 plus
order statistics over the whole corpus). It is wrapped in a try/except and
**cannot take down the report** — which is precisely how two collections were
lost in the previous run to an unguarded plotting call.

```bash
# standalone, against any existing interventions.parquet — no GPU needed
python -m admission                     # audit + panel + contrast + CATE
python -m admission --no-cate --limit 500

# as part of a full run
python -m run_all --merge --n-shards 4 --only 7
```

Artefacts, per dataset: `admission_panel.parquet`, `admission_audit.json`,
`admission_by_support.csv`, `admission_cate.csv`. All match `publish.py`'s
existing include globs, so they publish without changes.

**Cost.** Postings build is O(corpus tokens), cached per dataset: 2.3s on
NFCorpus, ~2 min on TREC-COVID. The panel loop is two order statistics per row
over the full corpus, so it scales with rows × N — Quora (59k rows, 523k docs)
is the worst case at roughly 10 minutes. Budget ~20 minutes total across six
collections.

**Always read `admission_audit.json` first.** If `exact` is false, the closed
form disagrees with that collection's index and every number in the panel
inherits the error. The stage prints a warning but does not stop.

---

## 8. Why this is the stronger paper

You have two things almost nobody has together: a probabilistic retrieval model
that predicts admission (query likelihood / DFR — and BM25 already *is* a
support/lift model, `log P(q|d)` being a monotone transform of lift plus a
length term), and **exact counterfactuals** for what the pipeline does.

So you can ask whether the probabilistic model correctly predicts the *causal*
response to a perturbation — and localise exactly where it fails. It will fail
on interference, because independent scoring cannot predict the cut moving when
`df(t)` competitors rise at once. That is a claim the score-distribution
literature structurally cannot make, because it has to fit distributions and
hope.

The remaining unknown is what sets the collection-level churn rate. Normalising
support by `K/N` **failed** to collapse the six collections onto one curve
(SciFact topped out at 9% first-stage activation, TREC-COVID at 51% at the same
normalised support). Boundary crowding is the candidate third input, which is
why `boundary_cut_gap` and `boundary_n_within_5pct_below` are logged per query.
Confirming or refuting that is the first thing the full run can settle.
