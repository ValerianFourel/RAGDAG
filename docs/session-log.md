# Session log — MVP build, HoreKa runs, and what we learned

Written 2026-07-31. Covers the work from an empty directory to a six-dataset
multi-GPU run with published provenance. Companion to `audit.md`, which is the
pre-refactor code audit.

---

## 1. What exists now

A self-contained experiment that treats a three-stage retrieval pipeline as a
structural causal model and measures four things on BEIR collections.

```
Q ──► M1 = BM25(Q)     ┐
  └─► M2 = Dense(Q)    ├─► C = TopK(M1) ∪ TopK(M2) ──► R = CE(Q,C) ──► Y = rank(d)
                       ┘
```

| module | file | what it does |
|---|---|---|
| 1 | `pipeline.py` | the SCM; stages independently overridable, `run()` takes a different query per stage |
| 2 | `interventions.py` | `do(Q := Q + t)` with a matched control arm, cluster-bootstrap CIs |
| 3 | `mediation.py` | five counterfactual worlds per treated pair → path decomposition |
| 4 | `dml_analysis.py` | naive OLS vs `DoubleMLPLR` with LightGBM nuisances |
| 5 | `stability.py` | RBO@10 under meaning-preserving query rewrites |
| 6 | `run_all.py`, `report.py` | orchestration, sharding, mechanical PASS/FAIL verdict |

Supporting: `scripts/{prefetch,check,publish}.py`, `scripts/{env.sh,setup_login.sh,run_multigpu.sh,horeka.sbatch}`.

The methodological core is that the pipeline is a **deterministic function we
own**, so the cross-world quantity `Y(q1, M(q0))` can be *evaluated* by feeding
one stage the old query and the rest the new one. No sequential-ignorability
assumption to defend.

---

## 2. Results

### Baseline quality (323 queries, GPU profile K=50 / CE-512)

| | nDCG@10 |
|---|---|
| BM25 only | 0.3233 |
| Dense only | 0.3455 |
| Full pipeline | 0.3568 |

Close to published BEIR figures; full beats both channels. This is a gate, not a
finding — a mis-wired reranker would make every causal number downstream an
artefact.

### The headline: credit assignment is architectural

NFCorpus, 323 queries, 9,024 interventions over 1,504 (query, document) pairs:

| config | reranker | lexical | dense | interaction |
|---|---|---|---|---|
| union (BM25 + dense) | **81.0%** | 6.2% | 7.8% | 5.0% |
| BM25-only first stage | 39.5% | **60.5%** | — | — |

Signed effects in the union config: reranker **+12.75**, lexical **−0.89**,
dense **−1.08**, interaction **+0.75**.

The first stages have *negative* effects. Mechanism: they decide membership in a
fixed-size pool, and the target is already in it — so raising its BM25 score
buys it nothing while pulling competitors in to displace it. Only the reranker
rescores it directly. Change the first stage and the attribution inverts.

**Internal check:** `12.7549 − 0.8898 − 1.0791 + 0.7451 = 11.5304`, exactly the
measured treatment Δrank. The decomposition reconstructs the total effect rather
than approximating it.

### Six collections

| dataset | queries | C1 contrast | largest share | C2 | C3 | verdict |
|---|---|---|---|---|---|---|
| nfcorpus | 323 | +6.979 | 81.0% | PASS | 3/3 | YES |
| trec-covid | 50 | +6.092 | 75.4% | PASS | 2/3 | YES |
| fiqa | 648 | +3.967 | 84.2% | PASS | 1/1 | YES |
| scifact | 300 | +1.222 | **92.8%** | **FAIL** | 2/3 | MIXED |
| scidocs | 1,000 | — | — | — | — | no report produced |
| quora | 10,000 | — | — | — | — | no report produced |

**The share is not invariant.** Across four collections it spans 75–93%. On the
CPU profile (K=20) SciFact gave 81.9%; at the full design (K=50) it gives 92.8%
and fails the degeneracy criterion. An earlier claim in this session that SciFact
"replicates NFCorpus to within one point" was true only at *matched* settings and
does not survive the change in pool depth.

### DoubleML — naive attribution is inflated ~5×

| concept | naive | DML-adjusted |
|---|---|---|
| risk | +0.4386 [+0.250, +0.627] | +0.0906 [+0.024, +0.157] |
| dietary | +0.4646 [+0.242, +0.687] | +0.0851 [+0.015, +0.155] |
| diet | +0.4442 [+0.199, +0.689] | **−0.0114 [−0.082, +0.059]** |

All three naive intervals *exclude* zero, so a naive analyst would confidently
report an effect. After adjustment two shrink by ~80% and one vanishes. Three
near-identical naive estimates across unrelated words is the tell: it is one
shared confounder — document length — being credited to whatever term is tested.

### Stability — the dense encoder is the instability

RBO@10 under meaning-preserving rewrites (50 queries):

| | BM25 | dense | full |
|---|---|---|---|
| stopword removal | 0.948 | 0.899 | 0.922 |
| word-order shuffle | **1.000** | **0.838** | 0.872 |
| synonym | 0.951 | 0.941 | 0.943 |

Ordering is consistent: **BM25 > full > dense**. The cross-encoder *partly
repairs* dense's instability, so reranking buys robustness and not only accuracy
— a claim that requires the decomposition to make.

### SciFact: the reranker actively hurts

Full pipeline 0.6876 vs dense 0.7127 at CE-512 (0.6695 at CE-192, so longer
sequences helped but not enough). ms-marco-MiniLM is trained on web-style
queries; SciFact queries are scientific claims. The mediation shares there
describe causal responsibility for an intervention's effect **on a pipeline whose
reranking stage is not beneficial**. `config.KNOWN_RERANKER_HARMFUL` records
this and the report emits a blockquote so the two cannot be conflated.

---

## 3. Defects found and fixed

Listed because the failure *mode* matters more than the fix: almost all of these
produced plausible output rather than an error.

### Would have corrupted results silently

| # | defect | consequence |
|---|---|---|
| 1 | Per-query RNGs seeded from Python's `hash()` | salted per process — every worker and every rerun sampled different terms while claiming a fixed seed. Found when sharding exposed it. Now blake2b. |
| 2 | Control terms rejected by **surface form** | BM25 indexes stems, so `model`/`models`, `fruit`/`fruits` leaked. **2.7% of control terms were weak treatments.** |
| 3 | Control terms rejected by **stem + sklearn stopwords** | BM25 strips 33 words, sklearn 318. `having`/`have` leaked, 1 in 4,500. Fixed by deciding absence with the BM25 tokeniser itself. |
| 4 | `run_meta.json` recorded config but not code | a corrected sampler left the fingerprint identical, so `run_all` would have reused the pre-fix parquet and reported it as fresh. Now a source fingerprint. |
| 5 | `RESULTS_DIR` not dataset-scoped | running SciFact **destroyed** the NFCorpus results rather than sitting beside them. |
| 6 | Corpus cache not versioned | adding `titles` still loaded old pickles via a `.get()` fallback with every title empty — blanking `term_in_title` and every origin-document title. |
| 7 | Cache keys not dataset-scoped | `load_corpus_and_queries()` checked the cache *before* reading `DATASET`, so switching collections returned the previous corpus **and** its queries. |
| 8 | Synonym variant produced nonsense | `ECMO→ecmos`, `Ornish→ornishes`, `deafness→deafnesses`. Not meaning-preserving, so "instability" was partly the rewriter's own errors. Number flips are now validated against the corpus vocabulary. |
| 9 | Targets drawn from the reranked top-k | conditions the experiment on retrieval success and mechanically inflates the reranker's share — the quantity under study. Now drawn from the whole pool, with censoring flagged. |
| 10 | Preflight under-powered | scored ~54 control terms on 12 queries; a 1-in-4,500 leak passes that ~99.7% of the time. It reported a false all-clear on defect 3. Replaced with set intersection over the real tokenisation, swept across all queries. |
| 11 | Report hardcoded prose | claimed "reduced to fit a CPU-only machine" inside a full-design 4×A100 report. Now profile-aware. |
| 12 | Recorded nDCG ranges applied to smoke runs | a 10-query NFCorpus sample lands at 0.31 by sampling alone and tripped a false regression alarm. Ranges now gate full runs only. |

### Environment and tooling

| defect | fix |
|---|---|
| Pickled dataclasses bound to `__main__` | caches unloadable from any other module; serialise primitives |
| torch using 4 of 8 threads | explicit `set_num_threads` |
| matplotlib 3.11 removed `boxplot(labels=)` | `tick_labels=` |
| sklearn ≥1.9 removed `check_X_y(force_all_finite=)` | pinned `scikit-learn>=1.4,<1.8`; doubleml 0.10.1 needs it |
| `module load devel/python/3.11` does not exist on HoreKa | fell back to system Python 3.9, where pip aborted the whole requirements file at the first `>=3.10` pin and installed **nothing** — surfacing as a dozen unrelated `ModuleNotFoundError`s. `config.py` now refuses to import below 3.10; `setup_login.sh` fetches CPython 3.11 via uv |
| `pip install Stemmer` | installs an unrelated stub; the package is **PyStemmer** |
| `env.sh` forced `HF_HUB_OFFLINE=1` unconditionally | broke `huggingface-cli login` on the login node. Offline is a compute-node property; now keyed on `SLURM_JOB_ID` |
| `nvidia-smi` ignores `CUDA_VISIBLE_DEVICES` | would spawn workers on GPUs belonging to another job; launcher now respects the allocation |

---

## 4. Compute

| | |
|---|---|
| CPU, before tuning | 21.5 s/query |
| CPU, after thread + sequence-length + ONNX tuning | 2.73 s/query (**8×**) |
| 4×A100, 323 queries, full design | **~9 minutes** |
| cross-encoder pairs in that run | 1.72 M |
| throughput | 798 pairs/s/GPU = 44% of A100 fp32 peak |
| same work on CPU | ~24 hours |

Speed comes from three places: GPU vs CPU on the cross-encoder (~40×), 4-way
query sharding (4×), and the pair cache (4.4× on mediation — the five freeze
worlds share most `(query, document)` pairs, measured 77% hit rate).

**Sharding is by query, not by tensor.** A rerank batch is ~85 pairs through a
22M-parameter model — far too small to split four ways — and stages within a
query are serial. Sharded output was verified **bit-identical** to a
single-process run.

**MS MARCO is the one blocker.** At 8.8M documents `np.argpartition` costs
280 ms/call → **420 minutes** of pure top-K selection. The audit corrected an
earlier overstatement here: memory is *not* the problem (the score-cache working
set is 4 arrays, not the 8,192 the cap allowed), and FAISS is unnecessary —
exhaustive fp16 GEMV is 4.4 ms/query and only ~18k distinct queries need one.
The single genuine rewrite is `_topk_ids` → `torch.topk` on GPU.

---

## 5. Provenance

Every `do()` operation is recorded with enough context to audit it — 38 columns
per intervention, including `term_tf_in_doc`, `term_df_corpus`, `term_idf`,
`term_tfidf_weight`, **`select_prob`** (the exact probability the word was drawn
with), `n_candidate_terms`, `term_in_title` and `term_bm25_form`. Plus
`origin_documents.parquet` giving the title, snippet and length of every source
document.

`select_prob` is the load-bearing one: it distinguishes "we injected the
document's most characteristic word" from "we injected a marginal one", which is
the first thing a reviewer will ask.

`scripts/publish.py` pushes `results/<dataset-tag>/` to a private HuggingFace
dataset repo with a `MANIFEST.json` per dataset pinning git SHA, dirty flag,
source fingerprint and per-file checksums. Artefacts with different
`code_fingerprint` values are **not comparable** — the sampler was corrected
twice while config stayed fixed.

---

## 6. Literature findings (four research agents)

### DML sensitivity — our adjusted estimates are fragile

Derived and numerically verified: **RV ≈ |t|/√n**. Applied to our results:

| concept | RV | vs DoubleML's 3% default |
|---|---|---|
| risk | 1.51% | below |
| dietary | 1.34% | below |
| diet | 0.18% | below |

So: **"unadjusted attribution is biased" is defensible; "the adjusted +0.09 is
the true effect" is not.** Saying so strengthens the paper. Also `RV ∝ √(p(1−p))`
— rare terms get low robustness values for arithmetic reasons, so never pool
across frequency deciles.

### DML clustering — SEs are understated

~34–90 documents share each query. Measured on our own panel: SEs understated
**1.55–1.81×**, score ICC 0.033–0.051. A synthetic replica of our design showed
the "naive falls outside the DML CI" test firing on 48.7% of *null* datasets with
i.i.d. SEs versus 5.3% cluster-robust.

**C3 survives on real data** (2/3 concepts still outside the clustered interval),
but this must be refit with `DoubleMLClusterData` — and the GPU profile's larger
pools will make the understatement worse. Note doubleml 0.10.1 needs
`DoubleMLClusterData`; `DoubleMLData(cluster_cols=…)` only exists from 0.11.0.

A specific leakage channel: `query_len` is constant within a query, so with
i.i.d. folds LightGBM can fit an effective per-query intercept. This
*manufactures* precision — DML's i.i.d. SE came out smaller than OLS's — and
hides itself, since measured ICC was 8× lower on leaky folds.

### Mediation terminology — `residual` is misnamed

It is exactly `−[r(1,1,1) − r(1,0,1) − r(0,1,1) + r(0,0,1)]`, a 2×2 additive
interaction contrast between the lexical and dense channels. The literature name
is **mediator–mediator interaction** (VanderWeele & Vansteelandt). "Residual"
reads as regression noise; "synergy" would be worse, since it means
*sufficient-cause synergism* — a different concept.

Correct names: `reranker` = natural (pure) direct effect; `first_stage_total` =
natural (pure) indirect effect. **No recanting-witness problem** (verified two
ways — the union-and-truncate collider does not create a witness because C's
parents are mediators, not treatment copies).

**Positioning correction:** mechanistic interpretability made the determinism
argument first (Vig et al. 2020), and its "treeification" is Robins &
Richardson's edge-expanded graph rediscovered. The novelty is the application to
retrieval and the two-regime result, not the observation about determinism.

### IR terminology — a citation landmine

**Zobel (1998) and Voorhees (2000) are *reassuring* papers.** Both conclude
evaluation is stable despite judgment differences. Citing either as evidence that
shallow qrels break perturbation evaluation is a misreading a reviewer will
catch. Use BEIR's Hole@10 and Craswell et al. instead.

Corrected lineage: `judged@k` is MacAvaney, Soldaini & Goharian (ECIR 2020);
"hole rate" was coined in ANCE (ICLR 2021); BEIR popularised `Hole@10` citing
ANCE. Nothing in the TREC overviews. Also: `trec_eval` ships `-m unj`, an
unjudged-at-cutoff "red flag" measure — usable instead of an ad-hoc script.

---

## 7. Open decisions and known gaps

**Not done, and flagged as such:**

- **GPU determinism untested.** `use_deterministic_algorithms(False)` is set, and
  cross-encoder batch composition depends on cache state. The bit-identical
  sharded-vs-single test was CPU/ONNX. Two GPU runs *did* produce identical DML
  and stability numbers, which is encouraging but not conclusive (same node).
- **Control terms are not frequency-matched.** Median df 0.36% vs treatment's
  1.27% (p≈3e-56). Rarer words perturb encoders more regardless of relevance,
  which likely inflates the C1 contrast.
- **`residual` not yet renamed** to `mediator_interaction`; `INT_med` not yet
  reported although the worlds for it are computed.
- **DML not yet refit cluster-robust.**
- **scidocs and quora produced no report** in the six-dataset run; cause not yet
  diagnosed.
- The mediation-novelty literature search was budget-truncated and needs
  redoing before any novelty claim appears in print.

**The scope limit that matters most.** Targets are relevant documents *already in
the pool*, so everything measured is **reordering, not recall**. That is exactly
why the reranker dominates, and it leaves a competing explanation open: the ~81%
may be partly mechanical, a consequence of conditioning on pool membership rather
than a fact about these models. Replication across collections is consistent with
both readings. **Only the recall-side experiment — targeting relevant documents
outside the pool — distinguishes them**, which makes it decisive rather than
merely valuable.

**Pool depth is not a free parameter.** `K` sets both the conditioning event and
the outcome scale (`MISSING_RANK = K + 1`), so a deeper pool mechanically
inflates the rank-magnitude of first-stage effects. Cross-`K` comparisons
conflate a behavioural shift with a change of units. This is why SciFact moved
from 81.9% to 92.8%.
