# Pre-refactor audit

Written before any WP-0 work, per the brief. Scope: module boundaries, hard-coded
dataset assumptions, determinism gaps, where the freezing logic lives, and what
breaks at 8.8M passages. No code has been changed.

Repo state: `f03a54e`, 4,184 LOC across 8 modules + 3 scripts.

---

## 1. Module boundaries as they stand

| file | LOC | role | coupling |
|---|---|---|---|
| `config.py` | 346 | seeds, device/profile resolution, cache keys, all knobs | imported by everything; module-level side effects |
| `pipeline.py` | 814 | corpus loading, BM25, dense, candidates, rerank, nDCG, baseline | the SCM; everything depends on it |
| `interventions.py` | 522 | term sampling, `do()` loop, cluster bootstrap | depends on `pipeline` |
| `mediation.py` | 363 | the five freeze worlds, path decomposition | depends on `pipeline`, reads `interventions` parquet |
| `dml_analysis.py` | 359 | panel construction, naive OLS, `DoubleMLPLR` | depends on `pipeline` |
| `stability.py` | 478 | query variants, RBO, per-config rankings | depends on `pipeline` |
| `run_all.py` | 373 | worker/merge orchestration, sharding | imports all of the above |
| `report.py` | 535 | REPORT.md, criteria adjudication | reads artifacts only |

**The good news for WP-0.** The stage boundaries the brief asks for already exist
as methods with the right signatures — `bm25_scores`, `dense_scores`,
`candidates`, `rerank`, `run` — and `run()` already accepts a distinct query per
stage. That is the load-bearing abstraction and it survives the refactor intact.

**The bad news.** They are *methods on one 814-line class*, not interfaces.
`RetrievalPipeline` owns model loading, caching, scoring, ranking, covariates and
device policy simultaneously. There is no `FirstStage` or `Reranker` type to
swap. Adding FAISS or a generator means editing that class rather than
implementing an interface.

**No trace object exists.** `PipelineResult` is close — it carries
`bm25_full`, `dense_full`, `candidates`, `provenance`, `reranked`, `ranks` — but
it holds *full-corpus score arrays* rather than references, which is exactly what
cannot survive 8.8M passages (§5). The brief's `PipelineRunner → trace` design is
right; `PipelineResult` should become a trace of node values with lazy/top-K
score access.

---

## 2. Dataset-specific assumptions, hard-coded

Four sites. Two are blocking for multi-dataset work, two are cosmetic.

**Blocking:**

1. **`dml_analysis.MEDICAL_LEXICON`** — a hand-written 132-word biomedical
   vocabulary used to pick DML concept terms. On TREC-DL or Quora this selects
   nothing and Module 4 silently produces an empty concept list. WP-6's
   "200–500 terms stratified by frequency decile" removes the need for it
   entirely; it should be deleted, not ported.

2. **`stability.SYNONYMS`** — 33 hand-written medical/general pairs, plus
   `_IRREGULAR` (4 entries). Off-domain these fire on almost nothing, so the
   synonym variant degenerates to a no-op and RBO ≈ 1.0 by construction rather
   than by stability. **This would read as "the pipeline is stable on SciFact"
   when it actually means "the rewriter did nothing."** The `unchanged_text`
   fraction is already tracked and must become a hard gate, not a printed note.

**Cosmetic but worth noting:**

3. `config.DATASET` is a single string; nothing else is parameterised per
   dataset. Baseline nDCG expectations live in prose in `pipeline.py`, not as
   per-dataset assertions (WP-4 needs these as data).

4. `Corpus.__post_init__` builds `ce_texts` by truncating to
   `CE_MAX_LENGTH * 0.9` **words** — a heuristic tuned for NFCorpus's ~234-word
   abstracts. On MS MARCO passages (~60 words) it is a no-op; on long documents
   it silently changes what the reranker sees.

---

## 3. Determinism: assumed, not enforced

This is the weakest area and the brief is right to lead with it.

**Explicitly disabled.** `config.py:292` sets
`torch.use_deterministic_algorithms(False)`. `CUBLAS_WORKSPACE_CONFIG` is never
set. `torch.backends.cudnn.benchmark` is never pinned. The comment justifying it
("CE inference only; no autograd") is a weaker argument than I thought when I
wrote it — see the next point.

**The real risk is batch composition, and we have it.** `rerank()` scores only
*cache-missing* pairs, so the batch handed to the model depends on cache state,
which depends on execution order. The same `(query, doc)` pair can be scored in a
batch of 64 on one run and a batch of 17 on another. GPU GEMM tiling is
batch-shape dependent, so those are not guaranteed bit-identical, and at a
near-tie a 1e-6 difference flips a rank.

**What has and has not been verified.** Sharded-vs-single-process output was
verified bit-identical for interventions, mediation and stability — **on CPU via
the ONNX path**. The same test has *never been run on GPU*. The published HoreKa
numbers therefore rest on an untested determinism assumption. This is the single
most important gap in the audit.

**Tie-breaking is total, and correctly so.** Three sites, all sound:
`_topk_ids` uses `np.lexsort((idx, -arr[idx]))`; `rerank` sorts on
`(-score, corpus_index)`; `load_corpus_and_queries` sorts query ids. Set and dict
iteration is `sorted()`-wrapped at every site I checked. No stable-sort accidents
found.

**Seeding is sound.** `config.stable_seed()` uses blake2b rather than Python's
salted `hash()` — this was a bug, found and fixed when sharding exposed it.
Per-query RNGs are derived from it, so worker assignment cannot change sampling.

**Provenance is thin.** `run_meta.json` covers 12 config values. It does **not**
cover: git SHA, HF model revision SHAs (models are pinned by *tag*, so
`BAAI/bge-small-en-v1.5` silently changes if the repo is updated), corpus content
hash, BM25 index build params (k1/b/stemmer/stopword list are fixed in code but
unhashed), or the ONNX-vs-torch backend flag. WP-3's content-addressed cache
needs all of these.

**Resumability is shard-level only.** A worker that dies at 90% loses its whole
shard; only the baseline is separately cached. Nothing checkpoints mid-module.

---

## 4. Where the freezing logic lives

Two places, and the split is cleaner than I expected:

- **Mechanism:** `pipeline.RetrievalPipeline.run()` (~line 620). Freezing is
  implemented purely as per-stage query routing — `bm25_query`, `dense_query`,
  `rerank_query`, each defaulting to `query`. There is no special-case code; a
  frozen stage is just a stage handed a different string. This generalises
  cleanly to WP-7's freeze-D / freeze-prompt, because those are also "hand this
  node a different value."

- **Policy:** `mediation.decompose()` (~line 100), which enumerates the five
  worlds and differences them. 40 lines, no model code, already testable in
  isolation given a `run`-like callable.

**A serious problem with the brief's proposed invariant.** WP-0 asks to assert
`|total − Σ paths − residual| < 1e-6`. As currently written that assertion is
**vacuous** — `mediation.py:138` defines

```python
residual = total - (reranker + lexical + dense_)
```

so `total − (reranker + lexical + dense + residual) ≡ 0` identically, for any
inputs whatsoever, including a completely broken freezing implementation. The
test can never fail and therefore tests nothing.

The invariants that *do* have content are the brief's other two, and they should
be the ones encoded:

- freeze-all ⇒ effect exactly 0
- freeze-none ⇒ effect exactly equals the total

Plus a third worth adding: in the `bm25_only` config the residual must be exactly
0 (the dense path is absent by construction, so the decomposition is genuinely
additive). That one *can* fail and would catch a broken freeze. It is currently
verified in `scripts/check.py` but not as a unit test.

---

## 5. What breaks at 8.8M passages

Measured on this machine, MS MARCO passage v1 (8,841,823 docs, 384 dims):

| quantity | value | verdict |
|---|---|---|
| one full-corpus score array | 35.4 MB | — |
| `_dense_cache` at its 8192 limit | **290 GB** | OOM |
| `_bm25_cache` at its 8192 limit | **290 GB** | OOM |
| `doc_emb` resident, fp32 | 13.6 GB | fits A100-40, tight |
| `doc_emb` resident, fp16 | 6.8 GB | fine |
| CE pair cache, 6M entries | 0.90 GB | fine |
| `np.argpartition` over 8.8M | **158 ms/call** | — |
| mediation argpartition calls (~9k pairs × 5 worlds × 2 configs) | 90,000 | **3.9 h of pure argpartition** |

### 5b. Correction after measurement — the memory figure was misleading

The 290 GB above is what the *current cache cap* would try to allocate, not what
the experiment needs. Measured working set during a mediation query group:

```
peak distinct query score-arrays live : 4
current cache cap                     : 8192
```

Four. The five freeze worlds reuse only `q0` and `q1`, for two channels. The 8192
cap was over-provisioning that cost nothing at 3,633 docs (119 MB) and is simply
wrong at 8.8M. Dropping it to 32 gives **2.3 GB** and loses no cache hits.

So memory is a one-line change, not an architecture problem. The real blocker is
**time, and specifically top-K selection on CPU**:

| operation at 8.8M | cost | total over mediation |
|---|---|---|
| `np.argpartition` (CPU) | 164 ms | **4.1 h** ← the actual blocker |
| `torch.topk` (CPU) | 67 ms | 1.7 h |
| `torch.topk` (GPU, est.) | ~3 ms | 4.5 min |
| exhaustive dense GEMV, fp16 on A100 | 4.4 ms/query | 1.3 min* |

\* only ~18k *distinct* queries need a dense array across the whole mediation
(q0 and q1 per pair); the other 72k runs are cache hits.

**Consequence: MS MARCO does not need an ANN index.** Exhaustive fp16 GEMV plus
GPU top-K handles the full corpus in single-digit minutes. `doc_emb` at fp16 is
6.8 GB resident, leaving ~33 GB on an A100-40 for models and activations.

This is also *methodologically preferable*, and it dissolves open decision 2:
no ANN approximation, no index build artifact to hash, no risk of silently mixing
two index builds across runs, and "same query → same output" stays trivially
true rather than true-by-pinned-seed.

**Revised breakage list:**

1. **Cache cap** — `8192` → ~32. One line. (was: "must become top-K caching")
2. **`_topk_ids`** — must select on GPU via `torch.topk`, not `np.argpartition`.
   This is the only genuine hot-path rewrite.
3. **`dense_scores()` returning a full dict** — builds an 8.8M-entry Python dict;
   minutes and gigabytes per call. Public API only, not the hot path, but it must
   become top-K or be deleted.
4. **`bm25_full` / `dense_full` on `PipelineResult`** — holds two full-corpus
   arrays per result object. Fine while one result is live; fatal if any code
   retains a list of them. Becomes top-K in the `Trace` refactor.

Not needed, contrary to the original WP-3 framing: FAISS, HNSW, and the ANN
build-config hashing that came with them.

**Not a problem:** the CE pair cache, the BM25 index itself (`bm25s` is sparse),
the corpus pickle (would need streaming, but that is routine), and the sharding
design, which scales unchanged.

---

## 6. Findings the brief did not ask for but should know

**The estimand is conditioned in code, silently.** `interventions.select_targets`
picks targets from `set(run.candidates) & relevant` — so every published number
is conditional on *the target already being retrieved*. This is exactly the
reordering-vs-admission split WP-5 addresses, but right now that conditioning
event appears nowhere in the report's tables. `docs/estimand.md` (WP-2) is more
urgent than its position in the ordering suggests.

**Two term-sampling bugs were found and fixed during the MVP run**, both from
approximating BM25's tokenisation instead of using it:
surface-form matching let `model`/`models` through (2.7% of control terms);
stem matching against sklearn's 318-word stopword list still let `having`/`have`
through (1 in 4,500), because bm25s strips only 33 words. Absence is now decided
with the BM25 tokeniser itself. WP-1's matching must reuse that same predicate or
it will reintroduce the class of bug.

**Known open flaw, unfixed:** control terms are drawn uniformly and are
consequently *rarer* than the TF-IDF-weighted treatment terms (median df 0.36% vs
1.27%, Mann-Whitney p≈3e-56). This is WP-1 and it should inflate the current
+6.97 contrast.

**`report.py` hard-codes prose that can contradict the run.** Fixed once already
(a limitations line claimed CPU-profile reductions in a full-design 4×A100
report). Any text asserting a fact about the configuration must be derived from
config, not written inline. Worth a lint rule in WP-9.

---

## 7. Proposed WP-0 sequencing (for approval, not yet started)

1. Extract `Corpus`/`FirstStage`/`Reranker`/`Intervention`/`FreezeSpec` as
   protocols; keep `RetrievalPipeline` as the first `FirstStage`+`Reranker`
   implementation so nothing moves yet.
2. Convert `PipelineResult` → `Trace` with **lazy top-K** score access rather
   than full arrays (this is the change that unblocks §5, so it belongs in WP-0
   even though it looks like WP-3).
3. Move `decompose()` to operate on traces, not on a pipeline handle.
4. Golden test pinning the current NFCorpus headline numbers.
5. Pydantic config + `configs/nfcorpus.yaml`.

Determinism enforcement (`use_deterministic_algorithms(True)`, CUBLAS config,
GPU bit-identity test) I would like to do **inside WP-0 rather than later** —
if it turns out GPU runs are not reproducible, that invalidates the golden test
this whole refactor is validated against, and I would rather find out on day one.

---

## 8. Open decisions — I am not guessing on these

1. **Generator model (WP-7).** Determinism and prompt headroom over quality, per
   the brief. My proposal: Qwen2.5-7B-Instruct at greedy/bs=1 via HF `generate`,
   with Qwen2.5-3B for iteration. Needs your call before I pin a revision.

2. **MS MARCO dense index.** `IndexFlatIP` at fp16 is 6.8 GB and fits one A100
   with room for the model — exact, no ANN approximation, no build-config in the
   provenance hash. HNSW would be faster per query but introduces an
   approximation *and* a build artifact that must be hashed and can never be
   silently rebuilt. Given the measured budget I lean **flat/exact**, accepting
   slower queries, because it keeps "same query → same output" trivially true.
   Your call.

3. **top-50 per channel at MS MARCO scale.** The brief correctly flags this as
   estimand-changing, not a free parameter. At 8.8M docs, top-50 is a
   ~5.7e-6 slice versus 1.4e-2 at NFCorpus — four orders of magnitude
   different as a *conditioning event*, which makes cross-dataset comparison of
   the reordering regime hard to defend. Options: fix K, fix the recall level, or
   fix K/|corpus|. I have a recommendation but want to discuss it rather than
   pick.

4. **DML vocabulary sample.** Proposal: stratify by corpus-frequency decile
   within the `[0.5%, 40%]` band, sample n per decile with a fixed seed, exclude
   terms appearing in fewer than X documents for overlap, pre-register the scheme
   in `docs/predictions.md`. Needs sign-off on the band and n.

5. **Determinism escalation.** If the GPU bit-identity test fails, the options
   are (a) force bs=1 for the cross-encoder, costing throughput, (b) pin batch
   composition independently of cache state, or (c) weaken the exactness claim to
   a tolerance. These have very different costs and (c) changes what the paper
   can claim. I will surface the measurement before choosing.

---

## 9. One dataset at a time — footprint

Per-dataset embedding cache, fp16, 384 dims:

| dataset | docs | embedding cache |
|---|---|---|
| NFCorpus | 3,633 | 3 MB |
| SciFact | 5,183 | 4 MB |
| SciDocs | 25,657 | 20 MB |
| BEIR-Quora | 522,931 | 402 MB |
| MS MARCO passage | 8,841,823 | 6.8 GB |
| HotpotQA | 5,233,329 | 4.0 GB |
| **all cached simultaneously** | | **11.2 GB** |

TREC-DL 2019/2020 reuse the MS MARCO corpus, so they cost no extra embedding.

Nothing here forces one-at-a-time on resource grounds — 11.2 GB of workspace is
cheap and only one dataset's embeddings are ever *resident* anyway. The reasons
to do it sequentially are operational, and they are good ones:

- one config hash per run, so provenance stays unambiguous;
- a failed dataset does not poison a multi-dataset artifact set;
- incremental, reviewable progress (NFCorpus golden → SciFact falsification →
  the rest), with a checkpoint after each;
- GPU memory stays predictable: one `doc_emb` resident per worker, 6.8 GB worst
  case, leaving ~33 GB on an A100-40.

Recommended order, cheapest-and-most-informative first: **NFCorpus** (golden,
already passing) → **SciFact** (falsification test; 4 MB, minutes) → **TREC-DL**
(needs the MS MARCO corpus, so it forces the scale work) → **MS MARCO dev** →
**Quora** → **HotpotQA** (WP-7 only).

Putting SciFact second is deliberate: it is the cheapest dataset in the set *and*
the one that can falsify the architectural claim. If the mediation shares do not
shift there, that is worth knowing before building any scale infrastructure at
all.
