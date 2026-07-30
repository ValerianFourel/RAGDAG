# RAGDAG

*Causal retrieval MVP — treating a RAG retrieval pipeline as a DAG you can intervene on.*

Can a retrieval pipeline be explained **causally**? When a document comes back,
what caused it — the lexical stage, the dense stage, or the reranker?

This is a self-contained research experiment on BEIR NFCorpus that treats a
three-stage retrieval pipeline as a structural causal model and measures four
things on it:

1. **do()-interventions** on the query (term injection, with a matched control arm)
2. **Exact path-specific effects** by freezing pipeline stages — computed, not estimated
3. A **DoubleML** check on whether naive concept attribution is confounded
4. **Counterfactual stability** under meaning-preserving query rewrites

## The SCM

```
Q ──► M1 = BM25(Q)     ┐
  └─► M2 = Dense(Q)    ├─► C = TopK(M1) ∪ TopK(M2) ──► R = CE(Q, C) ──► Y = rank(d)
                       ┘
```

The central move: because the pipeline is a deterministic function we own,
the cross-world quantity `Y(Q1, M(Q0))` — the outcome under the treated query
with a mediator held at its baseline value — can be **evaluated directly** by
re-running the pipeline with that stage fed the old query. Ordinary causal
mediation must assume sequential ignorability to identify this. Here there is
nothing to assume and nothing to estimate.

`RetrievalPipeline.run()` accepts a separate query for each stage, which is
what makes the freezing exact:

```python
pipe.run(q1, bm25_query=q0, dense_query=q0, rerank_query=q1)  # reranker path only
pipe.run(q1, bm25_query=q1, dense_query=q1, rerank_query=q0)  # first-stage path only
pipe.run(q1, bm25_query=q1, dense_query=q0, rerank_query=q1)  # lexical + reranker
```

## Setup

### A100 / CUDA (recommended)

```bash
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install torch --index-url https://download.pytorch.org/whl/cu124
VIRTUAL_ENV=.venv uv pip install -r requirements-gpu.txt
```

The code detects CUDA at import and switches to a GPU profile that runs the
**design document's settings verbatim** — `K_CANDIDATES=50`,
`MAX_TARGET_DOCS_PER_QUERY=10`, `CE_MAX_LENGTH=512`, batch size 256. No flags
needed. Check what it resolved before committing to a long run:

```bash
python -c "import config; print(config.device_banner()); print(config.summary())"
```

### HoreKa (KIT)

HoreKa's `accelerated` partition gives 4× A100-40 per node. The one thing that
will bite you: **compute nodes have no outbound internet**, so models and data
must be fetched on a login node first and the job must run offline.

```bash
# ── login node ────────────────────────────────────────────────────────────
ws_allocate ragdag 30                       # workspace; $HOME is quota'd
export RAGDAG_WS=$(ws_find ragdag)
cd "$RAGDAG_WS" && git clone https://github.com/ValerianFourel/RAGDAG.git && cd RAGDAG

module avail python 2>&1 | head            # find the Python 3.11+ module
module load devel/python/3.11              # adjust to what's actually there
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt

export HF_HOME="$RAGDAG_WS/hf" IR_DATASETS_HOME="$RAGDAG_WS/ir_datasets"
python scripts/prefetch.py                 # ~1 min; downloads models + NFCorpus

# ── submit ────────────────────────────────────────────────────────────────
sbatch --export=ALL,RAGDAG_WS="$RAGDAG_WS",N_QUERIES=30 \
       --partition=dev_accelerated --time=00:30:00 scripts/horeka.sbatch   # smoke
sbatch --export=ALL,RAGDAG_WS="$RAGDAG_WS" scripts/horeka.sbatch           # full
```

Then `squeue --me`, and read the verdict from the end of the job log or
`results/REPORT.md`.

`scripts/prefetch.py` populates `$HF_HOME`, `$IR_DATASETS_HOME`, `cache/corpus.pkl`
and `cache/bm25_index/` — about 110 MB. It deliberately skips the document
embeddings: those are keyed on the dense sequence length, which differs between
the CPU and GPU profiles, and they take seconds on an A100 against minutes on a
shared login node. The job builds them.

`scripts/horeka.sbatch` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so an
air-gapped node fails loudly instead of hanging on a socket. Verified: every
resource loads from cache with both flags set.

Check `--gres=gpu:4` and the module name against your allocation before the full
run — `scontrol show partition accelerated` and `module avail` are authoritative,
not this README.

### Multi-GPU (4-GPU node)

```bash
./scripts/run_multigpu.sh          # one worker per visible GPU, then merge
./scripts/run_multigpu.sh 4        # force 4 workers
sbatch scripts/run_a100.sbatch     # SLURM, --gres=gpu:a100:4
```

**Sharding is by query, not by tensor.** The GPU-bound stages — baseline,
interventions, mediation, stability — are independent per query, so each worker
takes a strided slice (`qids[i::N]`) of the 323 queries, owns one GPU, and
writes `results/shards/*_<i>of<N>.parquet`. A merge pass concatenates them,
runs the global steps, and writes the report.

DataParallel would be the wrong tool: a rerank batch is ~85 pairs through a
22M-parameter model, far too small to split four ways profitably, and the
stages within a query are inherently serial. Query sharding scales close to
linearly instead. Striding rather than contiguous blocks matters for balance —
per-query cost varies with how many judged-relevant documents land in the pool.

What does *not* shard: the **DoubleML stage needs the full (query, document)
panel**, so it runs once in the merge pass, on CPU via LightGBM. In practice
this costs about a minute, so it is not a bottleneck — the merge pass is
dominated by nothing much and the GPU stages set the wall clock. Budget roughly
**15–25 minutes** on 4×A100 against ~1 hour on one.

`DML_N_JOBS` defaults to **1** on purpose. Raising it deadlocked at 0% CPU
during development: torch and LightGBM each load an OpenMP runtime, and two in
one process can hang on thread-pool init. `deterministic=True` makes the
results thread-count independent anyway, so extra threads buy only speed on a
stage that already takes a minute. If you do raise it and see a 0%-CPU stall,
that is this issue, not slowness.

Correctness guarantees, both verified in development:

- Sharded output is **bit-identical** to a single-process run. Per-query RNGs
  are seeded with a stable blake2b digest, not Python's `hash()`, which is
  salted per process and would otherwise make every worker — and every rerun —
  sample different injection terms.
- A missing or failed shard makes the merge **abort loudly** rather than
  aggregate a partial result set, which would look like a completed run over
  fewer queries.

The launcher warms the shared read-only caches (corpus, BM25 index, document
embeddings) in one process before forking workers, so N processes don't race to
build and write the same files.

Manual equivalent, if you'd rather not use the script:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -m run_all --shard $i --n-shards 4 &
done; wait
python -m run_all --merge --n-shards 4
```

### Single GPU

`python -m run_all` with no flags does the whole thing in one process. Expect
roughly **1 hour** end to end. Both models are tiny (bge-small 33M params,
MiniLM-L-6 22M); the cost is the *number* of sequential cross-encoder calls.

### CPU fallback

```bash
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

`requirements.txt` carries macOS-Intel-specific pins (see *Deviations*). On CPU
the profile automatically shrinks the candidate pool and target count to keep
the run tractable, at the cost of a narrower measurable range.

## Run

```bash
# Smoke test the whole chain
N_QUERIES=30 python -m run_all

# Full run, all 323 queries
python -m run_all

# Individual modules
python -m pipeline        # baseline + nDCG sanity check
python -m interventions
python -m mediation
python -m dml_analysis
python -m stability
python -m report
```

### Environment overrides

| variable | effect |
|---|---|
| `DEVICE` | `cuda`, `cuda:1`, `cpu`, `mps`. Default: auto-detect. |
| `K_CANDIDATES` | Top-K per first-stage channel. Default 50 (GPU) / 20 (CPU). |
| `MAX_TARGET_DOCS` | Target documents per query. Default 10 (GPU) / 3 (CPU). |
| `CE_MAX_LENGTH` | Cross-encoder sequence length. Default 512 (GPU) / 192 (CPU). |
| `CE_BATCH_SIZE` | Default 256 (GPU) / 64 (CPU). |
| `N_QUERIES` | Query subset for smoke tests. Default: all 323. |
| `USE_FP16` | Half precision. **Off by default even on A100** — see below. |
| `ALLOW_TF32` | TF32 matmuls on Ampere. Off by default — see below. |

### Precision is deliberately conservative

`USE_FP16` and `ALLOW_TF32` are both **off by default on GPU**. The outcome
variable of this experiment is a *rank change*, so any numerical perturbation
can manufacture or erase an effect at a near-tie. int8 dynamic quantisation was
measured during development at 9/10 top-10 agreement and rejected on exactly
these grounds; fp16 and TF32 are far gentler but are opt-in rather than
assumed. An A100 completes the full design at fp32 in about an hour, so there
is no throughput argument for spending fidelity here.

Cache keys embed the precision tag and both sequence lengths, so artefacts
computed under one setting are never silently reused under another.

`run_all` executes everything in **one process against one pipeline instance**,
so the cross-encoder pair cache — keyed on `(query text, document)` — is shared
across modules. The mediation module re-scores heavily overlapping
`(query, pool)` combinations that modules 2 and 5 already paid for; sharing the
cache removes a large fraction of the reranking work.

Artefacts land in `results/`; the corpus, BM25 index, document embeddings,
ONNX graph and baseline runs are cached in `cache/` and never recomputed.

## Modules

| file | role |
|---|---|
| `config.py` | Seeds, depths, model names, paths. Every knob that affects the SCM. |
| `pipeline.py` | The SCM as code: independently overridable stages, covariates, nDCG. |
| `interventions.py` | `do(Q := Q + t)` with treatment/control arms, cluster-bootstrap CIs. |
| `mediation.py` | Five counterfactual worlds per pair → additive path decomposition. |
| `dml_analysis.py` | Naive OLS vs `DoubleMLPLR` with LightGBM nuisances. |
| `stability.py` | Meaning-preserving rewrites, RBO@10 per retriever configuration. |
| `run_all.py` | Orchestration, caching, per-module wall clock. |
| `report.py` | `results/REPORT.md` + automated PASS/FAIL on the three MVP criteria. |

## Success criteria

The report adjudicates these mechanically from the artefacts on disk; a
criterion whose artefact is missing is reported `INCONCLUSIVE`, never as a pass.

1. **Intervention effects exist** — treatment-arm mean `|Δrank|` exceeds control,
   bootstrap CI for the difference excluding 0.
2. **Mediation is non-degenerate** — no single path holds >90% of the share,
   and shares shift ≥10 points between the two first-stage configurations.
3. **Confounding is live** — at least one concept whose naive estimate falls
   outside the DML confidence interval.

## Deviations from the design document

Everything here is a deliberate, documented choice — no silent substitutions.

**On A100 there are no compute deviations.** The GPU profile runs
`K_CANDIDATES=50`, `MAX_TARGET_DOCS_PER_QUERY=10`, `CE_MAX_LENGTH=512` and all
323 queries — the design document as written, at fp32.

**scikit-learn is pinned to 1.7.2** on *every* platform. This is a genuine
incompatibility, not caution: `doubleml==0.10.1` calls
`sklearn.utils.validation.check_X_y(force_all_finite=...)`, an argument
deprecated in scikit-learn 1.6 and **removed in 1.9**. With sklearn ≥1.9,
`DoubleMLPLR.fit()` raises `TypeError` and Module 4 cannot run.

**CPU-profile compromises** (only when no GPU is present). A CPU cross-encoder
sustains ~20 pairs/s, and the experiment needs ~15k full pipeline executions;
at `K_CANDIDATES=50` that is over 20 hours of reranking alone. Following the
working agreement — *reduce candidate pools before reducing query count*:

| setting | design | CPU profile | why |
|---|---|---|---|
| `K_CANDIDATES` | 50 | 20 | pool ~32 instead of ~85 |
| `MAX_TARGET_DOCS_PER_QUERY` | 10 | 3 | linear in pipeline executions |
| `CE_MAX_LENGTH` | 512 | 192 | ~2x throughput; docs average 234 words |
| `N_QUERIES` | 323 | **323 (unchanged)** | queries are the bootstrap clusters |

All 323 queries are retained either way, so the number of bootstrap clusters —
which drives CI width — is unaffected. The narrower pool compresses the range
of measurable rank movement and rank is censored at `K_CANDIDATES + 1`, both of
which make CPU-profile effects *conservative* relative to the GPU run.

**macOS Intel pins** in `requirements.txt` only. PyTorch published its last
macOS x86_64 wheel at **2.2.2**, which forces `transformers==4.44.2` and
`sentence-transformers==3.0.1` (transformers 5.x calls `torch.rms_norm`, added
in torch 2.5). None of this applies on Linux+CUDA — use `requirements-gpu.txt`.
Both specified models are used unchanged on every platform.

**ONNX Runtime cross-encoder — CPU only.** `pipeline.CrossEncoderBackend`
exports the *same* HuggingFace checkpoint to ONNX and runs it on onnxruntime.
Measured agreement with the torch path is ~5e-6 absolute, so it is an execution
optimisation, not a model substitution. It is **disabled on CUDA**, where plain
PyTorch is faster. int8 dynamic quantisation was tested and **rejected**: ~17%
extra throughput for enough score perturbation to reorder the top-10, which
would corrupt a measurement whose outcome variable is rank *change*.

**RBO implemented in-file.** As the design document prefers, rather than
depending on the `rbo` package. `stability.rbo` implements the *extrapolated*
form (Webber, Moffat & Zobel 2010, eq. 32), not the truncated sum: the
truncated sum cannot reach 1.0 at finite depth — two identical top-10 lists
would score 0.651 at p=0.9 — which would make "RBO near 1 means stable"
meaningless. A self-test covering identical, disjoint, adjacent-swap and
reversed lists runs before the module does any work.

**Cluster bootstrap.** Intervention CIs resample whole *queries*, not rows.
Rows within a query share a baseline and overlapping candidate pools, so
resampling rows would understate the standard errors.

**DML units are the candidate pool**, not the top-20 of the reranked list. The
design says "top-20 candidates"; taking the reranked top-20 would select units
on the outcome Y and add collider bias on top of the confounding being
measured. The pool averages ~32 documents per query and is selected by the
first stage only.

## Built-in correctness checks

These are load-bearing — each one would catch a specific silent failure:

- **Baseline nDCG@10** for BM25-only, dense-only and full pipeline. The run
  aborts if the full pipeline scores below 0.15, and warns if it fails to beat
  both single channels.
- **Control-arm ΔBM25 must be exactly 0.** Control terms are drawn to be absent
  from the target document, so that document's BM25 score cannot move. A
  non-zero value means term sampling is leaking.
- **BM25 under word-order shuffle must give RBO exactly 1.0.** BM25 is a bag of
  words; anything else is a bug in the pipeline or the metric.
- **The mediation residual is reported, never absorbed.** In the `bm25_only`
  configuration the dense path is absent by construction and the decomposition
  is exactly additive, so its residual must be zero.
- **RBO self-test** on identical / disjoint / swapped / reversed lists.

## Known limitations

Rank is a censored outcome (truncated at `K_CANDIDATES + 1`), so large
displacements are recorded as equal and path effects are conservative. The
sentinel is a fixed depth shared by both first-stage configurations, which
keeps the `union` vs `bm25_only` mediation comparison on a common scale.
DML standard errors treat
(query, document) units as independent while the naive OLS CIs are
query-clustered — this makes the confounding test conservative in the direction
of *not* flagging disagreement. And this is one dataset with one model pair:
nothing here shows the mediation shares transfer to other corpora or rerankers.
