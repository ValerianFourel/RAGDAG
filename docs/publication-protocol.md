# Publication protocol — competitive admission in hybrid retrieval

This is the canonical scientific specification for experiment version
`admission-v2`. Older rank mediation, DoubleML, reranker-credit and rewrite
stability results are legacy exploratory work and are not evidence for this
study.

## Claim and estimand

The system under study is a retrieval subsystem used by RAG, not an end-to-end
RAG generator. The unit is `(query, judged-relevant target document, appended
term)`. The target population is every mapped judged-relevant document, whether
inside or outside the factual hybrid pool.

The treatment is a document-derived indexed term absent from the factual query.
The estimand is its population-weighted effect under this designed intervention;
it is not the effect of naturally occurring query edits. Matched controls are
frequency-matched perturbation comparisons, not the identification strategy.

For each channel, the score perturbation is evaluated in four surgical worlds:

```
Y00  no score deltas              Y10  target delta only
Y01  competitor deltas only       Y11  all deltas (the real query edit)
```

`Y01-Y00` is called **competitive spillover**. These worlds are interventions
on score components and need not correspond to realizable text queries.
Reference-dependent factorial effects and symmetric Shapley allocations are
both reported. Hybrid admission is `S = A_BM25 OR A_dense`; every channel
effect is reported beside its hybrid-gate effect.

## Frozen design

- Collections: NFCorpus, TREC-COVID, FiQA, SciDocs, SciFact and Quora.
- Dense encoders: `BAAI/bge-small-en-v1.5` and `intfloat/e5-small-v2`, each at
  an immutable Hugging Face revision.
- Primary depth: K=50. Sensitivity depths: K=20 and K=100 on the identical
  sampled K=50 trials.
- Target strata use `h = K - min(full-corpus BM25 rank, dense rank)`: deep-in,
  boundary-in, boundary-out, mid-out and deep-out. At most two targets are
  sampled uniformly per nonempty stratum and exact inclusion probabilities are
  retained.
- Treatment terms are sampled uniformly, at most one from each support band:
  rare (`<=1%`), medium (`1–10%`) and common (`>10%`). Controls must be absent
  from query and target and satisfy `|delta log(df)| <= 0.1`; unmatched draws
  are omitted rather than weakly matched.
- Point estimates use inverse target-inclusion probabilities. Percentile 95%
  intervals use 10,000 whole-query bootstrap replicates. Prediction uses
  five-fold grouped cross-validation by query and inverse-probability weights.
- No arbitrary PASS/FAIL effect thresholds are used.

Primary hypotheses are: competitive spillover concentrates near the boundary;
threat/pressure predicts displacement better than support alone; hybrid
admission masks most channel changes; lexical absence is a BM25 structural zero
but not a dense structural zero; rescue declines with term frequency conditional
on margin; and qualitative findings replicate across encoders and K arms.

## Running the release experiment

On a networked login node, cache both models at immutable revisions and cache
all six datasets. Export the resolved 40-character commits as `BGE_REVISION`
and `E5_REVISION`. Compute nodes run offline.

```bash
bash scripts/prefetch_publication.sh
```

Submit `scripts/admission_k.sbatch`. It runs the six collections and both
encoders at K=50 on four A100s, then evaluates the same trials at K=20/100. The
job ends with:

```bash
python scripts/check_admission.py --release
```

Any failed audit, missing shard, dirty run, unresolved model revision, mixed
configuration, absent sampling frame, failed Shapley reconstruction or missing
release artefact makes the job fail.

## Interpretation limits

Defensible claims concern competitive admission, rescue/displacement, model
differences and masking under the specified intervention. Do not claim
assumption-free causal truth, natural-query effects, end-to-end answer quality,
unconditional RAG performance, or a universal numerical masking ratio.
