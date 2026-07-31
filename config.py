"""Global configuration for the causal-retrieval-mvp experiment.

Every knob that affects the structural causal model (SCM) or its estimation lives
here so that a run is fully reproducible from ``config.py`` + the fixed seed.

Causal role: this module fixes the *exogenous* parts of the experiment (seeds,
model identities, truncation depths). Nothing downstream may introduce
non-determinism that is not derived from :data:`SEED`.
"""

from __future__ import annotations

import hashlib
import os
import random
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    raise RuntimeError(
        f"RAGDAG needs Python >= 3.10 (3.11 recommended); this is "
        f"{sys.version_info.major}.{sys.version_info.minor} at {sys.executable}.\n"
        "\n"
        "On an HPC login node the usual cause is that a `module load` for Python\n"
        "silently failed, so `python -m venv` fell back to the system interpreter.\n"
        "Check with:  python -V   and   module spider python\n"
        "\n"
        "The most reliable fix needs no module system at all - uv fetches a\n"
        "standalone CPython:\n"
        "    curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        "    export PATH=\"$HOME/.local/bin:$PATH\"\n"
        "    rm -rf .venv && uv venv --python 3.11 .venv\n"
        "    source .venv/bin/activate\n"
    )

import numpy as np

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
#: ir_datasets identifier. Override with the ``DATASET`` env var; every cache
#: key and the run metadata are scoped by it, so switching is safe.
DATASET: str = os.environ.get("DATASET", "beir/nfcorpus/test").strip()

#: Number of queries to run. ``None`` means "all 323 test queries".
#: Override with the ``N_QUERIES`` environment variable for smoke tests.
_n_env = os.environ.get("N_QUERIES", "").strip()
N_QUERIES: int | None = int(_n_env) if _n_env else None


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
#: Modules whose source affects the numbers. Hashed into the run fingerprint so
#: that a code change invalidates cached artefacts.
_CODE_MODULES = (
    "config.py", "pipeline.py", "interventions.py",
    "mediation.py", "dml_analysis.py", "stability.py",
)


def code_fingerprint() -> str:
    """Content hash of the modules that determine the results.

    Without this, ``run_meta.json`` records only configuration, so a pure code
    change leaves the fingerprint identical and ``run_all`` silently reuses
    artefacts computed by the *old* code. That is not hypothetical: the
    control-term sampler was corrected twice while the config stayed fixed, and
    a rerun would have reported the pre-fix parquet as a fresh result.

    Hashes source content rather than the git SHA on purpose - it catches
    uncommitted edits, and it works in a clone without git available.
    """
    h = hashlib.blake2b(digest_size=8)
    root = Path(__file__).parent
    for name in _CODE_MODULES:
        p = root / name
        h.update(name.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    """Process-independent 32-bit seed derived from ``parts``.

    Python's built-in ``hash()`` is salted per interpreter unless PYTHONHASHSEED
    is set *before* the process starts, so ``hash(query_id)`` yields different
    values on every run and in every worker. Seeding per-query RNGs with it
    would make term sampling irreproducible across reruns and inconsistent
    between a single-process run and a sharded multi-GPU run. blake2b is
    stable everywhere.
    """
    payload = "\x1f".join(repr(p) for p in (SEED, *parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def _detect_device() -> str:
    """Resolve the compute device once, at import.

    Force with ``DEVICE=cuda`` / ``DEVICE=cuda:1`` / ``DEVICE=cpu``.
    """
    forced = os.environ.get("DEVICE", "").strip()
    if forced:
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEVICE: str = _detect_device()
ON_GPU: bool = DEVICE.startswith("cuda")


def _int_env(name: str, gpu_default: int, cpu_default: int) -> int:
    return int(os.environ.get(name, gpu_default if ON_GPU else cpu_default))

# --------------------------------------------------------------------------- #
# Pipeline depths
# --------------------------------------------------------------------------- #
#: Top-K taken from *each* first-stage channel (BM25 and dense) before union.
#:
#: The design document specifies 50, and that is what runs on a GPU. The CPU
#: fallback of 20 exists because a CPU cross-encoder sustains only ~20 pairs/s,
#: and the experiment needs ~15k full pipeline executions; at K=50 (pool ~85)
#: that is >20 h of reranking. Per the working agreement the candidate pool
#: shrinks before the query count does - all 323 queries are kept either way.
K_CANDIDATES: int = _int_env("K_CANDIDATES", 50, 20)

#: Depth of the final reranked list used for nDCG and RBO.
K_FINAL: int = 10

#: Sentinel rank assigned to a document that falls out of the candidate pool.
#: Observed ranks are truncated to this value too, so ``rank`` is a censored
#: measure on ``[1, MISSING_RANK]`` and Delta-rank is always well defined.
MISSING_RANK: int = K_CANDIDATES + 1

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
DENSE_MODEL: str = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: BGE models are trained with an asymmetric query instruction; omitting it
#: costs several nDCG points. Documents are embedded with no prefix.
BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

EMBED_BATCH_SIZE: int = _int_env("EMBED_BATCH_SIZE", 256, 64)
CE_BATCH_SIZE: int = _int_env("CE_BATCH_SIZE", 256, 64)

#: Dense encoder sequence length (documents are ~234 words / ~310 tokens).
MAX_SEQ_LENGTH: int = _int_env("MAX_SEQ_LENGTH", 512, 350)

#: Cross-encoder sequence length. On GPU this is the model's full 512, which
#: covers every NFCorpus document. The CPU fallback of 192 buys ~2x throughput
#: at the cost of truncating late-abstract content, because the reranker is the
#: bottleneck there; the quality cost shows up in the baseline sanity check.
CE_MAX_LENGTH: int = _int_env("CE_MAX_LENGTH", 512, 192)

#: Use ONNX Runtime for the cross-encoder. Numerically identical to the torch
#: path (max abs deviation ~5e-6 on this model) and ~15% faster on CPU, where
#: it is on by default. Off on GPU: CUDA torch is already faster than
#: onnxruntime's CPU provider, and it keeps the GPU path plain PyTorch.
USE_ONNX_CE: bool = os.environ.get("USE_ONNX_CE", "0" if ON_GPU else "1") == "1"

#: Half precision for the encoders. **Off by default even on A100.** The
#: outcome variable of this experiment is a *rank change*, so anything that
#: perturbs scores can manufacture or erase an effect. int8 quantisation was
#: measured at 9/10 top-10 agreement and rejected for exactly this reason; fp16
#: is far gentler but is still opt-in rather than assumed. An A100 runs the
#: whole experiment in well under an hour at fp32, so there is nothing to buy.
USE_FP16: bool = os.environ.get("USE_FP16", "0") == "1"

#: Allow TF32 matmuls on Ampere. Also off by default, same reasoning: TF32 has
#: a 10-bit mantissa and can flip near-ties in the ranking.
ALLOW_TF32: bool = os.environ.get("ALLOW_TF32", "0") == "1"

# --------------------------------------------------------------------------- #
# Module 2 - interventions
# --------------------------------------------------------------------------- #
#: Design document says 10, which is the GPU default. The CPU fallback of 3 is
#: the same budget compromise as :data:`K_CANDIDATES`. All 323 queries are
#: retained either way, so the number of bootstrap clusters - which drives CI
#: width - is unchanged.
MAX_TARGET_DOCS_PER_QUERY: int = _int_env("MAX_TARGET_DOCS", 10, 3)
N_TREATMENT_TERMS: int = 3
N_CONTROL_TERMS: int = 3
N_BOOTSTRAP: int = 1000

# --------------------------------------------------------------------------- #
# Module 4 - DoubleML
# --------------------------------------------------------------------------- #
DML_TOP_K_CANDIDATES: int = 20
DML_N_CONCEPTS: int = 3
DML_DF_MIN: float = 0.05
DML_DF_MAX: float = 0.40
DML_N_FOLDS: int = 5
DML_N_REP: int = 3

#: LightGBM threads for the nuisance learners. Default 1, deliberately.
#:
#: Raising this deadlocked at 0% CPU during development: torch and LightGBM each
#: load an OpenMP runtime, and two OpenMP runtimes in one process can hang on
#: ``fork``/thread-pool init (classic libomp-vs-libgomp duplicate-runtime
#: problem, most often seen on macOS). Combined with ``deterministic=True`` the
#: results are thread-count independent anyway, so raising it buys speed only.
#: Try ``DML_N_JOBS=8`` on a Linux node if this stage dominates your wall clock,
#: but treat a 0%-CPU stall as this issue rather than as slowness.
DML_N_JOBS: int = int(os.environ.get("DML_N_JOBS", 1))

# --------------------------------------------------------------------------- #
# Module 5 - stability
# --------------------------------------------------------------------------- #
STABILITY_N_QUERIES: int = int(os.environ.get("STABILITY_N_QUERIES", 50))
RBO_P: float = 0.9

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT: Path = Path(__file__).parent.resolve()
CACHE_DIR: Path = ROOT / "cache"

#: Results are dataset-scoped. Without this a second dataset overwrites the
#: first: every artefact path (parquets, figures, REPORT.md, run_meta.json) is
#: a fixed filename, so running SciFact after NFCorpus silently destroyed the
#: NFCorpus results rather than sitting alongside them.
RESULTS_ROOT: Path = ROOT / "results"
RESULTS_DIR: Path = RESULTS_ROOT / DATASET.replace("/", "-")

CACHE_DIR.mkdir(exist_ok=True)
RESULTS_ROOT.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#: Precision tag for cache keys. fp16/TF32 change the numbers, so artefacts
#: computed under one precision must not be silently reused under another.
PRECISION_TAG: str = "fp16" if USE_FP16 else ("tf32" if ALLOW_TF32 else "fp32")

#: Cache keys embed every setting that changes the stored numbers. Without
#: this, moving a checkout from the CPU profile (dense len 350, CE len 192) to
#: the GPU profile (512 / 512) would reuse the short-sequence embeddings and
#: cross-encoder scores and silently mix two different models' outputs.
_DENSE_TAG: str = DENSE_MODEL.split("/")[-1]

#: Every cache key is dataset-scoped. Without this, `load_corpus_and_queries`
#: checks `CORPUS_CACHE.exists()` *before* the dataset string is ever read, so
#: switching DATASET silently returns the previously cached corpus AND its
#: queries - a complete substitution that runs to completion and reports wrong
#: numbers without erroring.
DATASET_TAG: str = DATASET.replace("/", "-")

EMBEDDINGS_CACHE: Path = (
    CACHE_DIR
    / f"doc_emb_{DATASET_TAG}_{_DENSE_TAG}_L{MAX_SEQ_LENGTH}_{PRECISION_TAG}.npz"
)
BM25_INDEX_CACHE: Path = CACHE_DIR / f"bm25_{DATASET_TAG}"
CORPUS_CACHE: Path = CACHE_DIR / f"corpus_{DATASET_TAG}.pkl"


def baseline_cache_path(qids: list[str]) -> Path:
    """Cache path for a baseline run set.

    Keyed on the *content* of the query list, not its length: under sharding,
    worker 1 of 4 and worker 2 of 4 both hold ~81 queries, and a length-based
    key would make them overwrite each other's cache with different data.
    """
    digest = hashlib.blake2b(
        "\n".join(sorted(qids)).encode(), digest_size=6
    ).hexdigest()
    return CACHE_DIR / (
        f"baseline_{DATASET_TAG}_n{len(qids)}_{digest}_k{K_CANDIDATES}"
        f"_ce{CE_MAX_LENGTH}_d{MAX_SEQ_LENGTH}_{PRECISION_TAG}.pkl"
    )


# --------------------------------------------------------------------------- #
# Per-dataset baseline expectations (WP-4 gate, as data rather than prose)
# --------------------------------------------------------------------------- #
#: dataset -> (min, max) for full-pipeline nDCG@10 under *this* profile.
#: ``None`` means "no expectation recorded yet"; the run reports the observed
#: value and applies only the universal broken-pipeline floor. Fill in after the
#: first clean run on a dataset rather than guessing a range up front.
BASELINE_EXPECTATIONS: dict[str, tuple[float, float] | None] = {
    "beir/nfcorpus/test": (0.33, 0.38),   # observed 0.3568 at K=50/ce512
    # SciFact observed 0.6695 at K=20/ce192 (CPU profile), 300 queries. NOTE:
    # this is a *degraded* pipeline - see RERANKER_HELPS below.
    "beir/scifact/test": (0.62, 0.78),
    "beir/scidocs": None,
    "beir/quora/test": None,
    "beir/fiqa/test": None,
}

#: Collections where the cross-encoder is known NOT to improve over the best
#: single channel. ms-marco-MiniLM is trained on web-style queries; on SciFact's
#: scientific claims it actively degrades the ranking (0.6695 full vs 0.7127
#: dense). Mediation shares on such a collection describe causal responsibility
#: for an intervention's effect, NOT a well-configured pipeline - the report
#: must say so rather than let the two be conflated.
KNOWN_RERANKER_HARMFUL: frozenset[str] = frozenset({"beir/scifact/test"})

#: Universal floor. Below this the pipeline is broken on any collection.
BASELINE_FLOOR: float = 0.15


# --------------------------------------------------------------------------- #
# Multi-GPU sharding
# --------------------------------------------------------------------------- #
def shard_queries(qids: list[str], shard: int, n_shards: int) -> list[str]:
    """Slice a query list for one worker.

    Strided rather than contiguous: per-query cost varies a lot (number of
    judged-relevant documents in the pool drives how many interventions a query
    generates), and striding averages that out across workers far better than
    handing one worker a contiguous block.
    """
    if n_shards <= 1:
        return list(qids)
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} out of range for n_shards={n_shards}")
    return list(qids[shard::n_shards])


SHARD_DIR: Path = RESULTS_DIR / "shards"


def shard_path(name: str, shard: int, n_shards: int) -> Path:
    """Per-worker artefact path, e.g. ``results/shards/interventions_0of4.parquet``."""
    SHARD_DIR.mkdir(exist_ok=True, parents=True)
    return SHARD_DIR / f"{name}_{shard}of{n_shards}.parquet"

#: Cross-encoder pair cache is cleared once it exceeds this many entries.
#: Work is grouped by originating query, so locality keeps the hit rate high.
#: ~150 bytes/entry, so 1.5M entries is roughly 250 MB - worth it, since
#: ``run_all`` shares one cache across modules 2-5. The GPU profile runs the
#: full design (pool ~85, 10 targets/query) and generates several million
#: distinct pairs, so it gets a larger ceiling to avoid mid-run eviction.
CE_CACHE_MAX: int = _int_env("CE_CACHE_MAX", 6_000_000, 1_500_000)


#: Torch intra-op threads. Left at torch's default of physical/2 on this box,
#: which measurably under-uses the machine; set explicitly.
N_TORCH_THREADS: int = int(os.environ.get("N_TORCH_THREADS", os.cpu_count() or 4))


def set_seeds(seed: int = SEED) -> None:
    """Pin every RNG the pipeline can touch, and configure device precision.

    Causal role: interventions must be the *only* source of variation between
    two runs. Any residual randomness would contaminate the estimated effects -
    which is also why TF32 and fp16 are opt-in rather than default.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # CE inference only; no autograd
    torch.set_num_threads(N_TORCH_THREADS)
    torch.set_grad_enabled(False)  # pure inference everywhere
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = ALLOW_TF32


def device_banner() -> str:
    """One-line description of what the run is executing on."""
    if not ON_GPU:
        return f"device={DEVICE} threads={N_TORCH_THREADS} onnx_ce={USE_ONNX_CE}"
    try:
        import torch

        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        n = torch.cuda.device_count()
        return (
            f"device={DEVICE} ({name}, {mem:.0f} GB, {n} visible) "
            f"fp16={USE_FP16} tf32={ALLOW_TF32}"
        )
    except Exception:
        return f"device={DEVICE} fp16={USE_FP16} tf32={ALLOW_TF32}"


def summary() -> dict[str, object]:
    """Return the config as a flat dict for the report header."""
    return {
        "seed": SEED,
        "dataset": DATASET,
        "device": DEVICE,
        "precision": "fp16" if USE_FP16 else ("tf32" if ALLOW_TF32 else "fp32"),
        "n_queries": N_QUERIES if N_QUERIES is not None else "all",
        "K_CANDIDATES": K_CANDIDATES,
        "K_FINAL": K_FINAL,
        "ce_max_length": CE_MAX_LENGTH,
        "dense_model": DENSE_MODEL,
        "cross_encoder_model": CROSS_ENCODER_MODEL,
        "max_target_docs_per_query": MAX_TARGET_DOCS_PER_QUERY,
        "n_treatment_terms": N_TREATMENT_TERMS,
        "n_control_terms": N_CONTROL_TERMS,
        "n_bootstrap": N_BOOTSTRAP,
        "dml_folds": DML_N_FOLDS,
        "dml_reps": DML_N_REP,
        "stability_n_queries": STABILITY_N_QUERIES,
        "rbo_p": RBO_P,
    }


if __name__ == "__main__":
    for k, v in summary().items():
        print(f"{k:32s} {v}")
