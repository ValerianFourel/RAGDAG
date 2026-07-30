"""Preflight: verify the environment and the experiment's load-bearing invariants.

Fast enough to run before every job (a minute on a GPU, a few on CPU). It checks
the things that fail *silently* — a CPU profile on a GPU node, a stale cache, a
control arm that is quietly contaminated — rather than the things that already
crash loudly on their own.

    python scripts/check.py

Exits non-zero if any check fails, so it can gate a batch script.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator: run a check, capture its verdict, never let it abort the run.

    A check returns ``True`` (pass), ``False`` (fail, gates the run) or the
    string ``"warn"`` — for properties that are statistical rather than
    structural and so cannot be asserted on a 12-query preflight.
    """

    def deco(fn):
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
            if "-v" in sys.argv:
                traceback.print_exc()
        RESULTS.append((name, ok, detail))
        mark = "WARN" if ok == "warn" else ("PASS" if ok else "FAIL")
        print(f"  [{mark}] {name:<34} {detail}  ({time.time() - t0:.1f}s)", flush=True)
        return fn

    return deco


print("=" * 78)
print("RAGDAG PREFLIGHT")
print("=" * 78)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
@check("python >= 3.10")
def _py():
    v = sys.version_info
    return v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}"


@check("all imports resolve")
def _imports():
    import importlib.util

    need = ["torch", "transformers", "sentence_transformers", "numpy", "pandas",
            "scipy", "sklearn", "doubleml", "lightgbm", "statsmodels", "bm25s",
            "Stemmer", "ir_datasets", "matplotlib", "pyarrow"]
    missing = [m for m in need if importlib.util.find_spec(m) is None]
    return not missing, "all present" if not missing else "MISSING " + ", ".join(missing)


@check("sklearn/doubleml compatible")
def _sk():
    import inspect

    import sklearn
    from sklearn.utils.validation import check_X_y

    ok = "force_all_finite" in inspect.signature(check_X_y).parameters
    return ok, f"sklearn {sklearn.__version__}" + ("" if ok else " dropped force_all_finite")


import config  # noqa: E402


@check("device and profile")
def _dev():
    import torch

    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    s = config.summary()
    detail = (f"{config.DEVICE}, {n} gpu(s), K={s['K_CANDIDATES']}, "
              f"ce_len={s['ce_max_length']}, targets={s['max_target_docs_per_query']}, "
              f"{s['precision']}")
    # On a GPU box the design profile must be active; on CPU we only report.
    if config.ON_GPU:
        ok = n > 0 and s["K_CANDIDATES"] == 50 and s["ce_max_length"] == 512
    else:
        ok = True
    return ok, detail


@check("caches present (offline-ready)")
def _cache():
    import os

    missing = []
    if not config.CORPUS_CACHE.exists():
        missing.append("corpus.pkl")
    if not config.BM25_INDEX_CACHE.exists():
        missing.append("bm25_index/")
    hf = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    if not hf.exists():
        missing.append(f"HF_HOME={hf}")
    if missing:
        return False, "run scripts/prefetch.py — missing " + ", ".join(missing)
    return True, f"corpus + bm25 index cached; HF_HOME={hf}"


# --------------------------------------------------------------------------- #
# Scientific invariants
# --------------------------------------------------------------------------- #
config.set_seeds()
print("-" * 78)

from pipeline import (  # noqa: E402
    RetrievalPipeline, compute_baseline, content_tokens, load_corpus_and_queries,
    ndcg_at_k, select_queries,
)

_corpus, _queries = load_corpus_and_queries()
_pipe = RetrievalPipeline(_corpus, verbose=False)
_qids = select_queries(_queries, 12)
_base = compute_baseline(_pipe, _queries, _qids)


@check("RBO metric self-test")
def _rbo():
    import stability as S

    S._self_test()
    return True, "identical=1.0, disjoint~0, ordering respected"


@check("pipeline is not broken (nDCG)")
def _ndcg():
    """Gate on the hard floor only.

    "Full pipeline beats both channels" is a *large-sample* property — it holds
    comfortably on all 323 queries (0.3548 / 0.3449 / 0.3235) but is well within
    noise on a 12-query preflight, so asserting it here would produce false
    alarms. The genuine broken-pipeline signal is the 0.15 floor.
    """
    import numpy as np

    nb, nd, nf = [], [], []
    for qid, run in _base.items():
        rel = _queries.qrels.get(qid, {})
        nb.append(ndcg_at_k(run.bm25_top[: config.K_FINAL], rel))
        nd.append(ndcg_at_k(run.dense_top[: config.K_FINAL], rel))
        nf.append(ndcg_at_k([d for d, _ in run.reranked[: config.K_FINAL]], rel))
    b, d, f = float(np.mean(nb)), float(np.mean(nd)), float(np.mean(nf))
    detail = f"n={len(nf)} bm25={b:.4f} dense={d:.4f} full={f:.4f}"
    if f < 0.15:
        return False, detail + "  (below the 0.15 floor — pipeline is broken)"
    if f <= max(b, d):
        return "warn", detail + "  (full <= best channel; expected at n=12, check the real run)"
    return True, detail


@check("control arm delta-BM25 is exactly 0")
def _ctrl():
    """The control term is absent from the target document *after stemming*, so
    that document's BM25 score cannot move. A non-zero value means the term
    sampler is leaking weak treatments into the control arm."""
    import numpy as np

    import interventions as I

    vs = I.build_vocab_stats(_corpus)
    worst, n = 0.0, 0
    for qid in _qids:
        run = _base[qid]
        qtok = set(content_tokens(run.query_text))
        qs = frozenset(I.stem(sorted(qtok))) if qtok else frozenset()
        rng = np.random.default_rng(config.stable_seed(qid))
        for d in I.select_targets(run, _queries.relevant(qid), rng):
            di = _corpus.idx(d)
            dtok = _corpus.doc_content_tokens[di]
            ds = frozenset(I.stem(sorted(dtok))) if dtok else frozenset()
            I.sample_treatment_terms(_corpus, d, qtok, vs, rng)  # keep RNG in step
            for t in I.sample_control_terms(dtok, qtok, vs, rng, doc_stems=ds, query_stems=qs):
                base = run.bm25_scores_cand.get(d, 0.0)
                new = float(_pipe._bm25_array(f"{run.query_text} {t}")[di])
                worst = max(worst, abs(new - base))
                n += 1
    return worst == 0.0, f"max |delta| over {n} control terms = {worst:.2e}"


@check("BM25 is order-invariant under shuffle")
def _shuf():
    """BM25 is a bag of words, so permuting the query must not move its ranking.
    Anything but exactly 1.0 is a bug in the pipeline or in RBO."""
    import numpy as np

    import stability as S

    vals = []
    for qid in _qids:
        q0 = _queries.texts[qid]
        rng = np.random.default_rng(config.stable_seed("stab", qid))
        qv = S.variant_shuffle(q0, rng)
        vals.append(S.rbo(_pipe.rank_bm25_only(q0), _pipe.rank_bm25_only(qv)))
    m = float(np.mean(vals))
    return m == 1.0, f"mean RBO = {m:.6f} over {len(vals)} queries"


@check("lexical-only mediation residual is 0")
def _resid():
    """With no dense channel the decomposition is exactly additive by
    construction, so a non-zero residual would mean the freezing logic is wrong."""
    import numpy as np

    import interventions as I
    import mediation as M

    vs = I.build_vocab_stats(_corpus)
    worst, n = 0, 0
    for qid in _qids[:4]:
        run = _base[qid]
        q0 = run.query_text
        qtok = set(content_tokens(q0))
        rng = np.random.default_rng(config.stable_seed(qid))
        tgts = I.select_targets(run, _queries.relevant(qid), rng)
        if not tgts:
            continue
        d = tgts[0]
        terms = I.sample_treatment_terms(_corpus, d, qtok, vs, rng)[:1]
        r0 = _pipe.run(q0, first_stage="bm25").rank_of(d)
        for t in terms:
            out = M.decompose(_pipe, q0, f"{q0} {t}", d, "bm25", r0)
            worst = max(worst, abs(out["residual"]))
            n += 1
    return worst == 0, f"max |residual| over {n} pairs = {worst}"


# --------------------------------------------------------------------------- #
print("=" * 78)
_failed = [n for n, ok, _ in RESULTS if ok is False]
_warned = [n for n, ok, _ in RESULTS if ok == "warn"]
if _failed:
    print(f"FAILED {len(_failed)}/{len(RESULTS)}: " + ", ".join(_failed))
    print("=" * 78)
    raise SystemExit(1)
msg = f"ALL {len(RESULTS)} CHECKS PASSED"
if _warned:
    msg += f" ({len(_warned)} warning: {', '.join(_warned)})"
print(msg + " — safe to submit the full run.")
print("=" * 78)
