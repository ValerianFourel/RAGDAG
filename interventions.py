"""Module 2 - do()-interventions on the query.

The treatment is *term injection*::

    do(Q := Q0 + " " + t)

This is a genuine intervention rather than a conditioning operation: we set the
query to a new value and re-run the entire pipeline, so nothing upstream of Q
can confound the comparison. The control arm injects a term drawn from the
corpus vocabulary that appears in neither the query nor the target document -
it holds the *form* of the intervention (one extra token) fixed while removing
its *content* (relevance to the target). The treatment-minus-control contrast
therefore isolates the effect of injecting a topically relevant term, not the
effect of lengthening the query.

Run standalone::

    python -m interventions
"""

from __future__ import annotations

import math
import pickle
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from pipeline import (
    BaselineRun,
    Corpus,
    Queries,
    RetrievalPipeline,
    compute_baseline,
    content_tokens,
    load_corpus_and_queries,
    select_queries,
)

OUT_PARQUET = config.RESULTS_DIR / "interventions.parquet"
OUT_ORIGIN_DOCS = config.RESULTS_DIR / "origin_documents.parquet"
FIG_EFFECTS = config.RESULTS_DIR / "fig_intervention_effects.png"


# --------------------------------------------------------------------------- #
# Vocabulary statistics
# --------------------------------------------------------------------------- #
@dataclass
class VocabStats:
    """Corpus-level lexical statistics used to choose injection terms.

    Carries the two quantities the do()-operator is parameterised on:

    * **support** - ``df(t)/N``, the fraction of documents containing ``t``.
      Controls *how many documents the injection moves*, because appending
      ``t`` to a query changes the BM25 score of exactly the ``df(t)``
      documents that contain it and of no others.
    * **lift** - ``P(t|d) / P(t|corpus)``, a density ratio. Controls *how much
      this particular document moves*.

    These are orthogonal and they drive different stages: support drives
    candidate-pool churn (a first-stage effect), lift drives the target's own
    rescoring (a reranker effect). The previous TF-IDF sampler collapsed both
    into one weight and could not separate them - and because ``tf`` dominated
    that product, it drew *topical* words rather than *distinctive* ones.
    """

    df: dict[str, int]
    cf: dict[str, int]
    n_docs: int
    total_tokens: int
    control_pool: list[str]
    bin_edges: np.ndarray
    control_pool_by_bin: list[list[str]]

    def idf(self, term: str) -> float:
        return math.log(self.n_docs / (1 + self.df.get(term, 0)))

    def support(self, term: str) -> float:
        """Fraction of documents containing ``term``."""
        return self.df.get(term, 0) / max(1, self.n_docs)

    def lift(self, term: str, tf_in_doc: int, doc_len: int) -> float:
        """Density of ``term`` in one document relative to the whole corpus."""
        if tf_in_doc <= 0 or doc_len <= 0:
            return 0.0
        p_corpus = self.cf.get(term, 0) / max(1, self.total_tokens)
        if p_corpus <= 0:
            return float("inf")
        return (tf_in_doc / doc_len) / p_corpus

    def bin_of(self, term: str) -> int:
        """Support-bin index, or -1 if the term is outside the admissible band."""
        s = self.support(term)
        if s < self.bin_edges[0] or s > self.bin_edges[-1]:
            return -1
        return int(min(np.searchsorted(self.bin_edges, s, side="right") - 1,
                       len(self.bin_edges) - 2))

    @property
    def n_bins(self) -> int:
        return len(self.bin_edges) - 1


def _is_word(t: str) -> bool:
    return t.isalpha() and len(t) >= 3


def _frequency_tables(corpus: Corpus) -> tuple[dict[str, int], dict[str, int], int, int]:
    """``(df, cf, n_docs, total_tokens)``, cached to disk.

    Collection frequency needs per-document term *counts*; the corpus cache
    stores presence sets only, so this is a full re-tokenisation of the
    collection - minutes on the larger BEIR corpora. It is read-only and
    identical for every shard worker, so it is computed once and reused.
    Serialised as plain dicts rather than as a dataclass: a pickled dataclass
    binds to its defining module and has already been unloadable here once.
    """
    p = config.VOCAB_STATS_CACHE
    if p.exists():
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            if d.get("n_docs") == len(corpus):
                return d["df"], d["cf"], d["n_docs"], d["total_tokens"]
            print(f"[interventions] {p.name} is for a different corpus size - rebuilding")
        except Exception as e:  # noqa: BLE001 - a bad cache must not be fatal
            print(f"[interventions] could not read {p.name} ({e}) - rebuilding")

    t0 = time.time()
    df: Counter[str] = Counter()
    cf: Counter[str] = Counter()
    for i in range(len(corpus)):
        toks = [t for t in content_tokens(corpus.texts[i]) if _is_word(t)]
        cf.update(toks)
        df.update(set(toks))
    n, total = len(corpus), int(sum(cf.values()))
    print(f"[interventions] built vocabulary tables in {time.time() - t0:.1f}s "
          f"({n} docs, {total} tokens, {len(df)} terms) -> {p.name}")
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"df": dict(df), "cf": dict(cf), "n_docs": n, "total_tokens": total}, f)
    tmp.replace(p)  # atomic: 4 workers may race on first run
    return dict(df), dict(cf), n, total


def build_vocab_stats(corpus: Corpus, min_df: int = 5) -> VocabStats:
    """Document/collection frequencies, support bins, and per-bin control pools.

    Bins are **log-spaced** over support: vocabulary is Zipfian, so linear bins
    would put essentially the whole vocabulary in the lowest bucket and give no
    coverage of the range where pool churn actually happens.

    A control pool is built *per bin* so the control arm can be matched to the
    treatment arm on support. Without that match the two arms differ on two
    dimensions at once - in-document vs not, and common vs rare - and the
    contrast is not identified on either.
    """
    df, cf, n, total = _frequency_tables(corpus)

    lo = max(min_df / n, config.SUPPORT_MIN)
    hi = config.SUPPORT_MAX
    edges = np.logspace(np.log10(lo), np.log10(hi), config.N_SUPPORT_BINS + 1)
    # logspace does not round-trip its own endpoints (10**log10(x) != x in
    # binary floating point), which silently dropped every term sitting exactly
    # at min_df - the rarest bin's whole boundary. Pin the ends.
    edges[0], edges[-1] = lo, hi

    vs = VocabStats(
        df=dict(df), cf=dict(cf), n_docs=n, total_tokens=total,
        control_pool=[], bin_edges=edges, control_pool_by_bin=[],
    )
    by_bin: list[list[str]] = [[] for _ in range(vs.n_bins)]
    for t, c in df.items():
        if c < min_df:
            continue
        b = vs.bin_of(t)
        if b >= 0:
            by_bin[b].append(t)
    vs.control_pool_by_bin = [sorted(v) for v in by_bin]
    vs.control_pool = sorted(t for v in by_bin for t in v)
    return vs


def tfidf_terms(corpus: Corpus, doc_id: str, vs: VocabStats) -> tuple[list[str], np.ndarray]:
    """TF-IDF weighted candidate injection terms for one document."""
    i = corpus.idx(doc_id)
    tf = Counter(t for t in content_tokens(corpus.texts[i]) if _is_word(t))
    if not tf:
        return [], np.zeros(0)
    terms = sorted(tf)
    w = np.array([tf[t] * vs.idf(t) for t in terms], dtype=np.float64)
    w = np.clip(w, 0.0, None)
    return terms, w


# --------------------------------------------------------------------------- #
# Term sampling
# --------------------------------------------------------------------------- #
def sample_treatment_terms(
    corpus: Corpus,
    doc_id: str,
    query_tokens: set[str],
    vs: VocabStats,
    rng: np.random.Generator,
    n: int = config.N_TREATMENT_TERMS,
    query_bm25: frozenset[str] = frozenset(),
    bm25_df: dict[str, int] | None = None,
) -> list[dict]:
    """Sample at most one target term from each prespecified support band.

    Two-stage draw:

    1. pick a support bin uniformly at random among the bins this document can
       actually populate (without replacement while distinct bins remain);
    2. within that bin pick a term with probability proportional to its
       **lift** in this document.

    Stage 1 is the fix for the old sampler. Under TF-IDF weighting the realised
    treatment terms landed ~3x more common than the controls they were
    contrasted against (median support 1.3% vs 0.4%), and support turns out to
    be the strongest single moderator in the data - the reranker's share of the
    decomposition moves from 73% to 91% across support quartiles *within one
    collection*. Sweeping support deliberately turns that confound into a
    designed factor and yields a dose-response curve instead of a point
    estimate.

    Stage 2 preserves the original intent - inject a word characteristic of
    *this* document - but now expressed as a density ratio rather than as a
    tf-idf product in which ``tf`` silently dominated.

    Terms already present in the query are excluded (against the BM25 view of
    the query, not just its surface tokens), otherwise the intervention would
    be close to a no-op on the lexical channel.

    Returns one provenance dict per sampled term, so the recorded
    ``select_prob`` is the sampler's own number rather than a reconstruction
    that can drift away from it.
    """
    i = corpus.idx(doc_id)
    dl = int(corpus.doc_len[i])
    tf_all = Counter(t for t in content_tokens(corpus.texts[i]) if _is_word(t))
    cands = [
        t for t in sorted(tf_all)
        if t not in query_tokens
        and not (bm25_terms(t) & query_bm25)
        and vs.bin_of(t) >= 0
    ]
    if not cands:
        return []

    def indexed_support(term: str) -> float:
        bt = bm25_terms(term)
        return ((bm25_df or {}).get(next(iter(bt)), 0) / vs.n_docs
                if len(bt) == 1 and bm25_df is not None else vs.support(term))

    bands = (("rare", 0.0, 0.01), ("medium", 0.01, 0.10),
             ("common", 0.10, 1.01))
    out: list[dict] = []
    for band, lo, hi in bands[:n]:
        pool = [t for t in cands if len(bm25_terms(t)) == 1
                and lo <= indexed_support(t) < hi]
        if not pool:
            continue
        term = pool[int(rng.integers(len(pool)))]
        b = vs.bin_of(term)
        out.append({
            "term": term,
            "support_bin": b,
            "support_band": band,
            "term_bm25_df": int(round(indexed_support(term) * vs.n_docs)),
            "select_prob": float(1.0 / len(pool)),
            "term_lift": float(vs.lift(term, tf_all[term], dl)),
            "n_candidate_terms": len(cands),
            "n_candidate_bins": 3,
        })
    return out


_STEMMER = None
_BM25_TOK_CACHE: dict[str, frozenset[str]] = {}


def stem(words: list[str]) -> list[str]:
    """Snowball stems, matching what the BM25 index actually stores."""
    global _STEMMER
    if _STEMMER is None:
        import Stemmer

        _STEMMER = Stemmer.Stemmer("english")
    return _STEMMER.stemWords(words)


def bm25_terms(text: str) -> frozenset[str]:
    """Exactly the terms BM25 will score for ``text``.

    This is the ground truth for "does this term touch that document's BM25
    score", and it is *not* the same as stopword-filtered stems: the project's
    covariate tokeniser uses scikit-learn's stopword list (318 words) while the
    BM25 index uses bm25s' (33). The 285-word gap contains ordinary words -
    have, about, between, back - which BM25 indexes and scores but which the
    covariate view discards. Approximating with the wrong filter is what let
    control term "having" (stem "have") move a document's BM25 score.
    """
    global _STEMMER
    hit = _BM25_TOK_CACHE.get(text)
    if hit is not None:
        return hit
    import bm25s

    if _STEMMER is None:
        stem(["x"])
    toks = bm25s.tokenize(
        text, stopwords="en", stemmer=_STEMMER, return_ids=False, show_progress=False
    )[0]
    out = frozenset(toks)
    if len(_BM25_TOK_CACHE) > 20000:
        _BM25_TOK_CACHE.clear()
    _BM25_TOK_CACHE[text] = out
    return out


def build_bm25_df(corpus: Corpus) -> dict[str, int]:
    """Document frequencies in the index tokenizer's exact term space."""
    if config.BM25_DF_CACHE.exists():
        try:
            with open(config.BM25_DF_CACHE, "rb") as f:
                cached = pickle.load(f)
            if cached.get("n_docs") == len(corpus):
                return {str(t): int(v) for t, v in cached["df"].items()}
        except Exception:  # noqa: BLE001
            pass
    counts: Counter = Counter()
    for text in corpus.texts:
        counts.update(bm25_terms(text))
    tmp = config.BM25_DF_CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"n_docs": len(corpus), "df": dict(counts)}, f)
    tmp.replace(config.BM25_DF_CACHE)
    return dict(counts)


def sample_control_terms(
    doc_tokens: frozenset[str],
    query_tokens: set[str],
    vs: VocabStats,
    rng: np.random.Generator,
    match_bins: list[int] | None = None,
    n: int = config.N_CONTROL_TERMS,
    doc_bm25: frozenset[str] = frozenset(),
    query_bm25: frozenset[str] = frozenset(),
    match_df: list[int] | None = None,
    bm25_df: dict[str, int] | None = None,
) -> list[dict]:
    """Sample corpus terms that cannot touch the document's BM25 score,
    **matched to the treatment arm on support bin**.

    One control is drawn per entry of ``match_bins``, from that same bin. This
    is the identification fix: with an unmatched control pool the two arms
    differ simultaneously in whether the term is in the document *and* in how
    common the term is, and support is known to move the outcome strongly. A
    bin-matched control isolates the first difference.

    High-support bins are hard to match - a word in 30% of documents is likely
    to be in this document too, and would then fail the BM25-absence test. When
    a bin cannot be filled the search widens to the nearest bins and
    ``control_bin_matched`` records that it did, so the analysis can drop or
    reweight those rows rather than silently treating them as matched.

    Absence is decided with **the BM25 tokeniser itself** - same stopword list,
    same stemmer, same regex as the index - rather than with an approximation.
    Anything less exact has failed twice here:

    * surface-form matching let model/models and fruit/fruits through
      (2.7% of control terms);
    * stem matching over scikit-learn's stopword list still let "having"
      through, because sklearn strips 318 words and bm25s only 33, so ordinary
      indexed terms were invisible to the filter (1 in 4500).

    A control term that survives this check provably contributes zero to the
    target document's BM25 score, which is the invariant the whole
    treatment-versus-control contrast rests on.
    """
    if match_df is not None:
        forbidden = doc_bm25 | query_bm25
        available = [t for t in vs.control_pool
                     if t not in doc_tokens and t not in query_tokens
                     and not (bm25_terms(t) & forbidden)]
        out = []
        taken: set[str] = set()
        for pair_slot, wanted in enumerate(match_df):
            def indexed_df(t):
                bt = bm25_terms(t)
                return (bm25_df or {}).get(next(iter(bt)), 0) if len(bt) == 1 else 0
            candidates = [t for t in available if t not in taken and wanted > 0
                          and indexed_df(t) > 0
                          and abs(np.log(indexed_df(t)) - np.log(wanted)) <= 0.1]
            if not candidates:
                continue
            distances = np.array([abs(np.log(indexed_df(t)) - np.log(wanted))
                                  for t in candidates])
            best = np.flatnonzero(distances == distances.min())
            term = candidates[int(best[int(rng.integers(len(best)))])]
            taken.add(term)
            out.append({
                "term": term, "support_bin": vs.bin_of(term),
                "pair_slot": pair_slot,
                "support_bin_requested": vs.bin_of(term),
                "control_bin_matched": True,
                "df_match_target": int(wanted),
                "term_bm25_df": int(indexed_df(term)),
                "df_match_log_distance": float(abs(np.log(indexed_df(term)) - np.log(wanted))),
                "select_prob": float(1.0 / len(best)), "term_lift": 0.0,
                "n_candidate_terms": len(candidates), "n_candidate_bins": 3,
            })
        return out
    if match_bins is None:
        match_bins = [int(rng.integers(vs.n_bins)) for _ in range(n)]
    forbidden = doc_bm25 | query_bm25
    taken: set[str] = set()
    out: list[dict] = []

    def _try(bin_idx: int) -> str | None:
        pool = vs.control_pool_by_bin[bin_idx]
        if not pool:
            return None
        for _ in range(200):
            t = pool[int(rng.integers(len(pool)))]
            if t in taken or t in doc_tokens or t in query_tokens:
                continue
            if bm25_terms(t) & forbidden:
                continue
            return t
        return None

    for want in match_bins:
        term, got, matched = None, want, True
        # widen outward: want, want-1, want+1, want-2, want+2, ...
        for step in range(vs.n_bins):
            for cand_bin in ({want} if step == 0 else {want - step, want + step}):
                if not 0 <= cand_bin < vs.n_bins:
                    continue
                term = _try(int(cand_bin))
                if term is not None:
                    got, matched = int(cand_bin), (cand_bin == want)
                    break
            if term is not None:
                break
        if term is None:
            continue
        taken.add(term)
        pool_n = len(vs.control_pool_by_bin[got])
        out.append({
            "term": term,
            "support_bin": got,
            "support_bin_requested": int(want),
            "control_bin_matched": bool(matched),
            "select_prob": float(1.0 / max(1, pool_n)),
            "term_lift": 0.0,  # absent from the document by construction
            "n_candidate_terms": pool_n,
            "n_candidate_bins": vs.n_bins,
        })
    return out


# --------------------------------------------------------------------------- #
# do() provenance
# --------------------------------------------------------------------------- #
#: Name of the intervention operator. Recorded explicitly so that later
#: operators (permute, delete, replace) are distinguishable in the same table
#: rather than being inferred from which columns happen to be populated.
OPERATOR = "append_term"


def term_provenance(
    corpus: Corpus,
    doc_id: str,
    draw: dict,
    vs: VocabStats,
    arm: str,
) -> dict:
    """Everything about *why this word* was injected into *this query*.

    ``draw`` is the record the sampler returned, so ``select_prob`` and
    ``support_bin`` are the sampler's own values rather than a reconstruction.
    The previous version recomputed them from the same inputs, which was
    correct only for as long as the two code paths stayed in step; returning
    them directly removes that failure mode entirely.

    Recording only ``(doc_id, term)`` would make the published log unauditable:
    a reader could not tell whether a term was the document's most distinctive
    word or its least, how common it is in the corpus, nor how likely it was to
    be drawn at all.
    """
    term = draw["term"]
    i = corpus.idx(doc_id)
    tf_all = Counter(t for t in content_tokens(corpus.texts[i]) if _is_word(t))
    df = vs.df.get(term, 0)
    b = int(draw.get("support_bin", -1))
    lo, hi = (float(vs.bin_edges[b]), float(vs.bin_edges[b + 1])) if b >= 0 else (float("nan"),) * 2
    return {
        "operator": OPERATOR,
        "sampler": "support_lift_stratified",
        "term_source": "target_document" if arm == "treatment" else "corpus_vocabulary",
        "term_tf_in_doc": int(tf_all.get(term, 0)),
        "term_df_corpus": int(df),
        "term_bm25_df_corpus": int(draw.get("term_bm25_df", df)),
        "term_cf_corpus": int(vs.cf.get(term, 0)),
        # Kept under the historical name (it holds a *fraction*, not a percent)
        # so old analyses keep working; term_support is the correctly-named one.
        "term_doc_freq_pct": float(vs.support(term)),
        "term_support": float(vs.support(term)),
        "term_lift": float(draw.get("term_lift", float("nan"))),
        "support_bin": b,
        "support_bin_lo": lo,
        "support_bin_hi": hi,
        "support_bin_requested": int(draw.get("support_bin_requested", b)),
        "support_band": str(draw.get("support_band", "matched_control")),
        "control_bin_matched": bool(draw.get("control_bin_matched", True)),
        "df_match_target": int(draw.get("df_match_target", draw.get("term_bm25_df", 0))),
        "df_match_log_distance": float(draw.get("df_match_log_distance", 0.0)),
        "term_idf": float(vs.idf(term)),
        "term_tfidf_weight": float(tf_all.get(term, 0) * vs.idf(term)),
        "term_in_title": bool(term in set(content_tokens(corpus.titles[i]))),
        "term_bm25_form": " ".join(sorted(bm25_terms(term))) or "<stopword/dropped>",
        "injection_position": "append",
        "select_prob": float(draw.get("select_prob", float("nan"))),
        "n_candidate_terms": int(draw.get("n_candidate_terms", 0)),
        "n_candidate_bins": int(draw.get("n_candidate_bins", vs.n_bins)),
        # Exactly the number of documents whose BM25 score this injection
        # changes: appending t moves every document containing t and no other.
        # The ex-ante interference footprint of the operator.
        "pred_docs_moved": int(df),
    }


def origin_documents(corpus: Corpus, doc_ids: list[str], snippet_chars: int = 400) -> pd.DataFrame:
    """The documents that treatment terms were drawn from.

    Published alongside the intervention log so the record is self-contained:
    a reader can see *which* paper a word like "anterior" came from without
    re-downloading and re-parsing the collection. Text is truncated to a
    snippet - enough to identify and audit the document, short of
    redistributing the corpus.
    """
    rows = []
    for d in sorted(set(doc_ids)):
        i = corpus.idx(d)
        rows.append({
            "doc_id": d,
            "title": corpus.titles[i],
            "text_snippet": corpus.texts[i][:snippet_chars],
            "text_chars": len(corpus.texts[i]),
            "doc_len_tokens": int(corpus.doc_len[i]),
            "n_content_tokens": len(corpus.doc_content_tokens[i]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Target selection
# --------------------------------------------------------------------------- #
def select_targets(pipe: RetrievalPipeline, query: str, relevant: set[str],
                   rng: np.random.Generator, return_frame: bool = False) -> list[dict]:
    """Population-aware sample from every mapped judged-relevant document.

    Eligibility is membership of the pool, not of the reranked top-K. Requiring
    a non-censored baseline rank would restrict targets to documents the
    pipeline already ranks well, which conditions the whole experiment on
    retrieval success and mechanically concentrates the measured effect in the
    reranker path. Targets whose baseline rank is censored are kept and flagged
    (``base_censored``) so the analysis can condition on it instead.

    The sample is drawn with a per-query RNG rather than taking the best-ranked
    documents, which would leave no headroom for a positive effect.
    """
    eligible = sorted(d for d in relevant if d in pipe.corpus.doc_index)
    if not eligible:
        return []
    bm = pipe._bm25_array(query)
    de = pipe._dense_array(query)
    idx = np.arange(len(pipe.corpus))

    def rank(arr: np.ndarray, di: int) -> int:
        return 1 + int((arr > arr[di]).sum()) + int(((arr == arr[di]) & (idx < di)).sum())

    records = []
    for doc_id in eligible:
        di = pipe.corpus.idx(doc_id)
        rb, rd = rank(bm, di), rank(de, di)
        h = config.K_CANDIDATES - min(rb, rd)
        if h >= 10:
            stratum = "deep_in"
        elif h >= 0:
            stratum = "boundary_in"
        elif h >= -10:
            stratum = "boundary_out"
        elif h >= -50:
            stratum = "mid_out"
        else:
            stratum = "deep_out"
        records.append({"doc_id": doc_id, "bm25_rank_corpus": rb,
                        "dense_rank_corpus": rd, "hybrid_rank_margin": h,
                        "target_stratum": stratum})
    if len(eligible) <= config.MAX_TARGET_DOCS_PER_QUERY:
        return [{**r, "target_selected": True, "target_select_prob": 1.0,
                 "target_population_n": len(eligible)} for r in records]
    framed = []
    for stratum in ("deep_in", "boundary_in", "boundary_out", "mid_out", "deep_out"):
        pool = [r for r in records if r["target_stratum"] == stratum]
        if not pool:
            continue
        take = min(2, len(pool))
        chosen = rng.choice(len(pool), size=take, replace=False)
        chosen_set = {int(j) for j in chosen}
        framed.extend({**r, "target_selected": j in chosen_set,
                       "target_select_prob": take / len(pool),
                       "target_population_n": len(eligible)}
                      for j, r in enumerate(pool))
    return framed if return_frame else [r for r in framed if r["target_selected"]]


def build_target_sampling_frame(pipe: RetrievalPipeline, queries: Queries,
                                qids: list[str]) -> pd.DataFrame:
    """Complete qrel target population and its realized stratified sample."""
    rows = []
    for qid in qids:
        rng = np.random.default_rng(config.stable_seed(qid))
        for r in select_targets(pipe, queries.texts[qid], queries.relevant(qid), rng,
                                return_frame=True):
            rows.append({"query_id": qid, **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The intervention loop
# --------------------------------------------------------------------------- #
def run_interventions(
    pipe: RetrievalPipeline,
    queries: Queries,
    baseline: dict[str, BaselineRun],
    qids: list[str],
) -> pd.DataFrame:
    """Apply every do(Q -> Q + t) and record the induced change in the outcome."""
    corpus = pipe.corpus
    vs = build_vocab_stats(corpus)
    indexed_df = build_bm25_df(corpus)
    print(f"[interventions] vocab: {len(vs.df)} terms, control pool {len(vs.control_pool)}")

    rows: list[dict] = []
    t0 = time.time()
    for n, qid in enumerate(qids, 1):
        run = baseline[qid]
        q0 = run.query_text
        qtok = set(content_tokens(q0))
        qbm25 = bm25_terms(q0)
        rel = queries.relevant(qid)
        # Per-query RNG: reproducible, independent of iteration order, and
        # identical whether this query is processed by a single-process run or
        # by one worker of a sharded multi-GPU run.
        rng = np.random.default_rng(config.stable_seed(qid))
        targets = select_targets(pipe, q0, rel, rng)
        bm25_full0 = pipe._bm25_array(q0)
        dense_full0 = pipe._dense_array(q0)

        for target in targets:
            doc_id = target["doc_id"]
            base_rank = run.ranks.get(doc_id, config.MISSING_RANK)
            base_ce = dict(run.reranked).get(doc_id, float("nan"))
            di = corpus.idx(doc_id)
            base_bm25 = float(bm25_full0[di])
            base_dense = float(dense_full0[di])
            cov = pipe.covariates(q0, doc_id)
            prov = run.provenance.get(doc_id, "none")

            dtok = corpus.doc_content_tokens[corpus.idx(doc_id)]
            dbm25 = bm25_terms(corpus.texts[corpus.idx(doc_id)])
            # Treatment first: its support bins define what the control arm
            # must be matched to, so the two arms differ only in whether the
            # term came from the document.
            treat = sample_treatment_terms(
                corpus, doc_id, qtok, vs, rng, query_bm25=qbm25,
                bm25_df=indexed_df,
            )
            for pair_slot, draw in enumerate(treat):
                draw["pair_slot"] = pair_slot
            controls = sample_control_terms(
                dtok, qtok, vs, rng,
                match_bins=[d["support_bin"] for d in treat],
                doc_bm25=dbm25, query_bm25=qbm25,
                match_df=[indexed_df[next(iter(bm25_terms(d["term"])))] for d in treat],
                bm25_df=indexed_df,
            )
            matched_slots = {d["pair_slot"] for d in controls}
            for draw in treat:
                draw["control_match_status"] = (
                    "matched" if draw["pair_slot"] in matched_slots
                    else "no_control_within_log_df_caliper"
                )
            for draw in controls:
                draw["control_match_status"] = "matched"
            arms = {
                "treatment": treat,
                "control": controls,
            }
            for arm, draws in arms.items():
                for draw in draws:
                    term = draw["term"]
                    q1 = f"{q0} {term}"
                    bm25_new = pipe._bm25_array(q1)
                    dense_new = pipe._dense_array(q1)
                    candidates_new, provenance_new = pipe.candidates(bm25_new, dense_new)
                    new_rank = config.MISSING_RANK
                    new_ce = None
                    prov_term = term_provenance(corpus, doc_id, draw, vs, arm)
                    rows.append(
                        {
                            "query_id": qid,
                            "doc_id": doc_id,
                            "term": term,
                            "arm": arm,
                            "pair_slot": int(draw.get("pair_slot", -1)),
                            "control_match_status": str(draw.get("control_match_status", "legacy")),
                            **target,
                            "analysis_weight": 1.0 / target["target_select_prob"],
                            **prov_term,
                            "query_text": q0,
                            "injected_query": q1,
                            "base_rank": base_rank,
                            "new_rank": new_rank,
                            # Baseline already at the censoring sentinel: this
                            # target's Delta-rank can only move in one direction.
                            "base_censored": base_rank >= config.MISSING_RANK,
                            # Positive = document moved *up* the ranking.
                            "delta_rank": base_rank - new_rank,
                            "base_ce": base_ce,
                            "new_ce": new_ce if new_ce is not None else float("nan"),
                            "delta_ce": (new_ce - base_ce)
                            if new_ce is not None
                            else float("nan"),
                            "base_bm25": base_bm25,
                            "new_bm25": float(bm25_new[di]),
                            "delta_bm25": float(bm25_new[di]) - base_bm25,
                            "base_dense": base_dense,
                            "new_dense": float(dense_new[di]),
                            "delta_dense": float(dense_new[di]) - base_dense,
                            "in_candidates": doc_id in provenance_new,
                            "provenance_base": prov,
                            "provenance_new": provenance_new.get(doc_id, "none"),
                            **cov,
                        }
                    )
        if n % 10 == 0 or n == len(qids):
            el = time.time() - t0
            print(
                f"[interventions] {n}/{len(qids)} queries, {len(rows)} rows, "
                f"{el:.0f}s ({el / n:.2f}s/query), "
                f"CE pairs={pipe.ce_pairs_scored} hits={pipe.ce_cache_hits}",
                flush=True,
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def cluster_bootstrap(
    df: pd.DataFrame,
    value_col: str,
    cluster_col: str = "query_id",
    stat=np.mean,
    n_boot: int = config.N_BOOTSTRAP,
    seed: int = config.SEED,
) -> tuple[float, float, float]:
    """Cluster bootstrap over queries.

    Rows sharing a query are not independent (same baseline, overlapping
    candidate pools), so resampling rows would understate the standard error.
    We resample whole queries with replacement instead.
    """
    rng = np.random.default_rng(seed)
    groups = [g[value_col].to_numpy(dtype=float) for _, g in df.groupby(cluster_col, sort=True)]
    groups = [g[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if len(g)]
    if not groups:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(np.concatenate(groups)))
    n = len(groups)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        boots[b] = stat(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def cluster_bootstrap_diff(
    df: pd.DataFrame,
    value_col: str,
    arm_col: str = "arm",
    a: str = "treatment",
    b: str = "control",
    cluster_col: str = "query_id",
    absolute: bool = False,
    n_boot: int = config.N_BOOTSTRAP,
    seed: int = config.SEED,
) -> tuple[float, float, float]:
    """Bootstrap CI for ``mean(arm a) - mean(arm b)``, resampling queries.

    Both arms are resampled *together* within a query, preserving the paired
    structure of the design.
    """
    rng = np.random.default_rng(seed + 1)
    work = df.copy()
    v = work[value_col].to_numpy(dtype=float)
    work["_v"] = np.abs(v) if absolute else v
    grouped: list[tuple[np.ndarray, np.ndarray]] = []
    for _, g in work.groupby(cluster_col, sort=True):
        ga = g.loc[g[arm_col] == a, "_v"].to_numpy(dtype=float)
        gb = g.loc[g[arm_col] == b, "_v"].to_numpy(dtype=float)
        ga, gb = ga[~np.isnan(ga)], gb[~np.isnan(gb)]
        if len(ga) and len(gb):
            grouped.append((ga, gb))
    if not grouped:
        return float("nan"), float("nan"), float("nan")

    def diff(sel: list[int]) -> float:
        A = np.concatenate([grouped[i][0] for i in sel])
        B = np.concatenate([grouped[i][1] for i in sel])
        return float(A.mean() - B.mean())

    point = diff(list(range(len(grouped))))
    boots = np.array([diff(list(rng.integers(0, len(grouped), size=len(grouped)))) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Arm-level effect sizes with cluster-bootstrap 95% CIs."""
    recs = []
    for arm in ("treatment", "control"):
        sub = df[df["arm"] == arm]
        for col, label, absolute in (
            ("delta_rank", "mean Delta-rank", False),
            ("delta_rank", "mean |Delta-rank|", True),
            ("delta_ce", "mean Delta-CE", False),
            ("delta_bm25", "mean Delta-BM25", False),
            ("delta_dense", "mean Delta-dense", False),
        ):
            s = sub.copy()
            if absolute:
                s = s.assign(**{col: s[col].abs()})
            pt, lo, hi = cluster_bootstrap(s, col)
            recs.append(
                {"arm": arm, "metric": label, "estimate": pt, "ci_lo": lo, "ci_hi": hi, "n": len(sub)}
            )
    return pd.DataFrame(recs)


def print_summary(df: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Print the arm comparison and return the treatment-vs-control contrasts."""
    summ = summarize(df)
    print("\n" + "=" * 78)
    print("INTERVENTION EFFECTS (cluster-bootstrapped over queries, 95% CI)")
    print("=" * 78)
    print(f"{'arm':<11}{'metric':<20}{'estimate':>12}{'95% CI':>26}{'n':>8}")
    for _, r in summ.iterrows():
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        print(f"{r['arm']:<11}{r['metric']:<20}{r['estimate']:>12.4f}{ci:>26}{r['n']:>8}")

    contrasts: dict[str, tuple[float, float, float]] = {}
    print("-" * 78)
    print("Treatment - control contrasts:")
    for col, label, absolute in (
        ("delta_rank", "Delta-rank", False),
        ("delta_rank", "|Delta-rank|", True),
        ("delta_ce", "Delta-CE", False),
    ):
        pt, lo, hi = cluster_bootstrap_diff(df, col, absolute=absolute)
        key = f"{'abs_' if absolute else ''}{col}"
        contrasts[key] = (pt, lo, hi)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {label:<14} {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}")
    print("=" * 78 + "\n")
    return contrasts


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot_effects(df: pd.DataFrame, path=FIG_EFFECTS) -> None:
    """Histogram of Delta-rank by arm, plus the Delta-CE distribution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = {"treatment": "#2c7fb8", "control": "#d95f0e"}

    ax = axes[0]
    lo = int(df["delta_rank"].min())
    hi = int(df["delta_rank"].max())
    bins = np.arange(lo - 0.5, hi + 1.5, max(1, (hi - lo) // 40))
    for arm in ("control", "treatment"):
        s = df.loc[df["arm"] == arm, "delta_rank"]
        ax.hist(s, bins=bins, alpha=0.6, label=f"{arm} (n={len(s)})", color=colors[arm])
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Delta-rank  (baseline - post-intervention; >0 = moved up)")
    ax.set_ylabel("count")
    ax.set_title("Effect of do(Q := Q + term) on target-doc rank")
    ax.legend()

    ax = axes[1]
    for arm in ("control", "treatment"):
        s = df.loc[df["arm"] == arm, "delta_ce"].dropna()
        ax.hist(s, bins=50, alpha=0.6, label=arm, color=colors[arm])
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Delta cross-encoder score")
    ax.set_ylabel("count")
    ax.set_title("Effect on reranker score")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[interventions] wrote {path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    config.set_seeds()
    t0 = time.time()
    corpus, queries = load_corpus_and_queries()
    pipe = RetrievalPipeline(corpus)
    qids = select_queries(queries)
    baseline = compute_baseline(pipe, queries, qids)
    df = run_interventions(pipe, queries, baseline, qids)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"[interventions] wrote {OUT_PARQUET}  ({len(df)} rows)")
    print_summary(df)
    plot_effects(df)
    print(f"[interventions] wall clock {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
