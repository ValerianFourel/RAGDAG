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
    """Corpus-level lexical statistics used to choose injection terms."""

    df: dict[str, int]
    n_docs: int
    control_pool: list[str]

    def idf(self, term: str) -> float:
        return math.log(self.n_docs / (1 + self.df.get(term, 0)))


def _is_word(t: str) -> bool:
    return t.isalpha() and len(t) >= 3


def build_vocab_stats(corpus: Corpus, min_df: int = 5) -> VocabStats:
    """Document frequencies plus the pool of admissible control terms.

    Control terms are restricted to reasonably common alphabetic words so that
    the control arm is not secretly an "inject a junk token" arm - that would
    make the contrast trivially significant for the wrong reason.
    """
    df: Counter[str] = Counter()
    for toks in corpus.doc_content_tokens:
        df.update(t for t in toks if _is_word(t))
    n = len(corpus)
    pool = sorted(t for t, c in df.items() if min_df <= c <= 0.20 * n)
    return VocabStats(df=dict(df), n_docs=n, control_pool=pool)


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
) -> list[str]:
    """Sample ``n`` terms from the target document, TF-IDF weighted.

    Terms already present in the query are excluded, otherwise the
    intervention would be a no-op on the lexical channel.
    """
    terms, w = tfidf_terms(corpus, doc_id, vs)
    # Exclude against the BM25 view of the query, not just its surface tokens:
    # injecting a term the query already contributes is close to a no-op on the
    # lexical channel, diluting the treatment arm the mirror-image way the
    # stopword/stem mismatch inflated the control arm.
    keep = [
        j
        for j, t in enumerate(terms)
        if t not in query_tokens and not (bm25_terms(t) & query_bm25)
    ]
    if not keep:
        return []
    terms = [terms[j] for j in keep]
    w = w[keep]
    if w.sum() <= 0:
        w = np.ones(len(terms))
    p = w / w.sum()
    n = min(n, len(terms))
    idx = rng.choice(len(terms), size=n, replace=False, p=p)
    return [terms[j] for j in idx]


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


def sample_control_terms(
    doc_tokens: frozenset[str],
    query_tokens: set[str],
    vs: VocabStats,
    rng: np.random.Generator,
    n: int = config.N_CONTROL_TERMS,
    doc_bm25: frozenset[str] = frozenset(),
    query_bm25: frozenset[str] = frozenset(),
) -> list[str]:
    """Sample ``n`` corpus terms that cannot touch the document's BM25 score.

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
    out: list[str] = []
    forbidden = doc_bm25 | query_bm25
    pool = vs.control_pool
    for _ in range(50 * n):
        if len(out) == n:
            break
        t = pool[int(rng.integers(len(pool)))]
        if t in doc_tokens or t in query_tokens or t in out:
            continue
        if bm25_terms(t) & forbidden:
            continue
        out.append(t)
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
    term: str,
    vs: VocabStats,
    arm: str,
    query_tokens: set[str],
    query_bm25: frozenset[str],
) -> dict:
    """Everything about *why this word* was injected into *this query*.

    Computed after the fact from the same inputs the sampler saw, so it adds no
    RNG draws and cannot perturb which terms were chosen. Recording only
    ``(doc_id, term)`` would make the published log unauditable: a reader could
    not tell whether a term was the document's most distinctive word or its
    least, nor how likely it was to be drawn at all.
    """
    i = corpus.idx(doc_id)
    tf_all = Counter(t for t in content_tokens(corpus.texts[i]) if _is_word(t))
    df = vs.df.get(term, 0)
    rec = {
        "operator": OPERATOR,
        "term_source": "target_document" if arm == "treatment" else "corpus_vocabulary",
        "term_tf_in_doc": int(tf_all.get(term, 0)),
        "term_df_corpus": int(df),
        "term_doc_freq_pct": float(df / max(1, vs.n_docs)),
        "term_idf": float(vs.idf(term)),
        "term_in_title": bool(term in set(content_tokens(corpus.titles[i]))),
        "term_bm25_form": " ".join(sorted(bm25_terms(term))) or "<stopword/dropped>",
        "injection_position": "append",
        "select_prob": float("nan"),
        "n_candidate_terms": 0,
    }
    if arm == "treatment":
        # Reconstruct the sampler's candidate pool and the exact probability
        # this term carried. Same filter as sample_treatment_terms.
        terms, w = tfidf_terms(corpus, doc_id, vs)
        keep = [
            j for j, t in enumerate(terms)
            if t not in query_tokens and not (bm25_terms(t) & query_bm25)
        ]
        if keep:
            kt = [terms[j] for j in keep]
            kw = w[keep]
            if kw.sum() <= 0:
                kw = np.ones(len(kt))
            p = kw / kw.sum()
            rec["n_candidate_terms"] = len(kt)
            rec["term_tfidf_weight"] = float(kw[kt.index(term)]) if term in kt else float("nan")
            rec["select_prob"] = float(p[kt.index(term)]) if term in kt else float("nan")
        else:
            rec["term_tfidf_weight"] = float("nan")
    else:
        rec["n_candidate_terms"] = len(vs.control_pool)
        rec["term_tfidf_weight"] = float("nan")
        rec["select_prob"] = 1.0 / max(1, len(vs.control_pool))
    return rec


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
def select_targets(
    run: BaselineRun, relevant: set[str], rng: np.random.Generator
) -> list[str]:
    """Judged-relevant documents in the baseline *candidate pool*, capped at
    :data:`config.MAX_TARGET_DOCS_PER_QUERY`.

    Eligibility is membership of the pool, not of the reranked top-K. Requiring
    a non-censored baseline rank would restrict targets to documents the
    pipeline already ranks well, which conditions the whole experiment on
    retrieval success and mechanically concentrates the measured effect in the
    reranker path. Targets whose baseline rank is censored are kept and flagged
    (``base_censored``) so the analysis can condition on it instead.

    The sample is drawn with a per-query RNG rather than taking the best-ranked
    documents, which would leave no headroom for a positive effect.
    """
    eligible = sorted(set(run.candidates) & relevant)
    if len(eligible) <= config.MAX_TARGET_DOCS_PER_QUERY:
        return eligible
    idx = rng.choice(
        len(eligible), size=config.MAX_TARGET_DOCS_PER_QUERY, replace=False
    )
    return [eligible[j] for j in sorted(idx)]


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
        targets = select_targets(run, rel, rng)

        for doc_id in targets:
            base_rank = run.ranks.get(doc_id, config.MISSING_RANK)
            base_ce = dict(run.reranked).get(doc_id, float("nan"))
            base_bm25 = run.bm25_scores_cand.get(doc_id, 0.0)
            base_dense = run.dense_scores_cand.get(doc_id, 0.0)
            cov = pipe.covariates(q0, doc_id)
            prov = run.provenance.get(doc_id, "none")

            dtok = corpus.doc_content_tokens[corpus.idx(doc_id)]
            dbm25 = bm25_terms(corpus.texts[corpus.idx(doc_id)])
            arms = {
                "treatment": sample_treatment_terms(
                    corpus, doc_id, qtok, vs, rng, query_bm25=qbm25
                ),
                "control": sample_control_terms(
                    dtok, qtok, vs, rng, doc_bm25=dbm25, query_bm25=qbm25
                ),
            }
            for arm, terms in arms.items():
                for term in terms:
                    q1 = f"{q0} {term}"
                    res = pipe.run(q1)
                    new_rank = res.rank_of(doc_id)
                    di = corpus.idx(doc_id)
                    new_ce = res.ce_score_of(doc_id)
                    prov_term = term_provenance(
                        corpus, doc_id, term, vs, arm, qtok, qbm25
                    )
                    rows.append(
                        {
                            "query_id": qid,
                            "doc_id": doc_id,
                            "term": term,
                            "arm": arm,
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
                            "new_bm25": float(res.bm25_full[di]),
                            "delta_bm25": float(res.bm25_full[di]) - base_bm25,
                            "base_dense": base_dense,
                            "new_dense": float(res.dense_full[di]),
                            "delta_dense": float(res.dense_full[di]) - base_dense,
                            "in_candidates": doc_id in res.provenance,
                            "provenance_base": prov,
                            "provenance_new": res.provenance.get(doc_id, "none"),
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
