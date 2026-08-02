"""Module 7 (trial) - model whether a document is *retrieved at all*.

Modules 2-3 measure where a document lands **given** it is in the candidate
pool. This module measures whether it is in the pool, which is the question a
RAG system actually turns on: no amount of reranking recovers a document the
first stage never returned.

Scope is deliberately the **lexical channel only**, because there BM25 gives an
exact closed form and the whole analysis can be validated rather than fitted::

    A = 1{d in TopK(BM25)}          the outcome
    delta(t, d') = IDF(t) * tf(t,d')(k1+1) / (tf(t,d') + k1(1 - b + b|d'|/avgdl))

Appending a term ``t`` to a query changes the BM25 score of *exactly* the
``df(t)`` documents containing ``t``, by that amount, and of no others. So the
whole-corpus response to a do()-operation is computable **before running the
pipeline**, and the prediction can be checked against what the pipeline
actually did. That check is `audit_predictor`, and it should come out exact -
it is arithmetic, not a model.

The interesting structure is that admission is competitive::

    A = 1{ s_d > s_(K) }

Both sides move. ``s_d`` moves by lift (how much *this* document gains);
``s_(K)`` moves because the ``df(t)`` co-treated competitors gain too, which is
governed by support. That second term is interference - it is why the target's
own score rising does not guarantee admission, and no independent-scoring
retrieval model predicts it.

This yields a clean decomposition of the two arms:

* **control** terms are absent from the target by construction, so their direct
  effect on ``s_d`` is *exactly* zero. Any admission change they cause is
  **pure interference** - competitors displacing the target.
* **treatment** terms move both. The contrast therefore identifies the direct
  effect, and the control arm alone identifies the interference effect.

The arm contrast compares two *different* words matched on support. The panel
additionally evaluates a **surgical same-word decomposition**: four worlds per
injection, from the same closed-form deltas, with the two channels switched on
and off separately::

    Y00  nobody boosted            Y10  target only
    Y01  competitors only          Y11  both  (= the real intervention)

so ``Y10-Y00`` is the word's direct benefit, ``Y01-Y00`` its interference, and
``Y11-Y10-Y01+Y00`` their interaction - with the word, its IDF, its co-treated
set and every score increment held fixed across the four worlds.

Run standalone::

    python -m admission                    # audit + panel + CATE on $DATASET
    python -m admission --no-cate          # skip the DoubleML stage
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

import config
from interventions import bm25_terms
from pipeline import RetrievalPipeline, load_corpus_and_queries

OUT_PANEL = config.RESULTS_DIR / "admission_panel.parquet"
OUT_AUDIT = config.RESULTS_DIR / "admission_audit.json"
OUT_CATE = config.RESULTS_DIR / "admission_cate.csv"
OUT_SURGICAL = config.RESULTS_DIR / "admission_surgical.csv"

#: bm25s "lucene" method, matching RetrievalPipeline's index exactly. A
#: mismatch here would silently turn an exact prediction into an approximate
#: one, so it is asserted against the live index in `audit_predictor`.
K1, B = 1.5, 0.75

POSTINGS_CACHE = config.CACHE_DIR / f"postings_{config.DATASET_TAG}_v1.pkl"


# --------------------------------------------------------------------------- #
# Corpus-side term statistics
# --------------------------------------------------------------------------- #
def _bm25_token_list(text: str) -> list[str]:
    """The token *stream* bm25s indexes for this text, counts preserved.

    ``bm25_terms`` returns a set, which loses term frequency and document
    length - both of which the scoring function needs. Going through
    ``content_tokens`` instead is wrong for a subtler reason: it applies
    scikit-learn's 318-word stoplist while the index applies bm25s' 33-word
    one, so ordinary indexed terms are invisible to it. That mismatch has
    already caused one defect in this project and it silently shifts every
    document length here by a few percent.
    """
    import bm25s

    import interventions as _I

    if _I._STEMMER is None:
        _I.stem(["x"])  # initialise the shared stemmer the index was built with
    return bm25s.tokenize(text, stopwords="en", stemmer=_I._STEMMER,
                          return_ids=False, show_progress=False)[0]


def build_postings(corpus) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """``(postings, bm25_doc_lengths)``.

    Postings map ``bm25_term -> (document indices, term frequencies)``, keyed on
    the **BM25 form** (stemmed, stopword-filtered) because that is what the
    index scores. Lengths are counted in the same space - ``corpus.doc_len``
    comes from a different tokeniser and is a few percent off, which is enough
    to break an otherwise exact prediction.
    """
    if POSTINGS_CACHE.exists():
        try:
            with open(POSTINGS_CACHE, "rb") as f:
                d = pickle.load(f)
            if d.get("n_docs") == len(corpus) and "dl" in d:
                return ({t: (np.asarray(v[0]), np.asarray(v[1], dtype=float))
                         for t, v in d["post"].items()},
                        np.asarray(d["dl"], dtype=float))
        except Exception as e:  # noqa: BLE001
            print(f"[admission] could not read postings cache ({e}) - rebuilding")

    t0 = time.time()
    acc: dict[str, list[tuple[int, float]]] = {}
    dl = np.zeros(len(corpus), dtype=float)
    for i in range(len(corpus)):
        toks = _bm25_token_list(corpus.texts[i])
        dl[i] = len(toks)
        for bt, c in Counter(toks).items():
            acc.setdefault(bt, []).append((i, float(c)))
    post = {t: (np.array([x[0] for x in v]), np.array([x[1] for x in v], dtype=float))
            for t, v in acc.items()}
    print(f"[admission] built postings for {len(post)} terms in {time.time() - t0:.1f}s "
          f"(bm25 avg doc len {dl.mean():.1f})")
    with open(POSTINGS_CACHE, "wb") as f:
        pickle.dump({"n_docs": len(corpus), "dl": dl.tolist(),
                     "post": {t: (v[0].tolist(), v[1].tolist()) for t, v in post.items()}}, f)
    return post, dl


def bm25_delta(term: str, post: dict, dl: np.ndarray, avgdl: float, n_docs: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Exact BM25 score change for every document, when ``term`` is appended.

    Returns ``(doc_indices, deltas)``. Documents not containing the term are
    omitted because their delta is *exactly* zero - that sparsity is the whole
    reason the corpus-wide response is cheap to compute.

    Uses bm25s' ``lucene`` variant, which - unlike the textbook Robertson form -
    **omits the ``(k1+1)`` numerator factor**. It is a constant multiplier so it
    cannot change a ranking, but it scales every score by 2.5 at k1=1.5, which
    is fatal to a prediction that is supposed to match the index exactly.
    """
    hit = post.get(term)
    if hit is None:
        return np.zeros(0, dtype=int), np.zeros(0)
    ids, tf = hit
    idf = np.log(1.0 + (n_docs - len(ids) + 0.5) / (len(ids) + 0.5))
    d = idf * tf / (tf + K1 * (1.0 - B + B * dl[ids] / avgdl))
    return ids, d


# --------------------------------------------------------------------------- #
# The competitive outcome
# --------------------------------------------------------------------------- #
def boundary(scores: np.ndarray, k: int = config.K_CANDIDATES) -> dict[str, float]:
    """Where the top-K cut sits, and how crowded it is.

    ``cut`` is the K-th largest score - the bar a document must clear. The
    density terms say how many competitors sit just under it, which is what
    decides whether a perturbation reshuffles membership at all. Normalising
    support by K/N failed to explain cross-collection differences in churn;
    boundary crowding is the missing third input.
    """
    part = np.partition(scores, -(k + 1))[-(k + 1):]
    top = np.sort(part)[::-1]
    cut = float(top[k - 1])
    gap = float(top[k - 1] - top[k])
    band = max(cut * 0.05, 1e-9)
    return {
        "cut": cut,
        "cut_gap": gap,
        "n_within_5pct_below": int(((scores >= cut - band) & (scores < cut)).sum()),
        "n_nonzero": int((scores > 0).sum()),
    }


def admitted(scores: np.ndarray, doc_idx: int, k: int = config.K_CANDIDATES) -> bool:
    """Is this document in the top-K by these scores?"""
    return bool(scores[doc_idx] >= np.partition(scores, -k)[-k])


def predict_admission(base: np.ndarray, ids: np.ndarray, deltas: np.ndarray,
                      doc_idx: int, k: int = config.K_CANDIDATES) -> dict:
    """Predict, without running the pipeline, whether ``doc_idx`` is admitted.

    Applies the closed-form delta to every co-treated document, then re-reads
    the order statistic. The point of separating this from the pipeline is that
    the two must agree exactly; where they disagree, the closed form is wrong.
    """
    new = base.copy()
    new[ids] += deltas
    top_old = set(np.argpartition(base, -k)[-k:].tolist())
    top_new = set(np.argpartition(new, -k)[-k:].tolist())
    return {
        "pred_admitted": doc_idx in top_new,
        "pred_entrants": len(top_new - top_old),
        "pred_churn": len(top_new ^ top_old),
        "pred_score_new": float(new[doc_idx]),
        "pred_margin_new": float(new[doc_idx] - np.partition(new, -k)[-k]),
        "n_cotreated": int(len(ids)),
    }


# --------------------------------------------------------------------------- #
# GPU backend for the world evaluations
# --------------------------------------------------------------------------- #
class _WorldsGPU:
    """Evaluates the four worlds and top-K stats on a CUDA device.

    fp64 throughout: additions and >=-comparisons are exact IEEE operations,
    so the GPU changes wall-clock, not results (injected-term postings have
    unique doc indices, so the scatter-add has no nondeterministic collisions).
    Membership is decided by ``score >= K-th largest``, matching `admitted`.

    Per-query state (base tensor, its K-th score, its top-K set) lives in a
    small LRU rather than an unbounded cache: intervention rows arrive grouped
    by query, and a Quora-sized fp64 base vector is ~4 MB - caching every
    query would not fit a 40 GB card.
    """

    def __init__(self, k: int):
        import torch

        self.torch = torch
        self.dev = torch.device("cuda")
        self.k = k
        self._cache: OrderedDict[str, tuple] = OrderedDict()

    def _query_state(self, key: str, base: np.ndarray) -> tuple:
        st = self._cache.get(key)
        if st is None:
            t = self.torch.as_tensor(base, dtype=self.torch.float64, device=self.dev)
            topv, topi = self.torch.topk(t, self.k)
            st = (t, float(topv[-1].item()), set(topi.tolist()))
            self._cache[key] = st
            if len(self._cache) > 8:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return st

    def evaluate(self, qkey: str, base: np.ndarray, ids: np.ndarray,
                 deltas: np.ndarray, di: int, d_target: float
                 ) -> tuple[bool, bool, bool, bool, dict]:
        torch = self.torch
        base_t, kth0, top_old = self._query_state(qkey, base)
        y00 = bool(base[di] >= kth0)

        ids_t = torch.as_tensor(ids, dtype=torch.long, device=self.dev)
        del_t = torch.as_tensor(deltas, dtype=torch.float64, device=self.dev)
        new = base_t.clone()
        new.index_put_((ids_t,), del_t, accumulate=True)
        topv, topi = torch.topk(new, self.k)
        s_new, kth11 = new[di], topv[-1]
        y11 = bool((s_new >= kth11).item())
        top_new = set(topi.tolist())
        pred = {
            "pred_admitted": y11,
            "pred_entrants": len(top_new - top_old),
            "pred_churn": len(top_new ^ top_old),
            "pred_score_new": float(s_new.item()),
            "pred_margin_new": float((s_new - kth11).item()),
            "n_cotreated": int(len(ids)),
        }
        if d_target:
            new[di] -= d_target                       # competitors only
            kth01 = torch.topk(new, self.k).values[-1]
            y01 = bool((new[di] >= kth01).item())
            v = base_t.clone()                        # target only
            v[di] += d_target
            kth10 = torch.topk(v, self.k).values[-1]
            y10 = bool((v[di] >= kth10).item())
        else:
            # control: the target's delta is exactly 0, so the mixed worlds
            # coincide with the pure ones - no extra kernels needed.
            y01, y10 = y11, y00
        return y00, y10, y01, y11, pred


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
def build_admission_panel(pipe: RetrievalPipeline, inter: pd.DataFrame,
                          limit: int | None = None) -> pd.DataFrame:
    """One row per logged do()-operation, with lift, support, boundary state,
    the closed-form prediction and the pipeline's actual answer.

    Built from an existing ``interventions.parquet``, so it can be run against
    artefacts that already exist rather than requiring a fresh GPU run.
    """
    corpus = pipe.corpus
    post, dl = build_postings(corpus)
    avgdl = float(dl.mean())
    n_docs = len(corpus)
    k = config.K_CANDIDATES
    total_tokens = float(dl.sum())

    rows: list[dict] = []
    base_cache: dict[str, np.ndarray] = {}
    bnd_cache: dict[str, dict] = {}
    if limit:
        inter = inter.head(limit)

    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = _WorldsGPU(k)
            print(f"[admission] world evaluations on "
                  f"{torch.cuda.get_device_name(0)} (fp64)")
    except Exception as e:  # noqa: BLE001 - GPU is an accelerator, never a requirement
        print(f"[admission] GPU backend unavailable ({e}) - using numpy")

    t0 = time.time()
    for n, r in enumerate(inter.itertuples(), 1):
        q0, term = r.query_text, r.term
        if q0 not in base_cache:
            base_cache[q0] = np.asarray(pipe.bm25.get_scores(sorted(bm25_terms(q0))))
        base = base_cache[q0]
        try:
            di = corpus.idx(r.doc_id)
        except Exception:
            continue

        bt = sorted(bm25_terms(term))
        if len(bt) != 1:
            continue  # multi-stem or stopword-only injection: not a clean single-term do()
        ids, deltas = bm25_delta(bt[0], post, dl, avgdl, n_docs)
        hit_d = np.where(ids == di)[0]
        d_target = float(deltas[hit_d[0]]) if len(hit_d) else 0.0
        bnd = bnd_cache.get(q0)
        if bnd is None:
            bnd = bnd_cache[q0] = boundary(base, k)  # depends on base only

        # --- surgical same-word decomposition (four worlds) ------------------
        # Same term, same deltas; only who receives the boost changes, so the
        # contrast is within-word rather than across two matched words. All
        # four worlds use the same >= K-th-value membership rule so tie
        # handling is identical. For a control term the target delta is
        # exactly 0, hence Y10 == Y00 and Y11 == Y01 by construction.
        if gpu is not None:
            y00, y10, y01, y11, pred = gpu.evaluate(q0, base, ids, deltas, di, d_target)
        else:
            pred = predict_admission(base, ids, deltas, di, k)
            y00 = admitted(base, di, k)
            world = base.copy()
            world[ids] += deltas
            y11 = admitted(world, di, k)            # both channels on
            world[di] -= d_target
            y01 = admitted(world, di, k)            # competitors only
            if d_target:
                world = base.copy()
                world[di] += d_target
                y10 = admitted(world, di, k)        # target only
            else:
                y10 = y00

        # Boundary-threat count: co-treated competitors whose own boost is
        # large enough to overtake the target's *baseline* score. Support
        # counts every document the word touches; only these can displace.
        mask = ids != di
        threat = int((base[ids[mask]] + deltas[mask] > base[di]).sum())

        # Support and lift in BM25 space, so the model's coordinates are the
        # scoring function's own rather than a differently-tokenised proxy.
        t_ids, t_tf = post.get(bt[0], (np.zeros(0, int), np.zeros(0)))
        support = len(t_ids) / n_docs
        hit = np.where(t_ids == di)[0]
        tf_in_doc = float(t_tf[hit[0]]) if len(hit) else 0.0
        cf = float(t_tf.sum())
        p_corpus = cf / total_tokens if total_tokens else 0.0
        lift = (tf_in_doc / dl[di]) / p_corpus if (p_corpus > 0 and dl[di] > 0) else 0.0

        rows.append({
            "query_id": r.query_id, "doc_id": r.doc_id, "term": term, "arm": r.arm,
            "select_prob": float(getattr(r, "select_prob", np.nan)),
            # --- treatment coordinates -------------------------------------
            "support": float(support),
            "lift": float(lift),
            "idf": float(np.log(1.0 + (n_docs - len(t_ids) + 0.5) / (len(t_ids) + 0.5))),
            # --- pre-treatment state ---------------------------------------
            "base_score": float(base[di]),
            "base_margin": float(base[di] - bnd["cut"]),
            "base_admitted": bool(y00),
            **{f"boundary_{kk}": v for kk, v in bnd.items()},
            # --- closed-form prediction ------------------------------------
            **pred,
            "direct_delta": d_target,
            # --- surgical worlds: channels switched separately --------------
            "y00": bool(y00), "y10": bool(y10), "y01": bool(y01), "y11": bool(y11),
            "surg_direct": int(y10) - int(y00),
            "surg_interference": int(y01) - int(y00),
            "surg_total": int(y11) - int(y00),
            "surg_interaction": int(y11) - int(y10) - int(y01) + int(y00),
            "threat_count": threat,
        })
        if n % 500 == 0:
            print(f"[admission] {n}/{len(inter)} rows, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    print(f"[admission] panel: {len(df)} rows over {df['query_id'].nunique()} queries")
    return df


# --------------------------------------------------------------------------- #
# Validation: the closed form must reproduce the pipeline exactly
# --------------------------------------------------------------------------- #
def audit_predictor(pipe: RetrievalPipeline, inter: pd.DataFrame, n: int = 200) -> dict:
    """Run the pipeline on the injected queries and compare with the closed form.

    This is the load-bearing check of the whole module. BM25 is additive over
    query terms, so appending a term must shift scores by exactly the closed
    form - agreement is arithmetic, not empirical. Any mismatch means the
    postings, the tokenisation or the k1/b constants disagree with the live
    index, and every downstream number would inherit that error.
    """
    corpus = pipe.corpus
    post, dl = build_postings(corpus)
    avgdl, n_docs, k = float(dl.mean()), len(corpus), config.K_CANDIDATES

    worst_score, max_score, adm_agree, checked = 0.0, 0.0, 0, 0
    sub = inter.drop_duplicates(subset=["query_id", "term"]).head(n)
    for r in sub.itertuples():
        bt = sorted(bm25_terms(r.term))
        if len(bt) != 1 or bt[0] in bm25_terms(r.query_text):
            continue
        base = np.asarray(pipe.bm25.get_scores(sorted(bm25_terms(r.query_text))))
        truth = np.asarray(pipe.bm25.get_scores(sorted(bm25_terms(r.injected_query))))
        ids, deltas = bm25_delta(bt[0], post, dl, avgdl, n_docs)
        pred = base.copy()
        pred[ids] += deltas
        worst_score = max(worst_score, float(np.abs(pred - truth).max()))
        max_score = max(max_score, float(np.abs(truth).max()))
        try:
            di = corpus.idx(r.doc_id)
        except Exception:
            continue
        adm_agree += int(admitted(pred, di, k) == admitted(truth, di, k))
        checked += 1

    # The index stores float32, so the best achievable agreement is a few ulps
    # of the *largest score in play* - an absolute cutoff flags fp32 rounding
    # as "inexact" on any collection whose BM25 scores exceed ~8 (every error
    # ever observed here is exactly 1 ulp of the max score). 8 ulps is still
    # orders of magnitude below the defects this audit exists to catch (a 2.5x
    # scale factor, percent-level document-length errors).
    tol = 8.0 * float(np.finfo(np.float32).eps) * max(1.0, max_score)
    out = {
        "n_checked": checked,
        "max_abs_score_error": worst_score,
        "max_abs_score": max_score,
        "tolerance": tol,
        "admission_agreement": adm_agree / max(1, checked),
        "exact": bool(worst_score <= tol),
        # Recorded because K is profile-dependent (50 on GPU, 20 on CPU) and it
        # sets both the conditioning event and the bar: a panel built under a
        # different K is not comparable and must not be joined to this one.
        "k_candidates": int(config.K_CANDIDATES),
        "dataset": config.DATASET,
    }
    print(f"[admission] predictor audit: {checked} injections, "
          f"max |score error| = {worst_score:.3e} (tol {tol:.3e}), "
          f"admission agreement = {out['admission_agreement']:.4f}")
    return out


# --------------------------------------------------------------------------- #
# Effect estimation
# --------------------------------------------------------------------------- #
def contrast_by_support(panel: pd.DataFrame) -> pd.DataFrame:
    """Direct vs interference effect on admission, by support decile.

    Control terms have ``direct_delta == 0`` by construction, so the control
    arm isolates the interference channel: any admission loss there is caused
    purely by co-treated competitors crossing the cut. The treatment-control
    contrast is then the direct effect, net of interference.
    """
    p = panel[panel["base_admitted"]].copy()
    if p.empty:
        return pd.DataFrame()
    p["bin"] = pd.qcut(p["support"], min(6, p["support"].nunique()), duplicates="drop")
    recs = []
    for b, g in p.groupby("bin", observed=True):
        t = g[g.arm == "treatment"]
        c = g[g.arm == "control"]
        if len(t) < 10 or len(c) < 10:
            continue
        recs.append({
            "support_bin": str(b),
            "median_support": float(g["support"].median()),
            "n_treat": len(t), "n_ctrl": len(c),
            "keep_treat": float(t["pred_admitted"].mean()),
            "keep_ctrl": float(c["pred_admitted"].mean()),
            "interference_effect": float(c["pred_admitted"].mean() - 1.0),
            "direct_effect": float(t["pred_admitted"].mean() - c["pred_admitted"].mean()),
            "mean_entrants": float(g["pred_entrants"].mean()),
            "mean_cotreated": float(g["n_cotreated"].mean()),
        })
    return pd.DataFrame(recs)


def surgical_by_support(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the four-world same-word decomposition.

    Complements the arm contrast: there, two *different* words are compared,
    matched on support but potentially touching differently-placed competitor
    sets; here nothing varies between the compared worlds except which channel
    is switched on. Two arithmetic facts shape the table:

    * conditional on baseline admission, ``Y10 == Y00`` always - boosting an
      admitted target cannot eject it - so in that stratum the entire benefit
      of the target's own boost surfaces as the **interaction**: the boost
      rescuing the target from displacement that interference alone would
      have caused;
    * conditional on baseline exclusion, ``Y10 - Y00`` is **pure own-lift
      rescue** - the first recall-side quantity this module produces.
    """
    if panel.empty:
        return pd.DataFrame()
    recs = []
    for (arm, adm), g0 in panel.groupby(["arm", "base_admitted"]):
        if len(g0) < 10:
            continue
        bins = pd.qcut(g0["support"], min(6, g0["support"].nunique()),
                       duplicates="drop")
        for b, g in g0.groupby(bins, observed=True):
            if len(g) < 10:
                continue
            recs.append({
                "arm": arm, "base_admitted": bool(adm),
                "support_bin": str(b),
                "median_support": float(g["support"].median()),
                "n": len(g),
                "direct_y10_y00": float(g["surg_direct"].mean()),
                "interference_y01_y00": float(g["surg_interference"].mean()),
                "total_y11_y00": float(g["surg_total"].mean()),
                "interaction": float(g["surg_interaction"].mean()),
                "mean_threat_count": float(g["threat_count"].mean()),
                "mean_cotreated": float(g["n_cotreated"].mean()),
            })
    return pd.DataFrame(recs)


#: Covariates for the arm contrast. Deliberately **excludes lift**.
#:
#: Control terms are absent from the target document by construction, so their
#: lift is exactly 0 while every treatment term has lift > 0. Lift therefore
#: *is* the treatment, not a covariate: conditioning on it drives the propensity
#: to {0,1} and destroys overlap. The symptom is spectacular rather than subtle
#: - the partially linear score divides by Var(D - E[D|X]), so a near-perfect
#: propensity returns an enormous coefficient with an enormous standard error.
#:
#: Support is also excluded from X, for the opposite reason: the sampler
#: matches the arms bin-for-bin on support, so it is balanced by design and
#: conditioning on it buys nothing. It belongs on the dose-response axis, not
#: in the adjustment set.
#:
#: ``threat_count`` is excluded too: it is a function of the injected term
#: (post-treatment), so it is a mediator of the effect, not a confounder.
#: It belongs in the structural analysis of *why* displacement happens, next
#: to support and margin - never in X.
ARM_CONTRAST_X = ["base_margin", "boundary_cut_gap", "boundary_n_within_5pct_below",
                  "boundary_n_nonzero"]


def _overlap_ok(p: pd.DataFrame, x_cols: list[str]) -> tuple[bool, float]:
    """Can the arm be predicted from X? If yes, the contrast is not identified.

    Cheap guard against the failure above: fit the propensity directly and look
    at how extreme it gets. Positivity is an assumption that fails loudly here
    if you let it, and silently if you do not check.
    """
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import cross_val_predict

    m = LGBMClassifier(n_estimators=100, num_leaves=15, verbose=-1,
                       random_state=config.SEED, n_jobs=1, deterministic=True)
    ps = cross_val_predict(m, p[x_cols], p["D"], cv=5, method="predict_proba")[:, 1]
    worst = float(max(ps.max(), 1.0 - ps.min()))
    return worst < 0.95, worst


def fit_cate(panel: pd.DataFrame) -> pd.DataFrame:
    """DoubleML for the admission effect of a document-derived term, clustered
    by query.

    The propensity is **known by design** - the sampler assigned the arm 1:1
    within each support bin, so E[D|X] = 0.5 rather than something to estimate.
    The machine learner is therefore responsible only for the outcome
    regression, which is where it helps (variance, not bias).

    Clustering is not optional: ~34-90 documents share each query and i.i.d.
    standard errors on this panel were measured 1.55-1.81x too small.
    """
    import doubleml as dml
    from lightgbm import LGBMRegressor

    p = panel[panel["base_admitted"]].dropna(subset=ARM_CONTRAST_X).copy()
    p["D"] = (p["arm"] == "treatment").astype(float)
    p["Y"] = p["pred_admitted"].astype(float)
    if p["Y"].nunique() < 2 or p["D"].nunique() < 2 or len(p) < 50:
        print("[admission] insufficient variation for a CATE fit - skipping")
        return pd.DataFrame()

    ok, worst = _overlap_ok(p, ARM_CONTRAST_X)
    print(f"[admission] overlap check: worst cross-fitted propensity = {worst:.3f} "
          f"({'OK' if ok else 'DEGENERATE - contrast not identified'})")
    if not ok:
        return pd.DataFrame()

    common = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1,
                  random_state=config.SEED, n_jobs=config.DML_N_JOBS,
                  deterministic=True, force_row_wise=True)
    frame = p[["Y", "D"] + ARM_CONTRAST_X].astype(float).copy()
    frame["cluster"] = p["query_id"].astype("category").cat.codes.to_numpy()
    try:
        data = dml.DoubleMLClusterData(frame, y_col="Y", d_cols="D",
                                       x_cols=ARM_CONTRAST_X, cluster_cols="cluster")
        clustered = True
    except Exception as e:  # noqa: BLE001
        print(f"[admission] DoubleMLClusterData unavailable ({e}); "
              "falling back to i.i.d. - standard errors will be optimistic")
        data = dml.DoubleMLData(frame.drop(columns=["cluster"]), y_col="Y", d_cols="D",
                                x_cols=ARM_CONTRAST_X)
        clustered = False

    obj = dml.DoubleMLPLR(data, ml_l=LGBMRegressor(**common), ml_m=LGBMRegressor(**common),
                          n_folds=config.DML_N_FOLDS, n_rep=config.DML_N_REP)
    obj.fit()
    ci = obj.confint(level=0.95)
    return pd.DataFrame([{
        "n_units": len(p), "n_queries": p["query_id"].nunique(),
        "clustered": clustered,
        "coef": float(obj.coef[0]), "se": float(obj.se[0]),
        "ci_lo": float(ci.iloc[0, 0]), "ci_hi": float(ci.iloc[0, 1]),
        "pval": float(obj.pval[0]),
    }])


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="cap panel rows (smoke test)")
    ap.add_argument("--audit-n", type=int, default=200, help="injections to audit")
    ap.add_argument("--no-cate", action="store_true", help="skip the DoubleML stage")
    args = ap.parse_args()

    config.set_seeds()
    src = config.RESULTS_DIR / "interventions.parquet"
    if not src.exists():
        print(f"error: {src} not found - run `python -m run_all --only 1 2` first")
        return 1

    corpus, _ = load_corpus_and_queries()
    pipe = RetrievalPipeline(corpus)
    inter = pd.read_parquet(src)
    print(f"[admission] {len(inter)} logged interventions, dataset {config.DATASET}")

    audit = audit_predictor(pipe, inter, args.audit_n)
    import json
    OUT_AUDIT.write_text(json.dumps(audit, indent=2))
    if not audit["exact"]:
        print("[admission] WARNING: closed form does not reproduce the index exactly. "
              "Downstream numbers inherit this error - fix before interpreting.")

    panel = build_admission_panel(pipe, inter, args.limit)
    panel.to_parquet(OUT_PANEL, index=False)
    print(f"[admission] wrote {OUT_PANEL.name}")

    tab = contrast_by_support(panel)
    if not tab.empty:
        print("\nAdmission retention by support "
              "(control = pure interference; treatment - control = direct effect)")
        print(tab.to_string(index=False))

    surg = surgical_by_support(panel)
    if not surg.empty:
        surg.to_csv(OUT_SURGICAL, index=False)
        print("\nSurgical same-word decomposition "
              "(Y10=target only, Y01=competitors only, Y11=both):")
        print(surg.to_string(index=False))

    if not args.no_cate:
        res = fit_cate(panel)
        if not res.empty:
            res.to_csv(OUT_CATE, index=False)
            print("\n[admission] DoubleML admission effect:")
            print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
