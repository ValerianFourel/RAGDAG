"""Module 8 (trial) - dense-channel admission: the same do() operator, no closed form.

Module 7 models BM25 admission exactly, because BM25 is additive and sparse.
The dense channel is neither: appending a term changes the *query embedding*,
and every document's score moves. What survives is an exact linear identity.
The production score is a dot product against fixed document embeddings::

    s_j(q) = e_d(d_j)^T e_q(q)        =>       Δs_j(t) = e_d(d_j)^T Δq_t,
    Δq_t = e_q(q ⊕ t) - e_q(q)

There is no a-priori prediction of Δq (the encoder must run), but *given* Δq
the corpus response is exact linear algebra - auditable, and cheap enough to
evaluate the same four surgical worlds as the lexical module:

    Y00  nobody moves              Y10  target's Δs only
    Y01  competitors' Δs only      Y11  every Δs (= the real intervention)

Two identifications that held for BM25 do **not** transfer, by design of the
analysis rather than by oversight:

* a control term absent from the target can still move the target's dense
  score (semantic proximity), so "control = pure interference" fails here;
  the same-word worlds are the only clean decomposition in this channel;
* there is no term-level boost cap, so the immunity certificate is the
  Cauchy-Schwarz bound |Δs_j| <= ||e_j|| ||Δq|| - valid but conservative.

Scoring facts this module replicates exactly (verified by `audit_dense`):
document embeddings are L2-normalised fp32, cached; queries are encoded with
``config.BGE_QUERY_PREFIX`` and ``normalize_embeddings=True``; the score is an
exhaustive dot product (no ANN); the production top-K tie rule is lexsort on
``(-score, doc index)``. All reference arithmetic here is fp64 over the fp32
embeddings, so the linear identity is exact to fp64 rounding; the audit also
reports how often fp32-production and fp64-reference membership disagree.

Run standalone::

    python -m dense_admission                    # audit + panel + tables
    python -m dense_admission --limit 300 --audit-n 25 --no-models
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

import config
from pipeline import RetrievalPipeline, load_corpus_and_queries

OUT_PANEL = config.RESULTS_DIR / "dense_admission_panel.parquet"
OUT_AUDIT = config.RESULTS_DIR / "dense_admission_audit.json"
OUT_BY_BIN = config.RESULTS_DIR / "dense_admission_by_bin.csv"
OUT_MODELS = config.RESULTS_DIR / "dense_admission_models.csv"
OUT_UNION = config.RESULTS_DIR / "union_gate.csv"

#: query-embedding cache - keyed on the exact text, tagged with everything that
#: changes the vector (model, sequence length, precision).
QEMB_CACHE = config.CACHE_DIR / (
    f"qemb_{config.DATASET_TAG}_{config.DENSE_MODEL.split('/')[-1]}"
    f"_L{config.MAX_SEQ_LENGTH}_{config.PRECISION_TAG}_v1.pkl"
)

#: |Δs| magnitudes worth counting (cosine units). 1e-3 is roughly the median
#: baseline margin scale on these collections; 1e-2 is a large move.
DELTA_THRESHOLDS = (1e-3, 1e-2)

_BOOTSTRAP_B = 1000


# --------------------------------------------------------------------------- #
# Membership under the production tie rule
# --------------------------------------------------------------------------- #
# pipeline._topk_ids takes exactly K documents, ordered by lexsort on
# (-score, doc index). Membership is therefore "fewer than K documents beat
# you", where j beats the target iff s_j > s_t, or s_j == s_t and j < target.
# This differs from admission.py's >=K-th-value rule (which admits all ties);
# the dense module follows the production rule the pipeline actually uses.

def beat_count(scores, s_target: float, di: int, idx) -> int:
    """Documents ranked above a target scoring ``s_target`` at index ``di``.

    The target's own slot is excluded regardless of the (possibly stale)
    value stored there - callers exploit this to evaluate hypothetical
    target scores without copying the corpus vector.
    """
    gt = scores > s_target
    eq = scores == s_target
    n = int(gt.sum()) + int((eq & (idx < di)).sum())
    if bool(scores[di] > s_target):
        n -= 1  # remove the target's own stale slot
    return n


def dense_worlds(s, ds, di: int, k: int, idx) -> dict:
    """Four-world membership for one trial. Backend-agnostic (numpy or torch
    fp64 tensors on any device); no corpus-sized copies are made."""
    d_t = float(ds[di])
    s_t0 = float(s[di])
    s_t1 = s_t0 + d_t
    s11 = s + ds
    n0 = beat_count(s, s_t0, di, idx)
    n11 = beat_count(s11, s_t1, di, idx)
    return {
        "y00": n0 < k,
        "y11": n11 < k,
        "y01": beat_count(s11, s_t0, di, idx) < k,   # competitors move, target holds
        "y10": beat_count(s, s_t1, di, idx) < k,     # target moves alone
        "n_above_00": n0,
        "n_above_11": n11,
        "target_delta": d_t,
        "_s11": s11,
    }


# --------------------------------------------------------------------------- #
# Reference scorer (fp64 over the cached fp32 embeddings)
# --------------------------------------------------------------------------- #
class DenseRef:
    """Holds E (n x d, fp64) on the fastest available device and evaluates
    scores, deltas and world metrics. torch is the backend on CPU too, so
    there is exactly one implementation of the math."""

    def __init__(self, pipe: RetrievalPipeline):
        import torch

        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.E = torch.as_tensor(pipe.doc_emb, dtype=torch.float64, device=self.dev)
        # norms of the *scoring copy* (rows are L2-normalised, so ~1.0; using
        # the measured values keeps the Cauchy-Schwarz bound honest anyway)
        self.doc_norm = self.E.norm(dim=1)
        self.idx = torch.arange(self.E.shape[0], device=self.dev)
        self.n = int(self.E.shape[0])
        self._q: dict[str, tuple] = {}
        print(f"[dense_admission] reference scorer on {self.dev} "
              f"({self.n} docs, fp64)")

    def vec(self, emb: np.ndarray):
        return self.torch.as_tensor(emb, dtype=self.torch.float64, device=self.dev)

    def scores(self, e):
        return self.E @ e

    def query_state(self, qtext: str, e0: np.ndarray, k: int) -> tuple:
        """(base scores, K-th value, e0 tensor), cached per query text."""
        st = self._q.get(qtext)
        if st is None:
            e0t = self.vec(e0)
            s = self.scores(e0t)
            cut = float(self.torch.topk(s, k).values[-1].item())
            st = (s, cut, e0t)
            self._q[qtext] = st
            if len(self._q) > 8:
                self._q.pop(next(iter(self._q)))
        return st


# --------------------------------------------------------------------------- #
# Encoding (batched, deduplicated, cached; identical to the production path)
# --------------------------------------------------------------------------- #
def encode_unique(pipe: RetrievalPipeline, texts: list[str]) -> dict[str, np.ndarray]:
    """fp32 unit query embeddings for every unique text, exactly as
    ``RetrievalPipeline._dense_array`` produces them (prefix + normalise)."""
    cache: dict[str, np.ndarray] = {}
    if QEMB_CACHE.exists():
        try:
            with open(QEMB_CACHE, "rb") as f:
                cache = pickle.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"[dense_admission] could not read {QEMB_CACHE.name} ({e}) - re-encoding")
            cache = {}
    todo = sorted({t for t in texts if t not in cache})
    if todo:
        t0 = time.time()
        vecs = pipe.dense_model.encode(
            [config.BGE_QUERY_PREFIX + t for t in todo],
            batch_size=config.EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        cache.update({t: v for t, v in zip(todo, vecs)})
        tmp = QEMB_CACHE.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(cache, f)
        tmp.replace(QEMB_CACHE)
        print(f"[dense_admission] encoded {len(todo)} unique queries "
              f"in {time.time() - t0:.1f}s")
    return cache


# --------------------------------------------------------------------------- #
# Phase 2 - the audit
# --------------------------------------------------------------------------- #
def audit_dense(pipe: RetrievalPipeline, inter: pd.DataFrame, n: int = 200) -> dict:
    """Verify the finite-perturbation identity and our replication of the
    production scoring path, on real injections.

    Three distinct comparisons, reported separately because they fail for
    different reasons:

    1. identity: s(q) + E·Δq  vs  E·e(q⊕t), both fp64 - must agree to
       rounding; a mismatch means the linear algebra is wrong (H1);
    2. production: fp64-reference membership vs the pipeline's own fp32
       single-encode path - characterises dtype/batching effects, and would
       expose ANN artefacts if there were any (there is no ANN: exhaustive);
    3. Cauchy-Schwarz: |Δs_j| <= ||e_j|| ||Δq|| for every document.
    """
    ref = DenseRef(pipe)
    torch = ref.torch
    sub = inter.drop_duplicates(subset=["query_id", "term"]).head(n)
    texts = list(sub["query_text"]) + list(sub["injected_query"])
    emb = encode_unique(pipe, texts)

    # batch-vs-single encoding drift, on a small deterministic sample
    single = pipe.dense_model.encode(
        [config.BGE_QUERY_PREFIX + t for t in sorted(set(texts))[:16]],
        batch_size=1, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    enc_drift = float(max(
        np.abs(emb[t] - single[i]).max()
        for i, t in enumerate(sorted(set(texts))[:16])
    ))

    max_abs, max_rel, cs_viol = 0.0, 0.0, 0
    topk_exact, jac_sum, adm_agree, checked = 0, 0.0, 0, 0
    k = config.K_CANDIDATES
    for r in sub.itertuples():
        e0, e1 = emb[r.query_text], emb[r.injected_query]
        e0t, e1t = ref.vec(e0), ref.vec(e1)
        dq = e1t - e0t
        pred = ref.scores(e0t) + ref.scores(dq)
        truth = ref.scores(e1t)
        err = float((pred - truth).abs().max().item())
        scale = float(truth.abs().max().item())
        max_abs = max(max_abs, err)
        max_rel = max(max_rel, err / max(scale, 1e-12))

        dqn = float(dq.norm().item())
        ds = ref.scores(dq)
        cs_viol += int((ds.abs() > ref.doc_norm * dqn * (1 + 1e-9) + 1e-12).sum().item())

        # production fp32 path vs fp64 reference, on the injected query
        prod = pipe._dense_array(r.injected_query)  # noqa: SLF001 - the audited path
        prod_t = torch.as_tensor(prod, dtype=torch.float64, device=ref.dev)
        ref_top = set(torch.topk(truth, k).indices.tolist())
        prod_top = set(torch.topk(prod_t, k).indices.tolist())
        topk_exact += int(ref_top == prod_top)
        jac_sum += len(ref_top & prod_top) / len(ref_top | prod_top)
        try:
            di = pipe.corpus.idx(r.doc_id)
        except Exception:
            continue
        adm_agree += int((di in ref_top) == (di in prod_top))
        checked += 1

    out = {
        "n_checked": checked,
        "max_abs_identity_error": max_abs,
        "max_rel_identity_error": max_rel,
        "cs_bound_violations": cs_viol,
        "enc_batch_vs_single_max_diff": enc_drift,
        "prod_topk_exact_frac": topk_exact / max(1, len(sub)),
        "prod_topk_jaccard_mean": jac_sum / max(1, len(sub)),
        "prod_admission_agreement": adm_agree / max(1, checked),
        "ann": "none (exhaustive dot product)",
        "reference_dtype": "float64 over cached float32 embeddings",
        "production_dtype": "float32",
        "exact": bool(max_rel < 1e-9 and cs_viol == 0),
        "k_candidates": int(config.K_CANDIDATES),
        "dataset": config.DATASET,
    }
    print(f"[dense_admission] audit: {checked} injections, "
          f"identity err {max_abs:.3e} (rel {max_rel:.3e}), "
          f"CS violations {cs_viol}, "
          f"prod top-K exact {out['prod_topk_exact_frac']:.3f}, "
          f"prod admission agreement {out['prod_admission_agreement']:.4f}")
    return out


# --------------------------------------------------------------------------- #
# Phase 3+4 - the panel
# --------------------------------------------------------------------------- #
def build_dense_panel(pipe: RetrievalPipeline, inter: pd.DataFrame,
                      limit: int | None = None) -> pd.DataFrame:
    """One row per logged do() operation: dense four worlds, mechanistic
    coordinates, boundary pressure, threat measures and immunity certificates.
    """
    ref = DenseRef(pipe)
    torch = ref.torch
    k = config.K_CANDIDATES
    if limit:
        inter = inter.head(limit)
    emb = encode_unique(pipe, list(inter["query_text"]) + list(inter["injected_query"]))
    qtiles = torch.tensor([0.01, 0.50, 0.99], dtype=torch.float64, device=ref.dev)

    rows: list[dict] = []
    t0 = time.time()
    for n, r in enumerate(inter.itertuples(), 1):
        try:
            di = pipe.corpus.idx(r.doc_id)
        except Exception:
            continue
        e0, e1 = emb.get(r.query_text), emb.get(r.injected_query)
        if e0 is None or e1 is None:
            continue
        s, cut0, e0t = ref.query_state(r.query_text, e0, k)
        dq = ref.vec(e1) - e0t
        dqn = float(dq.norm().item())
        ds = ref.scores(dq)

        w = dense_worlds(s, ds, di, k, ref.idx)
        s11 = w.pop("_s11")
        d_t = w["target_delta"]
        s_t0 = float(s[di].item())

        comp = ref.idx != di
        beats0 = (s > s_t0) | ((s == s_t0) & (ref.idx < di))
        beats01 = (s11 > s_t0) | ((s11 == s_t0) & (ref.idx < di))
        overtake_y01 = int((comp & ~beats0 & beats01).sum().item())
        cross_cut = int((comp & (s <= cut0) & (s11 > cut0)).sum().item())

        band = max(abs(cut0) * 0.05, 1e-9)
        in_band = (s >= cut0 - band) & (s < cut0)
        ds_band = ds[in_band]
        n_band = int(in_band.sum().item())
        pos_band = float(ds_band.clamp(min=0).sum().item()) if n_band else 0.0
        net_band = float(ds_band.sum().item()) if n_band else 0.0

        # a-priori exposure: competitors close enough that the pairwise
        # Cauchy-Schwarz bound allows an overtake (pre-Δs, needs only ||Δq||)
        pair_bound = dqn * (ref.doc_norm + ref.doc_norm[di])
        exposure = int((comp & ~beats0 & ((s_t0 - s) < pair_bound)).sum().item())
        n_above0 = w["n_above_00"]
        immune_cs = bool(w["y00"] and (n_above0 + exposure) < k)

        # exact post-encoding certificate: with the observed Δs vector the
        # world outcome is fully determined; log the identity's ingredients so
        # a violation (impossible unless the implementation is wrong) shows.
        immune_exact = bool(w["y00"] and w["n_above_11"] < k)

        q = torch.quantile(ds, qtiles)
        ds_abs = ds.abs()
        rows.append({
            "query_id": r.query_id, "doc_id": r.doc_id, "term": r.term, "arm": r.arm,
            "select_prob": float(getattr(r, "select_prob", np.nan)),
            # --- coordinates -------------------------------------------------
            "target_delta": d_t,
            "query_delta_norm": dqn,
            "cos_dq_target": d_t / dqn if dqn > 0 else 0.0,
            "cos_dq_query": float((e0t @ dq).item()) / dqn if dqn > 0 else 0.0,
            "cos_dq_band_mean": (float(ds_band.mean().item()) / dqn
                                 if (n_band and dqn > 0) else 0.0),
            # --- Δs distribution --------------------------------------------
            "ds_mean": float(ds.mean().item()),
            "ds_std": float(ds.std().item()),
            "ds_min": float(ds.min().item()),
            "ds_max": float(ds.max().item()),
            "ds_frac_pos": float((ds > 0).double().mean().item()),
            "ds_q01": float(q[0].item()), "ds_q50": float(q[1].item()),
            "ds_q99": float(q[2].item()),
            **{f"n_abs_gt_{str(t).replace('.', 'p')}":
               int((ds_abs > t).sum().item()) for t in DELTA_THRESHOLDS},
            # --- boundary state ----------------------------------------------
            "base_score": s_t0,
            "base_cut": cut0,
            "base_margin": s_t0 - cut0,
            "n_above_00": n_above0,
            "n_within_5pct_below": n_band,
            "max_pos_comp_delta": float(ds[comp].max().item()),
            "band_pressure_pos": pos_band,
            "band_pressure_net": net_band,
            "overtake_y01": overtake_y01,
            "cross_cut": cross_cut,
            "threat_exposure_cs": exposure,
            "immune_cs": immune_cs,
            "immune_exact_consistent": bool(immune_exact == w["y11"]) if w["y00"] else True,
            # --- worlds -------------------------------------------------------
            "y00": bool(w["y00"]), "y10": bool(w["y10"]),
            "y01": bool(w["y01"]), "y11": bool(w["y11"]),
            "surg_direct": int(w["y10"]) - int(w["y00"]),
            "surg_interference": int(w["y01"]) - int(w["y00"]),
            "surg_total": int(w["y11"]) - int(w["y00"]),
            "surg_interaction": (int(w["y11"]) - int(w["y10"])
                                 - int(w["y01"]) + int(w["y00"])),
        })
        if n % 1000 == 0:
            print(f"[dense_admission] {n}/{len(inter)} rows, "
                  f"{time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    print(f"[dense_admission] panel: {len(df)} rows over "
          f"{df['query_id'].nunique()} queries in {time.time() - t0:.1f}s")
    return df


def join_lexical(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach lexical coordinates and BM25 worlds from the stage-7 panel.

    Left join on the trial key; the dense panel stands alone when the
    admission panel is absent (union columns simply stay NaN).
    """
    import admission as A

    if not A.OUT_PANEL.exists():
        print("[dense_admission] no admission_panel.parquet - skipping lexical join")
        return panel
    # K is profile-dependent (50 on GPU, 20 on CPU) and sets the bar itself, so
    # a stage-7 panel built under a different K describes a different
    # conditioning event. Joining the two silently manufactures documents that
    # appear to be outside *both* top-K lists - impossible, since targets are
    # drawn from the union pool. Refuse rather than produce that artefact.
    lex_k = None
    if A.OUT_AUDIT.exists():
        try:
            lex_k = json.loads(A.OUT_AUDIT.read_text()).get("k_candidates")
        except Exception:  # noqa: BLE001
            lex_k = None
    if lex_k is None:
        print("[dense_admission] WARNING: the lexical audit records no "
              "k_candidates (panel predates the guard). Verify both panels were "
              "built at K=%d before trusting union-gate numbers."
              % config.K_CANDIDATES)
    elif int(lex_k) != int(config.K_CANDIDATES):
        print(f"[dense_admission] REFUSING lexical join: admission_panel was "
              f"built at K={lex_k}, this run is K={config.K_CANDIDATES}. "
              f"Re-run stage 7 at the same K.")
        return panel
    lex = pd.read_parquet(A.OUT_PANEL)
    cols = ["query_id", "doc_id", "term", "arm", "support", "lift", "idf",
            "base_margin", "y00", "y10", "y01", "y11", "threat_count"]
    lex = lex[[c for c in cols if c in lex.columns]].rename(columns={
        "base_margin": "bm25_margin", "threat_count": "bm25_threat_count",
        "y00": "bm25_y00", "y10": "bm25_y10", "y01": "bm25_y01", "y11": "bm25_y11",
    })
    merged = panel.merge(lex, on=["query_id", "doc_id", "term", "arm"], how="left")
    if len(merged) != len(panel):
        # a non-unique key would silently duplicate trials - refuse instead
        print(f"[dense_admission] WARNING: lexical join changed the row count "
              f"({len(panel)} -> {len(merged)}); keeping the unjoined panel")
        return panel
    for w in ("00", "10", "01", "11"):
        b = merged[f"bm25_y{w}"]
        both = merged[f"y{w}"].astype(bool) | b.fillna(False).astype(bool)
        merged[f"u{w}"] = np.where(b.isna(), np.nan, both.astype(float))
    return merged


def union_gate(panel: pd.DataFrame) -> pd.DataFrame:
    """Phase 6: classify channel-level changes at the BM25 OR dense gate."""
    if "u00" not in panel.columns:
        return pd.DataFrame()
    p = panel.dropna(subset=["u00", "u11"]).copy()
    if p.empty:
        return pd.DataFrame()
    p["dense_change"] = p["y11"] != p["y00"]
    p["bm25_change"] = p["bm25_y11"] != p["bm25_y00"]
    p["union_change"] = p["u11"] != p["u00"]
    recs = []
    for (arm, adm), g in p.groupby(["arm", "y00"]):
        n = len(g)
        recs.append({
            "arm": arm, "dense_y00": bool(adm), "n": n,
            "dense_changes": int(g["dense_change"].sum()),
            "bm25_changes": int(g["bm25_change"].sum()),
            "union_changes": int(g["union_change"].sum()),
            "dense_masked": int((g["dense_change"] & ~g["union_change"]).sum()),
            "redundant_dense_rescue": int(
                ((~g["y00"]) & g["y11"] & (g["bm25_y00"] == True)).sum()),  # noqa: E712
            "union_rescue": int(((g["u00"] == 0) & (g["u11"] == 1)).sum()),
            "union_displacement": int(((g["u00"] == 1) & (g["u11"] == 0)).sum()),
        })
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Phase 8 - inference
# --------------------------------------------------------------------------- #
def _cluster_ci(values: np.ndarray, clusters: np.ndarray,
                rng: np.random.Generator, b: int = _BOOTSTRAP_B) -> tuple[float, float]:
    """Percentile 95% CI for the mean, resampling whole queries."""
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return float("nan"), float("nan")
    by = {c: values[clusters == c] for c in uniq}
    means = np.empty(b)
    for i in range(b):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        means[i] = float(np.concatenate([by[c] for c in draw]).mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def dense_by_bin(panel: pd.DataFrame) -> pd.DataFrame:
    """Four-world effects by arm x dense-baseline stratum x lexical support
    bin, with query-cluster bootstrap CIs. Support bins are the lexical
    coordinate, deliberately: H8 asks whether dense effects are organised by
    lexical support at all."""
    if panel.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(config.SEED)
    p = panel.copy()
    binner = "support" if "support" in p.columns and p["support"].notna().any() else None
    recs = []
    for (arm, adm), g0 in p.groupby(["arm", "y00"]):
        if len(g0) < 10:
            continue
        groups = (g0.groupby(pd.qcut(g0[binner], min(6, g0[binner].nunique()),
                                     duplicates="drop"), observed=True)
                  if binner else [("all", g0)])
        for b, g in groups:
            if len(g) < 10:
                continue
            cl = g["query_id"].astype("category").cat.codes.to_numpy()
            rec = {"arm": arm, "dense_y00": bool(adm), "support_bin": str(b),
                   "median_support": (float(g[binner].median()) if binner else np.nan),
                   "n": len(g), "n_queries": int(g["query_id"].nunique())}
            for eff in ("surg_direct", "surg_interference", "surg_total",
                        "surg_interaction"):
                v = g[eff].to_numpy(dtype=float)
                lo, hi = _cluster_ci(v, cl, rng)
                rec[eff] = float(v.mean())
                rec[f"{eff}_lo"], rec[f"{eff}_hi"] = lo, hi
            rec["mean_target_delta"] = float(g["target_delta"].mean())
            rec["mean_query_delta_norm"] = float(g["query_delta_norm"].mean())
            recs.append(rec)
    return pd.DataFrame(recs)


#: Feature sets for the mechanistic comparison (predictive, NOT causal).
#: Only pre-outcome quantities: baseline state, ||Δq||, target_delta and
#: boundary-pressure summaries derived from Δs - never the world outcomes.
MODEL_FEATURES = {
    "margin": ["base_margin"],
    "margin+support": ["base_margin", "support"],
    "margin+qnorm": ["base_margin", "query_delta_norm"],
    "margin+target_delta": ["base_margin", "target_delta"],
    "margin+pressure": ["base_margin", "n_within_5pct_below", "band_pressure_pos",
                        "threat_exposure_cs", "max_pos_comp_delta"],
    "full": ["base_margin", "support", "query_delta_norm", "target_delta",
             "n_within_5pct_below", "band_pressure_pos", "threat_exposure_cs",
             "max_pos_comp_delta", "cos_dq_target", "ds_frac_pos"],
}


def mechanistic_models(panel: pd.DataFrame) -> pd.DataFrame:
    """Grouped-CV comparison of what predicts admission change (H2/H3).

    Leave-queries-out (GroupKFold by query) logistic regressions; metrics on
    pooled out-of-fold predictions. Two tasks: ejection among baseline-admitted
    trials, rescue among baseline-excluded trials.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 log_loss, roc_auc_score)
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    recs = []
    tasks = [
        ("ejection", panel[panel["y00"]].copy(), lambda g: (~g["y11"]).astype(int)),
        ("rescue", panel[~panel["y00"]].copy(), lambda g: g["y11"].astype(int)),
    ]
    for task, p, make_y in tasks:
        if p.empty:
            continue
        for name, feats in MODEL_FEATURES.items():
            cols = [c for c in feats if c in p.columns]
            sub = p.dropna(subset=cols)
            y = make_y(sub).to_numpy()
            if len(sub) < 100 or y.sum() < 10 or y.sum() > len(y) - 10:
                continue
            groups = sub["query_id"].to_numpy()
            n_splits = min(5, len(np.unique(groups)))
            if n_splits < 2:
                continue
            oof = np.full(len(sub), np.nan)
            X = sub[cols].to_numpy(dtype=float)
            for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
                if len(np.unique(y[tr])) < 2:
                    continue
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=1000, C=1.0))
                m.fit(X[tr], y[tr])
                oof[te] = m.predict_proba(X[te])[:, 1]
            ok = ~np.isnan(oof)
            if ok.sum() < 100 or len(np.unique(y[ok])) < 2:
                continue
            yo, po = y[ok], np.clip(oof[ok], 1e-6, 1 - 1e-6)
            recs.append({
                "task": task, "model": name, "n": int(ok.sum()),
                "n_features": len(cols), "base_rate": float(yo.mean()),
                "log_loss": float(log_loss(yo, po)),
                "brier": float(brier_score_loss(yo, po)),
                "roc_auc": float(roc_auc_score(yo, po)),
                "pr_auc": float(average_precision_score(yo, po)),
            })
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="cap panel rows (smoke test)")
    ap.add_argument("--audit-n", type=int, default=200, help="injections to audit")
    ap.add_argument("--no-models", action="store_true", help="skip the CV model stage")
    args = ap.parse_args()

    config.set_seeds()
    src = config.RESULTS_DIR / "interventions.parquet"
    if not src.exists():
        print(f"error: {src} not found - run `python -m run_all --only 1 2` first")
        return 1
    corpus, _ = load_corpus_and_queries()
    pipe = RetrievalPipeline(corpus)
    inter = pd.read_parquet(src)
    print(f"[dense_admission] {len(inter)} logged interventions, dataset {config.DATASET}")

    audit = audit_dense(pipe, inter, args.audit_n)
    OUT_AUDIT.write_text(json.dumps(audit, indent=2))
    if not audit["exact"]:
        print("[dense_admission] ABORT: the finite-perturbation identity or the "
              "Cauchy-Schwarz bound failed on real injections. Fix the scoring "
              "replication before interpreting anything downstream.")
        return 1

    panel = build_dense_panel(pipe, inter, args.limit)
    panel = join_lexical(panel)
    panel.to_parquet(OUT_PANEL, index=False)
    print(f"[dense_admission] wrote {OUT_PANEL.name}")

    bad_cert = int((panel["immune_cs"] & panel["y00"] & ~panel["y11"]).sum())
    n_cert = int((panel["immune_cs"] & panel["y00"]).sum())
    n_disp = int((panel["y00"] & ~panel["y11"]).sum())
    incons = int((~panel["immune_exact_consistent"]).sum())
    print(f"[dense_admission] CS immunity: certified {n_cert}, displaced overall "
          f"{n_disp}, certificate violations {bad_cert} (must be 0), "
          f"exact-certificate inconsistencies {incons} (must be 0)")

    tab = dense_by_bin(panel)
    if not tab.empty:
        tab.to_csv(OUT_BY_BIN, index=False)
        print("\nDense four-world decomposition (cluster-bootstrap 95% CIs):")
        show = [c for c in tab.columns if not c.endswith(("_lo", "_hi"))]
        print(tab[show].to_string(index=False))

    ug = union_gate(panel)
    if not ug.empty:
        ug.to_csv(OUT_UNION, index=False)
        print("\nUnion gate (BM25 OR dense):")
        print(ug.to_string(index=False))

    if not args.no_models:
        models = mechanistic_models(panel)
        if not models.empty:
            models.to_csv(OUT_MODELS, index=False)
            print("\nMechanistic prediction (grouped CV, out-of-fold):")
            print(models.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
