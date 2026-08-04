"""Architecture-robustness experiment for retrieval admission.

Re-evaluates the *same* logged query interventions under three retrieval
families: parallel union, weighted reciprocal-rank fusion across a
lexical/dense spectrum, and two-stage cascades in both orders. For every
architecture it evaluates the
four channel worlds (Q0/Q1 at BM25 x Q0/Q1 at dense), giving an exact channel
Shapley decomposition and an order-sensitivity test.

This is retrieval-only: no cross-encoder or generator is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from dense_admission import encode_unique
from pipeline import RetrievalPipeline, load_corpus_and_queries

OUT_PANEL = config.RESULTS_DIR / "architecture_admission_panel.parquet"
OUT_SUMMARY = config.RESULTS_DIR / "architecture_admission_summary.csv"
OUT_CONTRASTS = config.RESULTS_DIR / "architecture_admission_contrasts.csv"
OUT_ORDER = config.RESULTS_DIR / "architecture_order_sensitivity.csv"
OUT_META = config.RESULTS_DIR / "architecture_admission_meta.json"


def _csv_numbers(name: str, default: str, cast=float) -> tuple:
    raw = os.environ.get(name, default)
    return tuple(cast(x.strip()) for x in raw.split(",") if x.strip())


FUSION_LAMBDAS = _csv_numbers("ARCH_FUSION_LAMBDAS", "0,.25,.5,.75,1")
CASCADE_BUDGETS = _csv_numbers("ARCH_CASCADE_BUDGETS", "50,100,500,1000", int)
FINAL_K = int(os.environ.get("ARCH_FINAL_K", config.K_CANDIDATES))
N_BOOT = int(os.environ.get("ARCH_N_BOOTSTRAP", "2000"))
RRF_C = int(os.environ.get("ARCH_RRF_C", "60"))


def topk(scores: np.ndarray, k: int, candidates: np.ndarray | None = None) -> np.ndarray:
    """Exactly k global document indices under (-score, corpus-index) order."""
    ids = np.arange(len(scores)) if candidates is None else np.asarray(candidates, dtype=int)
    if not len(ids):
        return ids
    k = min(k, len(ids))
    vals = scores[ids]
    cut = np.partition(vals, -k)[-k]
    above = ids[vals > cut]
    tied = ids[vals == cut]
    chosen = np.concatenate((above, tied[: k - len(above)]))
    return chosen[np.lexsort((chosen, -scores[chosen]))]


def _ranking(scores: np.ndarray, depth: int,
             cache: dict[int, np.ndarray]) -> np.ndarray:
    """Memoize each channel/world ranking once per intervention."""
    key = id(scores)
    if key not in cache:
        cache[key] = topk(scores, min(depth, len(scores)))
    return cache[key]


@dataclass(frozen=True)
class Architecture:
    name: str
    family: str
    parameter: float | int | None = None

    def admitted(self, bm25: np.ndarray, dense: np.ndarray, target: int,
                 rank_cache: dict[int, np.ndarray] | None = None) -> bool:
        rank_cache = {} if rank_cache is None else rank_cache
        k = min(FINAL_K, len(bm25))
        depth = max(k, max(CASCADE_BUDGETS, default=k))
        br = _ranking(bm25, depth, rank_cache)
        dr = _ranking(dense, depth, rank_cache)
        if self.family == "union":
            return target in set(br[:k]) | set(dr[:k])
        if self.family == "fusion":
            lam = float(self.parameter)
            ids = np.union1d(br, dr)
            rb = {int(x): i + 1 for i, x in enumerate(br)}
            rd = {int(x): i + 1 for i, x in enumerate(dr)}
            fused = np.array([
                lam / (RRF_C + rb[int(x)]) if int(x) in rb else 0.0
                for x in ids
            ]) + np.array([
                (1.0 - lam) / (RRF_C + rd[int(x)]) if int(x) in rd else 0.0
                for x in ids
            ])
            chosen = ids[np.lexsort((ids, -fused))[:k]]
            return target in set(chosen)
        m = min(int(self.parameter), len(bm25))
        if self.family == "bm25_then_dense":
            return target in set(topk(dense, k, br[:m]))
        if self.family == "dense_then_bm25":
            return target in set(topk(bm25, k, dr[:m]))
        raise ValueError(f"unknown architecture family {self.family!r}")


def architectures() -> list[Architecture]:
    out = [Architecture("parallel_union", "union")]
    out += [Architecture(f"rrf_lambda_{x:g}", "fusion", x) for x in FUSION_LAMBDAS]
    for m in CASCADE_BUDGETS:
        out.append(Architecture(f"bm25_then_dense_m{m}", "bm25_then_dense", m))
        out.append(Architecture(f"dense_then_bm25_m{m}", "dense_then_bm25", m))
    return out


def four_worlds(a: Architecture, b0: np.ndarray, b1: np.ndarray,
                d0: np.ndarray, d1: np.ndarray, target: int,
                rank_cache: dict[int, np.ndarray] | None = None) -> dict[str, float | bool]:
    cache = {} if rank_cache is None else rank_cache
    y00 = a.admitted(b0, d0, target, cache)
    y10 = a.admitted(b1, d0, target, cache)
    y01 = a.admitted(b0, d1, target, cache)
    y11 = a.admitted(b1, d1, target, cache)
    lexical = .5 * ((int(y10) - int(y00)) + (int(y11) - int(y01)))
    dense = .5 * ((int(y01) - int(y00)) + (int(y11) - int(y10)))
    return {
        "y00": y00, "y10": y10, "y01": y01, "y11": y11,
        "lexical_shapley": lexical, "dense_shapley": dense,
        "interaction": int(y11) - int(y10) - int(y01) + int(y00),
        "total_change": int(y11) - int(y00),
        "rescue": int((not y00) and y11),
        "displacement": int(y00 and (not y11)),
    }


def build_panel(pipe: RetrievalPipeline, inter: pd.DataFrame,
                limit: int | None = None) -> pd.DataFrame:
    if limit:
        inter = inter.head(limit)
    inter = inter.sort_values(["query_id", "doc_id", "term", "arm"]).reset_index(drop=True)
    emb = encode_unique(pipe, list(inter["query_text"]) + list(inter["injected_query"]))
    archs, rows = architectures(), []
    started, done = time.time(), 0
    for _, group in inter.groupby("query_id", sort=False):
        q0 = str(group.iloc[0]["query_text"])
        b0 = pipe._bm25_array(q0)  # audited production scoring path
        d0 = pipe.doc_emb @ emb[q0]
        depth = max(FINAL_K, max(CASCADE_BUDGETS, default=FINAL_K))
        baseline_rankings: dict[int, np.ndarray] = {}
        _ranking(b0, depth, baseline_rankings)
        _ranking(d0, depth, baseline_rankings)
        for r in group.itertuples():
            target = pipe.corpus.idx(r.doc_id)
            b1 = pipe._bm25_array(r.injected_query)
            d1 = pipe.doc_emb @ emb[r.injected_query]
            rank_cache = dict(baseline_rankings)
            base = {
                "query_id": r.query_id, "doc_id": r.doc_id, "term": r.term,
                "arm": r.arm, "target_stratum": r.target_stratum,
                "analysis_weight": float(getattr(r, "analysis_weight", 1.0)),
            }
            for a in archs:
                rows.append({**base, "architecture": a.name, "family": a.family,
                             "parameter": a.parameter,
                             **four_worlds(a, b0, b1, d0, d1, target, rank_cache)})
            # Never retain one full-corpus array per injection (fatal on Quora).
            pipe.__dict__.setdefault("_bm25_cache", {}).pop(r.injected_query, None)
            done += 1
            if done % 500 == 0:
                print(f"[architecture] {done}/{len(inter)} interventions, {time.time()-started:.0f}s", flush=True)
        pipe.__dict__.setdefault("_bm25_cache", {}).pop(q0, None)
    return pd.DataFrame(rows)


METRICS = ("y00", "y11", "total_change", "rescue", "displacement",
           "lexical_shapley", "dense_shapley", "interaction")


def _wmean(g: pd.DataFrame, metric: str) -> float:
    return float(np.average(g[metric].astype(float), weights=g["analysis_weight"]))


def summarize(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (architecture, family, parameter, arm), g in panel.groupby(
            ["architecture", "family", "parameter", "arm"], dropna=False):
        row = {"architecture": architecture, "family": family, "parameter": parameter,
               "arm": arm, "n": len(g), "n_queries": g.query_id.nunique()}
        row.update({m: _wmean(g, m) for m in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def clustered_contrasts(panel: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Treatment-minus-control contrasts with a query-cluster bootstrap."""
    rng = np.random.default_rng(config.stable_seed("architecture-bootstrap"))
    rows = []
    for architecture, p in panel.groupby("architecture", sort=False):
        qids = np.asarray(p.query_id.unique())
        qpos = {q: i for i, q in enumerate(qids)}
        point = {arm: p[p.arm == arm] for arm in ("treatment", "control")}
        for metric in METRICS[2:]:
            est = _wmean(point["treatment"], metric) - _wmean(point["control"], metric)
            boots = []
            if n_boot and len(qids) > 1:
                totals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for arm in ("treatment", "control"):
                    g = point[arm].copy()
                    g["_wy"] = g[metric].astype(float) * g["analysis_weight"]
                    agg = g.groupby("query_id").agg(wy=("_wy", "sum"), w=("analysis_weight", "sum"))
                    wy, w = np.zeros(len(qids)), np.zeros(len(qids))
                    for q, x in agg.iterrows():
                        wy[qpos[q]], w[qpos[q]] = x.wy, x.w
                    totals[arm] = wy, w
                for _ in range(n_boot):
                    counts = np.bincount(rng.integers(0, len(qids), len(qids)), minlength=len(qids))
                    means = {}
                    for arm, (wy, w) in totals.items():
                        denom = float(counts @ w)
                        means[arm] = float(counts @ wy) / denom if denom else np.nan
                    if not np.isnan(means["treatment"] + means["control"]):
                        boots.append(means["treatment"] - means["control"])
            lo, hi = (np.quantile(boots, [.025, .975]) if boots else (np.nan, np.nan))
            rows.append({"architecture": architecture, "metric": metric,
                         "estimate": est, "ci_lo": lo, "ci_hi": hi,
                         "n_queries": len(qids), "n_bootstrap": len(boots)})
    return pd.DataFrame(rows)


def order_sensitivity(panel: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Direct paired test of BM25->dense versus dense->BM25 ordering."""
    keys = ["query_id", "doc_id", "term", "arm", "analysis_weight"]
    rng = np.random.default_rng(config.stable_seed("architecture-order-bootstrap"))
    rows = []
    for m in CASCADE_BUDGETS:
        left = panel[panel.architecture == f"bm25_then_dense_m{m}"]
        right = panel[panel.architecture == f"dense_then_bm25_m{m}"]
        if left.empty or right.empty:
            continue
        paired = left[keys + list(METRICS[2:])].merge(
            right[keys + list(METRICS[2:])], on=keys, suffixes=("_bd", "_db"),
            validate="one_to_one")
        qids = np.asarray(paired.query_id.unique())
        qpos = {q: i for i, q in enumerate(qids)}
        for metric in METRICS[2:]:
            paired["_difference"] = paired[f"{metric}_bd"] - paired[f"{metric}_db"]
            arm_stats = {}
            arm_boots = {}
            for arm in ("treatment", "control"):
                g = paired[paired.arm == arm]
                arm_stats[arm] = float(np.average(g._difference, weights=g.analysis_weight))
                g = g.assign(_wy=g._difference * g.analysis_weight)
                agg = g.groupby("query_id").agg(wy=("_wy", "sum"), w=("analysis_weight", "sum"))
                wy, w = np.zeros(len(qids)), np.zeros(len(qids))
                for q, x in agg.iterrows():
                    wy[qpos[q]], w[qpos[q]] = x.wy, x.w
                arm_boots[arm] = wy, w
            estimate = arm_stats["treatment"] - arm_stats["control"]
            boots = []
            for _ in range(n_boot):
                counts = np.bincount(rng.integers(0, len(qids), len(qids)), minlength=len(qids))
                means = {}
                for arm, (wy, w) in arm_boots.items():
                    denom = float(counts @ w)
                    means[arm] = float(counts @ wy) / denom if denom else np.nan
                if not np.isnan(means["treatment"] + means["control"]):
                    boots.append(means["treatment"] - means["control"])
            lo, hi = (np.quantile(boots, [.025, .975]) if boots else (np.nan, np.nan))
            rows.append({"first_stage_budget": m, "metric": metric,
                         "order_difference_treatment": arm_stats["treatment"],
                         "order_difference_control": arm_stats["control"],
                         "order_difference_in_differences": estimate,
                         "ci_lo": lo, "ci_hi": hi, "n_queries": len(qids),
                         "n_bootstrap": len(boots)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test intervention limit")
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()
    config.set_seeds()
    src = config.RESULTS_DIR / "interventions.parquet"
    if not src.exists():
        print(f"error: {src} missing; produce/copy the K=50 intervention artefact first")
        return 1
    corpus, _ = load_corpus_and_queries()
    pipe = RetrievalPipeline(corpus)
    inter = pd.read_parquet(src)
    panel = build_panel(pipe, inter, args.limit)
    panel.to_parquet(OUT_PANEL, index=False)
    summary = summarize(panel)
    summary.to_csv(OUT_SUMMARY, index=False)
    contrasts = clustered_contrasts(panel, 0 if args.no_bootstrap else N_BOOT)
    contrasts.to_csv(OUT_CONTRASTS, index=False)
    order = order_sensitivity(panel, 0 if args.no_bootstrap else N_BOOT)
    order.to_csv(OUT_ORDER, index=False)
    OUT_META.write_text(json.dumps({
        "estimand": "retrieval admission under paired channel interventions",
        "fusion": "weighted reciprocal-rank fusion over declared retrieval depth",
        "rrf_c": RRF_C,
        "final_k": FINAL_K, "fusion_lambdas": FUSION_LAMBDAS,
        "cascade_budgets": CASCADE_BUDGETS, "n_bootstrap": 0 if args.no_bootstrap else N_BOOT,
        "dataset": config.DATASET, "dense_model": config.DENSE_MODEL,
        "dense_model_revision": config.DENSE_MODEL_REVISION,
        "code_fingerprint": config.code_fingerprint(),
    }, indent=2))
    print(f"[architecture] wrote {OUT_PANEL.name}, {OUT_SUMMARY.name}, "
          f"{OUT_CONTRASTS.name}, {OUT_ORDER.name}")
    print(contrasts[contrasts.metric.isin(["total_change", "lexical_shapley", "dense_shapley"])]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
