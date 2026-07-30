"""Module 3 - exact path-specific effects by freezing pipeline stages.

Standard causal mediation has to *estimate* natural direct and indirect effects,
because the analyst cannot observe a unit whose mediator takes its control value
while its treatment takes the treated value. A retrieval pipeline is different:
it is a deterministic function we own. We can literally evaluate the
cross-world quantity

    Y(Q1, M(Q0))

by executing the pipeline with the mediator stage fed ``Q0`` and everything else
fed ``Q1``. No sequential-ignorability assumption, no estimation error - the
path-specific effects are *computed*.

Five worlds are evaluated per treated pair (Q0 -> Q1 = Q0 + " " + t):

===================== ============ ============ ============ =================
world                 BM25 sees    dense sees   reranker sees carries
===================== ============ ============ ============ =================
baseline              Q0           Q0           Q0            nothing
full                  Q1           Q1           Q1            total effect
freeze_candidates     Q0           Q0           Q1            reranker path
freeze_rerank         Q1           Q1           Q0            first-stage path
freeze_dense          Q1           Q0           Q1            rerank + lexical
freeze_bm25           Q0           Q1           Q1            rerank + dense
===================== ============ ============ ============ =================

from which the additive decomposition is

    reranker  = r(baseline)          - r(freeze_candidates)
    lexical   = r(freeze_candidates) - r(freeze_dense)
    dense     = r(freeze_candidates) - r(freeze_bm25)
    residual  = total - (reranker + lexical + dense)

The residual is the *non-additivity* of the two first-stage channels: they feed
a shared pool via a union-and-truncate operation, which is not additive. It is
reported, never absorbed.

Run standalone::

    python -m mediation
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from interventions import OUT_PARQUET as INTERVENTIONS_PARQUET
from pipeline import (
    BaselineRun,
    RetrievalPipeline,
    compute_baseline,
    load_corpus_and_queries,
    select_queries,
)

OUT_PARQUET = config.RESULTS_DIR / "mediation.parquet"
OUT_TABLE = config.RESULTS_DIR / "mediation_ratio.csv"
FIG_MEDIATION = config.RESULTS_DIR / "fig_mediation_ratio.png"

#: The two first-stage architectures whose attributions we compare.
CONFIGS: dict[str, str] = {
    "union": "union",       # (a) BM25 + dense union, as in the main pipeline
    "bm25_only": "bm25",    # (b) lexical first stage only
}

PATHS = ["reranker", "lexical", "dense", "residual"]


@dataclass
class MediationRow:
    """Path decomposition for one (query, doc, injected term) triple."""

    query_id: str
    doc_id: str
    term: str
    cfg: str
    r_baseline: int
    r_full: int
    r_freeze_candidates: int
    r_freeze_rerank: int
    r_freeze_dense: int
    r_freeze_bm25: int
    total: int
    reranker: int
    lexical: int
    dense: int
    residual: int
    first_stage_total: int


def decompose(
    pipe: RetrievalPipeline,
    q0: str,
    q1: str,
    doc_id: str,
    first_stage: str,
    r_baseline: int,
) -> dict[str, int]:
    """Evaluate the five counterfactual worlds and return the decomposition.

    Every quantity is a *rank improvement*: positive means the target document
    moved up. Ranks are censored at :data:`config.MISSING_RANK`.
    """
    kw = dict(first_stage=first_stage)

    r_full = pipe.run(q1, **kw).rank_of(doc_id)
    # Q1 reaches the reranker only; candidate generation is held at Q0.
    r_fc = pipe.run(
        q1, bm25_query=q0, dense_query=q0, rerank_query=q1, **kw
    ).rank_of(doc_id)
    # Q1 reaches candidate generation only; the reranker is held at Q0.
    r_fr = pipe.run(
        q1, bm25_query=q1, dense_query=q1, rerank_query=q0, **kw
    ).rank_of(doc_id)
    # Dense channel frozen: Q1 flows through BM25 and the reranker.
    r_fd = pipe.run(
        q1, bm25_query=q1, dense_query=q0, rerank_query=q1, **kw
    ).rank_of(doc_id)
    # BM25 channel frozen: Q1 flows through dense and the reranker.
    r_fb = pipe.run(
        q1, bm25_query=q0, dense_query=q1, rerank_query=q1, **kw
    ).rank_of(doc_id)

    total = r_baseline - r_full
    reranker = r_baseline - r_fc
    lexical = r_fc - r_fd
    dense_ = r_fc - r_fb
    if first_stage == "bm25":
        # The dense channel is not wired in at all: its path effect is zero by
        # construction, and the decomposition is then exactly additive.
        dense_ = 0
    residual = total - (reranker + lexical + dense_)
    return {
        "r_full": r_full,
        "r_freeze_candidates": r_fc,
        "r_freeze_rerank": r_fr,
        "r_freeze_dense": r_fd,
        "r_freeze_bm25": r_fb,
        "total": total,
        "reranker": reranker,
        "lexical": lexical,
        "dense": dense_,
        "residual": residual,
        "first_stage_total": r_baseline - r_fr,
    }


def run_mediation(
    pipe: RetrievalPipeline,
    treated: pd.DataFrame,
    baseline: dict[str, BaselineRun],
) -> pd.DataFrame:
    """Compute the path decomposition for every treated pair, in both configs.

    Rows are processed grouped by query so that the cross-encoder pair cache -
    which is keyed on (query text, doc) - stays hot: the five worlds share most
    of their (query, document) pairs.
    """
    rows: list[dict] = []
    t0 = time.time()
    groups = list(treated.groupby("query_id", sort=True))
    for n, (qid, g) in enumerate(groups, 1):
        q0 = g["query_text"].iloc[0]
        # Baseline rank per config (the union baseline is already cached).
        r0: dict[str, dict[str, int]] = {}
        for cfg_name, fs in CONFIGS.items():
            res0 = pipe.run(q0, first_stage=fs)
            r0[cfg_name] = dict(res0.ranks)

        for _, row in g.iterrows():
            doc_id, term, q1 = row["doc_id"], row["term"], row["injected_query"]
            for cfg_name, fs in CONFIGS.items():
                rb = r0[cfg_name].get(doc_id, config.MISSING_RANK)
                d = decompose(pipe, q0, q1, doc_id, fs, rb)
                rows.append(
                    {
                        "query_id": qid,
                        "doc_id": doc_id,
                        "term": term,
                        "cfg": cfg_name,
                        "r_baseline": rb,
                        **d,
                    }
                )
        if n % 10 == 0 or n == len(groups):
            el = time.time() - t0
            print(
                f"[mediation] {n}/{len(groups)} queries, {len(rows)} rows, "
                f"{el:.0f}s ({el / n:.2f}s/query), CE pairs={pipe.ce_pairs_scored} "
                f"hits={pipe.ce_cache_hits}",
                flush=True,
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def mediation_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Share of total effect magnitude attributable to each path, per config.

    Shares are computed on *absolute* path effects: paths routinely push in
    opposite directions, and a signed normalisation would let them cancel into
    meaningless percentages. The residual is included in the denominator so the
    four shares sum to 100% and non-additivity cannot be hidden.
    """
    recs = []
    for cfg, g in df.groupby("cfg", sort=True):
        mags = {p: float(np.abs(g[p]).mean()) for p in PATHS}
        denom = sum(mags.values())
        rec: dict[str, object] = {
            "cfg": cfg,
            "n_pairs": int(len(g)),
            "mean_abs_total": float(np.abs(g["total"]).mean()),
            "mean_total": float(g["total"].mean()),
        }
        for p in PATHS:
            rec[f"{p}_mean_abs"] = mags[p]
            rec[f"{p}_mean"] = float(g[p].mean())
            rec[f"{p}_share_pct"] = 100.0 * mags[p] / denom if denom > 0 else float("nan")
        # How badly additivity fails, as a fraction of the total effect size.
        rec["nonadditivity_pct_of_total"] = (
            100.0 * mags["residual"] / rec["mean_abs_total"]
            if rec["mean_abs_total"] > 0
            else float("nan")
        )
        recs.append(rec)
    return pd.DataFrame(recs)


def print_ratio_table(ratio: pd.DataFrame) -> bool:
    """Print the mediation-ratio table; return whether the configs differ."""
    print("\n" + "=" * 88)
    print("MEDIATION RATIO - share of mean |total effect| by causal path")
    print("=" * 88)
    hdr = f"{'config':<12}{'n':>6}{'mean|TE|':>10}"
    for p in PATHS:
        hdr += f"{p:>13}"
    print(hdr)
    for _, r in ratio.iterrows():
        line = f"{r['cfg']:<12}{int(r['n_pairs']):>6}{r['mean_abs_total']:>10.3f}"
        for p in PATHS:
            line += f"{r[f'{p}_share_pct']:>12.1f}%"
        print(line)
    print("-" * 88)
    print("Mean signed path effects (rank improvement):")
    for _, r in ratio.iterrows():
        line = f"  {r['cfg']:<12}"
        for p in PATHS:
            line += f"{p}={r[f'{p}_mean']:+.3f}  "
        print(line)
    print("-" * 88)
    for _, r in ratio.iterrows():
        print(
            f"  {r['cfg']:<12} non-additivity (interaction residual) = "
            f"{r['nonadditivity_pct_of_total']:.1f}% of mean |total effect|"
        )

    differential = False
    if len(ratio) == 2:
        a, b = ratio.iloc[0], ratio.iloc[1]
        deltas = {p: abs(a[f"{p}_share_pct"] - b[f"{p}_share_pct"]) for p in PATHS}
        max_shift = max(deltas.values())
        differential = max_shift >= 10.0
        print("-" * 88)
        print(
            f"Share shift between '{a['cfg']}' and '{b['cfg']}' (percentage points): "
            + ", ".join(f"{p}={deltas[p]:.1f}" for p in PATHS)
        )
        print(f"Max shift = {max_shift:.1f} pp")
        print(f"MEDIATION DIFFERENTIAL: {'YES' if differential else 'NO'}")
    print("=" * 88 + "\n")
    return differential


def plot_mediation(ratio: pd.DataFrame, path=FIG_MEDIATION) -> None:
    """Stacked bar of the three path shares plus the interaction residual."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "reranker": "#2c7fb8",
        "lexical": "#7fcdbb",
        "dense": "#c7e9b4",
        "residual": "#d95f0e",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    cfgs = list(ratio["cfg"])
    bottom = np.zeros(len(cfgs))
    for p in PATHS:
        vals = ratio[f"{p}_share_pct"].to_numpy(dtype=float)
        ax.bar(cfgs, vals, bottom=bottom, label=p, color=colors[p], edgecolor="white")
        for i, v in enumerate(vals):
            if v > 4:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9)
        bottom += vals
    ax.set_ylabel("share of mean |total effect| (%)")
    ax.set_title("Mediation ratio by first-stage architecture")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    x = np.arange(len(PATHS))
    w = 0.38
    for i, (_, r) in enumerate(ratio.iterrows()):
        ax.bar(
            x + (i - 0.5) * w,
            [r[f"{p}_mean"] for p in PATHS],
            w,
            label=r["cfg"],
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(PATHS)
    ax.set_ylabel("mean signed path effect (rank improvement)")
    ax.set_title("Signed path effects")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[mediation] wrote {path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    config.set_seeds()
    t0 = time.time()
    if not INTERVENTIONS_PARQUET.exists():
        raise SystemExit(
            f"{INTERVENTIONS_PARQUET} not found - run `python -m interventions` first."
        )
    inter = pd.read_parquet(INTERVENTIONS_PARQUET)
    treated = inter[inter["arm"] == "treatment"].reset_index(drop=True)
    print(f"[mediation] {len(treated)} treated pairs over {treated['query_id'].nunique()} queries")

    corpus, queries = load_corpus_and_queries()
    pipe = RetrievalPipeline(corpus)
    baseline = compute_baseline(pipe, queries, select_queries(queries))

    df = run_mediation(pipe, treated, baseline)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"[mediation] wrote {OUT_PARQUET} ({len(df)} rows)")

    ratio = mediation_ratio(df)
    ratio.to_csv(OUT_TABLE, index=False)
    print_ratio_table(ratio)
    plot_mediation(ratio)
    print(f"[mediation] wall clock {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
