"""Module 5 - counterfactual stability under meaning-preserving query edits.

A meaning-preserving rewrite of a query is a *null intervention*: the causal
content of Q is unchanged, so a pipeline whose behaviour tracks meaning rather
than surface form should return the same ranking. RBO@10 between the original
and rewritten rankings therefore measures how much of the pipeline's output is
driven by query form rather than query content. The gap ``1 - RBO`` is the
instability.

Three rewrites, all LLM-free so the MVP stays dependency-light:

* **A - destop**  : drop stopwords. Should be a near no-op for BM25, which
  already removes them at index time; a useful implementation check.
* **B - shuffle** : permute word order. BM25 is a bag of words and must score
  *identically* (RBO exactly 1.0); any deviation is a bug. The dense encoder
  and cross-encoder are order-sensitive, so their instability here is real.
* **C - synonym** : singular/plural flips plus substitution from a hand-rolled
  medical/general synonym map.

Run standalone::

    python -m stability
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config
from pipeline import (
    RetrievalPipeline,
    load_corpus_and_queries,
    select_queries,
    stopwords,
)

OUT_CSV = config.RESULTS_DIR / "stability.csv"
OUT_PARQUET = config.RESULTS_DIR / "stability_per_query.parquet"
FIG_STABILITY = config.RESULTS_DIR / "fig_stability.png"

VARIANTS = ["destop", "shuffle", "synonym"]
CONFIGS = ["bm25_only", "dense_only", "full"]

#: Hand-rolled meaning-preserving substitutions. Kept deliberately small and
#: conservative: a substitution that changes meaning would turn this from a
#: null intervention into a real one and invalidate the measurement.
SYNONYMS: dict[str, str] = {
    "cancer": "carcinoma",
    "tumor": "tumour",
    "heart": "cardiac",
    "stroke": "cerebrovascular accident",
    "high": "elevated",
    "low": "reduced",
    "big": "large",
    "small": "little",
    "fat": "lipid",
    "sugar": "glucose",
    "salt": "sodium",
    "doctor": "physician",
    "drug": "medication",
    "illness": "disease",
    "sick": "ill",
    "kids": "children",
    "elderly": "older adults",
    "women": "females",
    "men": "males",
    "food": "nutrition",
    "eating": "consuming",
    "helps": "aids",
    "harmful": "damaging",
    "dangerous": "hazardous",
    "prevent": "avert",
    "cause": "induce",
    "reduce": "lower",
    "increase": "raise",
    "treat": "manage",
    "effect": "impact",
    "risk": "hazard",
    "study": "research",
    "benefits": "advantages",
}

_IRREGULAR = {"children": "child", "women": "woman", "men": "man", "people": "person"}


# --------------------------------------------------------------------------- #
# RBO
# --------------------------------------------------------------------------- #
def rbo(list1: list[str], list2: list[str], p: float = config.RBO_P) -> float:
    """Rank-Biased Overlap, extrapolated (Webber, Moffat & Zobel 2010, eq. 32).

    The extrapolated form is used rather than the truncated sum because the
    truncated sum cannot reach 1.0 at finite depth - two identical top-10 lists
    would score 0.651 at p=0.9, which would make "RBO close to 1 means stable"
    meaningless. RBO_EXT returns exactly 1.0 for identical lists at any depth.
    """
    if not list1 and not list2:
        return 1.0
    if not list1 or not list2:
        return 0.0

    s_list, l_list = (list1, list2) if len(list1) <= len(list2) else (list2, list1)
    s, l = len(s_list), len(l_list)

    seen_s: set[str] = set()
    seen_l: set[str] = set()
    overlap = 0
    x: list[int] = []  # x[d-1] = |S_1:d ∩ T_1:d|
    for d in range(1, l + 1):
        item_l = l_list[d - 1]
        if d <= s:
            item_s = s_list[d - 1]
            if item_s == item_l:
                overlap += 1
            else:
                if item_s in seen_l:
                    overlap += 1
                if item_l in seen_s:
                    overlap += 1
            seen_s.add(item_s)
        else:
            if item_l in seen_s:
                overlap += 1
        seen_l.add(item_l)
        x.append(overlap)

    x_s, x_l = x[s - 1], x[l - 1]
    term1 = sum((x[d - 1] / d) * p**d for d in range(1, l + 1))
    term2 = sum((x_s * (d - s) / (s * d)) * p**d for d in range(s + 1, l + 1))
    tail = ((x_l - x_s) / l + x_s / s) * p**l
    return float((1 - p) / p * (term1 + term2) + tail)


# --------------------------------------------------------------------------- #
# Query variants
# --------------------------------------------------------------------------- #
def variant_destop(query: str, rng: np.random.Generator) -> str:
    """A: remove stopwords."""
    sw = stopwords()
    kept = [w for w in query.split() if w.lower().strip(".,?!;:") not in sw]
    return " ".join(kept) if kept else query


def variant_shuffle(query: str, rng: np.random.Generator) -> str:
    """B: permute word order (bag-of-words content held fixed)."""
    words = query.split()
    if len(words) < 2:
        return query
    for _ in range(10):
        perm = list(rng.permutation(len(words)))
        out = [words[i] for i in perm]
        if out != words:
            return " ".join(out)
    return " ".join(reversed(words))


def _depluralize(w: str) -> str | None:
    lw = w.lower()
    if lw in _IRREGULAR:
        return _IRREGULAR[lw]
    if lw.endswith("ies") and len(lw) > 4:
        return lw[:-3] + "y"
    if lw.endswith("ses") and len(lw) > 4:
        return lw[:-2]
    if lw.endswith("s") and not lw.endswith("ss") and len(lw) > 3:
        return lw[:-1]
    return None


def _pluralize(w: str) -> str:
    lw = w.lower()
    if lw.endswith(("s", "x", "z", "ch", "sh")):
        return lw + "es"
    if lw.endswith("y") and len(lw) > 2 and lw[-2] not in "aeiou":
        return lw[:-1] + "ies"
    return lw + "s"


def _split_punct(w: str) -> tuple[str, str]:
    """Split a token into its alphabetic core and trailing punctuation."""
    i = len(w)
    while i > 0 and not w[i - 1].isalnum():
        i -= 1
    return w[:i], w[i:]


def variant_synonym(
    query: str, rng: np.random.Generator, vocab: frozenset[str] | None = None
) -> str:
    """C: synonym substitution plus a singular/plural flip.

    Both edits are guarded, because an edit that is *not* meaning-preserving
    turns this from a null intervention into a real one and the measurement
    stops meaning anything:

    * Number flips are attempted only on tokens that are all-lowercase in the
      original, so acronyms ("ECMO") and proper nouns ("Ornish") are left
      alone.
    * A flip is accepted only if the flipped form actually occurs in the corpus
      vocabulary. That rejects the confident nonsense a naive rule produces -
      "deafness" -> "deafnesses", "scoring" -> "scorings" - while accepting
      genuine alternations like "acid" -> "acids".
    """
    words = query.split()
    out = list(words)
    changed = False

    for i, w in enumerate(words):
        core, punct = _split_punct(w)
        if core.lower() in SYNONYMS:
            repl = SYNONYMS[core.lower()]
            if core[:1].isupper():
                repl = repl[:1].upper() + repl[1:]
            out[i] = repl + punct
            changed = True

    sw = stopwords()
    cands = [
        i
        for i, w in enumerate(words)
        if (c := _split_punct(w)[0])
        and c.islower()
        and c.isalpha()
        and len(c) > 3
        and c not in sw
    ]
    for j in rng.permutation(len(cands)) if cands else []:
        i = cands[int(j)]
        core, punct = _split_punct(words[i])
        flipped = _depluralize(core) or _pluralize(core)
        if not flipped or flipped == core:
            continue
        if vocab is not None and flipped not in vocab:
            continue  # not a real word in this corpus - would not preserve meaning
        out[i] = flipped + punct
        changed = True
        break

    return " ".join(out) if changed else query


VARIANT_FNS = {
    "destop": variant_destop,
    "shuffle": variant_shuffle,
    "synonym": variant_synonym,
}


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def rankings(pipe: RetrievalPipeline, query: str, k: int = config.K_FINAL) -> dict[str, list[str]]:
    """Top-k ranking under each retriever configuration."""
    return {
        "bm25_only": pipe.rank_bm25_only(query, k),
        "dense_only": pipe.rank_dense_only(query, k),
        "full": pipe.run(query).top(k),
    }


def sample_queries(qids: list[str], n: int | None = None) -> list[str]:
    """Deterministic random sample of queries.

    Not the first ``n`` by id: NFCorpus ids are grouped so that a prefix is
    dominated by single-word queries ("deafness", "ECMO"), for which stopword
    removal and word-order shuffling are no-ops. That would make the pipeline
    look far more stable than it is.
    """
    n = n or config.STABILITY_N_QUERIES
    if n >= len(qids):
        return list(qids)
    rng = np.random.default_rng(config.SEED)
    idx = sorted(rng.choice(len(qids), size=n, replace=False).tolist())
    return [qids[i] for i in idx]


def corpus_vocab(pipe: RetrievalPipeline) -> frozenset[str]:
    """Every content token that occurs anywhere in the corpus.

    Used as a cheap "is this a real word here" oracle when deciding whether a
    singular/plural flip preserves meaning.
    """
    v: set[str] = set()
    for toks in pipe.corpus.doc_content_tokens:
        v |= toks
    return frozenset(v)


def run_stability(pipe: RetrievalPipeline, queries, qids: list[str]) -> pd.DataFrame:
    """RBO@10 between each variant's ranking and the original, per config."""
    rows: list[dict] = []
    vocab = corpus_vocab(pipe)
    t0 = time.time()
    for n, qid in enumerate(qids, 1):
        q0 = queries.texts[qid]
        base = rankings(pipe, q0)
        rng = np.random.default_rng(config.stable_seed("stab", qid))
        for vname in VARIANTS:
            fn = VARIANT_FNS[vname]
            qv = fn(q0, rng, vocab) if vname == "synonym" else fn(q0, rng)
            var = rankings(pipe, qv)
            for cfg in CONFIGS:
                rows.append(
                    {
                        "query_id": qid,
                        "variant": vname,
                        "config": cfg,
                        "rbo": rbo(base[cfg], var[cfg]),
                        "identical": base[cfg] == var[cfg],
                        "query_original": q0,
                        "query_variant": qv,
                        "unchanged_text": qv == q0,
                    }
                )
        if n % 10 == 0 or n == len(qids):
            print(
                f"[stability] {n}/{len(qids)} queries, {time.time() - t0:.0f}s",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean RBO per (variant, config), with the instability gap.

    Two means are reported. Some rewrites are genuine no-ops - a single-word
    query has no stopwords to drop and no word order to permute - and those
    contribute a trivial RBO of exactly 1.0. ``mean_rbo`` includes them (it
    answers "how stable is the pipeline over this query set"), while
    ``mean_rbo_changed`` conditions on the rewrite having actually altered the
    text (it answers "how stable is the pipeline when the query really is
    rewritten"). The second is the informative one.
    """
    g = (
        df.groupby(["variant", "config"], sort=True)
        .agg(
            mean_rbo=("rbo", "mean"),
            std_rbo=("rbo", "std"),
            min_rbo=("rbo", "min"),
            frac_identical=("identical", "mean"),
            n=("rbo", "size"),
        )
        .reset_index()
    )
    ch = df[~df["unchanged_text"]]
    gc = (
        ch.groupby(["variant", "config"], sort=True)
        .agg(mean_rbo_changed=("rbo", "mean"), n_changed=("rbo", "size"))
        .reset_index()
    )
    g = g.merge(gc, on=["variant", "config"], how="left")
    g["instability"] = 1.0 - g["mean_rbo"]
    g["instability_changed"] = 1.0 - g["mean_rbo_changed"]
    return g


def print_summary(summ: pd.DataFrame, df: pd.DataFrame) -> None:
    print("\n" + "=" * 84)
    print("COUNTERFACTUAL STABILITY - RBO@10 (p=0.9) vs the original ranking")
    print("A meaning-preserving edit should give RBO = 1; the gap is instability.")
    print("=" * 84)
    print(
        f"{'variant':<10}{'config':<13}{'mean RBO':>10}{'std':>8}{'min':>8}"
        f"{'identical':>11}{'RBO|changed':>13}{'n chg':>7}"
    )
    for _, r in summ.iterrows():
        mc = r["mean_rbo_changed"]
        mc_s = f"{mc:.4f}" if pd.notna(mc) else "n/a"
        nc = int(r["n_changed"]) if pd.notna(r["n_changed"]) else 0
        print(
            f"{r['variant']:<10}{r['config']:<13}{r['mean_rbo']:>10.4f}"
            f"{r['std_rbo']:>8.3f}{r['min_rbo']:>8.3f}"
            f"{r['frac_identical']:>10.0%}{mc_s:>13}{nc:>7}"
        )
    print("-" * 84)
    # Implementation check: BM25 is a bag of words, so shuffling must not move it.
    chk = summ[(summ["variant"] == "shuffle") & (summ["config"] == "bm25_only")]
    if len(chk):
        v = float(chk["mean_rbo"].iloc[0])
        ok = abs(v - 1.0) < 1e-9
        print(
            f"  SANITY: BM25 under word-order shuffle -> mean RBO = {v:.6f} "
            f"({'as expected, bag of words is order-invariant' if ok else 'UNEXPECTED - investigate'})"
        )
    noop = df.groupby("variant")["unchanged_text"].mean()
    print("  Fraction of queries the rewrite left textually unchanged: " +
          ", ".join(f"{k}={v:.0%}" for k, v in noop.items()))
    print("=" * 84 + "\n")


def plot_stability(summ: pd.DataFrame, df: pd.DataFrame, path=FIG_STABILITY) -> None:
    """Grouped bars of mean RBO plus the per-query distribution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = {"bm25_only": "#7fcdbb", "dense_only": "#2c7fb8", "full": "#253494"}

    ax = axes[0]
    x = np.arange(len(VARIANTS))
    w = 0.26
    for i, cfg in enumerate(CONFIGS):
        vals = [
            float(summ[(summ["variant"] == v) & (summ["config"] == cfg)]["mean_rbo"].iloc[0])
            for v in VARIANTS
        ]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=cfg, color=colors[cfg])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.axhline(1.0, color="k", ls="--", lw=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mean RBO@10 vs original")
    ax.set_title("Stability under meaning-preserving query edits")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    data, labels = [], []
    for v in VARIANTS:
        for cfg in CONFIGS:
            data.append(df[(df["variant"] == v) & (df["config"] == cfg)]["rbo"].to_numpy())
            labels.append(f"{v[:5]}\n{cfg[:5]}")
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(list(colors.values())[i % 3])
        patch.set_alpha(0.75)
    ax.axhline(1.0, color="k", ls="--", lw=0.9, alpha=0.7)
    ax.set_ylabel("RBO@10")
    ax.set_title("Per-query distribution")
    ax.tick_params(axis="x", labelsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[stability] wrote {path}")


# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """RBO edge cases. Cheap insurance against a silently wrong metric."""
    a = [f"d{i}" for i in range(10)]
    assert abs(rbo(a, a) - 1.0) < 1e-9, "identical lists must give RBO 1.0"
    assert rbo(a, [f"x{i}" for i in range(10)]) < 1e-6, "disjoint lists ~ 0"
    swapped = a.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    r = rbo(a, swapped)
    assert 0.8 < r < 1.0, f"adjacent swap should be near but below 1, got {r}"
    assert rbo(a, list(reversed(a))) < rbo(a, swapped), "reversal must score lower"
    print("[stability] RBO self-test passed")


def main() -> None:
    config.set_seeds()
    _self_test()
    t0 = time.time()
    corpus, queries = load_corpus_and_queries()
    pipe = RetrievalPipeline(corpus)
    qids = sample_queries(select_queries(queries))
    print(f"[stability] {len(qids)} queries x {len(VARIANTS)} variants x {len(CONFIGS)} configs")
    df = run_stability(pipe, queries, qids)
    df.to_parquet(OUT_PARQUET, index=False)
    summ = summarize(df)
    summ.to_csv(OUT_CSV, index=False)
    print(f"[stability] wrote {OUT_CSV}")
    print_summary(summ, df)
    plot_stability(summ, df)
    print(f"[stability] wall clock {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
