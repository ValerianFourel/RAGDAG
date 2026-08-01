"""Module 4 - is naive concept attribution confounded?

Modules 2 and 3 intervene. This module deliberately does *not*: it asks what an
analyst would conclude from the observational baseline run alone, and whether
that conclusion survives confounder adjustment.

Setup, on the un-intervened pipeline:

* unit      - a (query, candidate document) pair
* treatment D - 1 if the document contains concept term ``c``
* outcome   Y - the cross-encoder score the pipeline assigned that pair
* confounders X - document length, document embedding norm, query-document
  lexical overlap, query length

X is a genuine confounder set rather than a set of controls-of-convenience:
longer documents both contain more concept terms (affecting D) and give a
cross-encoder more material to match against (affecting Y), so ``doc_len``
opens a back-door path D <- doc_len -> Y. The same argument applies to lexical
overlap and embedding norm.

The contrast is between the naive OLS coefficient of Y on D - which leaves the
back-door open - and a partially linear DoubleML estimate with LightGBM
nuisance learners and cross-fitting, which closes it without imposing a
functional form on the confounding.

Run standalone::

    python -m dml_analysis
"""

from __future__ import annotations

import time
from collections import Counter

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
    jaccard,
    load_corpus_and_queries,
    select_queries,
    tokenize,
)

OUT_CSV = config.RESULTS_DIR / "dml_comparison.csv"
OUT_PANEL = config.RESULTS_DIR / "dml_panel.parquet"
FIG_DML = config.RESULTS_DIR / "fig_dml_naive_vs_adjusted.png"

CONFOUNDERS = ["doc_len", "doc_emb_norm", "lex_overlap", "query_len"]

#: Hand-rolled medical/biomedical vocabulary for concept selection. NFCorpus is
#: a nutrition-and-medicine corpus, so its highest-frequency terms are a mix of
#: medical content words and research boilerplate ("study", "results",
#: "data"). Restricting to this list keeps the concepts substantive; it is
#: written out in full so the selection is auditable rather than a black box.
MEDICAL_LEXICON: frozenset[str] = frozenset(
    """
    cancer tumor tumour carcinoma malignant metastasis oncology
    cardiovascular heart cardiac stroke hypertension cholesterol arterial
    diabetes insulin glucose glycemic obesity obese weight
    diet dietary nutrition nutrient food foods eating vegetarian vegan
    protein proteins fat fats fatty lipid lipids fiber carbohydrate
    vitamin vitamins mineral calcium iron zinc magnesium sodium potassium
    antioxidant antioxidants inflammation inflammatory oxidative
    disease diseases disorder syndrome symptoms clinical patients patient
    treatment therapy therapeutic drug drugs medication dose dosage
    risk mortality morbidity incidence prevalence epidemiological
    cells cell cellular tissue tissues plasma serum blood
    liver kidney renal hepatic intestinal gut microbiota bacteria
    immune immunity infection viral bacterial
    metabolism metabolic enzyme hormone hormones estrogen testosterone
    supplementation supplements intake consumption exposure
    breast prostate colorectal colon lung skin bone
    pregnancy infant children adults elderly women men
    meat dairy milk soy fruit fruits vegetables grain grains fish
    """.split()
)


# --------------------------------------------------------------------------- #
# Concept selection
# --------------------------------------------------------------------------- #
def select_concepts(
    corpus: Corpus, n: int = config.DML_N_CONCEPTS
) -> list[tuple[str, float]]:
    """Pick the ``n`` highest-document-frequency medical terms in the band
    ``[DML_DF_MIN, DML_DF_MAX]``.

    The band matters for identification: a term present in almost every
    document gives no treatment variation, and a term present in a handful
    gives no power. Both extremes make the DML variance blow up.
    """
    df: Counter[str] = Counter()
    for toks in corpus.doc_content_tokens:
        df.update(toks)
    n_docs = len(corpus)
    band = [
        (t, c / n_docs)
        for t, c in df.items()
        if config.DML_DF_MIN <= c / n_docs <= config.DML_DF_MAX
        and t in MEDICAL_LEXICON
    ]
    band.sort(key=lambda x: (-x[1], x[0]))
    chosen = band[:n]
    if not chosen:
        # MEDICAL_LEXICON is NFCorpus-specific. On a non-biomedical collection
        # the intersection is empty and every downstream table would be silently
        # blank rather than absent. Say so loudly; WP-6 replaces the lexicon
        # with a frequency-stratified vocabulary sample.
        print(
            f"[dml] WARNING: no concept terms found for {config.DATASET}. The "
            f"medical lexicon ({len(MEDICAL_LEXICON)} terms) does not intersect "
            f"this corpus in the df band [{config.DML_DF_MIN:.0%}, "
            f"{config.DML_DF_MAX:.0%}]. Module 4 will be reported as "
            "INCONCLUSIVE, not as a null result."
        )
        return []
    print(
        f"[dml] concept terms chosen (df in [{config.DML_DF_MIN:.0%}, "
        f"{config.DML_DF_MAX:.0%}], medical lexicon): "
        + ", ".join(f"'{t}' (df={d:.1%})" for t, d in chosen)
    )
    return chosen


# --------------------------------------------------------------------------- #
# Panel construction
# --------------------------------------------------------------------------- #
def build_panel(
    pipe: RetrievalPipeline,
    queries: Queries,
    baseline: dict[str, BaselineRun],
    concepts: list[str],
) -> pd.DataFrame:
    """Assemble the observational (query, candidate document) panel.

    Units are the *candidate pool*, not the top-k of the reranked list. Taking
    the reranked top-k would select units on the outcome Y and induce
    collider bias on top of the confounding we are trying to measure.
    """
    corpus = pipe.corpus
    rows: list[dict] = []
    for qid, run in baseline.items():
        qtext = run.query_text
        qtok = content_tokens(qtext)
        qlen = float(len(tokenize(qtext)))
        ce = dict(run.reranked)
        for doc_id in run.candidates:
            i = corpus.idx(doc_id)
            dtoks = corpus.doc_content_tokens[i]
            rec = {
                "query_id": qid,
                "doc_id": doc_id,
                "ce_score": float(ce.get(doc_id, np.nan)),
                "doc_len": float(corpus.doc_len[i]),
                "doc_emb_norm": float(pipe.doc_emb_norm[i]),
                "lex_overlap": jaccard(qtok, dtoks),
                "query_len": qlen,
                "bm25": run.bm25_scores_cand.get(doc_id, 0.0),
                "dense": run.dense_scores_cand.get(doc_id, 0.0),
                "provenance": run.provenance.get(doc_id, "none"),
            }
            for c in concepts:
                rec[f"has_{c}"] = int(c in dtoks)
            rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=["ce_score"]).reset_index(drop=True)
    print(
        f"[dml] panel: {len(df)} (query, doc) units over "
        f"{df['query_id'].nunique()} queries "
        f"({len(df) / max(1, df['query_id'].nunique()):.1f} docs/query)"
    )
    return df


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def naive_ols(df: pd.DataFrame, treat: str) -> dict[str, float]:
    """OLS of Y on D alone - the unadjusted, back-door-open estimate.

    Standard errors are clustered by query, because candidate documents from
    the same query share a query embedding and a reranker context.
    """
    import statsmodels.api as sm

    X = sm.add_constant(df[[treat]].astype(float))
    model = sm.OLS(df["ce_score"].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": df["query_id"]}
    )
    coef = float(model.params[treat])
    se = float(model.bse[treat])
    return {
        "naive_coef": coef,
        "naive_se": se,
        "naive_ci_lo": coef - 1.96 * se,
        "naive_ci_hi": coef + 1.96 * se,
    }


def dml_plr(df: pd.DataFrame, treat: str) -> dict[str, float]:
    """Partially linear DoubleML estimate with LightGBM nuisance learners.

    Cross-fitting (``DML_N_FOLDS`` folds, ``DML_N_REP`` repetitions) is what
    licenses the use of a flexible machine learner for the nuisance functions
    E[Y|X] and E[D|X] without the resulting bias contaminating the treatment
    coefficient.
    """
    import doubleml as dml
    from lightgbm import LGBMClassifier, LGBMRegressor

    data = dml.DoubleMLData(
        df[["ce_score", treat] + CONFOUNDERS].astype(float),
        y_col="ce_score",
        d_cols=treat,
        x_cols=CONFOUNDERS,
    )
    # ``deterministic`` + ``force_row_wise`` make LightGBM reproducible
    # regardless of how many threads it is given, so the merge pass on a
    # 32-core node produces exactly the results a single-core run would.
    common = dict(
        n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1,
        random_state=config.SEED, n_jobs=config.DML_N_JOBS,
        deterministic=True, force_row_wise=True,
    )
    ml_l = LGBMRegressor(**common)
    # D is binary, so E[D|X] is a probability: a classifier is the right
    # nuisance learner. Fall back to a regressor if the installed DoubleML
    # refuses classifiers for the PLR score.
    try:
        obj = dml.DoubleMLPLR(
            data, ml_l=ml_l, ml_m=LGBMClassifier(**common),
            n_folds=config.DML_N_FOLDS, n_rep=config.DML_N_REP,
        )
        obj.fit()
        learner_m = "LGBMClassifier"
    except Exception:
        obj = dml.DoubleMLPLR(
            data, ml_l=LGBMRegressor(**common), ml_m=LGBMRegressor(**common),
            n_folds=config.DML_N_FOLDS, n_rep=config.DML_N_REP,
        )
        obj.fit()
        learner_m = "LGBMRegressor"

    ci = obj.confint(level=0.95)
    return {
        "dml_coef": float(obj.coef[0]),
        "dml_se": float(obj.se[0]),
        "dml_ci_lo": float(ci.iloc[0, 0]),
        "dml_ci_hi": float(ci.iloc[0, 1]),
        "dml_pval": float(obj.pval[0]),
        "dml_ml_m": learner_m,
    }


def analyse(df: pd.DataFrame, concepts: list[tuple[str, float]]) -> pd.DataFrame:
    """Run both estimators for every concept and flag disagreement."""
    recs = []
    for term, dfreq in concepts:
        treat = f"has_{term}"
        t0 = time.time()
        rec: dict[str, object] = {
            "concept": term,
            "doc_freq": dfreq,
            "n_treated": int(df[treat].sum()),
            "n_units": int(len(df)),
        }
        rec.update(naive_ols(df, treat))
        rec.update(dml_plr(df, treat))
        # The criterion: does confounder adjustment move the estimate outside
        # what the naive analysis would have called plausible?
        rec["naive_outside_dml_ci"] = bool(
            rec["naive_coef"] < rec["dml_ci_lo"] or rec["naive_coef"] > rec["dml_ci_hi"]
        )
        rec["abs_shift"] = abs(float(rec["naive_coef"]) - float(rec["dml_coef"]))
        rec["seconds"] = time.time() - t0
        recs.append(rec)
        print(f"[dml] {term}: naive={rec['naive_coef']:+.4f} dml={rec['dml_coef']:+.4f} "
              f"({rec['seconds']:.1f}s)", flush=True)
    if not recs:
        # Zero concepts (the lexicon does not intersect this corpus). Return a
        # frame that still carries the schema: a column-less frame writes a
        # headerless CSV, which pd.read_csv cannot parse at all.
        return pd.DataFrame(columns=[
            "concept", "doc_freq", "n_treated", "n_units",
            "naive_coef", "naive_se", "naive_ci_lo", "naive_ci_hi",
            "dml_coef", "dml_se", "dml_ci_lo", "dml_ci_hi", "dml_pval",
            "naive_outside_dml_ci", "abs_shift", "seconds",
        ])
    return pd.DataFrame(recs)


def print_comparison(res: pd.DataFrame) -> bool:
    """Print the side-by-side table and return whether confounding was found."""
    if res is None or res.empty:
        print("\n[dml] no concepts analysed - Module 4 produced no estimates.")
        return False
    print("\n" + "=" * 96)
    print("DOUBLE MACHINE LEARNING - naive vs confounder-adjusted concept effect")
    print("outcome Y = cross-encoder score;  treatment D = document contains concept")
    print("=" * 96)
    print(
        f"{'concept':<14}{'df':>7}{'n_treat':>9}{'naive':>10}{'naive 95% CI':>22}"
        f"{'DML':>10}{'DML 95% CI':>22}{'outside':>9}"
    )
    for _, r in res.iterrows():
        nci = f"[{r['naive_ci_lo']:+.3f},{r['naive_ci_hi']:+.3f}]"
        dci = f"[{r['dml_ci_lo']:+.3f},{r['dml_ci_hi']:+.3f}]"
        print(
            f"{r['concept']:<14}{r['doc_freq']:>6.1%}{int(r['n_treated']):>9}"
            f"{r['naive_coef']:>10.4f}{nci:>22}{r['dml_coef']:>10.4f}{dci:>22}"
            f"{('YES' if r['naive_outside_dml_ci'] else 'no'):>9}"
        )
    detected = bool(res["naive_outside_dml_ci"].any())
    print("-" * 96)
    print(f"CONFOUNDING DETECTED: {'YES' if detected else 'NO'}")
    print("=" * 96 + "\n")
    return detected


def plot_dml(res: pd.DataFrame, path=FIG_DML) -> None:
    """Dot-and-whisker comparison, naive vs DML, per concept.

    Returns early on an empty frame. ``select_concepts`` legitimately returns
    nothing when the concept lexicon does not intersect the collection, and
    ``print_comparison`` already handles that by reporting INCONCLUSIVE. Without
    the same guard here the KeyError propagated out of the merge pass and killed
    the *whole* run - stability and the report included - for two collections
    whose GPU work had already completed. A stage with nothing to say must not
    be able to destroy the stages that do.
    """
    if res is None or res.empty or "naive_coef" not in res.columns:
        print("[dml] no concept estimates - skipping figure.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 1.4 + 1.1 * len(res)))
    y = np.arange(len(res))
    off = 0.16
    for label, colour, key, sign in (
        ("naive OLS", "#d95f0e", "naive", +1),
        ("DoubleML (adjusted)", "#2c7fb8", "dml", -1),
    ):
        c = res[f"{key}_coef"].to_numpy(dtype=float)
        lo = res[f"{key}_ci_lo"].to_numpy(dtype=float)
        hi = res[f"{key}_ci_hi"].to_numpy(dtype=float)
        ax.errorbar(
            c, y + sign * off, xerr=[c - lo, hi - c], fmt="o", color=colour,
            capsize=4, label=label, markersize=7, lw=2,
        )
    ax.axvline(0, color="k", lw=0.9, ls="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.concept}\n(df={r.doc_freq:.0%})" for r in res.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel("effect of concept presence on cross-encoder score")
    ax.set_title("Naive vs confounder-adjusted concept effects")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[dml] wrote {path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    config.set_seeds()
    t0 = time.time()
    corpus, queries = load_corpus_and_queries()
    pipe = RetrievalPipeline(corpus)
    qids = select_queries(queries)
    baseline = compute_baseline(pipe, queries, qids)

    concepts = select_concepts(corpus)
    panel = build_panel(pipe, queries, baseline, [c for c, _ in concepts])
    panel.to_parquet(OUT_PANEL, index=False)

    res = analyse(panel, concepts)
    res.to_csv(OUT_CSV, index=False)
    print(f"[dml] wrote {OUT_CSV}")
    print_comparison(res)
    plot_dml(res)
    print(f"[dml] wall clock {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
