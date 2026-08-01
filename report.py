"""Module 6b - assemble results/REPORT.md and adjudicate the MVP criteria.

The verdict logic is deliberately mechanical: each criterion is a predicate over
numbers already written to disk by modules 2-4, so the report cannot flatter the
experiment. A criterion that cannot be evaluated because its artefact is missing
is reported as ``INCONCLUSIVE``, never silently as a pass.

Run standalone (after ``run_all``)::

    python -m report
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
import dml_analysis as D
import interventions as I
import mediation as M
import stability as S

REPORT_PATH = config.RESULTS_DIR / "REPORT.md"

FIGURES = [
    (I.FIG_EFFECTS, "Intervention effect sizes"),
    (M.FIG_MEDIATION, "Mediation ratio by path"),
    (D.FIG_DML, "Naive vs DML-adjusted concept effects"),
    (S.FIG_STABILITY, "Counterfactual stability"),
]


# --------------------------------------------------------------------------- #
# Artefact loading
# --------------------------------------------------------------------------- #
@dataclass
class Artifacts:
    """Whatever the pipeline managed to produce. Missing pieces stay ``None``."""

    ndcg: dict | None = None
    timings: dict | None = None
    interventions: pd.DataFrame | None = None
    mediation: pd.DataFrame | None = None
    ratio: pd.DataFrame | None = None
    dml: pd.DataFrame | None = None
    stability: pd.DataFrame | None = None
    stability_raw: pd.DataFrame | None = None


def load_artifacts() -> Artifacts:
    a = Artifacts()

    def _j(p):
        return json.load(open(p)) if p.exists() else None

    a.ndcg = _j(config.RESULTS_DIR / "baseline_ndcg.json")
    a.timings = _j(config.RESULTS_DIR / "timings.json")
    if I.OUT_PARQUET.exists():
        a.interventions = pd.read_parquet(I.OUT_PARQUET)
    if M.OUT_PARQUET.exists():
        a.mediation = pd.read_parquet(M.OUT_PARQUET)
        a.ratio = M.mediation_ratio(a.mediation)
    if D.OUT_CSV.exists():
        try:
            a.dml = pd.read_csv(D.OUT_CSV)
        except pd.errors.EmptyDataError:
            # A zero-concept run used to write a headerless CSV; treat it the
            # same as the file being absent rather than killing the report.
            a.dml = None
    if S.OUT_PARQUET.exists():
        a.stability_raw = pd.read_parquet(S.OUT_PARQUET)
        a.stability = S.summarize(a.stability_raw)
    return a


# --------------------------------------------------------------------------- #
# Criteria
# --------------------------------------------------------------------------- #
@dataclass
class Criterion:
    name: str
    status: str  # PASS | FAIL | INCONCLUSIVE
    detail: str
    interpretation: str


def criterion_interventions(a: Artifacts) -> Criterion:
    """C1: treated |Delta-rank| exceeds control, bootstrap CI excluding zero."""
    if a.interventions is None or a.interventions.empty:
        return Criterion("INTERVENTION EFFECTS EXIST", "INCONCLUSIVE",
                         "interventions.parquet missing", "Module 2 did not run.")
    pt, lo, hi = I.cluster_bootstrap_diff(a.interventions, "delta_rank", absolute=True)
    ok = bool(lo > 0)
    detail = f"mean |Delta-rank| treatment - control = {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
    if ok:
        interp = (
            "Injecting a term drawn from the target document moves that document "
            "measurably more than injecting an unrelated term of the same form, so "
            "do() on the query has a real, isolable effect on the ranking."
        )
    else:
        interp = (
            "The treated and control arms are not distinguishable at 95%: query "
            "term injection does not produce an effect this design can separate "
            "from the mechanical effect of lengthening the query."
        )
    return Criterion("INTERVENTION EFFECTS EXIST", "PASS" if ok else "FAIL", detail, interp)


def criterion_mediation(a: Artifacts) -> Criterion:
    """C2: no path above 90% of the share, and shares move >=10pp across configs."""
    if a.ratio is None or len(a.ratio) < 2:
        return Criterion("MEDIATION IS NON-DEGENERATE", "INCONCLUSIVE",
                         "mediation.parquet missing or single config",
                         "Module 3 did not run in both configurations.")
    shares = {
        r["cfg"]: {p: float(r[f"{p}_share_pct"]) for p in M.PATHS}
        for _, r in a.ratio.iterrows()
    }
    max_share = max(max(v.values()) for v in shares.values())
    not_degenerate = max_share <= 90.0
    cfgs = list(shares)
    deltas = {p: abs(shares[cfgs[0]][p] - shares[cfgs[1]][p]) for p in M.PATHS}
    max_shift = max(deltas.values())
    shifts = max_shift >= 10.0
    ok = not_degenerate and shifts
    detail = (
        f"largest single-path share = {max_share:.1f}% (needs <=90%); "
        f"largest share shift between configs = {max_shift:.1f} pp (needs >=10 pp)"
    )
    if ok:
        interp = (
            "Retrieval credit is genuinely divided between the lexical, dense and "
            "reranking paths, and the division depends on the first-stage "
            "architecture - so path attribution carries information that a single "
            "end-to-end score does not."
        )
    elif not not_degenerate:
        interp = (
            "One path absorbs almost all of the effect, so the decomposition adds "
            "little over simply naming that stage."
        )
    else:
        interp = (
            "The shares barely move when the first stage is changed, so the "
            "attribution may be a property of the metric rather than the architecture."
        )
    return Criterion("MEDIATION IS NON-DEGENERATE", "PASS" if ok else "FAIL", detail, interp)


def criterion_confounding(a: Artifacts) -> Criterion:
    """C3: at least one concept whose naive estimate sits outside the DML CI."""
    if a.dml is None or a.dml.empty or "naive_outside_dml_ci" not in a.dml.columns:
        return Criterion(
            "CONFOUNDING IS LIVE", "INCONCLUSIVE",
            "no concept estimates available",
            "Module 4 produced no concepts on this collection - most likely the "
            "medical lexicon does not intersect this corpus. Absence of estimates "
            "is not evidence of no confounding.")
    hits = a.dml[a.dml["naive_outside_dml_ci"]]
    ok = len(hits) > 0
    names = ", ".join(f"'{r.concept}'" for r in hits.itertuples()) or "none"
    worst = a.dml.loc[a.dml["abs_shift"].idxmax()]
    detail = (
        f"{len(hits)}/{len(a.dml)} concepts with naive estimate outside the DML CI "
        f"({names}); largest naive-vs-DML shift = {worst['abs_shift']:.4f} "
        f"on '{worst['concept']}'"
    )
    if ok:
        interp = (
            "Unadjusted concept attribution is measurably biased: confounder "
            "adjustment moves the estimate outside the range the naive analysis "
            "would have accepted, so observational explanations of this pipeline "
            "need adjustment to be trustworthy."
        )
    else:
        interp = (
            "Confounder adjustment does not move the estimates outside their naive "
            "intervals here; on this panel the naive attribution is not detectably "
            "biased."
        )
    return Criterion("CONFOUNDING IS LIVE", "PASS" if ok else "FAIL", detail, interp)


def evaluate(a: Artifacts) -> tuple[list[Criterion], str]:
    crits = [criterion_interventions(a), criterion_mediation(a), criterion_confounding(a)]
    n_pass = sum(c.status == "PASS" for c in crits)
    n_incon = sum(c.status == "INCONCLUSIVE" for c in crits)
    if n_pass == 3:
        overall = "yes"
    elif n_pass <= 1 and n_incon == 0:
        overall = "no"
    else:
        overall = "mixed"
    return crits, overall


def verdict_text(a: Artifacts | None = None) -> str:
    a = a or load_artifacts()
    crits, overall = evaluate(a)
    lines = ["=" * 78, "MVP SUCCESS CRITERIA", "=" * 78]
    for i, c in enumerate(crits, 1):
        lines.append(f"[{c.status:^12}] {i}. {c.name}")
        lines.append(f"               {c.detail}")
        lines.append(f"               -> {c.interpretation}")
        lines.append("")
    lines.append("-" * 78)
    lines.append(f"PROCEED TO FULL PAPER: {overall.upper()}")
    lines.append(_overall_note(overall, crits))
    lines.append("=" * 78)
    return "\n".join(lines)


def _overall_note(overall: str, crits: list[Criterion]) -> str:
    if overall == "yes":
        return (
            "All three quantities exist, are measurable, and vary. The causal "
            "framing is doing work the standard evaluation does not."
        )
    if overall == "no":
        return (
            "The causal quantities are either absent or degenerate at this scale. "
            "Reconsider the design before investing in a full study."
        )
    failed = [c.name for c in crits if c.status != "PASS"]
    return (
        "Some quantities are live and some are not: "
        + ", ".join(failed)
        + " did not clear the bar. Worth pursuing, but that component needs "
        "rethinking or more data first."
    )


# --------------------------------------------------------------------------- #
# Markdown assembly
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        if isinstance(v, (bool, np.bool_)):
            return "YES" if v else "no"
        return str(v)

    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _fmt_time(s: float) -> str:
    return f"{s:.1f}s" if s < 90 else (f"{s / 60:.1f} min" if s < 5400 else f"{s / 3600:.2f} h")


def build_report(path=REPORT_PATH) -> str:
    a = load_artifacts()
    crits, overall = evaluate(a)
    L: list[str] = []
    add = L.append

    add("# Causal Retrieval MVP — Results\n")
    add(
        "Testing whether a retrieval pipeline's behaviour can be explained "
        "*causally*: do()-interventions on the query, exact mediation by "
        "stage-freezing, a DoubleML confounding check, and counterfactual "
        "stability — on BEIR NFCorpus.\n"
    )

    # ---- verdict up front ----
    add("## Verdict\n")
    add(f"**Proceed to full paper: {overall.upper()}**\n")
    add("| # | Criterion | Result | Evidence |")
    add("|---|---|---|---|")
    for i, c in enumerate(crits, 1):
        add(f"| {i} | {c.name} | **{c.status}** | {c.detail} |")
    add("")
    for i, c in enumerate(crits, 1):
        add(f"{i}. *{c.name}* — {c.interpretation}\n")
    add(f"\n{_overall_note(overall, crits)}\n")

    # ---- config ----
    add("## Configuration\n")
    cfg = config.summary()
    add("| setting | value |")
    add("|---|---|")
    for k, v in cfg.items():
        add(f"| {k} | {v} |")
    add(f"| ce_max_length | {config.CE_MAX_LENGTH} |")
    add(f"| candidate pool | union of top-{config.K_CANDIDATES} BM25 and top-{config.K_CANDIDATES} dense |")
    add("")
    if a.timings:
        add("Wall clock per module:\n")
        add("| module | time |")
        add("|---|---|")
        for k, v in a.timings.items():
            add(f"| {k} | {_fmt_time(v)} |")
        add("")

    # ---- baseline ----
    add("## 1. Baseline pipeline quality\n")
    if a.ndcg:
        add("nDCG@10 on NFCorpus test:\n")
        add("| configuration | nDCG@10 |")
        add("|---|---|")
        for k in ("bm25_only", "dense_only", "full_pipeline"):
            add(f"| {k} | {a.ndcg[k]:.4f} |")
        add(f"\nQueries evaluated: {a.ndcg['n_queries']}.\n")
        best_single = max(a.ndcg["bm25_only"], a.ndcg["dense_only"])
        if a.ndcg["full_pipeline"] > best_single:
            add(
                f"The full pipeline beats both single channels "
                f"(+{a.ndcg['full_pipeline'] - best_single:.4f} over the better of the two), "
                "so the reranker is wired in correctly and the SCM below describes a "
                "pipeline that actually works.\n"
            )
        else:
            known = a.ndcg.get("dataset") in config.KNOWN_RERANKER_HARMFUL
            add(
                f"> **The reranker degrades this collection.** Full pipeline "
                f"{a.ndcg['full_pipeline']:.4f} vs best single channel "
                f"{best_single:.4f}.\n>\n"
                + ("> This is a known property of this dataset rather than a "
                   "configuration error: the MS MARCO-trained cross-encoder does not "
                   "transfer to this query style.\n>\n" if known else
                   "> Check for a mis-wired reranker before trusting anything below.\n>\n")
                + "> Every mediation share below therefore describes causal "
                  "responsibility for the *intervention's effect*, on a pipeline whose "
                  "reranking stage is not beneficial. Causal responsibility and "
                  "usefulness are different questions; do not report the shares as "
                  "evidence that this configuration works.\n"
            )

    # ---- interventions ----
    add("## 2. do()-interventions on the query\n")
    if a.interventions is not None:
        df = a.interventions
        add(
            f"{len(df)} interventions over {df['query_id'].nunique()} queries and "
            f"{df.groupby(['query_id', 'doc_id']).ngroups} (query, target document) pairs. "
            "Treatment injects a TF-IDF-weighted term drawn from the target document; "
            "control injects a corpus term absent from both query and document. "
            "All CIs are cluster bootstraps over queries "
            f"({config.N_BOOTSTRAP} resamples).\n"
        )
        summ = I.summarize(df)
        summ_disp = summ.assign(
            estimate=summ["estimate"].map("{:+.4f}".format),
            ci=[f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(summ["ci_lo"], summ["ci_hi"])],
        )[["arm", "metric", "estimate", "ci", "n"]]
        add(_md_table(summ_disp))
        add("")
        add("Treatment − control contrasts:\n")
        rows = []
        for col, label, absolute in (
            ("delta_rank", "Delta-rank", False),
            ("delta_rank", "|Delta-rank|", True),
            ("delta_ce", "Delta cross-encoder score", False),
        ):
            pt, lo, hi = I.cluster_bootstrap_diff(df, col, absolute=absolute)
            rows.append({
                "contrast": label,
                "estimate": f"{pt:+.4f}",
                "95% CI": f"[{lo:+.4f}, {hi:+.4f}]",
                "excludes 0": "YES" if (lo > 0 or hi < 0) else "no",
            })
        add(_md_table(pd.DataFrame(rows)))
        add("")
        ctrl_bm25 = df.loc[df["arm"] == "control", "delta_bm25"].abs().max()
        add(
            f"*Design check:* the largest absolute BM25 change in the control arm is "
            f"{ctrl_bm25:.2e}. Control terms are drawn to be absent from the target "
            "document, so the document's own BM25 score cannot move; any control-arm "
            "rank change is transmitted through competition from other documents and "
            "through the dense and reranking stages.\n"
        )
        if "base_censored" in df.columns:
            frac = float(df["base_censored"].mean())
            add(
                f"*Censoring:* {frac:.0%} of interventions target a document whose "
                f"baseline rank is already at the sentinel ({config.MISSING_RANK}), so "
                "their Delta-rank can only be non-negative. Targets are drawn from the "
                "whole candidate pool rather than the reranked top-k on purpose — "
                "restricting to well-ranked targets would condition the experiment on "
                "retrieval success and mechanically inflate the reranker's share in "
                "section 3. Restricted to uncensored baselines the contrast is:\n"
            )
            sub = df[~df["base_censored"]]
            if len(sub) and sub["arm"].nunique() == 2:
                pt, lo, hi = I.cluster_bootstrap_diff(sub, "delta_rank", absolute=True)
                add(
                    f"- mean |Delta-rank| treatment − control = **{pt:+.4f}** "
                    f"95% CI [{lo:+.4f}, {hi:+.4f}] "
                    f"({'excludes' if (lo > 0 or hi < 0) else 'includes'} 0, n={len(sub)})\n"
                )
        add(f"![Intervention effects]({I.FIG_EFFECTS.name})\n")

    # ---- mediation ----
    add("## 3. Exact path-specific effects\n")
    if a.ratio is not None:
        add(
            "Each treated pair is evaluated in five worlds (baseline, full, and three "
            "freezes). Because the pipeline is deterministic, the cross-world quantity "
            "`Y(Q1, M(Q0))` is *computed*, not estimated — there is no "
            "sequential-ignorability assumption to defend.\n"
        )
        add(
            "```\n"
            "reranker = r(baseline)          - r(freeze_candidates)\n"
            "lexical  = r(freeze_candidates) - r(freeze_dense)\n"
            "dense    = r(freeze_candidates) - r(freeze_bm25)\n"
            "residual = total - (reranker + lexical + dense)\n"
            "```\n"
        )
        share_tbl = a.ratio[
            ["cfg", "n_pairs", "mean_abs_total"] + [f"{p}_share_pct" for p in M.PATHS]
        ].rename(columns={f"{p}_share_pct": f"{p} %" for p in M.PATHS})
        add("**Mediation ratio** — share of mean |total effect| by path:\n")
        add(_md_table(share_tbl, "{:.2f}"))
        add("")
        signed = a.ratio[["cfg"] + [f"{p}_mean" for p in M.PATHS]].rename(
            columns={f"{p}_mean": p for p in M.PATHS}
        )
        add("Mean **signed** path effects (positive = document moved up):\n")
        add(_md_table(signed))
        add("")
        for _, r in a.ratio.iterrows():
            add(
                f"- `{r['cfg']}`: interaction residual is "
                f"**{r['nonadditivity_pct_of_total']:.1f}%** of the mean |total effect|."
            )
        add(
            "\nThe residual is reported rather than absorbed. It measures the "
            "non-additivity of the two first-stage channels: they meet in a "
            "union-and-truncate operation, which is not an additive combination, so "
            "the path effects are not expected to sum exactly to the total. In the "
            "`bm25_only` configuration the dense path is absent by construction and "
            "the decomposition is exactly additive, which is why its residual is zero.\n"
        )
        if len(a.ratio) == 2:
            r0, r1 = a.ratio.iloc[0], a.ratio.iloc[1]
            deltas = {p: abs(r0[f"{p}_share_pct"] - r1[f"{p}_share_pct"]) for p in M.PATHS}
            add(
                f"Share shift between `{r0['cfg']}` and `{r1['cfg']}`: "
                + ", ".join(f"{p} {deltas[p]:.1f} pp" for p in M.PATHS)
                + f". **MEDIATION DIFFERENTIAL: "
                f"{'YES' if max(deltas.values()) >= 10 else 'NO'}**\n"
            )
        add(f"![Mediation ratio]({M.FIG_MEDIATION.name})\n")

    # ---- DML ----
    add("## 4. Is naive attribution confounded?\n")
    if a.dml is None or a.dml.empty:
        add(
            "**INCONCLUSIVE** — no concept terms were analysed. The concept "
            "lexicon does not intersect this corpus in the required document-"
            "frequency band, so Module 4 produced no estimates. This is a "
            "coverage gap of the lexicon, not evidence of no confounding.\n"
        )
    else:
        add(
            "Observational analysis of the **baseline** run: units are (query, "
            "candidate document) pairs, treatment D is *document contains concept c*, "
            "outcome Y is the cross-encoder score, and confounders X are document "
            "length, document embedding norm, query–document lexical overlap and "
            "query length. Naive OLS leaves the back-door path D ← doc_len → Y open; "
            f"DoubleML (partially linear, LightGBM nuisances, {config.DML_N_FOLDS}-fold "
            f"cross-fitting x {config.DML_N_REP} repetitions) closes it.\n"
        )
        disp = a.dml.assign(
            naive=[f"{c:+.4f} [{lo:+.4f}, {hi:+.4f}]" for c, lo, hi
                   in zip(a.dml["naive_coef"], a.dml["naive_ci_lo"], a.dml["naive_ci_hi"])],
            dml=[f"{c:+.4f} [{lo:+.4f}, {hi:+.4f}]" for c, lo, hi
                 in zip(a.dml["dml_coef"], a.dml["dml_ci_lo"], a.dml["dml_ci_hi"])],
            df_pct=a.dml["doc_freq"].map("{:.1%}".format),
        )[["concept", "df_pct", "n_treated", "naive", "dml", "abs_shift",
           "naive_outside_dml_ci"]]
        add(_md_table(disp))
        add("")
        det = bool(a.dml["naive_outside_dml_ci"].any())
        add(f"**CONFOUNDING DETECTED: {'YES' if det else 'NO'}**\n")
        add(f"![Naive vs DML]({D.FIG_DML.name})\n")

    # ---- stability ----
    add("## 5. Counterfactual stability\n")
    if a.stability is not None:
        add(
            f"{config.STABILITY_N_QUERIES} queries x 3 meaning-preserving rewrites "
            "(stopword removal, word-order shuffle, synonym/number substitution). "
            "RBO@10 with p=0.9, extrapolated form, so identical rankings score "
            "exactly 1.0. The gap from 1 is the instability.\n"
        )
        piv = a.stability.pivot(index="variant", columns="config", values="mean_rbo")
        piv = piv.reset_index()
        add(_md_table(piv))
        add("")
        chk = a.stability[
            (a.stability["variant"] == "shuffle") & (a.stability["config"] == "bm25_only")
        ]
        if len(chk):
            v = float(chk["mean_rbo"].iloc[0])
            add(
                f"*Implementation check:* BM25 under word-order shuffle scores "
                f"RBO = {v:.6f}. BM25 is a bag of words, so anything other than "
                "exactly 1.0 would indicate a bug in the pipeline or the metric.\n"
            )
        worst = a.stability.loc[a.stability["mean_rbo"].idxmin()]
        add(
            f"Least stable combination: **{worst['variant']}** on **{worst['config']}** "
            f"(mean RBO {worst['mean_rbo']:.4f}, instability {worst['instability']:.4f}).\n"
        )
        add(f"![Stability]({S.FIG_STABILITY.name})\n")

    # ---- limitations ----
    add("## Limitations\n")
    _full_design = config.K_CANDIDATES >= 50 and config.MAX_TARGET_DOCS_PER_QUERY >= 10
    if _full_design:
        add(
            f"- **Scale.** This run used the full design: top-{config.K_CANDIDATES} per "
            f"channel, up to {config.MAX_TARGET_DOCS_PER_QUERY} target documents per "
            f"query, cross-encoder sequence length {config.CE_MAX_LENGTH}, all "
            f"{a.ndcg['n_queries'] if a.ndcg else 323} queries. No compute reductions "
            "were applied.\n"
        )
    else:
        add(
            f"- **Compute budget.** The candidate pool is top-{config.K_CANDIDATES} per "
            f"channel and at most {config.MAX_TARGET_DOCS_PER_QUERY} target documents "
            "per query, reduced from the design's 50/10 to fit a CPU-only machine. All "
            "queries are retained, so the number of bootstrap clusters is unaffected; "
            "the narrower pool does compress the range of measurable rank movement.\n"
        )
    add(
        f"- **Censored outcome.** Rank is truncated at {config.MISSING_RANK}, so large "
        "displacements are recorded as equal. Path effects are correspondingly "
        "conservative.\n"
        "- **DML standard errors** treat (query, document) units as independent, while "
        "documents within a query share a query representation. The naive OLS CIs are "
        "query-clustered; the DML CIs are not, so the confounding test is if anything "
        "conservative in the direction of *not* flagging disagreement.\n"
        "- **Single dataset, single model pair.** NFCorpus is a small, dense-qrel "
        "medical collection; nothing here establishes that the mediation shares "
        "transfer to other corpora or rerankers.\n"
    )

    text = "\n".join(L)
    path.write_text(text)
    print(f"[report] wrote {path} ({len(text)} chars)")
    return text


def main() -> None:
    build_report()
    print()
    print(verdict_text())


if __name__ == "__main__":
    main()
