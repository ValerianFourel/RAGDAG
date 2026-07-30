"""Module 6a - orchestrate the whole experiment, on one GPU or several.

Single process, everything runs against one
:class:`~pipeline.RetrievalPipeline` instance. That is not just tidiness: the
cross-encoder pair cache is keyed on (query text, document) and the mediation
module re-scores heavily overlapping (query, pool) combinations that modules 2
and 5 have already paid for. Sharing one cache removes a large fraction of the
reranking work.

**Multi-GPU.** The GPU-bound work (modules 1, 2, 3, 5) is independent per
query - only the final aggregation is global - so the experiment shards by
query rather than by tensor. Each worker owns one GPU and a strided slice of
the queries, writes partial artefacts, and a merge pass concatenates them and
runs the global steps. This beats DataParallel here: a rerank batch is ~85
pairs of a 22M-parameter model, far too small to split four ways, and the
pipeline stages are inherently serial within a query.

Usage::

    python -m run_all                          # single process, all stages
    N_QUERIES=30 python -m run_all             # smoke test
    python -m run_all --force                  # ignore cached artefacts
    python -m run_all --only 2 3               # selected stages

    # 4-GPU node (see scripts/run_multigpu.sh, which does this for you)
    CUDA_VISIBLE_DEVICES=0 python -m run_all --shard 0 --n-shards 4 &
    CUDA_VISIBLE_DEVICES=1 python -m run_all --shard 1 --n-shards 4 &
    CUDA_VISIBLE_DEVICES=2 python -m run_all --shard 2 --n-shards 4 &
    CUDA_VISIBLE_DEVICES=3 python -m run_all --shard 3 --n-shards 4 &
    wait
    python -m run_all --merge --n-shards 4
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import pandas as pd

import config

STAGES = {
    1: "baseline",
    2: "interventions",
    3: "mediation",
    4: "dml",
    5: "stability",
    6: "report",
}

META_PATH = config.RESULTS_DIR / "run_meta.json"
BASELINE_JSON = config.RESULTS_DIR / "baseline_ndcg.json"
TIMINGS_JSON = config.RESULTS_DIR / "timings.json"


@dataclass
class Clock:
    """Per-module wall-clock accounting, persisted for the report."""

    times: dict[str, float] = field(default_factory=dict)
    tag: str = ""

    def record(self, name: str, seconds: float) -> None:
        self.times[name] = seconds
        print(f"\n[run_all{self.tag}] === {name} finished in {_fmt(seconds)} ===\n", flush=True)

    def save(self, path=TIMINGS_JSON) -> None:
        with open(path, "w") as f:
            json.dump(self.times, f, indent=2)


def _fmt(s: float) -> str:
    if s < 90:
        return f"{s:.1f}s"
    if s < 5400:
        return f"{s / 60:.1f}min"
    return f"{s / 3600:.2f}h"


def current_meta() -> dict:
    """Config fingerprint. A change here invalidates cached artefacts."""
    return {
        "n_queries": config.N_QUERIES,
        "k_candidates": config.K_CANDIDATES,
        "k_final": config.K_FINAL,
        "max_target_docs": config.MAX_TARGET_DOCS_PER_QUERY,
        "n_treatment_terms": config.N_TREATMENT_TERMS,
        "n_control_terms": config.N_CONTROL_TERMS,
        "ce_max_length": config.CE_MAX_LENGTH,
        "dense_max_length": config.MAX_SEQ_LENGTH,
        "precision": config.PRECISION_TAG,
        "dense_model": config.DENSE_MODEL,
        "ce_model": config.CROSS_ENCODER_MODEL,
        "seed": config.SEED,
        # Source fingerprint: a code change must invalidate cached artefacts,
        # otherwise a corrected sampler silently reports the old parquet.
        "code": config.code_fingerprint(),
    }


def meta_matches() -> bool:
    if not META_PATH.exists():
        return False
    with open(META_PATH) as f:
        return json.load(f) == current_meta()


# --------------------------------------------------------------------------- #
# Worker: the per-query, GPU-bound stages
# --------------------------------------------------------------------------- #
def run_worker(shard: int, n_shards: int, stale: bool, only: set[int]) -> None:
    """Compute baseline / interventions / mediation / stability for one query slice.

    Writes ``results/shards/*_{shard}of{n}.parquet``. Deliberately does not run
    the DoubleML stage or the report: both need the *whole* query set, so they
    belong to the merge pass.
    """
    import pipeline as P

    tag = f" shard {shard}/{n_shards}" if n_shards > 1 else ""
    clock = Clock(tag=tag)
    t_start = time.time()

    corpus, queries = P.load_corpus_and_queries()
    pipe = P.RetrievalPipeline(corpus)
    all_qids = P.select_queries(queries)
    qids = config.shard_queries(all_qids, shard, n_shards)
    print(
        f"[run_all{tag}] {config.device_banner()}\n"
        f"[run_all{tag}] corpus={len(corpus)} docs  "
        f"queries={len(qids)}/{len(all_qids)}",
        flush=True,
    )

    # -- 1. baseline (cached per query slice) --
    t0 = time.time()
    baseline = P.compute_baseline(pipe, queries, qids)
    if n_shards == 1:
        ndcg = P.baseline_sanity_check(pipe, queries, baseline)
        with open(BASELINE_JSON, "w") as f:
            json.dump(ndcg, f, indent=2)
        if ndcg["full_pipeline"] < 0.15:
            raise SystemExit(
                "Full-pipeline nDCG@10 < 0.15 - the pipeline is broken. Stopping "
                "before the causal modules, per the working agreement."
            )
    clock.record("baseline", time.time() - t0)

    # -- 2. interventions --
    import interventions as I

    p_inter = config.shard_path("interventions", shard, n_shards)
    if 2 in only:
        if p_inter.exists() and not stale:
            print(f"[run_all{tag}] reusing {p_inter.name}")
        else:
            t0 = time.time()
            df = I.run_interventions(pipe, queries, baseline, qids)
            df.to_parquet(p_inter, index=False)
            print(f"[run_all{tag}] wrote {p_inter.name} ({len(df)} rows)")
            clock.record("interventions", time.time() - t0)

    # -- 3. mediation --
    import mediation as M

    p_med = config.shard_path("mediation", shard, n_shards)
    if 3 in only:
        if p_med.exists() and not stale:
            print(f"[run_all{tag}] reusing {p_med.name}")
        else:
            t0 = time.time()
            inter = pd.read_parquet(p_inter)
            treated = inter[inter["arm"] == "treatment"].reset_index(drop=True)
            med = M.run_mediation(pipe, treated, baseline)
            med.to_parquet(p_med, index=False)
            print(f"[run_all{tag}] wrote {p_med.name} ({len(med)} rows)")
            clock.record("mediation", time.time() - t0)

    # -- 5. stability --
    import stability as S

    p_stab = config.shard_path("stability", shard, n_shards)
    if 5 in only:
        if p_stab.exists() and not stale:
            print(f"[run_all{tag}] reusing {p_stab.name}")
        else:
            t0 = time.time()
            S._self_test()
            # Sample from the full query set first, then shard the sample, so
            # the 50 stability queries are the same regardless of worker count.
            sampled = S.sample_queries(all_qids)
            mine = [q for q in sampled if q in set(qids)]
            sdf = (
                S.run_stability(pipe, queries, mine)
                if mine
                else pd.DataFrame(columns=["query_id", "variant", "config", "rbo"])
            )
            sdf.to_parquet(p_stab, index=False)
            print(f"[run_all{tag}] wrote {p_stab.name} ({len(sdf)} rows)")
            clock.record("stability", time.time() - t0)

    clock.times["worker_total"] = time.time() - t_start
    clock.save(
        config.SHARD_DIR / f"timings_{shard}of{n_shards}.json"
        if n_shards > 1
        else TIMINGS_JSON
    )
    print(f"[run_all{tag}] worker done in {_fmt(clock.times['worker_total'])}")


# --------------------------------------------------------------------------- #
# Merge: concatenate shards and run the global stages
# --------------------------------------------------------------------------- #
def _load_shards(name: str, n_shards: int) -> pd.DataFrame:
    """Concatenate every worker's artefact, failing loudly on a missing shard.

    A silently-skipped shard would look like a completed run over fewer
    queries, which is exactly the kind of quiet truncation that invalidates an
    aggregate without announcing itself.
    """
    frames, missing = [], []
    for i in range(n_shards):
        p = config.shard_path(name, i, n_shards)
        if not p.exists():
            missing.append(p.name)
            continue
        frames.append(pd.read_parquet(p))
    if missing:
        raise SystemExit(
            f"missing {len(missing)}/{n_shards} {name} shards: {', '.join(missing)}\n"
            f"Re-run the failed worker(s) before merging - refusing to aggregate "
            f"a partial result set."
        )
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "query_id" in df.columns:
        df = df.sort_values(["query_id"] + [c for c in ("doc_id", "term", "cfg", "variant", "config") if c in df.columns])
        df = df.reset_index(drop=True)
    return df


def run_merge(n_shards: int, stale: bool, only: set[int]) -> None:
    """Aggregate shard artefacts, run the global stages, write the report."""
    import pipeline as P

    print(f"[merge] combining {n_shards} shard(s)")
    clock = Clock()
    t_start = time.time()
    if TIMINGS_JSON.exists() and not stale:
        with open(TIMINGS_JSON) as f:
            clock.times = json.load(f)
    for i in range(n_shards):
        p = config.SHARD_DIR / f"timings_{i}of{n_shards}.json"
        if p.exists():
            with open(p) as f:
                for k, v in json.load(f).items():
                    clock.times[f"{k}[shard{i}]"] = v

    corpus, queries = P.load_corpus_and_queries()
    pipe = P.RetrievalPipeline(corpus)
    all_qids = P.select_queries(queries)

    # -- baseline: reassemble from the per-shard caches, no GPU work --
    baseline: dict[str, P.BaselineRun] = {}
    for i in range(n_shards):
        baseline.update(P.compute_baseline(pipe, queries, config.shard_queries(all_qids, i, n_shards)))
    ndcg = P.baseline_sanity_check(pipe, queries, baseline)
    with open(BASELINE_JSON, "w") as f:
        json.dump(ndcg, f, indent=2)
    if ndcg["full_pipeline"] < 0.15:
        raise SystemExit("Full-pipeline nDCG@10 < 0.15 - the pipeline is broken.")

    import interventions as I
    import mediation as M
    import stability as S

    if 2 in only:
        inter = _load_shards("interventions", n_shards)
        inter.to_parquet(I.OUT_PARQUET, index=False)
        print(f"[merge] wrote {I.OUT_PARQUET.name} ({len(inter)} rows)")
        I.print_summary(inter)
        I.plot_effects(inter)

    if 3 in only:
        med = _load_shards("mediation", n_shards)
        med.to_parquet(M.OUT_PARQUET, index=False)
        print(f"[merge] wrote {M.OUT_PARQUET.name} ({len(med)} rows)")
        ratio = M.mediation_ratio(med)
        ratio.to_csv(M.OUT_TABLE, index=False)
        M.print_ratio_table(ratio)
        M.plot_mediation(ratio)

    # -- 4. DoubleML: needs the whole panel, and is CPU/LightGBM bound --
    import dml_analysis as D

    if 4 in only:
        t0 = time.time()
        if D.OUT_CSV.exists() and not stale:
            res = pd.read_csv(D.OUT_CSV)
            print(f"[merge] reusing {D.OUT_CSV.name}")
        else:
            concepts = D.select_concepts(corpus)
            panel = D.build_panel(pipe, queries, baseline, [c for c, _ in concepts])
            panel.to_parquet(D.OUT_PANEL, index=False)
            res = D.analyse(panel, concepts)
            res.to_csv(D.OUT_CSV, index=False)
            clock.record("dml", time.time() - t0)
        D.print_comparison(res)
        D.plot_dml(res)

    if 5 in only:
        sdf = _load_shards("stability", n_shards)
        sdf.to_parquet(S.OUT_PARQUET, index=False)
        summ = S.summarize(sdf)
        summ.to_csv(S.OUT_CSV, index=False)
        S.print_summary(summ, sdf)
        S.plot_stability(summ, sdf)

    with open(META_PATH, "w") as f:
        json.dump(current_meta(), f, indent=2)
    clock.times["merge"] = time.time() - t_start
    clock.save()

    if 6 in only:
        import report as R

        R.build_report()
        print("\n[merge] verdict section:\n")
        print(R.verdict_text())

    print("\n[merge] per-stage timings:")
    for k, v in clock.times.items():
        print(f"    {k:24s} {_fmt(v)}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Run the causal retrieval MVP end to end.")
    ap.add_argument("--force", action="store_true", help="ignore cached artefacts")
    ap.add_argument(
        "--only", nargs="+", type=int, choices=sorted(STAGES), default=sorted(STAGES),
        help="run only these stage numbers",
    )
    ap.add_argument("--shard", type=int, default=None, help="worker index (multi-GPU)")
    ap.add_argument("--n-shards", type=int, default=1, help="number of workers")
    ap.add_argument("--merge", action="store_true", help="merge shards and finish")
    args = ap.parse_args()
    only = set(args.only)

    config.set_seeds()
    stale = args.force or not meta_matches()
    if stale and META_PATH.exists():
        print("[run_all] config changed since last run - recomputing artefacts")

    if args.shard is None and not args.merge:
        print("=" * 70)
        print("CAUSAL RETRIEVAL MVP")
        print(f"  {config.device_banner()}")
        for k, v in config.summary().items():
            print(f"  {k:28s} {v}")
        print("=" * 70 + "\n")

    if args.merge:
        run_merge(args.n_shards, stale, only)
    elif args.shard is not None:
        run_worker(args.shard, args.n_shards, stale, only)
    else:
        # Single process: worker over the whole query set, then merge.
        run_worker(0, 1, stale, only)
        run_merge(1, stale, only)


if __name__ == "__main__":
    main()
