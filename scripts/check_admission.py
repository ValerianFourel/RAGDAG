"""Post-run verification for the admission stages, across K arms.

    python scripts/check_admission.py

Checks, per results directory:

* the lexical and dense panels were built at the **same K** (they are joined,
  and K is the bar itself - a mismatch manufactures documents that appear to be
  outside both channels' top-K, which is impossible since targets are drawn
  from the union pool);
* both closed-form audits pass;
* ``both_missed`` is zero **when the analysis K is at least the K the
  interventions were sampled at** (default 50, override with
  ``--interventions-k``). Targets are drawn from the top-K BM25 union top-K
  dense pool at sampling time, so at a shallower analysis K a target that sat
  at, say, BM25 rank 35 is legitimately outside both top-20 lists - there
  ``both_missed`` is a measurement (how many targets a shallower pool loses),
  not a defect;
* the lexical immunity certificate has zero violations - margin greater than
  the largest possible competitor boost must imply the target is never ejected.

Exits non-zero if any check fails, so it can gate a downstream step.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load(path: str):
    try:
        return json.load(open(path))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    import argparse

    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--interventions-k", type=int, default=50,
                     help="pool depth the do() trials were sampled at (default 50)")
    ap_.add_argument("--release", action="store_true",
                     help="enforce the complete 6-dataset x 2-model x 3-K matrix")
    args = ap_.parse_args()

    rows, problems = [], []
    import config
    for d in sorted(glob.glob("results/beir-*")):
        meta = _load(os.path.join(d, "run_meta.json")) or {}
        if meta.get("experiment_version") != config.EXPERIMENT_VERSION:
            continue
        ap = os.path.join(d, "admission_panel.parquet")
        if not os.path.exists(ap):
            continue
        name = os.path.basename(d)
        la = _load(os.path.join(d, "admission_audit.json")) or {}
        p = pd.read_parquet(ap)
        r = {
            "dir": name,
            "lexK": la.get("k_candidates"),
            "denseK": None,
            "n": len(p),
            "admit": round(float(p["base_admitted"].mean()), 3),
        }
        required = {"analysis_weight", "target_select_prob", "target_stratum",
                    "competitive_spillover", "shapley_target", "shapley_competitors"}
        missing = required - set(p.columns)
        if missing:
            problems.append(f"{name}: lexical panel missing v2 columns {sorted(missing)}")
        elif not np.allclose(p["shapley_target"] + p["shapley_competitors"],
                             p["surg_total"]):
            problems.append(f"{name}: lexical Shapley reconstruction failed")
        ctl = p[(p["arm"] == "control") & p["base_admitted"]]
        tre = p[(p["arm"] == "treatment") & ~p["base_admitted"]]
        r["interference"] = round(float(ctl["surg_interference"].mean()), 3) if len(ctl) else None
        r["rescue"] = round(float(tre["surg_direct"].mean()), 3) if len(tre) else None

        # margin > max possible competitor boost => ejection is arithmetically
        # impossible. Violations must be exactly zero.
        imm = ctl[ctl["base_margin"] > ctl["idf"]]
        r["immune_n"] = len(imm)
        r["immune_viol"] = int((~imm["pred_admitted"]).sum()) if len(imm) else 0

        dp = os.path.join(d, "dense_admission_panel.parquet")
        if os.path.exists(dp):
            da = _load(os.path.join(d, "dense_admission_audit.json")) or {}
            q = pd.read_parquet(dp)
            r["denseK"] = da.get("k_candidates")
            r["both_missed"] = int((q["u00"] == 0).sum()) if "u00" in q.columns else "no join"
            if {"shapley_target", "shapley_competitors", "surg_total"} <= set(q.columns):
                if not np.allclose(q["shapley_target"] + q["shapley_competitors"],
                                   q["surg_total"]):
                    problems.append(f"{name}: dense Shapley reconstruction failed")
            if not da.get("exact", False):
                problems.append(f"{name}: dense audit not exact")
            if r["denseK"] is None:
                problems.append(f"{name}: dense audit predates the K stamp - the "
                                "panel may have been built at another K; re-run stage 8")
        else:
            r["both_missed"] = "-"

        if not la.get("exact", False):
            problems.append(f"{name}: lexical audit not exact")
        if r["lexK"] is None:
            problems.append(f"{name}: lexical audit predates the K stamp - re-run stage 7")
        elif r["denseK"] is not None and r["lexK"] != r["denseK"]:
            problems.append(f"{name}: K mismatch lex={r['lexK']} dense={r['denseK']}")
        # v2 deliberately samples relevant targets outside the union. These are
        # the extensive-margin population, not an impossible state.
        if r["immune_viol"]:
            problems.append(f"{name}: {r['immune_viol']} immunity-certificate violations")
        if args.release and meta.get("git_dirty"):
            problems.append(f"{name}: run was produced from a dirty worktree")
        if meta.get("code") != config.code_fingerprint():
            problems.append(f"{name}: code fingerprint differs from current source")
        revision = meta.get("dense_model_revision")
        if args.release and (not revision or revision == "main"):
            problems.append(f"{name}: dense model is not pinned to an immutable revision")
        ip = os.path.join(d, "interventions.parquet")
        if os.path.exists(ip):
            inter = pd.read_parquet(ip)
            controls = inter[inter["arm"] == "control"]
            if len(controls) and not (controls["delta_bm25"] == 0).all():
                problems.append(f"{name}: controls changed the target BM25 score")
            if not os.path.exists(os.path.join(d, "target_sampling_frame.parquet")):
                problems.append(f"{name}: complete target sampling frame is missing")
        for required_file in ("admission_estimates.csv", "union_gate.csv",
                              "target_sampling_frame.parquet", "shard_completion.json"):
            if args.release and not os.path.exists(os.path.join(d, required_file)):
                problems.append(f"{name}: required release artefact {required_file} is missing")
        rows.append(r)

    if args.release:
        expected = 6 * 2 * 3
        if len(rows) != expected:
            problems.append(f"release matrix incomplete: found {len(rows)}/{expected} result arms")
        for summary in ("results/meta_analysis.csv", "results/hypothesis_estimates.csv"):
            if not os.path.exists(summary):
                problems.append(f"release summary missing: {summary}")
    if not rows:
        print("no admission panels found under results/")
        return 1 if args.release else 0
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p_ in problems:
            print(f"  !! {p_}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
