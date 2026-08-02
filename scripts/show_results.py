"""Print the admission results as readable tables.

    python scripts/show_results.py                 # everything, K=50 arm
    python scripts/show_results.py --arm _k20      # a sensitivity arm
    python scripts/show_results.py --sweep         # K=20/50/100 side by side
    python scripts/show_results.py --only rescue   # one section

Sections: counts, interference, rescue, dense, union, models.
Reads only artefacts; computes nothing that the panels do not already contain.
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

SECTIONS = ("counts", "interference", "rescue", "dense", "union", "models")


def dirs_for(arm: str) -> list[str]:
    """Result directories for one arm ('' is the primary K=50 arm)."""
    out = []
    for d in sorted(glob.glob("results/beir-*")):
        base = os.path.basename(d)
        suffix = ""
        for s in ("_k20", "_k50", "_k100"):
            if base.endswith(s):
                suffix = s
        if suffix == arm:
            out.append(d)
    return out


def _read(d: str, name: str):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


def section_counts(ds: list[str]) -> None:
    rows = []
    for d in ds:
        p = _read(d, "admission_panel.parquet")
        if p is None:
            continue
        rows.append({
            "collection": os.path.basename(d),
            "trials": len(p),
            "queries": p["query_id"].nunique(),
            "pairs": p.groupby(["query_id", "doc_id"]).ngroups,
            "unique_terms": p["term"].nunique(),
            "treatment": int((p["arm"] == "treatment").sum()),
            "control": int((p["arm"] == "control").sum()),
            "admitted": int(p["base_admitted"].sum()),
            "excluded": int((~p["base_admitted"]).sum()),
        })
    if rows:
        print("\n=== TRIAL COUNTS (lexical panel) ===")
        print(pd.DataFrame(rows).to_string(index=False))


def section_interference(ds: list[str]) -> None:
    print("\n=== LEXICAL INTERFERENCE: one irrelevant word vs an admitted target ===")
    print("(control arm, surg_interference by support bin; negative = ejected)")
    for d in ds:
        t = _read(d, "admission_surgical.csv")
        if t is None:
            continue
        sub = t[(t["arm"] == "control") & (t["base_admitted"])]
        if sub.empty:
            continue
        print(f"\n-- {os.path.basename(d)}")
        cols = ["median_support", "n", "interference_y01_y00", "total_y11_y00",
                "mean_threat_count", "mean_cotreated"]
        print(sub[[c for c in cols if c in sub.columns]].round(4).to_string(index=False))


def section_rescue(ds: list[str]) -> None:
    print("\n=== LEXICAL RESCUE: own-lift admission of an excluded target ===")
    print("(treatment arm, base_admitted=False; direct = Y10-Y00, total = Y11-Y00)")
    for d in ds:
        t = _read(d, "admission_surgical.csv")
        if t is None:
            continue
        sub = t[(t["arm"] == "treatment") & (~t["base_admitted"])]
        if sub.empty:
            continue
        print(f"\n-- {os.path.basename(d)}")
        cols = ["median_support", "n", "direct_y10_y00", "total_y11_y00",
                "interaction", "mean_threat_count"]
        print(sub[[c for c in cols if c in sub.columns]].round(4).to_string(index=False))


def section_dense(ds: list[str]) -> None:
    print("\n=== DENSE FOUR WORLDS (cluster-bootstrap 95% CIs) ===")
    for d in ds:
        t = _read(d, "dense_admission_by_bin.csv")
        if t is None:
            continue
        print(f"\n-- {os.path.basename(d)}")
        show = [c for c in t.columns if not c.endswith(("_lo", "_hi"))]
        print(t[show].round(4).to_string(index=False))


def section_union(ds: list[str]) -> None:
    print("\n=== UNION GATE (BM25 OR dense) ===")
    for d in ds:
        t = _read(d, "union_gate.csv")
        if t is None:
            continue
        print(f"\n-- {os.path.basename(d)}")
        print(t.to_string(index=False))


def section_models(ds: list[str]) -> None:
    print("\n=== MECHANISTIC PREDICTION (leave-queries-out CV, out-of-fold) ===")
    print("predictive, not causal; higher ROC-AUC / lower log-loss is better")
    for d in ds:
        t = _read(d, "dense_admission_models.csv")
        if t is None:
            continue
        print(f"\n-- {os.path.basename(d)}")
        cols = ["task", "model", "n", "base_rate", "log_loss", "roc_auc", "pr_auc"]
        print(t[[c for c in cols if c in t.columns]].round(4).to_string(index=False))


def sweep() -> None:
    """K=20/50/100 side by side, from the lexical panels."""
    rows = []
    for arm, k in (("_k20", 20), ("", 50), ("_k100", 100)):
        for d in dirs_for(arm):
            p = _read(d, "admission_panel.parquet")
            if p is None:
                continue
            name = os.path.basename(d)
            for s in ("_k20", "_k100"):
                name = name.replace(s, "")
            ctl = p[(p["arm"] == "control") & p["base_admitted"]]
            tre = p[(p["arm"] == "treatment") & ~p["base_admitted"]]
            imm = ctl[ctl["base_margin"] > ctl["idf"]]
            rows.append({
                "collection": name, "K": k,
                "admit_rate": round(float(p["base_admitted"].mean()), 3),
                "interference": round(float(ctl["surg_interference"].mean()), 4) if len(ctl) else None,
                "rescue": round(float(tre["surg_direct"].mean()), 4) if len(tre) else None,
                "n_admitted": len(ctl), "n_excluded": len(tre),
                "immune_n": len(imm),
                "immune_viol": int((~imm["pred_admitted"]).sum()) if len(imm) else 0,
            })
    if not rows:
        print("no panels found")
        return
    df = pd.DataFrame(rows).sort_values(["collection", "K"])
    print("\n=== K SENSITIVITY (same do() trials, different pool depth) ===")
    print(df.to_string(index=False))
    print("\nInterference and rescue by K (pivot):")
    for metric in ("interference", "rescue", "admit_rate"):
        piv = df.pivot(index="collection", columns="K", values=metric)
        print(f"\n  {metric}")
        print(piv.to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="", help="'' (K=50), _k20 or _k100")
    ap.add_argument("--sweep", action="store_true", help="K arms side by side")
    ap.add_argument("--only", choices=SECTIONS, help="print one section")
    args = ap.parse_args()

    if args.sweep:
        sweep()
        return 0

    ds = dirs_for(args.arm)
    if not ds:
        print(f"no result directories for arm '{args.arm}'")
        return 1
    fns = {"counts": section_counts, "interference": section_interference,
           "rescue": section_rescue, "dense": section_dense,
           "union": section_union, "models": section_models}
    for name in ([args.only] if args.only else SECTIONS):
        fns[name](ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
