"""Post-run verification for the admission stages, across K arms.

    python scripts/check_admission.py

Checks, per results directory:

* the lexical and dense panels were built at the **same K** (they are joined,
  and K is the bar itself - a mismatch manufactures documents that appear to be
  outside both channels' top-K, which is impossible since targets are drawn
  from the union pool);
* both closed-form audits pass;
* ``both_missed`` is zero (the observable symptom of the above);
* the lexical immunity certificate has zero violations - margin greater than
  the largest possible competitor boost must imply the target is never ejected.

Exits non-zero if any check fails, so it can gate a downstream step.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd


def _load(path: str):
    try:
        return json.load(open(path))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    rows, problems = [], []
    for d in sorted(glob.glob("results/beir-*")):
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
        if isinstance(r["both_missed"], int) and r["both_missed"]:
            problems.append(f"{name}: {r['both_missed']} rows outside BOTH channels "
                            "(impossible - targets come from the union pool)")
        if r["immune_viol"]:
            problems.append(f"{name}: {r['immune_viol']} immunity-certificate violations")
        rows.append(r)

    if not rows:
        print("no admission panels found under results/")
        return 1
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
