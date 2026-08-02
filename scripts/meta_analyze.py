"""Aggregate clean K=50 admission-v2 estimates without pooling raw trials."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def random_effects(g: pd.DataFrame) -> dict[str, float]:
    y, se = g["estimate"].to_numpy(float), g["se"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, se = y[ok], se[ok]
    if len(y) < 2:
        return {"estimate": np.nan, "se": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "tau2": np.nan, "i2": np.nan}
    w = 1 / se**2
    fixed = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - fixed)**2)
    df = len(y) - 1
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1 / (se**2 + tau2)
    point = np.sum(wr * y) / np.sum(wr)
    pooled_se = math.sqrt(1 / np.sum(wr))
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return {"estimate": point, "se": pooled_se,
            "ci_lo": point - 1.96 * pooled_se,
            "ci_hi": point + 1.96 * pooled_se,
            "tau2": tau2, "i2": i2}


def main() -> int:
    rows = []
    for d in sorted(config.RESULTS_ROOT.glob("beir-*_admission-v2_*_k50")):
        mp, ep = d / "run_meta.json", d / "admission_estimates.csv"
        if not mp.is_file() or not ep.is_file():
            continue
        meta = json.loads(mp.read_text())
        if meta.get("experiment_version") != config.EXPERIMENT_VERSION:
            continue
        estimates = pd.read_csv(ep)
        estimates = estimates[estimates["target_stratum"] == "all"].copy()
        estimates["dataset"] = meta["dataset"]
        estimates["dense_model"] = meta["dense_model"]
        estimates["code"] = meta["code"]
        rows.append(estimates)
    if not rows:
        print("no K=50 admission-v2 estimates found", file=sys.stderr)
        return 1
    data = pd.concat(rows, ignore_index=True)
    data["se"] = (data["ci_hi"] - data["ci_lo"]) / 3.92
    data["z"] = data["estimate"] / data["se"]
    data["p_value"] = [math.erfc(abs(z) / math.sqrt(2)) for z in data["z"]]
    data["p_holm"] = np.nan
    for _, idx in data.groupby(["dense_model", "metric"]).groups.items():
        loc = list(idx)
        data.loc[loc, "p_holm"] = _holm(data.loc[loc, "p_value"].to_numpy())
    data.to_csv(config.RESULTS_ROOT / "hypothesis_estimates.csv", index=False)

    recs = []
    for (model, metric), g in data.groupby(["dense_model", "metric"]):
        recs.append({"dense_model": model, "metric": metric,
                     "n_collections": g["dataset"].nunique(), **random_effects(g)})
    out = pd.DataFrame(recs)
    out.to_csv(config.RESULTS_ROOT / "meta_analysis.csv", index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
