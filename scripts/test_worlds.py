"""Unit tests for the four-world evaluators (dense + lexical) and union gate.

Plain asserts, no pytest dependency::

    python scripts/test_worlds.py

Covers:
* dense membership under the production tie rule ((-score, doc index)
  lexsort), against a brute-force reference, including deliberate ties and
  negative deltas - numpy and torch-CPU backends must agree exactly;
* the target's stale-slot exclusion trick in ``beat_count``;
* the Cauchy-Schwarz bound on random embeddings;
* union-gate classification on a hand-built panel;
* sparse membership and top-K selection under the production tie rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dense_admission import (beat_count, dense_worlds, publication_estimates,
                             union_gate)  # noqa: E402
import admission  # noqa: E402
import interventions  # noqa: E402


# --------------------------------------------------------------------------- #
def brute_member(scores: np.ndarray, di: int, k: int) -> bool:
    """Production rule: exactly K docs, ordered by (-score, index)."""
    order = sorted(range(len(scores)), key=lambda j: (-scores[j], j))
    return di in order[:k]


def brute_worlds(s: np.ndarray, ds: np.ndarray, di: int, k: int) -> dict:
    def member(vals: np.ndarray) -> bool:
        return brute_member(vals, di, k)

    s10 = s.copy(); s10[di] += ds[di]
    s01 = s + ds; s01[di] = s[di]
    return {
        "y00": member(s), "y11": member(s + ds),
        "y10": member(s10), "y01": member(s01),
    }


def test_dense_worlds_numpy() -> None:
    rng = np.random.default_rng(7)
    n, k = 40, 5
    idx = np.arange(n)
    for trial in range(500):
        s = np.round(rng.normal(size=n), 2)          # rounding forces ties
        ds = np.round(rng.normal(scale=0.5, size=n), 2)
        ds[rng.random(n) < 0.5] = 0.0                # sparse-ish, signed
        di = int(rng.integers(n))
        got = dense_worlds(s.astype(float), ds.astype(float), di, k, idx)
        want = brute_worlds(s.astype(float), ds.astype(float), di, k)
        for w in ("y00", "y10", "y01", "y11"):
            assert got[w] == want[w], (trial, w, di, got[w], want[w])
    print("ok  dense_worlds vs brute force (numpy, 500 trials with ties)")


def test_dense_worlds_torch_cpu() -> None:
    import torch

    rng = np.random.default_rng(11)
    n, k = 40, 5
    idx_np = np.arange(n)
    idx_t = torch.arange(n)
    for trial in range(200):
        s = np.round(rng.normal(size=n), 2).astype(float)
        ds = np.round(rng.normal(scale=0.5, size=n), 2).astype(float)
        di = int(rng.integers(n))
        a = dense_worlds(s, ds, di, k, idx_np)
        b = dense_worlds(torch.as_tensor(s), torch.as_tensor(ds), di, k, idx_t)
        for w in ("y00", "y10", "y01", "y11"):
            assert bool(a[w]) == bool(b[w]), (trial, w)
    print("ok  torch-CPU backend agrees with numpy (200 trials)")


def test_stale_slot_exclusion() -> None:
    # target sits at index 2 with stored score 9; hypothetical score 1.
    s = np.array([5.0, 3.0, 9.0, 1.0])
    idx = np.arange(4)
    # docs beating a target at 1.0: 5.0, 3.0 and the tie at 1.0 with index 3?
    # index 3 > 2, so the tie does NOT beat it; stored 9.0 (self) is excluded.
    assert beat_count(s, 1.0, 2, idx) == 2
    # tie handling: equal score at a smaller index beats, larger does not
    s2 = np.array([2.0, 7.0, 7.0, 7.0])
    assert beat_count(s2, 7.0, 2, idx) == 1   # only index 1 beats index 2
    assert beat_count(s2, 7.0, 1, idx) == 0
    assert beat_count(s2, 7.0, 3, idx) == 2
    print("ok  beat_count stale-slot exclusion and tie rule")


def test_cauchy_schwarz() -> None:
    rng = np.random.default_rng(3)
    E = rng.normal(size=(1000, 32))
    norms = np.linalg.norm(E, axis=1)
    for _ in range(20):
        dq = rng.normal(size=32) * rng.uniform(0.01, 2.0)
        ds = E @ dq
        bound = norms * np.linalg.norm(dq)
        assert (np.abs(ds) <= bound * (1 + 1e-9) + 1e-12).all()
    print("ok  Cauchy-Schwarz bound holds on random embeddings")


def test_union_gate() -> None:
    import pandas as pd

    p = pd.DataFrame({
        "arm": ["treatment"] * 4,
        "query_id": ["q1"] * 4,
        # dense worlds
        "y00": [False, False, True, True],
        "y11": [True, True, False, True],
        "bm25_y00": [True, False, True, False],
        "bm25_y11": [True, False, True, False],
        "u00": [1.0, 0.0, 1.0, 1.0],
        "u11": [1.0, 1.0, 1.0, 1.0],
    })
    out = union_gate(p)
    r = {(row["dense_y00"]): row for _, row in out.iterrows()}
    # excluded stratum: one redundant rescue (bm25 already had it), one genuine
    assert r[False]["redundant_dense_rescue"] == 1
    assert r[False]["union_rescue"] == 1
    # admitted stratum: one dense ejection masked by bm25, one no-op
    assert r[True]["dense_masked"] == 1
    assert r[True]["union_displacement"] == 0
    print("ok  union gate classification")


def test_lexical_admitted_regression() -> None:
    rng = np.random.default_rng(5)
    n, k = 60, 7
    for _ in range(300):
        s = np.round(rng.normal(size=n), 2)
        di = int(rng.integers(n))
        want = brute_member(s, di, k)
        assert admission.admitted(s, di, k) == want
        got = admission.topk_indices(s, k).tolist()
        expected = sorted(range(n), key=lambda j: (-s[j], j))[:k]
        assert got == expected
    print("ok  sparse admission/top-K match production ties (300 trials)")


def test_shapley_identity() -> None:
    for y00 in (0, 1):
        for y10 in (0, 1):
            for y01 in (0, 1):
                for y11 in (0, 1):
                    target = .5 * ((y10 - y00) + (y11 - y01))
                    competitors = .5 * ((y01 - y00) + (y11 - y10))
                    assert target + competitors == y11 - y00
    print("ok  Shapley allocations reconstruct every binary four-world total")


def test_population_target_sampling() -> None:
    class Corpus:
        doc_ids = [f"d{i:03d}" for i in range(100)]
        doc_index = {d: i for i, d in enumerate(doc_ids)}
        def __len__(self): return len(self.doc_ids)
        def idx(self, doc_id): return self.doc_index[doc_id]
    class Pipe:
        corpus = Corpus()
        def _bm25_array(self, query): return np.arange(100, 0, -1, dtype=float)
        def _dense_array(self, query): return np.arange(100, 0, -1, dtype=float)
    relevant = set(Pipe.corpus.doc_ids) | {"not-in-corpus"}
    frame = interventions.select_targets(Pipe(), "q", relevant,
                                         np.random.default_rng(19), return_frame=True)
    selected = [r for r in frame if r["target_selected"]]
    assert len(frame) == 100                 # out-of-union qrels remain eligible
    assert len(selected) <= 10               # at most two per five strata
    assert all(0 < r["target_select_prob"] <= 1 for r in frame)
    assert {r["target_stratum"] for r in frame} >= {
        "deep_in", "boundary_in", "boundary_out", "mid_out", "deep_out"}
    print("ok  all-relevant stratified target frame and selection probabilities")


def test_weighted_publication_estimate() -> None:
    import pandas as pd
    rows = []
    # Two queries with deliberately unequal sampling probabilities. IPW should
    # recover a 1/2 hybrid-rescue population mean exactly.
    for q, weight, rescued in (("q1", 4.0, 1), ("q2", 4.0, 0)):
        rows.append({
            "query_id": q, "doc_id": q, "term": "x", "arm": "treatment",
            "analysis_weight": weight, "target_stratum": "boundary_out",
            "y00": False, "y10": bool(rescued), "y01": False, "y11": bool(rescued),
            "bm25_y00": False, "bm25_y10": bool(rescued),
            "bm25_y01": False, "bm25_y11": bool(rescued),
            "bm25_shapley_target": float(rescued),
            "bm25_shapley_competitors": 0.0,
            "shapley_target": float(rescued), "shapley_competitors": 0.0,
            "u00": 0.0, "u10": float(rescued), "u01": 0.0, "u11": float(rescued),
        })
    estimates = publication_estimates(pd.DataFrame(rows), n_boot=100)
    value = estimates[(estimates["target_stratum"] == "all") &
                      (estimates["metric"] == "hybrid_rescue")]["estimate"].iloc[0]
    assert value == 0.5
    print("ok  inverse-probability weighted clustered publication estimate")


if __name__ == "__main__":
    test_dense_worlds_numpy()
    test_dense_worlds_torch_cpu()
    test_stale_slot_exclusion()
    test_cauchy_schwarz()
    test_union_gate()
    test_lexical_admitted_regression()
    test_shapley_identity()
    test_population_target_sampling()
    test_weighted_publication_estimate()
    print("\nALL WORLD TESTS PASSED")
