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
* regression: ``admission.admitted`` (the lexical >=K-th-value rule) against
  its own brute-force reference, so a future edit cannot silently change the
  published BM25 semantics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dense_admission import beat_count, dense_worlds, union_gate  # noqa: E402
import admission  # noqa: E402


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
        # lexical rule: admitted iff score >= K-th largest VALUE (ties admit)
        want = s[di] >= np.sort(s)[::-1][k - 1]
        assert admission.admitted(s, di, k) == want
    print("ok  admission.admitted matches the >=K-th-value rule (300 trials)")


if __name__ == "__main__":
    test_dense_worlds_numpy()
    test_dense_worlds_torch_cpu()
    test_stale_slot_exclusion()
    test_cauchy_schwarz()
    test_union_gate()
    test_lexical_admitted_regression()
    print("\nALL WORLD TESTS PASSED")
