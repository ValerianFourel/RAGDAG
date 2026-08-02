"""Download everything the experiment needs, so the compute job can run offline.

Run this on a **login node** (which has internet); the batch job then runs with
``HF_HUB_OFFLINE=1`` and touches the network zero times. On clusters whose
compute nodes are air-gapped -- HoreKa among them -- forgetting this step is the
single most common way a job dies four seconds after it starts.

What it fetches / builds:

* the configured dense encoder checkpoint, into ``$HF_HOME``
* the BEIR NFCorpus test collection, into ``$IR_DATASETS_HOME``
* ``cache/corpus.pkl``   -- parsed documents, tokens, qrels (profile-independent)
* ``cache/bm25_index/``  -- the BM25 index (profile-independent)

It deliberately does **not** build the document embeddings. Those are keyed on
the dense sequence length, which differs between the CPU and GPU profiles, and
they take seconds on an A100 versus minutes on a shared login node. The job
builds them.

Usage::

    python scripts/prefetch.py            # download + light caches
    python scripts/prefetch.py --embed    # also embed, using the current profile
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--embed", action="store_true",
        help="also build document embeddings under the current profile "
             "(slow on a login node; normally left to the batch job)",
    )
    args = ap.parse_args()

    print("=" * 68)
    print("PREFETCH - run me on a login node, with internet")
    print("=" * 68)
    for k in ("HF_HOME", "IR_DATASETS_HOME"):
        print(f"  {k:18s} {os.environ.get(k, '(unset - will land in $HOME)')}")
    print(f"  cache dir          {config.CACHE_DIR}")
    print(f"  profile            {config.device_banner()}")
    if os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"):
        print("\n  !! HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE is set - unset it to download.")
        return 1
    print()

    config.set_seeds()
    t0 = time.time()

    # -- models -----------------------------------------------------------
    from sentence_transformers import SentenceTransformer

    print(f"[1/4] dense encoder      {config.DENSE_MODEL}", flush=True)
    t = time.time()
    m = SentenceTransformer(config.DENSE_MODEL, device="cpu",
                            revision=config.DENSE_MODEL_REVISION)
    print(f"      ok ({time.time() - t:.1f}s, dim={m.get_sentence_embedding_dimension()})")

    print("[2/4] cross-encoder      skipped (admission-v2 is retrieval-only)")

    # -- dataset + derived caches -----------------------------------------
    print(f"[3/4] dataset            {config.DATASET}", flush=True)
    t = time.time()
    from pipeline import RetrievalPipeline, load_corpus_and_queries

    corpus, queries = load_corpus_and_queries()
    print(f"      ok ({time.time() - t:.1f}s) "
          f"{len(corpus)} docs, {len(queries.query_ids)} queries")

    print("[4/4] BM25 index", flush=True)
    t = time.time()
    pipe = RetrievalPipeline(corpus, verbose=False)
    pipe.bm25
    print(f"      ok ({time.time() - t:.1f}s)")

    if args.embed:
        print(f"\n[+]   document embeddings (dense len {config.MAX_SEQ_LENGTH})", flush=True)
        t = time.time()
        pipe.doc_emb
        print(f"      ok ({time.time() - t:.1f}s) -> {config.EMBEDDINGS_CACHE.name}")

    # -- report -----------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"prefetch complete in {time.time() - t0:.1f}s")
    print("cache contents:")
    total = 0
    for p in sorted(config.CACHE_DIR.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            total += sz
            print(f"  {sz / 1e6:9.1f} MB  {p.relative_to(config.CACHE_DIR)}")
    print(f"  {total / 1e6:9.1f} MB  total")
    print("\nThe batch job can now run with HF_HUB_OFFLINE=1.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
