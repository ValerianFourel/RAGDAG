"""Publish results to a HuggingFace dataset repo (and fetch them back).

**Run this on a login node, never inside the job.** HoreKa compute nodes have no
outbound route, so an upload from a worker hangs until the walltime expires.

    python scripts/publish.py --repo <user>/ragdag-results --dry-run
    python scripts/publish.py --repo <user>/ragdag-results
    python scripts/publish.py --repo <user>/ragdag-results --dataset beir/scifact/test
    python scripts/publish.py --repo <user>/ragdag-results --download

Repos are created **private** by default: results are unpublished research, and
making them public is a deliberate act, not a default. Pass ``--public`` to
override.

Only ``results/`` is uploaded. ``cache/`` is excluded on purpose - it holds a
91 MB ONNX graph and the document embeddings, all of which are regenerable from
the models and the corpus, and none of which belong in a results record.

Authentication: ``huggingface-cli login``, or set ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

#: Uploaded. Everything else under results/ is ignored.
INCLUDE = ["*.md", "*.json", "*.csv", "*.parquet", "*.png"]

CARD = """---
license: cc-by-4.0
tags: [information-retrieval, causal-inference, rag, beir]
---

# RAGDAG results

Artefacts from **RAGDAG** - treating a multi-stage retrieval pipeline as a
structural causal model and computing path-specific effects exactly by freezing
stages, rather than estimating them.

Code: https://github.com/ValerianFourel/RAGDAG

## Layout

One directory per collection, named after its `ir_datasets` id:

```
<dataset-tag>/
  REPORT.md                 human-readable report incl. the PASS/FAIL verdict
  MANIFEST.json             provenance: git SHA, code fingerprint, config, checksums
  baseline_ndcg.json        nDCG@10 per configuration + reranker_helps flag
  interventions.parquet     one row per (query, doc, term, arm) with all deltas
  mediation.parquet         per-pair path decomposition, both first-stage configs
  mediation_ratio.csv       aggregated mediation shares
  dml_comparison.csv        naive OLS vs DoubleML per concept
  stability.csv             RBO@10 per (variant, retriever config)
  fig_*.png                 figures
  shards/                   per-worker partials (multi-GPU runs)
```

## Reading these numbers

Each `MANIFEST.json` pins the exact code that produced its directory. Artefacts
from different `code_fingerprint` values are **not** comparable - the term
sampler was corrected twice during development, and mixing pre- and post-fix
runs would silently blend two different experiments.

Check `baseline_ndcg.json` → `reranker_helps` before interpreting mediation
shares. On some collections the MS MARCO-trained cross-encoder *degrades*
retrieval; the shares there describe causal responsibility for an
intervention's effect, not a well-configured pipeline.
"""


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=config.ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def result_dirs(only: str | None) -> list[Path]:
    """Result subtrees to publish; one per dataset."""
    if not config.RESULTS_ROOT.exists():
        return []
    if only:
        d = config.RESULTS_ROOT / only.replace("/", "-")
        return [d] if d.is_dir() else []
    return sorted(p for p in config.RESULTS_ROOT.iterdir() if p.is_dir())


def write_manifest(d: Path) -> dict:
    """Provenance record. Results that cannot be traced to code are not results."""
    files = sorted(
        p for p in d.rglob("*")
        if p.is_file() and p.name != "MANIFEST.json"
        and any(p.match(g) for g in INCLUDE)
    )
    meta_path = d / "run_meta.json"
    manifest = {
        "dataset": d.name,
        "published_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "code_fingerprint": config.code_fingerprint(),
        "run_meta": json.loads(meta_path.read_text()) if meta_path.exists() else None,
        "files": {
            str(p.relative_to(d)): {"bytes": p.stat().st_size, "sha256_16": _sha256(p)}
            for p in files
        },
    }
    (d / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.environ.get("HF_REPO"),
                    help="target dataset repo, e.g. user/ragdag-results (or $HF_REPO)")
    ap.add_argument("--dataset", default=None, help="publish only this ir_datasets id")
    ap.add_argument("--public", action="store_true", help="create the repo public")
    ap.add_argument("--dry-run", action="store_true", help="show what would upload")
    ap.add_argument("--download", action="store_true", help="fetch results instead")
    ap.add_argument("--out", default=None, help="download destination")
    args = ap.parse_args()

    if not args.repo:
        print("error: --repo (or $HF_REPO) is required, e.g. user/ragdag-results")
        return 2

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()

    # ------------------------------------------------------------------ down --
    if args.download:
        out = Path(args.out or (config.ROOT / "results_downloaded"))
        print(f"downloading {args.repo} -> {out}")
        snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=str(out))
        print("done")
        return 0

    dirs = result_dirs(args.dataset)
    if not dirs:
        print(f"nothing to publish under {config.RESULTS_ROOT}")
        return 1

    # ------------------------------------------------------------------- plan --
    print("=" * 72)
    print(f"{'DRY RUN - ' if args.dry_run else ''}publishing to {args.repo} "
          f"({'public' if args.public else 'PRIVATE'}, dataset repo)")
    print("=" * 72)
    total = 0
    for d in dirs:
        m = write_manifest(d)
        n = sum(f["bytes"] for f in m["files"].values())
        total += n
        print(f"\n  {d.name}   {len(m['files'])} files, {n / 1e6:.2f} MB")
        print(f"    git {m['git_sha'][:8]}{'-dirty' if m['git_dirty'] else ''}  "
              f"code {m['code_fingerprint']}")
        for name in list(m["files"])[:6]:
            print(f"      {name}")
        if len(m["files"]) > 6:
            print(f"      ... and {len(m['files']) - 6} more")
    print(f"\n  TOTAL {total / 1e6:.2f} MB   (cache/ excluded by design)")

    if args.dry_run:
        print("\ndry run - nothing uploaded. Re-run without --dry-run to publish.")
        return 0

    # ------------------------------------------------------------------- auth --
    try:
        who = api.whoami()["name"]
    except Exception:
        print("\nnot authenticated. Run `huggingface-cli login` or set HF_TOKEN.")
        return 2
    print(f"\nauthenticated as {who}")

    api.create_repo(args.repo, repo_type="dataset", private=not args.public,
                    exist_ok=True)

    card = config.ROOT / "results" / "README.md"
    card.write_text(CARD)
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset",
                    commit_message="Update dataset card")

    for d in dirs:
        print(f"  uploading {d.name} ...", flush=True)
        api.upload_folder(
            repo_id=args.repo, repo_type="dataset",
            folder_path=str(d), path_in_repo=d.name,
            allow_patterns=[*INCLUDE, "MANIFEST.json"],
            commit_message=f"{d.name} @ {_git('rev-parse', '--short', 'HEAD')}",
        )
    print(f"\nhttps://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
