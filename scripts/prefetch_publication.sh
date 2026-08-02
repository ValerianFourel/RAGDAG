#!/usr/bin/env bash
# Networked-login-node prefetch for the complete admission-v2 release matrix.
set -euo pipefail
cd "$(dirname "$0")/.."

for value in BGE_REVISION E5_REVISION; do
  revision="${!value:-}"
  [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || {
    echo "$value must be an immutable 40-character Hugging Face commit" >&2
    exit 2
  }
done

datasets="${RAGDAG_DATASETS:-beir/nfcorpus/test beir/scifact/test beir/trec-covid beir/fiqa/test beir/scidocs beir/quora/test}"
for spec in "BAAI/bge-small-en-v1.5:$BGE_REVISION" "intfloat/e5-small-v2:$E5_REVISION"; do
  model="${spec%%:*}"
  revision="${spec##*:}"
  for dataset in $datasets; do
    echo "prefetch dataset=$dataset model=$model revision=$revision"
    DATASET="$dataset" DENSE_MODEL="$model" DENSE_MODEL_REVISION="$revision" \
      K_CANDIDATES=50 python scripts/prefetch.py
  done
done
