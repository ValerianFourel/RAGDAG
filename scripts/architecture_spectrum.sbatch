#!/bin/bash
# Architecture robustness over the existing paired admission interventions.
# One process deliberately: full-corpus score vectors are large on Quora.
#SBATCH --job-name=ragdag-arch
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ragdag-arch-%j.out
#SBATCH --error=logs/ragdag-arch-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source .venv/bin/activate

: "${RAGDAG_WS:=$PWD}"
export HF_HOME="${HF_HOME:-$RAGDAG_WS/hf}"
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RAGDAG_WS/ir_datasets}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ARCH_FINAL_K="${ARCH_FINAL_K:-50}"
export ARCH_FUSION_LAMBDAS="${ARCH_FUSION_LAMBDAS:-0,.25,.5,.75,1}"
export ARCH_CASCADE_BUDGETS="${ARCH_CASCADE_BUDGETS:-50,100,500,1000}"
export ARCH_N_BOOTSTRAP="${ARCH_N_BOOTSTRAP:-2000}"
export ARCH_RRF_C="${ARCH_RRF_C:-60}"

RAGDAG_DATASETS="${RAGDAG_DATASETS:-beir/nfcorpus/test beir/scifact/test beir/trec-covid beir/fiqa/test beir/scidocs beir/quora/test}"
RAGDAG_MODELS="${RAGDAG_MODELS:-BAAI/bge-small-en-v1.5 intfloat/e5-small-v2}"

python scripts/test_worlds.py
for model in $RAGDAG_MODELS; do
  case "$model" in
    BAAI/bge-small-en-v1.5) revision="${BGE_REVISION:-}" ;;
    intfloat/e5-small-v2) revision="${E5_REVISION:-}" ;;
    *) echo "unsupported model $model" >&2; exit 2 ;;
  esac
  [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo "immutable model revisions required" >&2; exit 2; }
  for ds in $RAGDAG_DATASETS; do
    echo "# ARCHITECTURE dataset=$ds model=$model $(date)"
    DATASET="$ds" DENSE_MODEL="$model" DENSE_MODEL_REVISION="$revision" K_CANDIDATES=50 \
      python -m architecture_admission
  done
done
