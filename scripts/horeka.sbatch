#!/bin/bash
# RAGDAG on HoreKa (KIT) - 4x A100-40 on one node, query-sharded.
#
#   sbatch scripts/horeka.sbatch                             # default dataset
#   sbatch --export=ALL,DATASET=beir/scifact/test scripts/horeka.sbatch
#   sbatch --export=ALL,N_QUERIES=30 scripts/horeka.sbatch    # smoke test
#
#   # several collections in ONE job - avoids the 1-job-per-user QOS limit on
#   # dev_accelerated, and reuses the warm GPU/model state between datasets:
#   sbatch --export=ALL,DATASETS="beir/nfcorpus/test beir/scifact/test" \
#          --partition=accelerated --time=04:00:00 scripts/horeka.sbatch
#
# Results are written to results/<dataset-tag>/ so datasets never overwrite
# each other. DATASET must also have been prefetched on the login node.
#
# Prerequisite: run scripts/prefetch.py on a LOGIN node first. HoreKa compute
# nodes have no outbound internet, so the models and dataset must already be in
# $HF_HOME / $IR_DATASETS_HOME before this job starts.
#
#SBATCH --job-name=ragdag
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=logs/ragdag-%j.out
#SBATCH --error=logs/ragdag-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# ---------------------------------------------------------------------------
# Caches must live on a workspace, not $HOME: the HF cache plus the ONNX graph
# and embeddings run to a few hundred MB, and $HOME is quota'd. RAGDAG_WS is
# expected to point at a workspace (see the README's HoreKa section).
# ---------------------------------------------------------------------------
: "${RAGDAG_WS:=$PWD}"
export HF_HOME="${HF_HOME:-$RAGDAG_WS/hf}"
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RAGDAG_WS/ir_datasets}"

# Compute nodes are air-gapped; fail loudly rather than hang on a socket.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

source .venv/bin/activate

echo "=================================================================="
echo "job          $SLURM_JOB_ID on $(hostname)"
echo "partition    ${SLURM_JOB_PARTITION:-?}   cpus/task ${SLURM_CPUS_PER_TASK:-?}"
echo "DATASET      ${DATASET:-<config default>}"
echo "HF_HOME      $HF_HOME"
echo "started      $(date)"
echo "=================================================================="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', torch.cuda.device_count())"
python -c "import config; print('profile:', config.device_banner()); print(config.summary())"
echo "------------------------------------------------------------------"

# One worker per GPU, then the merge pass. The launcher reads
# CUDA_VISIBLE_DEVICES (which SLURM sets) rather than trusting nvidia-smi.
export PYTHON=python

# DATASETS (plural, space-separated) runs several collections sequentially in
# one allocation. A failure on one is reported and does NOT abort the rest -
# losing four completed datasets because the fifth had no prefetched corpus
# would be the wrong trade.
FAILED=""
for ds in ${DATASETS:-${DATASET:-__default__}}; do
  if [ "$ds" != "__default__" ]; then export DATASET="$ds"; fi
  echo
  echo "##################################################################"
  echo "# dataset: ${DATASET:-<config default>}   $(date +%H:%M:%S)"
  echo "##################################################################"
  if srun --ntasks=1 --export=ALL bash scripts/run_multigpu.sh; then
    python -c "import config; print('   OK ->', config.RESULTS_DIR)"
  else
    echo "!! FAILED: ${DATASET:-<default>}" >&2
    FAILED="$FAILED ${DATASET:-<default>}"
  fi
done
[ -n "$FAILED" ] && echo "!! datasets that failed:$FAILED" >&2

echo "------------------------------------------------------------------"
echo "finished     $(date)"
python -c "import config; print('results ->', config.RESULTS_DIR)"
