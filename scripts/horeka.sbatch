#!/bin/bash
# RAGDAG on HoreKa (KIT) - 4x A100-40 on one node, query-sharded.
#
#   sbatch scripts/horeka.sbatch                             # default dataset
#   sbatch --export=ALL,DATASET=beir/scifact/test scripts/horeka.sbatch
#   sbatch --export=ALL,N_QUERIES=30 scripts/horeka.sbatch    # smoke test
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
srun --ntasks=1 --export=ALL bash scripts/run_multigpu.sh

echo "------------------------------------------------------------------"
echo "finished     $(date)"
python -c "import config; print('results ->', config.RESULTS_DIR)"
