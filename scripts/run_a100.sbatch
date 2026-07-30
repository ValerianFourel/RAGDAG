#!/bin/bash
#SBATCH --job-name=causal-retrieval
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/causal-retrieval-%j.out
#SBATCH --error=logs/causal-retrieval-%j.out
#
# Sharded across all 4 A100s: one worker per GPU, each owning a strided slice
# of the 323 queries, then a merge pass.
#
# The GPU-bound stages (baseline, interventions, mediation, stability) scale
# close to linearly because they are independent per query. The DoubleML stage
# does NOT: it needs the full (query, document) panel and is LightGBM on CPU,
# so it runs once in the merge pass. Request plenty of CPU cores for it.
#
# For a single GPU instead:  #SBATCH --gres=gpu:a100:1  and the script still
# works unchanged - it uses however many GPUs it can see.
#
# Submit with:  sbatch scripts/run_a100.sbatch
set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# module load cuda/12.4        # uncomment / adjust for your cluster
source .venv/bin/activate

# Keep HF and ir_datasets caches on fast local scratch if the cluster has it,
# otherwise they land in $HOME and can be slow or quota-limited.
export HF_HOME="${HF_HOME:-$PWD/cache/hf}"
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$PWD/cache/ir_datasets}"
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

# Design-document settings are the defaults on GPU (K_CANDIDATES=50,
# MAX_TARGET_DOCS=10, CE_MAX_LENGTH=512) - no overrides needed.
export PYTHON=python
# One srun task; the launcher itself forks one worker per visible GPU.
srun --ntasks=1 --export=ALL bash scripts/run_multigpu.sh
