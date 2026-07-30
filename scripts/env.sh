# Source me, don't execute me:   source scripts/env.sh
#
# Sets up a shell for RAGDAG on either a login node or a compute node.
# Safe to source repeatedly.
#
#   source scripts/env.sh          # auto: offline iff no internet expected
#   RAGDAG_ONLINE=1 source scripts/env.sh   # force online (login node, prefetching)

# --- locate the repo whether sourced from bash or zsh ----------------------
_ragdag_src="${BASH_SOURCE[0]:-${(%):-%x}}"
RAGDAG_DIR="$(cd "$(dirname "$_ragdag_src")/.." && pwd)"
cd "$RAGDAG_DIR" || return 1

: "${RAGDAG_WS:=$RAGDAG_DIR}"
export RAGDAG_WS
export HF_HOME="${HF_HOME:-$RAGDAG_WS/hf}"
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RAGDAG_WS/ir_datasets}"
export TOKENIZERS_PARALLELISM=false

# --- offline on compute nodes ---------------------------------------------
# HoreKa compute nodes have no outbound route. Setting these makes a missing
# cache fail immediately and legibly instead of hanging on a socket until the
# job's walltime expires.
if [ -n "${RAGDAG_ONLINE:-}" ]; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
else
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

# --- thread budget ---------------------------------------------------------
# Login nodes report every core on the box (152 on HoreKa). Letting torch grab
# all of them on a shared login node is both antisocial and slower than 8. In a
# job, divide the allocation across the workers we are about to fork.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  _cpus="${SLURM_CPUS_PER_TASK:-8}"
  _gpus=$(printf '%s' "${CUDA_VISIBLE_DEVICES:-0}" | awk -F, '{print NF}')
  [ "${_gpus:-1}" -lt 1 ] && _gpus=1
  export N_TORCH_THREADS=$(( _cpus / _gpus > 0 ? _cpus / _gpus : 1 ))
else
  export N_TORCH_THREADS="${N_TORCH_THREADS:-8}"
fi

[ -f .venv/bin/activate ] && . .venv/bin/activate

echo "RAGDAG env ready"
echo "  repo        $RAGDAG_DIR"
echo "  workspace   $RAGDAG_WS"
echo "  python      $(command -v python) ($(python -V 2>&1))"
echo "  offline     HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
echo "  threads     $N_TORCH_THREADS"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "  gpus        $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
else
  echo "  gpus        none visible (login node?)"
fi
unset _ragdag_src _cpus _gpus
