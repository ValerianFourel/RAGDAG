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

# --- offline on compute nodes, ONLINE on login nodes -----------------------
# HoreKa compute nodes have no outbound route, so inside a job the offline flags
# make a missing cache fail immediately instead of hanging on a socket until the
# walltime expires.
#
# Login nodes are the opposite case: that is where prefetch downloads models and
# where publish uploads results, both of which REQUIRE network. Forcing offline
# unconditionally broke `huggingface-cli login` with OfflineModeIsEnabled.
#
# SLURM_JOB_ID is set inside both sbatch and salloc, unset on a bare login node,
# which is exactly the distinction we need.
if [ -n "${RAGDAG_ONLINE:-}" ]; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  _ragdag_net="online (forced)"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  _ragdag_net="offline (in SLURM job $SLURM_JOB_ID)"
else
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  _ragdag_net="online (login node)"
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
echo "  network     $_ragdag_net"
echo "  threads     $N_TORCH_THREADS"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "  gpus        $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
else
  echo "  gpus        none visible (login node?)"
fi
unset _ragdag_src _cpus _gpus _ragdag_net
