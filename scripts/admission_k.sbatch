#!/bin/bash
# Admission stages (7 lexical + 8 dense) at one pool depth K.
#
# Stages 7 and 8 are re-analyses of an existing interventions.parquet: they
# need a GPU but no reranking, so this is one card for a few minutes rather
# than the four-way sharded job that produced the interventions.
#
#   # primary run - repairs/refreshes results/<dataset-tag>/ at K=50
#   sbatch --export=ALL,RAGDAG_WS="$RAGDAG_WS",K_CANDIDATES=50 \
#          scripts/admission_k.sbatch
#
#   # K-sensitivity arm - writes results/<dataset-tag>_k20/ instead
#   sbatch --export=ALL,RAGDAG_WS="$RAGDAG_WS",K_CANDIDATES=20,RESULTS_SUFFIX=_k20 \
#          scripts/admission_k.sbatch
#
#   # a subset of collections (RAGDAG_DATASETS, *not* DATASETS - see below)
#   sbatch --export=ALL,...,RAGDAG_DATASETS="beir/nfcorpus/test beir/scifact/test" \
#          scripts/admission_k.sbatch
#
# Why K must be passed explicitly: config defaults it to 50 on a GPU node and
# 20 on a CPU node, so an unpinned job silently changes the experiment with the
# partition it lands on. Why RESULTS_SUFFIX: K is the conditioning event and
# the bar, so a K=20 run is a different experiment and must not share a
# directory with the K=50 one - mixing them once manufactured 972 impossible
# union rescues before the K guard existed.
#
#SBATCH --job-name=ragdag-adm
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ragdag-adm-%j.out
#SBATCH --error=logs/ragdag-adm-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

: "${RAGDAG_WS:=$PWD}"
export HF_HOME="${HF_HOME:-$RAGDAG_WS/hf}"
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RAGDAG_WS/ir_datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

source .venv/bin/activate

if [ -z "${K_CANDIDATES:-}" ]; then
  echo "refusing to run: K_CANDIDATES must be set explicitly (it is"
  echo "partition-dependent by default, which is how a CPU job once wrote"
  echo "K=20 panels over K=50 ones)." >&2
  exit 2
fi
export K_CANDIDATES
export RESULTS_SUFFIX="${RESULTS_SUFFIX:-}"

# NOT named DATASETS: HoreKa exports DATASETS=/hkfs/home/dataset/datasets
# system-wide, and --export=ALL inherits it, so a ${DATASETS:-default} would
# silently iterate over that path instead of the collections (it did once).
RAGDAG_DATASETS="${RAGDAG_DATASETS:-beir/nfcorpus/test beir/scifact/test beir/trec-covid beir/fiqa/test beir/scidocs beir/quora/test}"
for _ds in $RAGDAG_DATASETS; do
  case "$_ds" in
    */*) : ;;                       # a dataset id, e.g. beir/nfcorpus/test
    *) echo "refusing to run: '$_ds' is not a dataset id - set RAGDAG_DATASETS" >&2
       exit 2 ;;
  esac
  [ -e "$_ds" ] && { echo "refusing to run: '$_ds' is a filesystem path, not a"
                     echo "dataset id - a stray DATASETS/RAGDAG_DATASETS leaked in." >&2; exit 2; }
done

echo "=================================================================="
echo "RAGDAG admission  K=$K_CANDIDATES  suffix='${RESULTS_SUFFIX}'"
echo "  node        $(hostname)"
echo "  started     $(date)"
echo "  datasets    $RAGDAG_DATASETS"
echo "=================================================================="

python scripts/test_worlds.py || { echo "unit tests failed - aborting"; exit 1; }

# A suffixed run starts from an empty directory, so seed it with the intervention
# log from the primary run. The sampled do() operations are deliberately held
# fixed across K: only the bar changes, which is the whole point of the sweep.
if [ -n "$RESULTS_SUFFIX" ]; then
  for ds in $RAGDAG_DATASETS; do
    tag="${ds//\//-}"
    src="results/${tag}/interventions.parquet"
    dst="results/${tag}${RESULTS_SUFFIX}"
    if [ -f "$src" ]; then
      mkdir -p "$dst"
      cp -n "$src" "$dst/" && echo "seeded $dst/interventions.parquet"
    else
      echo "!! no $src - $ds will be skipped"
    fi
  done
fi

failed=""
for ds in $RAGDAG_DATASETS; do
  echo ""
  echo "# dataset: $ds   $(date +%H:%M:%S)"
  DATASET="$ds" python -m run_all --merge --n-shards 4 --only 7 8 \
    || { echo "!! FAILED: $ds"; failed="$failed $ds"; }
done

echo ""
echo "------------------------------------------------------------------"
echo "finished     $(date)"
[ -n "$failed" ] && echo "!! datasets that failed:$failed"
exit 0
