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
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ragdag-adm-%j.out
#SBATCH --error=logs/ragdag-adm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

: "${RAGDAG_WS:=$PWD}"
export HF_HOME="${HF_HOME:-$RAGDAG_WS/hf}"
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RAGDAG_WS/ir_datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

source .venv/bin/activate

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
echo "RAGDAG admission-v2 publication experiment"
echo "  node        $(hostname)"
echo "  started     $(date)"
echo "  datasets    $RAGDAG_DATASETS"
echo "=================================================================="

python scripts/test_worlds.py || { echo "unit tests failed - aborting"; exit 1; }

RAGDAG_MODELS="${RAGDAG_MODELS:-BAAI/bge-small-en-v1.5 intfloat/e5-small-v2}"
for model in $RAGDAG_MODELS; do
  model_tag="${model##*/}"
  case "$model" in
    BAAI/bge-small-en-v1.5) revision="${BGE_REVISION:-}" ;;
    intfloat/e5-small-v2) revision="${E5_REVISION:-}" ;;
    *) echo "unsupported publication model: $model" >&2; exit 2 ;;
  esac
  [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || {
    echo "set BGE_REVISION and E5_REVISION to immutable 40-character HF commits" >&2
    exit 2
  }
  for ds in $RAGDAG_DATASETS; do
    echo ""
    echo "# PRIMARY dataset=$ds model=$model K=50 $(date +%H:%M:%S)"
    DATASET="$ds" DENSE_MODEL="$model" DENSE_MODEL_REVISION="$revision" K_CANDIDATES=50 \
      ./scripts/run_multigpu.sh 4 --only 2 7 8 --force

    # Hold the sampled trials fixed in the K arms; only the admission bar moves.
    src_dir="$(DATASET="$ds" DENSE_MODEL="$model" DENSE_MODEL_REVISION="$revision" K_CANDIDATES=50 python -c 'import config; print(config.RESULTS_DIR)')"
    src="$src_dir/interventions.parquet"
    for k in 20 100; do
      dst="$(DATASET="$ds" DENSE_MODEL="$model" DENSE_MODEL_REVISION="$revision" K_CANDIDATES="$k" python -c 'import config; print(config.RESULTS_DIR)')"
      mkdir -p "$dst"
      cp "$src" "$dst/interventions.parquet"
      cp "$src_dir/target_sampling_frame.parquet" "$dst/"
      cp "$src_dir/origin_documents.parquet" "$dst/"
      cp "$src_dir/shard_completion.json" "$dst/"
      echo "# SENSITIVITY dataset=$ds model=$model K=$k $(date +%H:%M:%S)"
      DATASET="$ds" DENSE_MODEL="$model" DENSE_MODEL_REVISION="$revision" K_CANDIDATES="$k" \
        python -m run_all --merge --n-shards 4 --only 7 8 --force
    done
  done
done

echo ""
echo "------------------------------------------------------------------"
echo "finished     $(date)"
python scripts/meta_analyze.py
python scripts/check_admission.py --release
