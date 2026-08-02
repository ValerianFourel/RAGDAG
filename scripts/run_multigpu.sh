#!/usr/bin/env bash
# Run the experiment across every visible GPU, then merge.
#
#   ./scripts/run_multigpu.sh            # use all GPUs nvidia-smi reports
#   ./scripts/run_multigpu.sh 4          # force 4 workers
#   N_QUERIES=40 ./scripts/run_multigpu.sh 4    # sharded smoke test
#
# Sharding is by query, one worker per GPU. Each worker writes
# results/shards/*_<i>of<N>.parquet; the merge pass concatenates them, runs the
# DoubleML stage (which needs the full panel and is CPU-bound), and writes
# results/REPORT.md.
set -euo pipefail
cd "$(dirname "$0")/.."

# Resolve which GPUs we may actually use.
#
# CUDA_VISIBLE_DEVICES takes precedence over nvidia-smi: on a shared cluster
# node SLURM allocates a subset (say "2,3,6,7") and sets this variable, while
# nvidia-smi cheerfully lists every physical GPU on the box. Trusting nvidia-smi
# would spawn workers on GPUs belonging to somebody else's job.
DEVS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  while IFS= read -r line; do DEVS+=("$line"); done < <(
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' '
  )
fi

# An explicit count argument caps (never extends) the resolved device list.
if [[ -n "${1:-}" ]]; then
  if (( ${#DEVS[@]} > $1 )); then
    DEVS=("${DEVS[@]:0:$1}")
  elif (( ${#DEVS[@]} < $1 )); then
    echo "Requested $1 worker(s) but only ${#DEVS[@]} GPU(s) are visible." >&2
    if (( ${#DEVS[@]} == 0 )); then
      echo "Running $1 CPU worker(s) instead." >&2
      for (( j=0; j<$1; j++ )); do DEVS+=(""); done
    fi
  fi
fi

if (( ${#DEVS[@]} == 0 )); then
  echo "No GPUs detected; falling back to a single CPU process." >&2
  DEVS=("")
fi
N_GPUS="${#DEVS[@]}"

PY="${PYTHON:-python}"
export TOKENIZERS_PARALLELISM=false
# Each worker gets one GPU and should not also try to grab every CPU core.
TOTAL_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
export N_TORCH_THREADS="${N_TORCH_THREADS:-$(( TOTAL_CPUS / N_GPUS > 0 ? TOTAL_CPUS / N_GPUS : 1 ))}"

DS_TAG=$(${PY:-python} -c "import config; print(config.RESULTS_DIR.name)" 2>/dev/null || echo default)
mkdir -p logs "results/$DS_TAG/shards"

echo "=== dataset: ${DATASET:-<config default>}  (results/$DS_TAG) ==="
echo "=== launching $N_GPUS worker(s) on GPU(s) [${DEVS[*]}], $N_TORCH_THREADS torch threads each ==="

# Warm the shared read-only caches (corpus, BM25 index, document embeddings)
# in ONE process first. Without this, N workers race to build and write the
# same cache files on first run and can corrupt each other's output.
CUDA_VISIBLE_DEVICES="${DEVS[0]}" $PY -u -c "
import config, pipeline, interventions
config.set_seeds()
print('[warmup]', config.device_banner())
corpus, queries = pipeline.load_corpus_and_queries()
p = pipeline.RetrievalPipeline(corpus)
p.bm25; p.doc_emb
# Term-sampler frequency tables: a full re-tokenisation of the collection,
# read-only and identical for every worker. Built here so the 4 shards load it
# instead of each spending minutes of allocated GPU time rebuilding it.
interventions.build_vocab_stats(corpus)
interventions.build_bm25_df(corpus)
print('[warmup] corpus, BM25 index, embeddings and vocabulary tables are cached')
" 2>&1 | tee "logs/${DS_TAG}_warmup.log"

pids=()
for (( i=0; i<N_GPUS; i++ )); do
  CUDA_VISIBLE_DEVICES="${DEVS[$i]}" $PY -u -m run_all \
    --shard "$i" --n-shards "$N_GPUS" "${@:2}" \
    > "logs/${DS_TAG}_shard_${i}of${N_GPUS}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "  worker $i -> GPU ${DEVS[$i]:-cpu} (pid $pid, logs/${DS_TAG}_shard_${i}of${N_GPUS}.log)"
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "!! worker $i FAILED - see logs/${DS_TAG}_shard_${i}of${N_GPUS}.log" >&2
    fail=1
  fi
done
if (( fail )); then
  echo "Refusing to merge: at least one worker failed." >&2
  exit 1
fi

echo "=== all workers finished; merging ==="
$PY -u -m run_all --merge --n-shards "$N_GPUS" "${@:2}" 2>&1 | tee "logs/${DS_TAG}_merge.log"
