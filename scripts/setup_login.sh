#!/usr/bin/env bash
# Login-node setup: build the venv, install everything, prefetch models + data.
#
#   export RAGDAG_WS=$(ws_find ragdag)      # or any writable dir
#   bash scripts/setup_login.sh
#
# Idempotent — safe to re-run after a failure.
#
# Uses uv to fetch a standalone CPython 3.11 rather than the module system.
# On HoreKa `module load devel/python/3.11` does not exist, and a failed
# `module load` leaves you on the system Python 3.9, where pip aborts the whole
# requirements file at the first pin needing >=3.10 and installs NOTHING. That
# failure presents as a dozen unrelated ModuleNotFoundErrors, so we avoid it.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

echo "=============================================================="
echo "RAGDAG login-node setup"
echo "  repo         $REPO"
echo "  RAGDAG_WS    ${RAGDAG_WS:-<unset, defaulting to repo dir>}"
echo "=============================================================="

: "${RAGDAG_WS:=$REPO}"
export HF_HOME="${HF_HOME:-$RAGDAG_WS/hf}"
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$RAGDAG_WS/ir_datasets}"
mkdir -p "$HF_HOME" "$IR_DATASETS_HOME" logs results/shards

# ---------------------------------------------------------------- 1. uv -----
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "[1/4] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "[1/4] uv already present ($(command -v uv))"
fi
command -v uv >/dev/null 2>&1 || {
  echo
  echo "uv is not installed and could not be fetched (network blocked?)."
  echo "Fallback: find a Python >=3.10 through the module system instead —"
  echo "    module spider python"
  echo "    module keyword python"
  echo "then:  <that-python> -m venv .venv && source .venv/bin/activate"
  exit 1
}

# ------------------------------------------------------------- 2. venv ------
NEED_VENV=1
if [[ -x .venv/bin/python ]]; then
  V=$(.venv/bin/python -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if [[ "$(printf '%s\n3.10\n' "$V" | sort -V | head -1)" == "3.10" ]]; then
    echo "[2/4] reusing existing .venv (Python $V)"
    NEED_VENV=0
  else
    echo "[2/4] existing .venv is Python $V (<3.10) — rebuilding"
    rm -rf .venv
  fi
fi
if (( NEED_VENV )); then
  echo "[2/4] creating .venv with Python 3.11"
  uv venv --python 3.11 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -V

# --------------------------------------------------------- 3. packages ------
echo "[3/4] installing torch (cu124) then the rest"
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements-gpu.txt

python - <<'PY'
import importlib, sys
need = ["torch","transformers","sentence_transformers","numpy","pandas","scipy",
        "sklearn","doubleml","lightgbm","statsmodels","bm25s","Stemmer",
        "ir_datasets","matplotlib","pyarrow"]
missing = [m for m in need if not importlib.util.find_spec(m)]
if missing:
    sys.exit("STILL MISSING: " + ", ".join(missing))
import sklearn, numpy, torch
print(f"  torch {torch.__version__} | numpy {numpy.__version__} | sklearn {sklearn.__version__}")
from sklearn.utils.validation import check_X_y
import inspect
assert "force_all_finite" in inspect.signature(check_X_y).parameters, (
    f"scikit-learn {sklearn.__version__} dropped force_all_finite; "
    "doubleml 0.10.1 needs it. Pin scikit-learn<1.8.")
print("  doubleml/sklearn compatibility OK")
PY

# --------------------------------------------------------- 4. prefetch ------
echo "[4/4] prefetching models and dataset"
python scripts/prefetch.py

cat <<EOF

==============================================================
Setup complete. Submit with:

  export RAGDAG_WS="$RAGDAG_WS"
  sbatch --export=ALL,RAGDAG_WS="\$RAGDAG_WS",N_QUERIES=30 \\
         --partition=dev_accelerated --time=00:30:00 scripts/horeka.sbatch

then the full run:

  sbatch --export=ALL,RAGDAG_WS="\$RAGDAG_WS" scripts/horeka.sbatch
==============================================================
EOF
