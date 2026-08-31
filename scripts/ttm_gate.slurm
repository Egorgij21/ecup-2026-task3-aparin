#!/bin/bash
#SBATCH --job-name=ttm_gate
#SBATCH --partition=computeshort
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:50:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err

# CLUSTER.md 4: `compute`/`computeshort` have outbound internet, apini does NOT. Weights are
# fetched HERE into the shared cache; the GPU job then runs with HF_HUB_OFFLINE=1.
set -e
ROOT=/path/to/ecup
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export HF_HOME="$ROOT/envs/hf_cache"        # same cache MOMENT and Chronos-Bolt already use
PY="$ROOT/envs/tsfm/bin/python"

# TTM ships in granite-tsfm; transformers' PatchTSMixerModel is the fallback the gate tries next.
# Installed with --no-deps so it cannot drag a different torch/transformers into a working env
# (SESSION 2026-08-24 6: that is exactly how envs/ag's torchvision got broken).
"$PY" -c "import tsfm_public" 2>/dev/null || {
  echo "=== installing granite-tsfm (no-deps) into envs/tsfm ==="
  "$ROOT/envs/tsfm/bin/pip" install -q --no-deps granite-tsfm 2>&1 | tail -5 || \
  "$ROOT/envs/tsfm/bin/pip" install -q --no-deps "tsfm_public @ git+https://github.com/ibm-granite/granite-tsfm.git" 2>&1 | tail -5 || \
  echo "  install failed -- the gate will fall back to transformers"
}

"$PY" src/ttm_gate.py
echo "=== TTM GATE DONE ==="
