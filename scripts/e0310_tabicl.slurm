#!/bin/bash
# e0310 -- TabICLv2 as a blend member. Two stages in one job: install (compute has internet,
# CLUSTER.md §3), then run. GPU: andrena with the TYPED gres string that `sinfo` reports --
# `gpu:nvidia_a100-pcie-40gb:1`. The bare `gpu:1` form is what the notes recorded as rejected.
#SBATCH --job-name=e0310tabicl
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof runs
PY="$ROOT/envs/seq/bin/python"          # the env that already has torch
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# install into the existing seq env if absent; BSD-3, no licence obstacle
"$PY" -c "import tabicl" 2>/dev/null || "$PY" -m pip install --quiet tabicl

# smoke: 2 folds, nothing written -- proves the model loads, fits and predicts at scale
"$PY" src/run_tabicl.py --config configs/e0310_tabicl.yaml --screen --no-log
echo "=== SMOKE OK, running full 5 folds ==="
"$PY" src/run_tabicl.py --config configs/e0310_tabicl.yaml
echo "=== e0310 DONE ==="
