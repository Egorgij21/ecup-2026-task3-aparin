#!/bin/bash
# e0913 -- RealMLP via pytabkit in a FRESH venv (bypasses the AG env's broken torchvision).
# Builds envs/realmlp once (compute has internet), gates on a clean pytabkit import, then runs
# the standalone runner. IDEAS.md §I22 -- the one open lever: does RealMLP hold rho~0.67 at
# NN_TORCH's r~0.96 (=> strongly positive excess)? Fold-4 feasibility, 1.2M subsample, CPU.
#SBATCH --job-name=e0913rmlp
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag envs
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 NUMEXPR_MAX_THREADS=16

# build the fresh env once (matched torch+pytabkit; standalone, no AG, no torchvision needed)
if [ ! -x envs/realmlp/bin/python ]; then
  echo "=== building envs/realmlp ==="
  /data/home/<user>/miniconda3/bin/python3 -m venv envs/realmlp
  envs/realmlp/bin/pip install -q --upgrade pip
  envs/realmlp/bin/pip install -q pytabkit polars pyarrow numpy scikit-learn 2>&1 | tail -8
fi
PY="$ROOT/envs/realmlp/bin/python"
"$PY" -c "import pytabkit,polars,numpy; from pytabkit import RealMLP_TD_Regressor; print('pytabkit import OK')" \
  || { echo "pytabkit import FAILED -- aborting (env unusable)"; exit 3; }

echo "=== running RealMLP fold 4 ==="
"$PY" src/run_realmlp.py
echo "=== e0913 DONE ==="
