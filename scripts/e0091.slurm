#!/bin/bash
# Tuned LightGBM: full CV (writes oof/e0091.parquet + experiments.csv row), then the submission.
#SBATCH --job-name=e0091_tuned2
#SBATCH --partition=compute
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
PY="$ROOT/envs/ecup/bin/python"
"$PY" src/run.py --config configs/e0091_tuned2.yaml
echo "=== CV DONE, building submission ==="
"$PY" src/predict.py --config configs/e0091_tuned2.yaml --rounds 894
echo "=== e0091 DONE ==="
