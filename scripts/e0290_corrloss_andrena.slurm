#!/bin/bash
# e0290-e0293 -- IDEAS.md §I19, the within-day Pearson-loss GRU sweep. Four full-CV runs in one
# GPU job (each e0101-family run is ~3.7 min for 5 folds, so all four fit comfortably).
# GPU job -> apini (CLUSTER.md rule 5b: this trains a torch model). --gres REQUIRED.
#SBATCH --job-name=e0290_e0293A
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs oof subs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="$ROOT/envs/seq/bin/python"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for cfg in e0290 e0291 e0292 e0293; do
  echo "=== RUN ${cfg} ==="
  "$PY" src/run_seq.py --config configs/${cfg}_corrloss.yaml
done
echo "=== CORRLOSS SWEEP DONE ==="
