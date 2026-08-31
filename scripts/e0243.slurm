#!/bin/bash
#SBATCH --job-name=e0243_mag
#SBATCH --partition=compute
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof runs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
# IDEAS.md I13. tasks 0-2 = BUYERS-only treatment (seeds 0/1/2);
#               tasks 3-5 = ALL-users same-harness CONTROL (seeds 0/1/2).
# The 3 seeds per arm double as the reseeded-bag control IDEAS.md E1 requires.
CFGS=(configs/e0243s0_magnitude.yaml configs/e0243s1_magnitude.yaml configs/e0243s2_magnitude.yaml \
      configs/e0244s0_magnitude.yaml configs/e0244s1_magnitude.yaml configs/e0244s2_magnitude.yaml)
CFG=${CFGS[$SLURM_ARRAY_TASK_ID]}
echo "=== task $SLURM_ARRAY_TASK_ID -> $CFG ==="
"$PY" src/metrics.py
"$PY" src/run.py --config "$CFG" ${SCREEN:+--screen}
echo "=== DONE $CFG ==="
