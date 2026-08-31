#!/bin/bash
# e0919 5-FOLD confirm -- Chronos-Bolt + scale-stats per-token GRU (IDEAS.md §I29). Fold-4 gave
# rho_B 0.664 (+0.013 over MOMENT/e0915) and rho_partial +0.0124, but fold-4 INFLATES (e0915's
# +0.0112 -> pooled -0.0004). This array gives the honest pooled rho_partial: pull the 5 OOFs, pool,
# run admissibility vs the family. One fold per task, ~15 min each in parallel.
#SBATCH --job-name=e0919cf
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --array=0-4
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 NUMEXPR_MAX_THREADS=16 HF_HOME="$ROOT/envs/hf_cache"
PY="$ROOT/envs/chronos/bin/python"
"$PY" -c "import chronos" || { echo "chronos env broken"; exit 3; }
"$PY" src/run_tsfm_chronos.py --exp-id e0919 --fold "$SLURM_ARRAY_TASK_ID"
echo "=== e0919 fold $SLURM_ARRAY_TASK_ID DONE ==="
