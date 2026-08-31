#!/bin/bash
# e0915 5-FOLD confirm of the per-patch GRU over MOMENT tokens (IDEAS.md §I28) -- the one positive
# lead. One fold per array task -> ~1h each in parallel. Pooled OOF -> LOFO admissibility vs champion.
#SBATCH --job-name=e0915cf
#SBATCH --partition=compute
#SBATCH --mem=240G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 NUMEXPR_MAX_THREADS=16 HF_HOME="$ROOT/envs/hf_cache"
PY="$ROOT/envs/tsfm/bin/python"
"$PY" -c "import momentfm" || { echo "env broken"; exit 3; }
"$PY" src/run_tsfm_gru.py --mode patch --exp-id e0915 --fold "$SLURM_ARRAY_TASK_ID"
echo "=== e0915 fold $SLURM_ARRAY_TASK_ID DONE ==="
