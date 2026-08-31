#!/bin/bash
# e0924 HYBRID: GMV per-patch GRU (strong, e0919) + 12 other channels mean-pooled (decorrelated, e0923)
# -> head. IDEAS.md §I29 strength-recovery. andrena GPU. Runs --smoke first (3 ch, small) to validate
# the combined path; flip off --smoke and resubmit for the full fold-4.
#SBATCH --job-name=e0924hyb
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 NUMEXPR_MAX_THREADS=16 HF_HOME="$ROOT/envs/hf_cache" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$ROOT/envs/chronos/bin/python"
"$PY" -c "import chronos, sklearn, torch; print('deps OK | cuda', torch.cuda.is_available())"
"$PY" src/run_tsfm_hybrid.py --exp-id e0924 --fold 4
echo "=== e0924 hybrid full fold-4 DONE ==="
