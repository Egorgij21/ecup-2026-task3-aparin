#!/bin/bash
# e0923 H1: multi-channel frozen-FM member (IDEAS.md §I29). All channels -> Chronos mean-pool -> concat
# -> PCA -> +scale-stats -> head. andrena GPU. Runs --smoke first (3 ch, 2 anchors x 3k, 6k val) to
# validate the multi-channel path; flip off --smoke and resubmit for the full 13-channel fold-4.
#SBATCH --job-name=e0923h1
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
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
"$PY" src/run_tsfm_h1.py --exp-id e0923 --fold 4
echo "=== e0923 H1 full 13-channel fold-4 DONE ==="
