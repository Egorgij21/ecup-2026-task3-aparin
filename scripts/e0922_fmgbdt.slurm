#!/bin/bash
# e0922 H2a: does a frozen Chronos embedding ADD signal to the GBDT as extra features? (IDEAS.md §I29)
# Chronos mean-pool embed (GPU) + 665 features -> LightGBM [665] vs [665 + FM-PCA32], fold-4.
# andrena: GPU for the embedding pass, 16 CPUs for LightGBM. envs/chronos has chronos+lightgbm+sklearn.
#SBATCH --job-name=e0922fmgb
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 NUMEXPR_MAX_THREADS=16 HF_HOME="$ROOT/envs/hf_cache" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$ROOT/envs/chronos/bin/python"
"$PY" -c "import chronos, lightgbm, sklearn, torch; print('deps OK | cuda', torch.cuda.is_available())"
"$PY" src/run_fm_gbdt.py
echo "=== e0922 H2a DONE ==="
