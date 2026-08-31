#!/bin/bash
#SBATCH --job-name=usercv_mn
#SBATCH --partition=apini
#SBATCH --account=pilot_apini
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/ecup/logs/usercv_mn_%j.out
#SBATCH --error=/path/to/ecup/logs/usercv_mn_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$ROOT/envs/seq/bin/python" src/run_usercv.py "$@"
echo "=== USERCV MONTHNORM DONE ==="
