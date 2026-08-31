#!/bin/bash
#SBATCH --job-name=e0070_e0073
#SBATCH --partition=apini
#SBATCH --account=pilot_apini
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1
C=(configs/e0070_recent6.yaml configs/e0071_recent10.yaml configs/e0072_recent14.yaml configs/e0073_recent18.yaml)
"$ROOT/envs/ecup/bin/python" src/run.py --config "${C[$SLURM_ARRAY_TASK_ID]}"
