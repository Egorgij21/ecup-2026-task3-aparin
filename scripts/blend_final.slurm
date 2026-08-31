#!/bin/bash
#SBATCH --job-name=blend_final
#SBATCH --partition=compute
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; export PYTHONUNBUFFERED=1
"$ROOT/envs/ecup/bin/python" src/blend_final.py "$@"
