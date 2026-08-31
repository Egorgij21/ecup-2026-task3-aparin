#!/bin/bash
# Tiny utility runner so nothing Python ever executes on the login node.
#   sbatch scripts/util.slurm src/collect.py
#SBATCH --job-name=util
#SBATCH --partition=computeshort
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:20:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"
export PYTHONUNBUFFERED=1
"$ROOT/envs/ecup/bin/python" "$@"
