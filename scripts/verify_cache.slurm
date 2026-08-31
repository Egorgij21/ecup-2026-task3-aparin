#!/bin/bash
# Prove the feature cache is byte-exact before any experiment is allowed to use it.
#   sbatch scripts/verify_cache.slurm                    # refuses if over quota
#   sbatch scripts/verify_cache.slurm --force            # ignore the quota refusal
# Memory: two full copies of the 665-feature matrix (~0.6G each) plus the panel.
#SBATCH --job-name=verify_cache
#SBATCH --partition=computeshort
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"
export PYTHONUNBUFFERED=1
"$ROOT/envs/ecup/bin/python" src/verify_cache.py "$@"
