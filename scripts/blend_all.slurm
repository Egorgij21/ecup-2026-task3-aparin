#!/bin/bash
# Assemble AutoGluon's OOF into blendable form, then blend across approaches.  CPU-only.
#   sbatch scripts/blend_all.slurm
#SBATCH --job-name=blend_all
#SBATCH --partition=compute
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:40:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err

set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
PY="$ROOT/envs/ecup/bin/python"

echo "########## assemble e0064 (AutoGluon, our LB best) ##########"
"$PY" src/ag_oof_to_parquet.py e0064

echo; echo "########## gbdt only -- the door EXPERIMENTS.md §2 measured as shut ##########"
"$PY" src/blend.py e0049 e0064

echo; echo "########## seq family only ##########"
"$PY" src/blend.py e0100 e0101 e0102

echo; echo "########## best gbdt + best seq ##########"
"$PY" src/blend.py e0049 e0101

echo; echo "########## everything ##########"
"$PY" src/blend.py e0049 e0064 e0100 e0101 e0102
echo "=== BLEND ALL DONE ==="
