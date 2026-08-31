#!/bin/bash
#SBATCH --job-name=mksub
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; export PYTHONUNBUFFERED=1
PY="$ROOT/envs/ecup/bin/python"
echo "### step 1: gbdt family average (e0049, e0064)"
"$PY" src/blend_submit.py --members e0049 e0064 --out e0120g
echo; echo "### step 2: seq family average (7 members)"
"$PY" src/blend_submit.py --members e0100 e0101 e0101s1 e0101s2 e0101s3 e0102 e0108 --out e0120s
echo; echo "### step 3: 50/50 across families -> e0120  (CV 1.76280)"
"$PY" src/blend_submit.py --members e0120g e0120s --out e0120
echo "=== MKSUB DONE ==="
