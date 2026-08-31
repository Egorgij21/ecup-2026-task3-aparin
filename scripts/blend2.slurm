#!/bin/bash
#SBATCH --job-name=blend2
#SBATCH --partition=compute
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:40:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PY="$ROOT/envs/ecup/bin/python"
OLD="e0100 e0101 e0101s1 e0101s2 e0101s3 e0102 e0108"
NEW="e0110 e0111 e0112 e0113"
echo "########## seq family: OLD 7 (reference = 1.76323 equal-weight) ##########"
"$PY" src/blend.py $OLD
echo; echo "########## seq family: OLD 7 + NEW 4 = 11 members ##########"
"$PY" src/blend.py $OLD $NEW
echo; echo "########## family weight sweep with the enlarged seq family ##########"
"$PY" src/blend_fixed.py --group gbdt e0049 e0064 --group seq $OLD $NEW
