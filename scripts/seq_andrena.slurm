#!/bin/bash
# Same as scripts/seq.slurm but targets `andrena` instead of `apini`.
#
# WHY: apini's GPU QOS is saturated by the co-tenant project (WorkDir
# /path/to), whose jobs CLUSTER.md forbids cancelling, and our
# seq job has sat PENDING on (QOSMaxGRESPerUser) for hours. `andrena` is a separate partition
# with a separate QOS and had an idle node. CLUSTER.md 4: andrena's QOS requires --gres=gpu:1
# even for CPU-only work, and needs --account=pilot_andrena.
#
# Memory: the seq panel is 2.86 GB fp16 and observed GPU-resident use is 3.27 GB, so an
# A100-40GB is ample -- this does not need an H200.
#SBATCH --job-name=seq_a
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err

set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs oof subs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="$ROOT/envs/seq/bin/python"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID): run_seq.py $@ ==="
"$PY" src/run_seq.py "$@"
echo "=== SEQ DONE ==="
