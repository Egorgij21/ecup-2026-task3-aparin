#!/bin/bash
# Generic runner for src/run_usercv.py (CAUSAL_EXP.md user-split GRU) on ANDRENA.
#
# apini's GPU QOS is saturated by the co-tenant project, whose jobs CLUSTER.md forbids
# cancelling; andrena is a separate partition with a separate QOS and started our work
# instantly. CLUSTER.md 4: andrena needs --account=pilot_andrena and --gres=gpu:1.
# Memory: the `full` panel is 250k x 409 x 85 fp16 = 17.4 GB, which fits an A100-40GB.
#
#   sbatch -J e0215 scripts/usercv.slurm --variant full --folds 0 1 2 3 4 --seeds 3 \
#          --epochs 60 --t-cut 2025-09-16
#
# Override the job name with -J so the log filename (%x) equals the exp_id -- CLUSTER.md 5b.
# Everything after the script name is forwarded verbatim to run_usercv.py, so the sbatch line
# IS the experiment record; run_usercv.py prints the protocol it is running on startup.
#SBATCH --job-name=usercv_a
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) : run_usercv.py $@ ==="
"$ROOT/envs/seq/bin/python" src/run_usercv.py "$@"
echo "=== USERCV DONE ==="
