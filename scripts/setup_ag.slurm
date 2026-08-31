#!/bin/bash
# AutoGluon in its OWN venv (envs/ag), deliberately not envs/ecup.
#
# AutoGluon pins numpy/scikit-learn/lightgbm to its own ranges. Installing it into envs/ecup
# would silently move lightgbm off 4.6.0 and make every future CV number incomparable with
# the 60+ rows already in experiments.csv -- CLAUDE.md rule 9 (deterministic runs) and rule 3
# (the log stays valid). A second venv costs ~6 GB and keeps the frozen stack frozen.
#SBATCH --job-name=setup_ag
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE      # `compute` has internet; profile may set these

BASEPY=/data/home/<user>/miniconda3/bin/python3   # envs/<project>'s ensurepip is broken
rm -rf "$ROOT/envs/ag"
"$BASEPY" -m venv --copies "$ROOT/envs/ag"
PY="$ROOT/envs/ag/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools

# CPU-only torch: the full CUDA wheel is ~2.5 GB of dead weight on a CPU partition, and
# scratch is shared and already over its soft quota.
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install "autogluon.tabular[all]"
"$PY" -m pip install autogluon.timeseries || echo "WARN: autogluon.timeseries failed"
"$PY" -m pip install polars pyarrow pyyaml || true

echo "=== versions ==="
"$PY" - <<'PYEOF'
import importlib, sys
print("python", sys.version)
for m in ["autogluon.tabular","autogluon.timeseries","autogluon.core",
          "numpy","pandas","sklearn","lightgbm","torch","polars"]:
    try:
        print(f"{m:22s} {getattr(importlib.import_module(m),'__version__','?')}")
    except Exception as e:
        print(f"{m:22s} FAILED: {type(e).__name__}: {e}")
# the one fit() kwarg the whole design depends on -- confirm it exists in THIS version
from autogluon.tabular import TabularPredictor
import inspect
src = inspect.getdoc(TabularPredictor.fit) or ""
for kw in ["groups", "tuning_data", "num_bag_folds", "num_stack_levels", "use_bag_holdout"]:
    print(f"fit kwarg {kw:18s} documented: {kw in src}")
PYEOF

"$PY" -m pip freeze > "$ROOT/envs/requirements.ag.lock.txt"
du -sh "$ROOT/envs/ag"
echo "=== AG ENV SETUP DONE ==="
