#!/bin/bash
# Drop stale feature-cache generations.
#   sbatch scripts/clear_cache.slurm         # keep the generation matching the current code
#   sbatch scripts/clear_cache.slurm --all   # drop everything, including the current one
#
# A "generation" is one hash of features.py + data.py. Editing either file starts a new one
# and orphans the old, so without this the cache grows by ~0.6G x 81 anchors every time the
# feature code is touched. Scratch is shared and already over its soft quota, so unbounded
# growth is not an option -- see CLUSTER.md gotcha 1.
#SBATCH --job-name=clear_cache
#SBATCH --partition=computeshort
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
C="$ROOT/data/featcache"
[ -d "$C" ] || { echo "no cache directory -- nothing to do"; exit 0; }

echo "before:"; du -sh "$C"
echo "generations:"; du -sh "$C"/*/ 2>/dev/null || true

if [ "$1" = "--all" ]; then
  KEEP="__none__"
  echo "--all given: dropping every generation"
else
  KEEP=$("$ROOT/envs/ecup/bin/python" -c \
    "import sys;sys.path.insert(0,'$ROOT/src');import features;print(features._code_hash())")
  echo "keeping generation $KEEP (matches the current features.py + data.py)"
fi

for d in "$C"/*/; do
  b=$(basename "$d")
  [ "$b" = "$KEEP" ] || { echo "  removing stale generation $b"; rm -rf "$d"; }
done

# Temp files from jobs that were killed mid-write. The rename is atomic, so these are never
# live cache entries -- only litter. Anything still being written is under an hour old.
find "$C" -name "*.tmp.npz" -mmin +60 -print -delete 2>/dev/null || true

echo "after:"; du -sh "$C"
/usr/lpp/mmfs/bin/mmlsquota --block-size auto gpfsFlash | tail -1
