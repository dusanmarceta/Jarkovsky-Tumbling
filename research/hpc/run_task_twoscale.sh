#!/usr/bin/env bash
# Run ONE two-scale task (research/superfast) for one manifest line. Same interface as run_task.sh:
#
#   run_task_twoscale.sh <manifest.jsonl> <task-list.txt> [results_root]
#
# Results go to <results_root>/runs_twoscale/<run_id>/result.json (never into the full solver's runs/<run_id>/).
# Optional: TWOSCALE_NUMERICS='{"n_loc": 48}' (JSON overrides of superfast.twoscale.Numerics).
set -euo pipefail
MANIFEST=$1; TASKLIST=$2; ROOT=${3:-$(dirname "$MANIFEST")}
IDX=${SLURM_ARRAY_TASK_ID:-${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-${SGE_TASK_ID:-${LSB_JOBINDEX:-${TASK_INDEX:?no array index}}}}}}
IDX=$((IDX - ${TASK_INDEX_BASE:-0}))
LINE=$(sed -n "$((IDX + 1))p" "$TASKLIST")
[ -n "$LINE" ] || { echo "task $IDX: no line in $TASKLIST"; exit 2; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MPLBACKEND=Agg PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$ROOT/numba_cache}
cd "$REPO/research"
if [ -n "${TWOSCALE_NUMERICS:-}" ]; then
  exec "$VENV/bin/python" -m superfast.run --manifest "$MANIFEST" --index "$LINE" --root "$ROOT" --numerics "$TWOSCALE_NUMERICS"
fi
exec "$VENV/bin/python" -m superfast.run --manifest "$MANIFEST" --index "$LINE" --root "$ROOT"
