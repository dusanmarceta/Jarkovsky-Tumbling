#!/usr/bin/env bash
# Run ONE simulation (one manifest line) as one scheduler task. Scheduler-agnostic.
#
#   run_task.sh <manifest.jsonl> <task-list.txt> [results_root]
#
# The task index comes from the scheduler's array variable (Slurm, PBS Pro/Torque, SGE/UGE, LSF) or $TASK_INDEX;
# line k of <task-list.txt> holds the manifest line number to run, so cost bins can be submitted as separate arrays.
# Requires: REPO (path of the research working copy) and VENV (its virtual environment) in the environment.
set -euo pipefail
MANIFEST=$1; TASKLIST=$2; ROOT=${3:-$(dirname "$MANIFEST")}
IDX=${SLURM_ARRAY_TASK_ID:-${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-${SGE_TASK_ID:-${LSB_JOBINDEX:-${TASK_INDEX:?no array index}}}}}}
IDX=$((IDX - ${TASK_INDEX_BASE:-0}))                    # SGE and LSF count from 1: set TASK_INDEX_BASE=1
LINE=$(sed -n "$((IDX + 1))p" "$TASKLIST")
[ -n "$LINE" ] || { echo "task $IDX: no line in $TASKLIST"; exit 2; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MPLBACKEND=Agg PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$ROOT/numba_cache}
cd "$REPO/research"
exec "$VENV/bin/python" -m tools.manifest run --manifest "$MANIFEST" --index "$LINE" --root "$ROOT"
