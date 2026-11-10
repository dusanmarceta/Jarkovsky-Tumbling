# Running the campaign on the cluster

One simulation = one scheduler task, single-threaded (`NUMBA_NUM_THREADS=1`), about 0.6–2 GB memory. Every run
writes only into `<root>/runs/<run_id>/`; meshes and caches are prepared once before any array starts.

## Steps

1. **Copy and environment.** Copy the research working copy (git clone or rsync) to the cluster. Create a
   Python ≥ 3.10 virtual environment and install `research/environment/requirements-lock.txt`. Use pip wheels if
   compute nodes have no internet.
2. **Tests.** `cd research && python -m pytest tests -q`. All must pass on the cluster's CPU.
3. **Speed factor.** Run `research/verification/run_quick_checks.py` once and compare the benchmark wall times with
   `research/runs/verification/raw/benchmark`. That gives the cluster/M4 speed factor.
4. **Prepare each manifest.**
   `python research/hpc/prepare.py research/campaign/manifests/<name>.jsonl --root $SCRATCH/<name> --speed-factor F`.
   This writes meshes, visible-facet caches, the Numba cache, and task lists by cost bin (1 h, 6 h, 24 h, 96 h).
5. **Array scripts.** `python research/hpc/make_slurm_arrays.py --root ... --manifest ... --repo ... --venv ... --account ... --partition ...`
   writes one script per bin. Review them, then rerun with `--submit` (or use `sbatch`). For PBS, SGE or LSF, call
   `research/hpc/run_task.sh` from that scheduler's array syntax; it reads the array index from the usual variables.
6. **Check and retry.** `python -m tools.manifest check --manifest ...` lists lines that are not accepted. The task
   lists can be rebuilt from that list. Accepted runs are skipped automatically.
7. **Collect.** `python research/hpc/aggregate.py --manifest ... --root ... --out results.csv` writes one row per run
   (inputs, derived quantities, policy, drift, per-orbit history, diagnostics, status, commit).

**Order.** The convergence matrix first, then A (Morris), D (phases), B (Sobol′) and C (conventions). Each stage is
checked before the next.

## Information needed from the administrators

- **Access:** login host(s), SSH key or 2FA procedure, VPN; data transfer method (scp, rsync or Globus).
- **Scheduler:** type and version (Slurm, PBS Pro/Torque, SGE/UGE, LSF); partitions and queues with maximum wall
  time; maximum array size; maximum queued and running jobs per user; QoS or accounting string.
- **Allocation:** core-hours (or node-hours) granted, expiry date, how single-core jobs are charged (per core or per
  whole node; if whole-node, tasks must be packed, e.g. with GNU parallel inside one job).
- **Nodes:** CPU model, cores per node, memory per node or per core, local scratch per node.
- **Storage:** shared file system paths for home, project and scratch; quotas in GB *and number of files*
  (about 6 files per run plus one mesh per state); purge policy on scratch.
- **Software:** module system and available Python versions (≥ 3.10); internet from compute nodes (pip) or not;
  Apptainer or Singularity support.
- **Policy:** any limits on array jobs or throttling, and on many small jobs.
