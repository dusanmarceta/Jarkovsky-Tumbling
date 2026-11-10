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

## Waves (branch `research/paper1-hpc-v2`)

The frozen manifests are not submitted one by one. `research/campaign/waves.py` checks them against
`FREEZE_checksums.json`, applies the exclusion list (`research/campaign/exclusions/outside_1d_v1.json`), puts every
unique run in exactly one wave and writes a deduplicated wave manifest. Counts and run-ID hashes are recorded in
`research/campaign/WAVES.json`.

| Wave | Content | Runs | Core-h (M4) | Starts when |
|---|---|---|---|---|
| 0 | convergence batch V (executable lines) | 78 | ~880 | HPC access |
| 1 | slow + ordinary, e ≤ 0.6 | 4,810 | ~23,800 | V passes |
| 1b | V2 convergence corners C1–C7 (`research/campaign/manifests_v2/`) | 106 | ~2,740 | alongside wave 1 |
| 2 | e ∈ (0.6, 0.8] and the executable fast stratum | 2,000 | ~28,400 | C1–C5 pass |
| 3 | preserved low-Γ ordinary runs (`full_long`, 101–256 core-h each) | 88 | ~14,600 | C7 (and C1–C3) pass; long queue or checkpointing |
| excluded | outside 1D validity (43 in executable routes) | 107 | – | never in production |
| deferred | super-fast stratum, non-executable fast runs | 602 | – | new numerics validated |

```
python research/campaign/waves.py write --wave 1 --out $SCRATCH/campaign/wave1.jsonl
python research/hpc/prepare.py $SCRATCH/campaign/wave1.jsonl --root $SCRATCH/campaign --speed-factor F
python research/hpc/make_slurm_arrays.py --root $SCRATCH/campaign --manifest $SCRATCH/campaign/wave1.jsonl ...
```

Use **one** `--root` for all waves. Results go to `<root>/runs/<run_id>/` and accepted runs are skipped, so a run
that occurs in two waves is computed once. For example, the V2 base variants are the production runs themselves.
Wave 3 and the C7 lines of wave 1b need `prepare.py --allow-long`. That fills a `400h` bin whose wall time
(25 days in `make_slurm_arrays.py`) must be checked against the cluster limit; if the limit is shorter, these runs
need orbit checkpointing first. `prepare.py` skips excluded runs in any manifest (`--exclusions ""` disables this).

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
