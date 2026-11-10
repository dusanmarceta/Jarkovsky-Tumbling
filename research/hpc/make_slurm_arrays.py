"""
Write Slurm array scripts for the task lists produced by prepare.py (one array per cost bin). Nothing is submitted
unless --submit is given. Partition, account, QoS and limits must come from the HPC centre's documentation.

    python research/hpc/make_slurm_arrays.py --root /scratch/.../B_sobol_slow --manifest .../B_sobol_slow.jsonl \
        --repo /home/.../Jarkovsky-Tumbling-research --venv /home/.../Jarkovsky-Tumbling-research/venv \
        --account PROJECT --partition PARTITION --max-array 1000 --throttle 200
"""
import argparse
import math
import subprocess
from pathlib import Path

WALL = {'1h': '02:00:00', '6h': '10:00:00', '24h': '1-12:00:00', '96h': '6-00:00:00'}   # ~1.5x margin; adjust to limits
MEM = {'1h': '2G', '6h': '2G', '24h': '3G', '96h': '4G'}

TEMPLATE = """#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={wall}
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem={mem}
#SBATCH --array=0-{last}%{throttle}
#SBATCH --output={root}/logs/%x_%A_%a.out
{extra}
export REPO={repo} VENV={venv} NUMBA_CACHE_DIR={root}/numba_cache TASK_INDEX_BASE=0
export SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID + {offset}))
bash {repo}/research/hpc/run_task.sh {manifest} {tasklist} {root}
"""

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('root', 'manifest', 'repo', 'venv', 'account', 'partition'):
        ap.add_argument(f'--{k}', required=True)
    ap.add_argument('--max-array', type=int, default=1000, help='MaxArraySize of the cluster')
    ap.add_argument('--throttle', type=int, default=200)
    ap.add_argument('--extra', default='', help='extra #SBATCH lines, e.g. "#SBATCH --qos=normal"')
    ap.add_argument('--submit', action='store_true')
    a = ap.parse_args()
    root = Path(a.root)
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    stem = Path(a.manifest).stem
    for b in WALL:
        tl = root / 'tasks' / f'{stem}_{b}.txt'
        if not tl.exists():
            continue
        n = sum(1 for _ in open(tl))
        for part in range(math.ceil(n / a.max_array)):
            off, cnt = part * a.max_array, min(a.max_array, n - part * a.max_array)
            script = root / 'tasks' / f'{stem}_{b}_part{part}.sbatch'
            script.write_text(TEMPLATE.format(name=f'{stem}_{b}_{part}', account=a.account, partition=a.partition,
                                              wall=WALL[b], mem=MEM[b], last=cnt - 1, throttle=a.throttle, root=root,
                                              extra=a.extra, repo=a.repo, venv=a.venv, offset=off,
                                              manifest=Path(a.manifest).resolve(), tasklist=tl))
            print(script, f'{cnt} tasks')
            if a.submit:
                subprocess.run(['sbatch', str(script)], check=True)
