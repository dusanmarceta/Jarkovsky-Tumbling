'''
Run Tempestovsky for the rows of an inputs CSV (make_inputs.py), one single-threaded process per run.

    python run_batch.py inputs_sim.csv --out runs_sim --chunk 0 --n-chunks 100 --workers 1
    python run_batch.py inputs_sim.csv --out runs_sim --workers 6          # all rows on one machine
    python run_batch.py inputs_sim.csv.gz --out runs_sim --last 10000 ...   # only runs 0..9999 (rows are random)

Rows are split round-robin over n-chunks (e.g. one chunk per job of a job array, or per processor).
A run whose output file already exists is skipped, so an interrupted batch can simply be restarted.
Run from the repository root (the shape models and config template are read with relative paths).
'''
import argparse, csv, gzip, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

TEMPLATE = 'data/config/config_apophis.yaml'
SPIN_KEYS = ('I1', 'I2', 'E_ratio', 'P_phi_h', 'psi_0', 'phi_0', 'lambda_L', 'beta_L')
THERMAL_KEYS = ('thermal_inertia', 'albedo', 'a_au', 'ecc')
# Batch defaults (override with --set): 100 steps per rotation changes the drift by < 1e-4 (relative) against 200;
# for a tumbling body the drift of single orbits scatters by 0.2-0.4 % (the rotation phases differ from orbit to
# orbit), so a 1e-3 tolerance is met only by chance and can take 10-20 orbits; 0.5 % stops after the thermal transient.
DEFAULTS = {'implicit_timesteps_per_day': '100', 'yarkovsky_tolerance': '0.005', 'max_orbits': '10'}


def make_config(row, out_dir, extra):
    run = int(row['run'])
    overrides = {k: row[k] for k in SPIN_KEYS + THERMAL_KEYS}
    overrides.update({'n_jobs': '1', 'silent_mode': 'true',
                      'output_file': f'"{out_dir}/run_{run:05d}.txt"',
                      'progress_file': f'"{out_dir}/progress/run_{run:05d}.txt"'})
    overrides.update(DEFAULTS)
    overrides.update(extra)
    lines = open(TEMPLATE).read().splitlines()
    seen = set()
    for i, line in enumerate(lines):
        m = re.match(r'^(\w+):', line)
        if m and m.group(1) in overrides:
            lines[i] = f'{m.group(1)}: {overrides[m.group(1)]}'
            seen.add(m.group(1))
    lines += [f'{k}: {v}' for k, v in overrides.items() if k not in seen]
    path = f'{out_dir}/configs/run_{run:05d}.yaml'
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def run_one(row, out_dir, extra):
    run = int(row['run'])
    if os.path.exists(f'{out_dir}/run_{run:05d}.txt'):
        return run, 'skipped'
    cfg = make_config(row, out_dir, extra)
    env = dict(os.environ, NUMBA_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with open(f'{out_dir}/logs/run_{run:05d}.log', 'w') as log:
        r = subprocess.run([sys.executable, 'tempest.py', '--config', cfg], stdout=log, stderr=subprocess.STDOUT, env=env)
    return run, 'ok' if r.returncode == 0 and os.path.exists(f'{out_dir}/run_{run:05d}.txt') else f'failed ({r.returncode})'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--n-chunks', type=int, default=1)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--first', type=int, default=0, help='only rows with run >= first')
    ap.add_argument('--last', type=int, default=None, help='only rows with run < last')
    ap.add_argument('--set', action='append', default=[], help='extra config override, e.g. --set implicit_timesteps_per_day=200')
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out)
    for sub in ('', 'configs', 'logs', 'progress'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    extra = dict(s.split('=', 1) for s in a.set)
    opener = gzip.open if a.inputs.endswith('.gz') else open
    rows = [r for r in csv.DictReader(opener(a.inputs, 'rt'))
            if int(r['run']) % a.n_chunks == a.chunk and int(r['run']) >= a.first and (a.last is None or int(r['run']) < a.last)]
    print(f'{len(rows)} runs in chunk {a.chunk}/{a.n_chunks}, {a.workers} workers, output in {out_dir}', flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for run, status in ex.map(lambda r: run_one(r, out_dir, extra), rows):
            print(f'run {run:05d}: {status}', flush=True)
