"""
Run the two-scale method for one manifest line (frozen campaign manifest, wave manifest or any other manifest built
by the campaign tools), as one scheduler task. Results go to <root>/runs_twoscale/<run_id>/result.json, next to
the full solver's <root>/runs/<run_id>/, so the two methods never overwrite each other.

    python -m superfast.run --manifest M.jsonl --index 17 --root $SCRATCH/campaign [--numerics '{"n_loc": 48}']

A result is accepted when every frozen position converged, the coupling iterations changed the orbit-averaged drift
by less than coupling_tol_rel |da/dt| + coupling_tol_frac x (drift scale), and the drift is finite.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
for p in (str(RESEARCH), str(RESEARCH.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from superfast.twoscale import Numerics, TwoScaleYarkovsky  # noqa: E402
from tools.manifest import ensure_mesh, full_config, read_manifest  # noqa: E402

COUPLING_TOL_REL = 2e-3


def git_commit():
    try:
        return subprocess.run(['git', '-C', str(RESEARCH), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def accepted(res):
    r = (res or {}).get('result') or {}
    d = r.get('dadt_m_s')
    if d is None or not np.isfinite(d) or not r.get('all_converged'):
        return False
    ch = r.get('coupling_change')
    return ch is None or ch <= COUPLING_TOL_REL * abs(d) + 1e-4 * r.get('drift_scale_m_s', 0.0)


def run_line(line, root, numerics=None):
    out_dir = Path(root) / 'runs_twoscale' / line['run_id']
    out = out_dir / 'result.json'
    if out.exists() and accepted(json.load(open(out))):
        return json.load(open(out))
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = full_config(line['config_overrides'])
    mesh_dir = ensure_mesh(line, root)
    cfg['shape_model_directory'] = str(Path(mesh_dir).resolve()) + '/'
    num = Numerics(**(numerics or {}))
    t0, c0 = time.perf_counter(), time.process_time()
    status, err, r = 'ok', None, None
    try:
        model = TwoScaleYarkovsky(cfg, Path(cfg['shape_model_directory']) / cfg['shape_model_file'], num)
        r = model.run()
    except Exception as e:                       # recorded, not raised: the task list goes on
        status, err = 'failed', repr(e)
    res = {'run_id': line['run_id'], 'state_id': line.get('state_id'), 'kind': line.get('kind'),
           'stratum': line.get('stratum'), 'status': status, 'error': err, 'method': 'twoscale-v2',
           'wall_time_s': time.perf_counter() - t0, 'cpu_time_s': time.process_time() - c0,
           'git_commit': git_commit(), 'python': platform.python_version(), 'host': platform.node(),
           'numerics': asdict(num), 'inputs': {k: line[k] for k in ('inputs', 'derived', 'policy') if k in line}}
    if r is not None:
        res['numerics'] = r['numerics']          # as resolved for this run (n_loc = 0 -> the eccentricity rule)
        last = r['iterations'][-1]
        scale = float(np.mean([p.get('scale', 0.0) for p in last['per_location']])) if last['per_location'] else 0.0
        res['result'] = {'dadt_m_s': r['dadt_m_s'], 'iterations_dadt_m_s': [it['dadt_m_s'] for it in r['iterations']],
                         'coupling_change': r['coupling_change'], 'all_converged': last['all_converged'],
                         'fallback_steps': last['fallbacks'], 'drift_scale_m_s': scale, 'window': r['window'],
                         'periods_s': r['periods'], 'n_fast_layers': r['n_fast_layers'], 'dt_s': r['dt_s'],
                         'dz_m': r['dz_m'], 'mass_kg': r['mass_kg'], 'total_fast_steps': r['total_fast_steps'],
                         'seasonal': last.get('seasonal'), 'per_location': last['per_location']}
        res['accepted'] = accepted(res)
    json.dump(res, open(out, 'w'), indent=1, default=float)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--numerics', default=None, help='JSON dict of Numerics overrides')
    a = ap.parse_args()
    line = read_manifest(a.manifest)[a.index]
    res = run_line(line, Path(a.root).resolve(), json.loads(a.numerics) if a.numerics else None)
    print(json.dumps({'run_id': res['run_id'], 'status': res['status'], 'accepted': res.get('accepted'),
                      'dadt_m_s': (res.get('result') or {}).get('dadt_m_s'), 'wall_s': res['wall_time_s']}))
