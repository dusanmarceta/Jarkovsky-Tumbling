"""
Numerical convergence of the two-scale method for one manifest line: each numerical parameter is refined on its own
and the change of the orbit-averaged da/dt relative to the base setting is reported.

    python -m superfast.validation.convergence --manifest M.jsonl --index i --root R --out conv.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[2]
for p in (str(RESEARCH), str(RESEARCH.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from superfast.twoscale import Numerics, TwoScaleYarkovsky  # noqa: E402
from tools.manifest import ensure_mesh, full_config, read_manifest  # noqa: E402

VARIANTS = {
    'base': {},
    'n_loc_12': {'n_loc': 12},
    'n_loc_48': {'n_loc': 48},
    'fast_depths_12': {'fast_depths': 12.0},
    'dt_half': {'_dt_factor': 0.5},
    'dz_half': {'_dz_factor': 0.5},
    'tol_tight': {'tol_rel': 5e-5, 'tol_energy': 5e-5},
    'seasonal_steps_x2': {'seasonal_steps': 8000},
    'seasonal_dz_half': {'seasonal_dz_over_ls': 0.025},
    'coupling_4': {'coupling_iterations': 4},
}


def run(line, root, base, variants):
    cfg = full_config(line['config_overrides'])
    mesh_dir = ensure_mesh(line, root)
    cfg['shape_model_directory'] = str(Path(mesh_dir).resolve()) + '/'
    mesh = Path(cfg['shape_model_directory']) / cfg['shape_model_file']
    probe = TwoScaleYarkovsky(cfg, mesh, Numerics(**base))
    out = {}
    for name in variants:
        v = dict(VARIANTS[name])
        num = dict(base)
        if '_dt_factor' in v:
            num['dt_s'] = probe.dt * v.pop('_dt_factor')
        if '_dz_factor' in v:
            num['dz_m'] = probe.dz * v.pop('_dz_factor')
        num.update(v)
        t0, c0 = time.perf_counter(), time.process_time()
        r = TwoScaleYarkovsky(cfg, mesh, Numerics(**num)).run()
        out[name] = {'dadt_m_s': r['dadt_m_s'], 'iterations': [it['dadt_m_s'] for it in r['iterations']],
                     'all_converged': r['iterations'][-1]['all_converged'], 'wall_s': time.perf_counter() - t0,
                     'cpu_s': time.process_time() - c0, 'fast_steps': r['total_fast_steps'], 'n_fast_layers': r['n_fast_layers']}
        b = out.get('base', out[name])['dadt_m_s']
        print(f"{name:20s} da/dt {r['dadt_m_s']:+.6e}  rel to base {r['dadt_m_s'] / b - 1:+.2e}  "
              f"cpu {out[name]['cpu_s']:.0f} s  converged {out[name]['all_converged']}", flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default='{"coupling_iterations": 3}')
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    a = ap.parse_args()
    line = read_manifest(a.manifest)[a.index]
    res = run(line, Path(a.root).resolve(), json.loads(a.base), a.variants)
    prev = json.load(open(a.out)) if Path(a.out).exists() else {}
    prev[line['run_id']] = {'state_id': line['state_id'], 'kind': line['kind'], 'variants': res}
    json.dump(prev, open(a.out, 'w'), indent=1, default=float)
