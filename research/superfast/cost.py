"""
Cost model of the two-scale method, for binning scheduler tasks (research/hpc/prepare.py --twoscale).

    cpu_s = C_CELL * n_loc * coupling_iterations * WINDOWS * window_steps * n_facets * n_fast_layers + C_RUN

window_steps, n_fast_layers and n_loc come from the model set-up for the line (no time stepping); C_CELL (core
seconds per column-layer update), WINDOWS (mean averaging windows per frozen position and coupling iteration) and
C_RUN (set-up, torus table, seasonal problem) are calibrated on the validation runs (runs/superfast_validation)
on an Apple M4 core, like the full solver's cost model.

    python -m superfast.cost --manifest deferred_twoscale.jsonl --out estimates.json
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
for p in (str(RESEARCH), str(RESEARCH.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

C_CELL = 7.0e-9      # core s per column-layer step (validation runs: median 6.6e-9 for runs above 1e10 steps)
WINDOWS = 6.0        # windows per frozen position and coupling iteration (validation runs: median 5, tumblers 4,
                     # super-fast tumblers up to 10)
C_RUN = 60.0         # core s per run outside the fast columns (set-up, torus table, seasonal problem)
# coupling: 3 passes minimum (Numerics.coupling_iterations), up to 6 when the drift still changes by > 1e-3


def setup_sizes(line, root, numerics=None):
    from superfast.twoscale import Numerics, TwoScaleYarkovsky
    from tools.manifest import ensure_mesh, full_config
    cfg = full_config(line['config_overrides'])
    mesh_dir = ensure_mesh(line, root)
    cfg['shape_model_directory'] = str(Path(mesh_dir).resolve()) + '/'
    with contextlib.redirect_stdout(io.StringIO()):
        m = TwoScaleYarkovsky(cfg, Path(cfg['shape_model_directory']) / cfg['shape_model_file'], Numerics(**(numerics or {})))
    return {'n_loc': m.num.n_loc, 'iterations': m.num.coupling_iterations, 'window_steps': m.window['steps'],
            'window_periods': m.window['periods'], 'n_facets': int(m.normals.shape[0]), 'n_fast_layers': m.n_fast,
            'dt_s': m.dt, 'dz_m': m.dz, 'periods_s': m.periods}


def estimate_core_h(sizes, c_cell=C_CELL, windows=WINDOWS, c_run=C_RUN):
    cells = sizes['n_loc'] * sizes['iterations'] * windows * sizes['window_steps'] * sizes['n_facets'] * sizes['n_fast_layers']
    return (c_cell * cells + c_run) / 3600.0


if __name__ == '__main__':
    from tools.manifest import read_manifest
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--root', required=True, help='where meshes are written (<root>/meshes)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--from-sizes', default=None, help='recompute the estimates from an earlier output (no set-up)')
    a = ap.parse_args()
    if a.from_sizes:
        prev = json.load(open(a.from_sizes))['runs']
        out = {rid: dict(r, twoscale_core_h_estimate=estimate_core_h(r)) for rid, r in prev.items()}
        json.dump({'model': {'C_CELL': C_CELL, 'WINDOWS': WINDOWS, 'C_RUN': C_RUN}, 'runs': out},
                  open(a.out, 'w'), indent=1, default=float)
        sys.exit(0)
    out = {}
    for ln in read_manifest(a.manifest):
        s = setup_sizes(ln, Path(a.root).resolve())
        out[ln['run_id']] = dict(s, twoscale_core_h_estimate=estimate_core_h(s),
                                 full_core_h_estimate=ln['policy']['cost_core_h_estimate'])
        print(ln['run_id'], f"{out[ln['run_id']]['twoscale_core_h_estimate']:.3g} core-h "
              f"(full solver {ln['policy']['cost_core_h_estimate']:.3g})", flush=True)
    json.dump({'model': {'C_CELL': C_CELL, 'WINDOWS': WINDOWS, 'C_RUN': C_RUN}, 'runs': out},
              open(a.out, 'w'), indent=1, default=float)
