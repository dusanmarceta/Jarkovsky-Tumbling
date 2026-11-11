"""
Prepare a manifest for distributed execution (run once, on a login or a single compute node, before any array job).

  1. writes every distinct ellipsoid mesh to <root>/meshes (atomic writes)
  2. computes the solver's visible-facet cache for each mesh (the solver writes it into the repository on first use;
     doing it here prevents concurrent jobs from racing on the same cache file)
  3. warms the Numba cache (NUMBA_CACHE_DIR) with one short run, so array tasks do not compile concurrently
  4. writes task lists by estimated cost bin: <root>/tasks/<manifest>_<bin>.txt (manifest line numbers), skipping
     lines whose route is not executable with the current solver and runs on the exclusion list
     (research/campaign/exclusions/outside_1d_v1.json), and a JSON summary of the bins

    python research/hpc/prepare.py $SCRATCH/campaign/wave1.jsonl --root $SCRATCH/campaign

Wave manifests (research/campaign/waves.py) carry an 'hpc.route' that replaces the manifest route. Lines with
hpc.route = 'full_long' (96-400 core-h: the preserved low-Gamma runs and V2 corner C7) are only binned with
--allow-long, into the '400h' bin, which needs a long-wall-time queue.

Deferred wave, two-scale method (research/superfast, branch research/superfast-method): write the wave with
`python -m campaign.waves write --wave deferred --twoscale --out deferred_twoscale.jsonl`, estimate the costs with
`python -m superfast.cost --manifest deferred_twoscale.jsonl --root R --out twoscale_costs.json`, then

    python research/hpc/prepare.py deferred_twoscale.jsonl --root R --twoscale-costs twoscale_costs.json

The task lists are the same format; submit them with make_slurm_arrays.py --runner run_task_twoscale.sh.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
from tools.manifest import ensure_mesh, read_manifest  # noqa: E402
from tools.runner import REPO  # noqa: E402

BINS = [('1h', 1.0), ('6h', 6.0), ('24h', 24.0), ('96h', 96.0)]      # estimated core-hours per run (Apple M4 cost model)
LONG_BIN = ('400h', 400.0)                                             # only with --allow-long
EXECUTABLE_ROUTES = {'full', 'full_costly'}
EXCLUSIONS = RESEARCH / 'campaign' / 'exclusions' / 'outside_1d_v1.json'


def line_cost(ln, twoscale_costs=None):
    """Estimated M4 core-hours: the full solver's cost model, or the two-scale estimate for hpc.route = 'twoscale'."""
    if (ln.get('hpc') or {}).get('route') == 'twoscale':
        return twoscale_costs[ln['run_id']]['twoscale_core_h_estimate']
    return ln['policy']['cost_core_h_estimate']


def task_bins(lines, speed_factor=1.0, allow_long=False, excluded=frozenset(), twoscale_costs=None):
    """Manifest line numbers per cost bin, and the counts of skipped lines by reason."""
    bins_def = BINS + ([LONG_BIN] if allow_long else [])
    bins = {b: [] for b, _ in bins_def}
    skipped = {}
    for i, ln in enumerate(lines):
        if ln['kind'] == 'rejected':
            reason = 'rejected'
        elif ln['run_id'] in excluded:
            reason = 'excluded'
        else:
            route = (ln.get('hpc') or {}).get('route') or ln['policy']['route']
            reason = None if (route in EXECUTABLE_ROUTES or (route == 'full_long' and allow_long)
                              or (route == 'twoscale' and twoscale_costs is not None and ln['run_id'] in twoscale_costs)) else route
        if reason is None:
            c = line_cost(ln, twoscale_costs) * speed_factor
            for b, lim in bins_def:
                if c <= lim:
                    bins[b].append(i)
                    break
            else:
                reason = f'over_{int(bins_def[-1][1])}h'
        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
    return bins, skipped


def warm_visible_facets(mesh_path):
    sys.path.insert(0, str(REPO))
    import numpy as np
    from src.utilities.config import Config
    from src.model.view_factors import calculate_and_cache_visible_facets
    from tempest import read_shape_model
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        f.write(f'shape_model_file: "{mesh_path.name}"\nshape_model_directory: "{mesh_path.parent}/"\nsilent_mode: true\nn_jobs: 1\n')
    cfg = Config(f.name)
    os.unlink(f.name)
    facets = read_shape_model(str(mesh_path), 1, 2, 1, False)
    pos = np.array([x.position for x in facets]); nrm = np.array([x.normal for x in facets]); ver = np.array([x.vertices for x in facets])
    calculate_and_cache_visible_facets(True, facets, pos, nrm, ver, cfg)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest')
    ap.add_argument('--root', required=True)
    ap.add_argument('--speed-factor', type=float, default=1.0, help='HPC core time / Apple M4 core time (measure!)')
    ap.add_argument('--skip-warm-run', action='store_true')
    ap.add_argument('--allow-long', action='store_true', help="bin hpc.route = 'full_long' lines (up to 400 core-h)")
    ap.add_argument('--exclusions', default=str(EXCLUSIONS), help='JSON list of excluded runs ("" = none)')
    ap.add_argument('--twoscale-costs', default=None, help="superfast.cost output: bin hpc.route = 'twoscale' lines")
    a = ap.parse_args()
    tsc = json.load(open(a.twoscale_costs))['runs'] if a.twoscale_costs else None
    excluded = {r['run_id'] for r in json.load(open(a.exclusions))['runs']} if a.exclusions else set()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    lines = read_manifest(a.manifest)
    seen = set()
    for ln in lines:
        if ln['kind'] == 'rejected':
            continue
        name = ln['config_overrides']['shape_model_file']
        if name not in seen:
            seen.add(name)
            mesh_dir = ensure_mesh(ln, root)
            if tsc is None:                      # the two-scale method needs no visible-facet cache (convex bodies)
                warm_visible_facets(mesh_dir / name)
    print(f'{len(seen)} meshes written and visible-facet caches computed')
    bins, skipped = task_bins(lines, a.speed_factor, a.allow_long, excluded, tsc)
    (root / 'tasks').mkdir(exist_ok=True)
    stem = Path(a.manifest).stem
    for b, idx in bins.items():
        (root / 'tasks' / f'{stem}_{b}.txt').write_text(''.join(f'{i}\n' for i in idx))
    summary = {'manifest': str(Path(a.manifest).resolve()), 'root': str(root.resolve()),
               'bins': {b: len(v) for b, v in bins.items()}, 'skipped': skipped,
               'est_core_h': sum(line_cost(lines[i], tsc) * a.speed_factor for v in bins.values() for i in v)}
    json.dump(summary, open(root / 'tasks' / f'{stem}_summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    if not a.skip_warm_run and tsc is not None:
        # compile and cache the two-scale kernels once (one frozen position, one short window)
        os.environ.setdefault('NUMBA_CACHE_DIR', str(root / 'numba_cache'))
        from superfast.cost import setup_sizes  # noqa: F401  (imports the kernels)
        from superfast.twoscale import Numerics, TwoScaleYarkovsky
        from tools.manifest import full_config
        first = next(ln for ln in lines if ln['kind'] != 'rejected')
        cfg = full_config(first['config_overrides'])
        cfg['shape_model_directory'] = str((root / 'meshes').resolve()) + '/'
        m = TwoScaleYarkovsky(cfg, root / 'meshes' / cfg['shape_model_file'],
                              Numerics(n_loc=8, coupling_iterations=1, max_windows=1, seasonal_min_orbits=1, seasonal_max_orbits=1))
        m.run(locations=[0])
        print('two-scale kernels compiled')
    elif not a.skip_warm_run:
        # one short run compiles and caches all Numba functions in NUMBA_CACHE_DIR
        os.environ.setdefault('NUMBA_CACHE_DIR', str(root / 'numba_cache'))
        from tools.runner import run_case
        from tools.manifest import full_config
        first = next(ln for ln in lines if ln['kind'] != 'rejected')
        cfg = full_config(first['config_overrides'])
        cfg.update(shape_model_directory=str(root / 'meshes'), a_au=0.3, min_orbits=1, max_orbits=1)
        rec = run_case(cfg, root / 'warmup', 1)
        print('warm-up run exit code', rec['exit_code'], 'wall', rec['wall_time_s'], 's')
