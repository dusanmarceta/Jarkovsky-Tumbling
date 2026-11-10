"""
Machine-readable simulation manifests (JSON Lines, one simulation per line) and the per-job runner.

    # build a manifest from design points
    python -m tools.manifest build --design sobol --n 64 --seed 2027 --out runs/prod/manifest.jsonl
    # run line k of a manifest (e.g. k = $SLURM_ARRAY_TASK_ID), one single-threaded job per simulation
    python -m tools.manifest run --manifest runs/prod/manifest.jsonl --index 17
    # validate all finished runs, list the ones to retry
    python -m tools.manifest check --manifest runs/prod/manifest.jsonl

Each manifest line holds: run_id, state_id, kind (tumbler / pa / pa_P_par / pa_P_phi / pa_P_lc), the unit-cube
design coordinates, the physical inputs, every derived rotation quantity, the numerical settings and the complete
solver configuration overrides. Results go to <root>/runs/<run_id>/ (isolated working directory), so concurrent
jobs never share output files. A run is accepted when exit code = 0, valid = True, converged = True and
fallback steps = 0; otherwise the retry policy doubles the steps per rotation (up to 2 retries).
"""
import argparse
import json
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from tools.population import BOUNDS, build_state, from_unit, full_config, sample_unit  # noqa: E402
from tools.runner import native, run_case  # noqa: E402


def lines_for_state(state_id, unit, phys, mesh_dir, numerics=None, fixed=None, twins=('P_par', 'P_phi', 'P_lc')):
    st = build_state(phys, mesh_dir, numerics=numerics, fixed=fixed, twins=twins)
    out = []
    for rid, kind, cfg, rec in st['runs']:
        out.append(native({'run_id': rid, 'state_id': state_id, 'kind': kind, 'unit': list(unit) if unit is not None else None,
                           'physical': phys, 'derived': st['state']['tumbler_derived'], 'E_ratio': st['state']['E_ratio'],
                           'mesh': st['state']['mesh'], 'fixed': st['state']['fixed'], 'numerics': st['state']['numerics'],
                           'run_numerics': rec, 'config_overrides': cfg}))
    return out


def write_manifest(lines, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for ln in lines:
            f.write(json.dumps(ln, default=float) + '\n')


def read_manifest(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def accepted(rec):
    """exit 0, valid temperatures, no backward-Euler fallback, and convergence of the orbit-averaged drift.
    Long-averaging runs (fixed number of orbits, used where a rotational period is not << the orbital period)
    are accepted without the convergence flag: their orbit-to-orbit variation is the quantity being measured."""
    r = (rec or {}).get('result') or {}
    inp = (rec or {}).get('inputs') or {}
    long_avg = inp.get('policy', {}).get('averaging') == 'long'
    # physical bound (the solver's own 'valid' flag misses runaway temperatures, see research/CHANGELOG.md):
    # no facet can exceed the subsolar temperature at perihelion (no self-heating in the orbit loop)
    T_bound = inp.get('thermal', {}).get('T_subsolar_perihelion_K')
    T_ok = T_bound is None or (r.get('T_surface_max_K') is not None and r['T_surface_max_K'] <= 1.05 * T_bound)
    return (rec is not None and rec.get('exit_code') == 0 and bool(r.get('valid')) and r.get('fallback_steps', 0) == 0
            and T_ok and (bool(r.get('converged')) or long_avg))


def ensure_mesh(line, root):
    """Campaign lines carry a mesh specification; the STL is written once (atomically) under <root>/meshes."""
    import os
    from tools.ellipsoid import ellipsoid_mesh, write_ascii_stl
    spec = line['mesh_spec']
    mesh_dir = Path(root) / 'meshes'
    mesh_dir.mkdir(parents=True, exist_ok=True)
    path = mesh_dir / line['config_overrides']['shape_model_file']
    if not path.exists():
        tmp = path.with_suffix(f'.tmp{os.getpid()}')
        write_ascii_stl(tmp, ellipsoid_mesh(spec['b_over_a'], spec['c_over_a'], spec['D_eq_m'], spec['n_subdiv']))
        os.replace(tmp, path)
    return mesh_dir


def run_line(line, root, attempt_numerics=None):
    run_dir = Path(root) / 'runs' / line['run_id']
    prov = run_dir / 'provenance.json'
    if prov.exists() and accepted(json.load(open(prov))):
        return json.load(open(prov))
    cfg = full_config(line['config_overrides'])
    if cfg.get('shape_model_directory') == 'MESH_DIR':
        cfg['shape_model_directory'] = str(ensure_mesh(line, root))
    for attempt in range(3):
        if attempt:
            cfg['implicit_timesteps_per_day'] = int(cfg['implicit_timesteps_per_day'] * 2)
        rec = run_case(cfg, run_dir, 1, extra_provenance={k: line[k] for k in line if k != 'config_overrides'} | {'attempt': attempt})
        if accepted(rec):
            return rec
        r = rec.get('result') or {}
        if r.get('valid') and not r.get('converged') and r.get('fallback_steps', 0) == 0:
            return rec          # not a numerical failure: drift did not settle within max_orbits; keep and flag
    return rec


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--design', choices=['sobol', 'lhs'], default='sobol')
    b.add_argument('--n', type=int, required=True)
    b.add_argument('--seed', type=int, required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--no-twins', action='store_true')
    r = sub.add_parser('run')
    r.add_argument('--manifest', required=True)
    r.add_argument('--index', type=int, required=True)
    r.add_argument('--root', default=None, help='results root (default: the manifest directory)')
    c = sub.add_parser('check')
    c.add_argument('--manifest', required=True)
    a = ap.parse_args()
    if a.cmd == 'build':
        out = Path(a.out)
        U = sample_unit(a.n, a.seed, a.design)
        lines = []
        for i, u in enumerate(U):
            lines += lines_for_state(f'{a.design}{a.seed}_{i:05d}', u, from_unit(u), out.parent / 'meshes',
                                     twins=() if a.no_twins else ('P_par', 'P_phi', 'P_lc'))
        write_manifest(lines, out)
        print(f'{len(lines)} runs from {a.n} states -> {out} (bounds: {json.dumps(BOUNDS)})')
    elif a.cmd == 'run':
        lines = read_manifest(a.manifest)
        rec = run_line(lines[a.index], Path(a.root) if a.root else Path(a.manifest).parent)
        print(json.dumps({'run_id': lines[a.index]['run_id'], 'exit': rec.get('exit_code'), 'result': rec.get('result')}, default=float))
    else:
        lines = read_manifest(a.manifest)
        root = Path(a.manifest).parent
        todo = []
        for i, ln in enumerate(lines):
            p = root / 'runs' / ln['run_id'] / 'provenance.json'
            if not (p.exists() and accepted(json.load(open(p)))):
                todo.append(i)
        print(json.dumps({'total': len(lines), 'accepted': len(lines) - len(todo), 'to_retry_indices': todo}))


if __name__ == '__main__':
    main()
