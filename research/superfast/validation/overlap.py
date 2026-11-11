"""
Step 2 of the validation: the two-scale method against the full solver in the overlap regime, on campaign states
whose full-solver runs are affordable on one core. The lines are copied verbatim from the frozen manifests (same
run IDs), so the full-solver results are ordinary campaign runs.

    python -m superfast.validation.overlap select --out runs/superfast_validation/overlap.jsonl
    # full solver, one process per line (background):
    python -m tools.manifest run --manifest runs/superfast_validation/overlap.jsonl --index i --root runs/superfast_validation/raw/full
    python -m superfast.validation.overlap new --manifest ... --root runs/superfast_validation/raw/full --out .../new.json [--index i]
    python -m superfast.validation.overlap compare --manifest ... --root ... --new .../new.json
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

from campaign import waves as W  # noqa: E402
from tools.manifest import accepted, ensure_mesh, full_config, read_manifest  # noqa: E402

STATES = [
    ('sobol2102_ordinary_00327', 'SAM, P_phi 23 h, e 0.14, Gamma 397, D 22 m (small body)'),
    ('sobol2102_ordinary_00353', 'LAM, e 0.38, Gamma 1970, D 179 m'),
    ('sobol2102_ordinary_00419', 'SAM, e 0.71, Gamma 2228, D 36 m'),
    ('morris1102_ordinary_00049', 'LAM, e 0.80, Gamma 2500, D 697 m'),
    ('sobol2102_ordinary_00930', 'LAM, e 0.47, Gamma 498, D 117 m'),
    ('sobol2102_ordinary_00561', 'LAM, e 0.06, Gamma 24 (low), D 53 m'),
    ('sobol2101_slow_00729', 'SAM, P_phi 90 h, e 0.28, Gamma 1657, D 3.7 km (frozen-Sun limit)'),
    ('sobol4101_fast_00228', 'SAM, P_phi 1.5 h, e 0, Gamma 143, D 149 m (fast stratum)'),
]


def select(out):
    runs = W.unique_runs(W.load_frozen())
    lines = []
    for sid, note in STATES:
        for kind in ('tumbler', 'pa_P_par'):
            ln = next(l for l, _ in runs.values() if l['state_id'] == sid and l['kind'] == kind)
            lines.append(dict(ln, validation_note=note))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        for ln in lines:
            f.write(json.dumps(ln, default=float) + '\n')
    print(f'{len(lines)} lines -> {out}; full-solver cost estimate {sum(l["policy"]["cost_core_h_estimate"] for l in lines):.1f} core-h')


def run_new(manifest, root, out, indices=None, numerics=None):
    from superfast.twoscale import Numerics, TwoScaleYarkovsky
    lines = read_manifest(manifest)
    res = json.load(open(out)) if Path(out).exists() else {}
    for i, ln in enumerate(lines):
        if indices is not None and i not in indices:
            continue
        cfg = full_config(ln['config_overrides'])
        mesh_dir = ensure_mesh(ln, root)
        cfg['shape_model_directory'] = str(mesh_dir) + '/'
        num = Numerics(**(numerics or {}))
        t0, c0 = time.perf_counter(), time.process_time()
        m = TwoScaleYarkovsky(cfg, mesh_dir / cfg['shape_model_file'], num)
        r = m.run()
        res[ln['run_id']] = {'state_id': ln['state_id'], 'kind': ln['kind'], 'dadt_m_s': r['dadt_m_s'],
                             'iterations': [it['dadt_m_s'] for it in r['iterations']], 'coupling_change': r['coupling_change'],
                             'all_converged': r['iterations'][-1]['all_converged'], 'wall_s': time.perf_counter() - t0,
                             'cpu_s': time.process_time() - c0,
                             'fast_steps': r['total_fast_steps'], 'window': r['window'], 'n_fast_layers': r['n_fast_layers'],
                             'periods': r['periods'], 'seasonal': r['iterations'][-1].get('seasonal'),
                             'numerics': r['numerics']}
        print(json.dumps({k: res[ln['run_id']][k] for k in ('state_id', 'kind', 'dadt_m_s', 'iterations', 'wall_s', 'fast_steps')}), flush=True)
        json.dump(res, open(out, 'w'), indent=1, default=float)
    return res


def compare(manifest, root, new):
    lines = read_manifest(manifest)
    res = json.load(open(new))
    rows = []
    for ln in lines:
        p = Path(root) / 'runs' / ln['run_id'] / 'provenance.json'
        full = json.load(open(p)) if p.exists() else None
        n = res.get(ln['run_id'])
        row = {'state_id': ln['state_id'], 'kind': ln['kind'], 'note': ln.get('validation_note', '')}
        if full:
            r = full.get('result') or {}
            row.update(full_dadt=r.get('drift_m_s'), full_accepted=accepted(full), full_wall_s=full.get('wall_time_s'),
                       full_orbits=r.get('orbits'), full_error=r.get('drift_error_m_s'))
        if n:
            row.update(new_dadt=n['dadt_m_s'], new_wall_s=n['wall_s'], coupling_change=n['coupling_change'])
        if full and n and row.get('full_dadt'):
            row['rel_diff'] = n['dadt_m_s'] / row['full_dadt'] - 1
            row['speedup'] = row['full_wall_s'] / n['wall_s'] if row.get('full_wall_s') else None
        rows.append(row)
    # tumbling effect Delta = tumbler - twin
    by = {}
    for r in rows:
        by.setdefault(r['state_id'], {})[r['kind']] = r
    for sid, d in by.items():
        t, p = d.get('tumbler', {}), d.get('pa_P_par', {})
        if all(x.get(k) is not None for x in (t, p) for k in ('full_dadt', 'new_dadt')):
            t['delta_full'] = t['full_dadt'] - p['full_dadt']
            t['delta_new'] = t['new_dadt'] - p['new_dadt']
    print(json.dumps(rows, indent=1, default=float))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select')
    s.add_argument('--out', required=True)
    n = sub.add_parser('new')
    n.add_argument('--manifest', required=True)
    n.add_argument('--root', required=True)
    n.add_argument('--out', required=True)
    n.add_argument('--index', type=int, nargs='*')
    n.add_argument('--numerics', default=None, help='JSON dict of Numerics overrides')
    c = sub.add_parser('compare')
    c.add_argument('--manifest', required=True)
    c.add_argument('--root', required=True)
    c.add_argument('--new', required=True)
    a = ap.parse_args()
    if a.cmd == 'select':
        select(a.out)
    elif a.cmd == 'new':
        run_new(a.manifest, a.root, a.out, a.index, json.loads(a.numerics) if a.numerics else None)
    else:
        compare(a.manifest, a.root, a.new)
