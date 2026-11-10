"""
Build campaign manifests (JSON Lines) for the 12-input study. No simulation is run here.

    python research/campaign/build_campaign.py design --stratum slow --design morris --r 20 --levels 8 --seed 101 \
        --twins P_par --out research/campaign/manifests/A_morris_slow.jsonl
    python research/campaign/build_campaign.py design --stratum ordinary --design sobol --n 1024 --seed 202 \
        --twins P_par --out research/campaign/manifests/B_sobol_ordinary.jsonl
    python research/campaign/build_campaign.py phase --source research/campaign/manifests/B_sobol_slow.jsonl \
        --n-states 48 --replicates 8 --seed 303 --out research/campaign/manifests/D_phase_slow.jsonl
    python research/campaign/build_campaign.py summary research/campaign/manifests/*.jsonl

Each line: run_id, state_id, kind (tumbler / pa_P_par / pa_P_phi / pa_P_lc), stratum, unit coordinates, the 12
inputs, derived rotational quantities, thermal/validity quantities, policy (averaging mode, numerics, route, cost
estimate, flags), mesh specification and the complete solver configuration overrides. A flat CSV with the same
content (one row per run) is written next to each manifest for statistical analysis.
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
from tools.campaign import STRATA, design_morris, design_saltelli, design_sobol, lines_for_unit  # noqa: E402


def _job(args):
    x, stratum, sid, twins, replicate = args
    try:
        return lines_for_unit(np.asarray(x), stratum, sid, twins=twins, replicate=replicate)
    except Exception as exc:                         # record, never silently drop, an inadmissible point
        return [{'state_id': sid, 'stratum': stratum, 'unit': list(map(float, x)), 'kind': 'rejected',
                 'error': f'{type(exc).__name__}: {exc}'}]


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def write(lines, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        for ln in lines:
            f.write(json.dumps(ln, default=float) + '\n')
    rows = [flatten({k: v for k, v in ln.items() if k != 'config_overrides'}) for ln in lines]
    keys = sorted({k for r in rows for k in r})
    with open(out.with_suffix('.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_jobs(jobs, workers):
    lines = []
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(_job, jobs, chunksize=4):
            lines += res
    return lines


def summary(paths):
    for p in paths:
        L = [json.loads(l) for l in open(p)]
        runs = [l for l in L if l['kind'] != 'rejected']
        tum = [l for l in runs if l['kind'] == 'tumbler']
        by_route, flags, modes = {}, {}, {}
        cost = {}
        for l in runs:
            r = l['policy']['route']
            by_route[r] = by_route.get(r, 0) + 1
            cost[r] = cost.get(r, 0.0) + l['policy']['cost_core_h_estimate']
            for fl in l['policy']['flags']:
                flags[fl] = flags.get(fl, 0) + 1
        for l in tum:
            m = l['derived']['mode']
            modes[m] = modes.get(m, 0) + 1
        avg = sum(1 for l in tum if l['policy']['averaging'] == 'long')
        print(json.dumps({'manifest': str(p), 'states': len({l['state_id'] for l in L}), 'runs': len(runs),
                          'rejected': len(L) - len(runs), 'routes': by_route,
                          'core_h_by_route': {k: round(v, 1) for k, v in cost.items()}, 'flags': flags,
                          'modes': modes, 'long_averaging_tumblers': avg}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('design')
    d.add_argument('--stratum', choices=list(STRATA), required=True)
    d.add_argument('--design', choices=['sobol', 'morris', 'saltelli'], required=True)
    d.add_argument('--n', type=int, default=0)
    d.add_argument('--r', type=int, default=20)
    d.add_argument('--levels', type=int, default=8)
    d.add_argument('--seed', type=int, required=True)
    d.add_argument('--twins', nargs='*', default=['P_par'])
    d.add_argument('--out', required=True)
    d.add_argument('--workers', type=int, default=8)
    p = sub.add_parser('phase')
    p.add_argument('--source', required=True)
    p.add_argument('--n-states', type=int, required=True)
    p.add_argument('--replicates', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('summary')
    s.add_argument('paths', nargs='+')
    a = ap.parse_args()

    if a.cmd == 'design':
        if a.design == 'sobol':
            X = design_sobol(a.n, a.seed)
        elif a.design == 'morris':
            X = design_morris(a.r, a.levels, a.seed)
        else:
            X = design_saltelli(a.n, a.seed)
        tag = f'{a.design}{a.seed}_{a.stratum}'
        jobs = [(x, a.stratum, f'{tag}_{i:05d}', tuple(a.twins), None) for i, x in enumerate(X)]
        write(run_jobs(jobs, a.workers), a.out)
        summary([a.out])
    elif a.cmd == 'phase':
        # same physical state, new initial phases phi_0 (x6) and psi_0 (x7); averaging forced to 'long' later
        src = [json.loads(l) for l in open(a.source) if json.loads(l)['kind'] == 'tumbler']
        rng = np.random.default_rng(a.seed)
        pick = rng.choice(len(src), size=min(a.n_states, len(src)), replace=False)
        jobs = []
        for i in pick:
            base = np.array(src[i]['unit'])
            for rep in range(a.replicates):
                x = base.copy()
                if rep:
                    x[6], x[7] = rng.random(), rng.random()
                jobs.append((x, src[i]['stratum'], f"{src[i]['state_id']}_ph{rep}", ('P_par',) if rep == 0 else (), rep))
        lines = run_jobs(jobs, a.workers)
        for ln in lines:                       # fixed, long averaging for every replicate (finite-time study)
            if ln['kind'] == 'rejected':
                continue
            n = ln['numerics']['long_orbits']
            ln['policy']['averaging'] = 'long'
            ln['config_overrides'].update(min_orbits=n, max_orbits=n)
        write(lines, a.out)
        summary([a.out])
    else:
        summary(a.paths)
