"""
Execution waves for the frozen campaign (tag hpc-freeze-v1) and its supplements. No manifest is changed: the
frozen .jsonl.gz files are read, checked against their SHA-256 checksums, and a selection is written elsewhere.

Every unique frozen run belongs to exactly one wave (the first rule that matches):
  excluded  listed in exclusions/outside_1d_v1.json: the finite-body factor of the linear theory deviates by
            more than 5 %, so the 1D conduction model is not valid
  0         convergence batch V, executable routes
  3         preserved low-Gamma runs of the ordinary stratum: 'multiscale_required' only because their cost
            exceeds the 96 core-h route limit (cost <= LONG_MAX_CORE_H); HPC route 'full_long'. Includes the
            8 such V runs. Gate: V2 corner C7 (and C1-C3 for the 53 runs with e > 0.6)
  1         slow and ordinary strata, e <= 0.6, executable routes. Gate: V
  2         executable routes with e > 0.6 or in the fast stratum. Gate: V2 corners C1-C5
  deferred  everything else (super-fast stratum, non-executable fast runs): waits for new numerics
Supplementary manifests (manifests_v2/) form their own waves:
  1b        V2 convergence corners (executable routes and 'full_long'); runs next to wave 1

    python research/campaign/waves.py summary [--write]       # counts and core-h per wave (-> campaign/WAVES.json)
    python research/campaign/waves.py exclusions               # rewrite exclusions/outside_1d_v1.json
    python research/campaign/waves.py write --wave 1 --out $SCRATCH/campaign/wave1.jsonl

A wave manifest holds each selected run once (deduplicated by run_id): the original manifest line plus an 'hpc'
block (wave, route, source manifest lines). prepare.py, run_task.sh and aggregate.py treat it like any manifest.
Use ONE results root for all waves, so a run that occurs in two waves (V2 base variants are production runs) is
computed only once.
"""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parent
FROZEN_DIR = CAMPAIGN / 'manifests'
SUPPLEMENT_DIR = CAMPAIGN / 'manifests_v2'
EXCLUSIONS = CAMPAIGN / 'exclusions' / 'outside_1d_v1.json'
SUMMARY = CAMPAIGN / 'WAVES.json'
EXECUTABLE = {'full', 'full_costly'}
LONG_MAX_CORE_H = 400.0          # 'full_long': 96 < cost <= 400 core-h (Apple M4); needs long wall time or checkpoints
FROZEN_WAVES = ('excluded', '0', '1', '2', '3', 'deferred')
SUPPLEMENT_WAVES = {'V2_convergence': '1b'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(directory, checksum_file):
    """{manifest stem: [lines]} for every manifest listed in checksum_file, after verifying each checksum."""
    out = {}
    for e in json.load(open(directory / checksum_file)):
        path = directory / e['file']
        if sha256(path) != e['sha256']:
            raise RuntimeError(f'{path} does not match its checksum in {checksum_file}')
        out[e['file'][:-len('.jsonl.gz')]] = [json.loads(l) for l in gzip.open(path, 'rt') if l.strip()]
    return out


def load_frozen():
    return load(FROZEN_DIR, 'FREEZE_checksums.json')


def load_supplements():
    return load(SUPPLEMENT_DIR, 'CHECKSUMS.json') if (SUPPLEMENT_DIR / 'CHECKSUMS.json').exists() else {}


def unique_runs(manifests):
    """{run_id: (first line, ['manifest:line', ...])}"""
    runs = {}
    for name, lines in manifests.items():
        for i, ln in enumerate(lines):
            if ln['kind'] == 'rejected':
                continue
            runs.setdefault(ln['run_id'], (ln, []))[1].append(f'{name}:{i}')
    return runs


def outside_1d(ln):
    return 'outside_1d_validity' in ln['policy']['flags']


def build_exclusions(frozen):
    runs = unique_runs(frozen)
    rows = []
    for rid, (ln, src) in sorted(runs.items()):
        if outside_1d(ln):
            rows.append({'run_id': rid, 'kind': ln['kind'], 'state_id': ln['state_id'], 'stratum': ln['stratum'],
                         'route': ln['policy']['route'], 'D_eq_m': ln['inputs']['D_eq_m'],
                         'thermal_inertia': ln['inputs']['thermal_inertia'],
                         'finite_body_factor_seasonal': ln['thermal']['finite_body_factor_seasonal'],
                         'finite_body_factor_diurnal': ln['thermal']['finite_body_factor_diurnal'], 'sources': src})
    by_route = {}
    for r in rows:
        by_route[r['route']] = by_route.get(r['route'], 0) + 1
    return {'name': 'outside_1d_v1', 'source_tag': 'hpc-freeze-v1',
            'rule': "policy.flags contains 'outside_1d_validity': the finite-sphere factor F(R/l, Theta)/F(inf, Theta) "
                    "of the linear theory deviates from 1 by more than 5 % at the seasonal or the diurnal period",
            'counts': {'runs': len(rows), 'in_executable_routes': sum(r['route'] in EXECUTABLE for r in rows),
                       'by_route': by_route},
            'manifest_sha256': {e['file']: e['sha256'] for e in json.load(open(FROZEN_DIR / 'FREEZE_checksums.json'))},
            'runs': rows}


def load_exclusions(path=EXCLUSIONS):
    return {r['run_id'] for r in json.load(open(path))['runs']} if Path(path).exists() else set()


def hpc_route(ln):
    route, cost = ln['policy']['route'], ln['policy']['cost_core_h_estimate']
    if route in EXECUTABLE:
        return route
    if route == 'multiscale_required' and cost <= LONG_MAX_CORE_H and ln['stratum'] != 'superfast':
        return 'full_long'
    return None


def frozen_wave(ln, in_v, excluded):
    """Wave of one frozen run (rules in the module docstring, applied in order)."""
    route = ln['policy']['route']
    if ln['run_id'] in excluded:
        return 'excluded'
    if in_v and route in EXECUTABLE:
        return '0'
    if (route == 'multiscale_required' and ln['stratum'] == 'ordinary'
            and ln['policy']['cost_core_h_estimate'] <= LONG_MAX_CORE_H):
        return '3'
    if route in EXECUTABLE and ln['stratum'] in ('slow', 'ordinary') and ln['inputs']['ecc'] <= 0.6:
        return '1'
    if route in EXECUTABLE and (ln['inputs']['ecc'] > 0.6 or ln['stratum'] == 'fast'):
        return '2'
    return 'deferred'


def assign(frozen=None, supplements=None, excluded=None):
    """{wave: {run_id: (line, sources, hpc route)}} for the frozen runs and the supplementary waves."""
    frozen = load_frozen() if frozen is None else frozen
    supplements = load_supplements() if supplements is None else supplements
    excluded = load_exclusions() if excluded is None else excluded
    v_ids = {ln['run_id'] for ln in frozen['V_convergence']}
    waves = {w: {} for w in FROZEN_WAVES}
    for rid, (ln, src) in unique_runs(frozen).items():
        waves[frozen_wave(ln, rid in v_ids, excluded)][rid] = (ln, src, hpc_route(ln))
    for name, lines in supplements.items():
        w = SUPPLEMENT_WAVES[name]
        waves.setdefault(w, {})
        for rid, (ln, src) in unique_runs({name: lines}).items():
            route = hpc_route(ln)
            if route is not None and rid not in excluded and not outside_1d(ln):
                waves[w][rid] = (ln, src, route)
    return waves


def summarise(waves):
    out = {}
    for w, runs in waves.items():
        ids = sorted(runs)
        routes = {}
        for _, _, r in runs.values():
            routes[str(r)] = routes.get(str(r), 0) + 1
        out[w] = {'runs': len(ids), 'core_h_m4': round(sum(ln['policy']['cost_core_h_estimate'] for ln, _, _ in runs.values()), 1),
                  'hpc_routes': routes, 'run_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
    return out


def write_wave(waves, wave, out, route_override=None):
    """route_override = 'twoscale' writes the deferred wave for the two-scale method (research/superfast); the frozen
    lines are copied unchanged, only the 'hpc' block says how they are run."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = waves[wave]
    with open(out, 'w') as f:
        for rid in sorted(runs, key=lambda r: (runs[r][1][0], r)):
            ln, src, route = runs[rid]
            f.write(json.dumps(dict(ln, hpc={'wave': wave, 'route': route_override or route, 'sources': src,
                                             'selection': 'waves-v1'}), default=float) + '\n')
    return len(runs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('summary')
    s.add_argument('--write', action='store_true', help=f'also write {SUMMARY.relative_to(CAMPAIGN.parent.parent)}')
    sub.add_parser('exclusions')
    w = sub.add_parser('write')
    w.add_argument('--wave', required=True)
    w.add_argument('--out', required=True)
    w.add_argument('--twoscale', action='store_true', help="write the deferred wave with hpc.route = 'twoscale'")
    a = ap.parse_args()
    if a.cmd == 'exclusions':
        ex = build_exclusions(load_frozen())
        EXCLUSIONS.parent.mkdir(parents=True, exist_ok=True)
        json.dump(ex, open(EXCLUSIONS, 'w'), indent=1)
        print(json.dumps(ex['counts']))
        sys.exit(0)
    waves = assign()
    if a.cmd == 'summary':
        summ = summarise(waves)
        if a.write:
            json.dump({'selection': 'waves-v1', 'long_max_core_h': LONG_MAX_CORE_H, 'waves': summ}, open(SUMMARY, 'w'), indent=1)
        print(json.dumps(summ, indent=1))
    else:
        if a.twoscale:
            if a.wave != 'deferred':
                sys.exit('--twoscale is only for the deferred wave')
        elif a.wave not in waves or a.wave in ('excluded', 'deferred'):
            sys.exit(f'wave must be one of {[x for x in waves if x not in ("excluded", "deferred")]} '
                     f'(deferred: only with --twoscale)')
        n = write_wave(waves, a.wave, a.out, 'twoscale' if a.twoscale else None)
        print(f'wave {a.wave}: {n} runs -> {a.out}')
