"""
Convergence-matrix manifests: corner states x numerical variants.

Acceptance (evaluated after the runs): for every corner, the production setting differs from the finer setting
by < 0.5 % in drift (or < 2x the phase-noise floor of study D, whichever is larger), and the tumbler-twin
difference Delta changes by < 10 % of itself. Otherwise the production numerics of that region are revised before
the runs that depend on it are used.

V (set v1, frozen at hpc-freeze-v1): five synthetic corners of the slow and ordinary strata, e <= 0.6.

    python research/campaign/build_convergence.py --out research/campaign/manifests/V_convergence.jsonl

V2 (set v2): seven corners taken from EXISTING campaign states, so each base variant is the production run itself
(same run_id, checked at build time) and the validation applies to points that are in the campaign.
  C1-C3  e in (0.6, 0.8]        gate for the e > 0.6 runs (wave 2) and the complete Morris trajectories
  C4-C5  fast stratum           gate for the executable fast runs (wave 2)
  C6     c/a = 0.12             mesh resolution for elongated bodies (adds a 5,120-facet variant)
  C7     Gamma = 6, ordinary    gate for the preserved low-Gamma runs (wave 3); reduced variant set, ~100 core-h runs

    python research/campaign/build_convergence.py --set v2 --out research/campaign/manifests_v2/V2_convergence.jsonl
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
from campaign.build_campaign import write  # noqa: E402
from tools.campaign import lines_for_unit  # noqa: E402

# unit coordinates x0..x11 = lambda, sin(beta), b/a, c/b, u, logP_phi, phi0, psi0, rho, logGamma, e, logD
CORNERS = {
    'slow_lowG_LAM_ecc': ('slow', [0.20, 0.75, 0.30, 0.50, 0.85, 0.80, 0.1, 0.3, 0.5, 0.05, 0.75, 0.6]),
    'slow_highG_SAM':    ('slow', [0.60, 0.30, 0.70, 0.70, 0.20, 0.10, 0.4, 0.6, 0.5, 0.95, 0.25, 0.6]),
    'ord_lowG_SAMsep':   ('ordinary', [0.40, 0.90, 0.50, 0.60, 0.45, 0.15, 0.7, 0.2, 0.5, 0.08, 0.40, 0.6]),
    'ord_highG_LAM':     ('ordinary', [0.80, 0.10, 0.20, 0.30, 0.65, 0.85, 0.2, 0.8, 0.5, 0.90, 0.10, 0.6]),
    'ord_midG_obl90':    ('ordinary', [0.10, 0.50, 0.60, 0.80, 0.30, 0.50, 0.5, 0.5, 0.5, 0.50, 0.50, 0.6]),
}
VARIANTS = {'base': {}, 'steps_x2': {'steps_factor': 2.0}, 'steps_x4': {'steps_factor': 4.0},
            'dz_0.05': {'dz_over_ld': 0.05}, 'dz_0.2': {'dz_over_ld': 0.2}, 'depth_3': {'n_seasonal_depths': 3.0},
            'depth_6': {'n_seasonal_depths': 6.0}, 'facets_1280': {'n_subdiv': 3},
            'orbits_30': {'long_orbits': 30, 'long_if_slowest_period_gt_orbit_fraction': 0.0}}

# V2: corner -> (manifest holding the state, state_id, variant names)
ALL9 = tuple(VARIANTS)
REDUCED = ('base', 'steps_x2', 'dz_0.05', 'dz_0.2', 'depth_3', 'depth_6', 'facets_1280')   # no steps_x4, no orbits_30
V2_VARIANTS = dict(VARIANTS, facets_5120={'n_subdiv': 4})
V2_CORNERS = {
    'C1_slow_e077_lowG':   ('B_sobol_slow', 'sobol2101_slow_01003', ALL9),
    'C2_ord_e080_highG':   ('B_sobol_ordinary', 'sobol2102_ordinary_00406', ALL9),
    'C3_ord_e076_lowG':    ('B_sobol_ordinary', 'sobol2102_ordinary_00315', REDUCED),
    'C4_fast_P1p8h':       ('E_sobol_fast', 'sobol4101_fast_00101', ALL9),
    'C5_fast_P0p4h_e068':  ('E_sobol_fast', 'sobol4101_fast_00217', REDUCED),
    'C6_shape_ca012':      ('A_morris_slow', 'morris1101_slow_00105', ALL9 + ('facets_5120',)),
    'C7_ord_G6_preserved': ('B_sobol_ordinary', 'sobol2102_ordinary_00739', ('base', 'steps_x2', 'dz_0.05')),
}
FROZEN = RESEARCH / 'campaign' / 'manifests'


def frozen_state(manifest, state_id):
    """The tumbler line and P_par twin of an existing campaign state (read from the frozen .jsonl.gz)."""
    out = {}
    for raw in gzip.open(FROZEN / f'{manifest}.jsonl.gz', 'rt'):
        if raw.strip():
            ln = json.loads(raw)
            if ln['state_id'] == state_id and ln['kind'] in ('tumbler', 'pa_P_par'):
                out[ln['kind']] = ln
    if 'tumbler' not in out:
        raise KeyError(f'{state_id} not found in {manifest}')
    return out


def build_v1():
    lines = []
    for name, (stratum, x) in CORNERS.items():
        for vname, num in VARIANTS.items():
            for ln in lines_for_unit(np.array(x), stratum, f'conv_{name}_{vname}', twins=('P_par',), numerics=num):
                ln['convergence'] = {'corner': name, 'variant': vname}
                lines.append(ln)
    return lines


def build_v2():
    lines = []
    for name, (manifest, sid, variants) in V2_CORNERS.items():
        src = frozen_state(manifest, sid)
        x, stratum = np.array(src['tumbler']['unit']), src['tumbler']['stratum']
        for vname in variants:
            new = lines_for_unit(x, stratum, f'conv2_{name}_{vname}', twins=('P_par',), numerics=V2_VARIANTS[vname],
                                 rotation=src['tumbler']['derived'])
            if vname == 'base':
                # the base variant must be the production run itself (identical solver configuration)
                for ln in new:
                    ref = src.get(ln['kind'])
                    if ref is None or ln['run_id'] != ref['run_id'] or ln['config_overrides'] != ref['config_overrides']:
                        raise RuntimeError(f'{name}: base {ln["kind"]} does not reproduce {sid} from {manifest}')
            for ln in new:
                ln['convergence'] = {'set': 'v2', 'corner': name, 'variant': vname, 'source_manifest': manifest,
                                     'source_state_id': sid}
                lines.append(ln)
    return lines


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', choices=('v1', 'v2'), default='v1')
    a = ap.parse_args()
    lines = build_v1() if a.set == 'v1' else build_v2()
    write(lines, a.out)
    print(json.dumps({'runs': len(lines), 'core_h': round(sum(l['policy']['cost_core_h_estimate'] for l in lines), 1),
                      'routes': {r: sum(l['policy']['route'] == r for l in lines) for r in {l['policy']['route'] for l in lines}}}))
