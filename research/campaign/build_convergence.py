"""
Convergence-matrix manifest (first HPC stage): corner states of the executable strata x numerical variants.

Acceptance (evaluated after the runs): for every corner, the production setting differs from the finer setting
by < 0.5 % in drift (or < 2x the phase-noise floor of study D, whichever is larger), and the tumbler-twin
difference Delta changes by < 10 % of itself. Otherwise the production numerics of that region are revised before
studies A-D are run.

    python research/campaign/build_convergence.py --out research/campaign/manifests/V_convergence.jsonl
"""
import argparse
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    lines = []
    for name, (stratum, x) in CORNERS.items():
        for vname, num in VARIANTS.items():
            for ln in lines_for_unit(np.array(x), stratum, f'conv_{name}_{vname}', twins=('P_par',), numerics=num):
                ln['convergence'] = {'corner': name, 'variant': vname}
                lines.append(ln)
    write(lines, a.out)
    print(json.dumps({'runs': len(lines), 'core_h': round(sum(l['policy']['cost_core_h_estimate'] for l in lines), 1),
                      'routes': {r: sum(l['policy']['route'] == r for l in lines) for r in {l['policy']['route'] for l in lines}}}))
