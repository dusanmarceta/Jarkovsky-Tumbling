"""
Solver-level verification runs (full tempest.py runs on generated ellipsoid tumblers).

Groups
  scaling      drift ~ 1/rho and ~ 1/D at fixed Gamma (identical numerics; 2 orbits; exact-scaling check)
  convergence  time step, layer thickness, grid depth, facet count and orbit number for a reference tumbler
  phase        finite-time noise: replicates of the reference tumbler with different initial phases phi_0

Acceptance criteria are evaluated in analyse_verification.py.
    python research/verification/run_verification.py scaling convergence phase --workers 8
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
from tools.population import build_state, full_config  # noqa: E402
from tools.runner import run_case  # noqa: E402

OUT = RESEARCH / 'runs' / 'verification'
MESHES = RESEARCH / 'runs' / 'meshes'

# Reference tumbler T1: moderately elongated SAM tumbler, intermediate thermal inertia and obliquity
T1 = {'cos_gamma': 0.70710678, 'lambda_L': 60.0, 'b_over_a': 0.8, 'c_over_b': 0.75, 'u_exc': 0.25,
      'P_par_h': 16.0, 'Gamma': 300.0, 'ecc': 0.2}
# Stiff corner T2: deep LAM, low thermal inertia, faster spin, retrograde, more eccentric
T2 = {'cos_gamma': -0.5, 'lambda_L': 200.0, 'b_over_a': 0.6, 'c_over_b': 0.7, 'u_exc': 0.85,
      'P_par_h': 8.0, 'Gamma': 30.0, 'ecc': 0.4}


def tumbler(phys, numerics=None, fixed=None):
    st = build_state(phys, MESHES, numerics=numerics, fixed=fixed, twins=())
    rid, kind, cfg, rec = st['runs'][0]
    return full_config(cfg), st


def jobs_scaling():
    fast = {'n_subdiv': 1, 'steps_per_rotation': 50, 'n_seasonal_depths': 3.0, 'dz_over_ld': 0.2,
            'min_orbits': 2, 'max_orbits': 2}
    phys = dict(T1, P_par_h=24.0)
    out = []
    for name, fixed in (('ref', {}), ('rho_half', {'density': 1000.0}), ('D_half', {'D_eq_m': 500.0}),
                        ('rho_half_D_half', {'density': 1000.0, 'D_eq_m': 500.0})):
        cfg, st = tumbler(phys, fast, fixed)
        cfg['yarkovsky_abs_tolerance'] = 0.0
        out.append((f'scaling/{name}', cfg, st))
    return out


def jobs_convergence():
    out = []
    variants = {'base': {}, 'steps50': {'steps_per_rotation': 50}, 'steps200': {'steps_per_rotation': 200},
                'dz0.2': {'dz_over_ld': 0.2}, 'dz0.05': {'dz_over_ld': 0.05}, 'depth3': {'n_seasonal_depths': 3.0},
                'depth6': {'n_seasonal_depths': 6.0}, 'facets1280': {'n_subdiv': 3},
                'orbits12': {'min_orbits': 12, 'max_orbits': 12}}
    for name, num in variants.items():
        cfg, st = tumbler(T1, num)
        out.append((f'convergence/T1_{name}', cfg, st))
    for name, num in {'base': {}, 'steps200': {'steps_per_rotation': 200}, 'dz0.05': {'dz_over_ld': 0.05},
                      'orbits12': {'min_orbits': 12, 'max_orbits': 12}}.items():
        cfg, st = tumbler(T2, num)
        out.append((f'convergence/T2_{name}', cfg, st))
    return out


def jobs_phase():
    out = []
    for phi0 in (60.0, 120.0, 180.0, 240.0, 300.0):
        cfg, st = tumbler(T1, None, {'phi_0': phi0})
        out.append((f'phase/T1_phi{int(phi0)}', cfg, st))
    for lam in (150.0, 300.0):            # lambda_L at e = 0 is a phase as well
        cfg, st = tumbler(dict(T1, ecc=0.0, lambda_L=lam))
        out.append((f'phase/T1_e0_lambda{int(lam)}', cfg, st))
    cfg, st = tumbler(dict(T1, ecc=0.0))
    out.append(('phase/T1_e0_lambda60', cfg, st))
    return out


def expected_cost(cfg, st):
    d = st['state']['tumbler_derived']
    steps = 365.25 * 24 * cfg['a_au'] ** 1.5 / d['P_dt_h'] * cfg['implicit_timesteps_per_day']
    return steps * cfg['n_layers'] * (20 * 4 ** st['state']['numerics']['n_subdiv'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('groups', nargs='+', choices=['scaling', 'convergence', 'phase'])
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    jobs = sum(({'scaling': jobs_scaling, 'convergence': jobs_convergence, 'phase': jobs_phase}[g]() for g in a.groups), [])
    jobs.sort(key=lambda j: -expected_cost(j[1], j[2]))      # longest first
    MESHES.mkdir(parents=True, exist_ok=True)

    def go(job):
        name, cfg, st = job
        run_dir = OUT / 'raw' / name
        if (run_dir / 'provenance.json').exists() and json.load(open(run_dir / 'provenance.json')).get('exit_code') == 0:
            return name, 'skipped'
        rec = run_case(cfg, run_dir, 1, extra_provenance=st['state'])
        r = rec['result'] or {}
        return name, f"exit {rec['exit_code']} drift {r.get('drift_m_s')} orbits {r.get('orbits')} valid {r.get('valid')} {rec['wall_time_s']} s"

    # warm the numba cache and the visible-facet caches with one short run first
    print(go(jobs[-1]), flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for name, status in ex.map(go, jobs[:-1]):
            print(name, status, flush=True)
