"""
Synthetic tumbling-asteroid generator: physical design variables -> validated repository configurations.

Physical variables of one state (all independent by construction, see research/README.md):
    cos_gamma   cosine of the obliquity of the angular momentum L            -> beta_L = 90 deg - gamma
    lambda_L    longitude of L in the orbit frame (from perihelion) [deg]     -> lambda_L
    b_over_a    intermediate / long semi-axis                                 \\ ellipsoid mesh, I1 and I2
    c_over_b    short / intermediate semi-axis (c/a = c_over_b * b_over_a)    /  computed from the mesh
    u_exc       normalised excitation in [0, 1] (0 = principal-axis rotation) -> E_ratio (spinstate.e_ratio_from_u)
    P_par_h     2*pi*L/(2E): conserved spin period about L [h]                -> P_phi_h (spinstate.p_phi_for_period)
    Gamma       thermal inertia [SI]                                          -> thermal_inertia
    ecc         eccentricity at fixed a                                       -> ecc
Fixed nuisance phases: phi_0 = 0, psi_0 = 90 deg (always reachable). Diameter and bulk density are reference
values (drift scales as 1/(rho D) in this 1D model; verified separately before use).

Each state can be expanded into a tumbler run and principal-axis (PA) twins at different period conventions.
Twins use the *same* time step and the *same physical depth grid* as their tumbler, so that discretisation
errors largely cancel in tumbler - twin differences.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .ellipsoid import mesh_file
from .runner import REPO, load_template
from .spinstate import derived, e_ratio_from_u, p_phi_for_period

YEAR_H = 365.25 * 24.0

FIXED = {'a_au': 1.0, 'albedo': 0.1, 'emissivity': 0.9, 'density': 2000.0, 'specific_heat_capacity': 1000.0,
         'D_eq_m': 1000.0, 'phi_0': 0.0, 'psi_0': 90.0}

NUMERICS = {'n_subdiv': 2, 'steps_per_rotation': 100, 'n_seasonal_depths': 4.5, 'dz_over_ld': 0.1,
            'yarkovsky_tolerance': 2e-3, 'yarkovsky_abs_tolerance': 2e-9, 'min_orbits': 4, 'max_orbits': 12}

# Provisional Paper-I design box (unit-cube coordinate -> physical value); see research/README.md
BOUNDS = {'cos_gamma': (-1.0, 1.0, 'lin'), 'lambda_L': (0.0, 360.0, 'lin'), 'b_over_a': (0.4, 0.95, 'lin'),
          'c_over_b': (0.4, 0.95, 'lin'), 'u_exc': (0.0, 1.0, 'lin'), 'P_par_h': (2.0, 50.0, 'log'),
          'Gamma': (10.0, 2500.0, 'log'), 'ecc': (0.0, 0.5, 'lin')}


def from_unit(x):
    out = {}
    for (k, (lo, hi, kind)), v in zip(BOUNDS.items(), x):
        out[k] = float(lo + v * (hi - lo)) if kind == 'lin' else float(10 ** (np.log10(lo) + v * (np.log10(hi) - np.log10(lo))))
    return out


def sample_unit(n, seed, method='sobol'):
    from scipy.stats import qmc
    d = len(BOUNDS)
    eng = qmc.Sobol(d, scramble=True, seed=seed) if method == 'sobol' else qmc.LatinHypercube(d, seed=seed)
    return eng.random(n)


def run_id(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=float).encode()).hexdigest()[:16]


def grid_for(P_dt_h, Gamma, numerics, a_au):
    """Uniform grid in the solver's units: n_skin_depths (diurnal skin depths of P_dt) and n_layers."""
    P_orb_h = YEAR_H * a_au ** 1.5
    n_skin = numerics['n_seasonal_depths'] * np.sqrt(P_orb_h / P_dt_h)
    return float(n_skin), int(np.ceil(n_skin / numerics['dz_over_ld']))


def build_state(phys, mesh_dir, numerics=None, fixed=None, twins=('P_par', 'P_phi', 'P_lc')):
    """Return {'state': physical + derived record, 'runs': [(run_id, kind, config_overrides, record)]}."""
    numerics = dict(NUMERICS, **(numerics or {}))
    fixed = dict(FIXED, **(fixed or {}))
    b_a = phys['b_over_a']
    c_a = phys['c_over_b'] * b_a
    mesh_path, mesh = mesh_file(b_a, c_a, fixed['D_eq_m'], numerics['n_subdiv'], mesh_dir)
    I1, I2 = mesh['I1'], mesh['I2']
    if not (0 < I1 <= I2 <= 1.0 and I1 + I2 >= 1.0):
        raise ValueError(f'inertia constraint violated: I1 = {I1}, I2 = {I2}')
    gamma = float(np.degrees(np.arccos(np.clip(phys['cos_gamma'], -1, 1))))
    E = e_ratio_from_u(phys['u_exc'], I1, I2)
    P_phi = p_phi_for_period(I1, I2, E, phys['P_par_h'], 'P_par', fixed['psi_0'])
    d = derived(I1, I2, E, P_phi, fixed['psi_0'], fixed['phi_0'])
    if abs(d['P_par_h'] / phys['P_par_h'] - 1) > 1e-8 or abs(d['E_ratio_check'] / E - 1) > 1e-10:
        raise RuntimeError('spin-state conversion failed its self-check')
    n_skin, n_layers = grid_for(d['P_dt_h'], phys['Gamma'], numerics, fixed['a_au'])
    base = {'shape_model_file': mesh_path.name, 'shape_model_directory': str(mesh_path.parent),
            'thermal_inertia': phys['Gamma'], 'density': fixed['density'],
            'specific_heat_capacity': fixed['specific_heat_capacity'], 'emissivity': fixed['emissivity'],
            'albedo': fixed['albedo'], 'beaming_factor': 1.0, 'a_au': fixed['a_au'], 'ecc': phys['ecc'],
            'lambda_L': phys['lambda_L'], 'beta_L': 90.0 - gamma, 'I1': I1, 'I2': I2,
            'phi_0': fixed['phi_0'], 'psi_0': fixed['psi_0'], 'version': 'new', 'temp_solver': 'tempestovsky_implicit',
            'include_shadowing': True, 'n_scatters': 0, 'include_self_heating': False, 'orbital_initialisation': 0,
            'deep_temperature_correction': True, 'yarkovsky_extrapolate': True,
            'yarkovsky_tolerance': numerics['yarkovsky_tolerance'],
            'yarkovsky_abs_tolerance': numerics['yarkovsky_abs_tolerance'] * 1000.0 / fixed['D_eq_m'],
            'min_orbits': numerics['min_orbits'], 'max_orbits': numerics['max_orbits'], 'max_insolation_memory_mb': 500}
    state = {'physical': phys, 'fixed': fixed, 'numerics': numerics, 'gamma_deg': gamma, 'c_over_a': c_a,
             'mesh': {'file': str(mesh_path), **{k: mesh[k] for k in ('I1', 'I2', 'D_eq_m', 'max_offdiag_rel', 'area_m2')}},
             'E_ratio': E, 'tumbler_derived': d, 'n_skin_depths': n_skin, 'n_layers': n_layers}
    runs = []
    dt_s = d['delta_t_s'] if d['mode'] != 'PA' else d['P_dt_h'] * 3600.0 / numerics['steps_per_rotation']

    def add(kind, E_ratio, P_phi_h, P_rot_h):
        # same time step and same physical grid as the tumbler: scale steps and n_skin with the rotation period
        steps = int(round(P_rot_h * 3600.0 / dt_s))
        skin = n_skin * np.sqrt(d['P_dt_h'] / P_rot_h)
        cfg = dict(base, E_ratio=E_ratio, P_phi_h=P_phi_h, implicit_timesteps_per_day=steps,
                   n_skin_depths=float(skin), n_layers=n_layers)
        rid = run_id({k: v for k, v in cfg.items() if k not in ('shape_model_directory',)} | {'kind': kind})
        runs.append((rid, kind, cfg, {'kind': kind, 'period_h': P_rot_h, 'steps_per_rotation': steps,
                                       'n_skin_depths': float(skin), 'n_layers': n_layers}))

    if d['mode'] == 'PA':
        add('pa', 1.0, P_phi, d['P_dt_h'])
    else:
        cfg = dict(base, E_ratio=E, P_phi_h=P_phi, implicit_timesteps_per_day=numerics['steps_per_rotation'],
                   n_skin_depths=n_skin, n_layers=n_layers)
        rid = run_id({k: v for k, v in cfg.items() if k not in ('shape_model_directory',)} | {'kind': 'tumbler'})
        runs.append((rid, 'tumbler', cfg, {'kind': 'tumbler', 'period_h': d['P_dt_h'],
                                           'steps_per_rotation': numerics['steps_per_rotation'],
                                           'n_skin_depths': n_skin, 'n_layers': n_layers}))
        for conv in twins:
            P_target = d[conv + '_h']
            if not np.isfinite(P_target):
                continue                         # P_lc is undefined in LAM
            add('pa_' + conv, 1.0, p_phi_for_period(I1, I2, 1.0, P_target, 'P_par', fixed['psi_0']), P_target)
    return {'state': state, 'runs': runs}


def full_config(overrides, template=REPO / 'data/config/config_apophis.yaml'):
    cfg = load_template(template)
    cfg.update(overrides)
    return cfg
