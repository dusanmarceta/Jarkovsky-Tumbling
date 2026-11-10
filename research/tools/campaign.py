"""
Campaign generator for the 12-input synthetic-asteroid study (all eight rotational keys varied).

Inputs (unit-cube coordinate x_k -> code key):
  x0 lambda_L     U[0, 360) deg
  x1 beta_L       sin(beta_L) ~ U[-1, 1]  (isotropic L; obliquity gamma = 90 - beta_L)
  x2 I1 \\        via homogeneous-ellipsoid axis ratios b/a ~ U[0.30, 0.98] and c/b ~ U[0.40, 0.98];
  x3 I2 /         I1 = (b^2+c^2)/(1+b^2), I2 = (1+c^2)/(1+b^2) -- one-to-one onto the admissible inertia region
  x4 E_ratio      balanced excitation u: (0, 0.5) SAM, (0.5, 1) LAM (spinstate.e_ratio_from_u)
  x5 P_phi_h      log-uniform in the stratum's P_phi range (mean precession period, the observed tumbler period)
  x6 phi_0        U[0, 360) deg
  x7 psi_0        uniform over the reachable set sin^2(psi_0) >= (E - 1/I2)/(1/I1 - 1/I2)
  x8 density      U[1000, 3500] kg m^-3 (bulk; the code uses the same rho in the thermal column, where at fixed
                  Gamma it cancels)
  x9 thermal_inertia  log-uniform [5, 2500] SI   (K = Gamma^2/(rho c_p) derived; c_p fixed at 800)
  x10 ecc         U[0, 0.8] at a = 1 au (e > 0.6 flagged for stiffness validation)
  x11 D_eq        log-uniform in the stratum's diameter range [m]
Rotation strata (P_phi range, D range) are defined in STRATA; see research/campaign/parameter_space.md.

Every manifest line stores: unit coordinates, the 12 inputs, all derived rotational quantities (mode, theta range,
P_psi, polhode period, P_par, P_dt, P_lc, period ratios and near-commensurabilities), thermal and validity
quantities (Theta, skin depths, finite-body factors), the solver route, numerics, cost estimate, the mesh
specification (mesh generated at run time) and the complete configuration overrides.
"""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from .ellipsoid import analytic_inertia_ratios
from .population import YEAR_H, run_id
from .spinstate import derived, e_ratio_from_u

FIXED = {'a_au': 1.0, 'albedo': 0.1, 'emissivity': 0.9, 'specific_heat_capacity': 800.0}
SHAPE = {'b_over_a': (0.30, 0.98), 'c_over_b': (0.40, 0.98)}
RHO = (1000.0, 3500.0)
GAMMA = (5.0, 2500.0)
ECC = (0.0, 0.8)
ECC_VALIDATED = 0.6
STRATA = {   # P_phi range [h], D range [m]
    'slow':      {'P_phi_h': (24.0, 1000.0), 'D_m': (20.0, 10000.0)},
    'ordinary':  {'P_phi_h': (2.2, 24.0), 'D_m': (20.0, 10000.0)},
    'fast':      {'P_phi_h': (10.0 / 60, 2.2), 'D_m': (10.0, 200.0)},
    'superfast': {'P_phi_h': (5.0 / 3600, 10.0 / 60), 'D_m': (3.0, 150.0)},
}
NUMERICS = {'n_subdiv': 2, 'steps_per_rotation': 100, 'n_seasonal_depths': 4.5, 'dz_over_ld': 0.1,
            'yarkovsky_tolerance': 2e-3, 'abs_tol_ref_m_s': 2e-9, 'min_orbits': 4, 'max_orbits': 12,
            'long_orbits': 30, 'long_if_slowest_period_gt_orbit_fraction': 0.05}
# Measured cost model (research/runs/verification/raw/benchmark, Apple M4, 1 thread, 4 concurrent runs):
# t_step = C0 + C1 * facets * layers [s]; tumbler C0 = 0.96 ms, PA C0 = 0.76 ms, C1 = 6.0 ns (fit residuals < 8 %)
COST = {'C0_s': 9.6e-4, 'C1_s': 6.0e-9}
# Surface stiffness of the Crank-Nicolson step: S = rad_coeff * T^3 = (2 pi / N) / (dz / l_d) / Theta(T_q), with
# Theta at the perihelion subsolar temperature. Measured: S ~ 0.7-1.05 -> no backward-Euler fallback (flyby
# campaign, scaling and c_p checks); S ~ 10-40 -> 67-1804 fallbacks and one divergent run (benchmark). The
# steps per rotation are raised until S <= S_TARGET, so the defective Newton branch is never reached.
S_TARGET = 1.0
ROUTE_LIMITS = {'full_max_core_h': 12.0, 'full_costly_max_core_h': 96.0}

SIGMA = 5.670374419e-8
S0 = 1361.0


def lin(x, lo, hi):
    return lo + x * (hi - lo)


def logu(x, lo, hi):
    return float(10 ** (math.log10(lo) + x * (math.log10(hi) - math.log10(lo))))


def psi0_from_unit(w, I1, I2, E):
    """Uniform over the reachable set: I3*A(psi_0) >= E with A = sin^2/I1 + cos^2/I2 (I3 = 1)."""
    s_min = max(0.0, (E - 1.0 / I2) / (1.0 / I1 - 1.0 / I2)) if I1 < I2 else 0.0
    alpha = math.asin(min(1.0, math.sqrt(s_min))) + (1e-6 if s_min > 0 else 0.0)
    span = math.pi - 2 * alpha
    psi = alpha + w * 2 * span if w < 0.5 else math.pi + alpha + (w - 0.5) * 2 * span
    return math.degrees(psi) % 360.0, s_min


def near_commensurability(P_phi_h, T_omega_h, max_order=3):
    """Slowest low-order combination of the two rotational frequencies, |p/T_omega - q/P_phi| with
    1 <= p, q <= max_order (the insolation of a convex body is dominated by low harmonics). Its period is the
    beat period: if it is comparable to the orbital period, the rotational phase pattern seen along the orbit
    repeats slowly and finite-time (orbit-to-orbit) averages can differ, so the state gets long averaging."""
    if not (np.isfinite(P_phi_h) and np.isfinite(T_omega_h)) or P_phi_h <= 0 or T_omega_h <= 0:
        return {'resonance': None, 'beat_period_h': None}
    best = None
    for p in range(1, max_order + 1):
        for q in range(1, max_order + 1):
            df = abs(p / T_omega_h - q / P_phi_h)
            if best is None or df < best[0]:
                best = (df, p, q)
    df, p, q = best
    return {'resonance': f'{p}:{q}', 'beat_period_h': (1.0 / df) if df > 0 else float('inf')}


def finite_body_factor(D, Gamma, rho, cp, period_s, eps, Tstar):
    """F(R/l, Theta)/F(inf, Theta) of the finite-sphere linear theory (Vokrouhlicky 1999; repo's Fnu_eval)."""
    import sys
    from .runner import REPO
    sys.path.insert(0, str(REPO / 'Vokrouhlicky model'))
    from linearni_model_eccentric import Fnu_eval
    w = 2 * math.pi / period_s
    l = Gamma / (rho * cp * math.sqrt(w))
    Th = Gamma * math.sqrt(w) / (eps * SIGMA * Tstar ** 3)
    return float(Fnu_eval(D / 2 / l, Th) / Fnu_eval(1e9, Th)), l, Th


def state_from_unit(x, stratum):
    s = STRATA[stratum]
    b_a = lin(x[2], *SHAPE['b_over_a'])
    c_b = lin(x[3], *SHAPE['c_over_b'])
    c_a = c_b * b_a
    I1, I2 = analytic_inertia_ratios(b_a, c_a)
    u = float(np.clip(x[4], 1e-9, 1 - 1e-9))
    E = e_ratio_from_u(u, I1, I2)
    psi0, s_min = psi0_from_unit(x[7], I1, I2, E)
    return {'stratum': stratum, 'lambda_L': lin(x[0], 0.0, 360.0),
            'beta_L': math.degrees(math.asin(lin(x[1], -1.0, 1.0))),
            'b_over_a': b_a, 'c_over_b': c_b, 'c_over_a': c_a, 'I1': I1, 'I2': I2, 'u_exc': u, 'E_ratio': E,
            'P_phi_h': logu(x[5], *s['P_phi_h']), 'phi_0': lin(x[6], 0.0, 360.0), 'psi_0': psi0,
            'psi0_reachable_sin2_min': s_min, 'density': lin(x[8], *RHO), 'thermal_inertia': logu(x[9], *GAMMA),
            'ecc': lin(x[10], *ECC), 'D_eq_m': logu(x[11], *s['D_m'])}


def derive_state(st, numerics=None, check_against_solver=False, rotation=None):
    """check_against_solver: also derive through the solver's own Simulation (slow) and require agreement.
    rotation: the 'derived' block of an existing manifest line for this state. Its rotational periods are reused
    instead of being derived again, so lines rebuilt for an existing state (e.g. convergence variants) reproduce its
    configuration exactly. Manifests A-D were derived through the Simulation, E and later through fastspin; the two
    agree to < 1e-8 but not to the last digit."""
    num = dict(NUMERICS, **(numerics or {}))
    if rotation is not None:
        d = dict(rotation)
    else:
        from .fastspin import derived_fast
        ref = derived_fast(st['I1'], st['I2'], st['E_ratio'], 10.0, st['psi_0'], st['phi_0'])
        if check_against_solver:
            slow = derived(st['I1'], st['I2'], st['E_ratio'], 10.0, st['psi_0'], st['phi_0'])
            diff = max(abs(ref[k] / slow[k] - 1) for k in ('P_par_h', 'P_dt_h', 'T_omega_h')
                       if np.isfinite(slow[k]) and slow[k])
            if diff > 1e-6:
                raise RuntimeError(f'fast derivation differs from the solver set-up by {diff:.2e}')
            ref['solver_check_max_rel_diff'] = diff
        k = st['P_phi_h'] / 10.0                               # torque-free motion is scale-invariant in time
        d = {key: (v * k if key.endswith('_h') and isinstance(v, float) else v) for key, v in ref.items()}
    d['P_phi_h'] = st['P_phi_h']
    d['delta_t_s'] = d['P_dt_h'] * 3600.0 / num['steps_per_rotation']
    P_orb_h = YEAR_H * FIXED['a_au'] ** 1.5
    d['ratio_Ppsi_Pphi'] = d['P_psi_h'] / d['P_phi_h'] if np.isfinite(d['P_psi_h']) else None
    d['ratio_Tomega_Pphi'] = d['T_omega_h'] / d['P_phi_h']
    d['resonance_phi_omega'] = near_commensurability(d['P_phi_h'], d['T_omega_h'])
    beat = d['resonance_phi_omega']['beat_period_h'] or 0.0
    slowest = max(x for x in (d['P_phi_h'], d['P_psi_h'], d['T_omega_h'], beat) if np.isfinite(x))
    d['slowest_rotational_period_h'] = slowest
    d['orbit_over_slowest'] = P_orb_h / slowest
    # thermal and validity
    rho, G, cp = st['density'], st['thermal_inertia'], FIXED['specific_heat_capacity']
    eps = FIXED['emissivity']
    Tstar = ((1 - FIXED['albedo']) * S0 / (eps * SIGMA)) ** 0.25
    f_s, l_s, Th_s = finite_body_factor(st['D_eq_m'], G, rho, cp, P_orb_h * 3600, eps, Tstar)
    f_d, l_d, Th_d = finite_body_factor(st['D_eq_m'], G, rho, cp, d['P_par_h'] * 3600, eps, Tstar)
    th = {'K_W_mK': G ** 2 / (rho * cp), 'CpK': G ** 2 / rho, 'kappa_m2_s': G ** 2 / (rho * cp) ** 2,
          'l_seasonal_m': l_s, 'l_diurnal_m': l_d, 'Theta_seasonal': Th_s, 'Theta_diurnal_Ppar': Th_d,
          'finite_body_factor_seasonal': f_s, 'finite_body_factor_diurnal': f_d,
          'mass_kg': rho * math.pi / 6 * st['D_eq_m'] ** 3, 'q_au': FIXED['a_au'] * (1 - st['ecc'])}
    # numerics in the solver's units (diurnal skin depths of P_dt)
    n_skin = num['n_seasonal_depths'] * math.sqrt(P_orb_h / d['P_dt_h'])
    n_layers = int(math.ceil(n_skin / num['dz_over_ld']))
    w_dt = 2 * math.pi / (d['P_dt_h'] * 3600)
    T_q = Tstar / math.sqrt(th['q_au'])
    Theta_q = G * math.sqrt(w_dt) / (eps * SIGMA * T_q ** 3)
    S_base = (2 * math.pi / num['steps_per_rotation']) / num['dz_over_ld'] / Theta_q
    N_rot = int(max(num['steps_per_rotation'], math.ceil(num['steps_per_rotation'] * S_base / S_TARGET)))
    N_rot = int(round(N_rot * num.get('steps_factor', 1.0)))        # convergence studies
    th.update({'Theta_dt_perihelion': Theta_q, 'T_subsolar_perihelion_K': T_q, 'cn_stiffness_at_base_steps': S_base,
               'cn_stiffness': S_base * num['steps_per_rotation'] / N_rot})
    d['delta_t_s'] = d['P_dt_h'] * 3600.0 / N_rot
    long_avg = slowest > num['long_if_slowest_period_gt_orbit_fraction'] * P_orb_h
    orbits = num['long_orbits'] if long_avg else 6
    steps = P_orb_h / d['P_dt_h'] * N_rot
    facets = 20 * 4 ** num['n_subdiv']
    cost_h = orbits * steps * (COST['C0_s'] + COST['C1_s'] * facets * n_layers) / 3600
    flags = []
    if abs(f_s - 1) > 0.05 or abs(f_d - 1) > 0.05:
        flags.append('outside_1d_validity')
    if st['ecc'] > ECC_VALIDATED:
        flags.append('ecc_needs_stiffness_validation')
    if st['D_eq_m'] > 200 and d['P_dt_h'] < 2.2:
        flags.append('above_spin_barrier')
    if cost_h <= ROUTE_LIMITS['full_max_core_h']:
        route = 'full'
    elif cost_h <= ROUTE_LIMITS['full_costly_max_core_h']:
        route = 'full_costly'
    else:
        route = 'multiscale_required'
    run_policy = {'averaging': 'long' if long_avg else 'converge', 'orbits_estimate': orbits, 'steps_per_rotation': N_rot,
                  'n_skin_depths': n_skin, 'n_layers': n_layers, 'steps_per_orbit': steps, 'facets': facets,
                  'cost_core_h_estimate': cost_h, 'route': route, 'flags': flags}
    return d, th, run_policy


def config_overrides(st, d, policy, numerics=None, kind='tumbler', twin_period_h=None, dt_s=None):
    num = dict(NUMERICS, **(numerics or {}))
    mesh_name = mesh_filename(st['b_over_a'], st['c_over_a'], st['D_eq_m'], num['n_subdiv'])
    cfg = {'shape_model_file': mesh_name, 'shape_model_directory': 'MESH_DIR',
           'thermal_inertia': st['thermal_inertia'], 'density': st['density'],
           'specific_heat_capacity': FIXED['specific_heat_capacity'], 'emissivity': FIXED['emissivity'],
           'albedo': FIXED['albedo'], 'beaming_factor': 1.0, 'a_au': FIXED['a_au'], 'ecc': st['ecc'],
           'lambda_L': st['lambda_L'], 'beta_L': st['beta_L'], 'I1': st['I1'], 'I2': st['I2'],
           'version': 'new', 'temp_solver': 'tempestovsky_implicit', 'include_shadowing': True, 'n_scatters': 0,
           'include_self_heating': False, 'orbital_initialisation': 0, 'deep_temperature_correction': True,
           'yarkovsky_extrapolate': True, 'max_insolation_memory_mb': 500,
           'yarkovsky_tolerance': num['yarkovsky_tolerance'],
           'yarkovsky_abs_tolerance': num['abs_tol_ref_m_s'] * (1000.0 / st['D_eq_m']) * (2000.0 / st['density'])}
    if policy['averaging'] == 'long':
        cfg.update(min_orbits=num['long_orbits'], max_orbits=num['long_orbits'])
    else:
        cfg.update(min_orbits=num['min_orbits'], max_orbits=num['max_orbits'])
    if kind == 'tumbler':
        cfg.update(E_ratio=st['E_ratio'], P_phi_h=st['P_phi_h'], phi_0=st['phi_0'], psi_0=st['psi_0'],
                   implicit_timesteps_per_day=policy['steps_per_rotation'], n_skin_depths=policy['n_skin_depths'],
                   n_layers=policy['n_layers'])
    else:
        # principal-axis twin: spin about the short axis along L with period twin_period_h; same time step and
        # same physical depth grid as the tumbler (psi_0 = 90 deg -> P_phi_h = P * I1 in the code's E = 1 branch)
        steps = max(20, int(round(twin_period_h * 3600.0 / dt_s)))
        cfg.update(E_ratio=1.0, P_phi_h=twin_period_h * st['I1'], phi_0=0.0, psi_0=90.0,
                   implicit_timesteps_per_day=steps,
                   n_skin_depths=policy['n_skin_depths'] * math.sqrt(d['P_dt_h'] / twin_period_h),
                   n_layers=policy['n_layers'])
    return cfg


def mesh_filename(b_a, c_a, D, n_subdiv):
    key = f'{b_a:.8f}_{c_a:.8f}_{D:.4f}_{n_subdiv}'
    return f'ell_{hashlib.sha256(key.encode()).hexdigest()[:12]}_n{20 * 4 ** n_subdiv}.stl'


def lines_for_unit(x, stratum, state_id, twins=('P_par',), numerics=None, replicate=None, check=False, rotation=None):
    st = state_from_unit(x, stratum)
    d, th, policy = derive_state(st, numerics, check_against_solver=check, rotation=rotation)
    num = dict(NUMERICS, **(numerics or {}))
    base = {'state_id': state_id, 'stratum': stratum, 'unit': [float(v) for v in x], 'inputs': st, 'derived': d,
            'thermal': th, 'policy': policy, 'fixed': FIXED, 'numerics': num, 'replicate': replicate,
            'mesh_spec': {'b_over_a': st['b_over_a'], 'c_over_a': st['c_over_a'], 'D_eq_m': st['D_eq_m'],
                          'n_subdiv': num['n_subdiv']}}
    out = []
    cfg = config_overrides(st, d, policy, numerics)
    out.append(dict(base, kind='tumbler', config_overrides=cfg))
    for conv in twins:
        P = d.get(conv + '_h')
        if P is None or not np.isfinite(P):
            continue
        c = config_overrides(st, d, policy, numerics, kind='pa', twin_period_h=P, dt_s=d['delta_t_s'])
        out.append(dict(base, kind='pa_' + conv, twin_period_h=P, config_overrides=c))
    for ln in out:
        ln['run_id'] = run_id({'kind': ln['kind'], **{k: v for k, v in ln['config_overrides'].items()}})
    return out


# ---------------------------------------------------------------- designs (unit hypercube, d = 12)
D_IN = 12


def design_sobol(n, seed):
    from scipy.stats import qmc
    return qmc.Sobol(D_IN, scramble=True, seed=seed).random(n)


def design_morris(r, levels, seed):
    """Morris (1991) trajectories: r trajectories of D_IN + 1 points, p levels, step Delta = p / (2 (p - 1))."""
    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    pts = []
    for _ in range(r):
        x = rng.choice(grid[grid <= 1 - delta + 1e-12], D_IN)
        traj = [x.copy()]
        for k in rng.permutation(D_IN):
            x = x.copy()
            x[k] = x[k] + delta if x[k] + delta <= 1 + 1e-12 else x[k] - delta
            traj.append(x)
        pts.append(np.array(traj))
    return np.clip(np.concatenate(pts), 1e-6, 1 - 1e-6)


def design_saltelli(n, seed):
    """A, B and the D_IN matrices A_B^(i) (Saltelli et al. 2010): n * (D_IN + 2) points, ordered by blocks."""
    from scipy.stats import qmc
    base = qmc.Sobol(2 * D_IN, scramble=True, seed=seed).random(n)
    A, B = base[:, :D_IN], base[:, D_IN:]
    blocks = [A, B] + [np.where(np.arange(D_IN) == i, B, A) for i in range(D_IN)]
    return np.concatenate(blocks)
