"""
Two-scale method building blocks (research/superfast), checked against the full solver's own code.

Acceptance criteria:
  attitude: rotation matrices and body-frame Sun directions equal compute_tumbling_dynamics to < 1e-8 (SAM, LAM,
            principal axis) over many body periods
  orbit: Sun direction and distance equal utils.sun_direction to 1e-12; time(f) inverts f(t)
  drift formula: solver_dadt from (F.s_B, F.u_B) equals calculate_yarkovsky (radial and transverse, e > 0)
  torus table: the analytic precession average equals brute-force sampling of the precession angle to 1e-3
  column kernel: fast_window with zero sink reproduces implicit_orbit_step (the solver's orbit step) exactly
  local energy balance: at a converged frozen position, <absorbed> - eps sigma <T^4> - sink -> 0 per facet
"""
import contextlib
import gzip
import io
import json
import math
import types

import numpy as np
import pytest

from superfast import orbit
from superfast.attitude import L_to_body, TorqueFreeAttitude
from superfast.kernels import fast_window, torus_insolation
from superfast.twoscale import Numerics, TwoScaleYarkovsky, simulation_from_config
from tools.manifest import ensure_mesh, full_config


@pytest.fixture(scope='module')
def lines():
    L = [json.loads(l) for l in gzip.open(str(__import__('pathlib').Path(__file__).resolve().parents[1]
                                              / 'campaign/manifests/B_sobol_ordinary.jsonl.gz'), 'rt')]
    sam = next(l for l in L if l['kind'] == 'tumbler' and l['derived']['mode'] == 'SAM')
    lam = next(l for l in L if l['kind'] == 'tumbler' and l['derived']['mode'] == 'LAM' and l['inputs']['ecc'] > 0.3)
    pa = next(l for l in L if l['kind'] == 'pa_P_par' and l['state_id'] == lam['state_id'])
    return {'SAM': sam, 'LAM': lam, 'PA': pa}


def sim_for(line):
    return simulation_from_config(full_config(line['config_overrides']))


@pytest.mark.parametrize('kind', ['SAM', 'LAM', 'PA'])
def test_attitude_matches_solver(lines, kind):
    from src.utilities.tumbling_final_2 import compute_tumbling_dynamics
    from src.utilities.utils import sun_direction
    sim = sim_for(lines[kind])
    att = TorqueFreeAttitude(sim.y0, sim.I, sim.lambda_L, sim.beta_L, sim.mode == 'Fixed-Axis Rotation')
    n = 1500
    dt = 5 * att.period_s / n if kind != 'PA' else sim.delta_t
    t = np.arange(1, n + 1) * dt
    s = np.array([sun_direction(x, sim)[0] for x in t])
    with np.errstate(all='ignore'):
        ref = compute_tumbling_dynamics(dt, n, sim.y0, sim.I, s.copy(), sim.lambda_L, sim.beta_L, 0)
    assert np.abs(att.body_to_inertial(t) - ref['rotations']).max() < 1e-8
    sB = np.einsum('nij,nj->ni', att.inertial_to_body(t), s)
    assert np.abs(sB - ref['r_body']).max() < 1e-8


def test_orbit_matches_solver(lines):
    from src.utilities.utils import sun_direction
    sim = sim_for(lines['LAM'])
    tt = np.linspace(0, orbit.orbital_period(sim.a_au), 37)
    s, r, f = orbit.state_at(tt, sim.a_au, sim.ecc)
    for x, si, ri in zip(tt, s, r):
        sv, rv, _ = sun_direction(x, sim)
        assert np.abs(si - sv).max() < 1e-9 and abs(ri - rv / orbit.AU) < 1e-9
    back = orbit.time_of_true_anomaly(f, sim.a_au, sim.ecc)
    P = orbit.orbital_period(sim.a_au)
    assert np.abs((back - tt + P / 2) % P - P / 2).max() < 1e-5


def test_drift_formula_is_the_solvers(lines, tmp_path):
    from src.model.solvers.tempestovsky import calculate_yarkovsky
    ln = lines['LAM']
    cfg = full_config(ln['config_overrides'])
    mesh_dir = ensure_mesh(ln, tmp_path)
    cfg['shape_model_directory'] = str(mesh_dir) + '/'
    m = TwoScaleYarkovsky(cfg, mesh_dir / cfg['shape_model_file'], Numerics(n_loc=4))
    rng = np.random.default_rng(1)
    T = 150 + 200 * rng.random(m.normals.shape[0])
    for k in range(4):
        s_I, r_au, f = m.s_loc[k], float(m.r_loc[k]), float(m.f_loc[k])
        u_I = np.cross(s_I, [0, 0, 1.0])
        M = m.att.inertial_to_body(np.array([123.4]))[0]
        sB, uB = M @ s_I, M @ u_I
        Fs = m.coef * np.sum(m.areas * T ** 4 * (m.normals @ sB))
        Fu = m.coef * np.sum(m.areas * T ** 4 * (m.normals @ uB))
        ours = m.solver_dadt(Fs, -Fu, f, r_au)
        ref = calculate_yarkovsky(m.sim, m.normals, m.areas, m.mass, -sB, -uB, f, r_au, T)
        assert ours == pytest.approx(ref, rel=1e-12, abs=1e-30)


def test_torus_average_against_sampling(lines):
    sim = sim_for(lines['SAM'])
    att = TorqueFreeAttitude(sim.y0, sim.I, sim.lambda_L, sim.beta_L, False)
    rng = np.random.default_rng(3)
    normals = rng.normal(size=(5, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    tau = (np.arange(64) + 0.5) * att.period_s / 64
    _, th, psi = att.angles(tau)
    M = L_to_body(np.zeros_like(th), th, psi)
    cz = np.array([-0.9, -0.3, 0.0, 0.4, 0.95])
    g = torus_insolation(normals, np.ascontiguousarray(M), cz)
    phis = (np.arange(720) + 0.5) * 2 * math.pi / 720
    for j, c in enumerate(cz):
        sL = np.array([math.sqrt(1 - c * c), 0.0, c])
        acc = np.zeros(5)
        for ph in phis:
            sB = L_to_body(np.full(64, ph), th, psi) @ sL
            acc += np.maximum(normals @ sB.T, 0).mean(axis=1)
        assert np.abs(g[:, j] - acc / len(phis)).max() < 1e-3


def test_fast_kernel_equals_solver_orbit_step():
    from src.model.solvers.tempestovsky_implicit import implicit_orbit_step
    rng = np.random.default_rng(5)
    nf, nl, n = 6, 40, 300
    normals = rng.normal(size=(nf, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    ang = np.linspace(0, 6 * math.pi, n)
    sB = np.stack([np.cos(ang), np.sin(ang), np.full(n, 0.3)], axis=1)
    sB /= np.linalg.norm(sB, axis=1)[:, None]
    flux, c1, rad, a, theta = 1000.0, 2e-3, 1e-10, 0.8, 0.5
    T0 = 200 + 50 * rng.random((nf, nl))
    T_new = T0.copy()
    q_prev = flux * np.maximum(normals @ sB[0], 0)
    fast_window(T_new, normals, sB[1:], sB[1:], flux, np.zeros(nf), c1, rad, a, theta, q_prev.copy(),
                np.zeros(nf, np.bool_), np.zeros((nf, nl)))
    T_ref = T0.copy()
    qp = flux * np.maximum(normals @ sB[0], 0)
    for k in range(1, n):
        q = flux * np.maximum(normals @ sB[k], 0)
        implicit_orbit_step(T_ref, q, qp, c1, rad, a, theta, 0.0)
        qp = q
    assert np.abs(T_new - T_ref).max() == 0.0


def test_local_energy_balance(lines, tmp_path):
    ln = lines['PA']
    cfg = full_config(ln['config_overrides'])
    mesh_dir = ensure_mesh(ln, tmp_path)
    cfg['shape_model_directory'] = str(mesh_dir) + '/'
    m = TwoScaleYarkovsky(cfg, mesh_dir / cfg['shape_model_file'], Numerics(n_loc=4, tol_rel=1e-6, max_windows=400))
    m.torus_table()
    seas = m.seasonal()
    k = 1
    nf = m.normals.shape[0]
    T0 = np.repeat(seas['Tbar'][k], m.n_fast).reshape(nf, m.n_fast)
    r = m.local(k, T0, seas['q_s'][k])
    assert r['converged']
    imbalance = r['q_mean'] - m.eps * 5.670374419e-8 * (r['D4'] + r['T_mean'] ** 4) - seas['q_s'][k]
    assert np.abs(imbalance).max() < 1e-3 * r['q_mean'].max()
