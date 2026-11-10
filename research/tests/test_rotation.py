"""
Verification of the solver's rotational dynamics (src/utilities/tumbling_final_2.py and the set-up in
src/model/simulation.py). These are numerical-agreement tests: they check that the code integrates the
torque-free Euler equations it claims to integrate, not that torque-free rotation is physically adequate.

Acceptance criteria
  conservation : relative drift of E and |L| < 1e-8, direction of L within 1e-7 rad of (lambda_L, beta_L),
                 over >= 10 polhode periods with the production time step (P_dt / 100)
  sections     : integrating in two consecutive sections from last_state reproduces one long integration
                 to 1e-7 in the attitude matrices (the solver integrates orbit sections this way)
  modes        : SAM keeps the sign of the body-frame L3, LAM keeps the sign of L1
  phase        : every derived period is independent of psi_0 to 1e-6 (psi_0 is a pure phase)
"""
import numpy as np
import pytest

from conftest import I_ELL, LAM_E, SAM_E
from src.utilities.tumbling_final_2 import compute_tumbling_dynamics
from tools.spinstate import derived, simulation_for

LAMBDA, BETA = 37.0, -52.0


def integrate(I1, I2, E, n_polhode=10, steps_per_rot=100, P_phi=10.0):
    sim = simulation_for(I1, I2, E, P_phi, extra={'lambda_L': LAMBDA, 'beta_L': BETA})
    d = derived(I1, I2, E, P_phi)
    dt = sim.delta_t
    n = int(n_polhode * d['T_omega_h'] * 3600 / dt) if np.isfinite(d['T_omega_h']) else 20 * steps_per_rot
    res = compute_tumbling_dynamics(dt, n, sim.y0, sim.I, np.tile([1.0, 0.0, 0.0], (n, 1)), LAMBDA, BETA, 0)
    return sim, res, dt, n


def invariants(res, I):
    w = res['omega_body']
    Lb = w * np.array(I)
    E = 0.5 * np.sum(Lb * w, axis=1)
    Lin = np.einsum('nij,nj->ni', res['rotations'], Lb)        # body -> inertial (orbit) frame
    return E, np.linalg.norm(Lb, axis=1), Lin


@pytest.mark.parametrize('E_ratio', [SAM_E, LAM_E, 1.0 + 1e-4, 1.20])
def test_energy_and_momentum_conserved(E_ratio):
    sim, res, dt, n = integrate(*I_ELL, E_ratio)
    E, Lmag, Lin = invariants(res, sim.I)
    assert np.max(np.abs(E / E[0] - 1)) < 1e-8
    assert np.max(np.abs(Lmag / Lmag[0] - 1)) < 1e-8
    target = np.array([np.cos(np.radians(BETA)) * np.cos(np.radians(LAMBDA)),
                       np.cos(np.radians(BETA)) * np.sin(np.radians(LAMBDA)), np.sin(np.radians(BETA))])
    # einsum instead of '@': Accelerate BLAS on macOS raises spurious floating-point warnings in matmul
    ang = np.arccos(np.clip(np.einsum('ij,j->i', Lin, target) / np.linalg.norm(Lin, axis=1), -1, 1))
    assert ang.max() < 1e-7
    assert abs(2 * E[0] / Lmag[0] ** 2 - E_ratio) < 1e-10     # I3 = 1


def test_sections_chain_like_one_integration():
    sim, res_full, dt, n = integrate(*I_ELL, SAM_E, n_polhode=4)
    half = n // 2
    r = np.tile([1.0, 0.0, 0.0], (n, 1))
    a = compute_tumbling_dynamics(dt, half, sim.y0, sim.I, r[:half], LAMBDA, BETA, 0)
    b = compute_tumbling_dynamics(dt, n - half, a['last_state'], sim.I, r[half:], LAMBDA, BETA, 0)
    R2 = np.concatenate([a['rotations'], b['rotations']])
    assert np.max(np.abs(R2 - res_full['rotations'])) < 1e-7


@pytest.mark.parametrize('E_ratio,mode', [(SAM_E, 'SAM'), (LAM_E, 'LAM')])
def test_mode_topology(E_ratio, mode):
    sim, res, dt, n = integrate(*I_ELL, E_ratio, n_polhode=3)
    Lb = res['omega_body'] * np.array(sim.I)
    assert derived(*I_ELL, E_ratio, 10.0)['mode'] == mode
    if mode == 'SAM':
        assert np.all(np.sign(Lb[:, 2]) == np.sign(Lb[0, 2]))   # L circulates about the short (z) axis
    else:
        assert np.all(np.sign(Lb[:, 0]) == np.sign(Lb[0, 0]))   # L circulates about the long (x) axis


@pytest.mark.parametrize('E_ratio', [SAM_E, LAM_E])
def test_psi0_is_a_phase(E_ratio):
    ref = derived(*I_ELL, E_ratio, 10.0, psi_0=90.0)
    for psi in (60.0, 75.0, 105.0, 250.0, 280.0):
        try:
            d = derived(*I_ELL, E_ratio, 10.0, psi_0=psi)
        except ValueError:
            assert E_ratio > 1 / I_ELL[1]                      # only LAM has unreachable psi_0
            continue
        for k in ('P_par_h', 'P_dt_h', 'T_omega_h'):
            assert abs(d[k] / ref[k] - 1) < 1e-6, (psi, k)


def test_unreachable_psi0_raises():
    with pytest.raises(ValueError):
        simulation_for(*I_ELL, LAM_E, 10.0, psi_0=0.0)
