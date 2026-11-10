"""
Physical spin variables <-> repository configuration keys, computed with the repository's own code.

The solver's rotation set-up (src/model/simulation.py, Simulation.load_configuration) takes
I1, I2 (I3 = 1), E_ratio, P_phi_h, phi_0, psi_0. P_phi_h is the mean precession period of the body z axis
about L, except for E_ratio = 1 where the code sets the spin period to P_phi_h * A(psi_0) * I3
(stage-1 audit). Neither is continuous across E_ratio = 1 when I1 != I2.

Here the primary spin variable is P_par = 2*pi*L/(2E): omega . L_hat = 2E/L is conserved exactly in torque-free
motion and equals the true spin rate at E_ratio = 1. Because torque-free motion is scale-invariant in time,
P_par / P_phi depends only on (I1, I2, E_ratio, psi_0), so P_phi_h = P_par_target / ratio.

Every derived quantity is obtained by constructing the repository's Simulation object and integrating with the
repository's mean_euler_rates / max_angular_speed, so the values are those the solver will actually use.
"""
import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.model.simulation import Simulation  # noqa: E402
from src.utilities.tumbling_final_2 import mean_euler_rates  # noqa: E402

BASE = {'a_au': 1.0, 'thermal_inertia': 300.0, 'density': 2000.0, 'specific_heat_capacity': 1000.0,
        'n_skin_depths': 10.0, 'n_layers': 100, 'lambda_L': 0.0, 'beta_L': 90.0, 'implicit_timesteps_per_day': 100}


def simulation_for(I1, I2, E_ratio, P_phi_h, psi_0=90.0, phi_0=0.0, extra=None):
    data = dict(BASE, I1=I1, I2=I2, E_ratio=E_ratio, P_phi_h=P_phi_h, psi_0=psi_0, phi_0=phi_0)
    if extra:
        data.update(extra)
    cfg = SimpleNamespace(config_data=data, temp_solver='tempestovsky_implicit')
    with contextlib.redirect_stdout(io.StringIO()):      # Simulation prints diagnostics
        return Simulation(cfg)


def _euler_rhs(t, y, I1, I2, I3):
    w1, w2, w3, phi, th, psi = y
    st = np.sin(th) if abs(np.sin(th)) > 1e-9 else 1e-9
    dphi = (w1 * np.sin(psi) + w2 * np.cos(psi)) / st
    return [((I2 - I3) * w2 * w3) / I1, ((I3 - I1) * w3 * w1) / I2, ((I1 - I2) * w1 * w2) / I3,
            dphi, w1 * np.cos(psi) - w2 * np.sin(psi), w3 - dphi * np.cos(th)]


def derived(I1, I2, E_ratio, P_phi_h, psi_0=90.0, phi_0=0.0):
    """All rotation timescales and invariants that the solver will use for these configuration values."""
    sim = simulation_for(I1, I2, E_ratio, P_phi_h, psi_0, phi_0)
    I = np.array(sim.I, float)
    w = np.array(sim.y0[:3], float)
    L = np.linalg.norm(I * w)
    E = 0.5 * np.sum(I * w * w)
    out = {'mode': 'PA' if sim.mode == 'Fixed-Axis Rotation' else ('SAM' if E_ratio < I[2] / I2 else 'LAM'),
           'theta0_deg': float(np.degrees(sim.y0[4])), 'E_ratio_check': float(2 * I[2] * E / L ** 2),
           'P_par_h': float(2 * np.pi * L / (2 * E) / 3600.0), 'P_dt_h': float(sim.rotation_period_s / 3600.0),
           'P_phi_h': float(P_phi_h), 'delta_t_s': float(sim.delta_t), 'L': float(L), 'E': float(E)}
    if out['mode'] == 'PA':
        out.update(P_psi_h=np.inf, P_lc_h=out['P_par_h'], T_omega_h=np.inf, theta_min_deg=0.0, theta_max_deg=0.0)
        return out
    mphi, mpsi, T = mean_euler_rates(sim.y0, tuple(I))
    out['mean_phi_dot_check'] = float(mphi * P_phi_h * 3600.0 / (2 * np.pi))   # must be 1
    out['T_omega_h'] = float(T / 3600.0)
    P_psi = 2 * np.pi / abs(mpsi) / 3600.0 if abs(mpsi) * T > 1e-6 else np.inf  # LAM: psi librates, mean ~ 0
    out['P_psi_h'] = float(P_psi)
    out['P_lc_h'] = float(1.0 / (1.0 / P_phi_h - 1.0 / P_psi)) if np.isfinite(P_psi) else np.nan
    s = solve_ivp(_euler_rhs, (0, T), sim.y0, args=tuple(I), method='DOP853', rtol=1e-11, atol=1e-13,
                  t_eval=np.linspace(0, T, 2001))
    out['theta_min_deg'] = float(np.degrees(s.y[4].min()))
    out['theta_max_deg'] = float(np.degrees(s.y[4].max()))
    return out


def p_phi_for_period(I1, I2, E_ratio, P_target_h, convention='P_par', psi_0=90.0):
    """Configuration P_phi_h that gives the requested period under the chosen convention
    ('P_par', 'P_dt', 'P_lc' or 'P_phi'). Uses time-scale invariance: periods scale linearly with P_phi_h."""
    if convention == 'P_phi':
        return float(P_target_h)
    ref = derived(I1, I2, E_ratio, 10.0, psi_0)
    value = ref[convention + '_h']
    if not np.isfinite(value):
        raise ValueError(f'convention {convention} undefined for mode {ref["mode"]}')
    return float(P_target_h * 10.0 / value)


def e_ratio_from_u(u, I1, I2, sep_band=0.005, e_min_frac=1e-3):
    """Normalised excitation u in [0, 1] -> E_ratio in [1 + e_min_frac*range, 1/I1], skipping a band of
    relative width sep_band (of the full range) around the SAM/LAM separatrix 1/I2. u <= 0 gives 1 (principal axis)."""
    if u <= 0.0:
        return 1.0
    lo, hi, sep = 1.0, 1.0 / I1, 1.0 / I2
    rng = hi - lo
    a0, a1 = lo + e_min_frac * rng, sep - 0.5 * sep_band * rng
    b0, b1 = sep + 0.5 * sep_band * rng, hi - 1e-6 * rng
    la, lb = max(a1 - a0, 0.0), max(b1 - b0, 0.0)
    x = u * (la + lb)
    return float(a0 + x if x <= la else b0 + (x - la))
