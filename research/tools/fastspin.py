"""
Fast derivation of rotation-state quantities for manifest generation (no thermal solver, no orbit).

Equivalent to spinstate.derived(), which builds the solver's Simulation object and calls the solver's
mean_euler_rates() twice. That function integrates 1000 rotations at rtol 1e-11 because its return-to-start event
is not terminal (src/utilities/tumbling_final_2.py, l. 61-67), which costs 2-4 s per call. Here the same initial
state (copied from Simulation.load_configuration, l. 33-96), the same right-hand side and the same tolerances are
used, but the integration stops after the first return of omega to omega_0, and max|omega| and the nutation range
come from the dense output of that single pass. Results are checked against the solver's Simulation in
research/tests/test_fastspin.py and on a subsample during manifest generation.
"""
import math

import numpy as np
from scipy.integrate import solve_ivp


def _rhs(t, y, I1, I2, I3):
    w1, w2, w3, phi, th, psi = y
    st = np.sin(th) if abs(np.sin(th)) > 1e-9 else 1e-9
    dphi = (w1 * np.sin(psi) + w2 * np.cos(psi)) / st
    return [((I2 - I3) * w2 * w3) / I1, ((I3 - I1) * w3 * w1) / I2, ((I1 - I2) * w1 * w2) / I3,
            dphi, w1 * np.cos(psi) - w2 * np.sin(psi), w3 - dphi * np.cos(th)]


def initial_state(I1, I2, E_ratio, P_phi_h, psi_0, phi_0):
    """Copy of the solver's set-up (simulation.py l. 33-67): unscaled y0 and the PA flag."""
    I3 = 1.0
    psi0, phi0 = math.radians(psi_0), math.radians(phi_0)
    A = math.sin(psi0) ** 2 / I1 + math.cos(psi0) ** 2 / I2
    s2 = (E_ratio - 1.0) / (I3 * A - 1.0)
    if 1.0 < s2 <= 1.0 + 1e-6:
        s2 = 1.0
    if not 0.0 <= s2 <= 1.0:
        raise ValueError(f'E_ratio = {E_ratio} not reachable with psi_0 = {psi_0} deg (sin^2 theta_0 = {s2:.4f})')
    th0 = math.asin(math.sqrt(s2))
    w_phi = 2 * math.pi / (P_phi_h * 3600.0)
    L = w_phi / A
    y0 = [L * math.sin(th0) * math.sin(psi0) / I1, L * math.sin(th0) * math.cos(psi0) / I2, L * math.cos(th0) / I3,
          phi0, th0, psi0]
    pa = bool(np.isclose(E_ratio, 1.0) or th0 < 1e-7)
    return y0, w_phi, A, pa


def derived_fast(I1, I2, E_ratio, P_phi_h, psi_0=90.0, phi_0=0.0, n_samples=2000):
    I = (I1, I2, 1.0)
    y0, w_phi, A, pa = initial_state(I1, I2, E_ratio, P_phi_h, psi_0, phi_0)
    w = np.array(y0[:3])
    if pa:
        P = P_phi_h * A
        return {'mode': 'PA', 'theta0_deg': 0.0, 'E_ratio_check': 1.0, 'P_par_h': P, 'P_dt_h': P, 'P_phi_h': P_phi_h,
                'delta_t_s': P * 3600 / 100, 'P_psi_h': math.inf, 'P_lc_h': P, 'T_omega_h': math.inf,
                'theta_min_deg': 0.0, 'theta_max_deg': 0.0, 'derivation': 'fastspin-v1'}
    dw0 = np.array(_rhs(0.0, y0, *I)[:3])

    def back(t, y, *a):
        return np.dot(np.asarray(y[:3]) - w, dw0)
    back.direction = 1
    back.terminal = 2                    # a spurious root at t ~ 0 may come first; the solver skips t < 1e-3 t_unit
    t_unit = 2 * math.pi / np.linalg.norm(w)
    t_end, sol = 0.0, None
    for span in (50, 1000):              # rotations; extend only if the polhode period is very long (near separatrix)
        sol = solve_ivp(_rhs, (0, span * t_unit), y0, args=I, method='DOP853', rtol=1e-11, atol=1e-13,
                        events=back, dense_output=True)
        te = sol.t_events[0]
        good = te[te > 1e-3 * t_unit]
        if good.size:
            T = float(good[0])
            break
    else:
        raise RuntimeError('period of the body-frame rotation not found within 1000 rotations')
    yT = sol.sol(T)
    mphi, mpsi = (yT[3] - y0[3]) / T, (yT[5] - y0[5]) / T
    scale = w_phi / mphi
    ts = np.linspace(0, T, n_samples)
    Y = sol.sol(ts)
    wmax = np.max(np.linalg.norm(Y[:3], axis=0)) * scale
    ws = w * scale
    Iv = np.array(I)
    Lm = np.linalg.norm(Iv * ws)
    E = 0.5 * np.sum(Iv * ws * ws)
    P_psi = 2 * math.pi / abs(mpsi * scale) / 3600.0 if abs(mpsi) * T > 1e-6 else math.inf
    P_dt = 2 * math.pi / wmax / 3600.0
    th = np.degrees(sol.sol(np.linspace(0, T, 2001))[4])
    return {'mode': 'SAM' if E_ratio < 1.0 / I2 else 'LAM', 'theta0_deg': math.degrees(y0[4]),
            'E_ratio_check': float(2 * E / Lm ** 2), 'P_par_h': float(2 * math.pi * Lm / (2 * E) / 3600.0),
            'P_dt_h': float(P_dt), 'P_phi_h': float(P_phi_h), 'delta_t_s': float(P_dt * 3600 / 100),
            'T_omega_h': float(T / scale / 3600.0), 'P_psi_h': float(P_psi),
            'P_lc_h': float(1.0 / (1.0 / P_phi_h - 1.0 / P_psi)) if math.isfinite(P_psi) else math.nan,
            'theta_min_deg': float(th.min()), 'theta_max_deg': float(th.max()), 'mean_phi_dot_check': 1.0,
            'derivation': 'fastspin-v1'}
