"""
Attitude of the full solver's torque-free motion at any absolute time, without integrating the whole run.

The solver (src/utilities/tumbling_final_2.compute_tumbling_dynamics) integrates the Euler equations with 3-1-3
Euler angles continuously from t = 0 (aphelion) with DOP853, rtol 1e-11, atol 1e-13, starting from Simulation.y0.
The body-frame motion is periodic: after one period T (omega back to omega_0) theta returns and phi, psi have
advanced by fixed increments. Here one period is integrated once, with the same right-hand side, tolerances and
initial state, and kept as a dense-output interpolant. The state at any t is the state at t mod T plus whole-period
increments of phi and psi. For a principal-axis state (E_ratio = 1) the solver's equations reduce to
psi = psi_0 + omega_3 t, theta = theta_0, phi = phi_0.

Frames, as in the solver: R_LI = Rz(lambda_L) Ry(pi/2 - beta_L) maps the L frame (z along L) to the inertial orbit
frame; Ri(phi, theta, psi) maps the L frame to the body frame; body -> inertial is R_LI @ Ri^T, and a vector v given
in the inertial frame is Ri @ R_LI^T @ v in the body frame.
"""
import math

import numpy as np
from scipy.integrate import solve_ivp


def solver_rhs(t, y, I1, I2, I3):
    """Right-hand side of compute_tumbling_dynamics (same floor on sin(theta))."""
    w1, w2, w3, phi, th, psi = y
    st = np.sin(th) if abs(np.sin(th)) > 1e-9 else 1e-9
    dphi = (w1 * np.sin(psi) + w2 * np.cos(psi)) / st
    return [((I2 - I3) * w2 * w3) / I1, ((I3 - I1) * w3 * w1) / I2, ((I1 - I2) * w1 * w2) / I3,
            dphi, w1 * np.cos(psi) - w2 * np.sin(psi), w3 - dphi * np.cos(th)]


def L_to_body(phi, th, psi):
    """Ri for arrays of angles, shape (..., 3, 3) (the matrix Ri of compute_tumbling_dynamics)."""
    c1, s1, c2, s2, c3, s3 = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)
    R = np.empty(np.shape(phi) + (3, 3))
    R[..., 0, 0] = c3 * c1 - s3 * c2 * s1
    R[..., 0, 1] = c3 * s1 + s3 * c2 * c1
    R[..., 0, 2] = s3 * s2
    R[..., 1, 0] = -s3 * c1 - c3 * c2 * s1
    R[..., 1, 1] = -s3 * s1 + c3 * c2 * c1
    R[..., 1, 2] = c3 * s2
    R[..., 2, 0] = s2 * s1
    R[..., 2, 1] = -s2 * c1
    R[..., 2, 2] = c2
    return R


def L_frame_to_inertial(lambda_L_deg, beta_L_deg):
    lam, b = math.radians(lambda_L_deg), math.radians(beta_L_deg)
    Rz = np.array([[math.cos(lam), -math.sin(lam), 0.0], [math.sin(lam), math.cos(lam), 0.0], [0.0, 0.0, 1.0]])
    g = math.pi / 2 - b
    Ry = np.array([[math.cos(g), 0.0, math.sin(g)], [0.0, 1.0, 0.0], [-math.sin(g), 0.0, math.cos(g)]])
    return Rz @ Ry


class TorqueFreeAttitude:
    """y0: the solver's (scaled) initial state [w1, w2, w3, phi, theta, psi]; I = (I1, I2, I3)."""

    def __init__(self, y0, I, lambda_L_deg, beta_L_deg, principal_axis):
        self.y0 = np.asarray(y0, dtype=float)
        self.I = tuple(float(x) for x in I)
        self.R_LI = L_frame_to_inertial(lambda_L_deg, beta_L_deg)
        self.L_hat = self.R_LI[:, 2].copy()
        self.pa = bool(principal_axis)
        w0 = self.y0[:3]
        if self.pa:
            self.w3 = float(self.y0[2])
            self.period_s = 2 * math.pi / abs(self.w3)
            self.dphi = 0.0
            self.dpsi = 2 * math.pi * math.copysign(1.0, self.w3)
            self.sol = None
            return
        dw0 = np.array(solver_rhs(0.0, self.y0, *self.I)[:3])

        def back(t, y, *a):
            return float(np.dot(np.asarray(y[:3]) - w0, dw0))
        back.direction = 1
        t_unit = 2 * math.pi / np.linalg.norm(w0)
        for span in (50, 2000):
            sol = solve_ivp(solver_rhs, (0.0, span * t_unit), self.y0, args=self.I, method='DOP853', rtol=1e-11,
                            atol=1e-13, events=back, dense_output=True)
            te = sol.t_events[0]
            good = te[te > 1e-3 * t_unit]
            if good.size:
                break
        else:
            raise RuntimeError('body-frame period not found within 2000 rotations (state too close to the separatrix)')
        T = float(good[0])
        # integrate exactly one period again so that the dense output ends at T (no extrapolation)
        self.sol = solve_ivp(solver_rhs, (0.0, T), self.y0, args=self.I, method='DOP853', rtol=1e-11, atol=1e-13,
                             dense_output=True).sol
        yT = self.sol(T)
        self.period_s = T
        self.dphi = float(yT[3] - self.y0[3])
        self.dpsi = float(yT[5] - self.y0[5])
        self.theta_return_error = float(abs(yT[4] - self.y0[4]))

    @property
    def mean_phi_rate(self):
        return self.dphi / self.period_s

    def angles(self, t):
        t = np.asarray(t, dtype=float)
        if self.pa:
            return (np.full_like(t, self.y0[3]), np.full_like(t, self.y0[4]), self.y0[5] + self.w3 * t)
        n = np.floor(t / self.period_s)
        tau = t - n * self.period_s
        y = self.sol(tau.ravel()).reshape((6,) + t.shape)
        return y[3] + n * self.dphi, y[4], y[5] + n * self.dpsi

    def omega_body(self, t):
        t = np.asarray(t, dtype=float)
        if self.pa:
            return np.broadcast_to(self.y0[:3], t.shape + (3,)).copy()
        n = np.floor(t / self.period_s)
        y = self.sol((t - n * self.period_s).ravel())
        return y[:3].T.reshape(t.shape + (3,))

    def inertial_to_body(self, t):
        """(..., 3, 3): v_B = M @ v_I."""
        return L_to_body(*self.angles(t)) @ self.R_LI.T

    def body_to_inertial(self, t):
        return np.swapaxes(self.inertial_to_body(t), -1, -2)
