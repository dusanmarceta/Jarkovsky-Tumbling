'''
Rotation of Apophis through the 2029 Earth encounter (reproduction of Benson et al. 2022, Sec. 3).

Rigid body with principal moments I_l <= I_i <= I_s = 1 (normalized), torque-free except for the
Earth's gravity gradient torque M = 3 mu/R^3 * r_hat x (I r_hat) (body frame), along the JPL
geocentric trajectory. Attitude as a quaternion (body -> J2000 ecliptic), fixed-step RK4.

Conventions: x = long axis (I_l), y = intermediate (I_i), z = short axis (I_s).
Periods: P_psi = period of omega in the body frame, P_phi_s / P_phi_l = mean precession period of the
short / long axis about H (Samarasinha & Mueller 2015; 1/P_phi_s = 1/P_phi_l + 1/P_psi for SAM).
'''
import numpy as np
from numba import njit, prange
from scipy.special import ellipk, ellipj
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from matplotlib.path import Path
import os
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
MU_EARTH = 398600.4418          # km^3/s^2
HOUR = 3600.0

# ---------------------------------------------------------------- torque-free rotation (Landau & Lifshitz)

def free_rotation(I1, I2, I_d, H=1.0):
    '''Jacobi parameters of torque-free rotation for I1 < I2 < I3 = 1 and dynamic inertia I_d = H^2/2T.
    Returns (sam, n, m, amplitudes) with omega(t): SAM (cn, sn, dn), LAM (dn, sn, cn) times the amplitudes,
    argument u = n t, parameter m = k^2.'''
    I3 = 1.0
    M2, E2 = H**2, H**2 / I_d      # M^2 and 2E
    a = E2 * I3 - M2               # 2E I3 - M^2 > 0
    b = M2 - E2 * I1               # > 0
    sam = I_d > I2
    if sam:
        n = np.sqrt((I3 - I2) * b / (I1 * I2 * I3))
        m = (I2 - I1) * a / ((I3 - I2) * b)
        amp = (np.sqrt(a / (I1 * (I3 - I1))), np.sqrt(a / (I2 * (I3 - I2))), np.sqrt(b / (I3 * (I3 - I1))))
    else:
        n = np.sqrt((I2 - I1) * a / (I1 * I2 * I3))
        m = (I3 - I2) * b / ((I2 - I1) * a)
        amp = (np.sqrt(a / (I1 * (I3 - I1))), np.sqrt(b / (I2 * (I2 - I1))), np.sqrt(b / (I3 * (I3 - I1))))
    return sam, n, m, amp


def omega_body(I1, I2, I_d, u, H=1.0):
    sam, n, m, amp = free_rotation(I1, I2, I_d, H)
    sn, cn, dn, _ = ellipj(u, m)
    if sam:
        return np.array([amp[0] * cn, amp[1] * sn, amp[2] * dn])
    return np.array([amp[0] * dn, amp[1] * sn, amp[2] * cn])


def periods(I1, I2, I_d, H=1.0, n_quad=4000):
    '''P_psi, P_phi_s, P_phi_l (same time unit as 1/omega) from mean precession rates over one period.'''
    sam, n, m, amp = free_rotation(I1, I2, I_d, H)
    K = ellipk(m)
    P_psi = 4 * K / n
    u = (np.arange(n_quad) + 0.5) * 4 * K / n_quad
    w = omega_body(I1, I2, I_d, u, H)
    I3 = 1.0
    phi_s = H * (I1 * w[0]**2 + I2 * w[1]**2) / (I1**2 * w[0]**2 + I2**2 * w[1]**2)
    phi_l = H * (I2 * w[1]**2 + I3 * w[2]**2) / (I2**2 * w[1]**2 + I3**2 * w[2]**2)
    return P_psi, 2 * np.pi / phi_s.mean(), 2 * np.pi / phi_l.mean()


@lru_cache(maxsize=None)
def initial_I_d(I1, I2, P_psi, P_phi_s):
    '''SAM dynamic inertia giving the period ratio P_psi/P_phi_s (None if not reachable).'''
    target = P_psi / P_phi_s
    def f(I_d):
        Pp, Ps, _ = periods(I1, I2, I_d)
        return Pp / Ps - target
    lo, hi = I2 + 1e-9 * (1 - I2), 1 - 1e-7
    if f(hi) > 0:          # ratio at the non-tumbling limit already exceeds the target
        return None
    return brentq(f, lo, hi, xtol=1e-13)

# ---------------------------------------------------------------- trajectory

def trajectory(dt, t_half=2 * 86400.0):
    '''Torque factor 3 mu/R^3 [s^-2] and unit vector Earth -> Apophis (J2000 ecliptic) on a half-step grid
    from closest approach - t_half to closest approach + t_half.'''
    a = np.load(os.path.join(HERE, 'geo_vectors_2029.npy'))
    t = (a[:, 0] - a[0, 0]) * 86400.0
    spl = CubicSpline(t, a[:, 1:4])
    tt = np.linspace(t[0], t[-1], 200001)
    t_ca = tt[np.argmin(np.linalg.norm(spl(tt), axis=1))]
    n_steps = int(round(2 * t_half / dt))
    th = t_ca - t_half + 0.5 * dt * np.arange(2 * n_steps + 1)
    r = spl(th)
    R = np.linalg.norm(r, axis=1)
    return 3 * MU_EARTH / R**3, r / R[:, None], n_steps, a[0, 0] + t_ca / 86400.0, R.min()

# ---------------------------------------------------------------- RK4 integration of all runs (numba)

@njit(cache=True)
def _rotmat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@njit(cache=True)
def _deriv(q, w, I, g, rhat, dq, dw):
    A = _rotmat(q)
    rb = A.T @ rhat                       # Earth direction in the body frame
    Ir0, Ir1, Ir2 = I[0] * rb[0], I[1] * rb[1], I[2] * rb[2]
    M0 = g * (rb[1] * Ir2 - rb[2] * Ir1)
    M1 = g * (rb[2] * Ir0 - rb[0] * Ir2)
    M2 = g * (rb[0] * Ir1 - rb[1] * Ir0)
    h0, h1, h2 = I[0] * w[0], I[1] * w[1], I[2] * w[2]
    dw[0] = (h1 * w[2] - h2 * w[1] + M0) / I[0]
    dw[1] = (h2 * w[0] - h0 * w[2] + M1) / I[1]
    dw[2] = (h0 * w[1] - h1 * w[0] + M2) / I[2]
    # q_dot = 1/2 q (x) (0, w)
    dq[0] = 0.5 * (-q[1] * w[0] - q[2] * w[1] - q[3] * w[2])
    dq[1] = 0.5 * (q[0] * w[0] + q[2] * w[2] - q[3] * w[1])
    dq[2] = 0.5 * (q[0] * w[1] + q[3] * w[0] - q[1] * w[2])
    dq[3] = 0.5 * (q[0] * w[2] + q[1] * w[1] - q[2] * w[0])


@njit(parallel=True, cache=True)
def integrate(q0, w0, inertia, g, rhat, dt, n_steps, record_ca):
    '''RK4 for all runs. Returns final q, w and the angular acceleration magnitude at closest approach.'''
    N = q0.shape[0]
    qf = np.empty_like(q0)
    wf = np.empty_like(w0)
    acc_ca = np.zeros(N)
    for k in prange(N):
        q = q0[k].copy(); w = w0[k].copy(); I = inertia[k]
        k1q = np.empty(4); k2q = np.empty(4); k3q = np.empty(4); k4q = np.empty(4)
        k1w = np.empty(3); k2w = np.empty(3); k3w = np.empty(3); k4w = np.empty(3)
        for s in range(n_steps):
            i0 = 2 * s
            _deriv(q, w, I, g[i0], rhat[i0], k1q, k1w)
            if s == record_ca:
                acc_ca[k] = np.sqrt((k1w ** 2).sum())
            _deriv(q + 0.5 * dt * k1q, w + 0.5 * dt * k1w, I, g[i0 + 1], rhat[i0 + 1], k2q, k2w)
            _deriv(q + 0.5 * dt * k2q, w + 0.5 * dt * k2w, I, g[i0 + 1], rhat[i0 + 1], k3q, k3w)
            _deriv(q + dt * k3q, w + dt * k3w, I, g[i0 + 2], rhat[i0 + 2], k4q, k4w)
            q = q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
            w = w + dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
            q = q / np.sqrt((q ** 2).sum())
        qf[k] = q; wf[k] = w
    return qf, wf, acc_ca

# ---------------------------------------------------------------- sampling of the pre-flyby state

def lb_to_vec(lam, beta):
    lam, beta = np.radians(lam), np.radians(beta)
    return np.stack([np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), np.sin(beta)], axis=-1)


def vec_to_lb(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360, np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))


# Fig. 4 projection (orthographic from the south ecliptic pole): pixel = center + 1326.5 cos(beta) * dir(lambda)
FIG4_C, FIG4_S = np.array([1385.846, 1339.068]), 1326.5

def lb_to_fig4(lam, beta):
    r = FIG4_S * np.cos(np.radians(beta))
    lam = np.radians(lam)
    return np.stack([FIG4_C[0] - r * np.cos(lam), FIG4_C[1] - r * np.sin(lam)], axis=-1)


def sample_poles(rng, n):
    region = Path(np.load(os.path.join(HERE, 'pole_region_pixels.npy')))
    out = []
    while sum(len(o) for o in out) < n:
        m = 4 * n
        lam = rng.uniform(0, 360, m)
        beta = np.degrees(np.arcsin(rng.uniform(-1, np.sin(np.radians(-45)), m)))     # uniform on the sphere
        inside = region.contains_points(lb_to_fig4(lam, beta))
        out.append(np.c_[lam[inside], beta[inside]])
    return np.concatenate(out)[:n]


def frame_from(v):
    '''Orthonormal basis (columns) with the first column along v.'''
    v = v / np.linalg.norm(v)
    a = np.array([1.0, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(v, a); e1 /= np.linalg.norm(e1)
    return np.c_[v, e1, np.cross(v, e1)]


def mat_to_quat(A):
    w = np.sqrt(max(0.0, 1 + A[0, 0] + A[1, 1] + A[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + A[0, 0] - A[1, 1] - A[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - A[0, 0] + A[1, 1] - A[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - A[0, 0] - A[1, 1] + A[2, 2])) / 2
    x = np.copysign(x, A[2, 1] - A[1, 2]); y = np.copysign(y, A[0, 2] - A[2, 0]); z = np.copysign(z, A[1, 0] - A[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def initial_states(I1, I2, poles, rng, P_psi_h=263.0, P_phi_s_h=27.38):
    '''Pre-flyby SAM states with the nominal periods; uniform phase of omega (tau) and of the
    attitude about H (phi). Returns q0, w0 (rad/s), I_d and a mask of dynamically viable inertia pairs.'''
    N = len(I1)
    q0 = np.zeros((N, 4)); w0 = np.zeros((N, 3)); Id = np.full(N, np.nan); ok = np.zeros(N, bool)
    for k in range(N):
        I_d = initial_I_d(float(I1[k]), float(I2[k]), P_psi_h, P_phi_s_h)
        if I_d is None:
            continue
        sam, n, m, amp = free_rotation(I1[k], I2[k], I_d)
        P1 = 4 * ellipk(m) / n                       # P_psi for H = 1
        scale = P1 / (P_psi_h * HOUR)                # H such that P_psi = 263 h
        u = rng.uniform(0, 4 * ellipk(m))
        w = omega_body(I1[k], I2[k], I_d, u) * scale
        h = np.array([I1[k], I2[k], 1.0]) * w
        A0 = frame_from(lb_to_vec(*poles[k])) @ frame_from(h).T      # maps h_body to the pole
        ph = rng.uniform(0, 2 * np.pi)
        p = lb_to_vec(*poles[k]); c, s = np.cos(ph), np.sin(ph)
        Kx = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
        Rp = np.eye(3) + s * Kx + (1 - c) * Kx @ Kx                  # rotation about the pole by ph
        q0[k] = mat_to_quat(Rp @ A0); w0[k] = w; Id[k] = I_d; ok[k] = True
    return q0, w0, Id, ok

# ---------------------------------------------------------------- post-flyby state

def final_state(qf, wf, I1, I2):
    N = len(I1)
    I = np.c_[I1, I2, np.ones(N)]
    h_b = I * wf
    H = np.linalg.norm(h_b, axis=1)
    T2 = (I * wf**2).sum(1)                      # 2T
    out = dict(H=H, I_d=H**2 / T2, omega_e=T2 / H)
    A = np.array([_rotmat(q) for q in qf])
    h_i = np.einsum('kij,kj->ki', A, h_b)
    out['lam'], out['beta'] = vec_to_lb(h_i)
    sam = out['I_d'] > I2
    out['sam'] = sam
    out['Id_scaled'] = np.where(sam, (out['I_d'] - I2) / (1 - I2), (out['I_d'] - I2) / (I2 - I1))
    P = np.array([periods(I1[k], I2[k], out['I_d'][k], 1.0) for k in range(N)])
    # periods() uses H = 1 with the given I_d; real periods scale as 1/H (I_d fixed)
    out['P_psi_h'], out['P_phi_s_h'], out['P_phi_l_h'] = (P / H[:, None] / HOUR).T
    # Euler angles of the short axis (z convention) for the Tempestovsky input
    out['theta_deg'] = np.degrees(np.arccos(np.clip(h_b[:, 2] / H, -1, 1)))
    out['psi_deg'] = np.degrees(np.arctan2(h_b[:, 0], h_b[:, 1])) % 360
    return out
