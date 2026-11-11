"""
Compiled kernels of the two-scale method. All column updates call the full solver's own Crank-Nicolson column step
(src/model/solvers/tempestovsky_implicit.implicit_column_step: same finite-volume surface node, same safeguarded
Newton solve, same backward-Euler fallback), with an insulating bottom as in the solver's orbit loop.
"""
import math

import numpy as np
from numba import njit, prange

from src.model.solvers.tempestovsky_implicit import elimination_coefficients, implicit_column_step


@njit(parallel=True, cache=True)
def fast_window(T, normals, sB, uB, flux, sink, const1, rad_coeff, a, theta, q_prev, dark, Tz_sum):
    """
    Advance every facet column len(sB) steps at a frozen heliocentric position (in place).

    T (nf, nl): temperatures; sB, uB (n, 3): body-frame unit vectors towards the Sun and along s x z_orbit at the
    end of each step; flux: absorbed flux at normal incidence [W m^-2]; sink (nf,): constant surface heat sink
    [W m^-2] (the seasonal heat flux into the ground); q_prev (nf,): absorbed flux at the start (updated on exit);
    dark (nf,): columns held at T = 0 (the reduced model's fixed solution for a facet that is never lit);
    Tz_sum (nf, nl): incremented by the column temperatures after every step (for the deep-temperature correction).
    Returns per-facet sums over the n end-of-step states, columns: T_s, T_s^4, T_s^4 (n.s_B), T_s^4 (n.u_B), q,
    and the number of backward-Euler fallback steps.
    """
    nf, nl = T.shape
    n = sB.shape[0]
    cq, cinv = elimination_coefficients(nl, a, theta, False)
    bq, binv = elimination_coefficients(nl, a, 1.0, False)
    sums = np.zeros((nf, 5))
    fb = np.zeros(nf, np.int64)
    for i in prange(nf):
        if dark[i]:
            continue
        p = np.empty(nl)
        col = T[i]
        nx, ny, nz = normals[i, 0], normals[i, 1], normals[i, 2]
        qo = q_prev[i]
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        for k in range(n):
            mu = nx * sB[k, 0] + ny * sB[k, 1] + nz * sB[k, 2]
            qn = flux * mu if mu > 0.0 else 0.0
            fb[i] += implicit_column_step(col, col, p, cq, cinv, bq, binv,
                                          (theta * qn + (1.0 - theta) * qo - sink[i]) * const1,
                                          (qn - sink[i]) * const1, rad_coeff, a, theta, False, 0.0)
            ts = col[0]
            t4 = ts * ts * ts * ts
            s0 += ts
            s1 += t4
            s2 += t4 * mu
            s3 += t4 * (nx * uB[k, 0] + ny * uB[k, 1] + nz * uB[k, 2])
            s4 += qn
            qo = qn
            for z in range(nl):
                Tz_sum[i, z] += col[z]
        q_prev[i] = qo
        sums[i, 0] = s0
        sums[i, 1] = s1
        sums[i, 2] = s2
        sums[i, 3] = s3
        sums[i, 4] = s4
    return sums, fb.sum()


@njit(parallel=True, cache=True)
def seasonal_orbit(T, qnet, kap, const1, rad_coeff, a, theta, Ts_out, Tz_sum):
    """
    One orbit of the seasonal columns (in place). qnet (nf, ns + 1): absorbed tumbling-averaged flux at the step
    boundaries (index 0 = start of the orbit); kap (nf, ns + 1): emission enhancement of the fast fluctuation,
    <T^4> / <T>^4 >= 1 (the mean emission is eps sigma kap T^4; 1 in the first pass); Ts_out (nf, ns): surface
    temperature after each step; Tz_sum (nf, nl): sum of the column temperatures over the steps (for the
    deep-temperature correction). Returns the number of backward-Euler fallback steps.
    """
    nf, nl = T.shape
    ns = qnet.shape[1] - 1
    cq, cinv = elimination_coefficients(nl, a, theta, False)
    bq, binv = elimination_coefficients(nl, a, 1.0, False)
    fb = np.zeros(nf, np.int64)
    for i in prange(nf):
        p = np.empty(nl)
        col = T[i]
        for z in range(nl):
            Tz_sum[i, z] = 0.0
        for k in range(ns):
            fb[i] += implicit_column_step(col, col, p, cq, cinv, bq, binv,
                                          (theta * qnet[i, k + 1] + (1.0 - theta) * qnet[i, k]) * const1,
                                          qnet[i, k + 1] * const1, rad_coeff * kap[i, k + 1], a, theta, False, 0.0)
            Ts_out[i, k] = col[0]
            for z in range(nl):
                Tz_sum[i, z] += col[z]
    return fb.sum()


@njit(parallel=True, cache=True)
def torus_insolation(normals, M, cz):
    """
    g[i, j]: time average of max(0, n_i . s_B) over the torque-free motion for the Sun at polar angle
    arccos(cz[j]) from L. M (nt, 3, 3): L-frame -> body matrices at phi = 0 for nt body phases spread uniformly
    over one body period; the average over the precession angle phi is done analytically:
    <max(0, A cos u + B)>_u = B (B >= A), 0 (B <= -A), (B arccos(-B/A) + sqrt(A^2 - B^2)) / pi otherwise.
    """
    nf = normals.shape[0]
    nt = M.shape[0]
    nzt = cz.shape[0]
    g = np.zeros((nf, nzt))
    for i in prange(nf):
        n0, n1, n2 = normals[i, 0], normals[i, 1], normals[i, 2]
        for t in range(nt):
            mx = M[t, 0, 0] * n0 + M[t, 1, 0] * n1 + M[t, 2, 0] * n2
            my = M[t, 0, 1] * n0 + M[t, 1, 1] * n1 + M[t, 2, 1] * n2
            mz = M[t, 0, 2] * n0 + M[t, 1, 2] * n1 + M[t, 2, 2] * n2
            rho = math.sqrt(mx * mx + my * my)
            for j in range(nzt):
                c = cz[j]
                A = math.sqrt(max(0.0, 1.0 - c * c)) * rho
                B = c * mz
                if B >= A:
                    h = B
                elif B <= -A:
                    h = 0.0
                else:
                    h = (B * math.acos(-B / A) + math.sqrt(A * A - B * B)) / math.pi
                g[i, j] += h
        for j in range(nzt):
            g[i, j] /= nt
    return g
