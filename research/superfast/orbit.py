"""
Keplerian orbit in the full solver's convention (src/utilities/utils.sun_direction): time t is counted from
aphelion (mean anomaly M = pi + n t), the asteroid is at r (cos f, sin f, 0) in the orbit frame, and the unit
vector towards the Sun is s = -(cos f, sin f, 0). The solver solves Kepler's equation to 1e-6 rad; here it is
solved to machine precision.
"""
import math

import astropy.constants as const
import numpy as np

AU = const.au.value
GM = const.GM_sun.value


def mean_motion(a_au):
    return math.sqrt(GM / (a_au * AU) ** 3)


def eccentric_anomaly(M, e):
    """Solution of E - e sin E = M (Newton from Danby's starting value; M in [0, 2 pi))."""
    M = np.asarray(M, dtype=float)
    E = M + 0.85 * e * np.sign(np.sin(M))
    for _ in range(60):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < 1e-15):
            break
    return E


def state_at(t, a_au, e):
    """(s_I (..., 3) towards the Sun, r_au, f) at time t [s] after aphelion."""
    t = np.asarray(t, dtype=float)
    M = math.pi + mean_motion(a_au) * t
    E = eccentric_anomaly(np.mod(M, 2 * math.pi), e)
    f = np.arctan2(math.sqrt(1 - e * e) * np.sin(E), np.cos(E) - e)
    r = a_au * (1 - e * e) / (1 + e * np.cos(f))
    s = np.stack([-np.cos(f), -np.sin(f), np.zeros_like(f)], axis=-1)
    return s, r, f


def time_of_true_anomaly(f, a_au, e):
    """Time after aphelion [s], in [0, P), at which the true anomaly is f."""
    f = np.asarray(f, dtype=float)
    E = 2 * np.arctan2(math.sqrt(1 - e) * np.sin(f / 2), math.sqrt(1 + e) * np.cos(f / 2))
    M = E - e * np.sin(E)
    return np.mod(M - math.pi, 2 * math.pi) / mean_motion(a_au)


def orbital_period(a_au):
    return 2 * math.pi / mean_motion(a_au)
