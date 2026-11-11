"""
Two-scale Yarkovsky drift for tumbling (and principal-axis) asteroids on eccentric orbits.

Scale separation. Rotation (seconds to hours) is much faster than the orbit (P_rot / P_orb <= 0.025 for every
super-fast campaign state), and the diurnal skin depth is much smaller than the seasonal one. The surface
temperature is split into a slow part, which carries the seasonal (orbital) thermal memory down to several
seasonal skin depths, and a fast part confined to a thin layer, which responds to the rotation at a frozen
heliocentric position.

  1. Tumbling-averaged insolation. For the Sun at polar angle zeta from L, the time average of max(0, n.s) over
     the torque-free motion is tabulated per facet in cos(zeta): one body period sampled uniformly, the
     precession angle averaged analytically (non-resonant motion).
  2. Seasonal problem. Each facet column (depth and seasonal resolution of the full solver's domain) is driven by
     the tumbling-averaged absorbed flux along the Kepler orbit, minus the extra mean emission of the fast
     fluctuation, eps sigma (<T^4> - <T>^4), from step 3 (zero in the first pass). Its periodic solution gives, at
     each orbital position, the mean surface temperature and the net heat flux into the ground, q_s.
  3. Fast problem (the IAC 2026 frozen-Sun method, extended). At N_loc positions (uniform in true anomaly), the
     Sun is frozen and the body keeps its exact torque-free attitude in absolute time; each facet column (a few
     skin depths of the slowest rotational period, the full solver's layer thickness and time step) starts from
     the seasonal mean temperature and carries the seasonal flux q_s as a constant surface sink. Windows of whole
     body periods with matching precession phase are averaged until the local drift converges. The window means
     give the local recoil force and <T^4> - <T>^4.
  4. Steps 2-3 are repeated (coupling iterations). The local da/dt (the full solver's own Gauss formula, radial
     and transverse terms) is averaged over time along the orbit (weights r^2 for nodes uniform in f).

mode = 'diurnal' reproduces the IAC 2026 method (no seasonal memory: no sink, never-lit facets at T = 0).
"""
import contextlib
import io
import math
import time
import types
from dataclasses import asdict, dataclass

import astropy.constants as const
import numpy as np

from src.model.simulation import Simulation
from src.model.solvers.tempestovsky import geometry_for_yarko
from tempest import read_shape_model

from . import orbit
from .attitude import L_to_body, TorqueFreeAttitude
from .kernels import fast_window, seasonal_orbit, torus_insolation

SIGMA = const.sigma_sb.value
C_LIGHT = const.c.value
L_SUN = const.L_sun.value
AU = const.au.value
GM = const.GM_sun.value


@dataclass
class Numerics:
    mode: str = 'twoscale'            # 'twoscale' (diurnal + seasonal) or 'diurnal' (frozen Sun only)
    n_loc: int = 24                   # orbital positions, uniform in true anomaly
    fast_depths: float = 8.0          # fast column depth, in skin depths of the slowest rotational period
    window_phi_tol_deg: float = 5.0   # precession-phase mismatch accepted for an averaging window
    window_max_periods: int = 100
    warm_windows: int = 2
    max_windows: int = 80
    tol_rel: float = 2e-4             # window-to-window change of the local da/dt
    tol_abs_frac: float = 1e-4        # ... or this fraction of the drift scale of the total emitted force
    tol_energy: float = 2e-4          # and |<absorbed> - eps sigma <T^4> - sink| / <absorbed> (area weighted)
    deep_correction: bool = True      # after each window shift each column so that its time mean is uniform
    consecutive: int = 2
    seasonal_steps: int = 4000
    seasonal_dz_over_ls: float = 0.05
    seasonal_tol_K: float = 1e-3
    seasonal_min_orbits: int = 3
    seasonal_max_orbits: int = 80
    coupling_iterations: int = 2
    n_tau: int = 256
    n_cz: int = 721
    theta: float = 0.5                # Crank-Nicolson in the fast columns
    chunk_steps: int = 100_000
    dt_s: float = None                # overrides (reference comparisons); None = the full solver's value
    dz_m: float = None
    n_fast_layers: int = None
    mass_kg: float = None
    dark_rule: bool = None            # T = 0 for never-lit facets; default True in 'diurnal' mode only


def simulation_from_config(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return Simulation(types.SimpleNamespace(config_data=dict(cfg), temp_solver=cfg.get('temp_solver', 'tempestovsky_implicit')))


class TwoScaleYarkovsky:
    def __init__(self, config, mesh_path, numerics=None):
        t0 = time.perf_counter()
        self.num = numerics or Numerics()
        self.sim = sim = simulation_from_config(config)
        with contextlib.redirect_stdout(io.StringIO()):
            facets = read_shape_model(str(mesh_path), 1, 2, 1, False)
        volume, normals, areas = geometry_for_yarko(facets)
        self.normals = np.ascontiguousarray(normals, dtype=np.float64)
        self.areas = np.asarray(areas, dtype=np.float64)
        self.mass = float(self.num.mass_kg or volume * sim.density)
        self.pa = sim.mode == 'Fixed-Axis Rotation'
        self.att = TorqueFreeAttitude(sim.y0, sim.I, sim.lambda_L, sim.beta_L, self.pa)
        self.rho, self.cp, self.eps, self.albedo = sim.density, sim.specific_heat_capacity, sim.emissivity, sim.albedo
        self.kappa = sim.thermal_diffusivity
        self.dt = float(self.num.dt_s or sim.delta_t)
        self.dz = float(self.num.dz_m or sim.layer_thickness)
        # rotational time scales
        T = self.att.period_s
        if self.pa:
            periods = {'P_rot_s': T}
        else:
            periods = {'P_dt_s': sim.rotation_period_s, 'T_body_s': T, 'P_phi_s': 2 * math.pi / abs(self.att.mean_phi_rate)}
            if abs(self.att.dpsi) > 1e-9:
                periods['P_psi_s'] = 2 * math.pi * T / abs(self.att.dpsi)
        self.periods = periods
        self.P_slow = max(periods.values())
        self.n_fast = int(self.num.n_fast_layers or math.ceil(self.num.fast_depths * math.sqrt(self.kappa * self.P_slow / (2 * math.pi)) / self.dz))
        hc = self.rho * self.cp * self.dz
        self.c1, self.rad, self.a = self.dt / hc, self.eps * SIGMA * self.dt / hc, self.kappa * self.dt / self.dz ** 2
        self.coef = 2.0 / 3.0 * self.eps * SIGMA / C_LIGHT
        self.window = self._window()
        # orbit and positions (uniform in f; the time average of g(f) is sum r_k^2 g_k / sum r_k^2)
        self.P_orb = orbit.orbital_period(sim.a_au)
        f = 2 * math.pi * np.arange(self.num.n_loc) / self.num.n_loc
        self.t_loc = orbit.time_of_true_anomaly(f, sim.a_au, sim.ecc)
        self.s_loc, self.r_loc, f_chk = orbit.state_at(self.t_loc, sim.a_au, sim.ecc)
        assert np.allclose(np.cos(f_chk - f), 1.0, atol=1e-12)
        self.f_loc = f
        self.w_loc = self.r_loc ** 2 / np.sum(self.r_loc ** 2)
        self.setup_s = time.perf_counter() - t0

    # -- helpers ---------------------------------------------------------------------------------------------
    def _window(self):
        T = self.att.period_s
        if self.pa:
            return {'periods': 1, 'steps': int(round(T / self.dt)), 'phi_mismatch_deg': 0.0, 'length_s': T}
        best = None
        for m in range(1, self.num.window_max_periods + 1):
            x = (m * self.att.dphi / (2 * math.pi)) % 1.0
            mis = min(x, 1 - x) * 360.0
            if best is None or mis < best[1]:
                best = (m, mis)
            if mis <= self.num.window_phi_tol_deg:
                best = (m, mis)
                break
        m, mis = best
        return {'periods': m, 'steps': int(round(m * T / self.dt)), 'phi_mismatch_deg': mis, 'length_s': m * T}

    def flux_at(self, r_au):
        return L_SUN * (1 - self.albedo) / (4 * math.pi * (np.asarray(r_au) * AU) ** 2)

    def solver_dadt(self, R, B, f, r_au):
        """calculate_yarkovsky of src/model/solvers/tempestovsky.py for given R = -F.r_rad and B = F.r_trans."""
        s = self.sim
        return (2 * s.mean_motion * (s.a_au * AU) ** 2 / GM
                * (R * s.a_au * AU * s.ecc * math.sin(f) / math.sqrt(1 - s.ecc ** 2)
                   + B * s.a_au ** 2 * AU * math.sqrt(1 - s.ecc ** 2) / r_au) / self.mass)

    def torus_table(self):
        cz = np.linspace(-1.0, 1.0, self.num.n_cz)
        if self.pa:
            M = L_to_body(np.zeros(1), np.full(1, self.att.y0[4]), np.full(1, self.att.y0[5]))
        else:
            tau = (np.arange(self.num.n_tau) + 0.5) * self.att.period_s / self.num.n_tau
            _, th, psi = self.att.angles(tau)
            M = L_to_body(np.zeros_like(th), th, psi)
        self.cz_grid = cz
        self.g = torus_insolation(self.normals, np.ascontiguousarray(M), cz)
        return self.g

    def g_at(self, cz):
        """Tumbling-averaged max(0, n.s) per facet at cos(zeta) values cz (linear interpolation): (nf, len(cz))."""
        cz = np.atleast_1d(cz)
        x = (cz + 1.0) / 2.0 * (self.num.n_cz - 1)
        j = np.clip(np.floor(x).astype(int), 0, self.num.n_cz - 2)
        w = x - j
        return self.g[:, j] * (1 - w) + self.g[:, j + 1] * w

    # -- seasonal problem ------------------------------------------------------------------------------------
    def seasonal(self, D4_loc=None, T_start=None):
        num, sim = self.num, self.sim
        ns = int(num.seasonal_steps)
        dts = self.P_orb / ns
        l_s = math.sqrt(self.kappa * self.P_orb / (2 * math.pi))
        depth = sim.n_skin_depths * sim.skin_depth                # the full solver's domain depth
        nl = max(10, int(math.ceil(depth / (num.seasonal_dz_over_ls * l_s))))
        dzs = depth / nl
        tj = np.arange(ns + 1) * dts
        s, r, f = orbit.state_at(tj, sim.a_au, sim.ecc)
        qabs = self.flux_at(r)[None, :] * self.g_at(np.einsum('ni,i->n', s, self.att.L_hat))
        qnet = qabs.copy()
        if D4_loc is not None:
            fl = np.concatenate([self.f_loc, [self.f_loc[0] + 2 * math.pi]])
            D4 = np.concatenate([D4_loc, D4_loc[:1]], axis=0)               # (n_loc + 1, nf), periodic in f
            fm = np.mod(f, 2 * math.pi)
            for i in range(qnet.shape[0]):
                qnet[i] -= self.eps * SIGMA * np.interp(fm, fl, D4[:, i])
        hc = self.rho * self.cp * dzs
        c1, rad, a = dts / hc, self.eps * SIGMA * dts / hc, self.kappa * dts / dzs ** 2
        T_q = (self.flux_at(sim.a_au * (1 - sim.ecc)) / (self.eps * SIGMA)) ** 0.25
        theta = 0.5 if rad * T_q ** 3 <= 1.0 else 1.0
        nf = qnet.shape[0]
        if T_start is not None and T_start.shape == (nf, nl):
            T = T_start.copy()
        else:
            T = np.repeat((np.maximum(qnet.mean(axis=1), 1e-3) / (self.eps * SIGMA)) ** 0.25, nl).reshape(nf, nl).copy()
        Ts = np.zeros((nf, ns))
        Tz = np.zeros((nf, nl))
        prev = None
        fb = 0
        for n_orb in range(1, num.seasonal_max_orbits + 1):
            fb += seasonal_orbit(T, qnet, c1, rad, a, theta, Ts, Tz)
            T += (Ts.mean(axis=1)[:, None] - Tz / ns)               # deep-temperature correction (as the solver)
            change = np.inf if prev is None else float(np.abs(Ts - prev).max())
            prev = Ts.copy()
            if n_orb >= num.seasonal_min_orbits and change < num.seasonal_tol_K:
                break
        # surface temperature at step boundaries 0..ns (index 0 = end of the previous orbit)
        Tsb = np.concatenate([Ts[:, -1:], Ts], axis=1)
        x = self.t_loc / dts
        j = np.clip(np.floor(x).astype(int), 0, ns - 1)
        w = x - j
        Tbar = (Tsb[:, j] * (1 - w) + Tsb[:, j + 1] * w).T                       # (n_loc, nf)
        qn = (qnet[:, j] * (1 - w) + qnet[:, j + 1] * w).T
        qs = qn - self.eps * SIGMA * Tbar ** 4                                    # net heat flux into the ground
        return {'Tbar': Tbar, 'q_s': qs, 'theta': theta, 'orbits': n_orb, 'last_change_K': change, 'fallbacks': int(fb),
                'layers': nl, 'dz_m': dzs, 'steps_per_orbit': ns, 'T_columns': T}

    # -- fast problem at one frozen position ------------------------------------------------------------------
    def local(self, k, T_init, sink, dark=None, max_windows=None):
        num = self.num
        s_I, r_au, f, t0 = self.s_loc[k], float(self.r_loc[k]), float(self.f_loc[k]), float(self.t_loc[k])
        u_I = np.cross(s_I, [0.0, 0.0, 1.0])
        flux = float(self.flux_at(r_au))
        nf = self.normals.shape[0]
        T = np.ascontiguousarray(T_init, dtype=np.float64)
        dark = np.zeros(nf, dtype=np.bool_) if dark is None else np.asarray(dark, dtype=np.bool_)
        T[dark] = 0.0
        sink = np.ascontiguousarray(sink, dtype=np.float64)
        M0 = self.att.inertial_to_body(np.array([t0]))[0]
        q_prev = flux * np.maximum(self.normals @ (M0 @ s_I), 0.0)
        nW = self.window['steps']
        t_cur, hist, steps, fb, conv_count = t0, [], 0, 0, 0
        lit = ~dark
        aq = None
        for w in range(max_windows or num.max_windows):
            acc = np.zeros((nf, 5))
            Tz = np.zeros_like(T)
            done = 0
            while done < nW:
                n = min(num.chunk_steps, nW - done)
                tt = t_cur + self.dt * np.arange(1, n + 1)
                M = self.att.inertial_to_body(tt)
                sB = np.ascontiguousarray(np.einsum('nij,j->ni', M, s_I))
                uB = np.ascontiguousarray(np.einsum('nij,j->ni', M, u_I))
                sums, nfb = fast_window(T, self.normals, sB, uB, flux, sink, self.c1, self.rad, self.a, num.theta,
                                        q_prev, dark, Tz)
                acc += sums
                fb += int(nfb)
                done += n
                t_cur += n * self.dt
            steps += nW
            m = acc / nW
            Fs = self.coef * np.dot(self.areas, m[:, 2])          # F.s_B (F along the emitting normals)
            Fu = self.coef * np.dot(self.areas, m[:, 3])
            dadt = self.solver_dadt(Fs, -Fu, f, r_au)            # R = -F.r_rad = F.s_B; B = F.r_trans = -F.u_B
            scale = 2.0 / self.sim.mean_motion * self.coef * np.dot(self.areas, m[:, 1]) / self.mass
            # heat still going into (or out of) the columns: zero in the attractor
            imb = m[:, 4] - self.eps * SIGMA * m[:, 1] - sink
            aq = float(np.dot(self.areas[lit], m[lit, 4]) / self.areas[lit].sum()) if lit.any() else 0.0
            energy = float(np.dot(self.areas[lit], np.abs(imb[lit])) / self.areas[lit].sum()) / max(aq, 1e-30)
            hist.append({'dadt': dadt, 'scale': scale, 'm': m, 'energy': energy})
            if num.deep_correction and energy > num.tol_energy:
                # in the periodic attractor (insulating bottom, sink at the surface) the time-mean temperature is
                # the same at every depth: remove the slow relaxation of the deep layers (as the solver does per
                # orbit). Only while heat is still going into the columns, so the attractor itself is not perturbed.
                T[lit] += (m[lit, 0][:, None] - Tz[lit] / nW)
            if w >= num.warm_windows and len(hist) >= 2:
                d = abs(hist[-1]['dadt'] - hist[-2]['dadt'])
                ok = d <= num.tol_rel * abs(dadt) + num.tol_abs_frac * scale and energy <= num.tol_energy
                conv_count = conv_count + 1 if ok else 0
                if conv_count >= num.consecutive:
                    break
        last = hist[-2:]
        dadt = float(np.mean([h['dadt'] for h in last]))
        m = np.mean([h['m'] for h in last], axis=0)
        D4 = m[:, 1] - m[:, 0] ** 4
        return {'k': k, 'f_deg': math.degrees(f), 'r_au': r_au, 't_s': t0, 'dadt': dadt, 'windows': len(hist),
                'scale': float(hist[-1]['scale']),
                'converged': conv_count >= num.consecutive, 'steps': steps, 'fallbacks': fb,
                'last_change': abs(hist[-1]['dadt'] - hist[-2]['dadt']) if len(hist) > 1 else None,
                'energy_imbalance': hist[-1]['energy'],
                'D4': D4, 'T_mean': m[:, 0], 'q_mean': m[:, 4], 'T_final': T}

    # -- full run --------------------------------------------------------------------------------------------
    def run(self, locations=None, log=None):
        t_start = time.perf_counter()
        num = self.num
        ks = list(range(num.n_loc)) if locations is None else list(locations)
        self.torus_table()
        nf = self.normals.shape[0]
        out = {'mode': num.mode, 'iterations': []}
        if num.mode == 'diurnal':
            dark_rule = True if num.dark_rule is None else num.dark_rule
            res = []
            for k in ks:
                gk = self.g_at(self.s_loc[k] @ self.att.L_hat)[:, 0]
                dark = (gk <= 0.0) if dark_rule else np.zeros(nf, bool)
                Teq = np.maximum(self.flux_at(self.r_loc[k]) * gk / (self.eps * SIGMA), 1.0) ** 0.25
                T0 = np.repeat(Teq, self.n_fast).reshape(nf, self.n_fast)
                res.append(self.local(k, T0, np.zeros(nf), dark))
                if log:
                    log(f'  k={k:3d} f={res[-1]["f_deg"]:7.2f} deg  da/dt={res[-1]["dadt"]:+.6e} m/s  windows={res[-1]["windows"]}')
            out['iterations'].append(self._orbit_summary(res, ks))
        else:
            seas = self.seasonal()
            prev = None
            for it in range(num.coupling_iterations):
                res = []
                for k in ks:
                    if prev is None:
                        T0 = np.repeat(seas['Tbar'][k], self.n_fast).reshape(nf, self.n_fast)
                    else:
                        T0 = prev[ks.index(k)]['T_final'] + (seas['Tbar'][k] - prev_Tbar[k])[:, None]
                    res.append(self.local(k, T0, seas['q_s'][k]))
                    if log:
                        log(f'  it={it} k={k:3d} f={res[-1]["f_deg"]:7.2f} deg  da/dt={res[-1]["dadt"]:+.6e} m/s  '
                            f'windows={res[-1]["windows"]}')
                summ = self._orbit_summary(res, ks)
                summ['seasonal'] = {k: v for k, v in seas.items() if k not in ('Tbar', 'q_s', 'T_columns')}
                out['iterations'].append(summ)
                if it + 1 < num.coupling_iterations:
                    D4 = np.zeros((num.n_loc, nf))
                    for r in res:
                        D4[r['k']] = r['D4']
                    prev, prev_Tbar = res, seas['Tbar']
                    seas = self.seasonal(D4, T_start=seas['T_columns'])
        last = out['iterations'][-1]
        out.update(dadt_m_s=last['dadt_m_s'], per_location=last['per_location'],
                   coupling_change=(abs(out['iterations'][-1]['dadt_m_s'] - out['iterations'][-2]['dadt_m_s'])
                                    if len(out['iterations']) > 1 else None),
                   wall_s=time.perf_counter() - t_start + self.setup_s, setup_s=self.setup_s,
                   numerics=asdict(num), window=self.window, periods=self.periods, n_fast_layers=self.n_fast,
                   dt_s=self.dt, dz_m=self.dz, mass_kg=self.mass, principal_axis=self.pa,
                   total_fast_steps=int(sum(p['steps'] for it in out['iterations'] for p in it['per_location'])))
        return out

    def _orbit_summary(self, res, ks):
        d = np.array([r['dadt'] for r in res])
        if len(ks) == self.num.n_loc:
            w = self.w_loc[ks]
            dadt = float(np.sum(w * d) / np.sum(w))
        else:
            dadt = None
        per = [{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()
                if k not in ('D4', 'T_mean', 'q_mean', 'T_final')} for r in res]
        return {'dadt_m_s': dadt, 'per_location': per,
                'all_converged': bool(all(r['converged'] for r in res)), 'fallbacks': int(sum(r['fallbacks'] for r in res))}
