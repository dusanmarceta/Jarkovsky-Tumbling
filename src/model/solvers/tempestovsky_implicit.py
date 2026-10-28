'''
Implicit version of the "tempestovsky" solver.

The physics (surface energy balance, 1D heat conduction, boundary conditions, Yarkovsky drift)
is identical to YarkovskySolver in tempestovsky.py. Only the time integration of the heat
equation is different: instead of the explicit FTCS scheme, which is stable only for
kappa*dt/dz^2 <= 0.5 and therefore forces very small timesteps, the theta scheme is used
(theta = 1: backward Euler, theta = 0.5: Crank-Nicolson). It is unconditionally stable, so the
timestep can be chosen by the accuracy needed to resolve the rotation, not by the layer thickness.

The insolation is time-centred in the same way as the conduction and emission terms
(theta*S_new + (1-theta)*S_old), so with theta = 0.5 the scheme is second order in time.

Each timestep, per facet, is one tridiagonal solve. The interior rows are linear, so they are
eliminated from the bottom up, which leaves a single scalar nonlinear equation for the surface
temperature (because of the T^4 emission term). That is solved exactly with Newton's method,
and the subsurface temperatures are then obtained by back substitution.

Config options (only used by this solver):
    implicit_timesteps_per_day: number of timesteps per rotation (default 200)
    implicit_theta:             0.5 (Crank-Nicolson, default) ... 1.0 (backward Euler)
    linearized_emission:        true replaces eps*sigma*T^4 by eps*sigma*(T0^4 + 4*T0^3*(T - T0)) in the
                                surface energy balance and in the Yarkovsky force (default false).
                                Used to compare with the linear Yarkovsky theory (Vokrouhlicky).
    linearization_temperature:  T0: "T_av" (default), "T_star" or a number in K
'''

import numpy as np
from numba import jit, prange
import astropy.constants as const

from .tempestovsky import YarkovskySolver, calculate_secondary_radiation


@jit(nopython=True, cache=True)
def elimination_coefficients(n, a, theta, bottom_fixed):
    '''
    Coefficients of the bottom-up elimination, T[i] = p[i] + q[i] * T[i-1], that do not depend on
    the temperatures (q and the reciprocal pivots inv_d). They are the same for every facet and every timestep.
    '''
    ta = theta * a
    q = np.empty(n)
    inv_d = np.ones(n)
    q[n - 1] = 0.0 if bottom_fixed else 1.0
    for i in range(n - 2, 0, -1):
        inv_d[i] = 1.0 / (1.0 + 2.0 * ta - ta * q[i + 1])
        q[i] = ta * inv_d[i]
    return q, inv_d


@jit(nopython=True, cache=True)
def surface_temperature(T_old, p, q, inv_d, source, rad_coeff, a, theta, T_ref):
    '''
    Eliminate the interior rows of one column from the bottom up (T[i] = p[i] + q[i] * T[i-1], fills p)
    and solve the surface energy balance for the new surface temperature. T_old is only read, and
    p[n-1] must already hold the bottom boundary. Returns the new surface temperature.
    '''
    n = T_old.shape[0]
    ta = theta * a
    ea = (1.0 - theta) * a

    # Eliminate interior rows from the bottom up: T[i] = p[i] + q[i] * T[i-1]
    for i in range(n - 2, 0, -1):
        rhs = T_old[i] + ea * (T_old[i + 1] - 2.0 * T_old[i] + T_old[i - 1])
        p[i] = (rhs + ta * p[i + 1]) * inv_d[i]

    # Surface node holds half a layer (finite volume), so its energy balance is multiplied by 2:
    # x*(1 + 2*ta*(1 - q1)) + 2*theta*rad*x^4 = R
    T0 = T_old[0]
    lin = 1.0 + 2.0 * ta * (1.0 - q[1])

    if T_ref > 0.0:
        # LINEARIZED EMISSION: eps*sigma*T^4 -> eps*sigma*(Tr^4 + 4*Tr^3*(T - Tr)) = eps*sigma*(4*Tr^3*T - 3*Tr^4)
        g1 = 4.0 * T_ref**3
        g0 = 3.0 * T_ref**4
        R = (T0 + 2.0 * (source
                         - (1.0 - theta) * rad_coeff * (g1 * T0 - g0)
                         + theta * rad_coeff * g0
                         + ea * (T_old[1] - T0)
                         + ta * p[1]))
        return R / (lin + 2.0 * theta * rad_coeff * g1)

    R = (T0 + 2.0 * (source
                     - (1.0 - theta) * rad_coeff * T0**4
                     + ea * (T_old[1] - T0)
                     + ta * p[1]))
    nl = 2.0 * theta * rad_coeff

    # f(x) is monotonic and convex for x > 0, so Newton from the old value converges quickly
    x = T0
    for _ in range(50):
        f = lin * x + nl * x**4 - R
        df = lin + 4.0 * nl * x**3
        dx = f / df
        x -= dx
        if abs(dx) < 1e-10:
            break
    return x


@jit(nopython=True, cache=True)
def implicit_column_step(T_old, T_new, p, q, inv_d, q_be, inv_d_be, source, source_be, rad_coeff, a, theta,
                         bottom_fixed, T_ref=0.0):
    '''
    Advance one facet's temperature column by one timestep. Returns 1 if the backward Euler fallback
    was used, otherwise 0.

    T_old, T_new: (n_layers,) temperatures before/after the step (may be the same array)
    p:            (n_layers,) scratch array
    q, inv_d:     from elimination_coefficients (with the same a, theta and bottom_fixed)
    q_be, inv_d_be: the same for theta = 1 (backward Euler fallback)
    source:       explicit surface heating for this step, in K (insolation*const1 + self heating)
    source_be:    the same with the insolation at the end of the step only (theta = 1)
    rad_coeff:    emission coefficient, in K^-3 (emissivity*sigma*dt / (dz*rho*c))
    a:            kappa*dt/dz^2
    theta:        implicitness, 0.5 = Crank-Nicolson, 1 = backward Euler
    bottom_fixed: True keeps the deepest layer at its current temperature (as in the explicit
                  diurnal loop), False makes it insulating, T[-1] = T[-2] (as in the explicit orbit loop)

    Crank-Nicolson is not L-stable: when the surface is stiff (low thermal inertia, kappa*dt/dz^2 > 1
    or emission time << dt) its explicit half can drive the surface temperature negative, and with
    T^4 emission a negative temperature runs away. Such a step is redone with backward Euler, which keeps
    the temperatures positive (all its coefficients are positive), at first order in time for that step.
    '''
    n = T_old.shape[0]

    # Bottom boundary, written as T[n-1] = p[n-1] + q[n-1] * T[n-2]
    p[n - 1] = T_old[n - 1] if bottom_fixed else 0.0

    fallback = 0
    x = surface_temperature(T_old, p, q, inv_d, source, rad_coeff, a, theta, T_ref)
    if not x >= 0.0:    # negative or NaN (0 K is a facet that starts at 0 K and stays dark)
        fallback = 1
        x = surface_temperature(T_old, p, q_be, inv_d_be, source_be, rad_coeff, a, 1.0, T_ref)
        q = q_be

    # Back substitution (T_old is not read any more, so T_new may be the same array)
    T_new[0] = x
    for i in range(1, n):
        T_new[i] = p[i] + q[i] * T_new[i - 1]
    return fallback


@jit(nopython=True, parallel=True, cache=True)
def implicit_orbit_step(layer_temperatures, insolation, prev_insolation, const1, rad_coeff, a, theta, T_ref):
    '''One orbit timestep for all facets, in place (parallel over facets). layer_temperatures: (n_facets, n_layers).
    Returns the number of facets that needed the backward Euler fallback.'''
    n_facets, n_layers = layer_temperatures.shape
    q, inv_d = elimination_coefficients(n_layers, a, theta, False)
    q_be, inv_d_be = elimination_coefficients(n_layers, a, 1.0, False)
    fallbacks = np.zeros(n_facets, dtype=np.int64)
    for i in prange(n_facets):
        p = np.empty(n_layers)
        fallbacks[i] = implicit_column_step(layer_temperatures[i], layer_temperatures[i], p, q, inv_d, q_be, inv_d_be,
                                            (theta * insolation[i] + (1.0 - theta) * prev_insolation[i]) * const1,
                                            insolation[i] * const1, rad_coeff, a, theta, False, T_ref)
    return fallbacks.sum()


@jit(nopython=True, cache=True)
def implicit_diurnal_day(temperatures, layer_temperatures, insolation, visible_facets_list,
                         view_factors_list, const1, const2, a, self_heating_const,
                         timesteps_per_day, n_layers, include_self_heating, theta, T_ref):
    '''Implicit counterpart of calculate_temperatures in tempestovsky.py (same column swapping).'''
    n_facets = temperatures.shape[0]
    current_column = 0
    prev_column = 1
    p = np.empty(n_layers)
    q, inv_d = elimination_coefficients(n_layers, a, theta, True)
    q_be, inv_d_be = elimination_coefficients(n_layers, a, 1.0, True)

    for time_step in range(timesteps_per_day):
        current_column, prev_column = prev_column, current_column

        for i in range(n_facets):
            prev_step = (time_step - 1) % timesteps_per_day  # the day is periodic
            source = (theta * insolation[i, time_step] + (1.0 - theta) * insolation[i, prev_step]) * const1
            source_be = insolation[i, time_step] * const1
            if include_self_heating:
                self_heating = calculate_secondary_radiation(
                    layer_temperatures[:, prev_column, 0],
                    visible_facets_list[i],
                    view_factors_list[i],
                    self_heating_const
                )
                source += self_heating
                source_be += self_heating

            implicit_column_step(layer_temperatures[i, prev_column], layer_temperatures[i, current_column],
                                 p, q, inv_d, q_be, inv_d_be, source, source_be, const2, a, theta, True, T_ref)
            temperatures[i, time_step] = layer_temperatures[i, current_column, 0]

    return temperatures


class YarkovskyImplicitSolver(YarkovskySolver):
    def __init__(self):
        super().__init__()
        self.name = "tempestovsky_implicit"
        self.prev_insolation = None
        self.fallback_steps = 0     # facet steps redone with backward Euler (see implicit_column_step)

    def theta(self, simulation):
        return float(getattr(simulation, 'implicit_theta', 0.5))

    def linearization_temperature(self, simulation):
        '''
        Reference temperature for linearized emission (0.0 = full nonlinear T^4).
        Config: linearized_emission: true/false, linearization_temperature: "T_av" (default),
        "T_star" or a number in K. With E_star = L_sun/(4 pi r^2) at the current heliocentric distance r:
            T_star^4 = (1-A) E_star / (eps sigma)      (subsolar temperature, Vokrouhlicky's T_star)
            T_av     = T_star / sqrt(2)                (eps sigma T_av^4 = (1-A) E_star / 4)
        T_av is the choice consistent with the linear theory (its Theta, defined with T_star, together
        with k1 = k2 = k3 = 1/2 is equivalent to linearizing the emission around T_star/sqrt(2)).
        '''
        if not bool(getattr(simulation, 'linearized_emission', False)):
            simulation.T_lin_ref = 0.0
            return 0.0
        # Current heliocentric distance (set by the orbit loop); before the orbit loop r = a
        r_au = getattr(simulation, 'current_r_au', simulation.a_au)
        E_star = const.L_sun.value / (4.0 * np.pi * (r_au * const.au.value)**2)
        T_star = ((1.0 - simulation.albedo) * E_star / (simulation.emissivity * const.sigma_sb.value))**0.25
        choice = getattr(simulation, 'linearization_temperature', 'T_av')
        if isinstance(choice, str):
            T_ref = T_star if choice == 'T_star' else T_star / np.sqrt(2.0)
        else:
            T_ref = float(choice)
        simulation.T_lin_ref = T_ref
        return T_ref

    def diurnal_day(self, thermal_data, simulation, config, const1, const2, const3, self_heating_const):
        return implicit_diurnal_day(
            thermal_data.temperatures,
            thermal_data.layer_temperatures,
            thermal_data.insolation,
            thermal_data.visible_facets,
            thermal_data.thermal_view_factors,
            const1, const2, const3, self_heating_const,
            simulation.timesteps_per_day, simulation.n_layers,
            config.include_self_heating, self.theta(simulation),
            self.linearization_temperature(simulation)
        )

    def orbit_step(self, thermal_data, current_insolation, simulation):
        T = thermal_data.layer_temperatures
        if not T.flags['C_CONTIGUOUS']:
            T = np.ascontiguousarray(T)

        heat_capacity = simulation.density * simulation.specific_heat_capacity * simulation.layer_thickness
        const1 = simulation.delta_t / heat_capacity
        rad_coeff = simulation.emissivity * const.sigma_sb.value * simulation.delta_t / heat_capacity
        a = simulation.thermal_diffusivity * simulation.delta_t / simulation.layer_thickness**2

        current_insolation = np.ascontiguousarray(current_insolation, dtype=np.float64)
        if self.prev_insolation is None:
            self.prev_insolation = current_insolation

        self.fallback_steps += implicit_orbit_step(T, current_insolation, self.prev_insolation,
                                                   const1, rad_coeff, a, self.theta(simulation),
                                                   self.linearization_temperature(simulation))
        self.prev_insolation = current_insolation
        return T
