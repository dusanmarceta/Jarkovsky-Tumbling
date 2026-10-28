#!/src/utilities/tumbling_matrices.py
"""
Compute free-precession rotation matrices for an arbitrary ASCII STL shape,
ensuring a rational number of spin and precession cycles.

This is a rough work in progress and TEMPEST does not currently support tumbling bodies.

Dependencies:
    pip install numpy-stl numpy
"""

import argparse
import numpy as np
from stl import mesh
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp


# from utils import (
#     conditional_tqdm,
#     conditional_print,
#     rays_triangles_intersection,
#     calculate_rotation_matrix, sun_direction
# )  





def mean_euler_rates(y0, I):
    """
    Mean rates of the Euler angles phi (precession about L) and psi for torque-free rotation.

    The body-frame angular velocity is periodic, so one period T of omega(t) is found (omega returns
    to omega0) and the mean rates are the increments of phi and psi over that period divided by T.
    Returns (mean_phi_dot, mean_psi_dot, T) in the units of y0 (rad/s, s).
    """
    I1, I2, I3 = I
    y0 = np.asarray(y0, dtype=float)
    w0 = y0[:3]

    def dynamics(t, y):
        w1, w2, w3, phi, theta, psi = y
        st = np.sin(theta) if abs(np.sin(theta)) > 1e-9 else 1e-9
        dphi = (w1 * np.sin(psi) + w2 * np.cos(psi)) / st
        return [((I2 - I3) * w2 * w3) / I1,
                ((I3 - I1) * w3 * w1) / I2,
                ((I1 - I2) * w1 * w2) / I3,
                dphi,
                w1 * np.cos(psi) - w2 * np.sin(psi),
                w3 - dphi * np.cos(theta)]

    # omega(t) - omega0 projected on the initial direction of motion of omega: zero and increasing
    # at t = 0 and again after each full period
    dw0 = np.array(dynamics(0.0, y0)[:3])

    def back_to_start(t, y):
        return np.dot(np.asarray(y[:3]) - w0, dw0)
    back_to_start.direction = 1

    t_unit = 2 * np.pi / np.linalg.norm(w0)
    sol = solve_ivp(dynamics, (0, 1000 * t_unit), y0, method='DOP853', rtol=1e-11, atol=1e-13,
                    events=back_to_start)
    t_ev, y_ev = sol.t_events[0], sol.y_events[0]
    k = np.argmax(t_ev > 1e-3 * t_unit)
    if t_ev.size == 0 or t_ev[k] <= 1e-3 * t_unit:
        raise RuntimeError('mean_euler_rates: period of the body-frame rotation not found')
    T = t_ev[k]
    return (y_ev[k][3] - y0[3]) / T, (y_ev[k][5] - y0[5]) / T, T


def max_angular_speed(w0, I, T, n_samples=2000):
    """
    Maximum of |omega| over one period T of the torque-free body-frame angular velocity (Euler equations only).
    Used as the rotation rate that sets the time step: unlike omega_3, |omega| never approaches zero
    (omega_3 oscillates around zero in long axis mode).
    """
    I1, I2, I3 = I

    def euler(t, w):
        return [((I2 - I3) * w[1] * w[2]) / I1,
                ((I3 - I1) * w[2] * w[0]) / I2,
                ((I1 - I2) * w[0] * w[1]) / I3]

    sol = solve_ivp(euler, (0, T), np.asarray(w0, dtype=float), method='DOP853', rtol=1e-11, atol=1e-13,
                    t_eval=np.linspace(0, T, n_samples))
    return np.max(np.linalg.norm(sol.y, axis=0))


def compute_tumbling_dynamics(dt, timesteps, y0, I, r_inertial, lambda_L_deg, beta_L_deg, initialisation):
    """
    Rešava Eulerove jednačine i vraća rezultate uključujući i fiksni vektor ugaonog momenta L.
    """
    # r_inertial[:, 0] = -r_inertial[:, 0]
    # r_inertial[:, 1] = -r_inertial[:, 1]
    t_inertial = np.zeros_like(r_inertial)
    
    for i in range(timesteps):
        t_inertial[i] = np.cross(r_inertial[i], np.array([0, 0, 1]))
        

    # Uglovi iz stepeni u radijane
    lambda_L = np.deg2rad(lambda_L_deg)
    beta_L = np.deg2rad(beta_L_deg)

    Rz = np.array([
    [np.cos(lambda_L), -np.sin(lambda_L), 0],
    [np.sin(lambda_L),  np.cos(lambda_L), 0],
    [0,            0,           1]
    ])

    Ry = np.array([
        [np.cos(np.pi/2 - beta_L), 0, np.sin(np.pi/2 - beta_L)],
        [0,                       1, 0],
        [-np.sin(np.pi/2 - beta_L), 0, np.cos(np.pi/2 - beta_L)]
    ])
    
  
    R_rotation = Rz @ Ry
    
    # R_rotation = Rz_180 @ Rz @ Ry
    
    r_inertial = (R_rotation.T @ r_inertial.T).T
    t_inertial = (R_rotation.T @ t_inertial.T).T
    

    
    I1, I2, I3 = I
    
    # --- 1. Definisanje dinamike ---
    def dynamics(t, y):
        w1, w2, w3, phi, theta, psi = y
        # Eulerove jednačine (Dinamika ugaonih brzina)
        dw1 = ((I2 - I3) * w2 * w3) / I1
        dw2 = ((I3 - I1) * w3 * w1) / I2
        dw3 = ((I1 - I2) * w1 * w2) / I3
        
        # Kinematika Eulerovih uglova (3-1-3 konvencija)
        st = np.sin(theta) if abs(np.sin(theta)) > 1e-9 else 1e-9
        dphi = (w1 * np.sin(psi) + w2 * np.cos(psi)) / st
        dtheta = w1 * np.cos(psi) - w2 * np.sin(psi)
        dpsi = w3 - dphi * np.cos(theta)
        
        return [dw1, dw2, dw3, dphi, dtheta, dpsi]

    # --- 2. Integracija ---
    
    t_eval = np.arange(1, timesteps + 1) * dt
    
       
    sol = solve_ivp(dynamics, (0, t_eval[-1]), y0, t_eval=t_eval, 
                    method='DOP853', rtol=1e-11, atol=1e-13)
    
    # --- 3. Proračun ugaonog momenta L (Fiksan u inercijalnom prostoru) ---
    # Uzimamo početne uslove (t=0) da odredimo L_hat
    w0 = np.array([y0[0], y0[1], y0[2]])
    phi0, theta0, psi0 = y0[3], y0[4], y0[5]
    
    c1, s1 = np.cos(phi0), np.sin(phi0)
    c2, s2 = np.cos(theta0), np.sin(theta0)
    c3, s3 = np.cos(psi0), np.sin(psi0)
    
    # Početna matrica R (Inercijalni -> Telo)
    R0 = np.array([
        [ c3*c1 - s3*c2*s1,  c3*s1 + s3*c2*c1, s3*s2],
        [-s3*c1 - c3*c2*s1, -s3*s1 + c3*c2*c1, c3*s2],
        [ s2*s1,            -s2*c1,            c2]
    ])
    
    # L u sistemu tela: [I1*w1, I2*w2, I3*w3]
    L_body = np.array([I1 * w0[0], I2 * w0[1], I3 * w0[2]])
    # L u inercijalnom sistemu: R.T @ L_body
    L_inertial = R0.T @ L_body
    L_hat = L_inertial / np.linalg.norm(L_inertial)

    # --- 4. Transformacije kroz vreme ---
    rotations = []
    r_body = []
    t_body = []
    spin_axis_iner = []


    for i in range(len(sol.t)):
        phi_i, theta_i, psi_i = sol.y[3, i], sol.y[4, i], sol.y[5, i]
        w_body = sol.y[0:3, i]
        
        ci, si = np.cos(phi_i), np.sin(phi_i)
        cj, sj = np.cos(theta_i), np.sin(theta_i)
        ck, sk = np.cos(psi_i), np.sin(psi_i)
        
        Ri = np.array([
            [ ck*ci - sk*cj*si,  ck*si + sk*cj*ci, sk*sj],
            [-sk*ci - ck*cj*si, -sk*si + ck*cj*ci, ck*sj],
            [ sj*si,            -sj*ci,            cj]
        ])
        
        
        # Frames: Ri maps the L-frame (z along L) to the body frame, R_rotation maps the L-frame to the
        # inertial (orbit) frame. The shape model (STL) is given in the body principal-axis frame
        # (x = I1, y = I2, z = I3), so body -> inertial is R_rotation @ Ri.T, and the Sun direction in the
        # body frame is Ri @ R_rotation.T @ r (r_inertial was already multiplied by R_rotation.T above).
        rotations.append(R_rotation @ Ri.T)
        r_body.append(Ri @ r_inertial[i])
        t_body.append(Ri @ t_inertial[i])
        

        # PREDLOG POPRAVKE:
        spin_axis_iner.append(Ri.T @ w_body)

        # r_body_rotated.append(Ri @ r_inertial[i])
        # r_body_rotated.append(Ri @ t_inertial[i])
    # --- NOVI DEO: Čuvanje poslednjeg stanja ---
    # sol.y[:,-1] uzima sve promenljive (w1, w2, w3, phi, theta, psi) iz poslednjeg vremenskog koraka
    last_state = sol.y[:, -1].tolist()

    # ... (tvoj postojeći kod za transformacije kroz vreme) ...

    

    r_body = np.stack(r_body)
    t_body = np.stack(t_body)

    return {
        'rotations': np.stack(rotations),
        'L_axis': L_hat,
        'r_body': r_body,
        't_body': t_body,
        # 'r_body_rotated': np.stack(r_body_rotated),
        # 't_body_rotated': np.stack(t_body_rotated),
        'G_axes': np.stack(spin_axis_iner),
        # 'omega_axes': np.stack(body_axis_iner),
        'time': sol.t,
        'omega_body': sol.y[0:3, :].T,
        # 'body_axis': np.stack(body_axis_iner),
        'last_state': last_state  # DODATO: spremno za sledeći y0
    }
    

# time_step = 0.1265
# N_steps = 100000
# y0 = [np.float64(-0.0), np.float64(-0.0), np.float64(0.0008726650031629793), np.float64(0.0), np.float64(-0.0), np.float64(0.0)]

# I1 = 1
# I2 = 1
# I3 = 0.3
# r_inertial = np.tile(np.array([1, 0, 0]), (N_steps, 1))
# lambda_L = 60
# beta_L = 38


# sim_napred = compute_tumbling_dynamics(dt = time_step, timesteps = N_steps, y0 = y0,
#                                     I = np.array([I1, I2, I3]), 
#                                     r_inertial = r_inertial,
#                                     lambda_L_deg = lambda_L,
#                                     beta_L_deg = beta_L
#                                     )


# y0 = sim_napred['last_state']

# print(y0)


# sim_nazad = compute_tumbling_dynamics(dt = -time_step, timesteps = N_steps, y0 = y0,
#                                     I = np.array([I1, I2, I3]), 
#                                     r_inertial = r_inertial,
#                                     lambda_L_deg = lambda_L,
#                                     beta_L_deg = beta_L
#                                     )

# print(sim_nazad['last_state'])






