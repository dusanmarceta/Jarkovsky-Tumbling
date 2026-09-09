import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def verify_apophis_tumbling():
    # --- 1. ULAZNI PARAMETRI IZ TABELE ---
    I1, I2, I3 = 0.61, 0.965, 1.0
    I = (I1, I2, I3)

    phi_0 = np.deg2rad(152.0)
    psi_0 = np.deg2rad(14.0)
    E_ratio_target = 1.024
    
    # Periodi iz tabele (u satima)
    P_phi_h = 27.38
    P_psi_h = 263.0

    # Proračun početnog ugla nutacije theta_0
    A = (np.sin(psi_0)**2 / I1) + (np.cos(psi_0)**2 / I2)
    sin2_theta0 = (E_ratio_target - 1.0) / (I3 * A - 1.0)
    theta_0 = np.arcsin(np.sqrt(sin2_theta0))

    # Brzina precesije i magnituda momenta impulsa L
    w_phi = (2 * np.pi) / (P_phi_h * 3600.0)
    L_mag = w_phi / A

    # Početne ugaone brzine
    w1_0 = (L_mag * np.sin(theta_0) * np.sin(psi_0)) / I1
    w2_0 = (L_mag * np.sin(theta_0) * np.cos(psi_0)) / I2
    w3_0 = (L_mag * np.cos(theta_0)) / I3
    
    print(f'period: {(2 * np.pi / w3_0) / 3600.0}')
    
    print

    y0 = [w1_0, w2_0, w3_0, phi_0, theta_0, psi_0]

    # --- 2. INTEGRACIJA DINAMIKE ---
    # Simulacija u trajanju od 300 sati (dovoljno da se vidi pun ciklus P_psi od 263h)
    t_span = (0, 300 * 3600)
    t_eval = np.linspace(t_span[0], t_span[1], 10000)

    def dynamics(t, y):
        w1, w2, w3, phi, theta, psi = y
        dw1 = ((I2 - I3) * w2 * w3) / I1
        dw2 = ((I3 - I1) * w3 * w1) / I2
        dw3 = ((I1 - I2) * w1 * w2) / I3

        st = np.sin(theta)
        if abs(st) < 1e-8:
            st = 1e-8

        dphi = (w1 * np.sin(psi) + w2 * np.cos(psi)) / st
        dtheta = w1 * np.cos(psi) - w2 * np.sin(psi)
        dpsi = w3 - dphi * np.cos(theta)

        return [dw1, dw2, dw3, dphi, dtheta, dpsi]

    sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval, method='DOP853', rtol=1e-11, atol=1e-13)

    t_hours = sol.t / 3600.0
    phi_deg = np.unwrap(sol.y[3]) * (180 / np.pi)
    theta_deg = np.rad2deg(sol.y[4])
    psi_deg = np.unwrap(sol.y[5]) * (180 / np.pi)

    # --- 3. GRAFIČKI PRIKAZ ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Plot 1: Precesioni ugao phi(t)
    axs[0].plot(t_hours, phi_deg, label=r'$\phi(t)$ (Precesija)', color='tab:blue')
    axs[0].set_ylabel('Nekontinuirani $\phi$ [deg]')
    axs[0].grid(True)
    axs[0].axvline(P_phi_h, color='r', linestyle='--', label=f'$P_\phi = {P_phi_h}h$')
    axs[0].legend(loc='upper left')
    axs[0].set_title('Verifikacija tumbling rotacije (Apophis model)')

    # Plot 2: Ugao nutacije theta(t)
    axs[1].plot(t_hours, theta_deg, label=r'$\theta(t)$ (Nutacija)', color='tab:green')
    axs[1].set_ylabel(r'$\theta$ [deg]')
    axs[1].grid(True)
    axs[1].legend(loc='upper left')

    # Plot 3: Ugao spina psi(t)
    axs[2].plot(t_hours, psi_deg, label=r'$\psi(t)$ (Spin)', color='tab:orange')
    axs[2].set_xlabel('Vreme [h]')
    axs[2].set_ylabel('Nekontinuirani $\psi$ [deg]')
    axs[2].grid(True)
    axs[2].axvline(P_psi_h, color='purple', linestyle='--', label=f'$P_\psi = {P_psi_h}h$')
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

verify_apophis_tumbling()