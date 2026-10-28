'''Same initial states, trajectory scaled to the paper's closest approach distance (37 730 km instead of 38 011 km).'''
import numpy as np
from flyby import *
from orbits import PRE, POST, obliquity
d = np.load('ensemble_50000.npz')
n = 10000
I1, I2 = d['I1'][:n], d['I2'][:n]
dt = 30.0
g, rhat, n_steps, jd_ca, Rmin = trajectory(dt)
out = {}
for label, f in (('38011 km (JPL now)', 1.0), ('37730 km (paper)', 37730.0 / Rmin)):
    qf, wf, acc = integrate(d['q0'][:n], d['w0'][:n], np.c_[I1, I2, np.ones(n)], g / f**3, rhat, dt, n_steps, n_steps // 2)
    fs = final_state(qf, wf, I1, I2)
    delta = np.degrees(np.arccos(np.clip((lb_to_vec(d['pre_lam'][:n], d['pre_beta'][:n]) * lb_to_vec(fs['lam'], fs['beta'])).sum(1), -1, 1)))
    dl = (fs['lam'] - d['pre_lam'][:n] + 180) % 360 - 180
    print(f'{label}: delta mean {delta.mean():.2f} max {delta.max():.1f} deg | lambda increases {100*np.mean(dl > 0):.1f}% | '
          f'LAM {100*np.mean(~fs["sam"]):.1f}% | acc max {np.degrees(acc.max())*HOUR**2:.2f} deg/h^2 | P_phi_l mean {fs["P_phi_l_h"].mean():.2f} h | P_psi mean {fs["P_psi_h"].mean():.1f} h', flush=True)
print('paper: delta mean 14.5 max 45.8 | lambda increases 93.5% | LAM 17.3% | acc max 4.5 | P_phi_l mean 30.9 | P_psi mean 266.0')
