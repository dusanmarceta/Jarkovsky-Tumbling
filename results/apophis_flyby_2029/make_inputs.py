'''
Tempestovsky inputs for the post-2029 Yarkovsky drift of Apophis: one row per run, for the reproduced flyby
ensemble (inputs_sim.csv) and for the ensemble reconstructed from the digitized figures (inputs_dig.csv).
Row k of both files has the same thermal and physical parameters, so the two ensembles differ only in the spin state.

Spin state (Tempestovsky tumbling parameters, I3 = 1):
  I1, I2       = I_l/I_s, I_i/I_s
  E_ratio      = 2 I3 E / L^2 = I3 / I_d
  P_phi_h      = mean precession period of the short axis about L (Tempestovsky's Euler angle convention)
  psi_0        = Euler angle psi of the post-flyby state (any reachable value gives the same rotation, other phase)
  phi_0        = 0 (phase, irrelevant for the orbit-averaged drift)
  lambda_L, beta_L = post-flyby pole in the post-flyby orbital frame (x towards perihelion, z orbit normal)
Thermal/physical Monte Carlo as in results/apophis_fig3/fig3_mc.py (Vokrouhlicky et al. 2015, Fig. 3):
  thermal_inertia, albedo go into Tempestovsky; D, rho, roughness, shape_noise only rescale the drift afterwards:
  da/dt = da/dt(Tempestovsky, D_SHAPE, RHO_SIM) * (D_SHAPE / D) * (RHO_SIM / rho) * roughness * shape_noise
'''
import numpy as np
from orbits import POST, ecliptic_to_orbital

N = 50000
GAMMA_RANGE = (100.0, 1200.0)     # truncation of the thermal inertia distribution (as in the Fig. 3 reproduction)
rng = np.random.default_rng(20290413)


def asym_normal(mode, s_lo, s_hi, n):
    z = rng.standard_normal(n)
    return np.where(z < 0, mode + s_lo * z, mode + s_hi * z)


def thermal_samples(n):
    Gam = asym_normal(600.0, 350.0, 200.0, n)
    while np.any(bad := (Gam < GAMMA_RANGE[0]) | (Gam > GAMMA_RANGE[1])):
        Gam[bad] = asym_normal(600.0, 350.0, 200.0, bad.sum())
    lg = np.log10(Gam / 600.0)
    D = 375.0 - 20.0 * lg + rng.standard_normal(n) * np.hypot(6.0, 0.025 * 375.0)
    A = 0.14 * (0.30 + 0.035 * lg) / 0.30 + asym_normal(0.0, 0.04, 0.03, n)
    A = np.clip(A, 0.02, 0.40)
    m, v = 2.20, 0.30
    s2 = np.log(1 + v / m**2)
    rho = np.exp(np.log(m) - s2 / 2 + np.sqrt(s2) * rng.standard_normal(n)) * 1000.0
    rough = 1.10 + (0.10 / 3.0) * rng.standard_normal(n)
    shape = 1.0 + 0.03 * rng.standard_normal(n)
    return dict(thermal_inertia=Gam, albedo=A, D=D, rho=rho, roughness=rough, shape_noise=shape)


def reachable_psi(I1, I2, E_ratio, n_draw):
    '''Uniform psi_0 among the values for which E_ratio is reachable: sin^2(psi) >= smin
    (smin = 0 for SAM, a band around 90 and 270 deg for LAM).'''
    smin = np.clip((E_ratio - 1 / I2) / (1 / I1 - 1 / I2), 0, 1)
    p0 = np.degrees(np.arcsin(np.sqrt(smin)))                       # |sin psi| >= sin p0
    psi = p0 + rng.uniform(0, 1, n_draw) * (180 - 2 * p0)           # in [p0, 180 - p0]
    return np.where(rng.uniform(size=n_draw) < 0.5, psi, psi + 180)


def spin_rows(d, psi):
    lam_L, beta_L = ecliptic_to_orbital(d['post_lam'], d['post_beta'], POST)
    return dict(I1=d['I1'], I2=d['I2'], E_ratio=1.0 / d['post_I_d'], P_phi_h=d['post_P_phi_s_h'],
                psi_0=psi, phi_0=np.zeros(len(psi)), lambda_L=lam_L, beta_L=beta_L,
                sam=d['post_sam'].astype(int), P_psi_h=d['post_P_psi_h'], P_phi_l_h=d['post_P_phi_l_h'])


def write(path, cols):
    names = list(cols)
    data = np.column_stack([np.asarray(cols[k], dtype=float) for k in names])
    np.savetxt(path, np.column_stack([np.arange(len(data)), data]), delimiter=',',
               header='run,' + ','.join(names), comments='', fmt=['%d'] + ['%.10g'] * len(names))


if __name__ == '__main__':
    th = thermal_samples(N)
    sim = np.load('ensemble_50000.npz')
    dig = np.load('digitized_50000.npz')
    n_dig = len(dig['I1'])
    common = dict(a_au=np.full(N, POST['a']), ecc=np.full(N, POST['e']))
    rows_sim = spin_rows(sim, sim['post_psi_deg'])
    write('inputs_sim.csv', {**rows_sim, **common, **th})
    idx = np.arange(N) % n_dig                         # the digitized set may have slightly fewer states
    d_dig = {k: dig[k][idx] for k in dig.files}
    rows_dig = spin_rows(d_dig, reachable_psi(d_dig['I1'], d_dig['I2'], 1.0 / d_dig['post_I_d'], N))
    write('inputs_dig.csv', {**rows_dig, **common, **th})
    print('written inputs_sim.csv and inputs_dig.csv,', N, 'rows each')
