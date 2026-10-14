'''
Reproduction of Fig. 3 of Vokrouhlicky et al. (2015, Icarus 252, 277) with Tempestovsky only.

<da/dt> is computed by Tempestovsky (Apophis_200.stl, tumbling, nominal rotation state) on a grid in
thermal inertia Gamma, Bond albedo A and obliquity gamma (angle between L and the orbit normal), and
interpolated for each Monte Carlo sample. Size D and bulk density rho enter exactly through
<da/dt> ~ 1/(rho D) (the temperatures do not depend on D or rho at fixed Gamma; force ~ D^2, mass ~ rho D^3).
Roughness enhancement and the 3% shape noise are applied as in the paper.
'''
import glob, os, re
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
MS2AUMYR = 3.15576e13 / 1.495978707e11          # m/s -> au/Myr
D_SHAPE = 334.6                                   # volume-equivalent diameter of Apophis_200.stl [m]
RHO_SIM = 2000.0                                  # bulk density used in the simulations [kg/m3]
OBL_NOM = 167.201                                 # obliquity of the nominal rotation state [deg]
rng = np.random.default_rng(20150131)
N = 200_000


def read(f):
    return float(open(f).read().split(':')[1]) * MS2AUMYR * 1e4      # 1e-4 au/Myr


# --- Tempestovsky grid: out3d/g{Gamma}_A{A}_o{obl}.txt plus the nominal runs out/ti{Gamma}.txt -------
grid = {}
for f in glob.glob(os.path.join(HERE, 'out3d', 'g*_A*_o[0-9][0-9][0-9].txt')):
    g, a, o = re.match(r'g(\d+)_A([\d.]+)_o(\d+)\.txt', os.path.basename(f)).groups()
    grid[(float(g), float(a), OBL_NOM if o == '167' else float(o))] = read(f)
G = sorted({k[0] for k in grid})
for g in G:
    grid[(g, 0.14, OBL_NOM)] = read(os.path.join(HERE, 'out', f'ti{int(g)}.txt'))
AL = sorted({k[1] for k in grid})
OB = sorted({k[2] for k in grid})
tab = np.array([[[grid[(g, a, o)] for o in OB] for a in AL] for g in G])
interp = RegularGridInterpolator((np.log(G), AL, OB), tab)

print('Tempestovsky <da/dt> [1e-4 au/Myr] for D = 334.6 m, rho = 2 g/cm3 (rows Gamma, columns obliquity)')
for ia, a in enumerate(AL):
    print(f'A = {a}:   obliquity ' + '  '.join(f'{o:7.1f}' for o in OB))
    for ig, g in enumerate(G):
        print(f'   Gamma {g:6.0f}        ' + '  '.join(f'{v:7.2f}' for v in tab[ig, ia]))

# --- parameter distributions -------------------------------------------------------------------------
def asym_normal(mode, s_lo, s_hi, n):
    '''Two-piece normal with given mode and lower/upper sigmas.'''
    z = rng.standard_normal(n)
    return np.where(z < 0, mode + s_lo * z, mode + s_hi * z)

# Thermal inertia (Mueller et al. 2014): best 600, most likely range 250-800 taken as 1 sigma,
# truncated to the simulated range
Gam = asym_normal(600.0, 350.0, 200.0, N)
while np.any(bad := (Gam < G[0]) | (Gam > G[-1])):
    Gam[bad] = asym_normal(600.0, 350.0, 200.0, bad.sum())
lg = np.log10(Gam / 600.0)
# Size (Mueller Fig. 4: D decreases ~20 m per decade of Gamma); 375 m at 600, scatter 6 m (+) 2.5% calib.
D = 375.0 - 20.0 * lg + rng.standard_normal(N) * np.hypot(6.0, 0.025 * 375.0)
# Bond albedo (Mueller Fig. 4: pV rises ~0.035 per decade of Gamma); A = 0.14 +0.03 -0.04
A = 0.14 * (0.30 + 0.035 * lg) / 0.30 + asym_normal(0.0, 0.04, 0.03, N)
# Obliquity (Pravec et al. 2014, Fig. 7): median 165, 16-84% range 157-172
obl = asym_normal(165.0, 8.0, 7.0, N)
obl = np.where(obl > 180.0, 360.0 - obl, obl)
# Bulk density (Farnocchia et al. 2013b): lognormal, mean 2.20 g/cm3, variance 0.3 g^2/cm^6
m, v = 2.20, 0.30
s2 = np.log(1 + v / m**2)
rho = np.exp(np.log(m) - s2 / 2 + np.sqrt(s2) * rng.standard_normal(N)) * 1000.0
# Roughness enhancement: up to 20%, median half of it, Gaussian reaching the limits at 3 sigma
rough = 1.10 + (0.10 / 3.0) * rng.standard_normal(N)
# Shape / rotation-state uncertainty: 3% random noise
shape = 1.0 + 0.03 * rng.standard_normal(N)

clipped = np.mean((A < AL[0]) | (A > AL[-1]) | (obl < OB[0]) | (obl > OB[-1]))
pts = np.column_stack([np.log(Gam), np.clip(A, AL[0], AL[-1]), np.clip(obl, OB[0], OB[-1])])
dadt = interp(pts) * (D_SHAPE / D) * (RHO_SIM / rho) * rough * shape

p = np.percentile(dadt, [0.135, 15.865, 50, 84.135, 99.865])
hist, edges = np.histogram(dadt, bins=140, range=(-35, 0), density=True)
cent = 0.5 * (edges[1:] + edges[:-1])
print(f'\nSamples outside the A/obliquity grid (clipped to its edge): {100*clipped:.1f}%')
print(f'Tempestovsky MC (N = {N}): mean {dadt.mean():.2f}, sd {dadt.std():.2f}, peak {cent[np.argmax(hist)]:.2f}, '
      f'median {p[2]:.2f}, 1 sigma [{p[1]:.2f}, {p[3]:.2f}], 3 sigma [{p[0]:.2f}, {p[4]:.2f}]')
print('Paper Fig. 3:              mean -12.80, sd 3.60, peak -11.80, median -12.40, 1 sigma [-15.70, -8.60], 3 sigma [-26.40, -3.40]')
np.savez(os.path.join(HERE, 'fig3_samples.npz'), dadt=dadt, hist=hist, cent=cent, pct=p)
