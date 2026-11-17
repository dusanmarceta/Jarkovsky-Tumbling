'''
Fig. 1 of Farnocchia et al. (2024, Apophis T-5 Years, abstract 2046) with Tempestovsky: bulk density of Apophis as a
function of thermal inertia needed to match the measured Yarkovsky drift da/dt = -199 +- 1 m/yr.

    python results/apophis_density/density_fig1.py        # -> density_fig1.png, paths relative to this file

Tempestovsky drift for the nominal tumbling state of Pravec et al. (2014) (P_phi 27.38 h, I1 0.61, I2 0.965,
E_ratio 1.024; pole (269.222, -77.201) in the orbital frame = obliquity 167.2 deg), Apophis_200.stl
(D_SHAPE = 334.6 m), rho = 2000 kg/m3, pre-flyby orbit. Runs:
  ../apophis_fig3/out/ti*.txt             A = 0.14, Gamma 100-1200
  ../apophis_fig3/out3d/g*_A*_o167.txt    A = 0.08, 0.20, Gamma 100-1200
  out/g*_A*.txt (run.sh)                  A = 0.08, 0.14, 0.20, Gamma 50 and 75
At fixed Gamma the temperatures do not depend on D and rho, so da/dt ~ 1/(rho D) and
    rho(Gamma) = RHO_SIM * (da/dt)_model(Gamma, A) * (D_SHAPE / D) / (da/dt)_measured.
Physical parameters as in the abstract: D = 340 +- 50 m, p_V = 0.29 +- 0.03, G = 0.24 +- 0.11, Bond albedo
A = q p_V with q = 0.290 + 0.684 G. +-1 sigma: linear propagation of D, A (from p_V and G) and the measured drift.
The curve is a PCHIP interpolation in log Gamma through the runs; the albedo correction from A = 0.14 to the
nominal A is a quadratic in A through the three albedos at each Gamma of the 3D grid.

100% roughness: the abstract uses the Yarkovsky enhancement model developed for Bennu (Chesley et al. 2014).
Tempestovsky's kernel roughness does not enter the drift calculation, so the enhancement factor (red / blue curve)
is taken from the digitized Fig. 1 (fig1_digitized.npz, digitize_fig1.py), smoothed, and applied to the Tempestovsky\ncurve.
'''
import glob, os, re
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
FIG3 = os.path.join(HERE, '..', 'apophis_fig3')
YR = 3.15576e7                                    # s
D_SHAPE, RHO_SIM = 334.6, 2000.0                  # volume-equivalent diameter of Apophis_200.stl [m], density in runs
DADT, SIG_DADT = -199.0, 1.0                      # measured drift [m/yr] (Farnocchia & Chesley 2022)
D_NOM, SIG_D = 340.0, 50.0                        # [m]
PV, SIG_PV, G_SL, SIG_G = 0.29, 0.03, 0.24, 0.11
TI_RANGE = (50.0, 500.0)                          # shaded range of thermal inertia in Fig. 1 (DellaGiustina et al. 2023)
RHO_ANALOG = (1400.0, 3000.0)                     # plausible densities with Itokawa as an analog (dashed black)
GAMMA = np.arange(50.0, 1200.001, 5.0)


def read(f):
    return float(open(f).readline().split(':')[1])   # drift [m/s], first line


def runs():
    '''Tempestovsky drift [m/s] of the nominal state, {(Gamma, A): drift}.'''
    d = {}
    for f in glob.glob(os.path.join(FIG3, 'out', 'ti*.txt')):
        m = re.fullmatch(r'ti(\d+)\.txt', os.path.basename(f))
        if m:
            d[(float(m.group(1)), 0.14)] = read(f)
    for f in glob.glob(os.path.join(FIG3, 'out3d', 'g*_A*_o167.txt')) + glob.glob(os.path.join(HERE, 'out', 'g*_A*.txt')):
        m = re.fullmatch(r'g(\d+)_A([\d.]+)(_o167)?\.txt', os.path.basename(f))
        if m:
            d[(float(m.group(1)), float(m.group(2)))] = read(f)
    return d


def model():
    '''Density curve for the nominal parameters, its relative 1-sigma error, and the run nodes.'''
    d = runs()
    q = 0.290 + 0.684 * G_SL
    A, sig_A = q * PV, np.hypot(q * SIG_PV, 0.684 * PV * SIG_G)
    g14 = np.array(sorted(g for g, a in d if a == 0.14))
    drift14 = np.array([d[(g, 0.14)] for g in g14])
    g3 = np.array(sorted(g for g in g14 if all((g, a) in d for a in (0.08, 0.14, 0.20))))
    corr, slope = [], []
    for g in g3:                                  # quadratic in A of ln|drift| through A = 0.08, 0.14, 0.20
        p = np.polyfit([0.08, 0.14, 0.20], np.log([-d[(g, a)] for a in (0.08, 0.14, 0.20)]), 2)
        corr.append(np.exp(np.polyval(p, A) - np.polyval(p, 0.14)))
        slope.append(np.polyval(np.polyder(p), A))
    lg = np.log(GAMMA)
    drift = (PchipInterpolator(np.log(g14), drift14)(lg)
             * PchipInterpolator(np.log(g3), corr)(lg))
    dlnA = PchipInterpolator(np.log(g3), slope)(lg)
    rho = RHO_SIM * drift * YR * (D_SHAPE / D_NOM) / DADT
    rel = np.sqrt((SIG_D / D_NOM) ** 2 + (dlnA * sig_A) ** 2 + (SIG_DADT / DADT) ** 2)
    rho_nodes = RHO_SIM * drift14 * YR * (D_SHAPE / D_NOM) / DADT * np.interp(np.log(g14), np.log(g3), corr)
    return dict(rho=rho, rel=rel, A=A, sig_A=sig_A, g_nodes=g14, rho_nodes=rho_nodes, runs=d)


def crossing(y, level):
    i = np.flatnonzero(np.diff(np.sign(y - level)))
    return [GAMMA[k] + (level - y[k]) * (GAMMA[k + 1] - GAMMA[k]) / (y[k + 1] - y[k]) for k in i]


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    m = model()
    rho, rel = m['rho'], m['rel']
    pap = np.load(os.path.join(HERE, 'fig1_digitized.npz'))
    # enhancement red / blue of the paper, smoothed with a quartic in log Gamma (digitization noise; residuals < 0.5%)
    enh = np.polyval(np.polyfit(np.log(pap['Gamma']), pap['red'] / pap['blue'], 4), np.log(GAMMA))
    rho_r = rho * enh
    p_blue, p_red = np.interp(GAMMA, pap['Gamma'], pap['blue']), np.interp(GAMMA, pap['Gamma'], pap['red'])

    # checks of the new runs against the old grid and against a finer time step
    out = os.path.join(HERE, 'out')
    for new, ref, what in (('chk_g100_A0.14', os.path.join(FIG3, 'out', 'ti100.txt'), 'current code vs apophis_fig3 grid, Gamma 100'),
                           ('conv_g50_A0.14_s1600', os.path.join(out, 'g50_A0.14.txt'), 'Gamma 50: 1600 vs 800 steps per day')):
        f = os.path.join(out, new + '.txt')
        if os.path.exists(f) and os.path.exists(ref):
            print(f'check, {what}: {100 * (read(f) / read(ref) - 1):+.2f}%')

    print(f'measured da/dt {DADT} +- {SIG_DADT} m/yr, D = {D_NOM} +- {SIG_D} m, Bond albedo A = {m["A"]:.4f} +- {m["sig_A"]:.4f}')
    print(f'{"Gamma":>6s} {"rho smooth":>11s} {"1 sigma":>8s} {"rho rough":>10s} {"paper smooth":>13s} {"paper rough":>12s} {"ratio":>6s}')
    for g in (50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1200):
        i = np.argmin(abs(GAMMA - g))
        print(f'{g:6.0f} {rho[i]:11.0f} {100 * rel[i]:7.1f}% {rho_r[i]:10.0f} {p_blue[i]:13.0f} {p_red[i]:12.0f} {rho[i] / p_blue[i]:6.3f}')
    k, kp = np.argmax(rho), np.argmax(p_blue)
    print(f'maximum, smooth: Tempestovsky {rho[k]:.0f} at Gamma {GAMMA[k]:.0f}; paper {p_blue[kp]:.0f} at Gamma {GAMMA[kp]:.0f}')
    sel = (GAMMA >= TI_RANGE[0]) & (GAMMA <= TI_RANGE[1])
    print(f'Gamma {TI_RANGE[0]:.0f}-{TI_RANGE[1]:.0f}: smooth {rho[sel].min():.0f}-{rho[sel].max():.0f}, '
          f'rough {rho_r[sel].min():.0f}-{rho_r[sel].max():.0f} kg/m3 (paper smooth {p_blue[sel].min():.0f}-{p_blue[sel].max():.0f})')
    for name, y in (('Tempestovsky smooth', rho), ('Tempestovsky rough', rho_r), ('Tempestovsky smooth -1 sigma', rho * (1 - rel)),
                    ('paper smooth', p_blue), ('paper rough', p_red)):
        c = crossing(y, RHO_ANALOG[0])
        print(f'{name:29s} reaches {RHO_ANALOG[0]:.0f} kg/m3 at Gamma = {", ".join(f"{x:.0f}" for x in c) if c else "-"}')

    # ---------------------------------------------------------------------------------------------- figure
    BLUE, RED, LBLUE, LRED = '#0000ff', '#ff0000', '#9db4ff', '#ffadad'
    plt.rcParams.update({'font.size': 12, 'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                         'xtick.top': True, 'ytick.right': True})
    fig, ax = plt.subplots(figsize=(7.6, 5.7))
    ax.axvspan(*TI_RANGE, color='#e5e5e5', lw=0, zorder=0)
    ax.grid(color='#d0d0d0', lw=0.6, zorder=1)
    for y in RHO_ANALOG:
        ax.axhline(y, color='black', ls='--', lw=1.6, dashes=(4, 3), zorder=2)
    ax.plot(GAMMA, p_red, color=LRED, lw=1.4, zorder=3, label='Farnocchia et al. 2024, 100% roughness')
    ax.plot(GAMMA, p_blue, color=LBLUE, lw=1.4, zorder=3, label='Farnocchia et al. 2024, smooth')
    ax.plot(GAMMA, rho_r, color=RED, lw=2.0, zorder=4, label='Tempestovsky, 100% roughness (Bennu enhancement)')
    ax.plot(GAMMA, rho, color=BLUE, lw=2.0, zorder=5, label='Tempestovsky, smooth')
    for s in (1, -1):
        ax.plot(GAMMA, rho * (1 + s * rel), color=BLUE, lw=1.6, ls='--', dashes=(4, 3), zorder=4,
                label='Tempestovsky, ±1σ (D, albedo, da/dt)' if s == 1 else None)
    ax.plot(m['g_nodes'], m['rho_nodes'], 'o', ms=4, mfc='white', mec=BLUE, mew=1.0, zorder=6, clip_on=False,
            label='Tempestovsky runs')
    ax.set_xlim(50, 1200); ax.set_ylim(500, 3050)
    ax.set_xlabel(r'Thermal inertia - J m$^{-2}$ K$^{-1}$ s$^{-1/2}$')
    ax.set_ylabel(r'Bulk density - kg/m$^3$')
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95, edgecolor='#cccccc')
    fig.tight_layout()
    f = os.path.join(HERE, 'density_fig1.png')
    fig.savefig(f, dpi=200)
    print('figure:', f)
