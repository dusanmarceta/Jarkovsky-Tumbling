'''
Yarkovsky drift of Apophis after the 2029 encounter (Tempestovsky batch, drift_runs.csv.gz) compared with the drift
before the encounter.

    python results/apophis_flyby_2029/drift_vs_preflyby.py        # paths are relative to this file

After the flyby: the sim and dig sets of the batch, one Tempestovsky run per post-flyby state.

Before the flyby: the Tempestovsky grid of results/apophis_fig3 (nominal tumbling state on the pre-flyby orbit,
<da/dt> in thermal inertia, Bond albedo and obliquity), used in two ways:
  * paired: every run of the sim set on its own pre-flyby pole (ensemble_50000.npz, row k = run k), with the same
    Gamma, A, D, rho, roughness and shape noise as after the flyby. D, rho, roughness and shape noise cancel in the
    ratio after/before, which is compared with the diurnal linear theory (Eq. 6 of Benson et al. 2022).
  * Fig. 3 MC: the independent reproduction of Vokrouhlicky et al. (2015) Fig. 3 (results/apophis_fig3/fig3_samples.npz;
    obliquity from Pravec et al. 2014 instead of the pre-flyby poles of the flyby ensemble).

Approximations of the paired pre-flyby drift: the grid has the nominal inertias (I1 = 0.61, I2 = 0.965) and the pole
longitude of the nominal state for every pre-flyby state (the flyby ensemble varies I1, I2 at the same periods), and
it is linear in log Gamma between the grid nodes. Outside the grid (A < 0.08 or > 0.20, obliquity < 145 or > 177 deg)
it is extrapolated linearly; fig3_mc.py clips to the grid edge instead.

load() returns the merged tables (post-flyby results + inputs + paired pre-flyby drift) for further analysis.
'''
import glob, os, re
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from collect import MS2AUMYR, D_SHAPE, RHO_SIM
from orbits import PRE, POST, obliquity

HERE = os.path.dirname(os.path.abspath(__file__))
FIG3 = os.path.join(HERE, '..', 'apophis_fig3')
OBL_NOM = 167.201                     # obliquity of the nominal pre-flyby state, as in fig3_mc.py [deg]
PCT = [0.135, 15.865, 50, 84.135, 99.865]
PAPER_FIG3 = dict(mean=-12.80, sd=3.60, pct=[-26.40, -15.70, -12.40, -8.60, -3.40])   # Vokrouhlicky et al. 2015


def preflyby_grid():
    '''Pre-flyby Tempestovsky drift [m/s] for D_SHAPE and RHO_SIM as a function of (log Gamma, A, obliquity).'''
    read = lambda f: float(open(f).read().split(':')[1])
    grid = {}
    for f in glob.glob(os.path.join(FIG3, 'out3d', 'g*_A*_o[0-9][0-9][0-9].txt')):
        g, a, o = re.match(r'g(\d+)_A([\d.]+)_o(\d+)\.txt', os.path.basename(f)).groups()
        grid[(float(g), float(a), OBL_NOM if o == '167' else float(o))] = read(f)
    G = sorted({k[0] for k in grid})
    for g in G:                                    # nominal state (A = 0.14, nominal obliquity) is in out/
        grid[(g, 0.14, OBL_NOM)] = read(os.path.join(FIG3, 'out', f'ti{int(g)}.txt'))
    AL, OB = sorted({k[1] for k in grid}), sorted({k[2] for k in grid})
    tab = np.array([[[grid[(g, a, o)] for o in OB] for a in AL] for g in G])
    return RegularGridInterpolator((np.log(G), AL, OB), tab, bounds_error=False, fill_value=None), (G, AL, OB)


def load():
    res = pd.read_csv(os.path.join(HERE, 'drift_runs.csv.gz'))
    res = res[res.valid]
    sets = {}
    for name in ('sim', 'dig'):
        x = res[res.set == f'runs_{name}'].merge(pd.read_csv(os.path.join(HERE, f'inputs_{name}.csv.gz')), on='run')
        x['dadt'] = x.dadt_au_Myr * 1e4                                       # 1e-4 au/Myr
        x['obl_post'] = 90.0 - x.beta_L
        sets[name] = x
    sim = sets['sim']
    ens = np.load(os.path.join(HERE, 'ensemble_50000.npz'))
    sim['obl_pre'] = obliquity(ens['pre_lam'], ens['pre_beta'], PRE)[sim.run]
    interp, (G, AL, OB) = preflyby_grid()
    sim['drift_pre_m_s'] = interp(np.column_stack([np.log(sim.thermal_inertia), sim.albedo, sim.obl_pre]))
    scale = (D_SHAPE / sim.D) * (RHO_SIM / sim.rho) * sim.roughness * sim.shape_noise
    sim['dadt_pre'] = sim.drift_pre_m_s * MS2AUMYR * 1e4 * scale
    sim['ratio'] = sim.drift_m_s / sim.drift_pre_m_s                          # after / before
    sim['ratio_eq6'] = (np.cos(np.radians(sim.obl_post)) / np.cos(np.radians(sim.obl_pre))
                        * np.sqrt(PRE['a'] / POST['a']))
    sim.attrs['outside_grid'] = float(np.mean((sim.albedo < AL[0]) | (sim.albedo > AL[-1])
                                              | (sim.obl_pre < OB[0]) | (sim.obl_pre > OB[-1])))
    pre_mc = np.load(os.path.join(FIG3, 'fig3_samples.npz'))['dadt']
    return sim, sets['dig'], pre_mc


def stats(name, d):
    p = np.percentile(d, PCT)
    print(f'{name:34s} {len(d):7d} {d.mean():7.2f} {d.std():5.2f} {p[2]:7.2f}   [{p[1]:6.2f}, {p[3]:6.2f}]'
          f'   [{p[0]:6.2f}, {p[4]:6.2f}]')
    return p


def ratio_stats(name, r):
    p = np.percentile(r, [2.5, 16, 50, 84, 97.5])
    print(f'{name:34s} {p[2]:6.3f}   [{p[1]:.3f}, {p[3]:.3f}]   [{p[0]:.3f}, {p[4]:.3f}]   {100 * np.mean(r < 1):5.1f}%')


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sim, dig, pre_mc = load()
    print(f'da/dt [1e-4 au/Myr] (paired pre-flyby drift extrapolated outside the grid for '
          f'{100 * sim.attrs["outside_grid"]:.1f}% of the runs)')
    print(f'{"":34s} {"N":>7s} {"mean":>7s} {"sd":>5s} {"median":>7s}   {"1 sigma":^16s}   {"3 sigma":^16s}')
    p_pre = stats('before, paired (sim pre-flyby poles)', sim.dadt_pre)
    p_mc = stats('before, Fig. 3 MC (apophis_fig3)', pre_mc)
    p = PAPER_FIG3['pct']
    print(f'{"before, Vokrouhlicky et al. Fig. 3":34s} {"":7s} {PAPER_FIG3["mean"]:7.2f} {PAPER_FIG3["sd"]:5.2f} '
          f'{p[2]:7.2f}   [{p[1]:6.2f}, {p[3]:6.2f}]   [{p[0]:6.2f}, {p[4]:6.2f}]')
    p_sim = stats('after, sim', sim.dadt)
    stats('after, dig', dig.dadt)
    for s, lab in ((1, 'SAM'), (0, 'LAM')):
        stats(f'after, sim, {lab} after the flyby', sim.dadt[sim.sam == s])

    print(f'\nratio after/before (sim, paired)   median   16-84%            2.5-97.5%        |da/dt| decreases')
    ratio_stats('Tempestovsky', sim.ratio)
    ratio_stats('Eq. 6 (diurnal linear theory)', sim.ratio_eq6)
    for s, lab in ((1, 'SAM'), (0, 'LAM')):
        ratio_stats(f'Tempestovsky, {lab} after the flyby', sim.ratio[sim.sam == s])
    q = sim.ratio / sim.ratio_eq6
    print(f'Tempestovsky / Eq. 6: median {q.median():.3f}, 16-84% [{q.quantile(0.16):.3f}, {q.quantile(0.84):.3f}], '
          f'Pearson r = {np.corrcoef(sim.ratio, sim.ratio_eq6)[0, 1]:.3f}')
    print('(orbit change alone, nominal state and pole, README: 0.927; Eq. 6: 0.912)')

    # ---------------------------------------------------------------------------------------------- figure
    BLUE, ORANGE, AQUA, GREY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#52514e'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6))

    bins = np.arange(-35, 0.001, 0.25)
    curves = [(sim.dadt_pre, ORANGE, '-', 'before, paired (sim pre-flyby poles)'),
              (pre_mc, GREY, ':', 'before, Fig. 3 MC (Pravec obliquity)'),
              (sim.dadt, BLUE, '-', 'after, sim'),
              (dig.dadt, AQUA, '--', 'after, dig')]
    hmax = 0
    for d, c, ls, lab in curves:
        h, _, _ = ax.hist(d, bins=bins, density=True, histtype='step', color=c, ls=ls, lw=1.5, label=lab)
        hmax = max(hmax, h.max())
    bars = [(p, GREY, 'before, Vokrouhlický et al. 2015 Fig. 3'), (p_pre, ORANGE, 'before, paired'),
            (p_sim, BLUE, 'after, sim')]
    for k, (pp, c, lab) in enumerate(bars):          # median, 1 sigma, 3 sigma, labelled directly
        y = hmax * (1.08 + 0.12 * k)
        ax.plot([pp[1], pp[3]], [y, y], color=c, lw=3, solid_capstyle='round')
        ax.plot([pp[0], pp[4]], [y, y], color=c, lw=1)
        ax.plot(pp[2], y, 'o', color=c, ms=5)
        ax.text(pp[0], y + 0.035 * hmax, f'{lab}  (median, 1σ, 3σ)', color=MUTED, fontsize=7.5, va='bottom')
    ax.set_xlim(-35, 0); ax.set_ylim(0, 1.48 * hmax)
    ax.set_xlabel(r'$da/dt$  [$10^{-4}$ au/Myr]'); ax.set_ylabel(r'probability density  [$10^{4}$ Myr/au]')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.69), fontsize=8, frameon=False)
    ax.set_title('Yarkovsky drift before and after the 2029 encounter', fontsize=10, color=INK, loc='left')

    rb = np.arange(0.4, 1.3001, 0.01)
    for r, c, ls, lab in [(sim.ratio, BLUE, '-', 'Tempestovsky'), (sim.ratio_eq6, GREY, '--', 'Eq. 6 (diurnal linear theory)')]:
        bx.hist(r, bins=rb, density=True, histtype='step', color=c, ls=ls, lw=1.5,
                label=f'{lab}: median {np.median(r):.3f}, |da/dt| decreases in {100 * np.mean(r < 1):.0f}%')
    bx.axvline(1.0, color=MUTED, lw=0.8, ls=':')
    bx.set_xlim(rb[0], rb[-1])
    bx.set_xlabel('da/dt after / before  (sim set, same thermal and physical parameters)')
    bx.set_ylabel('probability density')
    bx.legend(loc='upper left', fontsize=8, frameon=False)
    bx.set_title('Drift ratio per run', fontsize=10, color=INK, loc='left')
    for a in (ax, bx):
        a.grid(axis='y', color='#e4e3df', lw=0.6); a.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(HERE, 'drift_vs_preflyby.png')
    fig.savefig(out, dpi=150)
    print('figure:', out)
