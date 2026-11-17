'''
Figures of the Tempestovsky paper (Tumbling-Yarkovsky-paper repository). Run from the repository root:

    python results/paper/make_figures.py [output directory]      # default: my_paper/figures

All numbers come from the results directories of this repository:
  results/linear_comparison/results_2026-10-14.txt     validation against the linear theory
  documentation/readme/make_figures.py                  drift per orbit with/without the deep-temperature correction
  results/apophis_fig3/out/ti*.txt                      nominal pre-flyby drift of Apophis versus Gamma
  results/apophis_density/                              density versus Gamma, digitized curves of V2015 and F2024
  results/apophis_flyby_2029/                           post-flyby batch and the paired pre-flyby drift
The output directory gets a README with the commit of this repository the figures were made from.
'''
import os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'results', 'apophis_density'), os.path.join(REPO, 'results', 'apophis_flyby_2029'),
                os.path.join(REPO, 'documentation', 'readme')]
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(REPO, 'my_paper', 'figures')

# validated categorical slots (dataviz reference palette, light mode, all-pairs pass) + neutral greys
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
COL1 = 3.46                                    # A&A column width [in]
YR, AU = 3.15576e7, 1.495978707e11
MS2AUMYR = YR * 1e6 / AU                       # m/s -> au/Myr
D_SHAPE = 334.6                                # volume-equivalent diameter of Apophis_200.stl [m]
A_PRE, A_POST = 0.9223592206975018, 1.10312    # semi-major axis before / after the 2029 encounter [au]
MEASURED = -199.0 / AU * 1e6 * 1e4             # measured pre-flyby drift [1e-4 au/Myr]

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.linewidth': 0.6, 'axes.edgecolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'lines.linewidth': 1.2, 'legend.frameon': False, 'savefig.dpi': 300, 'pdf.fonttype': 42})


def grid(ax):
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('written', name)


def nominal_drift():
    '''Nominal pre-flyby drift of Apophis [m/s] (D_SHAPE, rho 2000, A 0.14) versus Gamma, from apophis_fig3.'''
    G = np.array([100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1200.])
    d = np.array([float(open(os.path.join(REPO, f'results/apophis_fig3/out/ti{int(g)}.txt')).readline().split(':')[1])
                  for g in G])
    return G, d


# ----------------------------------------------------------------------------------------------- validation
def fig_validation():
    from make_figures import read_comparison
    data = read_comparison()                   # gamma, P, TI, linear, full, linearized
    fig, axes = plt.subplots(2, 1, figsize=(COL1, 3.9), sharex=True)
    for ax, col, title in ((axes[0], 4, 'full model'), (axes[1], 5, 'linearized emission')):
        grid(ax)
        ax.axhline(0, color=MUTED, lw=0.6)
        for gamma, c, m in ((0, BLUE, 'o'), (45, ORANGE, 's'), (90, AQUA, '^')):
            for P, ms, ls in ((4, 3, ':'), (12, 4, '-'), (24, 3, '--')):
                s = (data[:, 0] == gamma) & (data[:, 1] == P)
                rel = 100 * (data[s, col] / data[s, 3] - 1)
                ax.plot(data[s, 2], rel, color=c, ls=ls, marker=m, ms=ms, mfc='white', mew=0.8, lw=0.9,
                        label=f'$\\gamma={gamma}^\\circ$' if P == 12 else None)
        ax.text(0.02, 0.92, f'({"ab"[col - 4]}) {title} vs. linear theory', transform=ax.transAxes, va='top', color=INK)
        ax.set_ylabel('relative difference [%]')
    axes[1].axhline(1.6, color=MUTED, lw=0.8, ls=(0, (4, 3)))       # area bias of the 80-facet sphere (caption)
    axes[0].plot([], [], color=INK2, ls=':', label='$P=4$ h')
    axes[0].plot([], [], color=INK2, ls='-', label='$P=12$ h')
    axes[0].plot([], [], color=INK2, ls='--', label='$P=24$ h')
    axes[0].legend(loc='upper right', ncol=2, handlelength=2.2, columnspacing=1.0)
    axes[0].set_ylim(-15, 62)
    axes[1].set_ylim(0, 5)
    axes[1].set_xscale('log')
    axes[1].set_xticks([200, 500, 1000, 2000])
    axes[1].set_xticklabels(['200', '500', '1000', '2000'])
    axes[1].minorticks_off()
    axes[1].set_xlim(150, 2700)
    axes[1].set_xlabel(r'thermal inertia $\Gamma$ [J m$^{-2}$ K$^{-1}$ s$^{-1/2}$]')
    fig.subplots_adjust(hspace=0.08)
    save(fig, 'fig_validation.pdf')


def fig_convergence():
    from make_figures import NO_CORRECTION, WITH_CORRECTION, LIMIT
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    grid(ax)
    ax.axhline(LIMIT, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(7.2, LIMIT + 0.01, 'converged drift', color=INK2, fontsize=6.5, ha='right', va='bottom')
    for v, c, m, lab in ((NO_CORRECTION, ORANGE, 's', 'without deep-temperature correction'),
                         (WITH_CORRECTION, BLUE, 'o', 'with deep-temperature correction')):
        ax.plot(np.arange(1, len(v) + 1), v, color=c, marker=m, ms=4, mfc='white', mew=0.9, label=lab)
    ax.set_xlabel('orbit')
    ax.set_ylabel(r'$\langle da/dt\rangle$ over the orbit [$10^{-6}$ m s$^{-1}$]')
    ax.set_xticks(range(1, 8))
    ax.set_xlim(0.7, 7.3)
    ax.set_ylim(-8.0, -7.25)
    ax.legend(loc='lower right')
    save(fig, 'fig_convergence.pdf')


# ----------------------------------------------------------------------------------------------- Apophis before 2029
def fig_drift_gamma():
    G, d = nominal_drift()
    t = d * D_SHAPE / 375.0 * MS2AUMYR * 1e4                     # V2015 setup: D = 375 m, rho 2000, A 0.14
    v = np.load(os.path.join(REPO, 'results/apophis_density/v2015_fig1_digitized.npz'))
    f = PchipInterpolator(np.log(G), t)
    g = np.exp(np.linspace(np.log(G[0]), np.log(G[-1]), 300))
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(COL1, 3.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    for a in (ax, bx):
        grid(a)
    ax.plot(v['Gamma'], v['dadt'], color=ORANGE, lw=1.4, label="Vokrouhlický et al. (2015), Fig. 1")
    ax.plot(g, f(np.log(g)), color=BLUE, lw=1.4, label='this work')
    ax.plot(G, t, 'o', color=BLUE, ms=3.5, mfc='white', mew=0.9)
    ax.set_ylabel(r'$\langle da/dt\rangle$ [$10^{-4}$ au Myr$^{-1}$]')
    ax.set_ylim(-16.5, -6)
    ax.legend(loc='upper right')
    sel = (G >= v['Gamma'].min()) & (G <= v['Gamma'].max())
    bx.plot(G[sel], t[sel] / np.interp(G[sel], v['Gamma'], v['dadt']), 'o-', color=INK2, ms=3.5, mfc='white', mew=0.9)
    bx.axhline(1, color=MUTED, lw=0.6)
    bx.set_ylim(0.98, 1.12)
    bx.set_ylabel('ratio')
    bx.set_xscale('log')
    bx.set_xticks([100, 200, 500, 1000])
    bx.set_xticklabels(['100', '200', '500', '1000'])
    bx.minorticks_off()
    bx.set_xlim(95, 1250)
    bx.set_xlabel(r'thermal inertia $\Gamma$ [J m$^{-2}$ K$^{-1}$ s$^{-1/2}$]')
    fig.subplots_adjust(hspace=0.07)
    save(fig, 'fig_drift_gamma.pdf')


def fig_density():
    import density_fig1 as dm
    m = dm.model()
    rho, rel, GAM = m['rho'], m['rel'], dm.GAMMA
    pap = np.load(os.path.join(REPO, 'results/apophis_density/fig1_digitized.npz'))
    enh = np.polyval(np.polyfit(np.log(pap['Gamma']), pap['red'] / pap['blue'], 4), np.log(GAM))
    p_blue, p_red = (np.interp(GAM, pap['Gamma'], pap[k]) for k in ('blue', 'red'))
    # V2015 Fig. 1 rescaled to the 2024 parameters (D 340 m, nominal Bond albedo via the Tempestovsky A dependence)
    v = np.load(os.path.join(REPO, 'results/apophis_density/v2015_fig1_digitized.npz'))
    r = m['runs']
    g3 = np.array(sorted(g for g in {k[0] for k in r} if all((g, a) in r for a in (0.08, 0.14, 0.20))))
    cA = [np.exp(np.polyval(c, m['A']) - np.polyval(c, 0.14)) for c in
          (np.polyfit([0.08, 0.14, 0.20], np.log([-r[(g, a)] for a in (0.08, 0.14, 0.20)]), 2) for g in g3)]
    gv = v['Gamma']
    rho_v = 2000 * v['dadt'] / MEASURED * 375 / dm.D_NOM * PchipInterpolator(np.log(g3), cA)(np.log(gv))
    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    ax.axvspan(*dm.TI_RANGE, color='#efeeea', lw=0, zorder=0)
    grid(ax)
    for y in dm.RHO_ANALOG:
        ax.axhline(y, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.fill_between(GAM, rho * (1 - rel), rho * (1 + rel), color=BLUE, alpha=0.13, lw=0, zorder=2)
    ax.plot(GAM, p_blue, color=ORANGE, lw=1.1, zorder=3, label='Farnocchia et al. (2024), smooth')
    ax.plot(GAM, p_red, color=ORANGE, lw=1.1, ls='--', zorder=3, label='Farnocchia et al. (2024), rough')
    ax.plot(gv, rho_v, color=MUTED, lw=1.1, ls=':', zorder=3, label="Vokrouhlický et al. (2015), rescaled")
    ax.plot(GAM, rho * enh, color=BLUE, lw=1.1, ls='--', zorder=4, label='this work, rough (Bennu enhancement)')
    ax.plot(GAM, rho, color=BLUE, lw=1.5, zorder=5, label='this work, smooth ($\\pm1\\sigma$ band)')
    ax.plot(m['g_nodes'], m['rho_nodes'], 'o', color=BLUE, ms=3, mfc='white', mew=0.8, zorder=6, clip_on=False)
    ax.set_xlim(50, 1200)
    ax.set_ylim(500, 3100)
    ax.set_xlabel(r'thermal inertia $\Gamma$ [J m$^{-2}$ K$^{-1}$ s$^{-1/2}$]')
    ax.set_ylabel(r'bulk density [kg m$^{-3}$]')
    ax.legend(loc='lower right', fontsize=6.3, handlelength=2.4)
    save(fig, 'fig_density.pdf')


# ----------------------------------------------------------------------------------------------- Apophis after 2029
def load_flyby():
    from drift_vs_preflyby import load
    return load()


def fig_drift_dist(sim, dig, pre_mc):
    from drift_vs_preflyby import PAPER_FIG3
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(COL1, 4.2), gridspec_kw=dict(height_ratios=[1.35, 1]))
    for a in (ax, bx):
        grid(a)
    bins = np.arange(-32, 0.001, 0.4)
    for d, c, ls, lab in ((pre_mc, MUTED, '-', 'before 2029 (MC as in V2015)'), (sim.dadt_pre, ORANGE, '-', 'before 2029, paired'),
                          (sim.dadt, BLUE, '-', 'after 2029')):
        ax.hist(d, bins=bins, density=True, histtype='step', color=c, ls=ls, lw=1.1, label=lab)
    p = PAPER_FIG3['pct']
    y = 0.142
    ax.plot([p[1], p[3]], [y, y], color=INK2, lw=2.2, solid_capstyle='butt')
    ax.plot([p[0], p[4]], [y, y], color=INK2, lw=0.7)
    ax.plot(p[2], y, 'o', color=INK2, ms=3)
    ax.text(p[0], y + 0.008, 'Vokrouhlický et al. (2015), Fig. 3', color=INK2, fontsize=6.5, ha='left', va='bottom')
    ax.axvline(MEASURED, color=INK, lw=0.7, ls=(0, (4, 3)))
    ax.text(MEASURED + 0.4, 0.205, 'measured', color=INK, fontsize=6.5, ha='left', va='center')
    ax.set_xlim(-32, 0)
    ax.set_ylim(0, 0.225)
    ax.set_xlabel(r'$\langle da/dt\rangle$ [$10^{-4}$ au Myr$^{-1}$]')
    ax.set_ylabel('probability density')
    ax.legend(loc='upper left', fontsize=6.3)
    ax.text(0.98, 0.95, '(a)', transform=ax.transAxes, ha='right', va='top')
    rb = np.arange(0.5, 1.4001, 0.01)
    bx.hist(sim.ratio_eq6, bins=rb, density=True, histtype='step', color=MUTED, lw=1.1, label='Eq. (6) of Benson et al. (2023)')
    bx.hist(sim.ratio, bins=rb, density=True, histtype='step', color=BLUE, lw=1.1, label='this work')
    bx.axvline(1, color=INK, lw=0.6)
    bx.set_xlim(0.5, 1.4)
    bx.set_xlabel(r'$\langle da/dt\rangle_{\rm after}\,/\,\langle da/dt\rangle_{\rm before}$')
    bx.set_ylabel('probability density')
    bx.legend(loc='upper right', fontsize=6.3)
    bx.text(0.02, 0.95, '(b)', transform=bx.transAxes, ha='left', va='top')
    fig.subplots_adjust(hspace=0.38)
    save(fig, 'fig_drift_dist.pdf')


def theta_scaling(Gq):
    '''Orbit change alone: (a_f/a_i)^(-1/2) D(Gamma (a_f/a_i)^(3/2)) / D(Gamma), D = nominal pre-flyby drift curve.'''
    G, d = nominal_drift()
    f = PchipInterpolator(np.log(G), np.log(-d))
    ar = A_POST / A_PRE
    return ar ** -0.5 * np.exp(f(np.log(Gq * ar ** 1.5)) - f(np.log(Gq)))


def fig_ratio_gamma(sim):
    fig, ax = plt.subplots(figsize=(COL1, 2.8))
    grid(ax)
    ax.hexbin(sim.thermal_inertia, sim.ratio, gridsize=(45, 40), xscale='log', extent=(2.0, np.log10(1200), 0.6, 1.4),
              cmap='Greys', mincnt=1, bins='log', linewidths=0, alpha=0.55, zorder=1)
    edges = np.exp(np.linspace(np.log(100), np.log(1200), 13))
    mid, q16, q50, q84 = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = sim.ratio[(sim.thermal_inertia >= lo) & (sim.thermal_inertia < hi)]
        if len(s) > 200:
            mid.append(np.sqrt(lo * hi)); q16.append(s.quantile(0.16)); q50.append(s.median()); q84.append(s.quantile(0.84))
    mid = np.array(mid)
    ax.errorbar(mid, q50, yerr=[np.array(q50) - q16, np.array(q84) - q50], fmt='o', color=BLUE, ms=3.5, mfc='white',
                mew=0.9, elinewidth=0.9, capsize=1.5, zorder=4, label='this work: median, 16-84%')
    g = np.exp(np.linspace(np.log(100), np.log(900), 200))
    ax.plot(g, theta_scaling(g), color=BLUE, lw=1.2, ls='--', zorder=3, label=r'orbit change only, $\Theta$ scaling')
    ax.plot(600, 0.927, 'D', color=BLUE, ms=3.5, zorder=5, label='orbit change only, full run')
    ax.axhline(np.median(sim.ratio_eq6), color=MUTED, lw=1.1, zorder=2, label='Benson et al. (2023), Eq. (6)')
    ax.axhline(A_POST / A_PRE, color=ORANGE, lw=1.1, ls='-.', zorder=2, label=r'Ďurech et al. (2026), $\propto a$')
    ax.axhline(1, color=INK, lw=0.6, zorder=2)
    ax.set_xscale('log')
    ax.set_xticks([100, 200, 500, 1000])
    ax.set_xticklabels(['100', '200', '500', '1000'])
    ax.minorticks_off()
    ax.set_xlim(100, 1200)
    ax.set_ylim(0.7, 1.32)
    ax.set_xlabel(r'thermal inertia $\Gamma$ [J m$^{-2}$ K$^{-1}$ s$^{-1/2}$]')
    ax.set_ylabel(r'$\langle da/dt\rangle_{\rm after}\,/\,\langle da/dt\rangle_{\rm before}$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=6.3, ncol=2, handlelength=2.2, columnspacing=1.0)
    save(fig, 'fig_ratio_gamma.pdf')
    return mid, q16, q50, q84


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_validation()
    fig_convergence()
    fig_drift_gamma()
    fig_density()
    sim, dig, pre_mc = load_flyby()
    fig_drift_dist(sim, dig, pre_mc)
    mid, q16, q50, q84 = fig_ratio_gamma(sim)
    for x, a, b, c in zip(mid, q16, q50, q84):
        print(f'Gamma {x:6.0f}: ratio median {b:.3f} [{a:.3f}, {c:.3f}], Theta scaling {theta_scaling(np.array([x]))[0]:.3f}')
    commit = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', REPO, 'status', '--porcelain', 'results'], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(OUT, 'README.md'), 'w') as f:
        f.write('Figures written by `results/paper/make_figures.py` of the Tempestovsky repository '
                '(github.com/dusanmarceta/Jarkovsky-Tumbling, branch tumbling), '
                f'commit {commit}{" + uncommitted changes in results/" if dirty else ""}.\n'
                'Do not edit them here; change the script and rerun it.\n')
