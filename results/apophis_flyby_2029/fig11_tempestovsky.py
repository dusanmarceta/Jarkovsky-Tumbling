'''
Fig. 11 of Benson et al. (2022) with the Tempestovsky drift: distribution of the relative post-flyby Yarkovsky
semi-major axis drift rate a_f'/a_i' for the 50 000 runs of the sim set.

    python results/apophis_flyby_2029/fig11_tempestovsky.py      # -> fig11_tempestovsky.png

a_f' is the Tempestovsky run of each post-flyby state, a_i' the pre-flyby drift of the same run (its own pre-flyby
pole, same thermal inertia and albedo) from the grid of results/apophis_fig3 (see drift_vs_preflyby.py). D, rho,
roughness and shape noise cancel in the ratio. The paper's ratio is Eq. 6 (diurnal linear theory),
sqrt(a_i/a_f) cos(gamma_f) / cos(gamma_i). The digitized set has no pre-flyby pole per run, so it has no ratio.
'''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from drift_vs_preflyby import HERE, load

BIN = 0.005                                     # bin width of the paper's histogram
XLIM = (0.55, 1.45)                             # paper: 0.55-1.15; the Tempestovsky ratios reach 1.42

sim, _, _ = load()
r = sim.ratio.values
for name, x, paper in (('Tempestovsky', r, ''), ('Eq. 6, same runs', sim.ratio_eq6.values, '   (paper: 0.57-1.12, 97%)')):
    print(f'{name:18s} min {x.min():.3f}  max {x.max():.3f}  median {np.median(x):.3f}  '
          f'reduction in {100 * np.mean(x < 1):.1f}% of runs{paper}')

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 15, 'mathtext.fontset': 'cm',
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'axes.linewidth': 0.8})
fig, ax = plt.subplots(figsize=(7.0, 5.0))
bins = np.arange(XLIM[0], XLIM[1] + BIN / 2, BIN)
ax.hist(r, bins=bins, weights=np.full(len(r), 1 / len(r)), color='#b3b3b3', edgecolor='black', linewidth=0.4)
ax.set_xlim(*XLIM)
ax.set_ylim(0, None)
ax.set_xlabel(r'$\dot{a}_f\,/\,\dot{a}_i$', fontsize=18)
ax.set_ylabel('Fraction of Runs', fontsize=18)
fig.tight_layout()
out = os.path.join(HERE, 'fig11_tempestovsky.png')
fig.savefig(out, dpi=200)
print('figure:', out)
