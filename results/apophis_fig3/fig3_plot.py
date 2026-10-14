import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = np.load('fig3_samples.npz')
h, c, p = d['hist'], d['cent'], d['pct']
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(c, h, 'k-', lw=1.5, label='Tempestovsky (Apophis_200, tumbling)')
y = 1.05 * h.max()
ax.plot([p[1], p[3]], [y, y], 'k-', lw=3); ax.plot([p[0], p[4]], [y * 1.04] * 2, 'k-', lw=1)
ax.plot(p[2], y, 'ko')
yp = 1.14 * h.max()   # paper Fig. 3 summary
ax.plot([-15.7, -8.6], [yp, yp], 'r-', lw=3); ax.plot([-26.4, -3.4], [yp * 1.035] * 2, 'r-', lw=1)
ax.plot(-12.4, yp, 'ro', label='Vokrouhlický et al. 2015, Fig. 3 (median, 1σ, 3σ)')
ax.set_xlim(-35, 0); ax.set_ylim(0, 1.25 * h.max())
ax.set_xlabel(r'$da/dt$  [$10^{-4}$ au/Myr]'); ax.set_ylabel(r'probability density  [$10^{4}$ Myr/au]')
ax.legend(loc='upper left', fontsize=8)
fig.tight_layout(); fig.savefig('fig3_tempestovsky.png', dpi=150)
