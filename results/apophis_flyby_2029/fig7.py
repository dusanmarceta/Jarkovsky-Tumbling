'''Fig. 7 of Benson et al.: post-flyby poles for the nominal pre-flyby pole and periods, 5000 runs per inertia pair.'''
import numpy as np, time
from flyby import *
from calib import to_pixel
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

cases = [(0.58, 0.965), (0.61, 0.965), (0.64, 0.965), (0.61, 0.95), (0.61, 0.974)]
dt = 30.0
g, rhat, n_steps, jd_ca, Rmin = trajectory(dt)
rng = np.random.default_rng(7)
res = {}
for I1v, I2v in cases:
    N = 5000
    I1 = np.full(N, I1v); I2 = np.full(N, I2v)
    poles = np.tile([250.0, -75.0], (N, 1))
    t = time.time()
    q0, w0, Id0, ok = initial_states(I1, I2, poles, rng)
    qf, wf, acc = integrate(q0, w0, np.c_[I1, I2, np.ones(N)], g, rhat, dt, n_steps, n_steps // 2)
    fs = final_state(qf, wf, I1, I2)
    res[(I1v, I2v)] = fs
    print(f'I_l {I1v} I_i {I2v}: pre I_d {Id0[0]:.5f}, {time.time()-t:.0f} s, lam {fs["lam"].min():.1f}..{fs["lam"].max():.1f}, '
          f'beta {fs["beta"].min():.1f}..{fs["beta"].max():.1f}, LAM {100*np.mean(~fs["sam"]):.1f}%', flush=True)
np.savez('fig7_runs.npz', **{f'{a}_{b}_{k}': v for (a, b), fs in res.items() for k, v in fs.items()})

calib = dict(cx=1352.4, cy=1309.4, S=1291.3)
panels = [('figs/img-012.png', [(0.58, 0.965), (0.61, 0.965), (0.64, 0.965)]),
          ('figs/img-014.png', [(0.61, 0.95), (0.61, 0.965), (0.61, 0.974)])]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for row, (path, cs) in enumerate(panels):
    img = np.asarray(Image.open(path).convert('RGB'))
    for col, c in enumerate(cs):
        ax = axes[row, col]
        ax.imshow(img)
        x, y = to_pixel(res[c]['lam'], res[c]['beta'], calib['cx'], calib['cy'], calib['S'])
        ax.plot(x, y, ',', color='k', alpha=0.5)
        ax.set_xlim(450, 1650); ax.set_ylim(2150, 1150)
        ax.set_title(f'I_l = {c[0]}, I_i = {c[1]}: this work (black) over Benson et al. Fig. 7', fontsize=9)
        ax.axis('off')
fig.tight_layout()
fig.savefig('fig7_compare.png', dpi=110)
