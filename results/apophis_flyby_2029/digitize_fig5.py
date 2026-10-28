'''Digitize Fig. 5 of Benson et al.: histogram of the angle delta between pre- and post-flyby poles (1 deg bins).'''
import numpy as np
from PIL import Image
im = np.asarray(Image.open('figs/img-006.png').convert('L')).astype(int)
xl, xr, yb, yt = 277, 2260, 1556, 26          # axes: delta 0..50 deg, fraction 0..0.06
frac = []
for k in range(50):
    c = int(round(xl + (k + (0.75 if k == 0 else 0.25 if k == 49 else 0.5)) * (xr - xl) / 50))   # avoid the axis ticks
    col = im[yt + 3:yb - 2, c]
    nz = np.where(col < 250)[0]
    top = (yt + 3 + nz[0]) if len(nz) else yb
    frac.append((yb - top) / (yb - yt) * 0.06)
frac = np.array(frac)
np.savetxt('fig5_digitized.txt', np.c_[np.arange(50) + 0.5, frac], fmt='%.5f', header='delta bin center [deg], fraction of runs (Benson et al. Fig. 5)')
c = np.arange(50) + 0.5
print('sum of fractions %.4f, mean delta %.2f deg, mode %.1f deg, last nonzero bin %.1f' % (frac.sum(), (frac * c).sum() / frac.sum(), c[np.argmax(frac)], c[np.nonzero(frac > 1e-4)[0][-1]]))
print(np.round(frac, 4))
