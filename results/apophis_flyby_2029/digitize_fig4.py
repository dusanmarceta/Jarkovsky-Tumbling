'''Digitize Fig. 4 of Benson et al.: density of the 50 000 post-flyby poles from the coverage of the scatter plot.
Overlapping dots saturate the dense core; the density there is a lower limit.'''
import numpy as np
from PIL import Image
from scipy import ndimage
im = np.asarray(Image.open('figs/img-004.png').convert('RGB')).astype(int)
R, G, B = im[..., 0], im[..., 1], im[..., 2]
cx, cy, S = 1385.3, 1339.5, 1327.2
black = (R < 110) & (G < 110) & (B < 110)
red = (R > 150) & (G < 120) & (B < 120)
# exclude text, axes and the outer circle: keep the annulus inside beta = -32 deg, and remove long thin structures
yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
rr = np.hypot(xx - cx, yy - cy)
inside = rr < S * np.cos(np.radians(-33))
# the beta tick labels and the 'beta' label lie in the empty sector lambda 90..195 deg (no poles there)
lam_px = np.degrees(np.arctan2(-(yy - cy), -(xx - cx))) % 360
inside &= ~((lam_px > 90) & (lam_px < 195))
dots = black & inside
lab, n = ndimage.label(dots)
sizes = ndimage.sum(dots, lab, index=np.arange(1, n + 1))
single = np.median(sizes[(sizes > 3) & (sizes < 60)])
print('components', n, 'typical isolated dot area %.1f px' % single)
# coverage in cells; red pixels (polygon, markers) hide dots -> treat cells by their non-red pixels only
cell = 24
ny, nx = im.shape[0] // cell, im.shape[1] // cell
cov = np.full((ny, nx), np.nan)
for j in range(ny):
    for i in range(nx):
        sl = (slice(j * cell, (j + 1) * cell), slice(i * cell, (i + 1) * cell))
        ok = inside[sl] & ~red[sl]
        if ok.sum() > 0.3 * cell * cell:
            cov[j, i] = dots[sl][ok].mean()
c = np.clip(np.nan_to_num(cov, nan=0.0), 0, 0.995)
dens = -np.log(1 - c) / single              # dots per px^2 (Poisson overlap)
counts = dens * cell * cell
# cells hidden by red were set to 0: fill them from neighbours
hidden = np.isnan(cov) & (ndimage.uniform_filter(np.nan_to_num(cov, nan=0), 3) > 0)
filled = ndimage.generic_filter(np.where(np.isnan(cov), np.nan, counts), np.nanmean, size=3, mode='constant', cval=np.nan)
counts = np.where(hidden, np.nan_to_num(filled), counts)
sat = c >= 0.98
print('estimated number of dots %.0f (true 50 000); saturated cells %d of %d nonempty' % (counts.sum(), sat.sum(), (c > 0).sum()))
np.savez('fig4_digitized.npz', counts=counts, cell=cell, cx=cx, cy=cy, S=S, saturated=sat)
