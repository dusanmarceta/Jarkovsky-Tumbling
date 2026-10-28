'''Digitize Fig. 2 of Benson et al.: 2D histogram of post-flyby (P_psi, P_phi_l), MATLAB "hot" colormap, 0-100 runs.'''
import numpy as np
from PIL import Image

im = np.asarray(Image.open('figs/img-000.png').convert('RGB')).astype(float) / 255
x0, x1, y0, y1 = 1816, 3108, 22, 1084          # colormap area (original pixels)
nx, ny = 120, 240                               # P_psi 100..700 h (5 h bins), P_phi_l 10..70 h (0.25 h bins)
xe = np.linspace(x0, x1 + 1, nx + 1); ye = np.linspace(y0, y1 + 1, ny + 1)
counts = np.zeros((ny, nx))
for j in range(ny):
    for i in range(nx):
        # median color of the inner part of the bin (avoid antialiased edges)
        xa, xb = int(xe[i] + 0.3 * (xe[i+1] - xe[i])), int(xe[i] + 0.7 * (xe[i+1] - xe[i])) + 1
        ya, yb = int(ye[j] + 0.3 * (ye[j+1] - ye[j])), int(ye[j] + 0.7 * (ye[j+1] - ye[j])) + 1
        r, g, b = np.median(im[ya:yb, xa:xb].reshape(-1, 3), axis=0)
        v = 3 / 8 * r + 3 / 8 * g + 1 / 4 * b          # inverse of hot: r ramps first 3/8, g next 3/8, b last 1/4
        counts[j, i] = 100 * v
# background color of hot(64)[0] is (1/24, 0, 0) -> about 1.6 runs; subtract the colormap floor
floor = 100 * 3 / 8 * (1 / 24 * 8 / 3) / 1.0
counts = np.clip(counts - np.median(counts[counts < 3]), 0, None)
P_psi_c = 100 + 5 * (np.arange(nx) + 0.5)
P_phl_c = 70 - 0.25 * (np.arange(ny) + 0.5)       # image rows go from 70 h (top) down to 10 h
np.savez('fig2_digitized.npz', counts=counts, P_psi_c=P_psi_c, P_phi_l_c=P_phl_c)
print('total runs in the window', counts.sum().round(), ' max bin', counts.max().round(1),
      ' bins at the colormap top (>= 99):', int((counts >= 99).sum()))
k = np.unravel_index(np.argmax(counts), counts.shape)
print('peak at P_psi %.1f h, P_phi_l %.2f h' % (P_psi_c[k[1]], P_phl_c[k[0]]))
mpsi = counts.sum(0); mphl = counts.sum(1)
print('P_psi mode %.1f h, P_phi_l mode %.2f h' % (P_psi_c[np.argmax(mpsi)], P_phl_c[np.argmax(mphl)]))
print('mean P_phi_l in window %.2f h, mean P_psi in window %.1f h' % ((mphl * P_phl_c).sum() / mphl.sum(), (mpsi * P_psi_c).sum() / mpsi.sum()))
