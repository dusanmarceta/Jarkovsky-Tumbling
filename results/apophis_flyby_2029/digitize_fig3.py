'''Digitize Fig. 3 of Benson et al.: region of the (omega_e, scaled I_d) plane populated by the 50 000 post-flyby states
(occupancy of the grey scatter dots; used as a constraint for the digitized ensemble).'''
import numpy as np
from PIL import Image
from scipy import ndimage
im = np.asarray(Image.open('figs/img-002.png').convert('RGB')).astype(int)
R, G, B = im[..., 0], im[..., 1], im[..., 2]
x0, x1, y0, y1 = 204, 2947, 29, 1698            # axes box: omega_e 4..24 deg/h, scaled I_d +1 (top) .. -1 (bottom)
grey = (np.abs(R - 153) < 30) & (np.abs(G - 153) < 30) & (np.abs(B - 153) < 30)
nx, ny = 80, 80                                  # 0.25 deg/h x 0.025 cells
occ = np.zeros((ny, nx), int)
xe = np.linspace(x0, x1, nx + 1).astype(int); ye = np.linspace(y0, y1, ny + 1).astype(int)
for j in range(ny):
    for i in range(nx):
        occ[j, i] = grey[ye[j] + 1:ye[j + 1] - 1, xe[i] + 1:xe[i + 1] - 1].sum()
mask = occ >= 3
mask = ndimage.binary_closing(mask, iterations=1)        # fill isolated holes (red line, dashed line)
lab, n = ndimage.label(mask)                             # the 'SAM'/'LAM' labels have grey antialiased edges:
mask = lab == 1 + np.argmax(ndimage.sum(mask, lab, range(1, n + 1)))   # keep only the main cloud
omega_e = 4 + 0.25 * (np.arange(nx) + 0.5)
Id = 1 - 0.025 * (np.arange(ny) + 0.5)
cell_area = np.mean(np.diff(xe) - 2) * np.mean(np.diff(ye) - 2)
np.savez('fig3_digitized.npz', mask=mask, counts=occ, cell_area=cell_area, omega_e_c=omega_e, Id_scaled_c=Id)
print('occupied cells', mask.sum(), 'of', mask.size, '| scaled I_d range', Id[mask.any(1)].min(), Id[mask.any(1)].max(),
      '| omega_e range', omega_e[mask.any(0)].min(), omega_e[mask.any(0)].max())
