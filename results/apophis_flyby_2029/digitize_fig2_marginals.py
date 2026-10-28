'''Digitize the marginal histograms of Fig. 2 (Benson et al.): P_phi_l (0.25 h bins, 10-70 h) and P_psi (5 h bins, 100-700 h).'''
import numpy as np
from PIL import Image
im = np.asarray(Image.open('figs/img-000.png').convert('L')).astype(int)
nonwhite = im < 235
# tick marks would read as bars: blank them (ticks are ~25 px long, the bars never reach these zones)
xl0_, xl1_, yl0_, yl1_ = 198, 1490, 22, 1084
nonwhite[:, :xl0_ + 30] = False                                     # left axis of the P_phi_l panel and its ticks
nonwhite[np.r_[yl0_:yl0_ + 45, yl1_ - 45:yl1_ + 1][:, None], np.arange(836, 853)[None, :]] = False   # 0.01 ticks
xb0_, xb1_, yb0_, yb1_ = 1816, 3108, 1382, 2443
nonwhite[yb1_ - 30:yb1_ + 5, xb0_:xb1_ + 1] = False                  # bottom axis of the P_psi panel and its ticks
for fr in (0.01, 0.02, 0.03):                                       # fraction ticks on the left and right axes
    y = int(round(yb0_ + fr / 0.04 * (yb1_ - yb0_)))
    nonwhite[y - 4:y + 5, xb0_:xb0_ + 32] = False
    nonwhite[y - 4:y + 5, xb1_ - 32:xb1_ + 1] = False
# left panel: fraction axis 0.02 (x = 198) .. 0 (x = 1490); P_phi_l 70 h (y = 22) .. 10 h (y = 1084)
xl0, xl1, yl0, yl1 = 198, 1490, 22, 1084
f_phl = []
for j in range(240):
    vals = []
    for t in (0.3, 0.5, 0.7):
        r = int(yl0 + (j + t) * (yl1 - yl0) / 240)
        nz = np.where(nonwhite[r, xl0 + 3:xl1 - 2])[0]
        vals.append((xl1 - (xl0 + 3 + nz[0])) / (xl1 - xl0) * 0.02 if len(nz) else 0.0)
    f_phl.append(np.median(vals))      # the median rejects the dashed line at 30.56 h
f_phl = np.array(f_phl)
P_phl_c = 70 - 0.25 * (np.arange(240) + 0.5)
# bottom panel: fraction 0 (y = 1382) .. 0.04 (y = 2443); P_psi 100 (x = 1816) .. 700 h (x = 3108)
xb0, xb1, yb0, yb1 = 1816, 3108, 1382, 2443
f_psi = []
for i in range(120):
    vals = []
    for t in (0.3, 0.5, 0.7):
        c = int(xb0 + (i + t) * (xb1 - xb0) / 120)
        nz = np.where(nonwhite[yb0 + 3:yb1 - 2, c])[0]
        vals.append(((yb0 + 3 + nz[-1]) - yb0) / (yb1 - yb0) * 0.04 if len(nz) else 0.0)
    f_psi.append(np.median(vals))      # rejects the dashed line at 263 h
f_psi = np.array(f_psi)
P_psi_c = 100 + 5 * (np.arange(120) + 0.5)
# bins crossed by the dashed pre-flyby lines (30.56 h, 263 h): interpolate from the neighbours
for arr, centers, value, width in ((f_phl, P_phl_c, 30.56, 0.25), (f_psi, P_psi_c, 263.0, 5.0)):
    bad = np.abs(centers - value) < 0.5 * width + (1.2 if width < 1 else 3.0)
    good = ~bad
    arr[bad] = np.interp(centers[bad], centers[good][::np.sign(centers[1] - centers[0]).astype(int)],
                         arr[good][::np.sign(centers[1] - centers[0]).astype(int)])
    print('interpolated bins at', centers[bad])
np.savez('fig2_marginals.npz', f_phi_l=f_phl, P_phi_l_c=P_phl_c, f_psi=f_psi, P_psi_c=P_psi_c)
print('P_phi_l marginal: sum %.4f (all runs), mode %.2f h, mean %.2f h' % (f_phl.sum(), P_phl_c[np.argmax(f_phl)], (f_phl * P_phl_c).sum() / f_phl.sum()))
print('P_psi marginal: sum %.4f (runs with P_psi < 700 h), mode %.1f h' % (f_psi.sum(), P_psi_c[np.argmax(f_psi)]))
