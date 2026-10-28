'''Side-by-side comparison of the reproduced ensemble with Figs. 2-5 of Benson et al.'''
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from flyby import lb_to_vec
from calib import to_pixel
d = np.load('ensemble_50000.npz')
fig = plt.figure(figsize=(16, 15))
# Fig. 2 heatmap
ax = fig.add_subplot(3, 2, 1)
img = np.asarray(Image.open('figs/img-000.png').convert('RGB'))
ax.imshow(img[0:1100, 1800:3120]); ax.axis('off'); ax.set_title('Benson et al. Fig. 2 (P_psi 100-700 h, P_phi_l 10-70 h)')
ax = fig.add_subplot(3, 2, 2)
h, _, _ = np.histogram2d(d['post_P_psi_h'], d['post_P_phi_l_h'], bins=[np.arange(100, 705, 5), np.arange(10, 70.25, 0.25)])
ax.imshow(h.T[::-1], cmap='hot', vmin=0, vmax=100, extent=[100, 700, 10, 70], aspect='auto')
ax.plot(263, 30.56, 'wD', ms=8, mec='k'); ax.set_xlabel('P_psi [h]'); ax.set_ylabel('P_phi_l [h]'); ax.set_title('this work, same bins and colormap')
# Fig. 3
ax = fig.add_subplot(3, 2, 3)
img = np.asarray(Image.open('figs/img-002.png').convert('RGB'))
ax.imshow(img); ax.axis('off'); ax.set_title('Benson et al. Fig. 3 (omega_e 4-24 deg/h, scaled I_d -1..1)')
ax = fig.add_subplot(3, 2, 4)
we = np.degrees(d['post_omega_e']) * 3600
ax.plot(we, d['post_Id_scaled'], ',', color='0.55')
ax.plot(np.degrees(d['pre_omega_e']) * 3600, d['pre_Id_scaled'], ',', color='r')
ax.axhline(0, color='k', ls='--'); ax.set_xlim(4, 24); ax.set_ylim(-1, 1)
ax.set_xlabel('omega_e [deg/h]'); ax.set_ylabel('scaled I_d'); ax.set_title('this work')
# Fig. 4: our poles drawn on the paper's figure
ax = fig.add_subplot(3, 2, 5)
img = np.asarray(Image.open('figs/img-004.png').convert('RGB')).astype(float)
ax.imshow((255 - 0.35 * (255 - img)).astype(np.uint8))
x, y = to_pixel(d['post_lam'], d['post_beta'], 1385.3, 1339.5, 1327.2)
ax.plot(x, y, '.', color='#1f4fd6', ms=0.6, alpha=0.25)
ax.set_xlim(250, 2100); ax.set_ylim(2450, 600); ax.axis('off')
ax.set_title('Fig. 4: paper (faded black) and this work (blue)')
# Fig. 5
ax = fig.add_subplot(3, 2, 6)
delta = np.degrees(np.arccos(np.clip((lb_to_vec(d['pre_lam'], d['pre_beta']) * lb_to_vec(d['post_lam'], d['post_beta'])).sum(1), -1, 1)))
f5 = np.loadtxt('fig5_digitized.txt')
ax.bar(f5[:, 0], f5[:, 1], width=1.0, color='0.75', edgecolor='0.4', label='Benson et al. Fig. 5 (digitized)')
hh, e = np.histogram(delta, bins=np.arange(0, 51))
ax.step(e[:-1], hh / len(delta), where='post', color='#1f4fd6', lw=2, label='this work')
ax.set_xlabel('delta [deg]'); ax.set_ylabel('fraction of runs'); ax.legend()
fig.tight_layout(); fig.savefig('figs_compare.png', dpi=80)
