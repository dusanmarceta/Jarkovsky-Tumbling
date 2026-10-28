'''Summary figure of the two post-flyby ensembles (no reproduction of the paper's figures, only digitized values).'''
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flyby import lb_to_vec
from orbits import PRE, POST, obliquity
sim = np.load('ensemble_50000.npz'); dig = np.load('digitized_50000.npz')
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
a = ax[0, 0]
h, _, _ = np.histogram2d(sim['post_P_psi_h'], sim['post_P_phi_l_h'], bins=[np.arange(100, 705, 5), np.arange(10, 70.25, 0.25)])
a.imshow(h.T[::-1], cmap='hot', vmin=0, vmax=100, extent=[100, 700, 10, 70], aspect='auto')
a.plot(263, 30.56, 'wD', ms=7, mec='k'); a.set_xlabel('P_psi [h]'); a.set_ylabel('P_phi_l [h]')
a.set_title('post-flyby periods (re-simulated, runs per bin)', fontsize=10)
a = ax[0, 1]
delta = np.degrees(np.arccos(np.clip((lb_to_vec(sim['pre_lam'], sim['pre_beta']) * lb_to_vec(sim['post_lam'], sim['post_beta'])).sum(1), -1, 1)))
f5 = np.loadtxt('fig5_digitized.txt')
a.bar(f5[:, 0], f5[:, 1], width=1.0, color='0.8', edgecolor='0.5', label='Benson et al. Fig. 5 (digitized)')
hh, e = np.histogram(delta, bins=np.arange(0, 51)); a.step(e[:-1], hh / len(delta), where='post', color='#2a78d6', lw=2, label='re-simulated')
a.set_xlabel('pole shift delta [deg]'); a.set_ylabel('fraction of runs'); a.legend(fontsize=8); a.set_title('pole shift through the flyby', fontsize=10)
a = ax[1, 0]
bins = np.arange(110, 181, 1.0)
a.hist(obliquity(sim['pre_lam'], sim['pre_beta'], PRE), bins=bins, histtype='step', color='0.4', lw=1.5, density=True, label='before (sampled region)')
a.hist(obliquity(sim['post_lam'], sim['post_beta'], POST), bins=bins, histtype='step', color='#2a78d6', lw=2, density=True, label='after, re-simulated')
a.hist(obliquity(dig['post_lam'], dig['post_beta'], POST), bins=bins, histtype='step', color='#eb6834', lw=1.5, density=True, label='after, digitized figures')
a.set_xlabel('obliquity [deg] (own orbit)'); a.set_ylabel('density'); a.legend(fontsize=8); a.set_title('obliquity', fontsize=10)
a = ax[1, 1]
bins = np.linspace(0.84, 1.0, 81)
a.hist(sim['post_I_d'], bins=bins, histtype='step', color='#2a78d6', lw=2, density=True, label='after, re-simulated')
a.hist(dig['post_I_d'], bins=bins, histtype='step', color='#eb6834', lw=1.5, density=True, label='after, digitized figures')
a.hist(sim['pre_I_d'], bins=bins, histtype='step', color='0.4', lw=1.5, density=True, label='before')
a.set_xlabel('I_d / I_s  (= 1 / E_ratio)'); a.set_ylabel('density'); a.legend(fontsize=8); a.set_title('dynamic moment of inertia (below I_i: LAM)', fontsize=10)
fig.tight_layout(); fig.savefig('ensemble_summary.png', dpi=110)
