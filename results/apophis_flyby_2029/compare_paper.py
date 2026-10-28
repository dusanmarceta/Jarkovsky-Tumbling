'''Compare the reproduced 50 000-run ensemble with the numbers quoted in Benson et al. (Sec. 3.2, 4.4).'''
import numpy as np
from flyby import lb_to_vec
from orbits import PRE, POST, obliquity
d = np.load('ensemble_50000.npz')
Pl, Pp = d['post_P_phi_l_h'], d['post_P_psi_h']
Pl0, Pp0 = d['pre_P_phi_l_h'], d['pre_P_psi_h']
def mode(x, w, lo=None):
    lo = np.floor(x.min() / w) * w if lo is None else lo
    h, e = np.histogram(x, bins=np.arange(lo, x.max() + w, w))
    return 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
rows = []
rows.append(('P_phi_l range [h]', f'{Pl.min():.1f} - {Pl.max():.1f}', '16.4 - 67.2'))
rows.append(('P_phi_l mean [h]', f'{Pl.mean():.1f}', '30.9'))
rows.append(('P_phi_l most likely [h]', f'{mode(Pl, 0.25):.1f}', '~21'))
rows.append(('P_psi range [h]', f'{Pp.min():.1f} - {Pp.max():.1f}', '95.8 - 1686.9'))
rows.append(('P_psi mean [h]', f'{Pp.mean():.1f}', '266.0'))
rows.append(('P_psi most likely [h]', f'{mode(Pp, 5.0, 100.0):.0f}', '~185'))
rows.append(('P_phi_l decreases', f'{100*np.mean(Pl < Pl0):.1f}%', '55.8%'))
rows.append(('P_psi decreases', f'{100*np.mean(Pp < Pp0):.1f}%', '59.3%'))
rows.append(('both decrease (hotspot)', f'{100*np.mean((Pl < Pl0) & (Pp < Pp0)):.1f}%', '51.2%'))
rows.append(('omega_e increases', f'{100*np.mean(d["post_omega_e"] > d["pre_omega_e"]):.1f}%', '55.8%'))
rows.append(('LAM after the flyby', f'{100*np.mean(~d["post_sam"]):.1f}%', '17.3%'))
dl = (d['post_lam'] - d['pre_lam'] + 180) % 360 - 180
rows.append(('pole longitude increases', f'{100*np.mean(dl > 0):.1f}%', '93.5%'))
rows.append(('pole closer to the ecliptic', f'{100*np.mean(np.abs(d["post_beta"]) < np.abs(d["pre_beta"])):.1f}%', '55.3%'))
delta = np.degrees(np.arccos(np.clip((lb_to_vec(d['pre_lam'], d['pre_beta']) * lb_to_vec(d['post_lam'], d['post_beta'])).sum(1), -1, 1)))
rows.append(('delta mean [deg]', f'{delta.mean():.1f}', '14.5'))
rows.append(('delta most likely [deg]', f'{mode(delta, 1.0, 0.0):.1f}', '~5'))
rows.append(('delta max [deg]', f'{delta.max():.1f}', '45.8'))
acc = d['acc_ca_deg_h2']
rows.append(('|d omega/dt| at CA max [deg/h^2]', f'{acc.max():.2f}', '4.5'))
rows.append(('|d omega/dt| at CA most likely [deg/h^2]', f'{mode(acc, 0.1, 0.0):.2f}', '2.4'))
# Eq. 6 (diurnal linear theory): cos(gamma_f)/cos(gamma_i) * sqrt(a_i/a_f)
cg = np.cos(np.radians(obliquity(d['post_lam'], d['post_beta'], POST))) / np.cos(np.radians(obliquity(d['pre_lam'], d['pre_beta'], PRE)))
ratio = cg * np.sqrt(PRE['a'] / POST['a'])
rows.append(('cos(g_f)/cos(g_i) range', f'{cg.min():.2f} - {cg.max():.2f}', '0.63 - 1.2'))
rows.append(('cos(g_f)/cos(g_i) < 1', f'{100*np.mean(cg < 1):.0f}%', '56%'))
rows.append(('Eq. 6 drift ratio range', f'{ratio.min():.2f} - {ratio.max():.2f}', '0.57 - 1.12'))
rows.append(('Eq. 6 drift ratio < 1', f'{100*np.mean(ratio < 1):.0f}%', '97%'))
print(f'{"quantity":42s} {"this work":>16s} {"Benson et al.":>16s}')
for r in rows:
    print(f'{r[0]:42s} {r[1]:>16s} {r[2]:>16s}')
print(f'\nclosest approach JD {float(d["jd_ca"]):.5f}, distance {float(d["R_ca_km"]):.0f} km (paper: 37 730 km, 2019 orbit solution)')
