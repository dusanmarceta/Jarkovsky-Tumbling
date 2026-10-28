'''
Figures of Benson et al. (2022) next to the same quantities from the re-simulated (sim) and the digitized (dig)
post-flyby ensembles, plus the distributions of all Tempestovsky input parameters. Written to local_figures/,
which git ignores (the paper's figures are not redistributed). Run from this folder after ensemble.py,
gen_digitized.py, fig7.py and make_inputs.py:

    python make_local_figures.py
'''
import gzip, os, shutil, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from flyby import lb_to_vec
from orbits import PRE, POST, obliquity
from calib import to_pixel

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, '..', '..', 'papers', 'Spin State Evolution of (99942) Apophis during its 2029 Earth Encounter.pdf')
OUT = os.path.join(HERE, 'local_figures')
FIGS = os.path.join(OUT, 'paper')
SIM, DIG, PAPER = '#2a78d6', '#eb6834', '0.55'
os.makedirs(FIGS, exist_ok=True)
with open(os.path.join(OUT, '.gitignore'), 'w') as f:
    f.write('*\n')

# ---------------------------------------------------------------- the paper's figures
subprocess.run(['pdfimages', '-png', '-f', '7', '-l', '18', PDF, os.path.join(FIGS, 'img')], check=True)
order = sorted(f for f in os.listdir(FIGS) if f.startswith('img-') and f.endswith('.png'))
paper = {}
# pdfimages numbers images and soft masks together: keep the even (image) numbers in page order
kept = [f for f in order if int(f[4:7]) % 2 == 0]
labels = ['fig2_periods', 'fig3_spin_rate_Id', 'fig4_poles', 'fig5_delta', 'fig6a_acceleration', 'fig6b_acceleration',
          'fig7a_inertia_Il', 'fig7b_inertia_Ii', 'fig8_slopes', 'fig9_accelerations', 'fig10_contact_binary', 'fig11_drift_ratio']
for f, lab in zip(kept, labels):
    dst = os.path.join(FIGS, f'Benson2022_{lab}.png')
    shutil.move(os.path.join(FIGS, f), dst)
    paper[lab] = dst
for f in os.listdir(FIGS):
    if f.startswith('img-'):
        os.remove(os.path.join(FIGS, f))

sim = np.load('ensemble_50000.npz')
dig = np.load('digitized_50000.npz')


def paper_img(ax, lab, crop=None, title=None):
    img = np.asarray(Image.open(paper[lab]).convert('RGB'))
    if crop:
        x0, x1, y0, y1 = crop
        img = img[y0:y1, x0:x1]
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title or 'Benson et al. (2022)', fontsize=11)


# ---------------------------------------------------------------- Fig. 2: periods
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(2, 3, 1)
paper_img(ax, 'fig2_periods', (1560, 3509, 0, 1300), 'Benson et al. Fig. 2')
for k, (d, name) in enumerate([(sim, 'flyby re-simulated'), (dig, 'from digitized figures')]):
    ax = fig.add_subplot(2, 3, 2 + k)
    h, _, _ = np.histogram2d(d['post_P_psi_h'], d['post_P_phi_l_h'], bins=[np.arange(100, 705, 5), np.arange(10, 70.25, 0.25)])
    im = ax.imshow(h.T[::-1], cmap='hot', vmin=0, vmax=100, extent=[100, 700, 10, 70], aspect='auto')
    ax.plot(263, 30.56, 'wD', ms=9, mec='k')
    ax.set_xlabel('P_psi [h]'); ax.set_ylabel('P_phi_l [h]'); ax.set_title(name + f' ({len(d["I1"])} runs)', fontsize=11)
    plt.colorbar(im, ax=ax, label='number of runs')
m = np.load('fig2_marginals.npz')
ax = fig.add_subplot(2, 3, 4)
f = m['f_phi_l'] / m['f_phi_l'].sum()
ax.bar(m['P_phi_l_c'], f, width=0.25, color='0.82', edgecolor='0.6', label='paper (digitized)')
for d, c, name in [(sim, SIM, 'sim'), (dig, DIG, 'dig')]:
    h, e = np.histogram(d['post_P_phi_l_h'], bins=np.arange(10, 70.25, 0.25))
    ax.step(e[:-1], h / len(d['I1']), where='post', color=c, lw=1.5, label=name)
ax.axvline(30.56, color='k', ls='--', lw=1); ax.set_xlabel('P_phi_l [h]'); ax.set_ylabel('fraction of runs'); ax.legend()
ax = fig.add_subplot(2, 3, 5)
f = m['f_psi'] / m['f_phi_l'].sum()
ax.bar(m['P_psi_c'], f, width=5, color='0.82', edgecolor='0.6', label='paper (digitized)')
for d, c, name in [(sim, SIM, 'sim'), (dig, DIG, 'dig')]:
    h, e = np.histogram(d['post_P_psi_h'], bins=np.arange(100, 705, 5))
    ax.step(e[:-1], h / len(d['I1']), where='post', color=c, lw=1.5, label=name)
ax.axvline(263, color='k', ls='--', lw=1); ax.set_xlabel('P_psi [h]'); ax.set_ylabel('fraction of runs'); ax.legend()
ax = fig.add_subplot(2, 3, 6)
for d, c, name in [(sim, SIM, 'sim'), (dig, DIG, 'dig')]:
    ax.hist(d['post_P_psi_h'], bins=np.logspace(np.log10(80), np.log10(2500), 80), histtype='step', color=c, lw=1.5,
            weights=np.full(len(d['I1']), 1 / len(d['I1'])), label=name)
ax.set_xscale('log'); ax.axvline(263, color='k', ls='--', lw=1)
ax.set_xlabel('P_psi [h] (full range; the paper shows only 100-700 h)'); ax.set_ylabel('fraction of runs'); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_periods.png'), dpi=90); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: effective spin rate and scaled I_d
fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
paper_img(axes[0], 'fig3_spin_rate_Id', None, 'Benson et al. Fig. 3')
for ax, d, name in [(axes[1], sim, 'flyby re-simulated'), (axes[2], dig, 'from digitized figures')]:
    ax.plot(np.degrees(d['post_omega_e']) * 3600, d['post_Id_scaled'], ',', color='0.5', alpha=0.6)
    if 'pre_omega_e' in d.files:
        ax.plot(np.degrees(d['pre_omega_e']) * 3600, d['pre_Id_scaled'], ',', color='r')
    ax.axhline(0, color='k', ls='--'); ax.set_xlim(4, 24); ax.set_ylim(-1, 1)
    ax.text(21, 0.55, 'SAM', fontsize=13); ax.text(13.5, -0.5, 'LAM', fontsize=13)
    ax.set_xlabel('omega_e [deg/h]'); ax.set_ylabel('scaled I_d')
    ax.set_title(f'{name}: LAM {100 * np.mean(~d["post_sam"].astype(bool)):.1f}%', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_spin_rate_Id.png'), dpi=90); plt.close(fig)


# ---------------------------------------------------------------- Fig. 4: poles, same projection as the paper
def polar_frame(ax):
    '''Orthographic view from the south ecliptic pole: lambda 0 left, 90 up, 180 right; r = cos(beta).'''
    t = np.linspace(0, 2 * np.pi, 361)
    for b in (-80, -70, -60, -50, -40, -30):
        r = np.cos(np.radians(b))
        ax.plot(r * np.cos(t), r * np.sin(t), color='k' if b == -30 else '0.8', lw=1 if b == -30 else 0.6,
                ls='-' if b == -30 else ':')
        ax.text(r + 0.01, 0.02, f'{b}', fontsize=8, color='0.3')
    for lam in range(0, 360, 30):
        x, y = -np.cos(np.radians(lam)), np.sin(np.radians(lam))
        r = np.cos(np.radians(-30))
        ax.plot([0, r * x], [0, r * y], color='0.85', lw=0.6, ls=':')
        ax.text(1.06 * r * x, 1.06 * r * y, f'{lam}', ha='center', va='center', fontsize=9)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-0.98, 0.98); ax.set_ylim(-0.98, 0.98)


def polar_xy(lam, beta):
    r = np.cos(np.radians(beta))
    return -r * np.cos(np.radians(lam)), r * np.sin(np.radians(lam))


region = np.load('pole_region_pixels.npy')
reg_lam = (np.degrees(np.arctan2(-(region[:, 1] - 1339.068), -(region[:, 0] - 1385.846))) % 360)
reg_beta = -np.degrees(np.arccos(np.clip(np.hypot(region[:, 0] - 1385.846, region[:, 1] - 1339.068) / 1326.5, 0, 1)))
fig, axes = plt.subplots(1, 3, figsize=(21, 7.3))
paper_img(axes[0], 'fig4_poles', None, 'Benson et al. Fig. 4')
for ax, d, name in [(axes[1], sim, 'flyby re-simulated'), (axes[2], dig, 'from digitized figures')]:
    polar_frame(ax)
    x, y = polar_xy(d['post_lam'], d['post_beta'])
    ax.plot(x, y, '.', color='k', ms=1.0, alpha=0.5)
    rx, ry = polar_xy(reg_lam, reg_beta)
    ax.plot(np.r_[rx, rx[:1]], np.r_[ry, ry[:1]], color='r', lw=2)
    for (l, b), mk in [((250, -75), 'D'), ((247, -59), 's')]:
        px, py = polar_xy(l, b); ax.plot(px, py, mk, color='r', ms=10)
    ax.set_title(f'{name}: post-flyby poles (J2000 ecliptic)', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_poles.png'), dpi=90); plt.close(fig)

# ---------------------------------------------------------------- Fig. 5: pole shift (only the simulation has pre-flyby poles)
delta = np.degrees(np.arccos(np.clip((lb_to_vec(sim['pre_lam'], sim['pre_beta']) * lb_to_vec(sim['post_lam'], sim['post_beta'])).sum(1), -1, 1)))
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
paper_img(axes[0], 'fig5_delta', None, 'Benson et al. Fig. 5')
ax = axes[1]
f5 = np.loadtxt('fig5_digitized.txt')
ax.bar(f5[:, 0], f5[:, 1], width=1.0, color='0.82', edgecolor='0.55', label='paper (digitized)')
h, e = np.histogram(delta, bins=np.arange(0, 51))
ax.step(e[:-1], h / len(delta), where='post', color=SIM, lw=2, label=f'flyby re-simulated (mean {delta.mean():.1f} deg)')
ax.set_xlabel('angle delta between pre- and post-flyby pole [deg]'); ax.set_ylabel('fraction of runs'); ax.legend()
ax.set_title('the digitized set has no pre-flyby poles, so no delta', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig5_delta.png'), dpi=90); plt.close(fig)

# ---------------------------------------------------------------- Fig. 7: inertia sensitivity (our points on the paper's panels)
f7 = np.load('fig7_runs.npz')
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
for row, (lab, cases) in enumerate([('fig7a_inertia_Il', [(0.58, 0.965, 'orange'), (0.61, 0.965, 'green'), (0.64, 0.965, 'purple')]),
                                    ('fig7b_inertia_Ii', [(0.61, 0.95, 'orange'), (0.61, 0.965, 'green'), (0.61, 0.974, 'purple')])]):
    paper_img(axes[row, 0], lab, (400, 2200, 250, 2300), f'Benson et al. Fig. 7{"ab"[row]}')
    img = np.asarray(Image.open(paper[lab]).convert('RGB')).astype(float)
    sat = img.max(2) - img.min(2)
    faded = np.where(sat[..., None] > 90, img, 255 - 0.25 * (255 - img)).astype(np.uint8)
    for col, (a, b, cname) in enumerate(cases):
        ax = axes[row, col + 1]
        ax.imshow(faded)
        x, y = to_pixel(f7[f'{a}_{b}_lam'], f7[f'{a}_{b}_beta'], 1352.4, 1309.4, 1291.3)
        ax.plot(x, y, '.', color=SIM, ms=1.2, alpha=0.35)
        ax.set_xlim(450, 1650); ax.set_ylim(2150, 1150); ax.axis('off')
        ax.set_title(f'I_l = {a}, I_i = {b}: ours (blue) inside the {cname} outline', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig7_inertia_sensitivity.png'), dpi=80); plt.close(fig)

# ---------------------------------------------------------------- Fig. 11: Eq. 6 drift ratio (diurnal linear theory)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
paper_img(axes[0], 'fig11_drift_ratio', None, 'Benson et al. Fig. 11')
ax = axes[1]
pre_cos = np.cos(np.radians(obliquity(sim['pre_lam'], sim['pre_beta'], PRE)))
r = np.cos(np.radians(obliquity(sim['post_lam'], sim['post_beta'], POST))) / pre_cos * np.sqrt(PRE['a'] / POST['a'])
ax.hist(r, bins=np.arange(0.55, 1.15, 0.005), histtype='bar', color=SIM, weights=np.full(len(r), 1 / len(r)),
        label=f'flyby re-simulated: {r.min():.2f}-{r.max():.2f}, {100 * np.mean(r < 1):.0f}% < 1 (paper 0.57-1.12, 97%)')
ax.set_xlim(0.55, 1.15)
ax.set_title('the digitized set has no pre-flyby pole per run, so it has no ratio', fontsize=10)
ax.set_xlabel('da/dt after / before (Eq. 6: cos(g_f)/cos(g_i) * sqrt(a_i/a_f))'); ax.set_ylabel('fraction of runs'); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig11_drift_ratio.png'), dpi=90); plt.close(fig)

# ---------------------------------------------------------------- all Tempestovsky input parameters
def load_csv(path):
    with gzip.open(path, 'rt') as f:
        return np.genfromtxt(f, delimiter=',', names=True)


a_sim, a_dig = load_csv('inputs_sim.csv.gz'), load_csv('inputs_dig.csv.gz')
panels = [('I1', 'I1 = I_l / I_s', None), ('I2', 'I2 = I_i / I_s', None), ('E_ratio', 'E_ratio = I_s / I_d', np.linspace(1, 1.2, 81)),
          ('P_phi_h', 'P_phi_h [h] (short-axis precession)', np.linspace(8, 65, 115)), ('P_psi_h', 'P_psi [h] (not an input)', np.logspace(np.log10(80), np.log10(2500), 80)),
          ('psi_0', 'psi_0 [deg]', np.linspace(0, 360, 73)), ('lambda_L', 'lambda_L [deg] (post-flyby orbital frame)', np.linspace(0, 360, 73)),
          ('beta_L', 'obliquity = 90 - beta_L [deg]', np.linspace(110, 180, 71)), ('thermal_inertia', 'thermal inertia [SI]', np.linspace(100, 1200, 56)),
          ('albedo', 'Bond albedo', np.linspace(0, 0.3, 61)), ('D', 'D [m] (rescaling only)', np.linspace(320, 430, 56)),
          ('rho', 'rho [kg/m3] (rescaling only)', np.linspace(500, 5000, 91)), ('roughness', 'roughness factor (rescaling only)', np.linspace(0.95, 1.25, 61)),
          ('shape_noise', 'shape noise factor (rescaling only)', np.linspace(0.85, 1.15, 61)), ('sam', 'SAM (1) / LAM (0)', np.array([-0.5, 0.5, 1.5]))]
fig, axes = plt.subplots(3, 5, figsize=(24, 13))
for ax, (key, label, bins) in zip(axes.ravel(), panels):
    for arr, c, name in [(a_sim, SIM, 'sim'), (a_dig, DIG, 'dig')]:
        v = 90 - arr['beta_L'] if key == 'beta_L' else arr[key]
        ax.hist(v, bins=bins if bins is not None else 60, histtype='step', color=c, lw=1.6,
                weights=np.full(len(v), 1 / len(v)), label=name)
    if key == 'P_psi_h':
        ax.set_xscale('log')
    if key == 'beta_L':
        pre = obliquity(sim['pre_lam'], sim['pre_beta'], PRE)
        ax.hist(pre, bins=bins, histtype='step', color='0.5', lw=1.2, weights=np.full(len(pre), 1 / len(pre)), label='before flyby')
    ax.set_xlabel(label); ax.legend(fontsize=8)
axes[0, 0].set_ylabel('fraction of runs'); axes[1, 0].set_ylabel('fraction of runs'); axes[2, 0].set_ylabel('fraction of runs')
fig.suptitle('Tempestovsky inputs (inputs_sim.csv.gz / inputs_dig.csv.gz, 50 000 rows each; '
             f'post-flyby orbit a = {POST["a"]:.5f} au, e = {POST["e"]:.5f} in every row)', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'inputs_tempestovsky.png'), dpi=80); plt.close(fig)
print('written to', OUT)
for f in sorted(os.listdir(OUT)):
    print('  ', f)
