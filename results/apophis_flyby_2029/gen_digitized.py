'''
Post-flyby spin states reconstructed from the digitized figures of Benson et al. (2022), as an alternative to
the reproduced flyby ensemble. The pole comes from its own figure (its correlation with the rest is lost):
  periods (P_psi, P_phi_l): Fig. 2 heatmap, raked to its two marginal histograms
  pole (lambda, beta):      Fig. 4 scatter density (Poisson coverage inversion; the saturated core is a lower limit)
  inertias:                 uniform, I_l in [0.53, 0.73], I_i in [0.95, 0.974] (Sec. 3.1), viable pairs only
  I_d:                      from the period ratio for the drawn inertias (SAM and/or LAM branch)
  joint weighting:          150 000 candidate states (Fig. 2 periods x uniform viable inertias, both branches) get
                            weights raked so that together they reproduce the Fig. 2 period histogram, the Fig. 3
                            (omega_e, scaled I_d) density and the 17.3% LAM of the text; 50 000 are drawn by weight.
                            The inertias are uniform only a priori: given the outcome they are not (LAM is more likely
                            for small I_i, see Fig. 7), which the raking recovers approximately.
Saves digitized_50000.npz with the same post-flyby keys as ensemble_50000.npz.
'''
import numpy as np
from scipy.optimize import brentq
from multiprocessing import Pool
from flyby import periods, initial_I_d, HOUR

N = 50000
rng = np.random.default_rng(42)


def rake(counts, rows, cols, n_iter=200):
    '''Iterative proportional fitting of a 2D table to row and column sums.'''
    t = counts.copy()
    for _ in range(n_iter):
        t *= (rows / np.maximum(t.sum(1), 1e-300))[:, None]
        t *= (cols / np.maximum(t.sum(0), 1e-300))[None, :]
    return t


def sample_periods(n):
    h = np.load('fig2_digitized.npz'); m = np.load('fig2_marginals.npz')
    counts = h['counts'].copy()                     # rows: P_phi_l from 70 h down, columns: P_psi from 100 h
    Pl_c, Pp_c = h['P_phi_l_c'], h['P_psi_c']
    # white diamond (pre-flyby periods) hides the bins around (263 h, 30.56 h): fill from the surrounding bins
    di = (np.abs(Pp_c - 263) < 15)[None, :] & (np.abs(Pl_c - 30.56) < 2.0)[:, None]
    ring = (np.abs(Pp_c - 263) < 25)[None, :] & (np.abs(Pl_c - 30.56) < 3.0)[:, None] & ~di
    counts[di] = counts[ring].mean()
    # the lowest colormap level hides 0-1 runs per bin: give such bins half a run inside the support
    rows = m['f_phi_l'] / m['f_phi_l'].sum()        # the P_phi_l marginal holds all runs
    cols = m['f_psi'] / m['f_phi_l'].sum()          # same bar-reading bias as the P_phi_l marginal
    support = (rows > 0)[:, None] & (cols > 0)[None, :]
    counts = np.where((counts == 0) & support, 0.5, counts)
    cols = cols / cols.sum() * min(1.0, cols.sum())
    rows = rows / rows.sum() * cols.sum()           # P_psi > 700 h (outside the figure) is neglected
    t = rake(counts, rows, cols)
    p = (t / t.sum()).ravel()
    k = rng.choice(p.size, size=n, p=p)
    j, i = np.unravel_index(k, t.shape)
    Pp = Pp_c[i] + rng.uniform(-2.5, 2.5, n)
    Pl = Pl_c[j] + rng.uniform(-0.125, 0.125, n)
    return Pp, Pl, cols.sum()


def sample_poles(n):
    d = np.load('fig4_digitized.npz')
    c, cell, cx, cy, S = d['counts'], int(d['cell']), float(d['cx']), float(d['cy']), float(d['S'])
    p = (c / c.sum()).ravel()
    k = rng.choice(p.size, size=n, p=p)
    j, i = np.unravel_index(k, c.shape)
    x = (i + rng.uniform(0, 1, n)) * cell; y = (j + rng.uniform(0, 1, n)) * cell
    u, v = -(x - cx), -(y - cy)
    lam = np.degrees(np.arctan2(v, u)) % 360
    beta = -np.degrees(np.arccos(np.clip(np.hypot(u, v) / S, 0, 1)))
    return lam, beta


def solve_branches(I1, I2, r):
    '''I_d giving P_psi/P_phi_l = r on the SAM and on the LAM branch (None where not reachable).'''
    out = []
    for lo, hi in ((I2 + 1e-9, 1 - 1e-7), (I1 + 1e-7, I2 - 1e-9)):
        f = lambda I_d: (lambda P: P[0] / P[2])(periods(I1, I2, I_d, n_quad=1500)) - r
        try:
            flo, fhi = f(lo), f(hi)
            out.append(brentq(f, lo, hi, xtol=1e-12) if flo * fhi < 0 else None)
        except Exception:
            out.append(None)
    return out


def candidates(args):
    '''All states (both branches) for the sampled periods and one viable inertia pair that reaches the ratio.'''
    P_psi, P_phi_l, seed = args
    r = np.random.default_rng(seed)
    for _ in range(50):
        I1, I2 = r.uniform(0.53, 0.73), r.uniform(0.95, 0.974)
        if initial_I_d(I1, I2, 263.0, 27.38) is None:
            continue
        out = []
        for I_d in solve_branches(I1, I2, P_psi / P_phi_l):
            if I_d is None:
                continue
            Pp = periods(I1, I2, I_d, n_quad=1500)[0]
            omega_e = np.degrees(Pp / (P_psi * HOUR) / I_d) * HOUR          # H / I_d with H from P_psi
            Id_s = (I_d - I2) / (1 - I2) if I_d > I2 else (I_d - I2) / (I2 - I1)
            out.append((P_psi, P_phi_l, I1, I2, I_d, omega_e, Id_s))
        if out:
            return out
    return []


def fig2_target():
    h = np.load('fig2_digitized.npz'); m = np.load('fig2_marginals.npz')
    counts = h['counts'].copy()
    Pl_c, Pp_c = h['P_phi_l_c'], h['P_psi_c']
    di = (np.abs(Pp_c - 263) < 15)[None, :] & (np.abs(Pl_c - 30.56) < 2.0)[:, None]
    ring = (np.abs(Pp_c - 263) < 25)[None, :] & (np.abs(Pl_c - 30.56) < 3.0)[:, None] & ~di
    counts[di] = counts[ring].mean()
    rows = m['f_phi_l'] / m['f_phi_l'].sum(); cols = m['f_psi'] / m['f_phi_l'].sum()
    support = (rows > 0)[:, None] & (cols > 0)[None, :]
    counts = np.where((counts == 0) & support, 0.5, counts)
    cols = cols / cols.sum(); rows = rows / rows.sum()
    return rake(counts, rows, cols), Pl_c, Pp_c


if __name__ == '__main__':
    M = 150000
    Pp, Pl, f_in = sample_periods(M)
    seeds = rng.integers(0, 2**63, M)
    with Pool(6) as pool:
        res = pool.map(candidates, list(zip(Pp, Pl, seeds)), chunksize=200)
    c = np.array([x for lst in res for x in lst])
    n_per = np.repeat([len(lst) for lst in res], [len(lst) for lst in res])
    P_psi, P_phi_l, I1, I2, I_d, omega_e, Id_s = c.T
    lam_mode = I_d < I2
    print(f'{len(c)} candidates from {M} period samples ({np.mean([len(x) == 2 for x in res]):.3f} with both branches), '
          f'LAM among candidates {lam_mode.mean():.3f}')
    # targets on coarse bins
    t2, Pl_c, Pp_c = fig2_target()                       # rows: P_phi_l descending (0.25 h), cols: P_psi (5 h)
    t2 = t2[::-1].reshape(30, 8, 24, 5).sum((1, 3))      # 2 h x 25 h bins, rows ascending from 10 h
    b2 = (np.clip(((P_phi_l - 10) // 2).astype(int), 0, 29), np.clip(((P_psi - 100) // 25).astype(int), 0, 23))
    d3 = np.load('fig3_digitized.npz')
    cov = np.clip(d3['counts'] / d3['cell_area'], 0, 0.98) * d3['mask']
    t3 = (-np.log(1 - cov)).reshape(40, 2, 40, 2).sum((1, 3))       # 0.5 deg/h x 0.05 bins, rows from +1 down
    j3 = np.clip(((1 - Id_s) / 0.05).astype(int), 0, 39); i3 = np.clip(((omega_e - 4) / 0.5).astype(int), 0, 39)
    w = 1.0 / n_per                                       # each period sample counts once a priori
    for it in range(300):
        for b, t in ((b2, t2 / t2.sum()), ((j3, i3), None), (None, None)):
            if b is None:                                 # LAM fraction of the text
                f = w[lam_mode].sum() / w.sum()
                w[lam_mode] *= 0.173 / f; w[~lam_mode] *= 0.827 / (1 - f)
                continue
            cur = np.zeros(t2.shape if t is not None else t3.shape)
            np.add.at(cur, b, w)
            if t is None:                                 # Fig. 3: shape within SAM and within LAM, totals from the text
                tt = t3.copy()
                lam_rows = (1 - (np.arange(40) + 0.5) * 0.05) < 0
                for sel in (lam_rows, ~lam_rows):
                    if cur[sel].sum() > 0 and tt[sel].sum() > 0:
                        tt[sel] *= cur[sel].sum() / tt[sel].sum()
                t = tt
            else:
                t = t * w.sum()
            ratio = np.where(cur > 0, t / np.maximum(cur, 1e-300), 0.0)
            w *= ratio[b]
    ess = w.sum()**2 / (w**2).sum()
    cur2 = np.zeros(t2.shape); np.add.at(cur2, b2, w)
    cur3 = np.zeros(t3.shape); np.add.at(cur3, (j3, i3), w)
    print(f'raked: LAM {w[lam_mode].sum() / w.sum():.3f}, effective sample size {ess:.0f}, '
          f'Fig. 2 mismatch {0.5 * np.abs(cur2 / cur2.sum() - t2 / t2.sum()).sum():.3f}, '
          f'Fig. 3 total-variation mismatch {0.5 * np.abs(cur3 / cur3.sum() - t3 / t3.sum()).sum():.3f}')
    k = rng.choice(len(w), size=N, replace=True, p=w / w.sum())
    lam, beta = sample_poles(N)
    P = np.array([periods(a, b_, cc) for a, b_, cc in zip(I1[k], I2[k], I_d[k])])
    H = P[:, 0] / (P_psi[k] * HOUR)
    out = dict(I1=I1[k], I2=I2[k], post_I_d=I_d[k], post_H=H, post_omega_e=H / I_d[k],
               post_P_psi_h=P_psi[k], post_P_phi_l_h=P_phi_l[k], post_P_phi_s_h=P[:, 1] / H / HOUR,
               post_sam=I_d[k] > I2[k], post_lam=lam, post_beta=beta, post_Id_scaled=Id_s[k])
    np.savez('digitized_50000.npz', **out)
    print(f'saved {N} states ({len(np.unique(k))} distinct candidates)')
