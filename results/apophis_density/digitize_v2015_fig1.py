'''
Digitize the solid curve of Fig. 1 of Vokrouhlicky et al. (2015, Icarus 252, 277): orbit-averaged da/dt of the
tumbling Apophis from their numerical model, as a function of the surface thermal conductivity K.
Their setup: convex shape of Pravec et al. (2014) scaled to D = 375 m (volume), rho = 2000 kg/m3, C = 680 J/kg/K,
Bond albedo 0.14, emissivity 0.86, so Gamma = sqrt(K rho C).

    pdftoppm -f 4 -l 4 -r 300 -png papers/Yarko_Apophis.pdf /tmp/v15
    python results/apophis_density/digitize_v2015_fig1.py /tmp/v15-4.png results/apophis_density/v2015_fig1_digitized.npz

The plot frame is found from the longest dark rows and columns (x: log K from 0.01 to 1, y: -6 at the top to -18 at
the bottom, in 1e-4 au/Myr). The solid curve is the upper of the two curves; it is tracked from the left edge with a
continuity condition, which skips the arrow above it. Output: Gamma and dadt [1e-4 au/Myr].
'''
import sys
import numpy as np
import matplotlib.image as mi

RHO, C = 2000.0, 680.0


def groups(col):
    ys = np.flatnonzero(col)
    if len(ys) == 0:
        return []
    return [s.mean() for s in np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1)]


if __name__ == '__main__':
    im = mi.imread(sys.argv[1])[150:1100, 100:1300].mean(axis=2)      # Fig. 1 region of the 300 dpi page
    dark, ink = im < 0.5, im < 0.85
    rows, cols = dark.sum(1), dark.sum(0)
    rr, cc = np.flatnonzero(rows > 0.8 * rows.max()), np.flatnonzero(cols > 0.8 * cols.max())
    top, bot, left, right = int(rr[0]) + 2, int(rr[-1]) - 1, int(cc[0]) + 2, int(cc[-1]) - 1
    m, pts, prev = 12, {}, None
    for x in range(left + m, right - m):
        gs = [g + top + m for g in groups(ink[top + m:bot - m, x])]
        if not gs:
            continue
        if prev is None:
            y = min(gs)
        else:
            near = [g for g in gs if abs(g - prev) < 6]
            if not near:
                continue
            y = min(near, key=lambda g: abs(g - prev))
        pts[x], prev = y, y
    x = np.array(sorted(pts))
    y = np.array([pts[k] for k in x])
    K = 10 ** (-2 + 2 * (x - left) / (right - left))
    dadt = -6 - 12 * (y - top) / (bot - top)
    G = np.sqrt(K * RHO * C)
    np.savez(sys.argv[2], Gamma=G, dadt=dadt)
    print(f'frame {top} {bot} {left} {right}, {len(x)} columns, Gamma {G.min():.0f}-{G.max():.0f}')
    for g in (150, 200, 300, 400, 500, 600, 700, 800, 1000):
        print(g, round(float(np.interp(g, G, dadt)), 2))
