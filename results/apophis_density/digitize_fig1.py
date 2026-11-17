'''
Digitize Fig. 1 of Farnocchia et al. (2024, Apophis T-5 Years, abstract 2046): bulk density of Apophis vs thermal
inertia needed to match the measured Yarkovsky drift of -199 m/yr.

    pdfimages -png -f 1 -l 1 "papers/THE YARKOVSKY EFFECT ON APOPHIS: BULK DENSITY CONSTRAINTS AND ANTICIPATED.pdf" /tmp/fig1
    python results/apophis_density/digitize_fig1.py /tmp/fig1-000.png results/apophis_density/fig1_digitized.npz

The figure image (1167 x 875 px) is calibrated with its grid lines. The solid blue curve (smooth surface) is tracked
by colour; the dashed blue curves (+-1 sigma) are the nearest blue pixels above and below it in each column, the red
curve (100% roughness) is the only red curve. Gaps (dashes, crossings with the black lines) are interpolated.
Output: Gamma (50-1200 step 5) and rho of the curves blue, blue_hi, blue_lo, red [kg/m3].
'''
import sys
import numpy as np
import matplotlib.image as mi

X2G = lambda x: 600 + (x - 583.5) / 0.78667       # vertical grid lines 600/800/1000/1200 at x = 583.5/741/898/1055.5
G2X = lambda g: 583.5 + (g - 600) * 0.78667
Y2R = lambda y: 2000 - (y - 358.5) / 0.27915      # horizontal grid lines 2500..500 at y = 219/358.5/498.5/638/777.5
R2Y = lambda rho: 358.5 - (rho - 2000) * 0.27915
Y0, Y1, XL, XR = 60, 790, 153, 1054               # plot area searched


def groups(col):
    '''Centres of the runs of True pixels in one image column.'''
    ys = np.flatnonzero(col)
    if len(ys) == 0:
        return []
    return [Y0 + seg.mean() for seg in np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1)]


def track(mask, x0, y0, xs, tol=8):
    '''Follow a continuous curve column by column, predicting the next point from the local slope.'''
    out, pts = {}, [(x0, y0)]
    for x in xs:
        if len(pts) >= 2:
            (xa, ya), (xb, yb) = pts[-2], pts[-1]
            pred = yb + (yb - ya) / (xb - xa) * (x - xb)
        else:
            pred = pts[-1][1]
        gs = groups(mask[Y0:Y1, x])
        if gs:
            k = int(np.argmin([abs(v - pred) for v in gs]))
            if abs(gs[k] - pred) <= tol:
                out[x] = gs[k]
                pts = (pts + [(x, gs[k])])[-6:]
                if len(pts) == 6:
                    pts = [pts[0], pts[-1]]
    return out


if __name__ == '__main__':
    im = mi.imread(sys.argv[1])[..., :3]
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    blue = (b > 0.6) & (r < 0.35) & (g < 0.35)
    red = (r > 0.6) & (g < 0.35) & (b < 0.35)

    x0 = int(round(G2X(1000)))                                    # solid blue curve at Gamma = 1000 (~2200 kg/m3)
    y0 = min(groups(blue[Y0:Y1, x0]), key=lambda v: abs(v - R2Y(2200)))
    solid = {x0: y0, **track(blue, x0, y0, range(x0 - 1, XL - 1, -1)), **track(blue, x0, y0, range(x0 + 1, XR))}
    hi, lo, rd = {}, {}, {}
    for x in range(XL, XR):
        gs = groups(blue[Y0:Y1, x])
        if x in solid:
            up = [v for v in gs if v < solid[x] - 4]
            dn = [v for v in gs if v > solid[x] + 4]
            if up:
                hi[x] = max(up)
            if dn:
                lo[x] = min(dn)
        gr = groups(red[Y0:Y1, x])
        if len(gr) == 1:
            rd[x] = gr[0]

    G = np.arange(50, 1200.1, 5.0)
    res = {'Gamma': G}
    for name, d in (('blue', solid), ('blue_hi', hi), ('blue_lo', lo), ('red', rd)):
        x = np.array(sorted(d))
        res[name] = np.interp(G, X2G(x), Y2R(np.array([d[k] for k in x])))
        print(f'{name:8s} {len(x):4d} columns, Gamma {X2G(x).min():.0f}-{X2G(x).max():.0f}')
    for gg in (50, 100, 200, 400, 600, 800, 1000, 1200):
        i = np.argmin(abs(G - gg))
        print(gg, {k: round(float(v[i])) for k, v in res.items() if k != 'Gamma'})
    np.savez(sys.argv[2], **res)
