'''Calibrate the polar pole plots of Benson et al. (orthographic from the south ecliptic pole).'''
import numpy as np
from PIL import Image

def calibrate(path, guess_c, guess_rout, angles=(12, 80)):
    im = np.asarray(Image.open(path).convert('RGB')).astype(int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    dark = (R < 60) & (G < 60) & (B < 60)
    nonwhite = (R < 235) | (G < 235) | (B < 235)
    cx, cy = guess_c
    pts = []
    for a in np.deg2rad(np.arange(angles[0], angles[1], 2.0)):
        for r in np.arange(guess_rout - 100, guess_rout + 100, 0.5):
            x, y = cx + r * np.cos(a), cy - r * np.sin(a)
            if dark[int(round(y)), int(round(x))]:
                pts.append((x, y)); break
    pts = np.array(pts)
    A = np.c_[2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))]
    sol = np.linalg.lstsq(A, (pts ** 2).sum(1), rcond=None)[0]
    cx, cy = sol[0], sol[1]
    Rout = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    print("  outer-circle fit residual %.2f px" % np.std(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
    hist = np.zeros(int(Rout) + 50)
    for a in np.deg2rad(np.arange(angles[0] + 8, angles[1], 0.25)):
        for r in np.arange(30, Rout - 5, 1.0):
            x, y = cx + r * np.cos(a), cy - r * np.sin(a)
            if nonwhite[int(round(y)), int(round(x))]:
                hist[int(r)] += 1
    rings = []
    for beta in (-80, -70, -60, -50, -40):
        r0 = Rout * np.cos(np.radians(beta)) / np.cos(np.radians(-30))
        w = np.arange(int(r0 - 25), int(r0 + 25))
        rings.append(w[np.argmax(hist[w])])
    cosb = np.cos(np.radians([-80, -70, -60, -50, -40, -30]))
    S = np.dot(np.r_[rings, Rout], cosb) / np.dot(cosb, cosb)
    return cx, cy, S, Rout, rings, np.r_[rings, Rout] / cosb

def to_pixel(lam, beta, cx, cy, S):
    r = S * np.cos(np.radians(beta)); lam = np.radians(lam)
    return cx - r * np.cos(lam), cy - r * np.sin(lam)
