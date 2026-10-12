#!/usr/bin/env python3
"""
3D vizuelizacija dva STL modela asteroida (npr. originalni i decimirani)
jedan pored drugog, sa sinhronizovanom rotacijom.

Potrebno: numpy, matplotlib.

Upotreba:
    python visualize_apophis.py
    python visualize_apophis.py Apophis_3996.stl Apophis_1000.stl
    python visualize_apophis.py --no-edges --save apophis.png

U prozoru:  miš - rotacija (oba modela se okreću zajedno)
            e   - uključi/isključi ivice trouglova
"""
import argparse
import struct

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (potrebno za matplotlib < 3.2)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_stl(path):
    """Učitava binarni ili ASCII STL; vraća niz trouglova oblika (M, 3, 3)."""
    with open(path, "rb") as fh:
        data = fh.read()
    n = struct.unpack("<I", data[80:84])[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * n:
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        return np.frombuffer(data, dt, count=n, offset=84)["v"].astype(float)
    txt = data.decode("ascii", errors="ignore").split("\n")
    pts = [list(map(float, l.split()[1:4])) for l in txt
           if l.strip().startswith("vertex")]
    return np.array(pts, float).reshape(-1, 3, 3)


def shade(tri, light=(0.8, -0.9, 0.9), base=(0.62, 0.58, 0.52)):
    """Lambertovo osenčenje: boja svakog trougla prema uglu ka izvoru svetlosti."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    k = 0.25 + 0.75 * np.clip(n @ L, 0, 1)
    return np.clip(np.outer(k, base), 0, 1)


def stats(tri):
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                         tri[:, 2] - tri[:, 0]), axis=1).sum()
    nvert = len(np.unique(tri.reshape(-1, 3), axis=0))
    return nvert, vol, area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=["Apophis_3996.stl", "Apophis_1000.stl"])
    ap.add_argument("--no-edges", action="store_true", help="bez ivica trouglova")
    ap.add_argument("--save", help="sačuvaj sliku u fajl umesto prikaza")
    a = ap.parse_args()

    meshes = [read_stl(p) for p in a.files]

    # zajednički centar i opseg da bi modeli bili u istoj razmeri
    allp = np.concatenate([m.reshape(-1, 3) for m in meshes])
    c = 0.5 * (allp.min(0) + allp.max(0))
    r = 0.5 * np.ptp(allp, axis=0).max()

    fig = plt.figure(figsize=(6 * len(meshes), 6.4))
    fig.suptitle("Asteroid (99942) Apophis — poređenje rezolucija", fontsize=14, y=0.98)
    axes, polys = [], []
    for i, (path, tri) in enumerate(zip(a.files, meshes)):
        ax = fig.add_subplot(1, len(meshes), i + 1, projection="3d")
        pc = Poly3DCollection(tri - c, facecolors=shade(tri),
                              edgecolors="k" if not a.no_edges else "none",
                              linewidths=0.15)
        ax.add_collection3d(pc)
        for setlim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setlim(-r, r)
        if hasattr(ax, "set_box_aspect"):      # matplotlib >= 3.3
            ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        nv, vol, area = stats(tri)
        ax.set_title(f"{path}\n{len(tri)} trouglova, {nv} temena\n"
                     f"V = {vol/1e6:.3f}·10⁶ m³,  S = {area/1e3:.1f}·10³ m²",
                     fontsize=10)
        ax.view_init(elev=20, azim=-60)
        axes.append(ax)
        polys.append(pc)

    # sinhronizacija rotacije svih panela
    def on_move(event):
        src = event.inaxes
        if src not in axes:
            return
        for ax in axes:
            if ax is not src:
                ax.view_init(elev=src.elev, azim=src.azim)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "e":
            for pc in polys:
                ec = pc.get_edgecolor()
                pc.set_edgecolor("none" if len(ec) and ec[0][3] > 0 else "k")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.03, top=0.83, wspace=0.1)

    if a.save:
        fig.savefig(a.save, dpi=150)
        print("Sačuvano:", a.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
