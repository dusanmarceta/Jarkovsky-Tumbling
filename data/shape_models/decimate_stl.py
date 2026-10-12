#!/usr/bin/env python3
"""
Smanjenje rezolucije (decimacija) STL modela asteroida.

Metod: Quadric Error Metric edge collapse (Garland & Heckbert, 1997).
Ivice se sažimaju redom po najmanjoj geometrijskoj grešci, uz proveru
da mreža ostane zatvorena (manifold) i da se trouglovi ne "prevrnu".

Potrebno: samo numpy.

Upotreba:
    python decimate_stl.py Apophis_3996.stl Apophis_low.stl --faces 1000
    python decimate_stl.py Apophis_3996.stl Apophis_low.stl --ratio 0.25
"""
import argparse
import heapq
import struct

import numpy as np


# ---------------------------------------------------------------- STL I/O
def read_stl(path):
    """Učitava binarni ili ASCII STL; vraća (vertices Nx3, faces Mx3)."""
    with open(path, "rb") as fh:
        data = fh.read()
    n = struct.unpack("<I", data[80:84])[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * n:                      # binarni STL
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        tri = np.frombuffer(data, dt, count=n, offset=84)["v"].astype(float)
    else:                                             # ASCII STL
        txt = data.decode("ascii", errors="ignore").split("\n")
        pts = [list(map(float, l.split()[1:4])) for l in txt
               if l.strip().startswith("vertex")]
        tri = np.array(pts, float).reshape(-1, 3, 3)
    v, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return v, f


def write_stl(path, v, f, header="decimated"):
    """Snima binarni STL."""
    tri = v[f].astype("<f4")
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = (nrm / np.where(ln == 0, 1, ln)).astype("<f4")
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec = np.zeros(len(f), dt)
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as fh:
        fh.write(header.encode()[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())


# ---------------------------------------------------------- QEM decimacija
def decimate(v, f, target_faces):
    v = v.astype(float).copy()
    f = f.copy()
    nv = len(v)

    # kvadrike ravni trouglova, sabrane po temenima
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    area2 = np.linalg.norm(nrm, axis=1)
    nrm = nrm / area2[:, None]
    plane = np.c_[nrm, -(nrm * p0).sum(1)]
    Kp = plane[:, :, None] * plane[:, None, :] * area2[:, None, None]
    Q = np.zeros((nv, 4, 4))
    for k in range(3):
        np.add.at(Q, f[:, k], Kp)

    alive_f = np.ones(len(f), bool)
    alive_v = np.ones(nv, bool)
    vfaces = [set() for _ in range(nv)]
    for i, tri in enumerate(f):
        for a in tri:
            vfaces[a].add(i)

    def neighbors(a):
        s = set()
        for fi in vfaces[a]:
            s.update(f[fi])
        s.discard(a)
        return s

    def best_point(a, b):
        q = Q[a] + Q[b]
        A = q.copy()
        A[3] = [0, 0, 0, 1]
        cands = []
        if abs(np.linalg.det(A)) > 1e-12:
            cands.append(np.linalg.solve(A, [0, 0, 0, 1])[:3])
        cands += [v[a], v[b], 0.5 * (v[a] + v[b])]
        best, berr = None, np.inf
        for p in cands:
            h = np.r_[p, 1.0]
            e = h @ q @ h
            if e < berr:
                best, berr = p, e
        return berr, best

    version = np.zeros(nv, int)
    heap = []

    def push(a, b):
        if a > b:
            a, b = b, a
        err, p = best_point(a, b)
        heapq.heappush(heap, (err, a, b, version[a], version[b], tuple(p)))

    edges = set()
    for tri in f:
        for i in range(3):
            a, b = sorted((tri[i], tri[(i + 1) % 3]))
            edges.add((a, b))
    for a, b in edges:
        push(a, b)

    n_faces = len(f)
    while n_faces > target_faces and heap:
        err, a, b, va, vb, p = heapq.heappop(heap)
        if not (alive_v[a] and alive_v[b]):
            continue
        if va != version[a] or vb != version[b]:
            continue
        p = np.array(p)
        shared = vfaces[a] & vfaces[b]
        if len(shared) != 2:                 # nije unutrašnja ivica
            continue
        # link uslov: zajednički susedi moraju biti tačno 2 (čuva topologiju)
        if len(neighbors(a) & neighbors(b)) != 2:
            continue
        # zabrana prevrtanja trouglova i nastanka "iglica"
        bad = False
        for x in (a, b):
            for fi in vfaces[x] - shared:
                tri = f[fi]
                old = v[tri]
                new = old.copy()
                new[tri == x] = p
                n_old = np.cross(old[1] - old[0], old[2] - old[0])
                n_new = np.cross(new[1] - new[0], new[2] - new[0])
                if np.dot(n_old, n_new) <= 0.2 * np.linalg.norm(n_old) * np.linalg.norm(n_new):
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue

        # sažimanje b -> a
        for fi in shared:
            alive_f[fi] = False
            for x in f[fi]:
                vfaces[x].discard(fi)
        n_faces -= 2
        for fi in vfaces[b]:
            f[fi][f[fi] == b] = a
            vfaces[a].add(fi)
        vfaces[b] = set()
        alive_v[b] = False
        v[a] = p
        Q[a] = Q[a] + Q[b]
        version[a] += 1
        for c in neighbors(a):
            push(a, c)

    f = f[alive_f]
    used = np.unique(f)
    remap = -np.ones(nv, int)
    remap[used] = np.arange(len(used))
    return v[used], remap[f]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--faces", type=int, help="ciljni broj trouglova")
    g.add_argument("--ratio", type=float, help="udeo zadržanih trouglova (0-1)")
    a = ap.parse_args()

    v, f = read_stl(a.input)
    target = a.faces if a.faces else int(len(f) * (a.ratio or 0.25))
    print(f"Ulaz : {len(v):6d} temena, {len(f):6d} trouglova")
    v2, f2 = decimate(v, f, target)
    write_stl(a.output, v2, f2, header=f"Decimated from {a.input}")
    print(f"Izlaz: {len(v2):6d} temena, {len(f2):6d} trouglova  -> {a.output}")


if __name__ == "__main__":
    main()
