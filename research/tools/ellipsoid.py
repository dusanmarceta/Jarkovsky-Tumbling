"""
Convex triaxial ellipsoid meshes and their uniform-density mass properties.

The mesh is an icosphere (20 * 4**n facets) scaled to semi-axes a >= b >= c along x, y, z, so that the body
x axis is the long axis (smallest moment, I1), y the intermediate (I2) and z the short axis (largest, I3).
This matches the principal-axis convention assumed by the solver (src/utilities/tumbling_final_2.py, comment
at l. 202-205). The scale is chosen so that the *faceted* body has exactly the requested volume-equivalent
diameter, and the inertia ratios are computed from the faceted body (not the analytic ellipsoid), so the
rotational dynamics and the thermal surface describe the same object.
"""
import hashlib
from pathlib import Path

import numpy as np


def icosphere(n_subdiv):
    t = (1.0 + 5 ** 0.5) / 2.0
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]]
    verts = [p / np.linalg.norm(p) for p in v]
    faces = [tuple(x) for x in f]
    for _ in range(n_subdiv):
        cache, new = {}, []

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = verts[i] + verts[j]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    return np.array(verts), np.array(faces)


def mass_properties(tri):
    """Volume, centre of mass and inertia tensor (unit density) of a closed triangulated surface."""
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    dV = np.einsum('ij,ij->i', v0, np.cross(v1, v2)) / 6.0
    V = dV.sum()
    com = (dV[:, None] * (v0 + v1 + v2) / 4.0).sum(0) / V
    S = v0 + v1 + v2
    C = (dV[:, None, None] / 20.0 * (S[:, :, None] * S[:, None, :]
         + v0[:, :, None] * v0[:, None, :] + v1[:, :, None] * v1[:, None, :] + v2[:, :, None] * v2[:, None, :])).sum(0)
    C = C - V * np.outer(com, com)
    I = np.trace(C) * np.eye(3) - C
    return V, com, I


def ellipsoid_mesh(b_over_a, c_over_a, D_eq, n_subdiv=2):
    """Triangles (n, 3, 3) of an ellipsoid with axis ratios b/a, c/a (1 >= b/a >= c/a > 0) and equal-volume
    diameter D_eq [m] of the faceted body. Facet vertex order gives outward normals."""
    if not (1.0 >= b_over_a >= c_over_a > 0.0):
        raise ValueError(f'need 1 >= b/a >= c/a > 0, got b/a = {b_over_a}, c/a = {c_over_a}')
    verts, faces = icosphere(n_subdiv)
    tri = verts[faces] * np.array([1.0, b_over_a, c_over_a])
    # outward orientation: normal must point away from the centre
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum('ij,ij->i', n, tri.mean(1)) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    V, _, _ = mass_properties(tri)
    tri *= ((np.pi / 6.0 * D_eq ** 3) / V) ** (1.0 / 3.0)
    return tri


def inertia_summary(tri):
    V, com, I = mass_properties(tri)
    w, R = np.linalg.eigh(I)
    off = np.abs(I - np.diag(np.diag(I))).max() / np.abs(np.diag(I)).max()
    Ix, Iy, Iz = np.diag(I)
    return {'volume_m3': V, 'D_eq_m': (6 * V / np.pi) ** (1 / 3), 'com_m': com.tolist(),
            'I1': Ix / Iz, 'I2': Iy / Iz, 'principal_ratios': (w / w.max()).tolist(), 'max_offdiag_rel': off,
            'axis_alignment_deg': np.degrees(np.arccos(np.clip(np.abs(np.diag(R)), 0, 1))).tolist(),
            'area_m2': 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()}


def analytic_inertia_ratios(b_over_a, c_over_a):
    """Uniform-density ellipsoid with semi-axes (1, b, c): I1 = (b^2+c^2)/(1+b^2), I2 = (1+c^2)/(1+b^2)."""
    b2, c2 = b_over_a ** 2, c_over_a ** 2
    return (b2 + c2) / (1 + b2), (1 + c2) / (1 + b2)


def write_ascii_stl(path, tri, name='ellipsoid'):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    lines = [f'solid {name}']
    for k in range(len(tri)):
        lines.append(f'facet normal {n[k, 0]:.10e} {n[k, 1]:.10e} {n[k, 2]:.10e}')
        lines.append('  outer loop')
        for p in tri[k]:
            lines.append(f'    vertex {p[0]:.10e} {p[1]:.10e} {p[2]:.10e}')
        lines.append('  endloop')
        lines.append('endfacet')
    lines.append(f'endsolid {name}')
    Path(path).write_text('\n'.join(lines) + '\n')


def mesh_file(b_over_a, c_over_a, D_eq, n_subdiv, directory):
    """Write (or reuse) the STL for these parameters; returns (path, inertia summary)."""
    key = f'{b_over_a:.6f}_{c_over_a:.6f}_{D_eq:.3f}_{n_subdiv}'
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    path = Path(directory) / f'ell_{h}_n{20 * 4 ** n_subdiv}.stl'
    tri = ellipsoid_mesh(b_over_a, c_over_a, D_eq, n_subdiv)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_ascii_stl(path, tri, name=f'ellipsoid_{key}')
    return path, inertia_summary(tri)
