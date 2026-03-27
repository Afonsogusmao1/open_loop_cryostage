# front_tracking.py

"""
Extract freezing front position from a temperature field.

This file samples temperature along specified lines or grids.
It locates the Tf isotherm using a sign change and linear interpolation.
It can return a centerline front z(t) and a near-wall front z(t).
It can also return a curved front z_front(r, t) if requested.

Outputs are written to CSV for later plotting.
This module is designed to be robust to MPI and mesh changes.

Notes.
Define clearly what “fully frozen” means (centerline vs wall).
Front extraction is sensitive when the entire column is above or below Tf.
"""


from __future__ import annotations

import numpy as np
from mpi4py import MPI
from dolfinx import geometry


def prepare_point_eval(domain, points_rz):
    """
    Prepare fast repeated point evaluation.

    points_rz: list[(r,z)] -> converted to (x=r, y=z, z=0) for dolfinx
    Returns:
      pts3  : (N,3) points
      cells : (N,) cell indices (or -1 if not found locally)
      mask  : (N,) True where this rank owns a valid cell for the point
      comm  : MPI communicator
    """
    comm = domain.comm

    pts_rz = np.array(points_rz, dtype=np.float64)
    pts3 = np.zeros((pts_rz.shape[0], 3), dtype=np.float64)
    pts3[:, 0] = pts_rz[:, 0]
    pts3[:, 1] = pts_rz[:, 1]
    pts3[:, 2] = 0.0

    bb = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(bb, pts3)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts3)

    cells = np.full(pts3.shape[0], -1, dtype=np.int32)
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            cells[i] = links[0]

    mask = cells >= 0
    return pts3, cells, mask, comm


def eval_prepared(Tfun, pts3, cells, mask, comm):
    """
    Evaluate fem.Function at prepared points across MPI ranks.
    Uses MIN-reduction with a sentinel so only owning rank contributes.
    """
    sentinel = 1.0e20
    vals_local = np.full(pts3.shape[0], sentinel, dtype=np.float64)

    if np.any(mask):
        out = Tfun.eval(pts3[mask], cells[mask])
        vals_local[mask] = np.asarray(out).reshape(-1)

    vals_global = np.empty_like(vals_local)
    comm.Allreduce(vals_local, vals_global, op=MPI.MIN)
    return vals_global


def front_from_samples(z: np.ndarray, Tz: np.ndarray, Tf: float) -> float:
    """
    Centerline front z_front where T crosses Tf (linear interpolation).
    Convention:
      - find first index where Tz > Tf (liquid)
      - interpolate between it and previous
      - if no Tz > Tf => fully frozen => z_front = z[-1]
    """
    z = np.asarray(z, dtype=np.float64)
    Tz = np.asarray(Tz, dtype=np.float64)

    idx = np.where(Tz > Tf)[0]
    if len(idx) == 0:
        return float(z[-1])

    j = int(idx[0])
    if j == 0:
        return float(z[0])

    z0, z1 = z[j - 1], z[j]
    T0, T1 = Tz[j - 1], Tz[j]
    if abs(T1 - T0) < 1e-12:
        return float(z1)

    zf = z0 + (Tf - T0) * (z1 - z0) / (T1 - T0)
    return float(np.clip(zf, z[0], z[-1]))
