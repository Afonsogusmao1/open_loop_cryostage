from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import ufl
from dolfinx import mesh, fem
from petsc4py import PETSc


@dataclass(frozen=True)
class GeometryParams:
    R_in: float = 7.5e-3       # cavity radius (m)
    t_wall: float = 2.0e-3     # mold wall thickness (m)
    t_base: float = 0.0        # mold base thickness inside the cavity radius (m)
    H_fill: float = 15.0e-3    # water fill height (m)
    H_total: float = 17.0e-3   # total mold height (m)

    @property
    def R_out(self) -> float:
        return self.R_in + self.t_wall

    def probe_radius(self, wall_inset_m: float = 0.5e-3) -> float:
        """
        Thermocouple radius for a probe located slightly inside the cavity,
        near the wall.
        """
        return max(1e-10, self.R_in - wall_inset_m)


def create_axisym_domain(comm, geom: GeometryParams, Nr: int, Nz: int) -> mesh.Mesh:
    """
    Axisymmetric rectangle domain:
      r in [0, R_out], z in [0, H_total]
    """
    return mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([geom.R_out, geom.H_total])],
        [Nr, Nz],
        cell_type=mesh.CellType.triangle,
    )


def tag_boundaries(domain: mesh.Mesh, geom: GeometryParams):
    """
    Boundary facet tags:
      1 = bottom (z=0)
      2 = top    (z=H_total)
      3 = outer  (r=R_out)
    Returns: (facet_tags, ds, bottom_facets)
    """
    fdim = domain.topology.dim - 1
    tol = 1e-12

    bottom = mesh.locate_entities_boundary(domain, fdim, lambda x: x[1] <= tol)
    top = mesh.locate_entities_boundary(domain, fdim, lambda x: x[1] >= geom.H_total - tol)
    outer = mesh.locate_entities_boundary(domain, fdim, lambda x: x[0] >= geom.R_out - tol)

    facets = np.concatenate([bottom, top, outer])
    markers = np.concatenate([
        np.full(bottom.shape, 1, dtype=np.int32),
        np.full(top.shape, 2, dtype=np.int32),
        np.full(outer.shape, 3, dtype=np.int32),
    ])

    order = np.argsort(facets)
    facet_tags = mesh.meshtags(domain, fdim, facets[order], markers[order])
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    return facet_tags, ds, bottom


def bottom_dirichlet_bc(V, bottom_facets, T_plate_C: float):
    """
    Dirichlet BC: T = T_plate_C at bottom boundary.
    Returns (bc_object, T_plate_constant).
    """
    domain = V.mesh
    fdim = domain.topology.dim - 1
    T_plate = fem.Constant(domain, PETSc.ScalarType(T_plate_C))
    dofs_bottom = np.array(fem.locate_dofs_topological(V, fdim, bottom_facets), dtype=np.int32)
    bc_bottom = fem.dirichletbc(T_plate, dofs_bottom, V)
    return bc_bottom, T_plate


def dof_region_masks(V, geom: GeometryParams, eps: float = 1e-12):
    """
    Dof-wise masks for the calibration geometry.

    Regions:
      cavity_all := r <= R_in and z >= t_base
      fillable   := r <  R_in and t_base < z <= H_fill
      headspace  := r <  R_in and H_fill < z <= H_total
      mold       := complement of cavity_all

    Notes.
    fillable intentionally excludes the shared wall/interface DOFs.
    This keeps the mold wall DOFs untouched when the cavity is filled.
    """
    X = V.tabulate_dof_coordinates()
    r = X[:, 0]
    z = X[:, 1]

    cavity_all = (r <= (geom.R_in + eps)) & (z >= (geom.t_base - eps))
    fillable = (r < (geom.R_in - eps)) & (z > (geom.t_base + eps)) & (z <= (geom.H_fill + eps))
    headspace = (r < (geom.R_in - eps)) & (z > (geom.H_fill + eps)) & (z <= (geom.H_total + eps))
    mold = ~cavity_all

    return r, z, cavity_all, fillable, headspace, mold


def overwrite_fillable_temperature(V, Tfun, geom: GeometryParams, value_C: float, eps: float = 1e-12):
    """
    Reset only the lower water-filled part of the cavity, keeping the mold,
    the wall interface DOFs, and the headspace air unchanged.
    """
    X = V.tabulate_dof_coordinates()
    r = X[:, 0]
    z = X[:, 1]
    mask = (r < (geom.R_in - eps)) & (z > (geom.t_base + eps)) & (z <= (geom.H_fill + eps))
    Tfun.x.array[mask] = value_C
    Tfun.x.scatter_forward()