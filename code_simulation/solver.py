# solver.py

"""
Solve the transient heat transfer problem for the axisymmetric mold + cavity.

This file contains the FEniCSx implementation of the PDE solve.
The domain is an r–z cross-section with an axisymmetric assumption.
The mold and cavity are treated as separate regions with different properties.

A “fill event” can be modeled.
Before fill, the cavity can be treated as air or a pre-cooled state.
At fill time, the cavity temperature is overwritten to the fill temperature.
After fill, the cavity evolves with water/ice properties and latent heat handling.

Boundary conditions typically include:
A fixed plate temperature at the bottom.
Convective losses at the top and/or outer surfaces.

Outputs are written as fields (XDMF/H5) and time series logs (CSV).
Front tracking and probe extraction are triggered from here or via helper modules.

Notes.
Latent heat is handled via an apparent heat capacity over a small mushy band.
Time stepping and property updates are designed to be stable and predictable.
"""


from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem

from geometry import (
    GeometryParams,
    create_axisym_domain,
    tag_boundaries,
    bottom_dirichlet_bc,
    dof_region_masks,
    overwrite_fillable_temperature,
)
from front_tracking import prepare_point_eval, eval_prepared, front_from_temperature_threshold
from materials import (
    SolidProps,
    PLA_DEFAULT,
    AIR_DEFAULT,
    WATER_CONST_DEFAULT,
    water_ice_k_cp_from_tables,
    apparent_cp_bump,
)


@dataclass(frozen=True)
class ThermalBCs:
    T_room_C: float = 21.0
    h_top: float = 2.0   # W/m^2/K
    h_side: float = 2.0  # W/m^2/K


@dataclass(frozen=True)
class PhaseChangeParams:
    Tf: float = 0.0
    L_latent: float = 334000.0
    dT_mushy: float = 0.5


@dataclass(frozen=True)
class PrefillOptions:
    mode: str = "steady"  # "steady", "transient", "fixed_time", "probe_stabilized"
    probe_window_s: float = 60.0
    probe_tol_C: float = 0.05
    min_prefill_s: float = 0.0
    max_prefill_s: float = 3600.0


@dataclass(frozen=True)
class FreezeStopOptions:
    mode: str = "fillable_region"
    extra_subcooling_C: float = 0.0


SUPPORTED_FRONT_DEFINITION_MODES = ("isotherm_Tf", "solidus", "freeze_threshold")
_FRONT_DEFINITION_MODE_LOOKUP = {mode.lower(): mode for mode in SUPPORTED_FRONT_DEFINITION_MODES}


def normalize_front_definition_mode(front_definition_mode: str) -> str:
    mode_key = str(front_definition_mode).strip().lower()
    try:
        return _FRONT_DEFINITION_MODE_LOOKUP[mode_key]
    except KeyError as exc:
        raise ValueError(
            "front_definition_mode must be one of "
            f"{SUPPORTED_FRONT_DEFINITION_MODES!r}, got {front_definition_mode!r}"
        ) from exc


def front_threshold_from_mode(
    front_definition_mode: str,
    *,
    phase: PhaseChangeParams,
    freeze_stop: FreezeStopOptions,
) -> float:
    """Resolve the extraction threshold for a named front definition."""
    mode = normalize_front_definition_mode(front_definition_mode)
    if mode == "isotherm_Tf":
        threshold_C = float(phase.Tf)
    elif mode == "solidus":
        threshold_C = float(phase.Tf) - 0.5 * float(phase.dT_mushy)
    elif mode == "freeze_threshold":
        threshold_C = float(phase.Tf) - 0.5 * float(phase.dT_mushy) - float(freeze_stop.extra_subcooling_C)
    else:
        raise AssertionError(f"Unhandled front_definition_mode={mode!r}")

    if not np.isfinite(threshold_C):
        raise ValueError(f"front threshold for mode {mode!r} is not finite: {threshold_C!r}")
    return float(threshold_C)


def _update_coefficients(
    *,
    k_eff_fun: fem.Function,
    rho_cp_eff_fun: fem.Function,
    T_prev: fem.Function,
    fillable_dof: np.ndarray,
    headspace_dof: np.ndarray,
    fill_flag: float,
    pla: SolidProps,
    air: SolidProps,
    water_const: SolidProps,
    phase: PhaseChangeParams,
    use_tabulated_water_ice: bool,
):
    """
    Update k_eff and rho*cp_eff at DOFs using T_prev (Picard-style).

    Regions:
      - mold      -> PLA always
      - headspace -> air always
      - fillable  -> air pre-fill, water/ice post-fill
    """
    Tloc = np.asarray(T_prev.x.array, dtype=np.float64)

    k_arr = np.full_like(Tloc, pla.k, dtype=np.float64)
    rho_cp_arr = np.full_like(Tloc, pla.rho * pla.cp, dtype=np.float64)

    # Headspace is always air.
    k_arr[headspace_dof] = air.k
    rho_cp_arr[headspace_dof] = air.rho * air.cp

    if fill_flag < 0.5:
        k_arr[fillable_dof] = air.k
        rho_cp_arr[fillable_dof] = air.rho * air.cp
    else:
        T_cav = Tloc[fillable_dof]

        if use_tabulated_water_ice:
            k_cav, cp_cav = water_ice_k_cp_from_tables(T_cav, phase.Tf)
        else:
            k_cav = np.full(T_cav.shape, water_const.k, dtype=np.float64)
            cp_cav = np.full(T_cav.shape, water_const.cp, dtype=np.float64)

        cp_cav = cp_cav + apparent_cp_bump(T_cav, phase.Tf, phase.L_latent, phase.dT_mushy)

        k_arr[fillable_dof] = k_cav
        rho_cp_arr[fillable_dof] = water_const.rho * cp_cav

    k_eff_fun.x.array[:] = k_arr
    rho_cp_eff_fun.x.array[:] = rho_cp_arr
    k_eff_fun.x.scatter_forward()
    rho_cp_eff_fun.x.scatter_forward()


def _solve_prefill_steady(
    *,
    V,
    T: fem.Function,
    geom: GeometryParams,
    ds,
    bc_bottom,
    k_eff_prefill: fem.Function,
    bcs: ThermalBCs,
):
    """
    Solve the empty-mold pre-fill steady state.
    The fillable cavity and headspace are both treated as air.
    """
    domain = V.mesh
    x = ufl.SpatialCoordinate(domain)
    r = x[0]
    v = ufl.TestFunction(V)
    Ttrial = ufl.TrialFunction(V)

    h_top_c = fem.Constant(domain, PETSc.ScalarType(bcs.h_top))
    h_side_c = fem.Constant(domain, PETSc.ScalarType(bcs.h_side))
    Troom_c = fem.Constant(domain, PETSc.ScalarType(bcs.T_room_C))

    a_ss = (
        k_eff_prefill * ufl.dot(ufl.grad(Ttrial), ufl.grad(v)) * r * ufl.dx
        + h_top_c * Ttrial * v * r * ds(2)
        + h_side_c * Ttrial * v * r * ds(3)
    )
    L_ss = (
        h_top_c * Troom_c * v * r * ds(2)
        + h_side_c * Troom_c * v * r * ds(3)
    )

    problem_ss = LinearProblem(
        a_ss,
        L_ss,
        bcs=[bc_bottom],
        u=T,
        petsc_options_prefix="heat_ss_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-14,
            "ksp_max_it": 3000,
        },
    )
    problem_ss.solve()
    T.x.scatter_forward()


def _probe_stabilized(times: deque, values: deque, *, window_s: float, tol_C: float) -> bool:
    """
    Probe stabilization criterion based only on the three probe temperatures.

    Over the trailing time window [t-window_s, t], each probe must vary by at most tol_C:
      max(T_probe) - min(T_probe) <= tol_C
    """
    if len(times) < 2:
        return False

    span = times[-1] - times[0]
    if span < window_s - 1e-12:
        return False

    arr = np.asarray(values, dtype=np.float64)
    probe_span = arr.max(axis=0) - arr.min(axis=0)
    return bool(np.all(probe_span <= tol_C))


def _global_fillable_max_temperature(comm, Tfun: fem.Function, fillable_dof: np.ndarray) -> float:
    local_vals = np.asarray(Tfun.x.array, dtype=np.float64)
    if np.any(fillable_dof):
        local_max = float(np.max(local_vals[fillable_dof]))
    else:
        local_max = -1.0e30
    return float(comm.allreduce(local_max, op=MPI.MAX))


def _format_probe_label_mm(z_m: float) -> str:
    """Return a filename/CSV-safe probe label in mm, e.g. 11.5 mm -> 11p5mm."""
    return f"{1000.0 * z_m:.1f}".replace(".", "p") + "mm"


def _format_probe_header(probe_z_m: tuple[float, ...]) -> str:
    cols = [f"T_z{_format_probe_label_mm(zm)}_C" for zm in probe_z_m]
    return "time_s,time_since_fill_s," + ",".join(cols) + "\n"


def run_case(
    *,
    out_dir: str | Path,
    prefix: str,

    # Geometry + discretization
    geom: GeometryParams = GeometryParams(),
    Nr: int = 180,
    Nz: int = 360,

    # Time
    dt: float = 0.25,
    pre_cool_s: float = 0.0,
    t_after_fill_s: float = 3600.0,
    write_every: float = 60.0,
    write_field_output: bool = True,
    write_probe_csv: bool = True,
    show_progress: bool = True,

    # Fill
    T_fill_C: float = 21.0,

    # BCs
    T_plate_C: float = -20.0,
    T_plate_profile_C: Callable[[float], float] | None = None,
    bcs: ThermalBCs = ThermalBCs(),

    # Materials
    pla: SolidProps = PLA_DEFAULT,
    air: SolidProps = AIR_DEFAULT,
    water_const: SolidProps = WATER_CONST_DEFAULT,

    # Phase change
    phase: PhaseChangeParams = PhaseChangeParams(),

    # Prefill
    prefill: PrefillOptions = PrefillOptions(),

    # Post-fill stop
    freeze_stop: FreezeStopOptions = FreezeStopOptions(),

    # Front observable definition
    front_definition_mode: str = "isotherm_Tf",

    # Probes / centerline front
    probe_z_m: tuple[float, float, float] = (3e-3, 7e-3, 12e-3),
    probe_wall_inset_m: float = 0.5e-3,
    Nz_front: int = 800,

    # Wall-line front tracking (slightly inside the cavity)
    wall_r_factor: float = 0.999,
    stop_when_wall_frozen: bool = False,

    # Curved front tracking
    enable_front_curve: bool = True,
    Nr_front_curve: int = 25,
    Nz_front_curve: int = 400,
    r_max_factor: float = 0.999,
    front_curve_every_s: float = 60.0,

    # Properties
    use_tabulated_water_ice: bool = False,

    # Debug / diagnostics
    debug_material_probe_index: int | None = None,
    debug_material_every_s: float = 5.0,
):
    """
    Axisymmetric 2D freezing model for the calibration geometry.

    Supported pre-fill modes:
      - "steady": solve empty-mold steady state, then optionally hold for pre_cool_s
      - "transient" / "fixed_time": cool the empty mold transiently for pre_cool_s
      - "probe_stabilized": cool the empty mold transiently until the three near-wall probes stabilize

    In all cases, filling overwrites only the lower H_fill region of the cavity with water at T_fill_C.
    The top headspace remains air throughout the run.

    Default post-fill stop:
    stop when max(T in fillable region) <= Tf - dT_mushy/2 - extra_subcooling_C.

    Lean runs may disable field output, probe CSV output, or progress printing without
    changing the physics, front CSV, or freeze-complete event logic.
    """
    comm = MPI.COMM_WORLD
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def plate_temperature_C(current_t: float) -> float:
        if T_plate_profile_C is None:
            return float(T_plate_C)
        return float(T_plate_profile_C(float(current_t)))

    xdmf_path = out_dir / f"{prefix}.xdmf"
    probes_path = out_dir / f"{prefix}_probes.csv"
    front_path = out_dir / f"{prefix}_front.csv"
    front_curve_path = out_dir / f"{prefix}_front_curve.csv"

    domain = create_axisym_domain(comm, geom, Nr, Nz)
    V = fem.functionspace(domain, ("CG", 1))

    T = fem.Function(V, name="T")
    T_prev = fem.Function(V, name="T_prev")
    T.x.array[:] = bcs.T_room_C
    T_prev.x.array[:] = bcs.T_room_C
    T.x.scatter_forward()
    T_prev.x.scatter_forward()

    facet_tags, ds, bottom_facets = tag_boundaries(domain, geom)
    bc_bottom, T_plate_const = bottom_dirichlet_bc(V, bottom_facets, plate_temperature_C(0.0))

    r_dof, z_dof, cavity_all_dof, fillable_dof, headspace_dof, mold_dof = dof_region_masks(V, geom)

    k_eff = fem.Function(V, name="k_eff")
    rho_cp_eff = fem.Function(V, name="rho_cp_eff")
    fill = fem.Constant(domain, PETSc.ScalarType(0.0))

    _update_coefficients(
        k_eff_fun=k_eff,
        rho_cp_eff_fun=rho_cp_eff,
        T_prev=T_prev,
        fillable_dof=fillable_dof,
        headspace_dof=headspace_dof,
        fill_flag=float(fill.value),
        pla=pla,
        air=air,
        water_const=water_const,
        phase=phase,
        use_tabulated_water_ice=use_tabulated_water_ice,
    )

    if prefill.mode.lower() == "steady":
        _solve_prefill_steady(
            V=V,
            T=T,
            geom=geom,
            ds=ds,
            bc_bottom=bc_bottom,
            k_eff_prefill=k_eff,
            bcs=bcs,
        )
        T_prev.x.array[:] = T.x.array
        T_prev.x.scatter_forward()

    x = ufl.SpatialCoordinate(domain)
    r = x[0]
    v = ufl.TestFunction(V)
    Ttrial = ufl.TrialFunction(V)

    h_top_c = fem.Constant(domain, PETSc.ScalarType(bcs.h_top))
    h_side_c = fem.Constant(domain, PETSc.ScalarType(bcs.h_side))
    Troom_c = fem.Constant(domain, PETSc.ScalarType(bcs.T_room_C))

    a = (
        (rho_cp_eff / dt) * Ttrial * v * r * ufl.dx
        + k_eff * ufl.dot(ufl.grad(Ttrial), ufl.grad(v)) * r * ufl.dx
        + h_top_c * Ttrial * v * r * ds(2)
        + h_side_c * Ttrial * v * r * ds(3)
    )

    L = (
        (rho_cp_eff / dt) * T_prev * v * r * ufl.dx
        + h_top_c * Troom_c * v * r * ds(2)
        + h_side_c * Troom_c * v * r * ds(3)
    )

    problem = LinearProblem(
        a,
        L,
        bcs=[bc_bottom],
        u=T,
        petsc_options_prefix="heat_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
            "ksp_max_it": 2000,
        },
    )

    probe_r = geom.probe_radius(probe_wall_inset_m)
    probe_points = [(probe_r, float(zm)) for zm in probe_z_m]
    probe_pts3, probe_cells, probe_mask, comm_pts = prepare_point_eval(domain, probe_points)

    eps_r = 1e-10
    z_samp = np.linspace(max(geom.t_base, 0.0), geom.H_fill, Nz_front)
    front_points = [(eps_r, float(zi)) for zi in z_samp]
    front_pts3, front_cells, front_mask, _ = prepare_point_eval(domain, front_points)

    r_wall = float(wall_r_factor) * float(geom.R_in)
    wall_points = [(r_wall, float(zi)) for zi in z_samp]
    wall_pts3, wall_cells, wall_mask, _ = prepare_point_eval(domain, wall_points)

    if enable_front_curve:
        r_max = float(r_max_factor) * geom.R_in
        r_samp = np.linspace(eps_r, r_max, int(Nr_front_curve))
        z_samp_curve = np.linspace(max(geom.t_base, 0.0), geom.H_fill, int(Nz_front_curve))
        curve_points = [(float(rj), float(zi)) for rj in r_samp for zi in z_samp_curve]
        curve_pts3, curve_cells, curve_mask, _ = prepare_point_eval(domain, curve_points)
    else:
        r_samp = None
        z_samp_curve = None
        curve_pts3 = curve_cells = curve_mask = None

    prefill_mode = prefill.mode.lower()
    if prefill_mode == "steady":
        t_prefill_budget = max(0.0, pre_cool_s)
    elif prefill_mode in {"transient", "fixed_time"}:
        t_prefill_budget = max(0.0, pre_cool_s)
    elif prefill_mode == "probe_stabilized":
        t_prefill_budget = max(0.0, prefill.max_prefill_s)
    else:
        raise ValueError(f"Unknown prefill.mode='{prefill.mode}'")

    t_end_total = t_prefill_budget + t_after_fill_s
    n_steps = int(np.ceil(t_end_total / dt)) if t_end_total > 0.0 else 0
    next_write = 0.0

    freeze_threshold_C = phase.Tf - 0.5 * phase.dT_mushy - freeze_stop.extra_subcooling_C
    front_definition_mode = normalize_front_definition_mode(front_definition_mode)
    front_threshold_C = front_threshold_from_mode(
        front_definition_mode,
        phase=phase,
        freeze_stop=freeze_stop,
    )

    if show_progress and comm.rank == 0:
        print(
            "Front definition "
            f"mode={front_definition_mode}, threshold={front_threshold_C:.6f} C"
        )

    probe_times = deque()
    probe_values = deque()
    fill_time: float | None = None

    xdmf_context = io.XDMFFile(comm, str(xdmf_path), "w") if write_field_output else nullcontext(None)

    with xdmf_context as xdmf:
        if xdmf is not None:
            xdmf.write_mesh(domain)

        f_probe = None
        f_front = None
        f_curve = None

        if comm.rank == 0:
            if write_probe_csv:
                f_probe = open(probes_path, "w", buffering=1)
                f_probe.write(_format_probe_header(probe_z_m))
            f_front = open(front_path, "w", buffering=1)
            f_front.write(
                "time_s,time_since_fill_s,fill_flag,"
                "front_definition_mode,front_threshold_C,"
                "z_front_m,z_front_mm,z_front_rel_mm,v_front_mm_per_s,"
                "z_front_wall_m,z_front_wall_mm,v_front_wall_mm_per_s,"
                "Tmax_fillable_C,freeze_complete_flag\n"
            )

            if enable_front_curve:
                f_curve = open(front_curve_path, "w", buffering=1)
                f_curve.write("# radii_mm: " + ",".join(f"{rr*1000.0:.6f}" for rr in r_samp) + "\n")
                cols = [
                    "time_s",
                    "time_since_fill_s",
                    "fill_flag",
                    "front_definition_mode",
                    "front_threshold_C",
                ] + [f"z_front_r{i:03d}_mm" for i in range(len(r_samp))]
                f_curve.write(",".join(cols) + "\n")

        def do_fill(current_t: float):
            nonlocal fill_time
            fill.value = PETSc.ScalarType(1.0)
            fill_time = float(current_t)
            overwrite_fillable_temperature(V, T, geom=geom, value_C=T_fill_C)
            T_prev.x.array[:] = T.x.array
            T_prev.x.scatter_forward()
            if show_progress and comm.rank == 0:
                print(f"\nFILL at t={current_t:.3f}s -> lower cavity WATER, reset to {T_fill_C} °C")

        if prefill_mode == "steady" and pre_cool_s <= 1e-12:
            do_fill(0.0)

        t = 0.0
        if xdmf is not None:
            xdmf.write_function(T, t)

        temps = eval_prepared(T, probe_pts3, probe_cells, probe_mask, comm_pts)
        if float(fill.value) < 0.5:
            probe_times.append(t)
            probe_values.append(np.array(temps, dtype=np.float64))

        t_since_fill = (t - fill_time) if (float(fill.value) > 0.5 and fill_time is not None) else np.nan

        if comm.rank == 0:
            if f_probe is not None:
                f_probe.write(f"{t:.6f},{t_since_fill:.6f},{temps[0]:.6f},{temps[1]:.6f},{temps[2]:.6f}\n")
            f_front.write(
                f"{t:.6f},{t_since_fill:.6f},{float(fill.value):.1f},"
                f"{front_definition_mode},{front_threshold_C:.9e},"
                "nan,nan,nan,nan,nan,nan,nan,nan,0\n"
            )
            if enable_front_curve:
                nan_line = ",".join(["nan"] * len(r_samp))
                f_curve.write(
                    f"{t:.6f},{t_since_fill:.6f},{float(fill.value):.1f},"
                    f"{front_definition_mode},{front_threshold_C:.9e},{nan_line}\n"
                )

        zf_prev = np.nan
        zf_wall_prev = np.nan
        next_curve_since_fill = 0.0
        next_material_debug_since_fill = 0.0
        t0_wall = time.perf_counter()

        for step in range(1, n_steps + 1):
            t = step * dt

            if float(fill.value) < 0.5:
                if prefill_mode in {"transient", "fixed_time"} and (t >= pre_cool_s - 1e-12):
                    do_fill(t)

            _update_coefficients(
                k_eff_fun=k_eff,
                rho_cp_eff_fun=rho_cp_eff,
                T_prev=T_prev,
                fillable_dof=fillable_dof,
                headspace_dof=headspace_dof,
                fill_flag=float(fill.value),
                pla=pla,
                air=air,
                water_const=water_const,
                phase=phase,
                use_tabulated_water_ice=use_tabulated_water_ice,
            )

            if (
                debug_material_probe_index is not None
                and float(fill.value) > 0.5
                and fill_time is not None
                and t >= fill_time - 1e-12
            ):
                t_since_fill_debug = t - fill_time
                due_debug = (next_material_debug_since_fill <= 0.0) or (
                    t_since_fill_debug >= next_material_debug_since_fill - 1e-12
                )
                if due_debug:
                    T_used = eval_prepared(T_prev, probe_pts3, probe_cells, probe_mask, comm_pts)
                    k_used = eval_prepared(k_eff, probe_pts3, probe_cells, probe_mask, comm_pts)
                    rho_cp_used = eval_prepared(rho_cp_eff, probe_pts3, probe_cells, probe_mask, comm_pts)

                    idx_dbg = int(debug_material_probe_index)
                    if idx_dbg < 0 or idx_dbg >= len(probe_z_m):
                        raise IndexError(
                            f"debug_material_probe_index={idx_dbg} is out of range for {len(probe_z_m)} probes"
                        )

                    T_dbg = float(T_used[idx_dbg])
                    k_dbg = float(k_used[idx_dbg])
                    cp_eff_dbg = float(rho_cp_used[idx_dbg] / water_const.rho)

                    if use_tabulated_water_ice:
                        k_tab, cp_tab = water_ice_k_cp_from_tables(np.array([T_dbg]), phase.Tf)
                        k_tab = float(k_tab[0])
                        cp_tab = float(cp_tab[0])
                    else:
                        k_tab = float(water_const.k)
                        cp_tab = float(water_const.cp)

                    cp_lat_dbg = float(apparent_cp_bump(np.array([T_dbg]), phase.Tf, phase.L_latent, phase.dT_mushy)[0])
                    phase_dbg = "ice" if T_dbg < phase.Tf else "liquid"

                    if show_progress and comm.rank == 0:
                        z_dbg_mm = 1000.0 * float(probe_z_m[idx_dbg])
                        print(
                            "[material-debug] "
                            f"t={t:.2f}s  t_since_fill={t_since_fill_debug:.2f}s  "
                            f"probe_z={z_dbg_mm:.1f} mm  "
                            f"T_used={T_dbg:.4f} °C  phase={phase_dbg}  "
                            f"k_used={k_dbg:.4f} W/m/K  "
                            f"cp_table={cp_tab:.1f} J/kg/K  "
                            f"cp_latent={cp_lat_dbg:.1f} J/kg/K  "
                            f"cp_eff_used={cp_eff_dbg:.1f} J/kg/K"
                        )

                    if debug_material_every_s > 0.0:
                        next_material_debug_since_fill = t_since_fill_debug + float(debug_material_every_s)
                    else:
                        next_material_debug_since_fill = t_since_fill_debug + dt

            T_plate_const.value = PETSc.ScalarType(plate_temperature_C(t))
            problem.solve()
            T_prev.x.array[:] = T.x.array
            T_prev.x.scatter_forward()

            temps = eval_prepared(T, probe_pts3, probe_cells, probe_mask, comm_pts)

            if float(fill.value) < 0.5:
                probe_times.append(t)
                probe_values.append(np.array(temps, dtype=np.float64))
                while len(probe_times) > 0 and (t - probe_times[0]) > prefill.probe_window_s + 1e-12:
                    probe_times.popleft()
                    probe_values.popleft()

                if prefill_mode == "probe_stabilized":
                    if t >= prefill.min_prefill_s and _probe_stabilized(
                        probe_times,
                        probe_values,
                        window_s=prefill.probe_window_s,
                        tol_C=prefill.probe_tol_C,
                    ):
                        if show_progress and comm.rank == 0:
                            spans = np.ptp(np.asarray(probe_values, dtype=np.float64), axis=0)
                            print(
                                "\nEMPTY-MOLD probes stabilized -> "
                                f"spans over last {prefill.probe_window_s:.1f}s = "
                                f"[{spans[0]:.4f}, {spans[1]:.4f}, {spans[2]:.4f}] °C"
                            )
                        do_fill(t)
                    elif t >= prefill.max_prefill_s - 1e-12:
                        raise RuntimeError(
                            "Probe stabilization criterion was not reached before max_prefill_s. "
                            "Increase max_prefill_s or relax probe_window_s / probe_tol_C."
                        )

            t_since_fill = (t - fill_time) if (float(fill.value) > 0.5 and fill_time is not None) else np.nan

            if float(fill.value) > 0.5:
                Tz = eval_prepared(T, front_pts3, front_cells, front_mask, comm_pts)
                zf = front_from_temperature_threshold(z_samp, Tz, front_threshold_C)
                zf_rel = zf - geom.t_base
                v_front = 0.0 if np.isnan(zf_prev) else (zf - zf_prev) * 1000.0 / dt
                zf_prev = zf
            else:
                zf = np.nan
                zf_rel = np.nan
                v_front = np.nan

            if float(fill.value) > 0.5:
                Tzw = eval_prepared(T, wall_pts3, wall_cells, wall_mask, comm_pts)
                zf_wall = front_from_temperature_threshold(z_samp, Tzw, front_threshold_C)
                v_wall = 0.0 if np.isnan(zf_wall_prev) else (zf_wall - zf_wall_prev) * 1000.0 / dt
                zf_wall_prev = zf_wall
                Tmax_fillable_C = _global_fillable_max_temperature(comm, T, fillable_dof)
                freeze_complete = int(Tmax_fillable_C <= freeze_threshold_C)
            else:
                zf_wall = np.nan
                v_wall = np.nan
                Tmax_fillable_C = np.nan
                freeze_complete = 0

            zf_r_mm = None
            curve_written_this_step = False
            if enable_front_curve and float(fill.value) > 0.5:
                due_by_time = (next_curve_since_fill <= 0.0) or (t_since_fill >= next_curve_since_fill - 1e-12)
                due_by_stop = np.isfinite(Tmax_fillable_C) and (Tmax_fillable_C <= freeze_threshold_C)
                if due_by_time or due_by_stop:
                    Tcurve = eval_prepared(T, curve_pts3, curve_cells, curve_mask, comm_pts)
                    Tcurve = Tcurve.reshape(len(r_samp), len(z_samp_curve))

                    zf_r_m = np.empty(len(r_samp), dtype=float)
                    for j in range(len(r_samp)):
                        zf_r_m[j] = front_from_temperature_threshold(z_samp_curve, Tcurve[j, :], front_threshold_C)
                    zf_r_mm = zf_r_m * 1000.0
                    curve_written_this_step = True

                    if due_by_time:
                        next_curve_since_fill = t_since_fill + float(front_curve_every_s)

            if comm.rank == 0:
                if f_probe is not None:
                    f_probe.write(f"{t:.6f},{t_since_fill:.6f},{temps[0]:.6f},{temps[1]:.6f},{temps[2]:.6f}\n")

                if float(fill.value) > 0.5:
                    f_front.write(
                        f"{t:.6f},{t_since_fill:.6f},{float(fill.value):.1f},"
                        f"{front_definition_mode},{front_threshold_C:.9e},"
                        f"{zf:.9e},{zf*1000.0:.6f},{zf_rel*1000.0:.6f},{v_front:.6f},"
                        f"{zf_wall:.9e},{zf_wall*1000.0:.6f},{v_wall:.6f},"
                        f"{Tmax_fillable_C:.6f},{freeze_complete:d}\n"
                    )
                else:
                    f_front.write(
                        f"{t:.6f},{t_since_fill:.6f},{float(fill.value):.1f},"
                        f"{front_definition_mode},{front_threshold_C:.9e},"
                        "nan,nan,nan,nan,nan,nan,nan,nan,0\n"
                    )

                if enable_front_curve and curve_written_this_step:
                    f_curve.write(
                        f"{t:.6f},{t_since_fill:.6f},{float(fill.value):.1f},"
                        f"{front_definition_mode},{front_threshold_C:.9e},"
                        + ",".join(f"{val:.6f}" for val in zf_r_mm)
                        + "\n"
                    )

            if xdmf is not None and t >= next_write - 1e-12:
                xdmf.write_function(T, t)
                next_write += write_every

            if float(fill.value) > 0.5 and freeze_complete:
                if show_progress and comm.rank == 0:
                    print(
                        f"\nWater-filled region fully frozen at t_since_fill={t_since_fill:.2f}s "
                        f"with Tmax_fillable={Tmax_fillable_C:.4f} °C <= {freeze_threshold_C:.4f} °C. Stopping."
                    )
                if xdmf is not None:
                    xdmf.write_function(T, t)
                break

            if stop_when_wall_frozen and float(fill.value) > 0.5 and np.isfinite(zf_wall):
                if zf_wall >= geom.H_fill - 1e-12:
                    if show_progress and comm.rank == 0:
                        print(
                            f"\nWall-line frozen at t_since_fill={t_since_fill:.2f}s "
                            f"(r_wall={r_wall*1000.0:.3f} mm). Stopping because stop_when_wall_frozen=True."
                        )
                    if xdmf is not None:
                        xdmf.write_function(T, t)
                    break

            if show_progress and comm.rank == 0 and (step % max(1, n_steps // 100) == 0 or step == n_steps):
                elapsed = time.perf_counter() - t0_wall
                sps = elapsed / step
                eta = sps * (n_steps - step)
                print(f"\rstep {step}/{n_steps}  {sps:.3f} s/step  ETA {eta:6.1f} s", end="", flush=True)

        if comm.rank == 0:
            if f_probe is not None:
                f_probe.close()
            f_front.close()
            if enable_front_curve and f_curve is not None:
                f_curve.close()

            if show_progress:
                print("\nDone.")
                if xdmf is not None:
                    print(f"  XDMF        : {xdmf_path}")
                if f_probe is not None:
                    print(f"  PROBES CSV  : {probes_path}")
                print(f"  FRONT CSV   : {front_path}")
                if enable_front_curve:
                    print(f"  FRONT CURVE : {front_curve_path}")
