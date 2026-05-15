# -*- coding: utf-8 -*-
"""
Lorenzo-style Python v13

Main Te-fit changes relative to v12
- Te mtanh fit now follows the uploaded IDL logic more closely:
    * fit cloud = all selected raw TS Te points with psi_N >= xpf
    * no Te-point rejection based on measured ete
    * Te fit weights use sqrt(Te), not measured ete
    * no hard bounds on Te mtanh fit by default
    * Te psi-shift is applied after the fit, like IDL
    * Te is NOT re-fit after the shift
    * Te max gradient is reported as max(-dTe/dpsi), like IDL
- ne path is kept close to the previous Python behavior
- Mapping logic is unchanged from v12

Important note
- Exact equivalence to IDL map3d(..., cubic=-0.5) still cannot be guaranteed
  because the IDL map3d source is unavailable.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from pyuda import Client
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit, OptimizeWarning

client = Client()
warnings.simplefilter("ignore", OptimizeWarning)


# =============================================================================
# USER INPUTS
# =============================================================================
# Each dictionary describes one analysis case.  For normal use, copy one of the
# examples, update the shot/time/ELM-fraction fields, and add it to `in_list`.
in1 = {
    "shot": 49107,          # MAST-U shot number.
    "tr": [0.60, 0.80],     # Profile sample time range [s].
    "perc_elm": [0.80, 0.95],  # Accepted fraction of each ELM cycle.

    # For HFS-only test: set r_shift_lfs = 0.0
    # For LFS-only test: set r_shift_hfs = 0.0
    "r_shift_hfs": 0.005,
    "r_shift_lfs": 0.0005,
#    "r_shift_hfs": 0.00,
#    "r_shift_lfs": 0.000,
    "te_sep": 0.05,        # Target separatrix Te used by the Te shift logic.
    "neoff": 1,            # Density scrape-off-layer offset switch.
    "teoff": 0,            # Temperature scrape-off-layer offset switch.
    "rem_ne_peak": [0.0, 0.0],  # Optional psi_N interval to remove from ne fit.
    "xpf": 0.8,            # Minimum psi_N included in edge fit clouds.
    "ndeg": 5.0,           # Polynomial degree for the constrained core fit.
    "equi": "EPM",        # Equilibrium tree prefix used for flux mapping.
}

in2 = {
    "shot": 52429,
    "tr": [0.38, 0.48],
    "perc_elm": [0.80, 0.95],

    # For HFS-only test: set r_shift_lfs = 0.0
    # For LFS-only test: set r_shift_hfs = 0.0
    "r_shift_hfs": 0.00,
    "r_shift_lfs": 0.000,
#    "r_shift_hfs": 0.00,
#    "r_shift_lfs": 0.000,
    "te_sep": 0.05,
    "neoff": 1,
    "teoff": 0,
    "rem_ne_peak": [0.0, 0.0],
    "xpf": 0.8,
    "ndeg": 5.0,
    "equi": "EPM",
}

# =============================================================================
#in1 = {
#    "shot": 52600,
#    "tr": [0.40, 0.60],
#    "perc_elm": [0.80, 0.99],
#
#    # For HFS-only test: set r_shift_lfs = 0.0
#    # For LFS-only test: set r_shift_hfs = 0.0
#    "r_shift_hfs": 0.001,
#    "r_shift_lfs": 0.0005,
##    "r_shift_hfs": 0.00,
##    "r_shift_lfs": 0.000,
#    "te_sep": 0.05,
#    "neoff": 0,
#    "teoff": 0,
#    "rem_ne_peak": [0.0, 0.0],
#    "xpf": 0.7,
#    "ndeg": 5.0,
#    "equi": "EPM",
#}


in_list = [in1]  # Add more dictionaries here for batch profile fitting.

ELM_DIR = "."       # Directory containing telm_<shot>.npz files from step 03.
SAVE_OUTPUT = True  # Save compressed .npz fit products and mapped-point CSVs.
OUTDIR = "."        # Directory for numerical output files.
SAVE_PLOTS = False  # Set True to save diagnostic figures for each case.
PLOT_DIR = "lorenzo_style_fit_plots_v13"


# -----------------------------------------------------------------------------
# MAPPING OPTIONS
# -----------------------------------------------------------------------------
USE_IDL_MEAN_Z = False
MAPPING_METHOD = "linear"   # "linear", "cubic_like", "idl_like_best_effort"
DEBUG_MAPPING_TIME_INDEX = 0

# In IDL the Te fit is first obtained, then a horizontal psi shift is applied
# so that Te(psi_N=1) = te_sep. This switch reproduces that behavior.
APPLY_GLOBAL_PSI_SHIFT = False

# Plot raw mapping while tuning side-specific shifts.
PLOT_RAW_MAPPING = True

# -----------------------------------------------------------------------------
# IDL-STYLE Te FIT OPTIONS
# -----------------------------------------------------------------------------
TE_USE_IDL_SQRT_WEIGHTS = True
TE_USE_BOUNDS = False
TE_REFIT_AFTER_SHIFT = False   # IDL-like = False


# =============================================================================
# BASIC HELPERS
# =============================================================================
def extract_time_and_data(d):
    """
    Return a pyuda signal as separate time, data, and dimension arrays without assuming one fixed signal layout.
    """
    data = np.asarray(d.data)
    dims = getattr(d, "dims", None)
    if dims is None:
        dims = []

    time = None
    if hasattr(d, "time") and hasattr(d.time, "data"):
        time = np.asarray(d.time.data)

    dim_arrays = []
    for dim in dims:
        if hasattr(dim, "data"):
            dim_arrays.append(np.asarray(dim.data))
        else:
            dim_arrays.append(None)

    return time, data, dim_arrays


def get_uda(sig, shot):
    """
    Fetch one signal for one shot through the shared pyuda client.
    """
    return client.get(sig, shot)


def load_elm_npz(shot, elm_dir="."):
    """
    Load ELM timing data saved by the ELM detection scripts and convert scalar arrays to plain Python values.
    """
    path = os.path.join(elm_dir, f"telm_{shot}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ELM file not found: {path}")

    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}

    for key in ["shot", "felm", "nelm", "thr", "smooth_pts", "min_gap_s"]:
        if key in out and np.ndim(out[key]) == 0:
            out[key] = out[key].item()

    if "event_mode" in out and np.ndim(out["event_mode"]) == 0:
        out["event_mode"] = str(out["event_mode"].item())

    return out


def select_indices_in_elm_fraction(sample_times, telm, perc_elm):
    """
    Keep profile sample indices whose times fall inside the requested fraction of the surrounding ELM period.
    """
    sample_times = np.asarray(sample_times).ravel()
    telm = np.asarray(telm).ravel()

    keep = []
    for j in range(len(sample_times)):
        tj = sample_times[j]
        ind = np.searchsorted(telm, tj, side="right") - 1
        if ind >= 0 and (ind + 1) < len(telm):
            dt = tj - telm[ind]
            dtelm = telm[ind + 1] - telm[ind]
            if dtelm > 0:
                frac = dt / dtelm
                if (frac >= perc_elm[0]) and (frac <= perc_elm[1]):
                    keep.append(j)

    return np.asarray(keep, dtype=int)


def reorder_ts_to_rt(data, time, r):
    """
    Normalize Thomson-scattering arrays to the repository convention of radius-by-time shape.
    """
    data = np.asarray(data)
    time = np.asarray(time).ravel()
    r = np.asarray(r)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    if data.shape[0] == len(time):
        data = data.T
    elif data.shape[1] == len(time):
        pass
    else:
        if r.ndim == 2 and data.shape == r.shape:
            pass
        else:
            raise ValueError(
                f"Could not match data shape {data.shape} to time length {len(time)}"
            )
    return data


def reorder_rz_to_rt(arr, time):
    """
    Normalize R or Z coordinate arrays to radius-by-time shape, adding a time axis when the coordinate is static.
    """
    arr = np.asarray(arr)
    time = np.asarray(time).ravel()

    if arr.ndim == 1:
        return arr[:, None]

    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D R/Z array, got shape {arr.shape}")

    if arr.shape[0] == len(time):
        arr = arr.T
    elif arr.shape[1] == len(time):
        pass
    else:
        raise ValueError(
            f"Could not match R/Z shape {arr.shape} to time length {len(time)}"
        )
    return arr


# =============================================================================
# FIT FUNCTIONS
# =============================================================================
def mtanh(x, a0, a1, a2, a3, a4=0.0):
    """
    Evaluate the modified hyperbolic tangent edge-profile model used for pedestal fits.
    """
    xx = (a0 - x) / (2.0 * a1)
    mt = ((1.0 + a3 * xx) * np.exp(xx) - np.exp(-xx)) / (np.exp(xx) + np.exp(-xx))
    edge = (a2 - a4) / 2.0 * (mt + 1.0) + a4
    return edge


def mtanh_off(x, a0, a1, a2, a3, a5):
    """
    Evaluate a non-negative modified hyperbolic tangent with an explicit scrape-off-layer offset.
    """
    xx = (a0 - x) / (2.0 * a1)
    mt = ((1.0 + a3 * xx) * np.exp(xx) - np.exp(-xx)) / (np.exp(xx) + np.exp(-xx))
    edge = (a2 - a5) / 2.0 * (mt + 1.0) + a5
    return np.maximum(edge, 0.0)


def build_idl_style_te_sigma(y):
    """
    Build IDL-style Te fit uncertainties proportional to sqrt(Te) with safe clipping at zero.
    """
    y = np.asarray(y, dtype=float).ravel()
    return np.sqrt(np.clip(y, 1e-8, None))


def fit_mtanh_profile(
    x,
    y,
    yerr=None,
    use_offset=False,
    quantity="Te",
    use_bounds=True,
):
    """
    Fit the generic modified-tanh pedestal model to finite profile points.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    m = np.isfinite(x) & np.isfinite(y)

    yerr_use = None
    if yerr is not None:
        yerr = np.asarray(yerr).ravel()
        m &= np.isfinite(yerr) & (yerr > 0)

    x = x[m]
    y = y[m]
    if yerr is not None:
        yerr_use = yerr[m]

    if len(x) < 6:
        raise RuntimeError(f"Not enough points to fit {quantity}: {len(x)} points")

    s = np.argsort(x)
    x = x[s]
    y = y[s]
    if yerr_use is not None:
        yerr_use = yerr_use[s]

    ymax = np.nanmax(y)
    ymin = np.nanmin(y)

    if quantity.lower() == "te":
        p0 = [0.97, 0.009, max(ymax, 0.2), 0.0]
    else:
        p0 = [0.97, 0.009, max(ymax, 1.0), 0.0]

    if use_offset:
        p0 = p0 + [max(ymin, 0.0)]

    if use_offset:
        fit_fun = mtanh_off
        if use_bounds:
            bounds = (
                [0.85, 1e-4, 0.0, -50.0, 0.0],
                [1.10, 0.08, 10.0 * max(ymax, 1.0), 50.0, 10.0 * max(ymax, 1.0)],
            )
        else:
            bounds = (-np.inf, np.inf)
    else:
        fit_fun = lambda xx, a0, a1, a2, a3: mtanh(xx, a0, a1, a2, a3, 0.0)
        if use_bounds:
            bounds = (
                [0.85, 1e-4, 0.0, -50.0],
                [1.10, 0.08, 10.0 * max(ymax, 1.0), 50.0],
            )
        else:
            bounds = (-np.inf, np.inf)

    popt, pcov = curve_fit(
        fit_fun,
        x,
        y,
        p0=p0,
        sigma=yerr_use if yerr_use is not None else None,
        absolute_sigma=(yerr_use is not None),
        bounds=bounds,
        maxfev=50000,
    )

    return {
        "x": x,
        "y": y,
        "yerr": yerr_use,
        "popt": popt,
        "pcov": pcov,
        "use_offset": use_offset,
        "quantity": quantity,
    }


def fit_mtanh_profile_te_idl_style(x, y, use_offset=False):
    """
    IDL-like Te fit:
    - keep points based only on finite x and finite y
    - use sigma = sqrt(Te)
    - ignore measured ete
    - by default no hard bounds
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 6:
        raise RuntimeError(f"Not enough points to fit Te: {len(x)} points")

    s = np.argsort(x)
    x = x[s]
    y = y[s]

    sigma = build_idl_style_te_sigma(y) if TE_USE_IDL_SQRT_WEIGHTS else None

    return fit_mtanh_profile(
        x,
        y,
        yerr=sigma,
        use_offset=use_offset,
        quantity="Te",
        use_bounds=TE_USE_BOUNDS,
    )


def eval_fit_on_grid(fitres, xgrid):
    """
    Evaluate a fitted mtanh result on a caller-supplied psi_N grid.
    """
    popt = fitres["popt"]
    if fitres["use_offset"]:
        return mtanh_off(xgrid, *popt)
    return mtanh(xgrid, *popt, 0.0)


def extract_pedestal_params(fitres, xgrid):
    """
    Derive pedestal height, width, position, and maximum gradient from a fitted profile on a dense grid.
    """
    popt = fitres["popt"]
    pos = popt[0]
    width = 4.0 * popt[1]
    height = popt[2]
    sol = popt[4] if fitres["use_offset"] else 0.0

    yfit = eval_fit_on_grid(fitres, xgrid)
    dy = np.gradient(yfit, xgrid)
    grad_idl = -dy

    return {
        "ped_pos": pos,
        "ped_width": width,
        "ped_height": height,
        "sol_offset": sol,
        "max_grad": np.nanmax(grad_idl),
        "xgrid": xgrid,
        "yfit": yfit,
        "dy": dy,
        "grad_idl": grad_idl,
    }


# =============================================================================
# CORE POLYNOMIAL FITS
# =============================================================================
def fit_poly_with_constraints(x, y, x0, y0, dy0, degree=5, yerr=None):
    """
    Fit a polynomial core profile while forcing its value and slope to match the pedestal fit at a join point.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    m = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        yerr = np.asarray(yerr).ravel()
        m &= np.isfinite(yerr) & (yerr > 0)

    x = x[m]
    y = y[m]
    if yerr is not None:
        yerr = yerr[m]

    if len(x) < max(4, int(degree) + 1):
        degree = min(int(degree), max(1, len(x) - 2))

    degree = int(max(1, degree))
    ncoef = degree + 1

    A = np.vstack([x ** (degree - i) for i in range(ncoef)]).T

    c0 = np.array([x0 ** (degree - i) for i in range(ncoef)], dtype=float)
    c1 = np.array(
        [
            (degree - i) * x0 ** (degree - i - 1) if (degree - i) >= 1 else 0.0
            for i in range(ncoef)
        ],
        dtype=float,
    )

    C = np.vstack([c0, c1])
    d = np.array([y0, dy0], dtype=float)

    if yerr is not None:
        W = np.diag(1.0 / np.maximum(yerr, 1e-12))
        Aw = W @ A
        yw = W @ y
    else:
        Aw = A
        yw = y

    ATA = Aw.T @ Aw
    ATy = Aw.T @ yw

    KKT = np.block([
        [ATA, C.T],
        [C,   np.zeros((2, 2))]
    ])
    rhs = np.concatenate([ATy, d])

    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    coeff = sol[:ncoef]

    return {
        "coeff": coeff,
        "degree": degree,
        "x0": x0,
        "y0": y0,
        "dy0": dy0,
    }


def eval_poly_core_fit(fitres, xgrid):
    """
    Evaluate a constrained polynomial fit result on a grid.
    """
    return np.polyval(fitres["coeff"], xgrid)


# =============================================================================
# ESSIVE FIT
# =============================================================================
def profile_fit_essive_py(x, p):
    """
    Evaluate the ESSIVE-style edge-plus-core profile model.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    height = p[0]
    offset = p[1]
    core_incline = p[2]
    pos = p[3]
    w4 = np.maximum(p[4], 1e-5)
    core_edge = p[5]
    c6 = p[6]
    c7 = p[7]
    c8 = p[8]

    xx = (pos - x) / (2.0 * w4)
    mt = ((1.0 + core_incline * xx) * np.exp(xx) - np.exp(-xx)) / (np.exp(xx) + np.exp(-xx))
    edge = (height - offset) / 2.0 * (mt + 1.0) + offset

    dx = np.clip(core_edge - x, 0.0, None)
    core_shape = c6 * dx**2 + c7 * dx**3 + c8 * dx**4

    return np.maximum(edge + core_shape, 0.0)


def fit_profile_fit_essive(x, y, yerr, p0):
    """
    Fit the ESSIVE-style profile model using bounded weighted nonlinear least squares.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    x = x[m]
    y = y[m]
    yerr = yerr[m]

    def model(xx, p0_, p1_, p2_, p3_, p4_, p5_, p6_, p7_, p8_):
        pars = np.array([p0_, p1_, p2_, p3_, p4_, p5_, p6_, p7_, p8_], dtype=float)
        return profile_fit_essive_py(xx, pars)

    lower = [0.0, -1.0, -20.0, 0.85, 1e-4, 0.3, -100.0, -100.0, -100.0]
    upper = [20.0, 10.0, 20.0, 1.10, 0.10, 1.20, 100.0, 100.0, 100.0]

    popt, pcov = curve_fit(
        model,
        x,
        y,
        p0=p0,
        sigma=yerr,
        absolute_sigma=True,
        bounds=(lower, upper),
        maxfev=100000,
    )
    return popt, pcov


# =============================================================================
# EQUILIBRIUM / PSI_N
# =============================================================================
def _coord_to_fractional_index(grid, x):
    """
    Convert physical grid coordinates to fractional array indices for scipy.ndimage interpolation.
    """
    grid = np.asarray(grid, dtype=float).ravel()
    x = np.asarray(x, dtype=float)

    if grid.size < 2:
        return np.full_like(x, np.nan, dtype=float)

    idx = np.arange(grid.size, dtype=float)
    return np.interp(x, grid, idx, left=np.nan, right=np.nan)


def _map_psin_3d_points(r_pts, z_pts, t_pts, fluxdb, method="idl_like_best_effort"):
    """
    Map R, Z, and time points onto normalized poloidal flux using the selected interpolation method.
    """
    r_pts = np.asarray(r_pts, dtype=float).ravel()
    z_pts = np.asarray(z_pts, dtype=float).ravel()
    t_pts = np.asarray(t_pts, dtype=float).ravel()

    psin_cube = fluxdb["psin"]
    r_grid = fluxdb["r_grid"]
    z_grid = fluxdb["z_grid"]
    t_grid = fluxdb["t_flux"]

    if method == "linear":
        interp3d = RegularGridInterpolator(
            (r_grid, z_grid, t_grid),
            psin_cube,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        pts = np.column_stack([r_pts, z_pts, t_pts])
        return interp3d(pts)

    if method in ("cubic_like", "idl_like_best_effort"):
        r_idx = _coord_to_fractional_index(r_grid, r_pts)
        z_idx = _coord_to_fractional_index(z_grid, z_pts)
        t_idx = _coord_to_fractional_index(t_grid, t_pts)

        out = np.full_like(r_pts, np.nan, dtype=float)
        valid = np.isfinite(r_idx) & np.isfinite(z_idx) & np.isfinite(t_idx)

        if np.any(valid):
            coords = np.vstack([r_idx[valid], z_idx[valid], t_idx[valid]])
            out[valid] = map_coordinates(
                psin_cube,
                coords,
                order=3,
                mode="constant",
                cval=np.nan,
                prefilter=True,
            )
        return out

    raise ValueError(f"Unknown mapping method: {method}")


def load_flux_grid(shot, equilibrium="EPM"):
    """
    Load equilibrium flux data and construct a normalized psi_N cube with aligned R, Z, and time axes.
    """
    prefix = equilibrium.lower()

    flux_sig = f"/{prefix}/output/profiles2D/poloidalFlux"
    psi_axis_sig = f"/{prefix}/output/globalParameters/psiAxis"
    psi_bnd_sig = f"/{prefix}/output/globalParameters/psiBoundary"

    dflux = get_uda(flux_sig, shot)
    t_flux, polflux, dims = extract_time_and_data(dflux)

    polflux = np.asarray(polflux)
    t_flux = np.asarray(t_flux).ravel() if t_flux is not None else None

    if polflux.ndim != 3:
        raise RuntimeError(f"Expected 3D poloidal flux array, got shape {polflux.shape}")

    dim_arrays = []
    for dd in dims:
        if dd is None:
            dim_arrays.append(None)
        else:
            dim_arrays.append(np.asarray(dd).ravel())

    time_axis = None
    if t_flux is not None:
        for ax, dd in enumerate(dim_arrays):
            if dd is not None and len(dd) == len(t_flux):
                time_axis = ax
                break

    if time_axis is None and t_flux is not None:
        for ax, n in enumerate(polflux.shape):
            if n == len(t_flux):
                time_axis = ax
                break

    if time_axis is None:
        raise RuntimeError(f"Could not identify time axis for polflux shape {polflux.shape}")

    spatial_axes = [ax for ax in range(3) if ax != time_axis]

    spatial_dim_arrays = []
    for ax in spatial_axes:
        if ax < len(dim_arrays):
            spatial_dim_arrays.append(dim_arrays[ax])
        else:
            spatial_dim_arrays.append(None)

    if spatial_dim_arrays[0] is None or len(spatial_dim_arrays[0]) != polflux.shape[spatial_axes[0]]:
        spatial_dim_arrays[0] = np.arange(polflux.shape[spatial_axes[0]], dtype=float)

    if spatial_dim_arrays[1] is None or len(spatial_dim_arrays[1]) != polflux.shape[spatial_axes[1]]:
        spatial_dim_arrays[1] = np.arange(polflux.shape[spatial_axes[1]], dtype=float)

    polflux = np.moveaxis(polflux, [spatial_axes[0], spatial_axes[1], time_axis], [0, 1, 2])

    r_grid = np.asarray(spatial_dim_arrays[0]).ravel()
    z_grid = np.asarray(spatial_dim_arrays[1]).ravel()

    if len(r_grid) > 1 and np.any(np.diff(r_grid) < 0):
        idx = np.argsort(r_grid)
        r_grid = r_grid[idx]
        polflux = polflux[idx, :, :]

    if len(z_grid) > 1 and np.any(np.diff(z_grid) < 0):
        idx = np.argsort(z_grid)
        z_grid = z_grid[idx]
        polflux = polflux[:, idx, :]

    daxis = get_uda(psi_axis_sig, shot)
    tb_axis, psi_axis, _ = extract_time_and_data(daxis)

    dbnd = get_uda(psi_bnd_sig, shot)
    tb_bnd, psi_bnd, _ = extract_time_and_data(dbnd)

    t_axis = np.asarray(tb_axis).ravel()
    psi_axis = np.asarray(psi_axis).ravel()
    t_bnd = np.asarray(tb_bnd).ravel()
    psi_bnd = np.asarray(psi_bnd).ravel()

    if t_flux is None:
        raise RuntimeError("Flux grid has no time axis")

    psi_axis_on_flux_t = np.interp(t_flux, t_axis, psi_axis, left=np.nan, right=np.nan)
    psi_bnd_on_flux_t = np.interp(t_flux, t_bnd, psi_bnd, left=np.nan, right=np.nan)

    denom = psi_bnd_on_flux_t - psi_axis_on_flux_t
    denom[np.abs(denom) < 1e-12] = np.nan

    psin = (polflux - psi_axis_on_flux_t[None, None, :]) / denom[None, None, :]

    return {
        "t_flux": np.asarray(t_flux).ravel(),
        "r_grid": np.asarray(r_grid).ravel(),
        "z_grid": np.asarray(z_grid).ravel(),
        "polflux": polflux,
        "psin": psin,
        "t_axis": t_axis,
        "psi_axis": psi_axis,
        "t_bnd": t_bnd,
        "psi_bnd": psi_bnd,
        "equilibrium": equilibrium,
    }


def psin_at_time_from_grid_old_2d(r_pts, z_pts, t0, fluxdb):
    """
    Evaluate psi_N at one time using the older 2D interpolation path retained for comparison.
    """
    t_flux = fluxdb["t_flux"]
    polflux = fluxdb["polflux"]
    r_grid = fluxdb["r_grid"]
    z_grid = fluxdb["z_grid"]

    it = np.argmin(np.abs(t_flux - t0))
    psi2d = polflux[:, :, it]

    psi_axis_t = np.interp(t0, fluxdb["t_axis"], fluxdb["psi_axis"], left=np.nan, right=np.nan)
    psi_bnd_t = np.interp(t0, fluxdb["t_bnd"], fluxdb["psi_bnd"], left=np.nan, right=np.nan)

    interp2d = RegularGridInterpolator(
        (r_grid, z_grid),
        psi2d,
        bounds_error=False,
        fill_value=np.nan,
    )

    pts = np.column_stack([r_pts, z_pts])
    psi = interp2d(pts)

    denom = psi_bnd_t - psi_axis_t
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return np.full_like(psi, np.nan, dtype=float)

    return (psi - psi_axis_t) / denom


def map_profile_to_psin_3d(
    shot,
    r_rt,
    z_rt,
    t_sel,
    equilibrium="EPM",
    use_idl_mean_z=True,
    mapping_method="idl_like_best_effort",
    fluxdb=None,
):
    """
    Map one profile slice from R/Z/time coordinates to psi_N using the 3D equilibrium grid.
    """
    if fluxdb is None:
        fluxdb = load_flux_grid(shot, equilibrium=equilibrium)

    nch, nt = r_rt.shape
    psin = np.full((nch, nt), np.nan, dtype=float)

    for k in range(nt):
        r_query = np.asarray(r_rt[:, k], dtype=float)

        if use_idl_mean_z:
            z_mean_k = float(np.nanmean(z_rt[:, k]))
            z_query = np.full_like(r_query, z_mean_k, dtype=float)
        else:
            z_query = np.asarray(z_rt[:, k], dtype=float)

        t_query = np.full_like(r_query, float(t_sel[k]), dtype=float)

        psin[:, k] = _map_psin_3d_points(
            r_query,
            z_query,
            t_query,
            fluxdb,
            method=mapping_method,
        )

    return psin, fluxdb


def map_profile_to_psin_old_pointwise_z(
    shot,
    r_rt,
    z_rt,
    t_sel,
    equilibrium="EPM",
    fluxdb=None,
):
    """
    Map one profile slice with the older pointwise-Z method retained for IDL comparison studies.
    """
    if fluxdb is None:
        fluxdb = load_flux_grid(shot, equilibrium=equilibrium)

    nch, nt = r_rt.shape
    psin = np.full((nch, nt), np.nan, dtype=float)

    for k in range(nt):
        r_query = np.asarray(r_rt[:, k], dtype=float)
        z_query = np.asarray(z_rt[:, k], dtype=float)
        psin[:, k] = psin_at_time_from_grid_old_2d(r_query, z_query, float(t_sel[k]), fluxdb)

    return psin, fluxdb


def map_ts_to_psin(
    shot,
    r_rt,
    z_rt,
    t_sel,
    equilibrium="EPM",
    use_idl_mean_z=True,
    mapping_method="idl_like_best_effort",
    fluxdb=None,
):
    """
    Map Thomson-scattering channel coordinates for selected time indices onto psi_N.
    """
    return map_profile_to_psin_3d(
        shot,
        r_rt,
        z_rt,
        t_sel,
        equilibrium=equilibrium,
        use_idl_mean_z=use_idl_mean_z,
        mapping_method=mapping_method,
        fluxdb=fluxdb,
    )


def map_ti_to_psin(
    shot,
    r_rt,
    z_rt,
    t_sel,
    equilibrium="EPM",
    use_idl_mean_z=True,
    mapping_method="idl_like_best_effort",
    fluxdb=None,
):
    """
    Map charge-exchange ion-temperature coordinates for selected time indices onto psi_N.
    """
    return map_profile_to_psin_3d(
        shot,
        r_rt,
        z_rt,
        t_sel,
        equilibrium=equilibrium,
        use_idl_mean_z=use_idl_mean_z,
        mapping_method=mapping_method,
        fluxdb=fluxdb,
    )


def build_mapping_debug(ts_time_sel, psin_old_2d, psin_new_3d, debug_time_index=0):
    """
    Collect the raw mapped coordinate arrays needed to inspect and tune the equilibrium mapping.
    """
    nt = psin_new_3d.shape[1]
    if nt == 0:
        return {
            "debug_time_index": None,
            "time_s": np.nan,
            "old_psin_slice": np.array([]),
            "new_psin_slice": np.array([]),
            "abs_diff_slice": np.array([]),
            "max_abs_diff": np.nan,
            "mean_abs_diff": np.nan,
        }

    k = int(np.clip(debug_time_index, 0, nt - 1))
    old_slice = np.asarray(psin_old_2d[:, k], dtype=float)
    new_slice = np.asarray(psin_new_3d[:, k], dtype=float)

    m = np.isfinite(old_slice) & np.isfinite(new_slice)
    abs_diff = np.abs(new_slice - old_slice)

    return {
        "debug_time_index": k,
        "time_s": float(ts_time_sel[k]),
        "old_psin_slice": old_slice,
        "new_psin_slice": new_slice,
        "abs_diff_slice": abs_diff,
        "max_abs_diff": float(np.nanmax(abs_diff[m])) if np.any(m) else np.nan,
        "mean_abs_diff": float(np.nanmean(abs_diff[m])) if np.any(m) else np.nan,
    }


# =============================================================================
# TS LOADING
# =============================================================================
def load_ts_data(shot, r_shift_hfs=0.0, r_shift_lfs=0.0):
    """
    Load Thomson-scattering density, temperature, uncertainty, and coordinate signals for one shot.
    """
    xr = get_uda("/ayc/r", shot)
    xz = get_uda("/ayc/z", shot)
    xte = get_uda("/ayc/T_e", shot)
    xne = get_uda("/ayc/n_e", shot)

    _, r_data, _ = extract_time_and_data(xr)
    _, z_data, _ = extract_time_and_data(xz)
    time_te, te_data, _ = extract_time_and_data(xte)
    _, ne_data, _ = extract_time_and_data(xne)

    time = np.asarray(time_te).ravel()

    te = reorder_ts_to_rt(te_data, time, r_data) / 1e3
    ne = reorder_ts_to_rt(ne_data, time, r_data) / 1e19

    r_rt = reorder_rz_to_rt(r_data, time).astype(float, copy=True)
    z_rt = reorder_rz_to_rt(z_data, time).astype(float, copy=True)

    if r_rt.shape[1] == 1 and len(time) > 1:
        r_rt = np.repeat(r_rt, len(time), axis=1)
    if z_rt.shape[1] == 1 and len(time) > 1:
        z_rt = np.repeat(z_rt, len(time), axis=1)

    ete = getattr(xte, "errors", None)
    if ete is None and hasattr(xte, "edata"):
        ete = np.asarray(xte.edata)
    if ete is not None:
        ete = reorder_ts_to_rt(ete, time, r_data) / 1e3
    else:
        ete = np.full_like(te, np.nan)

    ene = getattr(xne, "errors", None)
    if ene is None and hasattr(xne, "edata"):
        ene = np.asarray(xne.edata)
    if ene is not None:
        ene = reorder_ts_to_rt(ene, time, r_data) / 1e19
    else:
        ene = np.full_like(ne, np.nan)

    r_rt_unshifted = r_rt.copy()

    m_hfs = np.isfinite(r_rt_unshifted) & (r_rt_unshifted < 1.0)
    m_lfs = np.isfinite(r_rt_unshifted) & (r_rt_unshifted >= 1.0)

    r_rt[m_hfs] = r_rt[m_hfs] + float(r_shift_hfs)
    r_rt[m_lfs] = r_rt[m_lfs] + float(r_shift_lfs)

    return {
        "time": time,
        "r": r_rt,
        "r_unshifted": r_rt_unshifted,
        "z": z_rt,
        "te": te,
        "ne": ne,
        "ete": ete,
        "ene": ene,
    }


# =============================================================================
# CX / Ti LOADING
# =============================================================================
def fit_weighted_poly(x, y, yerr=None, deg=2):
    """
    Fit a weighted polynomial to finite points and return coefficients in numpy.polyval order.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    m = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        yerr = np.asarray(yerr).ravel()
        m &= np.isfinite(yerr) & (yerr > 0)

    x = x[m]
    y = y[m]
    if yerr is not None:
        yerr = yerr[m]

    if len(x) < max(4, deg + 2):
        raise RuntimeError(f"Not enough points for Ti polynomial fit: {len(x)}")

    s = np.argsort(x)
    x = x[s]
    y = y[s]
    if yerr is not None:
        yerr = yerr[s]

    deg_use = int(min(deg, max(1, len(x) - 2)))

    if yerr is not None:
        w = 1.0 / np.maximum(yerr, 1e-8)
        coeff = np.polyfit(x, y, deg_use, w=w)
    else:
        coeff = np.polyfit(x, y, deg_use)

    return {
        "x": x,
        "y": y,
        "yerr": yerr,
        "coeff": coeff,
        "deg": deg_use,
    }


def eval_poly_fit(fitres, xgrid):
    """
    Evaluate a polynomial fit dictionary returned by fit_weighted_poly.
    """
    return np.polyval(fitres["coeff"], xgrid)


def load_cx_ti_data(shot):
    """
    Load charge-exchange ion-temperature data and return it in a consistent dictionary format.
    """
    z_sig = "/ACT/CEL3/SS/Z"
    ti_sig = "/ACT/CEL3/SS/PVB/C5291/TEMPERATURE"
    dti_sig = "/ACT/CEL3/SS/PVB/C5291/TEMPERATURE_ERROR"

    dum2 = get_uda(ti_sig, shot)
    t_ti, ti_data, ti_dims = extract_time_and_data(dum2)

    if t_ti is None:
        raise RuntimeError("CX temperature has no time axis")

    t_ti = np.asarray(t_ti).ravel()
    ti_data = np.asarray(ti_data)

    dum3 = get_uda(dti_sig, shot)
    _, dti_data, _ = extract_time_and_data(dum3)
    dti_data = np.asarray(dti_data)

    xi = None
    if hasattr(dum2, "x"):
        try:
            xi = np.asarray(dum2.x).ravel()
        except Exception:
            xi = None

    if xi is None and len(ti_dims) > 0 and ti_dims[0] is not None:
        xi = np.asarray(ti_dims[0]).ravel()

    if xi is None:
        raise RuntimeError("Could not recover xi from CX temperature object")

    ti_rt = reorder_ts_to_rt(ti_data, t_ti, xi) / 1e3
    dti_rt = reorder_ts_to_rt(dti_data, t_ti, xi) / 1e3

    nch = ti_rt.shape[0]
    if xi.size != nch:
        if xi.size > nch:
            xi = xi[:nch]
        else:
            raise RuntimeError(f"xi length {xi.size} does not match Ti channels {nch}")

    zi_scalar = 0.01
    try:
        dum1 = get_uda(z_sig, shot)
        _, zi_data, _ = extract_time_and_data(dum1)
        zi_arr = np.asarray(zi_data).ravel()
        zi_arr = zi_arr[np.isfinite(zi_arr)]
        if zi_arr.size > 0:
            zi_scalar = float(np.nanmean(zi_arr))
    except Exception:
        pass

    r_rt = np.repeat(xi[:, None], len(t_ti), axis=1)
    z_rt = np.full((nch, len(t_ti)), zi_scalar, dtype=float)

    return {
        "time": t_ti,
        "r": r_rt,
        "z": z_rt,
        "ti": ti_rt,
        "dti": dti_rt,
        "xi": xi,
        "zi_scalar": zi_scalar,
    }


# =============================================================================
# EXTRA SELECTION HELPERS
# =============================================================================
def apply_rem_ne_peak(psin_flat, ne_flat, ene_flat, rem_ne_peak):
    """
    Remove density points from a configured psi_N interval before fitting, matching the legacy workflow option.
    """
    rem_ne_peak = np.asarray(rem_ne_peak, dtype=float).ravel()
    if rem_ne_peak.size != 2:
        return psin_flat, ne_flat, ene_flat

    left, right = rem_ne_peak
    if abs(left) == 0.0 and abs(right) == 0.0:
        return psin_flat, ne_flat, ene_flat

    if len(ne_flat) == 0:
        return psin_flat, ne_flat, ene_flat

    i_peak = np.nanargmax(ne_flat)
    x_peak = psin_flat[i_peak]

    keep = ~((psin_flat >= x_peak - left) & (psin_flat <= x_peak + right))
    return psin_flat[keep], ne_flat[keep], ene_flat[keep]


# =============================================================================
# SHIFT DIAGNOSTICS
# =============================================================================
def compute_side_shift_diagnostics(r_unshifted_2d, r_shifted_2d):
    """
    Report HFS/LFS psi_N diagnostics after applying side-specific radial shifts.
    """
    ru = np.asarray(r_unshifted_2d).ravel()
    rs = np.asarray(r_shifted_2d).ravel()

    m = np.isfinite(ru) & np.isfinite(rs)
    ru = ru[m]
    rs = rs[m]
    dr = rs - ru

    mhfs = ru < 1.0
    mlfs = ru >= 1.0

    def _stats(mask):
        if np.sum(mask) == 0:
            return {
                "n": 0,
                "mean_dR": np.nan,
                "min_dR": np.nan,
                "max_dR": np.nan,
            }
        return {
            "n": int(np.sum(mask)),
            "mean_dR": float(np.nanmean(dr[mask])),
            "min_dR": float(np.nanmin(dr[mask])),
            "max_dR": float(np.nanmax(dr[mask])),
        }

    return {
        "hfs": _stats(mhfs),
        "lfs": _stats(mlfs),
    }


# =============================================================================
# MAIN CASE RUNNER
# =============================================================================
def run_lorenzo_style_case(case_dict, elm_dir="."):
    """
    Run the full Lorenzo-style workflow for one case: load data, select ELM phase, map profiles, fit, plot, and package results.
    """
    shot = int(case_dict["shot"])
    tr = tuple(case_dict["tr"])
    perc_elm = tuple(case_dict["perc_elm"])
    r_shift_hfs = float(case_dict.get("r_shift_hfs", 0.0))
    r_shift_lfs = float(case_dict.get("r_shift_lfs", 0.0))
    te_sep = float(case_dict.get("te_sep", 0.05))
    neoff = bool(case_dict.get("neoff", 0))
    teoff = bool(case_dict.get("teoff", 0))
    rem_ne_peak = case_dict.get("rem_ne_peak", [0.0, 0.0])
    xpf = float(case_dict.get("xpf", 0.8))
    ndeg = float(case_dict.get("ndeg", 5.0))
    equi = str(case_dict.get("equi", "EPM"))

    elm = load_elm_npz(shot, elm_dir=elm_dir)
    telm = np.asarray(elm["telm"]).ravel()

    ts = load_ts_data(
        shot,
        r_shift_hfs=r_shift_hfs,
        r_shift_lfs=r_shift_lfs,
    )
    time = ts["time"]

    i_elm = select_indices_in_elm_fraction(time, telm, perc_elm)
    i_tr = np.where((time >= tr[0]) & (time <= tr[1]))[0]
    i_keep = np.array(sorted(set(i_elm).intersection(set(i_tr))), dtype=int)

    if len(i_keep) == 0:
        raise RuntimeError("No TS time points survived ELM-cycle + absolute-time selection")

    time_sel = time[i_keep]
    te_sel = ts["te"][:, i_keep]
    ne_sel = ts["ne"][:, i_keep]
    ete_sel = ts["ete"][:, i_keep]
    ene_sel = ts["ene"][:, i_keep]
    r_sel = ts["r"][:, i_keep]
    r_sel_unshifted = ts["r_unshifted"][:, i_keep]
    z_sel = ts["z"][:, i_keep]

    shift_diag = compute_side_shift_diagnostics(r_sel_unshifted, r_sel)

    fluxdb = load_flux_grid(shot, equilibrium=equi)

    psin_sel_raw, _ = map_ts_to_psin(
        shot,
        r_sel,
        z_sel,
        time_sel,
        equilibrium=equi,
        use_idl_mean_z=USE_IDL_MEAN_Z,
        mapping_method=MAPPING_METHOD,
        fluxdb=fluxdb,
    )

    psin_sel_old_method_raw, _ = map_profile_to_psin_old_pointwise_z(
        shot,
        r_sel,
        z_sel,
        time_sel,
        equilibrium=equi,
        fluxdb=fluxdb,
    )

    mapping_debug = build_mapping_debug(
        time_sel,
        psin_sel_old_method_raw,
        psin_sel_raw,
        debug_time_index=DEBUG_MAPPING_TIME_INDEX,
    )

    te_flat0 = te_sel.ravel()
    ne_flat0 = ne_sel.ravel()
    ete_flat0 = ete_sel.ravel()
    ene_flat0 = ene_sel.ravel()
    psin_flat_raw0 = psin_sel_raw.ravel()
    r_flat0 = r_sel.ravel()

    # -------------------------------------------------------------------------
    # Base finite mask exactly as before for common TS cloud
    # -------------------------------------------------------------------------
    m = np.isfinite(te_flat0) & np.isfinite(ne_flat0) & np.isfinite(psin_flat_raw0)

    te_flat = te_flat0[m]
    ne_flat = ne_flat0[m]
    ete_flat = ete_flat0[m]
    ene_flat = ene_flat0[m]
    psin_flat_raw = psin_flat_raw0[m]
    r_flat = r_flat0[m]

    # -------------------------------------------------------------------------
    # ne edge cloud
    # -------------------------------------------------------------------------
    psin_ne_raw, ne_use, ene_use = apply_rem_ne_peak(psin_flat_raw, ne_flat, ene_flat, rem_ne_peak)

    m_ne = psin_ne_raw >= xpf
    ne_edge = ne_use[m_ne]
    ene_edge = ene_use[m_ne]
    psin_ne_edge_raw = psin_ne_raw[m_ne]

    # -------------------------------------------------------------------------
    # Te edge cloud, IDL-like:
    # use all finite Te points with psi >= xpf
    # do NOT reject based on measured ete
    # -------------------------------------------------------------------------
    m_te_idl = np.isfinite(te_flat0) & np.isfinite(psin_flat_raw0)
    te_flat_idl = te_flat0[m_te_idl]
    psin_flat_raw_idl = psin_flat_raw0[m_te_idl]
    r_flat_idl = r_flat0[m_te_idl]
    ete_flat_idl = ete_flat0[m_te_idl]

    m_te_edge = psin_flat_raw_idl >= xpf
    te_edge = te_flat_idl[m_te_edge]
    psin_te_raw = psin_flat_raw_idl[m_te_edge]
    r_te_edge = r_flat_idl[m_te_edge]
    ete_edge = ete_flat_idl[m_te_edge]

    if len(psin_te_raw) < 6:
        raise RuntimeError("Too few Te edge points after xpf selection")
    if len(psin_ne_edge_raw) < 6:
        raise RuntimeError("Too few ne edge points after xpf selection")

    # -------------------------------------------------------------------------
    # First fits on raw psi_N
    # -------------------------------------------------------------------------
    te_fit_raw = fit_mtanh_profile_te_idl_style(
        psin_te_raw,
        te_edge,
        use_offset=teoff,
    )

    ne_fit_raw = fit_mtanh_profile(
        psin_ne_edge_raw,
        ne_edge,
        yerr=ene_edge if np.any(np.isfinite(ene_edge)) else None,
        use_offset=neoff,
        quantity="ne",
        use_bounds=True,
    )

    xfit_edge_raw = np.linspace(0.7, 1.1, 1001)
    y_te_raw = eval_fit_on_grid(te_fit_raw, xfit_edge_raw)

    # -------------------------------------------------------------------------
    # IDL-like Te psi shift: shift after fit, do not refit Te
    # -------------------------------------------------------------------------
    psi_shift = 0.0
    if APPLY_GLOBAL_PSI_SHIFT and np.isfinite(te_sep):
        i_sep = np.argmin(np.abs(y_te_raw - te_sep))
        psi_shift = 1.0 - xfit_edge_raw[i_sep]

    # Shift points
    psin_sel = psin_sel_raw + psi_shift
    psin_flat = psin_flat_raw + psi_shift
    psin_te = psin_te_raw + psi_shift
    psin_ne = psin_ne_raw + psi_shift
    psin_ne_edge = psin_ne_edge_raw + psi_shift

    # -------------------------------------------------------------------------
    # Final Te fit:
    # IDL-like default = do not refit after shift
    # -------------------------------------------------------------------------
    if TE_REFIT_AFTER_SHIFT:
        te_fit = fit_mtanh_profile_te_idl_style(
            psin_te,
            te_edge,
            use_offset=teoff,
        )
    else:
        te_fit = te_fit_raw

    # ne can keep previous behavior
    ne_fit = fit_mtanh_profile(
        psin_ne_edge,
        ne_edge,
        yerr=ene_edge if np.any(np.isfinite(ene_edge)) else None,
        use_offset=neoff,
        quantity="ne",
        use_bounds=True,
    )

    # -------------------------------------------------------------------------
    # Evaluate fits on shifted grid, like IDL plotting/output
    # -------------------------------------------------------------------------
    xfit_edge = xfit_edge_raw + psi_shift

    te_par = extract_pedestal_params(te_fit, xfit_edge_raw)
    ne_par = extract_pedestal_params(ne_fit, xfit_edge)

    tefit_edge_raw = te_par["yfit"]
    tefit_edge = tefit_edge_raw.copy()

    nefit_edge = ne_par["yfit"]

    gradt = -np.gradient(tefit_edge, xfit_edge)
    gradn = -np.gradient(nefit_edge, xfit_edge)

    # Update Te pedestal position with horizontal shift, like IDL
    te_par["ped_pos"] = te_par["ped_pos"] + psi_shift
    te_par["xgrid"] = xfit_edge
    te_par["yfit"] = tefit_edge
    te_par["dy"] = np.gradient(tefit_edge, xfit_edge)
    te_par["grad_idl"] = gradt
    te_par["max_grad"] = np.nanmax(gradt)

    # ne already fit on shifted coordinates, keep as is
    ne_par["grad_idl"] = gradn
    ne_par["max_grad"] = np.nanmax(gradn)

    # -------------------------------------------------------------------------
    # Core fits
    # -------------------------------------------------------------------------
    xfit_core = np.linspace(0.0, xpf, 1001)

    m_core_ne = np.isfinite(psin_flat) & np.isfinite(ne_flat) & (psin_flat <= xpf)
    fitn_core = fit_poly_with_constraints(
        psin_flat[m_core_ne],
        ne_flat[m_core_ne],
        x0=xpf,
        y0=float(np.interp(xpf, xfit_edge, nefit_edge)),
        dy0=float(-np.interp(xpf, xfit_edge, gradn)),
        degree=int(round(ndeg)),
        yerr=ene_flat[m_core_ne] if np.any(np.isfinite(ene_flat[m_core_ne])) else None,
    )
    necorefit = eval_poly_core_fit(fitn_core, xfit_core)

    # Te core: use IDL-like weights sqrt(Te), not measured ete
    te_core_sigma = None
    m_core_te = np.isfinite(psin_flat) & np.isfinite(te_flat) & (psin_flat <= xpf)
    if np.sum(m_core_te) > 0:
        te_core_sigma = build_idl_style_te_sigma(te_flat[m_core_te])

    fit_te_core = fit_poly_with_constraints(
        psin_flat[m_core_te],
        te_flat[m_core_te],
        x0=xpf,
        y0=float(np.interp(xpf, xfit_edge, tefit_edge)),
        dy0=float(-np.interp(xpf, xfit_edge, gradt)),
        degree=int(round(ndeg)),
        yerr=te_core_sigma,
    )
    tecorefit = eval_poly_core_fit(fit_te_core, xfit_core)

    inde = np.where(xfit_edge > xpf)[0]
    xfit_full = np.concatenate([xfit_core, xfit_edge[inde]])
    tefit_full = np.concatenate([tecorefit, tefit_edge[inde]])
    nefit_full = np.concatenate([necorefit, nefit_edge[inde]])

    pt = te_par["ped_pos"]
    wt = te_par["ped_width"]
    ht = te_par["ped_height"]
    te_sol = te_par["sol_offset"]

    pn = ne_par["ped_pos"]
    wn = ne_par["ped_width"]
    hn = ne_par["ped_height"]
    ne_sol = ne_par["sol_offset"]

    p0_ne_essive = np.array([hn, ne_sol, 0.0, pn, wn / 4.0, 0.8, 10.0, 0.0, 0.0], dtype=float)
    p0_te_essive = np.array([ht, te_sol, 0.0, pt, wt / 4.0, 0.7, 1.0, -0.7, 0.0], dtype=float)

    eyyin_ne = np.ones_like(nefit_full)
    fitn_essive, _ = fit_profile_fit_essive(xfit_full, nefit_full, eyyin_ne, p0_ne_essive)
    nefit_essive = profile_fit_essive_py(xfit_full, fitn_essive)

    eyyin_te = np.sqrt(np.maximum(tefit_full, 1e-6))
    fitt_essive, _ = fit_profile_fit_essive(xfit_full, tefit_full, eyyin_te, p0_te_essive)
    tefit_essive = profile_fit_essive_py(xfit_full, fitt_essive)

    # -------------------------------------------------------------------------
    # Ti
    # -------------------------------------------------------------------------
    ti_status = "missing"
    ti_sel = np.array([])
    dti_sel = np.array([])
    psin_ti_sel = np.array([])
    ti_fit = None
    ti_yfit = np.full_like(xfit_full, np.nan)
    ti_deg = np.nan
    ti_time_sel = np.array([])

    psin_ti_raw_2d = np.array([])
    psin_ti_shifted_2d = np.array([])
    psin_ti_sel_raw = np.array([])
    psin_ti_old_method_raw_2d = np.array([])

    try:
        cx = load_cx_ti_data(shot)
        ti_time = cx["time"]

        i_tr_ti = np.where((ti_time >= tr[0]) & (ti_time <= tr[1]))[0]
        tti = ti_time[i_tr_ti]
        ti_rt = cx["ti"][:, i_tr_ti]
        dti_rt = cx["dti"][:, i_tr_ti]
        r_ti = cx["r"][:, i_tr_ti]
        z_ti = cx["z"][:, i_tr_ti]

        i_elm_ti = select_indices_in_elm_fraction(tti, telm, perc_elm)
        if len(i_elm_ti) == 0:
            raise RuntimeError("No CX Ti time points survived ELM-cycle + absolute-time selection")

        ti_time_sel = tti[i_elm_ti]
        ti_rt = ti_rt[:, i_elm_ti]
        dti_rt = dti_rt[:, i_elm_ti]
        r_ti = r_ti[:, i_elm_ti]
        z_ti = z_ti[:, i_elm_ti]

        psin_ti_raw_2d, _ = map_ti_to_psin(
            shot,
            r_ti,
            z_ti,
            ti_time_sel,
            equilibrium=equi,
            use_idl_mean_z=USE_IDL_MEAN_Z,
            mapping_method=MAPPING_METHOD,
            fluxdb=fluxdb,
        )
        psin_ti_shifted_2d = psin_ti_raw_2d + psi_shift

        psin_ti_old_method_raw_2d, _ = map_profile_to_psin_old_pointwise_z(
            shot,
            r_ti,
            z_ti,
            ti_time_sel,
            equilibrium=equi,
            fluxdb=fluxdb,
        )

        ti_sel = ti_rt.ravel()
        dti_sel = dti_rt.ravel()
        psin_ti_sel_raw = psin_ti_raw_2d.ravel()
        psin_ti_sel = psin_ti_shifted_2d.ravel()

        mti = np.isfinite(ti_sel) & np.isfinite(psin_ti_sel)
        if np.any(np.isfinite(dti_sel)):
            mti &= np.isfinite(dti_sel) & (dti_sel > 0)

        ti_sel = ti_sel[mti]
        dti_sel = dti_sel[mti]
        psin_ti_sel_raw = psin_ti_sel_raw[mti]
        psin_ti_sel = psin_ti_sel[mti]

        print("Ti selected times:", ti_time_sel)
        print("Ti min/max [keV]:", np.nanmin(ti_sel), np.nanmax(ti_sel))
        print("Ti raw psin min/max:", np.nanmin(psin_ti_sel_raw), np.nanmax(psin_ti_sel_raw))
        print("Ti shifted psin min/max:", np.nanmin(psin_ti_sel), np.nanmax(psin_ti_sel))
        print("Number of Ti points:", len(ti_sel))

        ti_fit = fit_weighted_poly(
            psin_ti_sel,
            ti_sel,
            yerr=dti_sel if len(dti_sel) else None,
            deg=min(5, int(max(1, round(ndeg)))),
        )
        ti_yfit = eval_poly_fit(ti_fit, xfit_full)
        ti_deg = ti_fit["deg"]
        ti_status = "ok"

    except Exception as err:
        ti_status = f"fallback: {err}"
        ti_fit = None
        ti_yfit = np.full_like(xfit_full, np.nan)

    return {
        "shot": shot,
        "tr": np.asarray(tr),
        "perc_elm": np.asarray(perc_elm),
        "r_shift_hfs": r_shift_hfs,
        "r_shift_lfs": r_shift_lfs,
        "te_sep": te_sep,
        "neoff": int(neoff),
        "teoff": int(teoff),
        "rem_ne_peak": np.asarray(rem_ne_peak),
        "xpf": xpf,
        "ndeg": ndeg,
        "equi": equi,
        "psi_shift": psi_shift,
        "telm": telm,

        "use_idl_mean_z": bool(USE_IDL_MEAN_Z),
        "mapping_method": str(MAPPING_METHOD),
        "apply_global_psi_shift": bool(APPLY_GLOBAL_PSI_SHIFT),

        "te_use_idl_sqrt_weights": int(TE_USE_IDL_SQRT_WEIGHTS),
        "te_use_bounds": int(TE_USE_BOUNDS),
        "te_refit_after_shift": int(TE_REFIT_AFTER_SHIFT),

        "time_sel": time_sel,
        "r_sel": r_sel,
        "r_sel_unshifted": r_sel_unshifted,
        "z_sel": z_sel,

        "shift_diag": shift_diag,

        "psin_sel_old_method_raw": psin_sel_old_method_raw,
        "psin_sel_raw": psin_sel_raw,
        "psin_sel": psin_sel,

        "te_sel": te_sel,
        "ne_sel": ne_sel,
        "ete_sel": ete_sel,
        "ene_sel": ene_sel,

        "psin_flat_raw": psin_flat_raw,
        "psin_flat": psin_flat,

        "psin_te_raw": psin_te_raw,
        "psin_te": psin_te,
        "psin_ne_raw": psin_ne_raw,
        "psin_ne": psin_ne,
        "psin_ne_edge_raw": psin_ne_edge_raw,
        "psin_ne_edge": psin_ne_edge,

        "te_flat": te_flat,
        "ne_flat": ne_flat,
        "te_edge": te_edge,
        "ete_edge": ete_edge,
        "r_te_edge": r_te_edge,

        "te_popt": te_fit["popt"],
        "ne_popt": ne_fit["popt"],
        "te_ped": te_par,
        "ne_ped": ne_par,

        "xfit_edge": xfit_edge,
        "xfit_full": xfit_full,
        "tefit_full": tefit_full,
        "nefit_full": nefit_full,
        "tefit_essive": tefit_essive,
        "nefit_essive": nefit_essive,
        "fitt_essive": fitt_essive,
        "fitn_essive": fitn_essive,

        "ti_status": ti_status,
        "ti_time_sel": ti_time_sel,
        "ti_fit_x": ti_fit["x"] if ti_fit is not None else np.array([]),
        "ti_fit_y": ti_fit["y"] if ti_fit is not None else np.array([]),
        "ti_fit_coeff": ti_fit["coeff"] if ti_fit is not None else np.array([]),
        "ti_fit_deg": ti_deg,
        "ti_yfit": ti_yfit,
        "psin_ti_old_method_raw_2d": psin_ti_old_method_raw_2d,
        "psin_ti_raw_2d": psin_ti_raw_2d,
        "psin_ti_shifted_2d": psin_ti_shifted_2d,
        "psin_ti_sel_raw": psin_ti_sel_raw,
        "psin_ti_sel": psin_ti_sel,

        "mapping_debug": mapping_debug,

        "fluxdb_meta": {
            "t_flux": fluxdb["t_flux"],
            "r_grid": fluxdb["r_grid"],
            "z_grid": fluxdb["z_grid"],
        },
    }


# =============================================================================
# PLOTTING
# =============================================================================
def plot_lorenzo_style_result_v13(out):
    """
    Create the multi-panel diagnostic figure for one fitted case.
    """
    def _split_hfs_lfs(psin2d, r2d, y2d):
        ps = np.asarray(psin2d).ravel()
        rr = np.asarray(r2d).ravel()
        yy = np.asarray(y2d).ravel()
        m = np.isfinite(ps) & np.isfinite(rr) & np.isfinite(yy)
        ps, rr, yy = ps[m], rr[m], yy[m]

        hfs = rr <= 1.0
        lfs = rr > 1.0
        return (ps[hfs], yy[hfs]), (ps[lfs], yy[lfs])

    def _safe_interp(x0, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        m = np.isfinite(x) & np.isfinite(y)
        if np.sum(m) < 2:
            return np.nan
        return float(np.interp(x0, x[m], y[m]))

    shot = out["shot"]
    xfit = out["xfit_full"]

    tefit = out["tefit_full"]
    nefit = out["nefit_full"]

    pt = out["te_ped"]["ped_pos"]
    wt = out["te_ped"]["ped_width"]
    ht = out["te_ped"]["ped_height"]

    pn = out["ne_ped"]["ped_pos"]
    wn = out["ne_ped"]["ped_width"]
    hn = out["ne_ped"]["ped_height"]

    tesep_val = _safe_interp(1.0, xfit, tefit)
    nesep_val = _safe_interp(1.0, xfit, nefit)

    psi2d = out["psin_sel_raw"] if PLOT_RAW_MAPPING else out["psin_sel"]
    r2d = out["r_sel"]
    te2d = out["te_sel"]
    ne2d = out["ne_sel"]

    (psi_te_hfs, te_hfs), (psi_te_lfs, te_lfs) = _split_hfs_lfs(psi2d, r2d, te2d)
    (psi_ne_hfs, ne_hfs), (psi_ne_lfs, ne_lfs) = _split_hfs_lfs(psi2d, r2d, ne2d)

    ti_x = np.asarray(out.get("ti_fit_x", []))
    ti_y = np.asarray(out.get("ti_fit_y", []))
    tifit = np.asarray(out.get("ti_yfit", np.full_like(xfit, np.nan)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(direction="in", top=True, right=True, length=6, width=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    xr = (0.80, 1.10)
    yrt = (0.00, 0.51)
    yrn = (0.00, 6.00)

    grey = "0.75"
    red = "red"
    black = "black"
    green = "green"
    blue = "dodgerblue"
    cyan_dark = "deepskyblue"

    title_suffix = "raw" if PLOT_RAW_MAPPING else "shifted"

    # Te
    ax = axes[0]
    ax.set_title(f"{shot} ({title_suffix})", fontsize=18)
    ax.set_xlim(*xr)
    ax.set_ylim(*yrt)

    ax.axvline(1.0, color="k", linestyle="--", linewidth=1)
    ax.axhline(tesep_val, color=grey, linestyle=(0, (5, 5)), linewidth=1)
    ax.axhline(ht, color=red, linestyle=(0, (5, 5)), linewidth=1.2, label="Te pedestal height")
    ax.axvline(pt, color=red, linestyle=(0, (5, 5)), linewidth=1.2, label="Te pedestal position")
    ax.axvline(pt - wt / 2.0, color=grey, linestyle=(0, (5, 5)), linewidth=1)
    ax.axvline(pt + wt / 2.0, color=grey, linestyle=(0, (5, 5)), linewidth=1)

    ax.plot(psi_te_hfs, te_hfs, linestyle="None", marker="s", markersize=4, color=red, label="TS HFS")
    ax.plot(psi_te_lfs, te_lfs, linestyle="None", marker="s", markersize=4, color=green, label="TS LFS")
    ax.plot(xfit, tefit, color=black, linewidth=3, label="Te fit")

    ax.plot([1.0], [tesep_val], marker="o", markersize=10, markerfacecolor="none", markeredgecolor=red)
    ax.plot([1.0], [tesep_val], marker="o", markersize=6, color=black, label="Separatrix point")

    if np.size(tifit) == np.size(xfit) and np.any(np.isfinite(tifit)):
        ax.plot(xfit, tifit, color=cyan_dark, linewidth=1.5)

    ax.set_xlabel(r"$\psi_N$", fontsize=16)
    ax.set_ylabel(r"$T_e$ (keV)", fontsize=16)
    ax.legend(fontsize=9, loc="best")

    # ne
    ax = axes[1]
    ax.set_xlim(*xr)
    ax.set_ylim(*yrn)

    ax.axvline(1.0, color="k", linestyle="--", linewidth=1)
    ax.axhline(nesep_val, color=grey, linestyle=(0, (5, 5)), linewidth=1)
    ax.axhline(hn, color=red, linestyle=(0, (5, 5)), linewidth=1.2, label="ne pedestal height")
    ax.axvline(pn, color=red, linestyle=(0, (5, 5)), linewidth=1.2, label="ne pedestal position")
    ax.axvline(pn - wn / 2.0, color=grey, linestyle=(0, (5, 5)), linewidth=1)
    ax.axvline(pn + wn / 2.0, color=grey, linestyle=(0, (5, 5)), linewidth=1)

    ax.plot(psi_ne_hfs, ne_hfs, linestyle="None", marker="s", markersize=4, color=red, label="TS HFS")
    ax.plot(psi_ne_lfs, ne_lfs, linestyle="None", marker="s", markersize=4, color=green, label="TS LFS")
    ax.plot(xfit, nefit, color=black, linewidth=3, label="ne fit")

    ax.plot([1.0], [nesep_val], marker="o", markersize=10, markerfacecolor="none", markeredgecolor=red)
    ax.plot([1.0], [nesep_val], marker="o", markersize=6, color=black, label="Separatrix point")

    ax.set_xlabel(r"$\psi_N$", fontsize=16)
    ax.set_ylabel(r"$n_e$  $10^{19}$  $(m^{-3})$", fontsize=16)
    ax.legend(fontsize=9, loc="best")

    # Ti
    ax = axes[2]
    ax.set_xlim(*xr)
    ax.set_ylim(*yrt)

    ax.axvline(1.0, color="k", linestyle="--", linewidth=1)

    if len(ti_x) > 0:
        ax.plot(ti_x, ti_y, linestyle="None", marker="s", markersize=5, color=blue, label="CX Ti points")

    if np.size(tifit) == np.size(xfit) and np.any(np.isfinite(tifit)):
        ax.plot(xfit, tifit, color=black, linewidth=3, label="Ti fit")
        ax.plot(xfit, tifit, color=red, linewidth=1)

    ax.set_xlabel(r"$\psi_N$", fontsize=16)
    ax.set_ylabel(r"$T_i$ (keV)", fontsize=16)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    return fig, axes


def save_python_mapped_te_points_csv(out, outdir="."):
    """
    Export mapped Te points to CSV so the fit input cloud can be inspected outside Python.
    """
    import pandas as pd

    os.makedirs(outdir, exist_ok=True)

    rows = []

    for mapping_stage, psin_key in [
        ("old_pointwise_z_raw", "psin_sel_old_method_raw"),
        ("raw", "psin_sel_raw"),
        ("shifted", "psin_sel"),
    ]:
        psin2d = np.asarray(out[psin_key])
        te2d = np.asarray(out["te_sel"])
        r2d = np.asarray(out["r_sel"])
        z2d = np.asarray(out["z_sel"])
        time_sel = np.asarray(out["time_sel"])

        nch, nt = psin2d.shape

        for k in range(nt):
            z_mean_k = float(np.nanmean(z2d[:, k]))
            for i in range(nch):
                ps = psin2d[i, k]
                te = te2d[i, k]
                rr = r2d[i, k]
                zz = z2d[i, k]

                if np.isfinite(ps) and np.isfinite(te) and np.isfinite(rr):
                    rows.append({
                        "shot": out["shot"],
                        "mapping_stage": mapping_stage,
                        "mapping_method": out["mapping_method"],
                        "use_idl_mean_z": int(out["use_idl_mean_z"]),
                        "apply_global_psi_shift": int(out["apply_global_psi_shift"]),
                        "r_shift_hfs": float(out["r_shift_hfs"]),
                        "r_shift_lfs": float(out["r_shift_lfs"]),
                        "psi_shift": float(out["psi_shift"]),
                        "time_s": float(time_sel[k]),
                        "channel_index": i,
                        "R_m": float(rr),
                        "Z_m": float(zz) if np.isfinite(zz) else np.nan,
                        "Z_mean_slice_m": z_mean_k,
                        "psin": float(ps),
                        "Te_keV": float(te),
                        "side": "HFS" if rr <= 1.0 else "LFS"
                    })

    df = pd.DataFrame(rows)
    path = os.path.join(outdir, f"python_mapped_te_points_v13_{out['shot']}.csv")
    df.to_csv(path, index=False)
    print(f"Saved Python mapped Te points: {path}")
    return df, path


# =============================================================================
# SAVE / SUMMARY
# =============================================================================
def save_lorenzo_style_output_v13(out, outdir="."):
    """
    Save all headline fit outputs and selected intermediate arrays to a compressed npz file.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"TS_fit_lorenzo_style_v13_{out['shot']} .npz".replace(" ", ""))

    np.savez(
        path,
        shot=out["shot"],
        tr=out["tr"],
        perc_elm=out["perc_elm"],
        r_shift_hfs=out["r_shift_hfs"],
        r_shift_lfs=out["r_shift_lfs"],
        te_sep=out["te_sep"],
        neoff=out["neoff"],
        teoff=out["teoff"],
        rem_ne_peak=out["rem_ne_peak"],
        xpf=out["xpf"],
        ndeg=out["ndeg"],
        equi=out["equi"],
        psi_shift=out["psi_shift"],
        telm=out["telm"],
        use_idl_mean_z=int(out["use_idl_mean_z"]),
        mapping_method=out["mapping_method"],
        apply_global_psi_shift=int(out["apply_global_psi_shift"]),
        te_use_idl_sqrt_weights=int(out["te_use_idl_sqrt_weights"]),
        te_use_bounds=int(out["te_use_bounds"]),
        te_refit_after_shift=int(out["te_refit_after_shift"]),

        time_sel=out["time_sel"],
        r_sel=out["r_sel"],
        r_sel_unshifted=out["r_sel_unshifted"],
        psin_sel_old_method_raw=out["psin_sel_old_method_raw"],
        psin_sel_raw=out["psin_sel_raw"],
        psin_sel=out["psin_sel"],
        te_sel=out["te_sel"],
        ne_sel=out["ne_sel"],

        psin_flat_raw=out["psin_flat_raw"],
        psin_flat=out["psin_flat"],

        te_popt=out["te_popt"],
        ne_popt=out["ne_popt"],
        te_ped_pos=out["te_ped"]["ped_pos"],
        te_ped_width=out["te_ped"]["ped_width"],
        te_ped_height=out["te_ped"]["ped_height"],
        te_ped_max_grad=out["te_ped"]["max_grad"],
        ne_ped_pos=out["ne_ped"]["ped_pos"],
        ne_ped_width=out["ne_ped"]["ped_width"],
        ne_ped_height=out["ne_ped"]["ped_height"],
        ne_ped_max_grad=out["ne_ped"]["max_grad"],

        xfit_full=out["xfit_full"],
        tefit_full=out["tefit_full"],
        nefit_full=out["nefit_full"],
        tefit_essive=out["tefit_essive"],
        nefit_essive=out["nefit_essive"],
        fitt_essive=out["fitt_essive"],
        fitn_essive=out["fitn_essive"],

        ti_status=out["ti_status"],
        ti_time_sel=out["ti_time_sel"],
        ti_fit_x=out["ti_fit_x"],
        ti_fit_y=out["ti_fit_y"],
        ti_fit_coeff=out["ti_fit_coeff"],
        ti_fit_deg=out["ti_fit_deg"],
        ti_yfit=out["ti_yfit"],
        psin_ti_old_method_raw_2d=out["psin_ti_old_method_raw_2d"],
        psin_ti_raw_2d=out["psin_ti_raw_2d"],
        psin_ti_shifted_2d=out["psin_ti_shifted_2d"],
        psin_ti_sel_raw=out["psin_ti_sel_raw"],
        psin_ti_sel=out["psin_ti_sel"],

        mapping_debug_time_s=out["mapping_debug"]["time_s"],
        mapping_debug_time_index=out["mapping_debug"]["debug_time_index"],
        mapping_debug_old_psin_slice=out["mapping_debug"]["old_psin_slice"],
        mapping_debug_new_psin_slice=out["mapping_debug"]["new_psin_slice"],
        mapping_debug_abs_diff_slice=out["mapping_debug"]["abs_diff_slice"],
        mapping_debug_max_abs_diff=out["mapping_debug"]["max_abs_diff"],
        mapping_debug_mean_abs_diff=out["mapping_debug"]["mean_abs_diff"],

        shift_diag_hfs_n=out["shift_diag"]["hfs"]["n"],
        shift_diag_hfs_mean_dR=out["shift_diag"]["hfs"]["mean_dR"],
        shift_diag_hfs_min_dR=out["shift_diag"]["hfs"]["min_dR"],
        shift_diag_hfs_max_dR=out["shift_diag"]["hfs"]["max_dR"],
        shift_diag_lfs_n=out["shift_diag"]["lfs"]["n"],
        shift_diag_lfs_mean_dR=out["shift_diag"]["lfs"]["mean_dR"],
        shift_diag_lfs_min_dR=out["shift_diag"]["lfs"]["min_dR"],
        shift_diag_lfs_max_dR=out["shift_diag"]["lfs"]["max_dR"],
    )
    print(f"Saved: {path}")
    return path


def build_summary_row_v13(out):
    """
    Build one row of scalar summary metrics for the v13 profile-fit batch output.
    """
    return {
        "shot": out["shot"],
        "tr0": float(out["tr"][0]),
        "tr1": float(out["tr"][1]),
        "perc_elm_0": float(out["perc_elm"][0]),
        "perc_elm_1": float(out["perc_elm"][1]),
        "r_shift_hfs": float(out["r_shift_hfs"]),
        "r_shift_lfs": float(out["r_shift_lfs"]),
        "te_sep": float(out["te_sep"]),
        "neoff": int(out["neoff"]),
        "teoff": int(out["teoff"]),
        "xpf": float(out["xpf"]),
        "ndeg": float(out["ndeg"]),
        "equi": out["equi"],
        "use_idl_mean_z": int(out["use_idl_mean_z"]),
        "mapping_method": out["mapping_method"],
        "apply_global_psi_shift": int(out["apply_global_psi_shift"]),
        "te_use_idl_sqrt_weights": int(out["te_use_idl_sqrt_weights"]),
        "te_use_bounds": int(out["te_use_bounds"]),
        "te_refit_after_shift": int(out["te_refit_after_shift"]),
        "psi_shift": float(out["psi_shift"]),
        "te_ped_pos": float(out["te_ped"]["ped_pos"]),
        "te_ped_width": float(out["te_ped"]["ped_width"]),
        "te_ped_height_keV": float(out["te_ped"]["ped_height"]),
        "te_max_grad_keV_per_psin": float(out["te_ped"]["max_grad"]),
        "ne_ped_pos": float(out["ne_ped"]["ped_pos"]),
        "ne_ped_width": float(out["ne_ped"]["ped_width"]),
        "ne_ped_height_1e19m3": float(out["ne_ped"]["ped_height"]),
        "mapping_debug_max_abs_diff": float(out["mapping_debug"]["max_abs_diff"]),
        "mapping_debug_mean_abs_diff": float(out["mapping_debug"]["mean_abs_diff"]),
        "shift_diag_hfs_mean_dR": float(out["shift_diag"]["hfs"]["mean_dR"]),
        "shift_diag_lfs_mean_dR": float(out["shift_diag"]["lfs"]["mean_dR"]),
        "ti_status": out["ti_status"],
        "ti_fit_deg": out["ti_fit_deg"],
        "status": "ok",
        "error_message": "",
    }


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    rows = []

    if SAVE_PLOTS:
        os.makedirs(PLOT_DIR, exist_ok=True)

    for case in in_list:
        shot = case["shot"]
        print(f"\nProcessing shot {shot} ...")

        try:
            out = run_lorenzo_style_case(case, elm_dir=ELM_DIR)

            print("\n" + "=" * 60)
            print(f"Shot {out['shot']}")
            print("=" * 60)

            print("INPUTS")
            print(f"  tr                     = [{out['tr'][0]:.3f}, {out['tr'][1]:.3f}] s")
            print(f"  perc_elm               = [{out['perc_elm'][0]:.2f}, {out['perc_elm'][1]:.2f}]")
            print(f"  equilibrium            = {out['equi']}")
            print(f"  r_shift_hfs            = {out['r_shift_hfs']:.6f} m")
            print(f"  r_shift_lfs            = {out['r_shift_lfs']:.6f} m")
            print(f"  xpf                    = {out['xpf']:.3f}")
            print(f"  psi_shift              = {out['psi_shift']:.5f}")
            print(f"  USE_IDL_MEAN_Z         = {out['use_idl_mean_z']}")
            print(f"  MAPPING_METHOD         = {out['mapping_method']}")
            print(f"  APPLY_GLOBAL_PSI_SHIFT = {out['apply_global_psi_shift']}")
            print(f"  PLOT_RAW_MAPPING       = {PLOT_RAW_MAPPING}")

            print("\nTe IDL-LIKE OPTIONS")
            print(f"  TE_USE_IDL_SQRT_WEIGHTS = {out['te_use_idl_sqrt_weights']}")
            print(f"  TE_USE_BOUNDS           = {out['te_use_bounds']}")
            print(f"  TE_REFIT_AFTER_SHIFT    = {out['te_refit_after_shift']}")

            print("\nR-SHIFT DIAGNOSTICS")
            print(f"  HFS n                  = {out['shift_diag']['hfs']['n']}")
            print(f"  HFS mean dR            = {out['shift_diag']['hfs']['mean_dR']:.6e}")
            print(f"  HFS min/max dR         = {out['shift_diag']['hfs']['min_dR']:.6e}, {out['shift_diag']['hfs']['max_dR']:.6e}")
            print(f"  LFS n                  = {out['shift_diag']['lfs']['n']}")
            print(f"  LFS mean dR            = {out['shift_diag']['lfs']['mean_dR']:.6e}")
            print(f"  LFS min/max dR         = {out['shift_diag']['lfs']['min_dR']:.6e}, {out['shift_diag']['lfs']['max_dR']:.6e}")

            print("\nTS MAPPING")
            print(f"  old raw psi_N min      = {np.nanmin(out['psin_sel_old_method_raw']):.6f}")
            print(f"  old raw psi_N max      = {np.nanmax(out['psin_sel_old_method_raw']):.6f}")
            print(f"  new raw psi_N min      = {np.nanmin(out['psin_sel_raw']):.6f}")
            print(f"  new raw psi_N max      = {np.nanmax(out['psin_sel_raw']):.6f}")
            print(f"  shifted psi_N min      = {np.nanmin(out['psin_sel']):.6f}")
            print(f"  shifted psi_N max      = {np.nanmax(out['psin_sel']):.6f}")

            dbg = out["mapping_debug"]
            print("\nMAPPING DEBUG")
            print(f"  debug time index       = {dbg['debug_time_index']}")
            print(f"  debug time [s]         = {dbg['time_s']:.6f}")
            print(f"  max abs diff           = {dbg['max_abs_diff']:.6e}")
            print(f"  mean abs diff          = {dbg['mean_abs_diff']:.6e}")

            print("\nTe PEDESTAL")
            print(f"  Teped (ht)             = {out['te_ped']['ped_height']:.6f} keV")
            print(f"  pTe   (pt)             = {out['te_ped']['ped_pos']:.6f} psi_N")
            print(f"  wTe   (wt)             = {out['te_ped']['ped_width']:.6f} psi_N")
            print(f"  Te_sep                 = {np.interp(1.0, out['xfit_full'], out['tefit_full']):.6f} keV")
            print(f"  Te_SOL                 = {out['te_ped'].get('sol_offset', np.nan):.6f} keV")
            print(f"  max_grad_Te            = {out['te_ped']['max_grad']:.6f} keV/psi_N")

            print("\nne PEDESTAL")
            print(f"  neped (hn)             = {out['ne_ped']['ped_height']:.6f} x1e19 m^-3")
            print(f"  pne   (pn)             = {out['ne_ped']['ped_pos']:.6f} psi_N")
            print(f"  wne   (wn)             = {out['ne_ped']['ped_width']:.6f} psi_N")
            print(f"  ne_sep                 = {np.interp(1.0, out['xfit_full'], out['nefit_full']):.6f} x1e19 m^-3")
            print(f"  ne_SOL                 = {out['ne_ped'].get('sol_offset', np.nan):.6f} x1e19 m^-3")
            print(f"  max_grad_ne            = {out['ne_ped']['max_grad']:.6f} (1e19 m^-3)/psi_N")

            print("\nDERIVED")
            print(f"  pne - pTe              = {out['ne_ped']['ped_pos'] - out['te_ped']['ped_pos']:.6f} psi_N")
            print(f"  ne_sep / neped         = {np.interp(1.0, out['xfit_full'], out['nefit_full']) / out['ne_ped']['ped_height']:.6f}")

            print("\nTi")
            print(f"  Ti status              = {out['ti_status']}")
            if "ti_yfit" in out and np.any(np.isfinite(out["ti_yfit"])):
                print(f"  Ti(psi_N=1)            = {np.interp(1.0, out['xfit_full'], out['ti_yfit']):.6f} keV")

            fig, _ = plot_lorenzo_style_result_v13(out)

            df_pts, pts_path = save_python_mapped_te_points_csv(out, outdir=OUTDIR)

            if SAVE_PLOTS:
                plot_path = os.path.join(PLOT_DIR, f"lorenzo_style_fit_v13_{shot}.png")
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                print(f"Saved plot: {plot_path}")

            if SAVE_OUTPUT:
                save_lorenzo_style_output_v13(out, OUTDIR)

            rows.append(build_summary_row_v13(out))

        except Exception as err:
            print(f"FAILED on shot {shot}: {err}")
            rows.append({
                "shot": shot,
                "status": "failed",
                "error_message": str(err),
            })

    plt.show()

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        print("\nSummary:")
        print(df.to_string(index=False))
        csv_path = os.path.abspath("lorenzo_style_summary_v13.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary CSV: {csv_path}")
    except Exception as err:
        print(f"Could not build/save summary DataFrame: {err}")
