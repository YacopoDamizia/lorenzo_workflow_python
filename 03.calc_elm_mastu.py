# -*- coding: utf-8 -*-
"""
Python version of Lorenzo's calc_elm_mastu.pro
FREIA / Spyder ready

What it does:
- loads /XIM/DA/HM10/T
- smooths the signal with a moving average
- subtracts background
- applies threshold in a selected time window
- groups threshold crossings into ELM events using a minimum time gap
- picks one ELM time per event
- makes:
    1) Lorenzo-style 4-panel diagnostic plot
    2) overlay plot with raw signal, red background, and detected ELM peaks
- optionally saves the result for later Python pedestal fitting

Notes:
- This is intentionally close to the IDL logic, but not byte-for-byte identical.
- Saved format is .npz, not IDL .sav.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pyuda import Client

client = Client()


# =============================================================================
# USER SETTINGS
# =============================================================================
SHOT = 49107

SIGNAL = "/XIM/DA/HM10/T"
TSEL = (0.2, 0.9)       # IDL default
THR = 0.011            # IDL default
SMOOTH_PTS = 1000       # IDL default
MIN_GAP_S = 0.002       # IDL default

# Event time choice:
# "first" = first threshold crossing in each burst
# "max"   = maximum residual in each burst
# "last"  = last threshold crossing in each burst
EVENT_MODE = "max"

SAVE_ELM_TIMES = True
OUTDIR = "."


# =============================================================================
# HELPERS
# =============================================================================
def extract_time_and_data(d):
    """
    Convert a pyuda object into 1D time and data arrays.
    Compatible with the style used in your database scripts.
    """
    data = np.asarray(d.data).squeeze()
    dims = getattr(d, "dims", None)
    if dims is None:
        dims = []

    if hasattr(d, "time") and hasattr(d.time, "data"):
        t = np.asarray(d.time.data).squeeze()
    else:
        t = None

    if data.ndim == 1:
        if t is None:
            if len(dims) >= 1:
                t = np.asarray(dims[0].data).squeeze()
            else:
                return None, None
        return t, data

    if data.ndim == 2:
        if t is not None:
            if data.shape[0] == len(t):
                return t, np.nanmean(data, axis=1)
            elif data.shape[1] == len(t):
                return t, np.nanmean(data, axis=0)

        if len(dims) >= 2:
            d0 = np.asarray(dims[0].data).squeeze()
            d1 = np.asarray(dims[1].data).squeeze()

            if data.shape[0] == len(d0):
                return d0, np.nanmean(data, axis=1)
            elif data.shape[1] == len(d1):
                return d1, np.nanmean(data, axis=0)
            elif data.shape[1] == len(d0):
                return d0, np.nanmean(data, axis=0)
            elif data.shape[0] == len(d1):
                return d1, np.nanmean(data, axis=1)

    return None, None


def moving_average_same(y, npts):
    """
    IDL smooth-like moving average using centered convolution.
    Keeps output length the same.
    """
    y = np.asarray(y, dtype=float)
    npts = int(npts)

    if npts < 2:
        return y.copy()

    kernel = np.ones(npts, dtype=float) / float(npts)
    return np.convolve(y, kernel, mode="same")


def group_events_by_gap(t_above, y_above=None, min_gap_s=0.002, mode="max"):
    """
    Group threshold-crossing points into separate events.

    Parameters
    ----------
    t_above : array
        Times where residual >= threshold
    y_above : array or None
        Residual values at those times
    min_gap_s : float
        Minimum gap to start a new event
    mode : str
        'first', 'last', or 'max'

    Returns
    -------
    telm : np.ndarray
        One time per event
    """
    t_above = np.asarray(t_above)
    if t_above.size == 0:
        return np.array([])

    if y_above is not None:
        y_above = np.asarray(y_above)

    dt = np.diff(t_above)
    split_idx = np.where(dt >= min_gap_s)[0]

    starts = np.r_[0, split_idx + 1]
    ends = np.r_[split_idx, len(t_above) - 1]

    telm = []
    for s, e in zip(starts, ends):
        tg = t_above[s:e+1]

        if mode == "first":
            telm.append(tg[0])
        elif mode == "last":
            telm.append(tg[-1])
        elif mode == "max":
            if y_above is None:
                telm.append(tg[0])
            else:
                yg = y_above[s:e+1]
                imax = np.nanargmax(yg)
                telm.append(tg[imax])
        else:
            raise ValueError("mode must be 'first', 'last', or 'max'")

    return np.asarray(telm)


def detect_elm_mastu(
    shot,
    signal="/XIM/DA/HM10/T",
    tsel=(0.2, 0.9),
    thr=0.008,
    smooth_pts=1000,
    min_gap_s=0.002,
    event_mode="max",
):
    """
    Python reproduction of calc_elm_mastu.pro
    """
    d = client.get(signal, shot)
    t, da = extract_time_and_data(d)

    if t is None or da is None:
        raise RuntimeError(f"Could not extract time/data for signal {signal} on shot {shot}")

    t = np.asarray(t).squeeze()
    da = np.asarray(da).squeeze()

    m = np.isfinite(t) & np.isfinite(da)
    t = t[m]
    da = da[m]

    if len(t) < 2:
        raise RuntimeError(f"Not enough finite data points for shot {shot}")

    order = np.argsort(t)
    t = t[order]
    da = da[order]

    smda = moving_average_same(da, smooth_pts)
    das = da - smda

    mt = (t >= tsel[0]) & (t <= tsel[1])
    td = t[mt]
    dad = da[mt]
    smdad = smda[mt]
    dasd = das[mt]

    mabove = dasd >= thr
    t_above = td[mabove]
    y_above = dasd[mabove]

    telm = group_events_by_gap(
        t_above=t_above,
        y_above=y_above,
        min_gap_s=min_gap_s,
        mode=event_mode,
    )

    nelm = len(telm)
    duration = float(tsel[1] - tsel[0])
    felm = nelm / duration if duration > 0 else np.nan

    # residual at the chosen event times
    if len(telm) > 0:
        yelm = np.interp(telm, td, dasd, left=np.nan, right=np.nan)
        da_elm = np.interp(telm, td, dad, left=np.nan, right=np.nan)
        bg_elm = np.interp(telm, td, smdad, left=np.nan, right=np.nan)
    else:
        yelm = np.array([])
        da_elm = np.array([])
        bg_elm = np.array([])

    return {
        "shot": shot,
        "signal": signal,
        "t": t,
        "da": da,
        "smda": smda,
        "das": das,
        "td": td,
        "dad": dad,
        "smdad": smdad,
        "dasd": dasd,
        "t_above": t_above,
        "y_above": y_above,
        "telm": telm,
        "yelm": yelm,
        "da_elm": da_elm,
        "bg_elm": bg_elm,
        "nelm": nelm,
        "felm": felm,
        "tsel": tsel,
        "thr": thr,
        "smooth_pts": smooth_pts,
        "min_gap_s": min_gap_s,
        "event_mode": event_mode,
    }


# =============================================================================
# PLOTTING
# =============================================================================
def plot_elm_result(res):
    """
    Lorenzo-style 4-panel diagnostic plot.
    """
    shot = res["shot"]
    td = res["td"]
    dad = res["dad"]
    smdad = res["smdad"]
    dasd = res["dasd"]
    telm = res["telm"]
    thr = res["thr"]
    felm = res["felm"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    # 1) raw signal
    axes[0].plot(td, dad)
    axes[0].set_ylabel("Dalpha")
    axes[0].set_title(f"Shot {shot} - signal + smoothing + ELM detection")

    # 2) smoothed background
    axes[1].plot(td, smdad)
    axes[1].set_ylabel("Smoothed")

    # 3) residual + threshold
    axes[2].plot(td, dasd)
    axes[2].axhline(thr, linestyle="--")
    axes[2].set_ylabel("Residual")

    # 4) raw signal + vertical markers at detected ELM times
    axes[3].plot(td, dad)
    if len(telm) > 0:
        y0 = np.nanmin(dad)
        y1 = np.nanmax(dad)
        for te in telm:
            axes[3].vlines(te, y0, y1)
    axes[3].set_ylabel("Dalpha")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_title(f"Detected ELMs: {len(telm)}   f_ELM = {felm:.2f} Hz")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_elm_overlay(res):
    """
    Extra plot:
    - top: raw Dalpha + red smoothed background
    - bottom: residual + threshold + detected ELM peaks
    """
    td = res["td"]
    dad = res["dad"]
    smdad = res["smdad"]
    dasd = res["dasd"]
    telm = res["telm"]
    yelm = res["yelm"]
    thr = res["thr"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Top: raw + background
    axes[0].plot(td, dad, label="Dalpha")
    axes[0].plot(td, smdad, color="red", linewidth=2.0, label="Background")
    if len(telm) > 0:
        da_elm = np.interp(telm, td, dad, left=np.nan, right=np.nan)
        axes[0].plot(telm, da_elm, "o", label="Detected ELMs")
    axes[0].set_ylabel("Dalpha")
    axes[0].set_title(f"Shot {res['shot']} - raw signal and smoothed background")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: residual + threshold + event markers
    axes[1].plot(td, dasd, label="Residual")
    axes[1].axhline(thr, color="red", linestyle="--", label="Threshold")
    if len(telm) > 0:
        axes[1].plot(telm, yelm, "o", label=f"ELM {res['event_mode']} times")
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title(f"Detected ELM events: {len(telm)}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


# =============================================================================
# SAVING
# =============================================================================
def save_elm_times_for_fit(res, outdir="."):
    """
    Save ELM detection output in Python-native format for later profile fitting.
    """
    os.makedirs(outdir, exist_ok=True)
    shot = res["shot"]
    outpath = os.path.join(outdir, f"telm_{shot}.npz")

    np.savez(
        outpath,
        shot=shot,
        signal=res["signal"],
        telm=res["telm"],
        yelm=res["yelm"],
        da_elm=res["da_elm"],
        bg_elm=res["bg_elm"],
        felm=res["felm"],
        nelm=res["nelm"],
        tsel=np.asarray(res["tsel"]),
        thr=res["thr"],
        smooth_pts=res["smooth_pts"],
        min_gap_s=res["min_gap_s"],
        event_mode=res["event_mode"],
        td=res["td"],
        dad=res["dad"],
        smdad=res["smdad"],
        dasd=res["dasd"],
    )

    print(f"Saved ELM data for later fitting to: {outpath}")
    return outpath


def load_elm_times_npz(path):
    """
    Convenience loader for saved ELM files.
    """
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}

    # convert scalar arrays where convenient
    for key in ["shot", "felm", "nelm", "thr", "smooth_pts", "min_gap_s"]:
        if key in out and np.ndim(out[key]) == 0:
            out[key] = out[key].item()

    if "event_mode" in out and np.ndim(out["event_mode"]) == 0:
        out["event_mode"] = str(out["event_mode"].item())

    return out


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    res = detect_elm_mastu(
        shot=SHOT,
        signal=SIGNAL,
        tsel=TSEL,
        thr=THR,
        smooth_pts=SMOOTH_PTS,
        min_gap_s=MIN_GAP_S,
        event_mode=EVENT_MODE,
    )

    print(f"Shot:       {res['shot']}")
    print(f"Signal:     {res['signal']}")
    print(f"Window:     {res['tsel'][0]:.3f} - {res['tsel'][1]:.3f} s")
    print(f"Threshold:  {res['thr']}")
    print(f"Smooth pts: {res['smooth_pts']}")
    print(f"Min gap:    {res['min_gap_s']:.4f} s")
    print(f"Mode:       {res['event_mode']}")
    print(f"N ELMs:     {res['nelm']}")
    print(f"f_ELM:      {res['felm']:.2f} Hz")

    plot_elm_result(res)
    plot_elm_overlay(res)

    if SAVE_ELM_TIMES:
        save_elm_times_for_fit(res, OUTDIR)

    plt.show()