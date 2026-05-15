# -*- coding: utf-8 -*-
"""
Python batch version of Lorenzo's calc_elm_mastu.pro
FREIA / Spyder ready

What it does
- loops over a list of shots
- loads /XIM/DA/HM10/T
- smooths the signal with a moving average
- subtracts background
- applies threshold in a selected time window
- groups threshold crossings into ELM events using a minimum time gap
- picks one ELM time per event
- optionally saves:
    1) one telm_<shot>.npz file per shot
    2) a summary CSV for the whole batch
    3) optional diagnostic plots per shot

Notes
- This is intentionally close to the IDL logic, not byte-for-byte identical
- Main output for later pedestal fitting is telm_<shot>.npz
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyuda import Client

client = Client()


# =============================================================================
# USER SETTINGS
# =============================================================================
SHOTS = [
    49107,
     52600,
     52486,
     52483,
]


#SHOTS = [
#    50833, 50885, 52425, 52427, 52429, 52432, 52444, 52448, 52483, 52486,
#    52490, 52494, 52501, 52506, 52570, 52600, 52603, 52604, 51796, 52018,
#    52019, 52020, 52021, 52022, 52434, 52441, 50579, 50580, 50581, 50582,
#    52284, 52285, 52290, 52291, 52293, 52296, 52306, 52616, 52625, 52552,
#    52553, 52554, 52558, 52563
#]
#SHOTS = [
#    50833, 50885, 52425, 52427, 52429, 52444, 52448, 52490, 52506, 52569,
#    52612, 51796, 51797, 51798, 51799, 51800, 51802, 51803, 51804, 51805,
#    51806, 52018, 52019, 52020, 52021, 52022, 52248, 52249, 52312, 52314,
#    52434, 52441, 50579, 50580, 50581, 50582, 52284, 52285, 52290, 52291,
#    52293, 52296, 52616, 52618, 52619, 52621, 52625, 50885, 52432, 52483,
#    52486, 52494, 52501, 52570, 52571, 52572, 52600, 52615,51939, 51942        
#]



SIGNAL = "/XIM/DA/HM10/T"

# Same window/settings for all shots
TSEL = (0.2, 0.9)
THR = 0.01
SMOOTH_PTS = 1000
MIN_GAP_S = 0.002
EVENT_MODE = "max"   # "first", "max", "last"

# Saving
SAVE_ELM_TIMES = True
SAVE_SUMMARY_CSV = True
SAVE_PLOTS = False

OUTDIR = "."
PLOT_DIR = "elm_batch_plots"
SUMMARY_CSV_NAME = "elm_batch_summary.csv"


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
            if len(dims) >= 1 and hasattr(dims[0], "data"):
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
            d0 = np.asarray(dims[0].data).squeeze() if hasattr(dims[0], "data") else None
            d1 = np.asarray(dims[1].data).squeeze() if hasattr(dims[1], "data") else None

            if d0 is not None and data.shape[0] == len(d0):
                return d0, np.nanmean(data, axis=1)
            elif d1 is not None and data.shape[1] == len(d1):
                return d1, np.nanmean(data, axis=0)
            elif d0 is not None and data.shape[1] == len(d0):
                return d0, np.nanmean(data, axis=0)
            elif d1 is not None and data.shape[0] == len(d1):
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


# =============================================================================
# MAIN DETECTOR
# =============================================================================
def detect_elm_mastu(
    shot,
    signal="/XIM/DA/HM10/T",
    tsel=(0.2, 0.9),
    thr=0.01,
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

    axes[0].plot(td, dad)
    axes[0].set_ylabel("Dalpha")
    axes[0].set_title(f"Shot {shot} - signal + smoothing + ELM detection")

    axes[1].plot(td, smdad)
    axes[1].set_ylabel("Smoothed")

    axes[2].plot(td, dasd)
    axes[2].axhline(thr, linestyle="--")
    axes[2].set_ylabel("Residual")

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

    axes[0].plot(td, dad, label="Dalpha")
    axes[0].plot(td, smdad, color="red", linewidth=2.0, label="Background")
    if len(telm) > 0:
        da_elm = np.interp(telm, td, dad, left=np.nan, right=np.nan)
        axes[0].plot(telm, da_elm, "o", label="Detected ELMs")
    axes[0].set_ylabel("Dalpha")
    axes[0].set_title(f"Shot {res['shot']} - raw signal and smoothed background")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

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

    print(f"   Saved ELM data to: {outpath}")
    return outpath


def build_summary_row(res):
    telm = np.asarray(res["telm"])
    if len(telm) >= 2:
        dt_elm = np.diff(telm)
        mean_dt = float(np.nanmean(dt_elm))
        std_dt = float(np.nanstd(dt_elm))
    else:
        mean_dt = np.nan
        std_dt = np.nan

    return {
        "shot": int(res["shot"]),
        "status": "ok",
        "signal": res["signal"],
        "t0": float(res["tsel"][0]),
        "t1": float(res["tsel"][1]),
        "thr": float(res["thr"]),
        "smooth_pts": int(res["smooth_pts"]),
        "min_gap_s": float(res["min_gap_s"]),
        "event_mode": str(res["event_mode"]),
        "n_elm": int(res["nelm"]),
        "f_elm_hz": float(res["felm"]),
        "mean_dt_elm_s": mean_dt,
        "std_dt_elm_s": std_dt,
        "first_elm_s": float(telm[0]) if len(telm) > 0 else np.nan,
        "last_elm_s": float(telm[-1]) if len(telm) > 0 else np.nan,
        "error_message": "",
    }


# =============================================================================
# BATCH RUNNER
# =============================================================================
def run_elm_batch(
    shots,
    signal="/XIM/DA/HM10/T",
    tsel=(0.2, 0.9),
    thr=0.01,
    smooth_pts=1000,
    min_gap_s=0.002,
    event_mode="max",
    outdir=".",
    save_npz=True,
    save_plots=False,
    plot_dir="elm_batch_plots",
):
    rows = []

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    for i, shot in enumerate(shots, start=1):
        print(f"\n[{i}/{len(shots)}] Processing shot {shot} ...")

        try:
            res = detect_elm_mastu(
                shot=shot,
                signal=signal,
                tsel=tsel,
                thr=thr,
                smooth_pts=smooth_pts,
                min_gap_s=min_gap_s,
                event_mode=event_mode,
            )

            print(f"   OK | nELM={res['nelm']} | fELM={res['felm']:.2f} Hz")

            if save_npz:
                save_elm_times_for_fit(res, outdir=outdir)

            if save_plots:
                fig1, _ = plot_elm_result(res)
                fig2, _ = plot_elm_overlay(res)

                fig1.savefig(
                    os.path.join(plot_dir, f"elm_diag_{shot}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                fig2.savefig(
                    os.path.join(plot_dir, f"elm_overlay_{shot}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig1)
                plt.close(fig2)

            rows.append(build_summary_row(res))

        except Exception as err:
            print(f"   FAILED | {err}")
            rows.append({
                "shot": int(shot),
                "status": "failed",
                "signal": signal,
                "t0": float(tsel[0]),
                "t1": float(tsel[1]),
                "thr": float(thr),
                "smooth_pts": int(smooth_pts),
                "min_gap_s": float(min_gap_s),
                "event_mode": str(event_mode),
                "n_elm": np.nan,
                "f_elm_hz": np.nan,
                "mean_dt_elm_s": np.nan,
                "std_dt_elm_s": np.nan,
                "first_elm_s": np.nan,
                "last_elm_s": np.nan,
                "error_message": str(err),
            })

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    df = run_elm_batch(
        shots=SHOTS,
        signal=SIGNAL,
        tsel=TSEL,
        thr=THR,
        smooth_pts=SMOOTH_PTS,
        min_gap_s=MIN_GAP_S,
        event_mode=EVENT_MODE,
        outdir=OUTDIR,
        save_npz=SAVE_ELM_TIMES,
        save_plots=SAVE_PLOTS,
        plot_dir=PLOT_DIR,
    )

    print("\nBatch finished.")
    print(df.to_string(index=False))

    if SAVE_SUMMARY_CSV:
        csv_path = os.path.abspath(os.path.join(OUTDIR, SUMMARY_CSV_NAME))
        df.to_csv(csv_path, index=False)
        print("\nSummary CSV saved to:")
        print(csv_path)