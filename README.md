# MAST-U ELM Detection and Pedestal Profile Fitting

This repository contains Python scripts for detecting Edge Localized Modes (ELMs) in MAST-U D-alpha data and using those ELM times to perform Lorenzo-style Thomson-scattering pedestal profile fits.

The code was originally written to stay close to legacy IDL analysis workflows. The repository now adds documentation, usage notes, dependency lists, and clearer in-code explanations so new users can understand the workflow without first reading the IDL sources.

## Repository contents

| File | Purpose |
| --- | --- |
| `03.calc_elm_mastu.py` | Detects ELM times for one MAST-U shot, plots diagnostics, and saves `telm_<shot>.npz` for later profile fitting. |
| `03.calc_elm_mastu_MULTI_SHOT.py` | Batch version of the ELM detector for multiple shots; can save one `.npz` file per shot, diagnostic plots, and a CSV summary. |
| `04.V14_fit_mastu_ts_profiles_Te_fit.py` | Loads ELM timings, Thomson-scattering data, and equilibrium data; maps profiles to `psi_N`; fits `ne` and `Te` pedestal profiles; saves plots, CSVs, and `.npz` outputs. |
| `requirements.txt` | Python package requirements needed by the scripts. |
| `.gitignore` | Excludes generated data products, plots, caches, and local virtual environments from version control. |
| `LICENSE` | MIT license so the repository can be reused and adapted by others. |

## Workflow overview

1. **Detect ELMs** from a D-alpha signal.
2. **Save ELM times** to `telm_<shot>.npz`.
3. **Run pedestal fitting** using the saved ELM times and the selected ELM-cycle fraction.
4. **Inspect outputs**: diagnostic figures, mapped Te point CSVs, `.npz` fit products, and summary CSV rows.

The ELM detection scripts must normally be run before `04.V14_fit_mastu_ts_profiles_Te_fit.py`, because the profile-fitting script expects a `telm_<shot>.npz` file for each requested shot.

## Requirements

### Python

Python 3.9 or newer is recommended.

### Python packages

Install the public dependencies with:

```bash
python -m pip install -r requirements.txt
```

The scripts also require `pyuda`, which is usually provided by the UKAEA/MAST-U analysis environment rather than by the public Python Package Index. If `pip install pyuda` is not available in your environment, load the site-specific MAST-U Python environment before running these scripts.

### Data access

You need access to the MAST-U UDA signals used by the scripts, including:

- `/XIM/DA/HM10/T` for D-alpha ELM detection.
- Thomson-scattering density, temperature, uncertainty, and coordinate signals used inside `load_ts_data`.
- Equilibrium flux signals under the selected equilibrium tree, such as `EPM`.

The current fitting workflow intentionally focuses on Thomson-scattering
electron temperature and density only; ion-temperature analysis is disabled for
now to keep the output plots and saved products simpler.

## Quick start

### 1. Create and activate an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your laboratory environment already provides `pyuda`, activate that environment instead of a local virtual environment.

### 2. Detect ELMs for one shot

Open `03.calc_elm_mastu.py` and edit the **USER SETTINGS** block:

```python
SHOT = 49107
SIGNAL = "/XIM/DA/HM10/T"
TSEL = (0.2, 0.9)
THR = 0.011
SMOOTH_PTS = 1000
MIN_GAP_S = 0.002
EVENT_MODE = "max"
```

Run:

```bash
python 03.calc_elm_mastu.py
```

Expected outputs:

- Interactive diagnostic plots.
- `telm_<shot>.npz` if `SAVE_ELM_TIMES = True`.

### 3. Detect ELMs for many shots

Open `03.calc_elm_mastu_MULTI_SHOT.py` and edit:

```python
SHOTS = [49107, 52600, 52486, 52483]
TSEL = (0.2, 0.9)
THR = 0.01
SAVE_ELM_TIMES = True
SAVE_SUMMARY_CSV = True
```

Run:

```bash
python 03.calc_elm_mastu_MULTI_SHOT.py
```

Expected outputs:

- One `telm_<shot>.npz` file per successful shot.
- `elm_batch_summary.csv` when `SAVE_SUMMARY_CSV = True`.
- Optional plots in `elm_batch_plots/` when `SAVE_PLOTS = True`.

### 4. Fit Thomson-scattering pedestal profiles

Open `04.V14_fit_mastu_ts_profiles_Te_fit.py` and edit the case dictionaries near the top of the file. Important fields are:

| Field | Meaning |
| --- | --- |
| `shot` | MAST-U shot number. |
| `tr` | Time range used for selecting profile samples. |
| `perc_elm` | Accepted fraction of the ELM cycle, e.g. `[0.80, 0.95]`. |
| `r_shift_hfs`, `r_shift_lfs` | Side-specific radial shifts before mapping to `psi_N`. |
| `te_sep` | Separatrix electron temperature target used by the Te fit logic. |
| `xpf` | Minimum `psi_N` used for edge fit clouds. |
| `equi` | Equilibrium tree prefix, for example `EPM`. |

Run:

```bash
python 04.V14_fit_mastu_ts_profiles_Te_fit.py
```

Expected outputs depend on the `SAVE_OUTPUT`, `SAVE_PLOTS`, and output-directory settings. Common generated files include compressed fit products, mapped Te CSV files, and diagnostic figures.

## Important settings explained

### ELM detector settings

- `TSEL`: analysis time window in seconds.
- `THR`: residual D-alpha threshold after subtracting the smoothed background.
- `SMOOTH_PTS`: moving-average window length in samples.
- `MIN_GAP_S`: minimum time gap that separates two threshold-crossing bursts into different ELM events.
- `EVENT_MODE`:
  - `"first"`: event time is the first threshold crossing in a burst.
  - `"max"`: event time is the largest residual point in a burst.
  - `"last"`: event time is the final threshold crossing in a burst.

### Pedestal-fit settings

- `USE_IDL_MEAN_Z`: switches a mapping detail used while comparing with legacy IDL output.
- `MAPPING_METHOD`: interpolation method for converting R/Z/time coordinates to `psi_N`.
- `APPLY_GLOBAL_PSI_SHIFT`: applies a global Te `psi_N` shift so `Te(psi_N=1)` matches `te_sep`.
- `TE_USE_IDL_SQRT_WEIGHTS`: uses the IDL-style `sqrt(Te)` weights.
- `TE_REFIT_AFTER_SHIFT`: controls whether Te is refit after the separatrix shift; the IDL-like default is `False`.

## Generated files

These files are intentionally ignored by Git because they can be regenerated:

- `telm_*.npz`
- `elm_batch_summary.csv`
- `elm_batch_plots/`
- `lorenzo_style_fit_plots_v13/`
- `*_mapped_te_points.csv`
- `*.npz` fit products generated by the fitting workflow

## Development and validation

Run a syntax check before committing changes:

```bash
python -m py_compile 03.calc_elm_mastu.py 03.calc_elm_mastu_MULTI_SHOT.py 04.V14_fit_mastu_ts_profiles_Te_fit.py
```

A full scientific validation requires access to MAST-U UDA data and should compare:

- ELM times against diagnostic plots.
- Saved `telm_<shot>.npz` contents against expected shot windows.
- Mapped `psi_N` clouds against equilibrium expectations.
- Pedestal fit outputs against trusted IDL or previous Python baselines.

## Notes for new users

- Start with one shot before running the batch scripts.
- Keep the generated `telm_<shot>.npz` files in the same directory as the fitting script, or update `ELM_DIR`.
- Use `SAVE_PLOTS = True` when tuning thresholds or profile-fit settings.
- If a signal cannot be loaded, first verify UDA access and the exact signal path for the shot.
- The code intentionally keeps several legacy options because they are useful when comparing Python output with IDL output.
