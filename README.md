# Strain Gauge Filtering Lab — V53

## Run it

First time, in a terminal opened in this folder:

```
python -m pip install --upgrade -r requirements.txt
```

Then double-click **Start Strain Lab.bat**, or run `python -m streamlit run app.py`.
Keep `app.py`, the `strain_lab` folder and the `.streamlit` folder together. `.streamlit/config.toml` raises the
upload limit to 2,000 MB.

Your data is saved in `strain_lab_data` inside this folder (change it at the top of the Data page). Back it up.

## Online, shared version

See **HOSTING.md**: the app runs free on Streamlit Community Cloud and keeps its data in a private Hugging Face
dataset. Everyone who signs in with the password works on the same data.

## The five pages

**Data** — the permanent store.
- *Add raw files*: upload CSVs or ZIPs of CSVs (any folder layout). Location is read from the file, folder or ZIP
  name (`L3`, `Location 3`); date from the CSV name (`12_9_25`, `12-9-2025`). Files are read one at a time, stored
  exactly as uploaded (≈1/8 the CSV size) and never modified. Files already stored are skipped.
- *Records & temperatures*: type in temperatures (at recording, at sunrise the day before), fix locations/dates,
  add notes, view or re-download raw data, delete recordings.
- *Saved filtered runs*: rename runs, choose each recording's active run, delete runs (raw data is never touched).
- *Gauge types*: assigned from channel names; override any that are wrong.

**Filter** — choose *Data store* or *Upload files*, and *Single channel (preview)* or *Batch*.
- Single channel: filter one channel, see raw vs filtered, reproducible code, add to the session library.
- Batch from the store: filter many recordings/channels and save as a named run (≈18 MB per file).
- Batch from uploads: either save to the store (adds the raw files too) or download only
  (filtered CSVs by channel and date, time-history CSV, frequency CSVs).

**Plot & Compare** — add *lines*: any recording × channel × filtering (raw, a saved run, any preset, or custom
settings), from the store, the session library, or an uploaded frequency CSV. Compare locations, dates or filters
on one plot as a **time history**, **frequency of strain** curve, or **spectrum (PSD)**. Per-line label, color,
width, style and x-shift; auto-center at peak; zero lines at the window start; full figure styling and image export.

**Baseline Shift** — outliers can be hidden automatically (points that jump away from the line through their
neighbouring dates) or by clicking them on a chart and pressing *Hide selected*; hidden points are remembered, shown as
faint ×, left out of trend lines and the temperature table, and never deleted. *Vertical position* can show values as
measured, as change from the first date, or centered on zero (display only). Also: initial/end baseline (median of first/last N s), drift, whole-record median and mean, from raw
or filtered data, absolute or change since the first date. One chart per channel (locations overlaid), per gauge type,
per location × type, per location, or all on one chart; against date (with temperature) or temperature (with trend
lines); temperature-sensitivity table (µε/°F); CSV and Excel downloads.

**Exports** — from saved runs or the session library: filtered CSVs by channel (Time / Original / Filtered Corrected
per date), frequency CSVs by channel, appending dates to your existing frequency CSV ZIP, channel PNGs, one CSV per
recording. Built one channel at a time, so large selections don't run out of memory.

## Notes

- **Baseline correction** (hover the ? next to it in the app) subtracts one constant, so the record starts near
  zero and keeps its shape, including drift:
  - *Zero to start* (recommended): median of the first N seconds (default 10 s).
  - *Zero to a quiet window*: average over a window you choose.
  - *DPlot Integrated (legacy)* and *+ Shifted (legacy)*: DPlot's method, kept so old results can be reproduced.
  The corrected signals are **Raw (zeroed)** and **Filtered (zeroed)**; *Raw* and *Filtered* are always before correction.
- **Presets**: *Low-pass 20 Hz, zeroed (recommended)* keeps drift and temperature effects; *Band-pass 0.001–20 Hz,
  drift removed* is for event/fatigue work; *Bridge Traffic Default* is the original V46 setup.

- **Plot detail** (sidebar): each line is reduced to its minimum and maximum per slice before it's sent to the
  browser (20,000 points per line by default), so plots look identical, peaks included, but every click is fast.
  For full detail on a single event, narrow the *Show from / to* window instead of drag-zooming.

- Many raw files have a rounded TIME column (repeated time values after 10 s). The app warns about it; tick
  **Rebuild time from sampling frequency** in the filter settings.
- With the Bridge Traffic Default preset, results reproduce V46 exactly. *Passes = 1* matches DPlot's single pass.
- Filtered data is stored at 0.001 µε resolution (raw data resolution is 0.01 µε).

## Code layout

| File | Contents |
| --- | --- |
| `app.py` | Sidebar navigation |
| `strain_lab/processing.py` | Filters, baseline corrections, `process_trace()` |
| `strain_lab/store.py` | Data store: raw files, runs, temperatures, statistics |
| `strain_lab/workflows.py` | Import a file, filter recordings into the store, runs → traces |
| `strain_lab/baseline.py` | Baseline statistics, gauge types, temperature sensitivity |
| `strain_lab/frequency.py` | Frequency-of-strain curves and their ZIP exports |
| `strain_lab/figures.py` | Plot styling, channel PNGs |
| `strain_lab/io_utils.py` | CSV/ZIP reading, location/date from names |
| `strain_lab/cloud.py` | Mirrors the data store to a private Hugging Face dataset (online version) |
| `strain_lab/auth.py` | Sign-in for the online version |
| `strain_lab/ui.py` | Shared Streamlit pieces (settings form, caches) |
| `strain_lab/pages/` | One file per page |
| `tests/` | `python -m tests.test_equivalence <csv>` (vs V46), `tests.test_store <csv>`, `tests.test_pages <csv>` |
