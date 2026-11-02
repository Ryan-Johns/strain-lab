"""Streamlit-only helpers: session state, cached file loading and the shared settings form."""
import copy
import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from . import config as C
from .io_utils import read_csv_bytes, read_zip_csvs, numeric_columns, guess_time_column
from .processing import count_repeated_times
from .settings import FilterSettings

STATE_DEFAULTS = {
    "library": [],
    "save_counter": 0,
    "current_trace": None,
    "latest_settings": None,
    "latest_run": None,
    "filter_presets": None,      # filled from DEFAULT_PRESETS below
    "outputs": {},               # page -> {name: bytes / DataFrame}
}


def init_state():
    for k, v in STATE_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = copy.deepcopy(v)
    refresh_presets()


def refresh_presets():
    """Built-in presets plus everyone's saved presets from the Data store (shared between users)."""
    try:
        shared = get_store().presets
    except Exception:
        shared = {}
    st.session_state.filter_presets = {**copy.deepcopy(C.DEFAULT_PRESETS), **shared}


def user_name() -> str:
    """Name shown as 'added by' on records and runs."""
    return str(st.session_state.get("user_name", "") or "").strip()


def next_id() -> int:
    st.session_state.save_counter += 1
    return st.session_state.save_counter


def put_output(page: str, **kw):
    st.session_state.outputs.setdefault(page, {}).update(kw)


def get_output(page: str, name: str):
    return st.session_state.outputs.get(page, {}).get(name)


def show_table(df: Optional[pd.DataFrame]):
    if df is not None:
        st.dataframe(df, width="stretch", hide_index=True)


# ----------------------------------------------------------------------------- cached loading
@st.cache_data(show_spinner="Reading CSV…", max_entries=12)
def _load_csv(raw: bytes):
    df = read_csv_bytes(raw)
    num = numeric_columns(df)
    return df, num, guess_time_column(df, num)


@st.cache_data(show_spinner="Reading ZIP…", max_entries=4)
def _load_zip(raw: bytes):
    out = []
    for name, df in read_zip_csvs(raw):
        num = numeric_columns(df)
        out.append((name, df, num, guess_time_column(df, num)))
    return out


def load_csv(uploaded):
    """(df, numeric_columns, guessed_time). Cached, so the file is parsed once, not on every click."""
    return _load_csv(uploaded.getvalue())


def load_upload(uploaded):
    """List of dicts {file_name, df, numeric, time_guess} for a CSV or a ZIP of CSVs."""
    if uploaded.name.lower().endswith(".zip"):
        return [dict(file_name=n, df=d, numeric=num, time_guess=g) for n, d, num, g in _load_zip(uploaded.getvalue())]
    df, num, g = load_csv(uploaded)
    return [dict(file_name=uploaded.name, df=df, numeric=num, time_guess=g)]


def time_column_picker(df, numeric, guess, key, label="Time column"):
    idx = numeric.index(guess) if guess in numeric else 0
    col = st.selectbox(label, numeric, index=idx, key=key)
    rep = count_repeated_times(pd.to_numeric(df[col], errors="coerce").dropna().to_numpy())
    if rep:
        st.warning(f"`{col}` has repeated or backward time steps (the time values were rounded when saved). "
                   "Tick **Rebuild time from sampling frequency** in the filter settings to use an even time axis.")
    return col


def time_window(key: str, lo: float, hi: float, labels=("Start time", "End time")) -> Tuple[float, float]:
    c1, c2 = st.columns(2)
    start = c1.number_input(labels[0], value=float(lo), format="%.6f", key=f"{key}_start")
    end = c2.number_input(labels[1], value=float(hi), format="%.6f", key=f"{key}_end")
    if end <= start:
        st.error("End time must be greater than start time.")
        st.stop()
    return float(start), float(end)


# ----------------------------------------------------------------------------- shared settings form
def settings_form(prefix: str, x: Optional[np.ndarray] = None, default_preset: Optional[str] = None,
                  show_retry: bool = True) -> Tuple[FilterSettings, str]:
    """Preset picker + filter + baseline settings. Used by Filter, Batch Process and Filtered Export.

    Widget keys include the preset name, so choosing a preset reloads every field from it.
    Returns (settings, preset_name).
    """
    presets = st.session_state.filter_presets
    names = ["Manual / Custom"] + list(presets)
    idx = names.index(default_preset) if default_preset in names else 0
    preset_name = st.selectbox("Settings preset", names, index=idx, key=f"{prefix}_preset")
    d = FilterSettings.from_preset(presets.get(preset_name))
    k = lambda name: f"{prefix}_{preset_name}_{name}"

    c1, c2, c3, c4 = st.columns(4)
    ftype = c1.selectbox("Filter type", C.FILTER_TYPES, index=C.FILTER_TYPES.index(d.filter_type), key=k("type"))
    order = c2.number_input("Filter order", 1, 12, int(d.order), step=1, key=k("order"))
    low = c3.number_input("Low cutoff (Hz)", min_value=1e-9, value=float(d.cutoff_low), format="%.6f",
                          key=k("low"), disabled=ftype not in C.NEEDS_LOW_CUTOFF)
    high = c4.number_input("High cutoff (Hz)", min_value=1e-9, value=float(d.cutoff_high), format="%.6f",
                           key=k("high"), disabled=ftype not in C.NEEDS_HIGH_CUTOFF)

    c1, c2, c3 = st.columns(3)
    fs = c1.number_input("Sampling frequency (Hz)", min_value=1e-6, value=float(d.fs), format="%.4f", key=k("fs"))
    passes_opts = ["2 — zero-phase (forward + backward)", "1 — single forward pass (like DPlot Passes = 1)"]
    passes = c2.selectbox("Passes", passes_opts, index=0 if int(d.passes) == 2 else 1, key=k("passes"),
                          help="2 passes doubles the effective order and removes the time delay. "
                               "V46 always used 2 passes.")
    rebuild = c3.checkbox("Rebuild time from sampling frequency", value=bool(d.rebuild_time), key=k("rebuild"),
                          help="Replaces the time column with t0 + i/fs. Use when the CSV time column was rounded.")

    c1, c2 = st.columns(2)
    bmode = c1.selectbox("Baseline correction", C.BASELINE_MODES, index=C.BASELINE_MODES.index(d.baseline_mode),
                         key=k("bmode"), help=C.BASELINE_HELP)
    target = c2.selectbox("Apply baseline correction to", C.BASELINE_TARGETS,
                          index=C.BASELINE_TARGETS.index(d.baseline_target), key=k("target"),
                          disabled=bmode == C.BL_NONE,
                          help="Which signals get zeroed. The zeroed versions appear as 'Raw (zeroed)' and "
                               "'Filtered (zeroed)'; 'Raw' and 'Filtered' are always before correction.")

    qs = qe = None
    zero_window = d.zero_window_s
    if bmode == C.BL_ZERO_START:
        zero_window = st.number_input("N seconds at the start", min_value=0.1, max_value=3600.0,
                                      value=float(d.zero_window_s), step=5.0, key=k("zerowin"),
                                      help="The median of this many seconds at the start of the record is subtracted. "
                                           "10 s matches the 'initial baseline' used on the Baseline Shift page.")
    elif bmode == C.BL_QUIET:
        lo = float(x[0]) if x is not None and len(x) else 0.0
        hi = float(x[min(len(x) - 1, max(1, len(x) // 20))]) if x is not None and len(x) > 1 else 1.0
        qs, qe = time_window(k("quiet"), lo, hi, ("Quiet window start", "Quiet window end"))
        st.caption("The average over this window is subtracted from the whole record.")

    retry, step, max_trim = d.auto_retry, d.trim_step, d.max_trim
    if show_retry and bmode in C.INTEGRATED_MODES:
        with st.expander("DPlot Integrated auto-retry"):
            retry = st.checkbox("Auto-retry by trimming the start of the trace when the arrival interval is zero",
                                value=bool(d.auto_retry), key=k("retry"))
            c1, c2 = st.columns(2)
            step = c1.number_input("Auto-trim step", min_value=1e-9, value=float(d.trim_step), format="%.4f", key=k("step"))
            max_trim = c2.number_input("Maximum auto-trim", min_value=1e-9, value=float(d.max_trim), format="%.4f", key=k("maxtrim"))

    s = FilterSettings(
        filter_type=ftype, order=int(order), cutoff_low=float(low), cutoff_high=float(high),
        passes=2 if passes.startswith("2") else 1, fs=float(fs), rebuild_time=bool(rebuild),
        baseline_mode=bmode, baseline_target=target, quiet_start=qs, quiet_end=qe, zero_window_s=float(zero_window),
        auto_retry=bool(retry), trim_step=float(step), max_trim=float(max_trim),
        frequency_signal=d.frequency_signal, n_bins=d.n_bins, negative_only=d.negative_only,
        common_range=d.common_range, make_frequency=d.make_frequency,
    )
    return s, preset_name


def save_preset_box(s: FilterSettings, key: str):
    with st.expander("Save these settings as a preset"):
        st.caption("Presets are saved in the Data store, so everyone using this app can pick them.")
        name = st.text_input("Preset name", key=f"{key}_name")
        if st.button("Save preset", key=f"{key}_btn"):
            name = name.strip()
            if not name:
                st.warning("Enter a preset name first.")
            elif name in C.DEFAULT_PRESETS:
                st.warning("That name belongs to a built-in preset; choose another.")
            else:
                get_store().set_preset(name, s.to_preset() | {"saved_by": user_name()})
                refresh_presets()
                st.success(f"Saved preset: {name}")
        custom = sorted(n for n in st.session_state.filter_presets if n not in C.DEFAULT_PRESETS)
        if custom:
            c1, c2 = st.columns([3, 1])
            doomed = c1.selectbox("Delete a saved preset", ["—"] + custom, key=f"{key}_del_pick")
            if c2.button("Delete", key=f"{key}_del", disabled=doomed == "—"):
                get_store().delete_preset(doomed)
                refresh_presets()
                st.rerun()


def preset_sidebar():
    """Download/upload presets so they survive between sessions."""
    with st.sidebar.expander("Presets file"):
        st.download_button("Download presets (.json)", json.dumps(st.session_state.filter_presets, indent=2),
                           file_name="strain_lab_presets.json", mime="application/json", key="presets_dl")
        up = st.file_uploader("Load presets (.json)", type=["json"], key="presets_ul")
        if up is not None and st.button("Add presets from file", key="presets_add"):
            try:
                data = json.loads(up.getvalue())
                data = data.get("filter_presets", data)
                if not isinstance(data, dict):
                    raise ValueError("file does not contain a dictionary of presets")
                store = get_store()
                with store.transaction():
                    for name, preset in data.items():
                        if name not in C.DEFAULT_PRESETS:
                            store.presets[name] = preset
                refresh_presets()
                st.success(f"Added {len(data)} preset(s).")
            except Exception as e:
                st.error(f"Could not read presets: {e}")


def shift_control(key: str, limit: float, step: float):
    """Slider + exact number + nudge buttons, all kept in sync; returns the shift."""
    vk, sk, nk = f"{key}_v", f"{key}_s", f"{key}_n"
    for kk in (vk, sk, nk):
        st.session_state.setdefault(kk, 0.0)

    def set_value(v):
        st.session_state[vk] = float(v)
        st.session_state[nk] = float(v)
        st.session_state[sk] = float(min(max(v, -limit), limit))

    st.slider("X-shift", -limit, limit, step=step, key=sk, on_change=lambda: set_value(st.session_state[sk]))
    st.number_input("Exact x-shift", format="%.6f", key=nk, on_change=lambda: set_value(st.session_state[nk]))
    cols = st.columns(4)
    for col, delta in zip(cols, (-1.0, -0.1, 0.1, 1.0)):
        col.button(f"{delta:+g}", key=f"{key}_{delta}", on_click=lambda d=delta: set_value(st.session_state[vk] + d))
    return float(st.session_state[vk]), set_value


# ----------------------------------------------------------------------------- data store
def default_store_root() -> str:
    import os
    from pathlib import Path
    if os.environ.get("STRAIN_LAB_DATA"):
        return os.environ["STRAIN_LAB_DATA"]
    if hosted():                     # online: a local working copy; the real data lives in the Hugging Face dataset
        import tempfile
        return str(Path(tempfile.gettempdir()) / "strain_lab_data")
    return str(Path(__file__).resolve().parent.parent / "strain_lab_data")


def hosted() -> bool:
    """True when running online with the data kept in a Hugging Face dataset."""
    from . import cloud
    return cloud.configured()


def get_store():
    """The persistent data store. Online, its contents are mirrored to the private Hugging Face dataset."""
    from .store import DataStore
    from . import cloud
    root = default_store_root() if hosted() else st.session_state.setdefault("store_root", default_store_root())
    try:
        cloud.get_mirror(root)
    except Exception as e:
        st.error("Could not load the shared data from Hugging Face, so the app has stopped to protect it. "
                 f"Reload the page in a minute; if it keeps happening, check the app's Secrets settings.\n\n`{e}`")
        st.stop()
    return DataStore(root)


def _ago(ts) -> str:
    import time
    s = max(0, int(time.time() - ts))
    return "just now" if s < 60 else f"{s // 60} min ago" if s < 3600 else f"{s // 3600} h ago"


def sync_status():
    """Sidebar: whether changes are saved online, plus a Save now button that also checks the online copy."""
    from . import cloud
    m = cloud._MIRROR
    if m is None:
        return
    if m.last_error:
        st.sidebar.error(f"Not yet saved online ({m.pending} change(s)); retrying automatically. {m.last_error[:120]}")
    elif m.pending:
        st.sidebar.caption(f"⏳ Saving {m.pending} change(s) online…")
    else:
        st.sidebar.caption("✅ All changes saved online" + (f" (last {_ago(m.last_push_ts)})" if m.last_push_ts else ""))
    if st.sidebar.button("💾 Save now", key="sync_now", use_container_width=True,
                         help="Saves anything waiting and then checks the online copy against yours. "
                              "Your work is also saved automatically a few seconds after each change."):
        with st.spinner("Saving and checking the online copy…"):
            try:
                ok, msg = m.verify()
            except Exception as e:
                ok, msg = False, f"Couldn't reach Hugging Face to check ({type(e).__name__}). Try again in a minute."
        st.session_state["save_check"] = (ok, msg)
    res = st.session_state.pop("save_check", None)
    if res:
        (st.sidebar.success if res[0] else st.sidebar.warning)(res[1])
    st.sidebar.caption("Saved online: data-store files, temperatures, runs, presets, hidden points. "
                       "Not saved: the session library and plot settings.")


# ----------------------------------------------------------------------------- raw sources (uploads)
def read_source(src):
    """Full DataFrame for one RawSource; only the most recent one is kept in memory."""
    cache = st.session_state.setdefault("_source_cache", {})
    if src.key not in cache:
        cache.clear()
        with st.spinner(f"Reading {src.file_name}…"):
            cache[src.key] = src.read()
    return cache[src.key]


def preview(src):
    """(df, numeric_columns, time_guess) for a RawSource, or None (with an error shown) if it can't be read."""
    try:
        df = read_source(src)
    except Exception as e:
        st.error(f"Could not read {src.path}: {e}")
        return None
    num = numeric_columns(df)
    return df, num, guess_time_column(df, num)


# ----------------------------------------------------------------------------- series for plotting
@st.cache_resource(max_entries=24, show_spinner="Loading…")   # resource: arrays are shared, not copied each rerun
def store_series(root: str, record_id: str, channel: str, proc: str, settings_json: str, signal: str, stamp: str):
    """(time, values) for one stored channel, processed as requested.

    proc: 'raw' | 'run:<run_id>' | 'settings' (filter now with settings_json). stamp invalidates the cache.
    """
    from .store import DataStore
    from .processing import process_trace
    from .settings import FilterSettings
    from .config import SIGNALS
    store = DataStore(root)
    if proc.startswith("run:"):
        tr = store.run_traces(proc[4:], [channel])[channel]
        return tr["x"], tr[SIGNALS[signal]]
    t, raw = store.load_raw(record_id, [channel])
    y = raw[channel].to_numpy(float)
    if proc == "raw":
        return t, y
    res = process_trace(t, y, FilterSettings.from_preset(json.loads(settings_json)) if settings_json else FilterSettings())
    return res.x, getattr(res, SIGNALS[signal])


PLOT_DETAIL = {"Fast (5,000 points per line)": 5_000, "Standard (20,000 points per line)": 20_000,
               "High (100,000 points per line)": 100_000, "Full (every point — slow)": 0}


def plot_points() -> int:
    """Max points per line sent to the browser (0 = all), from the sidebar 'Plot detail' setting."""
    return PLOT_DETAIL[st.session_state.get("plot_detail", "Standard (20,000 points per line)")]


def plot_window(key: str, lo: float, hi: float):
    """'Show from / to' inputs. Narrowing the window re-selects points, so short events get full detail."""
    c1, c2 = st.columns(2)
    a = c1.number_input("Show from (s)", value=float(lo), format="%.3f", key=f"{key}_from_{lo:.3f}_{hi:.3f}")
    b = c2.number_input("Show to (s)", value=float(hi), format="%.3f", key=f"{key}_to_{lo:.3f}_{hi:.3f}")
    return (a, b) if b > a else (lo, hi)
