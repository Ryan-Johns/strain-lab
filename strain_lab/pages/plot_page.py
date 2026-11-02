"""Plot & Compare: overlay any channels, locations and filter settings as time histories,
frequency-of-strain curves or spectra, with publication styling."""
import json
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import welch

from .. import ui
from ..config import SIGNAL_NAMES, SIGNAL_HELP, COLOR_OPTIONS, DASH_OPTIONS, LEGEND_POSITIONS
from ..figures import apply_style, decimate_minmax
from ..frequency import compute_frequency_curve, smooth_curve
from ..io_utils import read_csv_bytes
from ..library import sort_items, short_label, display_title, trace_title, signal as item_signal
from ..settings import FilterSettings
from ..workflows import date_tag

PLOT_TYPES = ["Time history", "Frequency of strain", "Spectrum (PSD)"]
COLOR_CHOICES = [n for n, v in COLOR_OPTIONS.items() if v is not None]
# Order used when assigning colors to new lines (high-contrast first; yellow and light gray last).
AUTO_ORDER = ["Red", "Blue", "Green", "Orange", "Purple", "Black", "Teal", "Brown", "Magenta", "Navy", "Dark Green",
              "Dark Orange", "Dodger Blue", "Slate Blue", "Gray", "Maroon", "Cyan", "Pink", "Lime", "Yellow", "Light Gray"]


def _lines():
    return st.session_state.setdefault("plot_lines", [])


def _bump():
    st.session_state["plot_lines_version"] = st.session_state.get("plot_lines_version", 0) + 1


def _new_line(**kw):
    lines = _lines()
    taken = {l["label"] for l in lines}
    if kw.get("label") in taken and kw.get("proc_name"):          # same gauge & file, different filter
        kw["label"] = f"{kw['label']} · {kw['proc_name']}"
    base, n = kw.get("label", ""), 2
    while kw.get("label") in taken:
        kw["label"], n = f"{base} ({n})", n + 1
    used = {l["color"] for l in lines}
    color = next((c for c in AUTO_ORDER if c not in used), AUTO_ORDER[len(lines) % len(AUTO_ORDER)])
    lines.append(dict(uid=uuid.uuid4().hex[:8], color=color, width=1.5, dash="solid", shift=0.0, show=True) | kw)


# ---------------------------------------------------------------------------- adding lines
def _add_from_store(store):
    if not store.records:
        st.caption("The Data store is empty.")
        return
    recs = sorted(store.records, key=lambda r: (r["location"], r["date"]))
    locs = sorted({r["location"] for r in recs})
    c1, c2 = st.columns([1, 3])
    sel_locs = c1.multiselect("Locations", locs, default=locs, key="pa_locs")
    labels = {f"{r['location']} · {r['date']} · {r['file_name']}": r["id"] for r in recs if r["location"] in sel_locs}
    chosen = c2.multiselect("Recordings", list(labels), key="pa_recs",
                            help="Pick one recording per location to compare locations, or several dates of one location.")
    chans = sorted({c for l in chosen for c in store.record(labels[l])["channels"]}, key=str.lower)
    sel_ch = st.multiselect("Channels", chans, key="pa_chans")

    run_names = sorted({r["name"] for r in store.runs})
    options = ["Raw (unfiltered)"] + (["Active saved run"] + [f"Saved run: {n}" for n in run_names] if run_names else []) \
              + [f"Preset: {n}" for n in st.session_state.filter_presets] + ["Custom settings"]
    c1, c2 = st.columns([3, 1])
    procs = c1.multiselect("Filtering (pick several to compare filters)", options,
                           default=["Active saved run"] if run_names else ["Raw (unfiltered)"], key="pa_procs")
    sig = c2.selectbox("Signal", SIGNAL_NAMES, index=3, key="pa_sig", help=SIGNAL_HELP)
    custom, custom_name = None, "Custom"
    if "Custom settings" in procs:
        with st.container(border=True):
            st.markdown("**Custom settings**")
            custom_name = st.text_input("Name for these settings", value="Custom", key="pa_custom_name")
            custom, _ = ui.settings_form("pa_custom", None)

    combos = []
    for lab in chosen:
        rec = store.record(labels[lab])
        for ch in sel_ch:
            if ch not in rec["channels"]:
                continue
            for p in procs:
                if p == "Raw (unfiltered)":
                    combos.append(dict(proc="raw", settings_json="", proc_name="raw", signal="Raw"))
                elif p == "Active saved run" or p.startswith("Saved run: "):
                    runs = store.runs_for(rec["id"])
                    run = store.active_run(rec["id"]) if p == "Active saved run" else \
                        next((r for r in runs if r["name"] == p[len("Saved run: "):]), None)
                    if run is None or ch not in run["channels"]:
                        continue
                    combos.append(dict(proc=f"run:{run['id']}", settings_json="", proc_name=f"saved: {run['name']}", signal=sig))
                elif p.startswith("Preset: "):
                    n = p[len("Preset: "):]
                    combos.append(dict(proc="settings", proc_name=n, signal=sig,
                                       settings_json=json.dumps(FilterSettings.from_preset(st.session_state.filter_presets[n]).__dict__)))
                elif custom is not None:
                    combos.append(dict(proc="settings", proc_name=custom_name, signal=sig, settings_json=json.dumps(custom.__dict__)))
                else:
                    continue
                c = combos[-1]
                sig_txt = "" if c["signal"] in ("Raw", "Filtered (zeroed)") else f" [{c['signal']}]"
                c.update(source="store", record_id=rec["id"], channel=ch,
                         label=trace_title(ch, rec["file_name"]) + sig_txt)
    # When one gauge is added with several filters, name the filter in every one of its labels.
    per_gauge = {}
    for c in combos:
        per_gauge[(c["record_id"], c["channel"])] = per_gauge.get((c["record_id"], c["channel"]), 0) + 1
    for c in combos:
        if per_gauge[(c["record_id"], c["channel"])] > 1:
            c["label"] += f" · {c['proc_name']}"
    if st.button(f"Add {len(combos)} line(s)", type="primary", disabled=not combos, key="pa_add"):
        for c in combos:
            _new_line(**c)
        _bump()
        st.rerun()


def _add_from_library():
    lib = sort_items(st.session_state.library)
    if not lib:
        st.caption("No traces in the session library (traces from Filter → single channel or download-only batches).")
        return
    titles = {it["label"]: display_title(it) for it in lib}
    labels = st.multiselect("Traces", list(titles), format_func=titles.get, key="pa_lib")
    sig = st.selectbox("Signal", SIGNAL_NAMES, index=3, key="pa_lib_sig")
    if st.button(f"Add {len(labels)} line(s)", disabled=not labels, key="pa_lib_add"):
        for l in labels:
            it = next(i for i in lib if i["label"] == l)
            _new_line(source="library", lib_label=l, signal=sig, label=display_title(it))
        _bump()
        st.rerun()


def _add_from_freq_csv():
    up = st.file_uploader("Frequency CSV (columns: label, strain_bin_center, frequency_percent)", type=["csv"], key="pa_fcsv")
    if up is None:
        return
    try:
        df = read_csv_bytes(up.getvalue())
        for c in ("strain_bin_center", "frequency_percent"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["label", "strain_bin_center", "frequency_percent"])
    except Exception as e:
        st.error(f"Could not read it: {e}")
        return
    key = getattr(up, "file_id", up.name)
    st.session_state.setdefault("freq_csvs", {})[key] = df
    labels = st.multiselect("Curves", list(df["label"].unique()), format_func=short_label, key="pa_fcsv_labels")
    if st.button(f"Add {len(labels)} curve(s)", disabled=not labels, key="pa_fcsv_add"):
        for l in labels:
            _new_line(source="freqcsv", csv_key=key, csv_label=l, label=short_label(l))
        _bump()
        st.rerun()
    st.caption("Uploaded curves appear on the 'Frequency of strain' plot type only.")


# ---------------------------------------------------------------------------- data for a line
def _series(store, line):
    """(time, values) or None."""
    if line["source"] == "store":
        if line["record_id"] not in {r["id"] for r in store.records}:
            return None
        stamp = ""
        if line["proc"].startswith("run:"):
            try:
                stamp = store.run(line["proc"][4:])["created"]
            except StopIteration:
                return None
        return ui.store_series(str(store.root), line["record_id"], line["channel"], line["proc"],
                               line["settings_json"], line["signal"], stamp)
    if line["source"] == "library":
        it = next((i for i in st.session_state.library if i["label"] == line["lib_label"]), None)
        return None if it is None else (np.asarray(it["x"], float), item_signal(it, line["signal"]))
    return None


def _style_controls(ptype: str):
    defaults = {"Time history": ("Time history", "Time (s)", "Strain (µε)"),
                "Frequency of strain": ("Frequency of strain events vs strain", "Strain (µε)", "Frequency (%)"),
                "Spectrum (PSD)": ("Power spectral density", "Frequency (Hz)", "PSD (µε²/Hz)")}[ptype]
    with st.expander("Figure style and export"):
        c1, c2, c3 = st.columns(3)
        k = lambda n: f"ps_{ptype}_{n}"
        style = dict(
            title=c1.text_input("Title", defaults[0], key=k("title")),
            x_title=c1.text_input("X-axis title", defaults[1], key=k("xt")),
            y_title=c1.text_input("Y-axis title", defaults[2], key=k("yt")),
            height=c1.number_input("Height (px)", 300, 2000, 600, step=50, key=k("h")),
            fixed_width=c1.checkbox("Fixed width", key=k("fw")),
            width=c1.number_input("Width (px)", 400, 4000, 1200, step=50, key=k("w")),
            title_font_size=c2.number_input("Title font size", 8, 60, 22, key=k("tfs")),
            axis_title_font_size=c2.number_input("Axis title font size", 8, 60, 16, key=k("afs")),
            tick_font_size=c2.number_input("Tick font size", 6, 40, 12, key=k("kfs")),
            legend_font_size=c2.number_input("Legend font size", 6, 40, 12, key=k("lfs")),
            bold_title=c2.checkbox("Bold title", key=k("bt")),
            bold_axis_titles=c2.checkbox("Bold axis titles", key=k("ba")),
            show_legend=c3.checkbox("Show legend", value=True, key=k("leg")),
            legend_position=c3.selectbox("Legend position", LEGEND_POSITIONS, index=4, key=k("lpos")),
            legend_background=c3.checkbox("Legend background", value=True, key=k("lbg")),
            legend_font_color=c3.text_input("Legend text color", "black", key=k("lfc")),
            legend_border_color=c3.text_input("Legend border color", "black", key=k("lbc")),
            legend_border_width=c3.number_input("Legend border width", 0.0, 10.0, 1.0, step=0.5, key=k("lbw")),
            show_grid=c3.checkbox("Grid", value=True, key=k("grid")),
            minor_grid=c3.checkbox("Minor log gridlines", key=k("minor")),
        )
        c1, c2, c3 = st.columns(3)
        style.update(tick_color=c1.text_input("Tick color", "black", key=k("tc")),
                     title_color=c1.text_input("Title color", "black", key=k("ttc")),
                     grid_color=c2.text_input("Grid color", "lightgray", key=k("gc")),
                     grid_width=c2.number_input("Grid width", 0.0, 5.0, 1.0, step=0.5, key=k("gw")),
                     minor_grid_width=c2.number_input("Minor grid width", 0.0, 5.0, 0.5, step=0.1, key=k("mgw")),
                     outline_color=c3.text_input("Outline color", "black", key=k("oc")),
                     outline_width=c3.number_input("Outline width", 0.0, 5.0, 1.0, step=0.5, key=k("ow")),
                     plot_bgcolor=c3.text_input("Plot background", "white", key=k("pbg")),
                     paper_bgcolor=c3.text_input("Paper background", "white", key=k("ppbg")))
        c1, c2, c3 = st.columns(3)
        export = dict(filename=c1.text_input("Export file name", ptype.split(" ")[0].lower() + "_plot", key=k("fn")),
                      format=c2.selectbox("Export format", ["png", "svg", "jpeg", "webp"], key=k("fmt")),
                      scale=c3.number_input("Export scale", 1, 10, 3, key=k("scale")))
    return style, export


# ---------------------------------------------------------------------------- page
def render():
    st.title("Plot & Compare")
    store = ui.get_store()
    lines = _lines()

    with st.expander("Add lines", expanded=not lines):
        t1, t2, t3 = st.tabs(["From Data store", "From session library", "Frequency CSV"])
        with t1:
            _add_from_store(store)
        with t2:
            _add_from_library()
        with t3:
            _add_from_freq_csv()
    if not lines:
        st.info("Add lines to start. Each line is one channel from one recording with one filter setting — add the same "
                "channel from several locations, or with several filters, to compare them on one plot.")
        return

    st.subheader("Lines")
    table = pd.DataFrame([{k: l[k] for k in ("show", "label", "color", "width", "dash", "shift")} for l in lines])
    edited = st.data_editor(table, hide_index=True, width="stretch", num_rows="fixed",
                            key=f"pa_lines_{st.session_state.get('plot_lines_version', 0)}",
                            column_config={
                                "show": st.column_config.CheckboxColumn("show", width="small"),
                                "label": st.column_config.TextColumn("legend label", width="large"),
                                "color": st.column_config.SelectboxColumn("color", options=COLOR_CHOICES, required=True),
                                "width": st.column_config.NumberColumn("width", min_value=0.5, max_value=8, step=0.5),
                                "dash": st.column_config.SelectboxColumn("style", options=DASH_OPTIONS, required=True),
                                "shift": st.column_config.NumberColumn("x shift", format="%.3f",
                                                                       help="Moves the line along the x axis (time or strain)."),
                            })
    for l, row in zip(lines, edited.to_dict("records")):
        l.update(row)
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Remove hidden lines", key="pa_rm"):
        st.session_state["plot_lines"] = [l for l in lines if l["show"]]
        _bump()
        st.rerun()
    if c2.button("Clear all", key="pa_clear"):
        st.session_state["plot_lines"] = []
        _bump()
        st.rerun()

    st.subheader("Plot")
    ptype = st.radio("Plot type", PLOT_TYPES, horizontal=True, key="pa_type")
    shown = [l for l in lines if l["show"]]
    data = {}
    for l in shown:
        if l["source"] != "freqcsv":
            got = _series(store, l)
            if got is None:
                st.warning(f"'{l['label']}' is no longer available (deleted from the store or library).")
            else:
                data[l["uid"]] = got

    lo = min((float(np.nanmin(x)) for x, _ in data.values()), default=0.0)
    hi = max((float(np.nanmax(x)) for x, _ in data.values()), default=1.0)
    c1, c2, c3 = st.columns(3)
    t0 = c1.number_input("Window start (s)", value=lo, format="%.3f", key=f"pa_t0_{lo:.3f}")
    t1 = c2.number_input("Window end (s)", value=hi, format="%.3f", key=f"pa_t1_{hi:.3f}")
    align = c3.checkbox("Zero each line at the window start", key="pa_align",
                        help="Subtracts each line's median over the first second of the window, so lines with "
                             "different baselines can be compared by shape.") if ptype != "Spectrum (PSD)" else False

    def windowed(uid, line):
        x, y = data[uid]
        m = np.isfinite(y) & (x >= t0) & (x <= t1)
        x, y = x[m], y[m]
        if align and len(y):
            y = y - np.median(y[x <= x[0] + 1.0])
        return x, y

    fig = go.Figure()
    curves, export_rows = {}, []
    style, export = _style_controls(ptype)

    if ptype == "Time history":
        for l in shown:
            if l["uid"] in data:
                x, y = windowed(l["uid"], l)
                export_rows.append((l["label"], x + l["shift"], y))
                x, y = decimate_minmax(x, y, ui.plot_points())
                fig.add_trace(go.Scattergl(x=x + l["shift"], y=y, mode="lines", name=l["label"],
                                           line=dict(color=COLOR_OPTIONS[l["color"]], width=l["width"], dash=l["dash"])))
        st.caption("For full detail on one event, narrow the window rather than drag-zooming.")
        if any(l["source"] == "freqcsv" for l in shown):
            st.caption("Uploaded frequency curves are only drawn on the 'Frequency of strain' plot type.")

    elif ptype == "Frequency of strain":
        c1, c2, c3, c4 = st.columns(4)
        bins = c1.number_input("Bins", 20, 2000, 300, step=10, key="pa_bins")
        smooth = c2.number_input("Smoothing (1 = off)", 1, 101, 1, key="pa_smooth")
        log_y = c3.checkbox("Log y-axis (%)", value=True, key="pa_logy")
        neg = c4.checkbox("Negative strain only", key="pa_neg")
        c1, c2 = st.columns(2)
        common = c1.checkbox("Same bins for all lines", value=True, key="pa_common")
        limits = None
        if c2.checkbox("Custom strain range", key="pa_lim"):
            c1, c2 = st.columns(2)
            limits = (c1.number_input("Min strain", value=-100.0, key="pa_xmin"), c2.number_input("Max strain", value=20.0, key="pa_xmax"))
        vals = {}
        for l in shown:
            if l["uid"] in data:
                y = windowed(l["uid"], l)[1]
                vals[l["uid"]] = y[y <= 0] if neg else y
        rng = limits
        if rng is None and common and vals:
            nonempty = [v for v in vals.values() if len(v)]
            if nonempty:
                rng = (min(float(v.min()) for v in nonempty), max(float(v.max()) for v in nonempty))
        curves_xy = {}
        for l in shown:
            if l["uid"] in vals:
                c, f, _ = compute_frequency_curve(vals[l["uid"]], int(bins), *(rng or (None, None)))
                curves_xy[l["uid"]] = (c, smooth_curve(f, int(smooth)))
            elif l["source"] == "freqcsv":
                df = st.session_state.get("freq_csvs", {}).get(l["csv_key"])
                if df is None:
                    st.warning(f"Re-upload the frequency CSV for '{l['label']}' (uploads don't persist between sessions).")
                    continue
                part = df[df["label"] == l["csv_label"]]
                curves_xy[l["uid"]] = (part["strain_bin_center"].to_numpy(float), part["frequency_percent"].to_numpy(float))
        if st.button("Auto-center lines at their peak", key="pa_center",
                     help="Sets each line's x shift so its most frequent strain is at 0."):
            for l in shown:
                if l["uid"] in curves_xy and len(curves_xy[l["uid"]][1]):
                    c, f = curves_xy[l["uid"]]
                    l["shift"] = -float(c[int(np.nanargmax(f))])
            _bump()
            st.rerun()
        for l in shown:
            if l["uid"] in curves_xy:
                c, f = curves_xy[l["uid"]]
                m = f > 0 if log_y else np.ones(len(f), bool)
                fig.add_trace(go.Scatter(x=c[m] + l["shift"], y=f[m], mode="lines", name=l["label"],
                                         line=dict(color=COLOR_OPTIONS[l["color"]], width=l["width"], dash=l["dash"])))
                export_rows.append((l["label"], c + l["shift"], f))
        style["percent_log_y"] = log_y

    else:
        c1, c2, c3 = st.columns(3)
        nper = c1.selectbox("Segment length (samples)", [2048, 4096, 8192, 16384, 32768], index=2, key="pa_nper",
                            help="Longer segments give finer frequency resolution but noisier curves.")
        fmax = c2.number_input("Max frequency (Hz)", 0.1, 10000.0, 100.0, key="pa_fmax")
        style["log_x"] = c3.checkbox("Log frequency axis", value=True, key="pa_logx")
        style["log_y"] = True
        for l in shown:
            if l["uid"] in data:
                x, y = windowed(l["uid"], l)
                if len(y) < 16:
                    continue
                fs = (len(x) - 1) / (x[-1] - x[0]) if x[-1] > x[0] else 200.0   # robust to rounded time columns
                f, p = welch(y - np.mean(y), fs=fs, nperseg=min(int(nper), len(y)))
                m = (f > 0) & (f <= fmax)
                fig.add_trace(go.Scatter(x=f[m], y=p[m], mode="lines", name=l["label"],
                                         line=dict(color=COLOR_OPTIONS[l["color"]], width=l["width"], dash=l["dash"])))
                export_rows.append((l["label"], f[m], p[m]))

    for l in shown:
        curves[l["label"]] = dict(name=l["label"], color=COLOR_OPTIONS[l["color"]], width=l["width"], dash=l["dash"])
    apply_style(fig, style, {k: v for k, v in curves.items() if any(tr.name == k for tr in fig.data)})
    st.plotly_chart(fig, width="content" if style.get("fixed_width") else "stretch", key="pa_fig", theme=None,
                    config={"toImageButtonOptions": {"format": export["format"], "filename": export["filename"],
                                                     "scale": int(export["scale"])}, "displaylogo": False})
    st.caption("Use the camera button at the top-right of the plot to save an image with the export settings.")

    if export_rows and st.button("Prepare plotted data (CSV)", key="pa_prep"):
        frames = [pd.DataFrame({"line": lab, "x": x, "y": y}) for lab, x, y in export_rows]
        ui.put_output("plot", csv=pd.concat(frames, ignore_index=True).to_csv(index=False).encode())
    if ui.get_output("plot", "csv"):
        st.download_button("Download plotted data", ui.get_output("plot", "csv"), "plotted_data.csv", "text/csv", key="pa_dl")
