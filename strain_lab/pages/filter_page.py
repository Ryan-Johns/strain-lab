"""Filter: one channel with a preview, or many files/channels at once. From the Data store or from uploads."""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .. import ui
from ..config import SIGNAL_NAMES, SIGNAL_HELP, RECOMMENDED_PRESET
from ..figures import time_plot
from ..frequency import build_frequency_df, annotate, frequency_zip_by_channel
from ..io_utils import sources_from_upload, numeric_columns, safe_filename
from ..library import make_item, group_by_channel, trace_title
from ..processing import clean_xy, trim_arrays, process_trace
from ..store import DuplicateRecord
from ..workflows import filter_record, import_source, run_items, date_tag
from .code_page import reproducible_code


# ============================================================================ single channel
def _single_input(store, source):
    """Returns (x, y, meta) for the chosen channel, or None."""
    if source == "Data store":
        recs = sorted(store.records, key=lambda r: (r["location"], r["date"], r["file_name"]))
        labels = {store.record_label(r): r["id"] for r in recs}
        c1, c2 = st.columns([2, 1])
        rid = labels[c1.selectbox("Record", list(labels), key="flt_s_rec")]
        rec = store.record(rid)
        ch = c2.selectbox("Channel", rec["channels"], key=f"flt_s_ch_{rid}")
        t, raw = store.load_raw(rid, [ch])
        if ui.count_repeated_times(t):
            st.warning("This record's time column has repeated time steps; consider ticking "
                       "**Rebuild time from sampling frequency** below.")
        return t, raw[ch].to_numpy(float), dict(file_name=rec["file_name"], xcol=rec["time_col"], ycol=ch,
                                                 location=rec["location"], date=rec["date"], record_id=rid)
    up = st.file_uploader("Upload a CSV (or a ZIP of CSVs)", type=["csv", "zip"], key="flt_s_upload")
    if up is None:
        st.info("Upload a file to begin.")
        return None
    sources = sources_from_upload(up)
    if not sources:
        st.warning("No CSV files found.")
        return None
    src = sources[0]
    if len(sources) > 1:
        src = {s.path: s for s in sources}[st.selectbox("File in ZIP", [s.path for s in sources], key="flt_s_member")]
    p = ui.preview(src)
    if p is None:
        return None
    df, numeric, guess = p
    c1, c2 = st.columns(2)
    with c1:
        xcol = ui.time_column_picker(df, numeric, guess, "flt_s_xcol", "Time column")
    ycol = c2.selectbox("Channel", [c for c in numeric if c != xcol], key="flt_s_ycol")
    x, y = clean_xy(df, xcol, ycol)
    return x, y, dict(file_name=src.file_name, xcol=xcol, ycol=ycol)


def _single(store, source):
    got = _single_input(store, source)
    if got is None:
        return
    x, y, meta = got
    if len(x) < 3:
        st.error("Not enough valid points.")
        return
    if st.checkbox("Trim to a custom time window", key="flt_s_trim"):
        t0, t1 = ui.time_window("flt_s_trimwin", x.min(), x.max(), ("Trim start", "Trim end"))
        x, y = trim_arrays(x, y, t0, t1)

    st.subheader("Filter settings")
    s, _ = ui.settings_form("flt", x, default_preset=RECOMMENDED_PRESET)
    zeroing = s.baseline_mode != "None"
    show = st.multiselect("Show on plot", ["Raw", "Raw (zeroed)", "Filtered", "Filtered (zeroed)",
                                           "Raw baseline", "Filtered baseline"],
                          default=["Raw (zeroed)", "Filtered (zeroed)"] if zeroing else ["Raw", "Filtered"],
                          key=f"flt_s_show_{zeroing}", help=SIGNAL_HELP)
    if zeroing and not any("zeroed" in n for n in show):
        st.caption("⚠️ Baseline correction is on, but no '(zeroed)' series is selected, so its effect won't show.")
    if st.button("Run filter", type="primary", key="flt_s_run"):
        try:
            res = process_trace(x, y, s)
        except Exception as e:
            st.error(f"Processing failed: {e}")
            return
        st.session_state.current_trace = (res, dict(meta, settings=s))
    if st.session_state.current_trace is None:
        st.info("Click Run filter to see the result. To compare several channels or filter settings on one plot, "
                "use the Plot & Compare page.")
        return

    res, m = st.session_state.current_trace
    series = {"Raw": (res.y_raw, {"width": 1}, 0.6), "Raw (zeroed)": (res.y_raw_corrected, {"width": 1.5}, 1),
              "Filtered": (res.y_filtered, {"width": 1.5}, 1), "Filtered (zeroed)": (res.y_filtered_corrected, {"width": 1.5}, 1),
              "Raw baseline": (res.baseline_raw, {"width": 1, "dash": "dash"}, 1),
              "Filtered baseline": (res.baseline_filtered, {"width": 1, "dash": "dot"}, 1)}
    w0, w1 = ui.plot_window("flt_s_win", float(np.nanmin(res.x)), float(np.nanmax(res.x)))
    keep = (res.x >= w0) & (res.x <= w1)
    traces = [{"x": res.x[keep], "y": series[n][0][keep], "name": n, "line": series[n][1], "opacity": series[n][2]}
              for n in show]
    st.plotly_chart(time_plot(traces, trace_title(m["ycol"], m["file_name"]), "Time", "Strain (µε)",
                              max_points=ui.plot_points()), width="stretch")
    st.caption("For full detail on one event, narrow 'Show from / to' rather than drag-zooming. "
               "The sidebar's Plot detail setting controls speed vs detail.")
    st.write(f"Filter: `{res.filter_summary}`")
    st.write(f"Raw baseline: `{res.raw_baseline_summary}`")
    st.write(f"Filtered baseline: `{res.filtered_baseline_summary}`")
    if res.trim_used:
        st.info(f"Auto-retry trimmed {res.trim_used:.6g} s from the start.")

    ui.save_preset_box(m["settings"], "flt_s_preset")
    with st.expander("Reproducible code"):
        st.code(reproducible_code(m["settings"], m["file_name"], m["xcol"], m["ycol"]), language="python")
    c1, c2 = st.columns([3, 1])
    label = c1.text_input("Library label (used as the legend on Plot & Compare)",
                          value=trace_title(m["ycol"], m["file_name"]), key="flt_s_label")
    if c2.button("Add to session library", key="flt_s_save"):
        i = ui.next_id()
        st.session_state.library.append(make_item(i, f"{label} | #{i}", res,
                                                  {k: v for k, v in m.items() if k != "settings"}
                                                  | {"settings": m["settings"].to_preset(), "title": label.strip()}))
        st.success("Added. Session-library traces can be plotted on Plot & Compare and exported on Exports.")


# ============================================================================ batch
def _batch_store(store):
    recs = sorted(store.records, key=lambda r: (r["location"], r["date"], r["file_name"]))
    locs = sorted({r["location"] for r in recs})
    c1, c2 = st.columns([1, 3])
    sel_locs = c1.multiselect("Locations", locs, default=locs, key="flt_b_locs")
    labels = {store.record_label(r): r["id"] for r in recs if r["location"] in sel_locs}
    chosen = c2.multiselect("Records", list(labels), default=list(labels), key="flt_b_recs")
    if not chosen:
        st.warning("Select at least one record.")
        return
    all_ch = sorted({c for l in chosen for c in store.record(labels[l])["channels"]}, key=str.lower)
    chans = st.multiselect("Channels", all_ch, default=all_ch, key="flt_b_chans")

    st.subheader("Filter settings")
    s, preset = ui.settings_form("fltb", None, default_preset=RECOMMENDED_PRESET)
    st.subheader("Save")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Run name", value=preset if preset != "Manual / Custom" else "Custom", key="flt_b_name")
    active = c2.checkbox("Make this the active run", value=True, key="flt_b_active")
    replace = c3.checkbox("Replace an existing run with this name", value=True, key="flt_b_replace")
    st.caption("Results are saved in the Data store, so you never need to re-run this. Plot them on Plot & Compare, "
               "analyse them on Baseline Shift, and download them on Exports.")
    if st.button(f"Filter {len(chosen)} record(s) and save", type="primary", key="flt_b_go"):
        bar, log = st.progress(0.0), []
        for i, lab in enumerate(chosen, 1):
            bar.progress((i - 1) / len(chosen), text=f"Filtering {i} of {len(chosen)}: {lab}")
            try:
                _, rows = filter_record(store, labels[lab], chans, s, name.strip() or "Custom", active, replace,
                                        by=ui.user_name())
                log += rows
            except ValueError as e:
                log.append(dict(record=lab, channel="", status=f"failed: {e}"))
        bar.empty()
        ui.put_output("filter", log=pd.DataFrame(log))
    _show_log()


def _show_log():
    log = ui.get_output("filter", "log")
    if log is not None:
        ok = log["status"].str.startswith("ok")
        st.success(f"{ok.sum()} channel(s) filtered" + (f", {(~ok).sum()} failed." if (~ok).any() else "."))
        ui.show_table(log[~ok] if (~ok).any() else None)


def _batch_upload(store):
    ups = st.file_uploader("Upload CSVs or ZIPs of CSVs", type=["csv", "zip"], accept_multiple_files=True, key="flt_b_upload")
    if not ups:
        st.info("Upload files to begin. Large data sets are best added on the Data page first, then filtered from the store.")
        return
    sources = [s for up in ups for s in sources_from_upload(up)]
    if not sources:
        st.warning("No CSV files found.")
        return
    p = ui.preview(sources[0])
    if p is None:
        return
    df0, num0, guess0 = p
    c1, c2 = st.columns(2)
    with c1:
        tcol = ui.time_column_picker(df0, num0, guess0, "flt_b_tcol")
    cands = [c for c in num0 if c != tcol]
    chans = c2.multiselect("Channels", cands, default=cands, key="flt_b_uchans")

    dest = st.radio("Results", ["Save to the Data store (adds the raw files too)", "Download only (nothing is saved)"],
                    key="flt_b_dest", horizontal=True)
    to_store = dest.startswith("Save")
    rows = [dict(include=True, file=s.path, size_MB=round(s.size_mb, 1), location=s.location_guess,
                 date=s.date_guess or s.modified) for s in sources]
    table = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", key=f"flt_b_table_{len(rows)}",
                           disabled=["file", "size_MB"],
                           column_config={"include": st.column_config.CheckboxColumn("include", width="small"),
                                          "date": st.column_config.DateColumn("date")})
    todo = [(s, r) for s, r in zip(sources, table.to_dict("records")) if r["include"]]
    total_mb = sum(s.size_mb for s, _ in todo)
    if not to_store and total_mb > 1500:
        st.warning(f"About {total_mb:,.0f} MB of CSVs: download-only processing keeps everything in memory. "
                   "Saving to the Data store processes one file at a time instead.")
    window = None
    if not to_store and st.checkbox("Trim to a custom time window", key="flt_b_trim"):
        x0 = pd.to_numeric(df0[tcol], errors="coerce").dropna()
        window = ui.time_window("flt_b_trimwin", x0.min(), x0.max(), ("Trim start", "Trim end"))

    st.subheader("Filter settings")
    s, preset = ui.settings_form("fltbu", None, default_preset=RECOMMENDED_PRESET)

    if to_store:
        c1, c2 = st.columns(2)
        name = c1.text_input("Run name", value=preset if preset != "Manual / Custom" else "Custom", key="flt_bu_name")
        active = c2.checkbox("Make this the active run", value=True, key="flt_bu_active")
        st.caption("Temperatures can be added afterwards on Data → Records & temperatures.")
        if st.button(f"Add {len(todo)} file(s) and filter", type="primary", key="flt_bu_go", disabled=not todo):
            bar, log = st.progress(0.0), []
            for i, (src, row) in enumerate(todo, 1):
                bar.progress((i - 1) / len(todo), text=f"{i} of {len(todo)}: {src.file_name}")
                if pd.isna(row["date"]) or not str(row["location"] or "").strip():
                    log.append(dict(record=src.path, channel="", status="failed: location and date are required"))
                    continue
                loc, d = str(row["location"]).strip(), pd.Timestamp(row["date"]).date().isoformat()
                try:
                    rec = import_source(store, src, loc, d, tcol, by=ui.user_name())
                except DuplicateRecord:
                    rec = store.find_record(loc, d, src.file_name)
                except Exception as e:
                    log.append(dict(record=src.path, channel="", status=f"failed: {e}"))
                    continue
                try:
                    _, rows_ = filter_record(store, rec["id"], chans, s, name.strip() or "Custom", active, True,
                                             by=ui.user_name())
                    log += rows_
                except ValueError as e:
                    log.append(dict(record=src.path, channel="", status=f"failed: {e}"))
            bar.empty()
            ui.put_output("filter", log=pd.DataFrame(log))
        _show_log()
        return

    st.subheader("Downloads")
    c1, c2 = st.columns(2)
    want_zip = c1.checkbox("Filtered channel CSVs by date (ZIP: Time, Original, Filtered Corrected)", value=True, key="flt_bd_zip")
    want_hist = c1.checkbox("Processed time-history CSV (time + filtered corrected)", key="flt_bd_hist")
    want_freq = c2.checkbox("Frequency CSVs by channel (ZIP)", key="flt_bd_freq")
    want_lib = c2.checkbox("Add traces to the session library", key="flt_bd_lib",
                           help="Lets you plot and export them on the other pages during this session.")
    if want_freq:
        c1, c2, c3 = st.columns(3)
        fsig = c1.selectbox("Frequency signal", SIGNAL_NAMES, index=3, key="flt_bd_fsig")
        fbins = c2.number_input("Bins", 20, 1000, 300, step=10, key="flt_bd_fbins")
        fneg = c3.checkbox("Negative strain only", key="flt_bd_fneg")

    if st.button(f"Filter {len(todo)} file(s)", type="primary", key="flt_bd_go", disabled=not todo):
        multi_loc = len({r["location"] for _, r in todo}) > 1
        per_channel, cols, items, log = {c: [] for c in chans}, {}, [], []
        bar = st.progress(0.0)
        for i, (src, row) in enumerate(todo, 1):
            bar.progress((i - 1) / len(todo), text=f"{i} of {len(todo)}: {src.file_name}")
            d = row["date"]
            tag = (f"{row['location']} " if multi_loc and row["location"] else "") + \
                  (date_tag(pd.Timestamp(d).date().isoformat()) if not pd.isna(d) else Path(src.file_name).stem)
            try:
                df = src.read()
            except Exception as e:
                log.append(dict(record=src.path, channel="", status=f"failed: {e}"))
                continue
            for ch in chans:
                if tcol not in df.columns or ch not in df.columns:
                    log.append(dict(record=src.path, channel=ch, status="failed: column missing"))
                    continue
                try:
                    x, y = clean_xy(df, tcol, ch)
                    if window:
                        x, y = trim_arrays(x, y, *window)
                    r = process_trace(x, y, s)
                except Exception as e:
                    log.append(dict(record=src.path, channel=ch, status=f"failed: {e}"))
                    continue
                log.append(dict(record=src.path, channel=ch, status="ok" if not r.trim_used else f"ok, auto-trimmed {r.trim_used:.3g} s"))
                if want_zip:
                    per_channel[ch].append((tag, r.x, r.y_raw, r.y_filtered_corrected.astype(np.float32)))
                if want_hist:
                    cols.setdefault("time", pd.Series(r.x))
                    cols[f"{Path(src.file_name).stem}__{ch}__filtered_corrected"] = pd.Series(r.y_filtered_corrected)
                if want_freq or want_lib:
                    j = ui.next_id()
                    items.append(make_item(j, f"{preset} | {tag} | {ch} | batch | #{j}", r,
                                           dict(file_name=src.file_name, ycol=ch, xcol=tcol,
                                                title=trace_title(ch, src.file_name)), light=True))
            del df
        bar.empty()
        out = dict(zip=None, hist=None, freq=None)
        if want_zip:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for ch, res in per_channel.items():
                    if res:
                        frame = {}
                        for tag, x, raw, filt in res:
                            frame[f"{tag} Time"], frame[f"{tag} Original"], frame[f"{tag} Filtered Corrected"] = \
                                pd.Series(x), pd.Series(raw), pd.Series(filt)
                        zf.writestr(f"{safe_filename(ch)}.csv", pd.DataFrame(frame).to_csv(index=False))
                zf.writestr("Summary.csv", pd.DataFrame(log).to_csv(index=False))
            out["zip"] = buf.getvalue()
        if want_hist and cols:
            out["hist"] = pd.DataFrame(cols).to_csv(index=False).encode()
        if want_freq and items:
            out["freq"], _ = frequency_zip_by_channel(group_by_channel(items).items(), fsig, int(fbins), fneg)
        if want_lib:
            st.session_state.library += items
        ui.put_output("filter", log=pd.DataFrame(log), **out)

    _show_log()
    c1, c2, c3 = st.columns(3)
    if ui.get_output("filter", "zip"):
        c1.download_button("Filtered channel CSVs (ZIP)", ui.get_output("filter", "zip"), "filtered_channels_by_date.zip",
                           "application/zip", key="flt_bd_dl_zip")
    if ui.get_output("filter", "hist"):
        c2.download_button("Time-history CSV", ui.get_output("filter", "hist"), "processed_time_history.csv",
                           "text/csv", key="flt_bd_dl_hist")
    if ui.get_output("filter", "freq"):
        c3.download_button("Frequency CSVs (ZIP)", ui.get_output("filter", "freq"), "frequency_by_channel.zip",
                           "application/zip", key="flt_bd_dl_freq")


def render():
    st.title("Filter")
    store = ui.get_store()
    c1, c2 = st.columns(2)
    source = c1.radio("Data", ["Data store", "Upload files"], horizontal=True, key="flt_source",
                      index=0 if store.records else 1)
    mode = c2.radio("Mode", ["Single channel (preview)", "Batch (many files / channels)"], horizontal=True, key="flt_mode")
    if source == "Data store" and not store.records:
        st.info("The Data store is empty — add raw files on the Data page, or choose Upload files.")
        return
    if mode.startswith("Single"):
        _single(store, source)
    elif source == "Data store":
        _batch_store(store)
    else:
        _batch_upload(store)
