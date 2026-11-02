"""Exports: filtered CSVs by channel, frequency CSVs (new or appended), channel PNGs and per-recording CSVs."""
import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from .. import ui
from ..config import SIGNAL_NAMES
from ..figures import channel_png_zip
from ..frequency import frequency_zip_by_channel, update_frequency_zip
from ..io_utils import safe_filename
from ..library import sort_items, channels as lib_channels, channel_of, group_by_channel, label_parts, time_span
from ..workflows import run_items, date_tag


def _store_selection(store):
    """(run_ids, channels) chosen from the store, or (None, None)."""
    recs = sorted(store.records, key=lambda r: (r["location"], r["date"]))
    locs = sorted({r["location"] for r in recs})
    c1, c2 = st.columns([1, 3])
    sel_locs = c1.multiselect("Locations", locs, default=locs, key="ex_locs")
    labels = {store.record_label(r): r["id"] for r in recs if r["location"] in sel_locs}
    chosen = c2.multiselect("Recordings", list(labels), default=list(labels), key="ex_recs")
    names = sorted({r["name"] for r in store.runs})
    which = st.selectbox("Filtered run to use", ["Active run of each recording"] + names, key="ex_run")
    run_ids = []
    for lab in chosen:
        rid = labels[lab]
        run = store.active_run(rid) if which.startswith("Active") else \
            next((r for r in store.runs_for(rid) if r["name"] == which), None)
        if run is not None:
            run_ids.append(run["id"])
    missing = len(chosen) - len(run_ids)
    if missing:
        st.caption(f"{missing} recording(s) have no matching filtered run and are left out.")
    if not run_ids:
        st.info("No filtered runs selected. Filter recordings on the Filter page first.")
        return None, None
    chans = sorted({c for i in run_ids for c in store.run(i)["channels"]}, key=str.lower)
    sel = st.multiselect("Channels", chans, default=chans, key="ex_chans")
    return run_ids, sel


def render():
    st.title("Exports")
    store = ui.get_store()
    source = st.radio("Data", ["Data store (saved filtered runs)", "Session library"], horizontal=True, key="ex_source",
                      index=0 if store.runs else 1)

    if source.startswith("Data store"):
        if not store.runs:
            st.info("No saved filtered runs yet.")
            return
        run_ids, chans = _store_selection(store)
        if not run_ids or not chans:
            return
        multi_loc = len({store.record(store.run(i)["record_id"])["location"] for i in run_ids}) > 1

        def groups():
            """One channel at a time, so memory use stays small however many recordings are selected."""
            for ch in chans:
                items = [it for i in run_ids if ch in store.run(i)["channels"] for it in run_items(store, i, [ch], ui.next_id)]
                yield ch, items

        def tag(it):
            m = it["meta"]
            return (f"{m['location']} " if multi_loc else "") + date_tag(m["date"])
        span = None
    else:
        lib = sort_items(st.session_state.library)
        if not lib:
            st.info("The session library is empty.")
            return
        all_ch = lib_channels(lib)
        chans = st.multiselect("Channels", all_ch, default=all_ch, key="ex_lchans")
        labels = st.multiselect("Traces", [it["label"] for it in lib if channel_of(it) in chans],
                                default=[it["label"] for it in lib if channel_of(it) in chans], key="ex_ltraces")
        items_sel = [it for it in lib if it["label"] in set(labels)]
        if not items_sel:
            return
        groups = lambda: iter(group_by_channel(items_sel).items())
        tag = lambda it: label_parts(it["label"])[1] if len(label_parts(it["label"])) > 1 else it["label"]
        span = time_span(items_sel)

    tabs = st.tabs(["Filtered CSVs by channel", "Frequency CSVs", "Update existing frequency CSVs", "Channel PNGs",
                    "One CSV per recording"])

    with tabs[0]:
        st.caption("One CSV per channel; for each date: Time, Original and Filtered Corrected columns, in date order.")
        if st.button("Build ZIP", type="primary", key="ex_fz_go"):
            buf, n = io.BytesIO(), 0
            bar = st.progress(0.0)
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for k, (ch, items) in enumerate(groups(), 1):
                    bar.progress(k / max(len(chans), 1), text=f"{ch}")
                    if not items:
                        continue
                    cols = {}
                    for it in sorted(items, key=lambda it: it["meta"].get("date", "") or it["label"]):
                        t = tag(it)
                        cols[f"{t} Time"] = pd.Series(it["x"])
                        cols[f"{t} Original"] = pd.Series(it["y_raw"])
                        cols[f"{t} Filtered Corrected"] = pd.Series(np.round(it["y_filtered_corrected"], 4))
                    zf.writestr(f"{safe_filename(ch)}.csv", pd.DataFrame(cols).to_csv(index=False))
                    n += 1
            bar.empty()
            ui.put_output("exports", fzip=buf.getvalue())
            st.success(f"Built {n} channel file(s).")
        if ui.get_output("exports", "fzip"):
            st.download_button("Download filtered CSVs (ZIP)", ui.get_output("exports", "fzip"),
                               "filtered_channels_by_date.zip", "application/zip", key="ex_fz_dl")

    with tabs[1]:
        st.caption("One frequency CSV per channel (percent of samples in each strain bin), every selected date as a curve.")
        c1, c2, c3 = st.columns(3)
        sig = c1.selectbox("Signal", SIGNAL_NAMES, index=3, key="ex_fq_sig")
        bins = c2.number_input("Bins", 20, 1000, 300, step=10, key="ex_fq_bins")
        neg = c3.checkbox("Negative strain only", key="ex_fq_neg")
        if st.button("Build frequency ZIP", type="primary", key="ex_fq_go"):
            with st.spinner("Building…"):
                zb, combined = frequency_zip_by_channel(groups(), sig, int(bins), neg)
            ui.put_output("exports", fq=zb, fq_combined=combined.to_csv(index=False).encode())
        if ui.get_output("exports", "fq"):
            c1, c2 = st.columns(2)
            c1.download_button("Download frequency CSVs (ZIP)", ui.get_output("exports", "fq"), "frequency_by_channel.zip",
                               "application/zip", key="ex_fq_dl")
            c2.download_button("Download combined CSV", ui.get_output("exports", "fq_combined"),
                               "frequency_combined.csv", "text/csv", key="ex_fq_dl2")

    with tabs[2]:
        st.caption("Upload your existing ZIP of channel frequency CSVs; the selected dates are appended to each channel's "
                   "file, centered at their peak, and every file is recolored by date from red to purple (no pink).")
        existing = st.file_uploader("Existing channel frequency CSV ZIP", type=["zip"], key="ex_up_zip")
        c1, c2, c3 = st.columns(3)
        usig = c1.selectbox("Signal", SIGNAL_NAMES, index=3, key="ex_up_sig")
        ubins = c2.number_input("Bins", 20, 1000, 300, step=10, key="ex_up_bins")
        uneg = c3.checkbox("Negative strain only", key="ex_up_neg")
        ucommon = st.checkbox("Same strain range for the new curves within each channel", value=True, key="ex_up_common")
        if st.button("Build updated ZIP", type="primary", key="ex_up_go", disabled=existing is None):
            try:
                zb, summary = update_frequency_zip(existing.getvalue(), groups(), usig, int(ubins), uneg, ucommon)
                ui.put_output("exports", up=zb, up_summary=summary)
            except Exception as e:
                st.error(f"Could not update: {e}")
        ui.show_table(ui.get_output("exports", "up_summary"))
        if ui.get_output("exports", "up"):
            st.download_button("Download updated ZIP", ui.get_output("exports", "up"), "updated_frequency_channel_csvs.zip",
                               "application/zip", key="ex_up_dl")

    with tabs[3]:
        st.caption("One PNG per channel with every selected date overlaid, over the same time window.")
        lo, hi = span or (0.0, 3600.0)
        c1, c2, c3 = st.columns(3)
        psig = c1.selectbox("Signal", ["Filtered (zeroed)", "Raw (zeroed)"], key="ex_png_sig")
        t0 = c2.number_input("Start time (s)", value=float(lo), format="%.3f", key="ex_png_t0")
        t1 = c3.number_input("End time (s)", value=float(hi), format="%.3f", key="ex_png_t1")
        c1, c2, c3, c4 = st.columns(4)
        w = c1.number_input("Width (in)", 2.0, 30.0, 10.0, step=0.5, key="ex_png_w")
        h = c2.number_input("Height (in)", 2.0, 30.0, 6.0, step=0.5, key="ex_png_h")
        dpi = c3.number_input("DPI", 72, 600, 300, step=25, key="ex_png_dpi")
        maxpts = c4.number_input("Max points per line (0 = all)", 0, 500000, 50000, step=10000, key="ex_png_max",
                                 help="Drawing only; makes PNGs of hour-long records much faster.")
        dedupe = st.checkbox("Collapse repeated time stamps (drawing only)", value=True, key="ex_png_dedupe")
        if st.button("Build PNG ZIP", type="primary", key="ex_png_go"):
            try:
                with st.spinner("Drawing…"):
                    zb, summary = channel_png_zip(groups(), psig, t0, t1, w, h, int(dpi), int(maxpts) or None, dedupe)
                ui.put_output("exports", png=zb, png_summary=summary)
            except Exception as e:
                st.error(f"Could not build PNGs: {e}")
        ui.show_table(ui.get_output("exports", "png_summary"))
        if ui.get_output("exports", "png"):
            st.download_button("Download PNG ZIP", ui.get_output("exports", "png"), "channel_pngs.zip", "application/zip",
                               key="ex_png_dl")

    with tabs[4]:
        if not source.startswith("Data store"):
            st.caption("Available for saved runs only.")
        else:
            st.caption("One CSV per recording: time plus the filtered + corrected value of each selected channel.")
            if st.button("Build ZIP", key="ex_rc_go"):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for rid in run_ids:
                        run = store.run(rid)
                        rec = store.record(run["record_id"])
                        t, _ = store.load_raw(rec["id"], [])
                        df = store.load_run(rid, [c for c in chans if c in run["channels"]]).round(3)
                        df.insert(0, "time", store.run_time(run, t))
                        zf.writestr(f"{rec['location']}_{rec['date']}_{run['name']}.csv".replace(" ", "_"),
                                    df.to_csv(index=False))
                ui.put_output("exports", rc=buf.getvalue())
            if ui.get_output("exports", "rc"):
                st.download_button("Download ZIP", ui.get_output("exports", "rc"), "filtered_recordings.zip",
                                   "application/zip", key="ex_rc_dl")
