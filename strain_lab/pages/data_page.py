"""Data: keep raw files, edit temperatures and notes, manage saved filtered runs and gauge types."""
import pandas as pd
import streamlit as st

from .. import ui
from ..baseline import GAUGE_TYPES, gauge_type
from ..figures import time_plot
from ..io_utils import sources_from_upload
from ..store import DuplicateRecord
from ..workflows import import_source


def _changed_rows(before: pd.DataFrame, after: pd.DataFrame) -> list:
    """Index labels of rows the user edited (so saving never overwrites other people's changes to other rows)."""
    b = before.reindex(index=after.index, columns=after.columns).astype(object).where(lambda d: d.notna(), None)
    a = after.astype(object).where(lambda d: d.notna(), None)
    return [i for i in after.index if any(str(x) != str(y) for x, y in zip(a.loc[i], b.loc[i]))]


def _add_raw(store):
    st.caption("Upload CSVs, or ZIPs of CSVs (any folder layout). Files are read one at a time, so large ZIPs are fine. "
               "Raw data is stored exactly as uploaded, compressed to about 1/8 of the CSV size, and never modified.")
    ups = st.file_uploader("Raw CSV or ZIP files", type=["csv", "zip"], accept_multiple_files=True, key="ds_add_files")
    if not ups:
        return
    sources = []
    for up in ups:
        try:
            sources += sources_from_upload(up)
        except Exception as e:
            st.error(f"{up.name}: {e}")
    if not sources:
        st.warning("No CSV files found in the upload.")
        return

    first = ui.preview(sources[0])
    if first is None:
        return
    df0, num0, guess0 = first
    c1, c2 = st.columns(2)
    with c1:
        tcol = ui.time_column_picker(df0, num0, guess0, "ds_add_tcol")
    default_loc = c2.text_input("Location where it can't be read from the file or ZIP name", value="", key="ds_add_defloc")

    temps = {(r["location"], r["date"]): (r.get("temp_recording"), r.get("temp_sunrise")) for r in store.records}
    rows = []
    for s in sources:
        loc = s.location_guess or default_loc
        d = s.date_guess
        tr, ts = temps.get((loc, d.isoformat() if d else ""), (None, None))
        stored = d is not None and store.find_record(loc, d.isoformat(), s.file_name) is not None
        rows.append(dict(add=not stored, file=s.path, zip=s.container, size_MB=round(s.size_mb, 1), location=loc,
                         date=d or s.modified, date_from="name" if d else ("file timestamp" if s.modified else "—"),
                         temp_recording=tr, temp_sunrise=ts, in_store="yes" if stored else ""))
    n_missing_date = sum(r["date_from"] != "name" for r in rows)
    st.markdown(f"**{len(rows)} CSV file(s) found.** Check location, date and temperatures (temperatures can also be "
                "added later on the next tab), then click **Add to store**.")
    if n_missing_date:
        st.warning(f"{n_missing_date} file(s) have no date in their name; their date was taken from the file timestamp "
                   "or left blank. Please check those.")
    table = st.data_editor(pd.DataFrame(rows), hide_index=True, key=f"ds_add_table_{len(rows)}", width="stretch",
                           disabled=["file", "zip", "size_MB", "date_from", "in_store"],
                           column_config={"add": st.column_config.CheckboxColumn("add", width="small"),
                                          "date": st.column_config.DateColumn("date"),
                                          "temp_recording": st.column_config.NumberColumn("Temp @ recording (°F)"),
                                          "temp_sunrise": st.column_config.NumberColumn("Temp @ sunrise, day prior (°F)")})
    replace = st.checkbox("Replace files that are already in the store", key="ds_add_replace",
                          help="Otherwise files already in the store (same location, date and file name) are skipped.")
    todo = [(s, r) for s, r in zip(sources, table.to_dict("records")) if r["add"]]
    if st.button(f"Add {len(todo)} file(s) to store", type="primary", key="ds_add_go", disabled=not todo):
        bar, log = st.progress(0.0), []
        num = lambda v: None if v is None or pd.isna(v) else float(v)
        for i, (src, row) in enumerate(todo, 1):
            bar.progress((i - 1) / len(todo), text=f"Reading {i} of {len(todo)}: {src.file_name}")
            if pd.isna(row["date"]) or not str(row["location"] or "").strip():
                log.append(dict(file=src.path, result="skipped: location and date are required"))
                continue
            try:
                rec = import_source(store, src, str(row["location"]).strip(), pd.Timestamp(row["date"]).date().isoformat(),
                                    tcol, num(row["temp_recording"]), num(row["temp_sunrise"]), replace=replace,
                                    by=ui.user_name())
                log.append(dict(file=src.path, result=f"added as {store.record_label(rec)} ({len(rec['channels'])} channels)"))
            except DuplicateRecord:
                log.append(dict(file=src.path, result="already in store — skipped"))
            except Exception as e:
                log.append(dict(file=src.path, result=f"failed: {e}"))
        bar.empty()
        ui.put_output("data", add_log=pd.DataFrame(log))
    log = ui.get_output("data", "add_log")
    if log is not None:
        st.success(f"{log['result'].str.startswith('added').sum()} of {len(log)} file(s) added.")
        ui.show_table(log)


def _records(store):
    recs = store.records_df()
    if recs.empty:
        st.info("The store is empty. Add raw files on the first tab.")
        return
    st.caption("Edit locations, dates, temperatures and notes, then click Save changes.")
    if "added_by" not in recs:
        recs["added_by"] = ""
    view = recs.set_index("id")[["location", "date", "file_name", "n_channels", "duration_s", "runs",
                                 "temp_recording", "temp_sunrise", "notes", "added_by"]].fillna({"added_by": ""})
    edited = st.data_editor(view, key="ds_rec_editor", width="stretch", hide_index=True,
                            disabled=["file_name", "n_channels", "duration_s", "runs", "added_by"],
                            column_config={"temp_recording": st.column_config.NumberColumn("Temp @ recording (°F)"),
                                           "temp_sunrise": st.column_config.NumberColumn("Temp @ sunrise, day prior (°F)"),
                                           "duration_s": st.column_config.NumberColumn("duration (s)", format="%.0f")})
    if st.button("Save changes", key="ds_rec_save"):
        # Only rows you changed are written, so edits someone else made meanwhile are kept.
        changed = _changed_rows(view, edited)
        with store.transaction():
            for rid in changed:
                if not any(r["id"] == rid for r in store.records):
                    continue
                row = edited.loc[rid]
                store.update_record(rid, location=str(row["location"]).strip(), date=str(row["date"])[:10],
                                    temp_recording=row["temp_recording"], temp_sunrise=row["temp_sunrise"],
                                    notes=row["notes"] or "")
        st.success(f"Saved {len(changed)} change(s).")
        st.rerun()

    with st.expander("View or download raw data"):
        labels = {store.record_label(r): r["id"] for r in recs.to_dict("records")}
        rid = labels[st.selectbox("Record", list(labels), key="ds_raw_rec")]
        rec = store.record(rid)
        chans = st.multiselect("Channels to plot", rec["channels"], default=rec["channels"][:1], key=f"ds_raw_ch_{rid}")
        if chans:
            t, raw = store.load_raw(rid, chans)
            st.plotly_chart(time_plot([{"x": t, "y": raw[c].to_numpy(), "name": c} for c in chans],
                                      store.record_label(rec), "Time", "Strain (µε)", max_points=ui.plot_points()),
                            width="stretch")
        if st.button("Prepare raw CSV download", key="ds_raw_prep"):
            t, raw = store.load_raw(rid)
            raw.insert(0, rec["time_col"], t)
            ui.put_output("data", raw_csv=raw.to_csv(index=False).encode(), raw_name=rec["file_name"])
        if ui.get_output("data", "raw_csv"):
            st.download_button("Download raw CSV", ui.get_output("data", "raw_csv"), ui.get_output("data", "raw_name"),
                               "text/csv", key="ds_raw_dl")

    with st.expander("Delete records"):
        st.caption("Deleting a record removes its raw file, its saved filtered runs and its statistics.")
        labels = {store.record_label(r): r["id"] for r in recs.to_dict("records")}
        doomed = st.multiselect("Records to delete", list(labels), key="ds_del_recs")
        sure = st.checkbox("I understand this cannot be undone", key="ds_del_sure")
        if st.button("Delete selected records", disabled=not (doomed and sure), key="ds_del_go"):
            with store.transaction():
                for lab in doomed:
                    store.delete_record(labels[lab])
            st.success(f"Deleted {len(doomed)} record(s).")
            st.rerun()


def _runs(store):
    if not store.runs:
        st.info("No saved filtered runs yet. On the Filter page, tick 'Save to the Data store'.")
        return
    st.caption("Each record has one active run, used by Baseline Shift (filtered data) and as the default elsewhere. "
               "Rename runs or change the active one here, then click Save.")
    rows = []
    for run in store.runs:
        rec = store.record(run["record_id"])
        first = next(iter(run["summary"].values()), {})
        rows.append(dict(id=run["id"], location=rec["location"], date=rec["date"], file=rec["file_name"], name=run["name"],
                         active=bool(run.get("active")), channels=len(run["channels"]),
                         auto_trimmed=sum(v.get("n_trimmed", 0) > 0 for v in run["summary"].values()),
                         filter=first.get("filter_summary", ""), created=run["created"],
                         by=run.get("created_by", "")))
    df = pd.DataFrame(rows).sort_values(["location", "date", "name"])
    edited = st.data_editor(df.set_index("id"), key="ds_runs_editor", width="stretch", hide_index=True,
                            disabled=["location", "date", "file", "channels", "auto_trimmed", "filter", "created", "by"])
    if st.button("Save", key="ds_runs_save"):
        before = df.set_index("id")
        with store.transaction():
            for rid in _changed_rows(before, edited):
                if not any(r["id"] == rid for r in store.runs):
                    continue
                row = edited.loc[rid]
                if row["name"] != before.loc[rid, "name"]:
                    store.rename_run(rid, str(row["name"]))
                if row["active"] and not before.loc[rid, "active"]:
                    store.set_active(rid)
        st.rerun()
    with st.expander("Delete runs"):
        labels = {store.run_label(r): r["id"] for r in store.runs}
        doomed = st.multiselect("Runs to delete", list(labels), key="ds_runs_del_pick")
        sure = st.checkbox("Confirm (raw data is not affected)", key="ds_runs_sure")
        if st.button("Delete selected runs", disabled=not (doomed and sure), key="ds_runs_del"):
            with store.transaction():
                for lab in doomed:
                    store.delete_run(labels[lab])
            st.rerun()


def _gauge_types(store):
    chans = sorted({c for r in store.records for c in r["channels"]}, key=str.lower)
    if not chans:
        st.info("No channels yet.")
        return
    st.caption("Assigned from the channel name: BC bottom chord, DG diagonal, MS rebar, Z studs, and R…_xH / xV / xS "
               "gusset plate horizontal / vertical / slanted. Change any that are wrong.")
    df = pd.DataFrame({"channel": chans, "type": [gauge_type(c, store.gauge_types) for c in chans]})
    edited = st.data_editor(df, hide_index=True, key="ds_gt_editor", disabled=["channel"],
                            column_config={"type": st.column_config.SelectboxColumn("type", options=GAUGE_TYPES, required=True)})
    if st.button("Save gauge types", key="ds_gt_save"):
        store.set_gauge_types({r["channel"]: r["type"] for r in edited.to_dict("records")
                               if r["type"] != gauge_type(r["channel"])})
        st.success("Saved.")


def _maintenance(store):
    from .. import cloud
    m = cloud._MIRROR
    if m is None:
        return
    with st.expander("Online storage"):
        st.caption(f"Data is kept in the private Hugging Face dataset **{m.repo_id}**. Every save adds a version to its "
                   f"history ({m.commits} this session). After a few thousand, compacting the history keeps it fast; "
                   "your current data is not changed, but older versions are removed.")
        if st.button("Compact history", key="ds_compact"):
            with st.spinner("Compacting…"):
                try:
                    m.compact_history()
                    st.success("Done.")
                except Exception as e:
                    st.error(f"Could not compact: {e}")


def render():
    st.title("Data")
    if ui.hosted():
        store = ui.get_store()
        st.caption("Shared online data: everyone signed in to this app sees the same files, runs, temperatures and presets.")
    else:
        c1, c2 = st.columns([3, 1])
        root = c1.text_input("Store folder", value=st.session_state.get("store_root", ui.default_store_root()),
                             key="ds_root_input", help="All saved data lives in this folder. Back it up like any project folder.")
        if root != st.session_state.get("store_root"):
            st.session_state["store_root"] = root
        store = ui.get_store()
        c2.metric("Store size", f"{store.size_mb():,.0f} MB")
    locs = sorted({r["location"] for r in store.records})
    st.caption(f"{len(store.records)} raw files ({', '.join(locs) or 'none'}) · {len(store.runs)} saved filtered runs")
    _maintenance(store)
    tabs = st.tabs(["Add raw files", "Records & temperatures", "Saved filtered runs", "Gauge types"])
    with tabs[0]:
        _add_raw(store)
    with tabs[1]:
        _records(store)
    with tabs[2]:
        _runs(store)
    with tabs[3]:
        _gauge_types(store)
