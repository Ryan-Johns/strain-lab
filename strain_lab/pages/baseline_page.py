"""Baseline Shift: track how gauge baselines move between recording dates and with temperature."""
import io

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from .. import ui
from ..baseline import (METRICS, GAUGE_TYPES, OUTLIER_METHODS, analysis_table, ensure_stats,
                        temperature_sensitivity, flag_outliers)

MODES = {"As measured": "", "Change from first date": "Change in ", "Centered on zero (each line's median)": "Centered "}
GROUPINGS = {
    "Channel — locations overlaid": ("channel", "location"),
    "Gauge type — locations overlaid": ("gauge_type", "series"),
    "Location × gauge type": ("loc_type", "channel"),
    "Location — all gauges": ("location", "channel"),
    "Everything on one chart": (None, "series"),
}
X_AXES = {"Date": "date", "Temp @ recording (°F)": "temp_recording", "Temp @ sunrise, day prior (°F)": "temp_sunrise"}
PALETTE = pc.qualitative.Dark24 + pc.qualitative.Light24
LOC_COLORS = ["#1f6fd1", "#c0392b", "#2e8b57", "#7d5ba6", "#b8860b", "#17becf"]
DASHES = ["solid", "dash", "dot", "dashdot", "longdash"]


def _custom(g):
    return np.stack([g["location"], g["channel"], g["date"].dt.strftime("%Y-%m-%d"), g["temp_recording"].astype(object),
                     g["record_id"], g["hidden_by"]], axis=-1)


HOVER = ("%{customdata[0]} %{customdata[1]}<br>%{customdata[2]}<br>value %{y:.2f} µε"
         "<br>temp %{customdata[3]} °F<extra></extra>")


def _chart(df, title, series_col, x_col, y_col, y_label, height, temp_overlay, fit, color_of, dash_of, show_hidden):
    by_date = x_col == "date"
    fig = make_subplots(specs=[[{"secondary_y": bool(by_date and temp_overlay)}]])
    vis = df[df["hidden_by"] == ""]
    for name, g in vis.groupby(series_col, sort=True):
        g = g.dropna(subset=[x_col, y_col]).sort_values(x_col)
        if g.empty:
            continue
        color, dash = color_of(g.iloc[0]), dash_of(g.iloc[0])
        fig.add_trace(go.Scatter(x=g[x_col], y=g[y_col], name=str(name), mode="lines+markers" if by_date else "markers",
                                 line=dict(color=color, dash=dash, width=2), marker=dict(size=7, color=color),
                                 customdata=_custom(g), hovertemplate=HOVER))
        if fit and not by_date and len(g) >= 2 and g[x_col].nunique() >= 2:
            k, b = np.polyfit(g[x_col], g[y_col], 1)
            xs = np.array([g[x_col].min(), g[x_col].max()])
            fig.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=color, dash="dot", width=1.2)))
    hid = df[df["hidden_by"] != ""].dropna(subset=[x_col, y_col])
    if show_hidden and len(hid):
        fig.add_trace(go.Scatter(x=hid[x_col], y=hid[y_col], name="hidden points", mode="markers",
                                 marker=dict(symbol="x", size=9, color="rgba(120,120,120,0.6)"),
                                 customdata=_custom(hid),
                                 hovertemplate=HOVER.replace("<extra>", "<br>hidden (%{customdata[5]})<extra>")))
    if by_date and temp_overlay:
        for i, (loc, g) in enumerate(df.drop_duplicates(["location", "date"]).groupby("location")):
            g = g.dropna(subset=["temp_recording"]).sort_values("date")
            if len(g):
                fig.add_trace(go.Scatter(x=g["date"], y=g["temp_recording"], name=f"Temp {loc}", mode="lines+markers",
                                         line=dict(color="gray", dash=DASHES[i % len(DASHES)], width=1),
                                         marker=dict(symbol="x", size=6, color="gray"), hoverinfo="skip"),
                              secondary_y=True)
        fig.update_yaxes(title_text="Temperature (°F)", secondary_y=True, showgrid=False)
    fig.update_layout(title=title, height=height, template="plotly_white", margin=dict(l=60, r=20, t=50, b=50),
                      legend=dict(font=dict(size=10)), hovermode="closest", clickmode="event+select")
    fig.update_yaxes(title_text=y_label, secondary_y=False)
    fig.update_xaxes(title_text={"date": "Date", "temp_recording": "Temp @ recording (°F)",
                                 "temp_sunrise": "Temp @ sunrise, day prior (°F)"}[x_col])
    return fig


def _excel(df: pd.DataFrame) -> bytes:
    """One sheet per location laid out like the workbook: date + temps, then one column per channel, per metric."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for loc, g in df.groupby("location"):
            row = 0
            for title, col in (("INITIAL BASELINE", "initial"), ("END BASELINE", "end"),
                               ("LEVEL OF DRIFT - END MINUS INITIAL", "drift"), ("WHOLE-RECORD MEDIAN", "median")):
                wide = g.pivot_table(index=["date", "temp_sunrise", "temp_recording"], columns="channel", values=col,
                                     aggfunc="first", dropna=False).reset_index()
                wide["date"] = wide["date"].dt.date
                pd.DataFrame([[title]]).to_excel(xw, sheet_name=loc[:31], startrow=row, index=False, header=False)
                wide.to_excel(xw, sheet_name=loc[:31], startrow=row + 1, index=False)
                row += len(wide) + 4
    return buf.getvalue()


def render():
    st.title("Baseline Shift")
    store = ui.get_store()
    if not store.records:
        st.info("No data yet. Add raw files on the Data page first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    metric_name = c1.selectbox("Value", list(METRICS), key="bs_metric")
    basis = c2.selectbox("Data", ["Original (raw)", "Filtered (active run)"], key="bs_basis")
    window = c3.number_input("N seconds (initial/end windows)", 1.0, 600.0, 10.0, step=5.0, key="bs_window")
    mode = c4.selectbox("Vertical position", list(MODES), key="bs_mode",
                        help="Display only — the stored values never change.\n\n"
                             "- **As measured**: the value itself.\n"
                             "- **Change from first date**: each line starts at zero on its first shown date.\n"
                             "- **Centered on zero**: each line is shifted so its median is zero, to compare lines "
                             "with very different offsets on one scale.")

    bar = st.empty()
    added = ensure_stats(store, float(window), progress=lambda f: bar.progress(f, text="Computing baseline statistics…"))
    bar.empty()
    if added:
        st.toast(f"Computed statistics for {added} file(s) at N = {window:g} s.")

    df = analysis_table(store, "raw" if basis.startswith("Original") else "filtered", float(window))
    if df.empty:
        st.info("No statistics for this selection. Filtered data needs a saved run (Filter page → save to the Data store).")
        return
    y_col = METRICS[metric_name]

    with st.expander("Filters", expanded=True):
        c1, c2 = st.columns(2)
        locs = sorted(df["location"].unique())
        sel_locs = c1.multiselect("Locations", locs, default=locs, key="bs_locs")
        types = [t for t in GAUGE_TYPES if t in set(df["gauge_type"])]
        sel_types = c2.multiselect("Gauge types", types, default=types[:1], key="bs_types",
                                   help="Start with one type; add more as needed.")
        sub = df[df["location"].isin(sel_locs) & df["gauge_type"].isin(sel_types)]
        chans = sorted(sub["channel"].unique())
        sel_ch = st.multiselect("Channels (leave empty for all)", chans, key="bs_chans")
        if sel_ch:
            sub = sub[sub["channel"].isin(sel_ch)]
        dates = sorted(sub["date"].dt.date.unique())
        if len(dates) > 1:
            d0, d1 = st.select_slider("Date range", options=dates, value=(dates[0], dates[-1]), key="bs_dates")
            sub = sub[(sub["date"].dt.date >= d0) & (sub["date"].dt.date <= d1)]
    if sub.empty:
        st.warning("Nothing matches these filters.")
        return
    sub = sub.copy().sort_values("date")

    with st.expander("Outliers and hidden points", expanded=False):
        c1, c2, c3 = st.columns(3)
        method = c1.selectbox("Hide outliers automatically", OUTLIER_METHODS, key="bs_out_method",
                              help="Compares each point with the straight line through its neighbouring dates on the "
                                   "same line. Gradual (seasonal) change is kept; single-date jumps are hidden.\n\n"
                                   "- **Automatic**: hides a point that jumps more than *sensitivity* × the line's typical "
                                   "change between dates.\n- **Fixed limit**: hides a point more than the limit (µε) "
                                   "away.\n\nHidden points are left out of charts, trend lines and the temperature table, "
                                   "never deleted.")
        k = c2.number_input("Sensitivity (× typical change)", 2.0, 50.0, 5.0, step=0.5, key="bs_out_k",
                            disabled=not method.startswith("Auto"), help="Larger = only bigger jumps are hidden.")
        limit = c3.number_input("Fixed limit (µε)", 1.0, 100000.0, 100.0, step=10.0, key="bs_out_lim",
                                disabled=not method.startswith("Fixed"))
        show_hidden = st.checkbox("Show hidden points as faint ×", value=True, key="bs_show_hidden")
        st.caption("**Click a point** on any chart (or use the box/lasso tool in the chart's toolbar to pick several), "
                   "then use **Hide selected** below the charts. Click a × to bring it back.")

    hidden_you, kept = store.hidden_set(), store.kept_set()
    pair = list(zip(sub["record_id"], sub["channel"]))
    by_you = np.array([p in hidden_you for p in pair], bool)
    auto = flag_outliers(sub[~by_you], y_col, method, k=k, limit_ue=limit).reindex(sub.index, fill_value=False).to_numpy().copy()
    auto &= np.array([p not in kept for p in pair], bool)
    sub["hidden_by"] = np.where(by_you, "you", np.where(auto, "auto", ""))

    # Vertical position: one offset per line, from its visible points (display only).
    vis = sub[sub["hidden_by"] == ""]
    if mode.startswith("Change"):
        off = vis.dropna(subset=[y_col]).groupby(["location", "channel"])[y_col].first()
    elif mode.startswith("Centered"):
        off = vis.groupby(["location", "channel"])[y_col].median()
    else:
        off = None
    if off is None:
        sub["value"] = sub[y_col]
    else:
        sub["value"] = sub[y_col] - pd.Series(list(zip(sub["location"], sub["channel"])), index=sub.index).map(off).astype(float)
    sub["series"] = sub["location"] + " · " + sub["channel"]
    sub["loc_type"] = sub["location"] + " — " + sub["gauge_type"]

    st.subheader("Charts")
    c1, c2, c3, c4 = st.columns(4)
    grouping = c1.selectbox("One chart per", list(GROUPINGS), key="bs_group")
    x_name = c2.selectbox("X axis", list(X_AXES), key="bs_x")
    ncols = c3.selectbox("Charts per row", [1, 2, 3], index=1, key="bs_ncols")
    height = c4.number_input("Chart height", 250, 1200, 420, step=50, key="bs_height")
    c1, c2 = st.columns(2)
    temp_overlay = c1.checkbox("Overlay temperature (date axis)", value=True, key="bs_temp", disabled=X_AXES[x_name] != "date")
    fit = c2.checkbox("Trend lines (temperature axis)", value=True, key="bs_fit", disabled=X_AXES[x_name] == "date")

    x_col = X_AXES[x_name]
    if x_col != "date" and sub[x_col].isna().all():
        st.warning("No temperatures entered yet — add them on Data → Records & temperatures.")
        return
    group_col, series_col = GROUPINGS[grouping]
    loc_list = sorted(sub["location"].unique())
    ch_list = sorted(sub["channel"].unique())
    loc_color = {l: LOC_COLORS[i % len(LOC_COLORS)] for i, l in enumerate(loc_list)}
    ch_color = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(ch_list)}
    loc_dash = {l: DASHES[i % len(DASHES)] for i, l in enumerate(loc_list)}
    if series_col == "location":           # one channel per chart: color = location
        color_of, dash_of = (lambda r: loc_color[r["location"]]), (lambda r: "solid")
    elif series_col == "series":           # several locations per chart: color = channel, dash = location
        color_of, dash_of = (lambda r: ch_color[r["channel"]]), (lambda r: loc_dash[r["location"]])
    else:
        color_of, dash_of = (lambda r: ch_color[r["channel"]]), (lambda r: "solid")

    y_label = MODES[mode] + metric_name.split(" (")[0].lower() + " (µε)"
    if group_col is None:
        groups = [("All selected gauges", sub)]
    else:
        order = GAUGE_TYPES if group_col == "gauge_type" else None
        keys = sorted(sub[group_col].unique(), key=(lambda k: order.index(k)) if order else None)
        groups = [(k, sub[sub[group_col] == k]) for k in keys]
    if len(groups) > 60:
        st.warning(f"{len(groups)} charts — showing the first 60. Narrow the filters to see the rest.")
        groups = groups[:60]
    ver = st.session_state.get("bs_sel_ver", 0)
    selected = {}
    for i in range(0, len(groups), ncols):
        cols = st.columns(ncols)
        for col, (title, g) in zip(cols, groups[i:i + ncols]):
            with col:
                ev = st.plotly_chart(_chart(g, str(title), series_col, x_col, "value", y_label, height, temp_overlay,
                                            fit, color_of, dash_of, show_hidden),
                                     width="stretch", key=f"bs_chart_{ver}_{grouping}_{title}",
                                     on_select="rerun", selection_mode=("points", "box", "lasso"))
                for p in (ev.selection.points if ev and ev.selection else []):
                    cd = p.get("customdata")
                    if cd is not None and len(cd) >= 6:
                        selected[(cd[4], cd[1])] = cd
    if selected:
        to_hide = [k for k, cd in selected.items() if cd[5] == ""]
        to_show = [k for k, cd in selected.items() if cd[5] != ""]
        auto_sel = [k for k, cd in selected.items() if cd[5] == "auto"]
        names = ", ".join(f"{cd[0]} {cd[1]} {cd[2]}" for cd in list(selected.values())[:6]) + (" …" if len(selected) > 6 else "")
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        c1.markdown(f"**Selected:** {names}")
        if c2.button(f"Hide selected ({len(to_hide)})", disabled=not to_hide, type="primary", key="bs_hide"):
            store.hide_points(to_hide)
            st.session_state["bs_sel_ver"] = ver + 1
            st.rerun()
        if c3.button(f"Show again ({len(to_show)})", disabled=not to_show, key="bs_unhide"):
            store.show_points(to_show, keep=auto_sel)
            st.session_state["bs_sel_ver"] = ver + 1
            st.rerun()
        if c4.button("Clear selection", key="bs_clear_sel"):
            st.session_state["bs_sel_ver"] = ver + 1
            st.rerun()
    n_you, n_auto = int((sub["hidden_by"] == "you").sum()), int((sub["hidden_by"] == "auto").sum())
    if n_you or n_auto or store.kept_points:
        c1, c2 = st.columns([4, 1])
        c1.caption(f"Hidden here: {n_you} by you, {n_auto} automatically. Hidden points are never deleted.")
        if c2.button("Show all points again", key="bs_reset_pts",
                     help="Clears every point you hid or kept (automatic hiding still applies if switched on)."):
            store.reset_points()
            st.session_state["bs_sel_ver"] = ver + 1
            st.rerun()

    st.subheader("Temperature sensitivity")
    tcol = "temp_sunrise" if x_col == "temp_sunrise" else "temp_recording"
    sens = temperature_sensitivity(sub[sub["hidden_by"] == ""], "value", tcol)
    if sens.empty:
        st.caption("Needs at least three dates with temperatures per gauge.")
    else:
        st.caption(f"Least-squares slope of the selected value against {'sunrise' if tcol == 'temp_sunrise' else 'recording'} "
                   "temperature, per location and gauge. For reference, free thermal strain of steel is about 6.5 µε/°F.")
        sens = sens.sort_values(["gauge_type", "channel", "location"])
        st.dataframe(sens.round({"slope_ue_per_F": 3, "intercept": 1, "r2": 3}), hide_index=True, width="stretch")

    with st.expander("Data table and downloads"):
        cols = ["location", "date", "temp_sunrise", "temp_recording", "channel", "gauge_type", "initial", "end",
                "drift", "median", "mean", "value", "hidden_by", "file_name"]
        table = sub[cols].sort_values(["location", "channel", "date"]).rename(columns={"hidden_by": "hidden"})
        st.dataframe(table, hide_index=True, width="stretch")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download table (CSV)", table.to_csv(index=False).encode(), "baseline_shift_data.csv",
                           "text/csv", key="bs_dl_csv")
        c2.download_button("Download workbook-style Excel", _excel(sub[sub["hidden_by"] == ""]), "baseline_shift_summary.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="bs_dl_xlsx")
        if not sens.empty:
            c3.download_button("Download temperature sensitivity (CSV)", sens.to_csv(index=False).encode(),
                               "temperature_sensitivity.csv", "text/csv", key="bs_dl_sens")
