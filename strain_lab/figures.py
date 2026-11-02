"""Plotly figures and the matplotlib channel-PNG export (no Streamlit)."""
import io
import zipfile
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import UNIQUE_COLORS, COLOR_OPTIONS, AUTO_COLOR_CYCLE
from .io_utils import safe_filename
from .library import chronological, short_label, signal

PERCENT_TICKS = [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]


def resolve_color(selection: str, index: int) -> str:
    c = COLOR_OPTIONS.get(selection)
    return AUTO_COLOR_CYCLE[index % len(AUTO_COLOR_CYCLE)] if c is None else c


DEFAULT_MAX_POINTS = 10_000


def decimate_minmax(x, y, max_points=DEFAULT_MAX_POINTS):
    """Keep the minimum and maximum of each of max_points/2 equal slices, in time order.

    At screen resolution the line looks identical to the full data (every peak and valley is kept),
    but the browser receives a few thousand points instead of hundreds of thousands.
    max_points=None or 0 returns the data unchanged.
    """
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if not max_points or n <= max_points:
        return x, y
    buckets = max(1, int(max_points) // 2)
    size = int(np.ceil(n / buckets))
    pad = buckets * size - n
    yy = np.concatenate([y, np.full(pad, np.nan)]).reshape(buckets, size)
    nan = np.isnan(yy)
    lo = np.argmin(np.where(nan, np.inf, yy), axis=1)
    hi = np.argmax(np.where(nan, -np.inf, yy), axis=1)
    base = np.arange(buckets) * size
    idx = np.unique(np.concatenate([base + lo, base + hi]))
    idx = idx[idx < n]
    return x[idx], y[idx]


def time_plot(traces: List[dict], title: str, x_title="Time", y_title="Strain", height=600,
              max_points=DEFAULT_MAX_POINTS) -> go.Figure:
    """Line plot of time histories, each line reduced to max_points (min/max per slice) for speed."""
    fig = go.Figure()
    for tr in traces:
        x, y = decimate_minmax(tr["x"], tr["y"], max_points)
        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", name=tr.get("name", ""),
                                   line=tr.get("line", {}), opacity=tr.get("opacity", 1.0)))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, template="plotly_white",
                      height=height, showlegend=True)
    return fig


def set_percent_log_ticks(fig, show_minor_grid=False, minor_color="lightgray", minor_width=0.5):
    fig.update_yaxes(type="log", tickmode="array", tickvals=PERCENT_TICKS,
                     ticktext=[f"{v:g}%" for v in PERCENT_TICKS])
    if show_minor_grid:
        fig.update_yaxes(minor=dict(showgrid=True, gridcolor=minor_color, gridwidth=float(minor_width), ticks=""))


# ----------------------------------------------------------------------------- publication-style figure
_LEGEND_ANCHORS = {
    "Top Right (inside)": (0.88, 0.95, "right", "top"),
    "Top Left (inside)": (0.04, 0.95, "left", "top"),
    "Bottom Right (inside)": (0.88, 0.07, "right", "bottom"),
    "Bottom Left (inside)": (0.04, 0.07, "left", "bottom"),
    "Outside Right": (1.01, 0.98, "left", "top"),
}


def _inside_legend(fig, labels, st: dict, curves: dict):
    """Draws a boxed legend inside the plot area (Plotly's own legend can't be boxed tightly inside)."""
    if not labels:
        return
    names = [str(curves[l]["name"]) for l in labels]
    box_w = min(0.34, max(0.18, 0.0065 * max(map(len, names)) + 0.085))
    row_h, pad_x, pad_y, line_len = 0.052, 0.012, 0.015, 0.040
    box_h = 2 * pad_y + len(labels) * row_h
    pos = st["legend_position"]
    x0 = 0.955 - box_w if "Right" in pos else 0.045
    y1 = 0.95 if "Top" in pos else 0.07 + box_h
    x1, y0 = x0 + box_w, y1 - box_h
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=x0, x1=x1, y0=y0, y1=y1, layer="above",
                  line=dict(color=st["legend_border_color"], width=float(st["legend_border_width"])),
                  fillcolor="white" if st["legend_background"] else "rgba(0,0,0,0)")
    for i, (label, name) in enumerate(zip(labels, names)):
        y = y1 - pad_y - row_h * (i + 0.5)
        c = curves[label]
        fig.add_shape(type="line", xref="paper", yref="paper", x0=x0 + pad_x, x1=x0 + pad_x + line_len,
                      y0=y, y1=y, layer="above", line=dict(color=c["color"], width=c["width"], dash=c["dash"]))
        fig.add_annotation(xref="paper", yref="paper", x=x0 + pad_x + line_len + 0.008, y=y, text=name,
                           showarrow=False, xanchor="left", yanchor="middle", align="left",
                           font=dict(size=int(st["legend_font_size"]), color=st.get("legend_font_color", "black")))


def apply_style(fig: go.Figure, st: dict, curves: Dict[str, dict]):
    """Publication styling for any figure. st = style settings; curves[name] = {name, color, width, dash}
    for the boxed inside-legend."""
    bold = lambda text, on: f"<b>{text}</b>" if on else text
    builtin = st["show_legend"] and st["legend_position"] == "Outside Right"
    lx, ly, xa, ya = _LEGEND_ANCHORS.get(st["legend_position"], _LEGEND_ANCHORS["Top Right (inside)"])
    title_font = {"size": int(st["axis_title_font_size"]), "color": st["title_color"]}
    fig.update_layout(
        template="plotly_white", font={"color": st["tick_color"]},
        title={"text": bold(st["title"], st["bold_title"]), "font": {"size": int(st["title_font_size"]), "color": st["title_color"]}},
        xaxis_title={"text": bold(st["x_title"], st["bold_axis_titles"]), "font": title_font},
        yaxis_title={"text": bold(st["y_title"], st["bold_axis_titles"]), "font": title_font},
        height=int(st["height"]), width=int(st["width"]) if st.get("fixed_width") else None, showlegend=builtin,
        legend={"font": {"size": int(st["legend_font_size"]), "color": st.get("legend_font_color", "black")},
                "x": lx, "y": ly, "xanchor": xa, "yanchor": ya,
                "bgcolor": "white" if st["legend_background"] else "rgba(0,0,0,0)",
                "bordercolor": st["legend_border_color"], "borderwidth": float(st["legend_border_width"])},
        plot_bgcolor=st["plot_bgcolor"], paper_bgcolor=st["paper_bgcolor"],
        margin=dict(l=65, r=150 if builtin else 18, t=50, b=55),
    )
    axis = dict(tickfont={"size": int(st["tick_font_size"]), "color": st["tick_color"]}, showgrid=st["show_grid"],
                gridcolor=st["grid_color"], gridwidth=float(st["grid_width"]), zeroline=False, mirror=True,
                showline=True, linecolor=st["outline_color"], linewidth=float(st["outline_width"]))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if st.get("log_x"):
        fig.update_xaxes(type="log")
    if st.get("percent_log_y"):
        set_percent_log_ticks(fig, st["minor_grid"], st["grid_color"], st["minor_grid_width"])
    elif st.get("log_y"):
        fig.update_yaxes(type="log")
    if st["show_legend"] and not builtin:
        _inside_legend(fig, list(curves), st, curves)
    return fig


# ----------------------------------------------------------------------------- channel PNG export
def channel_png_zip(groups, signal_choice: str, t_start: float, t_end: float, width=10.0,
                    height=6.0, dpi=300, max_points=None, collapse_duplicates=True, smooth_window=None):
    """One PNG per channel (all dates overlaid) over a shared time window. groups: iterable of (channel, items).
    max_points / smooth_window / collapse_duplicates affect the drawing only, never the data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if t_end <= t_start:
        raise ValueError("End time must be greater than start time.")
    buf, rows = io.BytesIO(), []
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for channel, group in groups:
            group = chronological(group)
            if len(group) > len(UNIQUE_COLORS):
                raise ValueError(f"{len(group)} traces requested but only {len(UNIQUE_COLORS)} unique colors are available.")
            fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
            plotted = 0
            for color, it in zip(UNIQUE_COLORS, group):
                x = np.asarray(it["x"], dtype=float)
                y = signal(it, signal_choice)
                m = np.isfinite(x) & np.isfinite(y) & (x >= t_start) & (x <= t_end)
                if not m.any():
                    continue
                order = np.argsort(x[m], kind="stable")
                xp, yp = x[m][order], y[m][order]
                if collapse_duplicates and len(xp) > 1:
                    g = pd.DataFrame({"x": xp, "y": yp}).groupby("x", as_index=False)["y"].mean()
                    xp, yp = g["x"].to_numpy(), g["y"].to_numpy()
                if smooth_window and len(yp) > 3:
                    w = int(max(3, smooth_window)) | 1
                    w = min(w, len(yp) if len(yp) % 2 else len(yp) - 1)
                    if w >= 3:
                        yp = pd.Series(yp).rolling(w, center=True, min_periods=1).mean().to_numpy()
                if max_points and len(xp) > max_points:
                    step = int(np.ceil(len(xp) / max_points))
                    xp, yp = xp[::step], yp[::step]
                ax.plot(xp, yp, linewidth=1.5, color=color, label=short_label(it.get("label", "")))
                plotted += 1
            if not plotted:
                plt.close(fig)
                rows.append({"channel": channel, "status": "skipped - no data in time window", "file": "", "traces_plotted": 0})
                continue
            ax.set(title=str(channel), xlabel="Time", ylabel="Strain (με)")
            ax.grid(True, linewidth=0.4, alpha=0.5)
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            fname = f"{safe_filename(channel)}.png"
            zf.writestr(fname, png.getvalue())
            rows.append({"channel": channel, "status": "created", "file": fname, "traces_plotted": plotted})
    return buf.getvalue(), pd.DataFrame(rows)
