"""Frequency-of-strain curves (percent of samples per strain bin) and their CSV/ZIP exports."""
import io
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import RAINBOW_NO_PINK
from .io_utils import read_csv_bytes, csv_names_in_zip, safe_filename
from .library import signal, chronological, label_sort_key

FREQ_COLUMNS = ["label", "strain_bin_center", "frequency_percent"]


def compute_frequency_curve(values, n_bins: int, x_min=None, x_max=None):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([]), np.array([]), np.array([])
    x_min = float(np.min(v)) if x_min is None else x_min
    x_max = float(np.max(v)) if x_max is None else x_max
    if x_max <= x_min:
        return np.array([]), np.array([]), np.array([])
    counts, edges = np.histogram(v, bins=np.linspace(x_min, x_max, int(n_bins) + 1))
    return 0.5 * (edges[:-1] + edges[1:]), counts / len(v) * 100.0, counts


def smooth_curve(y, window: int):
    window = max(1, int(window))
    if len(y) == 0 or window == 1:
        return y
    return np.convolve(y, np.ones(window) / window, mode="same")


def _values(item: dict, signal_choice: str, negative_only: bool) -> np.ndarray:
    y = signal(item, signal_choice)
    y = y[np.isfinite(y)]
    return y[y <= 0] if negative_only else y


def build_frequency_df(items: List[dict], signal_choice: str, n_bins: int, negative_only: bool = False,
                       smooth_window: int = 1, common_range: bool = True,
                       x_limits: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
    """One curve per item. Bins span x_limits, else the common range over all items, else each item's own range."""
    x_min = x_max = None
    if x_limits is not None:
        x_min, x_max = x_limits
    elif common_range:
        lo = hi = None
        for it in items:          # running min/max: never concatenates traces
            y = _values(it, signal_choice, negative_only)
            if len(y):
                lo = y.min() if lo is None else min(lo, y.min())
                hi = y.max() if hi is None else max(hi, y.max())
        if lo is not None:
            x_min, x_max = float(lo), float(hi)
    frames = []
    for it in items:
        y = _values(it, signal_choice, negative_only)
        centers, freq, _ = compute_frequency_curve(y, n_bins, x_min, x_max)
        frames.append(pd.DataFrame({"label": it["label"], "strain_bin_center": centers,
                                    "frequency_percent": smooth_curve(freq, smooth_window)}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FREQ_COLUMNS)


def annotate(df: pd.DataFrame, items: List[dict]) -> pd.DataFrame:
    """Add source_file and channel columns from the items' metadata."""
    meta = {it["label"]: it.get("meta", {}) for it in items}
    df = df.copy()
    df["source_file"] = df["label"].map(lambda l: meta.get(l, {}).get("file_name", ""))
    df["channel"] = df["label"].map(lambda l: meta.get(l, {}).get("ycol", ""))
    return df


def peak_shift(df: pd.DataFrame, label: str) -> float:
    """Shift that puts a curve's most frequent strain at x = 0."""
    part = df[df["label"] == label]
    f = pd.to_numeric(part["frequency_percent"], errors="coerce")
    x = pd.to_numeric(part["strain_bin_center"], errors="coerce")
    ok = f.notna() & x.notna()
    return 0.0 if not ok.any() else -float(x[ok].loc[f[ok].idxmax()])


def center_at_peak(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    out["strain_bin_center"] = pd.to_numeric(out["strain_bin_center"], errors="coerce")
    for label in out["label"].dropna().unique():
        mask = out["label"] == label
        out.loc[mask, "strain_bin_center"] += peak_shift(out, label)
    return out


def add_chronological_colors(df: pd.DataFrame) -> pd.DataFrame:
    """plot_color/plot_order by date: oldest red through purple, no repeats, no pink."""
    if df is None or len(df) == 0 or "label" not in df.columns:
        return df
    labels = sorted(df["label"].dropna().unique(), key=label_sort_key)
    if len(labels) > len(RAINBOW_NO_PINK):
        raise ValueError(f"{len(labels)} traces are present, but only {len(RAINBOW_NO_PINK)} unique "
                         "non-pink rainbow colors are available.")
    out = df.copy()
    out["plot_color"] = out["label"].map({l: RAINBOW_NO_PINK[i] for i, l in enumerate(labels)})
    out["plot_order"] = out["label"].map({l: i for i, l in enumerate(labels)})
    return out


def frequency_zip_by_channel(groups, signal_choice: str = "Filtered (zeroed)", n_bins: int = 300,
                             negative_only: bool = False, common_range: bool = True):
    """One '<channel>_frequency.csv' per channel. groups: iterable of (channel, items) — a generator keeps
    only one channel in memory. Returns (zip_bytes, combined_df)."""
    buf, frames = io.BytesIO(), []
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for channel, group in groups:
            df = build_frequency_df(group, signal_choice, n_bins, negative_only, common_range=common_range)
            if len(df) == 0:
                continue
            df = annotate(df, group)
            zf.writestr(f"{safe_filename(channel)}_frequency.csv", df.to_csv(index=False))
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FREQ_COLUMNS)
    return buf.getvalue(), combined


def read_frequency_zip(existing_zip: bytes) -> dict:
    """{channel: DataFrame} from a ZIP of channel frequency CSVs."""
    existing = {}
    with zipfile.ZipFile(io.BytesIO(existing_zip)) as zin:
        for name in csv_names_in_zip(zin):
            channel = Path(name).stem
            if channel.lower().endswith("_frequency"):
                channel = channel[:-len("_frequency")]
            existing[channel] = read_csv_bytes(zin.read(name))
    return existing


def update_frequency_zip(existing_zip: bytes, groups, signal_choice: str, n_bins: int, negative_only: bool,
                         common_range: bool):
    """Append new curves (peak-centered, chronologically colored) to an existing channel ZIP.
    groups: iterable of (channel, items) holding only the traces to append."""
    existing = read_frequency_zip(existing_zip)
    buf, rows, done = io.BytesIO(), [], set()

    def write(zout, channel, old, new_items):
        out, appended = (old.copy() if old is not None else None), 0
        new_items = chronological(new_items)
        if new_items:
            new_df = build_frequency_df(new_items, signal_choice, n_bins, negative_only, common_range=common_range)
            if len(new_df):
                new_df = annotate(center_at_peak(new_df), new_items)
                if old is not None:
                    cols = list(old.columns) + [c for c in new_df.columns if c not in old.columns]
                    new_df = pd.concat([old.reindex(columns=cols), new_df.reindex(columns=cols)], ignore_index=True)
                out, appended = add_chronological_colors(new_df), len(new_items)
        if out is None:
            return
        fname = f"{safe_filename(channel)}.csv"
        zout.writestr(fname, out.to_csv(index=False))
        rows.append({"channel": channel, "existing_rows": 0 if old is None else len(old),
                     "new_traces_appended": appended, "updated_rows": len(out), "output_file": fname})

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for channel, items in groups:
            write(zout, channel, existing.get(channel), items)
            done.add(channel)
        for channel, old in existing.items():
            if channel not in done:
                write(zout, channel, old, [])
    return buf.getvalue(), pd.DataFrame(rows).sort_values("channel") if rows else pd.DataFrame(rows)
