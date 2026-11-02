"""Saved-trace library: building items, labels, sorting and trimming (no Streamlit).

A trace item is a plain dict so existing session code keeps working:
  {"id", "label", "x", "y_raw", "y_raw_corrected", "y_filtered", "y_filtered_corrected",
   "baseline_raw", "baseline_filtered", "meta": {...}}
"""
import re
from typing import Dict, List, Optional

import numpy as np

from .config import ARRAY_KEYS, SIGNALS
from .io_utils import date_tuple
from .processing import TraceResult


def make_item(item_id: int, label: str, result: TraceResult, meta: dict, light: bool = False) -> dict:
    """light=True (batch) keeps only the corrected arrays. The raw/filtered keys point at the
    same arrays instead of holding copies, which cuts memory per trace by about two thirds."""
    item = {"id": item_id, "label": label, "x": result.x, "meta": dict(meta, id=item_id)}
    if light:
        item.update(y_raw=result.y_raw_corrected, y_raw_corrected=result.y_raw_corrected,
                    y_filtered=result.y_filtered_corrected, y_filtered_corrected=result.y_filtered_corrected,
                    baseline_raw=None, baseline_filtered=None)
    else:
        item.update(y_raw=result.y_raw, y_raw_corrected=result.y_raw_corrected,
                    y_filtered=result.y_filtered, y_filtered_corrected=result.y_filtered_corrected,
                    baseline_raw=result.baseline_raw, baseline_filtered=result.baseline_filtered)
    return item


def signal(item: dict, choice: str) -> np.ndarray:
    return np.asarray(item[SIGNALS[choice]], dtype=float)


def channel_of(item: dict) -> str:
    return str(item.get("meta", {}).get("ycol", "") or "unknown_channel")


def label_parts(label: str) -> List[str]:
    return [p.strip() for p in str(label).split("|")]


def short_label(label: str) -> str:
    """'date/file | channel' from 'prefix | date/file | channel | filter | …'."""
    parts = [p for p in label_parts(label) if not re.fullmatch(r"(?:save|batch)?#\d+|store|batch", p)] or [str(label)]
    if len(parts) >= 3:
        return f"{parts[1]} | {parts[2]}"
    return parts[1] if len(parts) == 2 else parts[0]


def trace_title(channel: str, file_name: str) -> str:
    """The title used on the Filter plot, also used for legends: 'BC_T — 12_9_25.csv'."""
    return f"{channel} — {file_name}"


def display_title(item: dict) -> str:
    """Legend text for a library trace: its title, or a cleaned-up label for older traces."""
    return item.get("meta", {}).get("title") or short_label(item.get("label", ""))


def label_sort_key(label: str):
    return (date_tuple(label), short_label(label).lower(), str(label).lower())


def item_sort_key(item: dict):
    meta = item.get("meta", {})
    channel = str(meta.get("ycol", "") or "")
    if not channel:
        parts = label_parts(item.get("label", ""))
        channel = parts[2] if len(parts) >= 3 else (parts[1] if len(parts) == 2 else "")
    try:
        trace_id = int(item.get("id", 0))
    except (TypeError, ValueError):
        trace_id = 0
    return (channel.lower(), str(meta.get("file_name", "")).lower(), trace_id, str(item.get("label", "")).lower())


def sort_items(items: List[dict]) -> List[dict]:
    return sorted(items, key=item_sort_key)


def chronological(items: List[dict]) -> List[dict]:
    return sorted(items, key=lambda it: label_sort_key(it.get("label", "")))


def channels(items: List[dict]) -> List[str]:
    return sorted({channel_of(it) for it in items}, key=str.lower)


def group_by_channel(items: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for it in items:
        groups.setdefault(channel_of(it), []).append(it)
    return dict(sorted(groups.items(), key=lambda kv: kv[0].lower()))


def time_span(items: List[dict]):
    """(min, max) time over all items, or None."""
    lo, hi = [], []
    for it in items:
        x = np.asarray(it["x"], dtype=float)
        x = x[np.isfinite(x)]
        if len(x):
            lo.append(float(x.min())); hi.append(float(x.max()))
    return (min(lo), max(hi)) if lo else None


def common_window(items: List[dict]):
    """Overlapping (start, end) shared by all items, or None."""
    lo, hi = [], []
    for it in items:
        x = np.asarray(it["x"], dtype=float)
        x = x[np.isfinite(x)]
        if len(x) >= 2:
            lo.append(float(x.min())); hi.append(float(x.max()))
    if not lo or min(hi) <= max(lo):
        return None
    return max(lo), min(hi)


def trim_item(item: dict, start: float, end: float) -> Optional[dict]:
    if end <= start:
        return None
    x = np.asarray(item["x"], dtype=float)
    mask = np.isfinite(x) & (x >= start) & (x <= end)
    if not np.any(mask):
        return None
    out = dict(item)
    for k in ARRAY_KEYS:
        if out.get(k) is not None:
            out[k] = np.asarray(out[k])[mask]
    out["meta"] = dict(item.get("meta", {}), trim_start=float(start), trim_end=float(end))
    return out
