"""Multi-step operations shared by several pages (no Streamlit)."""
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .baseline import stats_rows
from .io_utils import RawSource, numeric_columns
from .processing import process_trace
from .settings import FilterSettings
from .library import trace_title


def import_source(store, src: RawSource, location: str, date: str, time_col: str,
                  temp_recording=None, temp_sunrise=None, replace=False, by="") -> dict:
    """Read one raw CSV, store it, and compute its baseline statistics (10 s windows)."""
    df = src.read()
    if time_col not in df.columns:
        raise ValueError(f"no '{time_col}' column")
    chans = [c for c in numeric_columns(df) if c != time_col]
    rec = store.add_raw(df, src.file_name, location, date, time_col, chans, temp_recording, temp_sunrise,
                        replace=replace, by=by)
    del df
    store.put_stats(stats_rows(store, rec, None, 10.0))
    return rec


def release_memory():
    """Hand freed memory back to the system (matters on the 2.7 GB online server)."""
    import gc
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def filter_record(store, rid: str, channels: Optional[List[str]], s: FilterSettings, name: str,
                  make_active=True, replace=True, by=""):
    """Filter a stored record's channels and save the result as a run. Returns (run or None, log rows).

    Works one channel at a time and keeps each result in its stored (integer) form, so memory stays low
    however many channels a recording has. Statistics are computed from those same stored values."""
    from .baseline import compute_stats
    from .store import QUANTUM
    store.reload()
    rec = next((r for r in store.records if r["id"] == rid), None)
    if rec is None:
        raise ValueError("This recording was deleted (possibly by someone else).")
    use = [c for c in (channels or rec["channels"]) if c in rec["channels"]]
    try:
        t, _ = store.load_raw(rid, [])
    except FileNotFoundError:
        raise ValueError("This recording was deleted (possibly by someone else).") from None
    settings = s.to_preset() | {"quiet_start": s.quiet_start, "quiet_end": s.quiet_end}
    t_run = store.run_time({"settings": settings}, t)
    windows = sorted(set(store.stats()["window_s"].astype(float))) or [10.0]
    n = len(t)
    results, stats, log = {}, [], []
    for ch in use:
        try:
            _, raw = store.load_raw(rid, [ch])
            r = process_trace(t, raw[ch].to_numpy(float), s)
            del raw
            results[ch] = store.compact_entry(r, n)
            trim = r.trim_used
            del r
            y = results[ch]["q"].to_numpy(dtype=float, na_value=np.nan) * QUANTUM   # exactly what is stored
            one = pd.DataFrame({ch: y})
            stats += [compute_stats(t_run, one, w).assign(window_s=float(w)) for w in windows]
            del y, one
            log.append(dict(record=store.record_label(rec), channel=ch,
                            status="ok" if not trim else f"ok, auto-trimmed {trim:.3g} s"))
        except Exception as e:
            log.append(dict(record=store.record_label(rec), channel=ch, status=f"failed: {e}"))
    if not results:
        return None, log
    run = store.save_run(rid, name, settings, results, make_active=make_active, replace_same_name=replace, by=by)
    del results
    if stats:
        store.put_stats(pd.concat(stats, ignore_index=True).assign(
            record_id=rec["id"], basis=run["id"], source="computed"))
    release_memory()
    return run, log


def date_tag(iso_date: str) -> str:
    y, m, d = (int(p) for p in iso_date.split("-"))
    return f"{m}_{d}_{y % 100:02d}"


def run_items(store, run_id: str, channels: Optional[List[str]], next_id: Callable[[], int]) -> List[dict]:
    """Library-style trace items for a saved run (labels: 'run | L3 12_9_25 | channel | store | #id')."""
    run = store.run(run_id)
    rec = store.record(run["record_id"])
    when = f"{rec['location']} {date_tag(rec['date'])}"
    use = [c for c in (channels or run["channels"]) if c in run["channels"]]
    items = []
    for ch, tr in store.run_traces(run_id, use).items():
        i = next_id()
        info = tr.pop("info")
        items.append(dict(id=i, label=f"{run['name']} | {when} | {ch} | store | #{i}", **tr,
                          baseline_raw=None, baseline_filtered=None,
                          meta=dict(id=i, title=trace_title(ch, rec["file_name"]), file_name=rec["file_name"],
                                    ycol=ch, xcol=rec["time_col"],
                                    location=rec["location"], date=rec["date"], store_run=run_id,
                                    filter_summary=info["filter_summary"],
                                    raw_baseline_summary=info["raw_baseline_summary"],
                                    filtered_baseline_summary=info["filtered_baseline_summary"],
                                    settings=run["settings"])))
    return items
