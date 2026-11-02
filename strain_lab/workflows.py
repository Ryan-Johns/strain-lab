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


def filter_record(store, rid: str, channels: Optional[List[str]], s: FilterSettings, name: str,
                  make_active=True, replace=True, by=""):
    """Filter a stored record's channels and save the result as a run. Returns (run or None, log rows)."""
    store.reload()
    rec = next((r for r in store.records if r["id"] == rid), None)
    if rec is None:
        raise ValueError("This recording was deleted (possibly by someone else).")
    use = [c for c in (channels or rec["channels"]) if c in rec["channels"]]
    try:
        t, raw = store.load_raw(rid, use)
    except FileNotFoundError:
        raise ValueError("This recording was deleted (possibly by someone else).") from None
    results, log = {}, []
    for ch in use:
        try:
            r = process_trace(t, raw[ch].to_numpy(float), s)
            results[ch] = store.run_entry(r)      # keep only what the run stores
            log.append(dict(record=store.record_label(rec), channel=ch,
                            status="ok" if not r.trim_used else f"ok, auto-trimmed {r.trim_used:.3g} s"))
        except Exception as e:
            log.append(dict(record=store.record_label(rec), channel=ch, status=f"failed: {e}"))
    if not results:
        return None, log
    settings = s.to_preset() | {"quiet_start": s.quiet_start, "quiet_end": s.quiet_end}
    run = store.save_run(rid, name, settings, results, make_active=make_active, replace_same_name=replace, by=by)
    windows = sorted(set(store.stats()["window_s"].astype(float))) or [10.0]
    store.put_stats(pd.concat([stats_rows(store, rec, run, w) for w in windows], ignore_index=True))
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
