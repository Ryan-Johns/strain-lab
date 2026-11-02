"""Baseline-shift statistics, gauge-type classification and temperature sensitivity."""
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

GAUGE_TYPES = ["Bottom chord", "Diagonal", "Rebar", "Studs", "Gusset plate – horizontal",
               "Gusset plate – vertical", "Gusset plate – slanted", "Other"]

METRICS = {
    "Whole-record median": "median",
    "Initial baseline (median of first N s)": "initial",
    "End baseline (median of last N s)": "end",
    "Drift (end − initial)": "drift",
    "Whole-record mean": "mean",
}


def gauge_type(channel: str, overrides: Optional[Dict[str, str]] = None) -> str:
    """BC_ bottom chord, DG_ diagonal, MS_ rebar, Z<n>_ studs, R…_ gusset plate by the H/V/S letter."""
    if overrides and channel in overrides:
        return overrides[channel]
    c = str(channel).upper()
    if c.startswith("BC"):
        return "Bottom chord"
    if c.startswith("DG"):
        return "Diagonal"
    if c.startswith("MS"):
        return "Rebar"
    if re.match(r"^Z\d", c):
        return "Studs"
    if c.startswith("R"):
        suffix = c.split("_", 1)[1] if "_" in c else ""
        orient = suffix[1:2] if len(suffix) >= 2 else ""
        return {"H": "Gusset plate – horizontal", "V": "Gusset plate – vertical",
                "S": "Gusset plate – slanted"}.get(orient, "Other")
    return "Other"


def compute_stats(t: np.ndarray, data: pd.DataFrame, window_s: float = 10.0) -> pd.DataFrame:
    """Per channel: median of the first/last window_s seconds, whole-record median and mean.
    NaNs (e.g. samples trimmed by auto-retry) are ignored, so the window starts at the first valid sample."""
    t = np.asarray(t, dtype=float)
    rows = []
    for ch in data.columns:
        y = data[ch].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(t)
        if not ok.any():
            continue
        tt, yy = t[ok], y[ok]
        first = yy[tt < tt[0] + window_s]
        last = yy[tt > tt[-1] - window_s]
        rows.append(dict(channel=ch, initial=float(np.median(first)), end=float(np.median(last)),
                         median=float(np.median(yy)), mean=float(np.mean(yy)), n=int(len(yy))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- analysis table
def analysis_table(store, basis: str = "raw", window_s: float = 10.0) -> pd.DataFrame:
    """Long table for plotting: one row per (record, channel) with temperatures, gauge type and metrics.

    basis 'raw' uses the original data; 'filtered' uses each record's active filtered run.
    """
    st = store.stats()
    recs = store.records_df()
    if st.empty or recs.empty:
        return pd.DataFrame()
    parts = []
    for _, rec in recs.iterrows():
        if basis == "raw":
            b = "raw"
        else:
            run = store.active_run(rec["id"])
            if run is None:
                continue
            b = run["id"]
        s = st[(st["record_id"] == rec["id"]) & (st["basis"] == b)]
        s = s[np.isclose(s["window_s"].astype(float), window_s)]
        if s.empty:
            continue
        s = s.assign(location=rec["location"], date=rec["date"], file_name=rec["file_name"],
                     temp_recording=rec["temp_recording"], temp_sunrise=rec["temp_sunrise"])
        parts.append(s)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["drift"] = df["end"] - df["initial"]
    df["gauge_type"] = df["channel"].map(lambda c: gauge_type(c, store.gauge_types))
    df["date"] = pd.to_datetime(df["date"])
    for c in ("temp_recording", "temp_sunrise"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["location", "channel", "date"]).reset_index(drop=True)


def temperature_sensitivity(df: pd.DataFrame, value_col: str, temp_col: str) -> pd.DataFrame:
    """Least-squares slope (µε per °F) of value vs temperature for each location/channel."""
    rows = []
    for (loc, ch), g in df.groupby(["location", "channel"]):
        gtype = g["gauge_type"].iloc[0] if "gauge_type" in g else gauge_type(ch)
        g = g[[temp_col, value_col]].dropna()
        if len(g) < 3 or g[temp_col].nunique() < 2:
            continue
        slope, icept = np.polyfit(g[temp_col], g[value_col], 1)
        pred = slope * g[temp_col] + icept
        ss_res = float(((g[value_col] - pred) ** 2).sum())
        ss_tot = float(((g[value_col] - g[value_col].mean()) ** 2).sum())
        rows.append(dict(location=loc, channel=ch, gauge_type=gtype, slope_ue_per_F=slope,
                         intercept=icept, r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan, n_dates=len(g)))
    return pd.DataFrame(rows)


def ensure_stats(store, window_s: float = 10.0, progress=None) -> int:
    """Compute any missing statistics for this window: raw data and every saved run. Returns how many were added."""
    st = store.stats()
    have = set(zip(st["record_id"], st["basis"], st["window_s"].astype(float))) if not st.empty else set()
    jobs = []
    for rec in store.records:
        if not rec.get("raw_file"):
            continue
        if (rec["id"], "raw", float(window_s)) not in have:
            jobs.append((rec, None))
        jobs += [(rec, run) for run in store.runs_for(rec["id"]) if (rec["id"], run["id"], float(window_s)) not in have]
    frames = []
    for i, (rec, run) in enumerate(jobs, 1):
        frames.append(stats_rows(store, rec, run, window_s))
        if progress:
            progress(i / len(jobs))
    if frames:
        store.put_stats(pd.concat(frames, ignore_index=True))
    return len(jobs)


def stats_rows(store, rec: dict, run: Optional[dict], window_s: float) -> pd.DataFrame:
    t, raw = store.load_raw(rec["id"])
    if run is None:
        s, basis = compute_stats(t, raw, window_s), "raw"
    else:
        s, basis = compute_stats(store.run_time(run, t), store.load_run(run["id"]), window_s), run["id"]
    return s.assign(record_id=rec["id"], basis=basis, window_s=float(window_s), source="computed")


OUTLIER_METHODS = ["Off", "Automatic (robust)", "Fixed limit (µε)"]


def _line_outliers(days: np.ndarray, v: np.ndarray, fixed: bool, k: float, limit_ue: float, floor_ue: float):
    """Iteratively drop the point farthest from the line through its neighbours until none exceed the limit."""
    n = len(v)
    keep = np.ones(n, bool)
    while keep.sum() >= 3 and (~keep).sum() < max(1, n // 3):
        idx = np.flatnonzero(keep)
        t, w = days[idx], v[idx]
        resid = np.empty(len(w))
        # interior: distance from the straight line joining the previous and next dates
        frac = (t[1:-1] - t[:-2]) / np.where(t[2:] > t[:-2], t[2:] - t[:-2], 1.0)
        resid[1:-1] = w[1:-1] - (w[:-2] + frac * (w[2:] - w[:-2]))
        # ends: distance from the single neighbour
        resid[0], resid[-1] = w[0] - w[1], w[-1] - w[-2]
        typical_step = float(np.median(np.abs(np.diff(w))))
        thr = float(limit_ue) if fixed else max(float(k) * typical_step, float(floor_ue))
        j = int(np.argmax(np.abs(resid)))
        if abs(resid[j]) <= thr:
            break
        keep[idx[j]] = False
    return ~keep


def flag_outliers(df: pd.DataFrame, value_col: str, method: str, k: float = 5.0, limit_ue: float = 100.0,
                  floor_ue: float = 25.0) -> pd.Series:
    """True for points that jump away from their own line (location/channel, in date order).

    Each point is compared with the straight line through its neighbouring dates (end points: with their one
    neighbour). The farthest point beyond the limit is removed, the rest re-checked, and so on (at most a third
    of a line's points). Limit:
      - Automatic: k × the line's typical change between consecutive dates, but at least floor_ue
      - Fixed limit: limit_ue microstrain
    Gradual (seasonal) change is not flagged; isolated jumps are.
    """
    flags = pd.Series(False, index=df.index)
    if method == "Off" or df.empty:
        return flags
    for _, g in df.dropna(subset=[value_col]).sort_values("date").groupby(["location", "channel"]):
        if len(g) < 3:
            continue
        days = (g["date"] - g["date"].iloc[0]).dt.total_seconds().to_numpy() / 86400.0
        out = _line_outliers(days, g[value_col].to_numpy(float), method.startswith("Fixed"), k, limit_ue, floor_ue)
        flags.loc[g.index[out]] = True
    return flags
