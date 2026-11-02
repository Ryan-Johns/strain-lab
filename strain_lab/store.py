"""Persistent data store: raw files, saved filtered runs, temperatures and baseline statistics.

Layout on disk (default folder: strain_lab_data next to app.py):
    catalog.json              records (one per raw file), runs (saved filtered results), gauge-type overrides,
                              shared presets, hidden Baseline Shift points
    raw/<record_id>.parquet   raw data exactly as uploaded (time column + numeric channels)
    filtered/<run_id>.parquet filtered + corrected channels, stored as int32 in units of 0.001 µε
    stats.parquet             baseline statistics (initial/end/median/mean) per record, basis and channel

Raw files are never modified. Deleting a run leaves its raw file alone; deleting a record removes
the raw file and every run made from it. Nothing here imports Streamlit.

Several people can use one store at the same time (e.g. the hosted app): every change runs inside
`transaction()`, which takes a lock, re-reads the catalog from disk, applies the change and writes it back,
so one person's change never overwrites another's.

Hooks (set by strain_lab.cloud when the store is mirrored to Hugging Face):
    on_write(relpath)   a file was written         on_delete(relpath)  a file was deleted
    fetch(relpath)      make sure a file is present locally before reading it
"""
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .processing import TraceResult

QUANTUM = 0.001          # stored resolution of filtered data, µε (raw data resolution is 0.01 µε)
STATS_COLUMNS = ["record_id", "basis", "channel", "window_s", "initial", "end", "median", "mean", "n", "source"]
RECORD_FIELDS = ["id", "location", "date", "file_name", "raw_file", "time_col", "channels", "n_samples",
                 "duration_s", "temp_recording", "temp_sunrise", "notes", "source", "added", "added_by"]


class DuplicateRecord(ValueError):
    pass


# ---------------------------------------------------------------------- hooks and locks
def _noop(_rel):
    return None


HOOKS = {"on_write": _noop, "on_delete": _noop, "fetch": _noop}
_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(root: Path) -> threading.RLock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(root), threading.RLock())


def _atomic_parquet(df: pd.DataFrame, path: Path, **kw):
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, **kw)
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


class DataStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        (self.root / "raw").mkdir(parents=True, exist_ok=True)
        (self.root / "filtered").mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / "catalog.json"
        self.stats_path = self.root / "stats.parquet"
        self._lock = _lock_for(self.root)
        self._depth = 0
        self._load()

    # ------------------------------------------------------------------ catalog I/O
    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _load(self):
        with self._lock:
            cat = json.loads(self.catalog_path.read_text()) if self.catalog_path.exists() else {}
        self.records: List[dict] = cat.get("records", [])
        self.runs: List[dict] = cat.get("runs", [])
        self.gauge_types: Dict[str, str] = cat.get("gauge_types", {})
        # Baseline Shift point overrides, as [record_id, channel] pairs: hidden by the user, or kept despite auto-outlier.
        self.hidden_points: List[list] = cat.get("hidden_points", [])
        self.kept_points: List[list] = cat.get("kept_points", [])
        self.presets: Dict[str, dict] = cat.get("presets", {})
        self.app_lock: dict = cat.get("app_lock", {})   # {"by", "key", "until", "note"}: one person has the app

    def reload(self):
        """Re-read the catalog (pick up changes made by other people)."""
        self._load()

    def _write_catalog(self):
        cat = {"version": 1, "records": self.records, "runs": self.runs, "gauge_types": self.gauge_types,
               "hidden_points": self.hidden_points, "kept_points": self.kept_points, "presets": self.presets,
               "app_lock": self.app_lock}
        tmp = self.catalog_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cat, indent=1, default=str))
        os.replace(tmp, self.catalog_path)        # atomic: never leaves a half-written catalog
        HOOKS["on_write"]("catalog.json")

    @contextmanager
    def transaction(self):
        """Lock, re-read the catalog, run the block, write the catalog. Nested transactions join the outer one.

            with store.transaction():
                store.update_record(...)
        """
        with self._lock:
            outer = self._depth == 0
            if outer:
                self._load()
            self._depth += 1
            try:
                yield self
            except BaseException:
                self._depth -= 1
                if outer:
                    self._load()          # discard the half-made change
                raise
            self._depth -= 1
            if outer:
                self._write_catalog()

    def save(self):
        """Kept for compatibility: inside a transaction the catalog is written when it ends."""
        if self._depth == 0:
            with self._lock:
                self._write_catalog()

    # ------------------------------------------------------------------ records
    def record(self, rid: str) -> dict:
        return next(r for r in self.records if r["id"] == rid)

    @staticmethod
    def record_label(r: dict) -> str:
        return f"{r['location']} · {r['date']} · {r['file_name']}"

    def find_record(self, location: str, date: str, file_name: str) -> Optional[dict]:
        return next((r for r in self.records if r["location"] == location and r["date"] == date
                     and r["file_name"] == file_name), None)

    def add_raw(self, df: pd.DataFrame, file_name: str, location: str, date: str, time_col: str,
                channels: List[str], temp_recording=None, temp_sunrise=None, notes="", replace=False, by="") -> dict:
        # Write the (large) raw file outside the lock, then register it in a short transaction.
        rid = f"{location}_{date}_{_short_id()}".replace(" ", "_")
        data = df[[time_col] + list(channels)]
        if not all(pd.api.types.is_numeric_dtype(data[c]) for c in data.columns):
            data = data.apply(pd.to_numeric, errors="coerce")
        raw_path = self.root / "raw" / f"{rid}.parquet"
        _atomic_parquet(data, raw_path, compression="zstd")
        t = data[time_col].to_numpy(float)
        try:
            with self.transaction():
                return self._register_raw(rid, raw_path, t, file_name, location, date, time_col, channels,
                                          len(data), temp_recording, temp_sunrise, notes, replace, by)
        except BaseException:
            raw_path.unlink(missing_ok=True)
            raise

    def _register_raw(self, rid, raw_path, t, file_name, location, date, time_col, channels, n,
                      temp_recording, temp_sunrise, notes, replace, by) -> dict:
        old = self.find_record(location, date, file_name)
        if old is not None:
            if not replace:
                raise DuplicateRecord(f"{location} {date} {file_name} is already in the store.")
            keep = {k: old.get(k) for k in ("temp_recording", "temp_sunrise", "notes")}
            self.delete_record(old["id"], save=False)
            temp_recording = keep["temp_recording"] if temp_recording is None else temp_recording
            temp_sunrise = keep["temp_sunrise"] if temp_sunrise is None else temp_sunrise
            notes = notes or keep["notes"] or ""
        rec = dict(id=rid, location=location, date=date, file_name=file_name, raw_file=f"raw/{rid}.parquet",
                   time_col=time_col, channels=list(channels), n_samples=int(n),
                   duration_s=float(np.nanmax(t) - np.nanmin(t)) if len(t) else 0.0,
                   temp_recording=temp_recording, temp_sunrise=temp_sunrise, notes=notes or "",
                   source="raw file", added=_now(), added_by=by or "")
        self.records.append(rec)
        HOOKS["on_write"](self._rel(raw_path))
        return rec

    def update_record(self, rid: str, **fields):
        """Change location, date, temperatures or notes. Use inside `with store.transaction():`."""
        rec = self.record(rid)
        for k, v in fields.items():
            if k in ("location", "date", "temp_recording", "temp_sunrise", "notes"):
                rec[k] = None if (isinstance(v, float) and np.isnan(v)) else v

    def delete_record(self, rid: str, save=True):
        with self.transaction():
            rec = next((r for r in self.records if r["id"] == rid), None)
            if rec is None:                       # already deleted by someone else
                return
            for run in [r for r in self.runs if r["record_id"] == rid]:
                self.delete_run(run["id"])
            if rec.get("raw_file"):
                (self.root / rec["raw_file"]).unlink(missing_ok=True)
                HOOKS["on_delete"](rec["raw_file"])
            self.records = [r for r in self.records if r["id"] != rid]
            self.hidden_points = [p for p in self.hidden_points if p[0] != rid]
            self.kept_points = [p for p in self.kept_points if p[0] != rid]
            self._drop_stats(record_id=rid)

    # ------------------------------------------------------------------ Baseline Shift point overrides
    def hidden_set(self) -> set:
        return {tuple(p) for p in self.hidden_points}

    def kept_set(self) -> set:
        return {tuple(p) for p in self.kept_points}

    def hide_points(self, pairs):
        """Hide (record_id, channel) points from Baseline Shift; data is untouched."""
        pairs = {tuple(p) for p in pairs}
        with self.transaction():
            self.hidden_points = [list(p) for p in self.hidden_set() | pairs]
            self.kept_points = [list(p) for p in self.kept_set() - pairs]

    def show_points(self, pairs, keep=()):
        """Un-hide points; those in `keep` are also kept visible even if the automatic outlier rule flags them."""
        pairs, keep = {tuple(p) for p in pairs}, {tuple(p) for p in keep}
        with self.transaction():
            self.hidden_points = [list(p) for p in self.hidden_set() - pairs]
            self.kept_points = [list(p) for p in self.kept_set() | keep]

    def reset_points(self):
        with self.transaction():
            self.hidden_points, self.kept_points = [], []

    # ------------------------------------------------------------------ shared presets and gauge types
    def set_preset(self, name: str, settings: dict):
        with self.transaction():
            self.presets[name] = settings

    def delete_preset(self, name: str):
        with self.transaction():
            self.presets.pop(name, None)

    # ------------------------------------------------------------------ app lock
    def current_lock(self, now: Optional[float] = None) -> dict:
        """The app lock if one is held and hasn't run out, else {}."""
        lk = self.app_lock or {}
        return lk if lk and float(lk.get("until", 0)) > (time.time() if now is None else now) else {}

    def take_lock(self, by: str, key: str, minutes: float, note: str = "") -> dict:
        """Lock the app for `minutes`. Returns the lock now in force (someone else's if they got there first)."""
        with self.transaction():
            cur = self.current_lock()
            if cur and cur.get("key") != key:
                return cur
            self.app_lock = {"by": by, "key": key, "until": time.time() + 60 * float(minutes), "note": note}
            return self.app_lock

    def release_lock(self, key: Optional[str] = None):
        """Unlock. With a key, only if that key holds the lock; without one, unlock regardless."""
        with self.transaction():
            if key is None or (self.app_lock or {}).get("key") == key:
                self.app_lock = {}

    def set_gauge_types(self, overrides: Dict[str, str]):
        with self.transaction():
            self.gauge_types = dict(overrides)

    def records_df(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=RECORD_FIELDS + ["n_channels", "runs"])
        df = pd.DataFrame(self.records)
        df["n_channels"] = df["channels"].map(len)
        df["runs"] = df["id"].map(lambda i: sum(r["record_id"] == i for r in self.runs))
        return df.sort_values(["location", "date", "file_name"]).reset_index(drop=True)

    def load_raw(self, rid: str, channels: Optional[List[str]] = None):
        """(time array, DataFrame of channels) exactly as uploaded."""
        rec = self.record(rid)
        if not rec.get("raw_file"):
            raise ValueError(f"{self.record_label(rec)} has no raw data.")
        cols = [rec["time_col"]] + (list(channels) if channels is not None else rec["channels"])
        HOOKS["fetch"](rec["raw_file"])
        df = pd.read_parquet(self.root / rec["raw_file"], columns=cols)
        return df[rec["time_col"]].to_numpy(float), df.drop(columns=rec["time_col"])

    # ------------------------------------------------------------------ filtered runs
    def run(self, run_id: str) -> dict:
        return next(r for r in self.runs if r["id"] == run_id)

    def run_label(self, run: dict) -> str:
        rec = self.record(run["record_id"])
        return f"{rec['location']} · {rec['date']} · {run['name']}" + (" (active)" if run.get("active") else "")

    def runs_for(self, rid: str) -> List[dict]:
        return [r for r in self.runs if r["record_id"] == rid]

    def active_run(self, rid: str) -> Optional[dict]:
        runs = self.runs_for(rid)
        return next((r for r in runs if r.get("active")), runs[-1] if runs else None)

    @staticmethod
    def run_entry(res: TraceResult) -> dict:
        """The part of a TraceResult a run keeps: the filtered + corrected signal and a small summary.
        Every baseline mode subtracts a constant, so the offsets rebuild the other signals exactly."""
        return dict(y=np.asarray(res.y_filtered_corrected, dtype=float), summary=dict(
            filter_summary=res.filter_summary, raw_baseline_summary=res.raw_baseline_summary,
            filtered_baseline_summary=res.filtered_baseline_summary, trim_used=res.trim_used,
            raw_offset=float(np.nanmean(res.y_raw - res.y_raw_corrected)),
            filtered_offset=float(np.nanmean(res.y_filtered - res.y_filtered_corrected))))

    def save_run(self, rid: str, name: str, settings: dict, results: Dict[str, object],
                 make_active=True, replace_same_name=True, by="") -> dict:
        """results: {channel: TraceResult or run_entry(...)}."""
        n = self.record(rid)["n_samples"]
        run_id = f"run_{_short_id()}"
        cols, summary = {}, {}
        for ch, res in results.items():
            e = res if isinstance(res, dict) else self.run_entry(res)
            a = np.full(n, np.nan)
            a[n - len(e["y"]):] = e["y"]          # auto-trim drops samples at the start
            cols[ch] = pd.array(np.round(a / QUANTUM), dtype="Int32")
            summary[ch] = dict(e["summary"], n_trimmed=int(n - len(e["y"])))
        path = self.root / "filtered" / f"{run_id}.parquet"
        _atomic_parquet(pd.DataFrame(cols), path, compression="zstd", compression_level=9)
        try:
            with self.transaction():
                if not any(r["id"] == rid for r in self.records):
                    raise ValueError("This recording was deleted by someone else while it was being filtered.")
                if replace_same_name:
                    for old in [r for r in self.runs_for(rid) if r["name"] == name]:
                        self.delete_run(old["id"])
                if make_active:
                    for r in self.runs_for(rid):
                        r["active"] = False
                run = dict(id=run_id, record_id=rid, name=name, settings=settings, file=f"filtered/{run_id}.parquet",
                           channels=list(results), summary=summary, created=_now(), created_by=by or "",
                           active=bool(make_active or not self.runs_for(rid)))
                self.runs.append(run)
                HOOKS["on_write"](self._rel(path))
                return run
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def set_active(self, run_id: str):
        with self.transaction():
            run = self.run(run_id)
            for r in self.runs_for(run["record_id"]):
                r["active"] = r["id"] == run_id

    def rename_run(self, run_id: str, name: str):
        with self.transaction():
            self.run(run_id)["name"] = name

    def delete_run(self, run_id: str, save=True):
        with self.transaction():
            run = next((r for r in self.runs if r["id"] == run_id), None)
            if run is None:
                return
            (self.root / run["file"]).unlink(missing_ok=True)
            HOOKS["on_delete"](run["file"])
            was_active = run.get("active")
            self.runs = [r for r in self.runs if r["id"] != run_id]
            self._drop_stats(basis=run_id)
            remaining = self.runs_for(run["record_id"])
            if was_active and remaining:
                remaining[-1]["active"] = True

    def load_run(self, run_id: str, channels: Optional[List[str]] = None) -> pd.DataFrame:
        """Filtered + corrected channels (µε, float64), aligned to the raw samples; trimmed samples are NaN."""
        run = self.run(run_id)
        cols = list(channels) if channels is not None else run["channels"]
        HOOKS["fetch"](run["file"])
        q = pd.read_parquet(self.root / run["file"], columns=cols)
        return q.astype("float64") * QUANTUM

    def run_time(self, run: dict, t_raw: np.ndarray) -> np.ndarray:
        s = run.get("settings", {})
        if s.get("rebuild_time"):
            return t_raw[0] + np.arange(len(t_raw)) / float(s.get("fs", 200.0))
        return t_raw

    def run_traces(self, run_id: str, channels: Optional[List[str]] = None) -> Dict[str, dict]:
        """Everything the session library needs per channel, rebuilt from raw + stored filtered data."""
        run = self.run(run_id)
        chans = list(channels) if channels is not None else run["channels"]
        t_raw, raw = self.load_raw(run["record_id"], chans)
        filt = self.load_run(run_id, chans)
        t = self.run_time(run, t_raw)
        out = {}
        for ch in chans:
            info = run["summary"][ch]
            k = info.get("n_trimmed", 0)
            y_raw = raw[ch].to_numpy(float)[k:]
            yfc = filt[ch].to_numpy(float)[k:]
            out[ch] = dict(x=t[k:], y_raw=y_raw, y_raw_corrected=y_raw - info["raw_offset"],
                           y_filtered=yfc + info["filtered_offset"], y_filtered_corrected=yfc, info=info)
        return out

    # ------------------------------------------------------------------ baseline statistics
    def stats(self) -> pd.DataFrame:
        if self.stats_path.exists():
            return pd.read_parquet(self.stats_path)
        return pd.DataFrame(columns=STATS_COLUMNS)

    def put_stats(self, rows: pd.DataFrame):
        """Insert/replace rows keyed by (record_id, basis, channel, window_s)."""
        if rows is None or rows.empty:
            return
        key = ["record_id", "basis", "channel", "window_s"]
        with self._lock:
            old = self.stats()
            if not old.empty:
                merged = old.merge(rows[key].drop_duplicates(), on=key, how="left", indicator=True)
                old = old[(merged["_merge"] == "left_only").to_numpy()]
            out = pd.concat([old, rows[STATS_COLUMNS]], ignore_index=True) if not old.empty else rows[STATS_COLUMNS]
            _atomic_parquet(out.reset_index(drop=True), self.stats_path)
            HOOKS["on_write"]("stats.parquet")

    def _drop_stats(self, record_id=None, basis=None):
        with self._lock:
            st = self.stats()
            if st.empty:
                return
            keep = np.ones(len(st), bool)
            if record_id is not None:
                keep &= (st["record_id"] != record_id).to_numpy()
            if basis is not None:
                keep &= (st["basis"] != basis).to_numpy()
            _atomic_parquet(st[keep].reset_index(drop=True), self.stats_path)
            HOOKS["on_write"]("stats.parquet")

    def size_mb(self) -> float:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file()) / 1e6
