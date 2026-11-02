"""Reading CSV/ZIP uploads and small naming helpers (no Streamlit)."""
import io
import re
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Try common encodings and separators; accept the first parse with 2+ columns."""
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "latin1", "cp1252"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep)
                if df.shape[1] >= 2:
                    df.columns = [str(c).strip() for c in df.columns]
                    return df
            except Exception as e:
                last_error = e
    raise ValueError(f"Could not read CSV file. Last error: {last_error}")


def csv_names_in_zip(zf: zipfile.ZipFile) -> List[str]:
    return [n for n in zf.namelist()
            if n.lower().endswith(".csv") and not n.startswith("__MACOSX/") and not Path(n).name.startswith(".")]


def read_zip_csvs(raw: bytes) -> List[Tuple[str, pd.DataFrame]]:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = csv_names_in_zip(zf)
        if not names:
            raise ValueError("No CSV files were found inside the ZIP.")
        return [(Path(n).name, read_csv_bytes(zf.read(n))) for n in names]


def numeric_columns(df: pd.DataFrame) -> List[str]:
    out = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().any():
            out.append(c)
    return out


def guess_time_column(df: pd.DataFrame, numeric: Optional[List[str]] = None) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for p in ("time", "timestamp", "t", "seconds", "sec", "sample", "index"):
        if p in lower:
            return lower[p]
    for c in numeric if numeric is not None else numeric_columns(df):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) >= 3 and s.is_monotonic_increasing:
            return c
    return None


def safe_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in str(name).strip())
    return safe or "channel"


# ----------------------------------------------------------------------------- dates in labels/file names
_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[_\-/.](\d{1,2})[_\-/.](\d{4}|\d{2})(?!\d)")


def date_tuple(text: str) -> Tuple[int, int, int]:
    """(year, month, day) from names like 10_24_25 or 10-24-2025; (9999, 99, 99) when absent."""
    m = _DATE_RE.search(str(text))
    if not m:
        return (9999, 99, 99)
    month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (year + 2000 if year < 100 else year, month, day)


def date_display(text: str) -> str:
    """'10_24_25' style date for a label/file name, or '' if none."""
    y, m, d = date_tuple(text)
    return "" if y == 9999 else f"{m}_{d}_{y % 100:02d}"


def date_sort_key(text: str):
    return (date_tuple(text), str(text).lower())


# ----------------------------------------------------------------------------- raw-file sources
import datetime as _dt
from dataclasses import dataclass, field
from typing import Callable

_LOC_RE = re.compile(r"(?:^|[^A-Za-z])L(?:oc(?:ation)?)?[\s_-]*(\d+)", re.I)


def guess_location(*texts: str) -> str:
    """'L3', from names like 'L3 12_9_25.csv' or 'Raw Data - Location 3 - T3.zip'; first text that matches wins."""
    for t in texts:
        m = _LOC_RE.search(str(t or ""))
        if m:
            return f"L{int(m.group(1))}"
    return ""


def guess_date(*texts: str) -> Optional[_dt.date]:
    """Date from the first name that has one. Paths are checked one folder at a time, file name first."""
    parts = [p for t in texts for p in reversed(re.split(r"[\\/]", str(t or ""))) if p]
    for t in parts:
        y, m, d = date_tuple(t)
        if y != 9999:
            try:
                return _dt.date(y, m, d)
            except ValueError:
                pass
    return None


@dataclass
class RawSource:
    """One raw CSV, either uploaded directly or inside an uploaded ZIP. Read on demand, one at a time."""
    key: str                      # unique id for caching
    file_name: str                # CSV file name
    path: str                     # path inside the ZIP (or the file name)
    container: str                # ZIP name, or ''
    size_mb: float
    modified: Optional[_dt.date]  # file timestamp inside the ZIP
    reader: Callable[[], pd.DataFrame] = field(repr=False)

    def read(self) -> pd.DataFrame:
        return self.reader()

    @property
    def location_guess(self) -> str:
        return guess_location(self.path, self.container)

    @property
    def date_guess(self) -> Optional[_dt.date]:
        return guess_date(Path(self.path).name, self.path, self.container)


def sources_from_upload(uploaded) -> List[RawSource]:
    """List the CSVs in an uploaded CSV or ZIP without reading them (ZIPs can hold many large files)."""
    name, fid = uploaded.name, getattr(uploaded, "file_id", uploaded.name)
    if not name.lower().endswith(".zip"):
        return [RawSource(f"{fid}", name, name, "", uploaded.size / 1e6, None,
                          lambda u=uploaded: read_csv_bytes(u.getvalue()))]
    out = []
    with zipfile.ZipFile(uploaded) as zf:
        for info in zf.infolist():
            n = info.filename
            if info.is_dir() or not n.lower().endswith(".csv") or n.startswith("__MACOSX/") or Path(n).name.startswith("."):
                continue
            def reader(u=uploaded, member=n):
                u.seek(0)
                with zipfile.ZipFile(u) as z:
                    return read_csv_bytes(z.read(member))
            try:
                mod = _dt.date(*info.date_time[:3])
            except ValueError:
                mod = None
            out.append(RawSource(f"{fid}::{n}", Path(n).name, n, name, info.file_size / 1e6, mod, reader))
    return sorted(out, key=lambda s: (s.location_guess, s.date_guess or _dt.date.max, s.path.lower()))
