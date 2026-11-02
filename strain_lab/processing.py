"""Filtering, baseline correction and the single processing pipeline used by every page.

Nothing in here imports Streamlit, so it can be tested or used from a script.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

from .config import (NEEDS_LOW_CUTOFF, NEEDS_HIGH_CUTOFF, INTEGRATED_MODES, LEGACY_BASELINE_NAMES,
                     BL_ZERO_START, BL_QUIET, BL_DPLOT_SHIFTED,
                     TARGET_RAW, TARGET_FILTERED, TARGET_BOTH)
from .settings import FilterSettings

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # np.trapz was removed in NumPy 2.4


class ArrivalIntervalZero(ValueError):
    """DPlot Integrated could not integrate because the pre-arrival interval has zero length."""


# ----------------------------------------------------------------------------- time
def rebuild_time(x: np.ndarray, fs: float) -> np.ndarray:
    """Even time axis t0 + i/fs. Fixes CSV time columns that were rounded (repeated timestamps)."""
    x = np.asarray(x, dtype=float)
    return (x[0] if len(x) else 0.0) + np.arange(len(x)) / float(fs)


def count_repeated_times(x: np.ndarray) -> int:
    return int(np.count_nonzero(np.diff(np.asarray(x, dtype=float)) <= 0))


# ----------------------------------------------------------------------------- filter
def design_filter(s: FilterSettings) -> Tuple[np.ndarray, str]:
    if s.fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    kind = {"Low-pass Butterworth": "lowpass", "High-pass Butterworth": "highpass",
            "Band-pass Butterworth": "bandpass"}.get(s.filter_type)
    if kind is None:
        raise ValueError(f"Unknown filter type: {s.filter_type}")
    if kind == "lowpass":
        wn, desc = s.cutoff_high, f"cutoff={s.cutoff_high} Hz"
    elif kind == "highpass":
        wn, desc = s.cutoff_low, f"cutoff={s.cutoff_low} Hz"
    else:
        wn, desc = [s.cutoff_low, s.cutoff_high], f"low={s.cutoff_low} Hz, high={s.cutoff_high} Hz"
    # Second-order sections are numerically stable even with a 0.001 Hz corner at 200 Hz.
    sos = butter(int(s.order), wn, btype=kind, fs=float(s.fs), output="sos")
    pass_txt = "zero-phase, 2 passes" if int(s.passes) == 2 else "single forward pass"
    return sos, f"{s.filter_type} (order={s.order}, {desc}, fs={s.fs:.6g} Hz, {pass_txt})"


def apply_filter(y: np.ndarray, s: FilterSettings) -> Tuple[np.ndarray, str]:
    y = np.asarray(y, dtype=float)
    sos, summary = design_filter(s)
    if int(s.passes) == 2:
        return sosfiltfilt(sos, y), summary
    # Single pass, started in steady state at the first value so the raw offset
    # (hundreds to thousands of µε) does not leave a long start-up transient.
    yf, _ = sosfilt(sos, y, zi=sosfilt_zi(sos) * y[0])
    return yf, summary


# ----------------------------------------------------------------------------- baseline
def apply_baseline(y, mode="None", x=None, quiet_start=None, quiet_end=None, zero_window_s=10.0):
    """Returns (corrected, baseline, summary). Every mode subtracts a single constant.
    The quiet-window and DPlot modes use the same maths as V46."""
    y = np.asarray(y, dtype=float)
    mode = LEGACY_BASELINE_NAMES.get(mode, mode)
    baseline = np.zeros_like(y)
    if len(y) == 0:
        return y.copy(), baseline, "No data"
    if mode == "None":
        return y.copy(), baseline, "No baseline correction"

    if mode == BL_ZERO_START:
        if x is None:
            raise ValueError("Zero to start requires the time array.")
        x = np.asarray(x, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            raise ValueError("No valid data.")
        t0 = float(x[ok][0])
        win = ok & (x < t0 + float(zero_window_s))
        ref = float(np.median(y[win]))
        baseline[:] = ref
        return y - ref, baseline, f"Zeroed to the median of the first {zero_window_s:g} s ({int(win.sum())} samples) = {ref:.6g}"

    if mode == BL_QUIET:
        if x is None or quiet_start is None or quiet_end is None or quiet_end <= quiet_start:
            raise ValueError("Quiet-window baseline shift requires valid time, quiet start and quiet end.")
        x = np.asarray(x, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x >= quiet_start) & (x <= quiet_end)
        if not np.any(mask):
            raise ValueError("No data points found in the selected quiet window.")
        ref = float(np.mean(y[mask]))
        baseline[:] = ref
        return y - ref, baseline, (f"Zeroed to the average of the quiet window "
                                   f"[{quiet_start:.6g}, {quiet_end:.6g}] with mean={ref:.6g}")

    if mode in INTEGRATED_MODES:
        if x is None:
            raise ValueError("DPlot Integrated requires the time array.")
        x = np.asarray(x, dtype=float)
        if len(y) < 3 or len(x) != len(y):
            raise ValueError("DPlot Integrated requires matching time and signal arrays with at least 3 points.")
        y0 = y[0]
        threshold = 0.05 * float(np.max(y) - np.min(y))
        if threshold <= 0:
            raise ValueError("Signal amplitude range is zero; cannot determine DPlot Integrated arrival.")
        departed = np.nonzero(np.abs(y[1:] - y0) > threshold)[0]
        if len(departed) == 0:
            raise ValueError("Could not determine arrival time using 5% amplitude-range threshold.")
        arrival = int(departed[0]) + 1
        pre = arrival - 1
        total_time = float(x[pre] - x[0])
        if total_time == 0:
            raise ArrivalIntervalZero("Arrival interval is zero; cannot compute DPlot Integrated shift.")
        impulse = float(_trapezoid(y[:pre + 1], x[:pre + 1]))
        shift = -impulse / total_time
        y_corr = y + shift
        baseline[:] = -shift
        info = (f"arrival_idx={arrival}, arrival_time={x[arrival]:.6g}, threshold={threshold:.6g}, "
                f"pre_integral={impulse:.6g}, shift={shift:.6g}")
        if mode == BL_DPLOT_SHIFTED:
            residual = float(np.mean(y_corr[:pre + 1]))
            return y_corr - residual, baseline + residual, f"DPlot Integrated + Shifted: {info}, residual_mean={residual:.6g}"
        return y_corr, baseline, f"DPlot Integrated: {info}"

    return y.copy(), baseline, "Unknown mode; no baseline correction applied"


def baseline_with_retry(y, x, s: FilterSettings):
    """apply_baseline, retrying DPlot Integrated after trimming the start if the arrival interval is zero.

    Returns (corrected, baseline, summary, start_index). start_index > 0 means the first
    start_index samples were dropped, and every companion array must be sliced the same way.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    try:
        corr, base, summary = apply_baseline(y, s.baseline_mode, x, s.quiet_start, s.quiet_end, s.zero_window_s)
        return corr, base, summary, 0
    except ArrivalIntervalZero as first_error:
        if not (s.auto_retry and len(x) >= 3):
            raise
        t0, t_end = float(np.min(x)), float(np.max(x))
        step = max(float(s.trim_step), 1e-9)
        max_trim = max(float(s.max_trim), step)
        trim, last_error = step, first_error
        while trim <= max_trim and t0 + trim < t_end:
            keep = x >= t0 + trim
            if np.count_nonzero(keep) < 3:
                break
            start = int(np.argmax(keep))
            qs = s.quiet_start if s.quiet_start is None else max(s.quiet_start, float(x[start]))
            qe = s.quiet_end if (s.quiet_end is not None and qs is not None and s.quiet_end > qs) else None
            try:
                corr, base, summary = apply_baseline(y[start:], s.baseline_mode, x[start:], qs, qe, s.zero_window_s)
                summary += (f" | auto-retry success after trimming {trim:.6g} time units from start "
                            f"({t0:.6g} to {float(x[start]):.6g})")
                return corr, base, summary, start
            except Exception as err:   # keep trimming
                last_error = err
                trim += step
        raise ValueError(f"{first_error} Auto-retry failed after trimming up to {max_trim:.6g} time units. "
                         f"Last retry error: {last_error}")


# ----------------------------------------------------------------------------- pipeline
@dataclass
class TraceResult:
    x: np.ndarray
    y_raw: np.ndarray
    y_raw_corrected: np.ndarray
    y_filtered: np.ndarray
    y_filtered_corrected: np.ndarray
    baseline_raw: np.ndarray
    baseline_filtered: np.ndarray
    filter_summary: str
    raw_baseline_summary: str
    filtered_baseline_summary: str
    trim_used: float = 0.0


def process_trace(x, y_raw, s: FilterSettings, correct_raw: bool = True) -> TraceResult:
    """Filter + baseline-correct one channel. Used by Filter, Batch Process and Filtered Export.

    correct_raw=False skips the raw-signal baseline even when the target includes it
    (Filtered Export only writes the original raw values, so it never needs it).
    """
    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    if len(x) < 3:
        raise ValueError("Not enough valid points.")
    if s.rebuild_time:
        x = rebuild_time(x, s.fs)
    t_start = float(x[0])

    y_filt, filter_summary = apply_filter(y_raw, s)
    y_raw_corr, base_raw = y_raw, np.zeros_like(y_raw)
    y_filt_corr, base_filt = y_filt, np.zeros_like(y_filt)
    raw_summary = filt_summary = "No baseline correction"

    if correct_raw and s.baseline_target in (TARGET_RAW, TARGET_BOTH):
        y_raw_corr, base_raw, raw_summary, start = baseline_with_retry(y_raw, x, s)
        if start:
            # Raw needed trimming: trim everything and refilter so all arrays line up.
            x, y_raw = x[start:], y_raw[start:]
            y_filt, filter_summary = apply_filter(y_raw, s)
            y_filt_corr, base_filt = y_filt, np.zeros_like(y_filt)

    if s.baseline_target in (TARGET_FILTERED, TARGET_BOTH):
        y_filt_corr, base_filt, filt_summary, start = baseline_with_retry(y_filt, x, s)
        if start:
            x, y_raw, y_filt = x[start:], y_raw[start:], y_filt[start:]
            y_raw_corr, base_raw = y_raw_corr[start:], base_raw[start:]

    return TraceResult(x, y_raw, y_raw_corr, y_filt, y_filt_corr, base_raw, base_filt,
                       filter_summary, raw_summary, filt_summary, float(x[0] - t_start))


# ----------------------------------------------------------------------------- helpers
def clean_xy(df, xcol: str, ycol: str):
    import pandas as pd
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    mask = x.notna() & y.notna()
    return x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float)


def trim_arrays(x, y, start: Optional[float], end: Optional[float]):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if start is not None:
        mask &= x >= start
    if end is not None:
        mask &= x <= end
    return x[mask], y[mask]
