import streamlit as st

from ..config import NEEDS_LOW_CUTOFF, NEEDS_HIGH_CUTOFF, TARGET_RAW, TARGET_FILTERED, TARGET_BOTH
from ..settings import FilterSettings

_BASELINE_FUNCS = {
    "Zero to start (median of first N s)": '''
def baseline(y, x):
    """Subtract the median of the first {zw} s."""
    return y - np.median(y[x < x[0] + {zw}])
''',
    "Zero to a quiet window (average)": '''
def baseline(y, x):
    """Subtract the mean over the quiet window."""
    m = (x >= {qs!r}) & (x <= {qe!r})
    return y - y[m].mean()
''',
    "DPlot Integrated (legacy)": '''
def baseline(y, x, shifted={shifted}):
    """DPlot Integrated: remove the average before the first 5%-of-range departure from y[0]."""
    thr = 0.05 * (y.max() - y.min())
    arrival = int(np.nonzero(np.abs(y[1:] - y[0]) > thr)[0][0]) + 1
    pre = arrival - 1
    shift = -np.trapezoid(y[:pre + 1], x[:pre + 1]) / (x[pre] - x[0])
    y = y + shift
    if shifted:   # "DPlot Integrated + Shifted"
        y = y - y[:pre + 1].mean()
    return y
''',
}


def reproducible_code(s: FilterSettings, file_name="your_file.csv", xcol="TIME", ycol="CHANNEL") -> str:
    kind = {"Low-pass Butterworth": "lowpass", "High-pass Butterworth": "highpass", "Band-pass Butterworth": "bandpass"}[s.filter_type]
    wn = ([s.cutoff_low, s.cutoff_high] if kind == "bandpass" else s.cutoff_low if kind == "highpass" else s.cutoff_high)
    lines = ["import numpy as np", "import pandas as pd",
             "from scipy.signal import butter, sosfiltfilt, sosfilt, sosfilt_zi", "",
             f'df = pd.read_csv("{file_name}")',
             f'x = pd.to_numeric(df["{xcol}"], errors="coerce")',
             f'y = pd.to_numeric(df["{ycol}"], errors="coerce")',
             "ok = x.notna() & y.notna()",
             "x, y = x[ok].to_numpy(float), y[ok].to_numpy(float)"]
    if s.rebuild_time:
        lines.append(f"x = x[0] + np.arange(len(x)) / {s.fs!r}   # rebuild time from the sampling frequency")
    lines += ["", f"# {s.filter_type}, order {s.order}, fs = {s.fs:g} Hz",
              f'sos = butter({s.order}, {wn!r}, btype="{kind}", fs={s.fs!r}, output="sos")']
    if int(s.passes) == 2:
        lines.append("y_filt = sosfiltfilt(sos, y)   # zero-phase (2 passes)")
    else:
        lines.append("y_filt, _ = sosfilt(sos, y, zi=sosfilt_zi(sos) * y[0])   # single forward pass")
    if s.baseline_mode != "None":
        tmpl = _BASELINE_FUNCS["DPlot Integrated (legacy)" if "Integrated" in s.baseline_mode else s.baseline_mode]
        lines.append(tmpl.format(qs=s.quiet_start, qe=s.quiet_end, zw=s.zero_window_s,
                                 shifted="Shifted" in s.baseline_mode))
        if s.baseline_target in (TARGET_RAW, TARGET_BOTH):
            lines.append("y_raw_corrected = baseline(y, x)")
        if s.baseline_target in (TARGET_FILTERED, TARGET_BOTH):
            lines.append("y_filt_corrected = baseline(y_filt, x)")
    return "\n".join(lines)
