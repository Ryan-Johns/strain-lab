"""Constants shared across the app: option lists, colors and default presets."""

APP_TITLE = "Strain Gauge Filtering Lab — V54"
DEFAULT_FS = 200.0

FILTER_TYPES = ["Low-pass Butterworth", "High-pass Butterworth", "Band-pass Butterworth"]
NEEDS_LOW_CUTOFF = {"High-pass Butterworth", "Band-pass Butterworth"}
NEEDS_HIGH_CUTOFF = {"Low-pass Butterworth", "Band-pass Butterworth"}

BL_NONE = "None"
BL_ZERO_START = "Zero to start (median of first N s)"
BL_QUIET = "Zero to a quiet window (average)"
BL_DPLOT = "DPlot Integrated (legacy)"
BL_DPLOT_SHIFTED = "DPlot Integrated + Shifted (legacy)"
BASELINE_MODES = [BL_NONE, BL_ZERO_START, BL_QUIET, BL_DPLOT, BL_DPLOT_SHIFTED]
INTEGRATED_MODES = {BL_DPLOT, BL_DPLOT_SHIFTED}
# Names used by V46–V54 presets and saved runs.
LEGACY_BASELINE_NAMES = {"DPlot constant shift (quiet window)": BL_QUIET, "DPlot Integrated": BL_DPLOT,
                         "DPlot Integrated + Shifted": BL_DPLOT_SHIFTED}

BASELINE_HELP = (
    "Baseline correction subtracts one constant from the whole record, so the result starts near zero. "
    "The shape of the record (including slow drift) is unchanged.\n\n"
    "- **None**: keep the gauge's own values.\n"
    "- **Zero to start** (recommended): subtract the median of the first N seconds. Predictable and the same for "
    "every channel and file.\n"
    "- **Zero to a quiet window**: subtract the average over a time window you choose (e.g. a period with no traffic).\n"
    "- **DPlot Integrated (legacy)**: DPlot's method — finds where the signal first moves more than 5% of its range "
    "and subtracts the average before that point. The window varies by channel and file. Kept for matching old results.\n"
    "- **+ Shifted (legacy)**: DPlot Integrated, then also removes any average left in that window.\n\n"
    "A band-pass filter's low cutoff also pulls the record toward zero, but by removing slow changes (drift), "
    "not by a single shift."
)

TARGET_RAW, TARGET_FILTERED, TARGET_BOTH = "Raw signal", "Filtered signal", "Both raw and filtered"
BASELINE_TARGETS = [TARGET_RAW, TARGET_FILTERED, TARGET_BOTH]

# Signal name shown in the UI -> key in a library trace item.
SIGNALS = {
    "Raw": "y_raw",
    "Raw (zeroed)": "y_raw_corrected",
    "Filtered": "y_filtered",
    "Filtered (zeroed)": "y_filtered_corrected",
}
SIGNAL_NAMES = list(SIGNALS)
SIGNALS.update({"Raw + corrected": "y_raw_corrected", "Filtered + corrected": "y_filtered_corrected"})  # old names
SIGNAL_HELP = ("'Raw' and 'Filtered' are before baseline correction; '(zeroed)' versions are after it. "
               "With baseline correction set to None they are identical.")
ARRAY_KEYS = ["x", "y_raw", "y_raw_corrected", "y_filtered", "y_filtered_corrected",
              "baseline_raw", "baseline_filtered"]

COLOR_OPTIONS = {
    "Auto": None, "Red": "red", "Orange": "orange", "Yellow": "gold", "Green": "green",
    "Blue": "blue", "Purple": "purple", "Pink": "deeppink", "Black": "black", "Gray": "gray",
    "Light Gray": "lightgray", "Cyan": "cyan", "Teal": "teal", "Brown": "brown",
    "Maroon": "maroon", "Navy": "navy", "Lime": "limegreen", "Magenta": "magenta",
    "Dark Green": "darkgreen", "Dark Orange": "darkorange", "Dodger Blue": "dodgerblue",
    "Slate Blue": "slateblue",
}
COLOR_NAMES = list(COLOR_OPTIONS)
UNIQUE_COLORS = [v for v in COLOR_OPTIONS.values() if v is not None]
AUTO_COLOR_CYCLE = ["red", "green", "blue", "orange", "purple", "black", "magenta", "teal",
                    "brown", "navy", "darkgreen", "darkorange", "dodgerblue", "slateblue",
                    "gold", "deeppink"]
# Chronological colors for appended frequency files: red -> purple, no pink.
RAINBOW_NO_PINK = ["red", "orange", "gold", "green", "teal", "cyan", "blue", "dodgerblue",
                   "navy", "slateblue", "purple"]
# Default color-picker positions for uploaded curves: Red, Orange, Yellow, Green, Blue, Purple, Pink.
RAINBOW_PICKER_NAMES = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Pink"]

DASH_OPTIONS = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
LEGEND_POSITIONS = ["Top Right (inside)", "Top Left (inside)", "Bottom Right (inside)",
                    "Bottom Left (inside)", "Outside Right"]

RECOMMENDED_PRESET = "Low-pass 20 Hz, zeroed (recommended)"
DEFAULT_PRESETS = {
    RECOMMENDED_PRESET: {
        "filter_type": "Low-pass Butterworth", "order": 4, "cutoff_high": 20.0, "cutoff_low": 0.1, "passes": 2,
        "fs": DEFAULT_FS, "rebuild_time": True, "baseline_mode": BL_ZERO_START, "zero_window_s": 10.0,
        "baseline_target": TARGET_BOTH, "frequency_signal": "Filtered (zeroed)", "n_bins": 200,
        "negative_only": False, "common_range": True, "make_frequency": True,
    },
    "Band-pass 0.001–20 Hz, drift removed": {
        "filter_type": "Band-pass Butterworth", "order": 1, "cutoff_low": 0.001, "cutoff_high": 20.0, "passes": 2,
        "fs": DEFAULT_FS, "rebuild_time": True, "baseline_mode": BL_ZERO_START, "zero_window_s": 10.0,
        "baseline_target": TARGET_BOTH, "frequency_signal": "Filtered (zeroed)", "n_bins": 200,
        "negative_only": False, "common_range": True, "make_frequency": True,
    },
    "Bridge Traffic Default": {   # the original V46 preset, unchanged
        "filter_type": "Band-pass Butterworth", "order": 1, "cutoff_low": 0.001, "cutoff_high": 20.0, "passes": 2,
        "fs": DEFAULT_FS, "baseline_mode": BL_DPLOT, "baseline_target": TARGET_BOTH,
        "frequency_signal": "Filtered (zeroed)", "n_bins": 200, "negative_only": False, "common_range": True,
        "make_frequency": True,
    },
}
