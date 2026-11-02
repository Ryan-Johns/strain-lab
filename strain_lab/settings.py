"""One settings object for filtering + baseline, shared by every page and by presets."""
from dataclasses import dataclass, asdict, fields
from typing import Optional

from .config import DEFAULT_FS, TARGET_FILTERED, LEGACY_BASELINE_NAMES


@dataclass
class FilterSettings:
    filter_type: str = "Low-pass Butterworth"
    order: int = 4
    cutoff_low: float = 0.1
    cutoff_high: float = 1.0
    passes: int = 2                  # 2 = zero-phase (forward + backward, V46 behaviour); 1 = single forward pass like DPlot
    fs: float = DEFAULT_FS
    rebuild_time: bool = False       # replace the CSV time column with t0 + i/fs
    baseline_mode: str = "None"
    baseline_target: str = TARGET_FILTERED
    zero_window_s: float = 10.0      # 'Zero to start': median of this many seconds
    quiet_start: Optional[float] = None
    quiet_end: Optional[float] = None
    auto_retry: bool = True          # DPlot Integrated: trim the start and retry if the arrival interval is zero
    trim_step: float = 0.5
    max_trim: float = 10.0
    # Frequency-curve defaults stored with presets (used by Batch Process).
    frequency_signal: str = "Filtered (zeroed)"
    n_bins: int = 200
    negative_only: bool = False
    common_range: bool = True
    make_frequency: bool = True

    @classmethod
    def from_preset(cls, preset: Optional[dict]) -> "FilterSettings":
        """Build settings from a preset dict; tolerant of old V22–V46 presets."""
        preset = dict(preset or {})
        if preset.get("fs") is None and preset.get("fs_override") is not None:
            preset["fs"] = preset["fs_override"]
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in preset.items() if k in known and v is not None}
        return cls(**clean)

    def __post_init__(self):
        self.baseline_mode = LEGACY_BASELINE_NAMES.get(self.baseline_mode, self.baseline_mode)
        self.frequency_signal = {"Raw + corrected": "Raw (zeroed)", "Filtered + corrected": "Filtered (zeroed)"}.get(
            self.frequency_signal, self.frequency_signal)

    def to_preset(self) -> dict:
        d = asdict(self)
        d.pop("quiet_start"); d.pop("quiet_end")   # quiet windows are file-specific
        return d
