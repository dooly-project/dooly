"""SGLang profiler package with fallback to shared modules."""

from pathlib import Path

_COMMON_PROFILER = Path(__file__).resolve().parents[2] / "common" / "profiler"
if str(_COMMON_PROFILER) not in __path__:
    __path__.append(str(_COMMON_PROFILER))
