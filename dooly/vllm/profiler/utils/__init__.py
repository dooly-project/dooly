"""vLLM profiler utils with fallback to shared common utils."""

from pathlib import Path

_COMMON_UTILS = Path(__file__).resolve().parents[3] / "common" / "profiler" / "utils"
if str(_COMMON_UTILS) not in __path__:
    __path__.append(str(_COMMON_UTILS))
