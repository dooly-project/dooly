"""
Simulator package for vLLM scheduler simulation.

This package provides tools to simulate and test vLLM scheduler behavior
without requiring GPU hardware.
"""

import sys
from pathlib import Path

# Add parent directory to path to enable scheduler imports
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

__all__ = ['emulator', 'request', 'scheduler', 'simulator', 'utils']
