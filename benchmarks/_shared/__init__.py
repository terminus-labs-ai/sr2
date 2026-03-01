"""Shared utilities for SR2 benchmarks."""

import os
import sys

# Ensure src/ is on sys.path so sr2 can be imported
_src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if os.path.isdir(_src_dir) and _src_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_src_dir))
