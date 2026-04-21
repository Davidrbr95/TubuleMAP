"""Cellpose tracking subpackage.

This file also registers a legacy module alias so existing absolute imports like
``from cellpose_tracker.tracking import ...`` keep working when this code is
imported as ``tubulemap.cellpose_tracker``.
"""

import sys as _sys

# Legacy compatibility for absolute imports used throughout this codebase.
_sys.modules.setdefault("cellpose_tracker", _sys.modules[__name__])

from . import tracking

__all__ = ["tracking"]
