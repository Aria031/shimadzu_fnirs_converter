"""shimadzu-fnirs-converter

Public API entrypoints.
Users should import from top-level:

    from shimadzu_fnirs_converter import convert, convert_fif, convert_snirf
"""

from __future__ import annotations

from .api import (
    ConvertConfig,
    ConvertResult,
    Target,
    convert,
    convert_fif,
    convert_snirf,
    convert_batch,
)

__all__ = [
    "ConvertConfig",
    "ConvertResult",
    "Target",
    "convert",
    "convert_fif",
    "convert_snirf",
    "convert_batch",
]

__version__ = "0.1.0"
