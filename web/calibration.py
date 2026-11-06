"""Qt-free color checker calibration for the web service.

Thin wrapper: the actual model lives in `pheno/color.py` so the desktop app and
this service cannot disagree about a calibrated pixel value. See that module for
why the matrix is fitted in the encoded sRGB space with an offset.
"""

from __future__ import annotations

import numpy as np

from pheno.color import (  # noqa: F401 - re-exported for callers and tests
    MAX_MEAN_DELTA_E,
    MCC24_REF_SRGB,
    CalibrationError,
    apply_matrix,
    correct_colors,
    delta_e,
    fit_matrix,
)


def color_correct_bgr(image: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Detect an MCC24 chart and return ``(corrected_bgr, patch_centers)``."""
    result = correct_colors(image)
    return result["corrected"], result["centers"]


def color_correct_with_report(image: np.ndarray) -> dict:
    """Same as above but keeps the fit diagnostics (dE, matrix, chart count)."""
    return correct_colors(image)
