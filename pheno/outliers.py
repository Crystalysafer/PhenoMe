"""One definition of "is this object an outlier?".

Four components apply the same median rule and must never disagree about it:

* the semantic-detection filter (``pheno/canvas.py``),
* the region-detection filter (``pheno/SAM.py``),
* the batch review window (``pheno/review.py``),
* the cohort analysis used for the manuscript (``tools/cohort_stats.py``).

Each of them used to carry its own copy of the numbers, which is how one copy
quietly drifts away from the others.

The accepted band was widened deliberately. A plate photographed for variety
description legitimately holds organs of very different sizes -- one large pod
next to a small one is the point of the image, not an error -- and the original
0.5x-2.0x band discarded exactly those: a plate with four real pods came back as
three, the largest having been dropped as an "outlier" (detections=4, created=3).

Filtering now starts at six objects per group and tolerates a sixteenfold
spread, which still removes obvious junk -- a speck of dust, a sliver of
background, a half-segmented object -- without arguing with the sample.

The major axis gets a second, parallel check because a drawn-out sliver (a
segmentation artefact stretched along a tray seam, a ruler edge) can sit
inside the area band while being ten times longer than everything else.
"""

import cv2
import numpy as np

# An object is an outlier when its area lies outside this multiple of the median
# of its own group.
AREA_LOW_RATIO = 0.25
AREA_HIGH_RATIO = 4.0

# ...and when its major axis does. For isometric organs a 4x area spread is
# only a 2x length spread, but elongated organs (pods) legitimately vary in
# length, so the band keeps extra slack -- it still rejects a tenfold sliver,
# which is the failure mode that prompted this criterion.
AXIS_LOW_RATIO = 0.35
AXIS_HIGH_RATIO = 3.0

# Groups smaller than this are never filtered: a median over a handful of organs
# describes that sample, it does not define a population to judge it against.
MIN_GROUP_FOR_OUTLIERS = 6

# SAM masks systematically over-include the object boundary by ~1.8-2.5 px
# per side (measured by sampling image intensity along mask-contour normals:
# the mask edge sits ~1.8 px beyond the true intensity transition on clean
# boundaries, more on fuzzy ones). Contracting the mask before measuring
# removes this additive boundary bias; the rim is a boundary-localization
# effect and roughly constant in pixels, so a fixed pixel radius is honest.
EDGE_EROSION_PX = 2


def contract_mask(mask):
    """Return ``mask`` shrunk by EDGE_EROSION_PX for trait measurement.

    An object that would erode away entirely keeps its original mask --
    better a slightly loose measurement than none at all.
    """
    binary = np.asarray(mask, dtype=np.uint8)
    if EDGE_EROSION_PX <= 0 or not np.any(binary):
        return binary
    k = 2 * EDGE_EROSION_PX + 1
    eroded = cv2.erode(binary, np.ones((k, k), np.uint8))
    return eroded if np.any(eroded) else binary


def area_is_outlier(area, median):
    """True when ``area`` lies outside the accepted band around ``median``."""
    if not area or not median:
        return True
    return area < AREA_LOW_RATIO * median or area > AREA_HIGH_RATIO * median


def mask_major_axis(mask):
    """Longer side of the mask's minimum-area rectangle, in pixels.

    Returns 0 for an empty mask; a degenerate contour (a point or a line)
    falls back to its bounding-box side so callers always get a length.
    The mask is contracted by EDGE_EROSION_PX first so the axis matches the
    convention used for reported traits.
    """
    binary = contract_mask(mask)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        axis = max(cv2.minAreaRect(contour)[1])
        if axis > 0:
            return float(axis)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return 0.0
    return float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def axis_is_outlier(axis, median):
    """True when ``axis`` lies outside the accepted band around ``median``."""
    if not axis or not median:
        return True
    return axis < AXIS_LOW_RATIO * median or axis > AXIS_HIGH_RATIO * median


def outlier_reason(area, median, axis=None, axis_median=None):
    """Short human-readable reason for the review table."""
    if not area:
        return "empty mask"
    reasons = []
    if area_is_outlier(area, median):
        reasons.append("area %.2fx median" % (area / median))
    if axis is not None and axis_median and axis_is_outlier(axis, axis_median):
        reasons.append("axis %.2fx median" % (axis / axis_median))
    return "; ".join(reasons) or "area %.2fx median" % (area / median)
