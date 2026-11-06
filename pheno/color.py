"""Colour-checker calibration shared by the desktop app and the web service.

Qt-free on purpose: `pheno/app.py` and `web/calibration.py` both call in here, so
the two front ends cannot drift apart on the numbers they report.

Model choice is measured, not assumed (`python -m tools.eval_color`, five DUS
photographs, leave-one-out over the 24 patches, mean CIE76 dE):

    3x3 + offset, gamma-encoded   5.94   <- used here
    quadratic polynomial          5.59      better on average but its worst held-out
                                            patch is much worse (28 vs 22 dE), and
                                            saturated greens/blacks are exactly what
                                            the beans are
    3x3, gamma-encoded            7.24      what this code used to do
    cv2.ccm (CIE2000, gamma lin)  8.34
    3x3 in linear light           9.74

The counter-intuitive part is that linearising first makes things *worse*. A
camera JPEG is not a plain gamma 2.2 encoding of the scene -- it carries a tone
curve -- so "physically correct" linear-light fitting fights the very non-linearity
the matrix has to absorb. Fitting in the encoded space with an offset wins on
every image tested.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised implicitly by the callers
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class CalibrationError(RuntimeError):
    """The image cannot be colour-calibrated."""


# Nominal sRGB values of the 24 Macbeth / MCC24 patches.
MCC24_REF_SRGB = np.array(
    [
        [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
        [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
        [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
        [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
        [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
        [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
    ],
    dtype=np.float32,
) / 255.0

# How many candidate charts to ask the detector for. More than one matters when
# the frame also holds a grey card or a second, smaller chart.
MAX_CHARTS = 3
# Mean patch error above this means the "chart" is not one, or the patches are
# mismatched; applying that matrix would wreck the picture silently.
MAX_MEAN_DELTA_E = 12.0
# Rows of pixels per block when applying the matrix: keeps the temporary float
# arrays of a 24 MPx photograph down to tens of megabytes.
BLOCK_ROWS = 512


def _to_lab(rgb01: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb01, 0.0, 1.0).astype(np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIE76 colour difference between two sets of sRGB values in [0, 1]."""
    return np.linalg.norm(_to_lab(a) - _to_lab(b), axis=1)


def fit_matrix(detected_rgb: np.ndarray, reference_rgb: np.ndarray = MCC24_REF_SRGB):
    """Least-squares 4x3 matrix (3x3 plus offset) mapping detected -> reference."""
    design = np.hstack([detected_rgb, np.ones((len(detected_rgb), 1), detected_rgb.dtype)])
    matrix, *_ = np.linalg.lstsq(design, reference_rgb, rcond=None)
    return matrix.astype(np.float32)


def apply_matrix(image_rgb01: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x3 (or 3x3) matrix to an RGB image in [0, 1], block by block."""
    height, width = image_rgb01.shape[:2]
    out = np.empty((height, width, 3), dtype=np.float32)
    with_offset = matrix.shape[0] == 4
    linear = matrix[:3]
    offset = matrix[3] if with_offset else None
    for start in range(0, height, BLOCK_ROWS):
        stop = min(height, start + BLOCK_ROWS)
        block = image_rgb01[start:stop].reshape(-1, 3)
        result = block @ linear
        if with_offset:
            result += offset
        np.clip(result, 0.0, 1.0, out=result)
        out[start:stop] = result.reshape(stop - start, width, 3)
    return out


def patch_centers(checker, scale: float = 1.0, offset=(0, 0)) -> list[list[int]]:
    """Centres of the 6x4 patch grid, bilinearly interpolated inside the box.

    ``scale`` maps detector coordinates back to the original image when the
    chart was located on a downscaled copy (see DETECT_LONG_SIDE).
    """
    box = checker.getBox()
    if box is None or len(box) != 4:
        return []
    top_left, top_right, bottom_right, bottom_left = box
    centers = []
    for row in range(4):
        for col in range(6):
            rel_x = (col + 0.5) / 6.0
            rel_y = (row + 0.5) / 4.0
            top_x = top_left[0] * (1 - rel_x) + top_right[0] * rel_x
            top_y = top_left[1] * (1 - rel_x) + top_right[1] * rel_x
            bottom_x = bottom_left[0] * (1 - rel_x) + bottom_right[0] * rel_x
            bottom_y = bottom_left[1] * (1 - rel_x) + bottom_right[1] * rel_x
            centers.append(
                [
                    int((top_x * (1 - rel_y) + bottom_x * rel_y) * scale) + offset[0],
                    int((top_y * (1 - rel_y) + bottom_y * rel_y) * scale) + offset[1],
                ]
            )
    return centers


def _patch_values(checker) -> np.ndarray | None:
    charts = checker.getChartsRGB()
    if charts is None or len(charts) == 0:
        return None
    # Column 1 is the per-patch mean; the rows are R, G, B interleaved.
    values = charts[:, 1].copy().reshape(-1, 3).astype(np.float32) / 255.0
    return values if values.shape == MCC24_REF_SRGB.shape else None


# The detector scans the whole frame, so it is the only MCC step whose cost
# grows with megapixels. Locating the chart on a long-side-capped copy is ~7x
# faster on 24 MP photos; patch means are linear statistics, so the fitted
# matrix is unchanged, and ``coord_scale`` maps the box/centres back.
DETECT_LONG_SIDE = 2000


def _projection_centers(values: np.ndarray, expected: int):
    """``(centers, indices)`` of lattice cells along one axis, or None.

    Humps closer than half a pitch are one cell; gaps of roughly a whole
    pitch are skipped lattice indices, so a fully unsaturated row or column
    does not break the fit. The humps must span the whole grid (first index 0,
    last ``expected - 1``) or the result is ambiguous and rejected.
    """
    low, high = float(values.min()), float(values.max())
    span = high - low
    if span < expected * 4 or not np.isfinite(span):
        return None
    profile = np.histogram(
        values, bins=min(int(round(span)), 600), range=(low, high)
    )[0].astype(np.float64)
    smooth = cv2.GaussianBlur(profile.reshape(1, -1), (1, 5), 0).ravel()
    if smooth.max() <= 0:
        return None
    for fraction in (0.15, 0.3, 0.45, 0.6):
        above = smooth > fraction * smooth.max()
        runs, index = [], 0
        while index < len(above):
            if above[index]:
                end = index
                while end < len(above) and above[end]:
                    end += 1
                runs.append((index, end))
                index = end
            else:
                index += 1
        if len(runs) < 2:
            continue
        centers = np.array(
            [low + (start + end) / 2.0 * span / len(profile) for start, end in runs]
        )
        diffs = np.diff(centers)
        steps = diffs / np.min(diffs)
        if (np.abs(steps - np.rint(steps)) > 0.2).any():
            continue
        indices = np.concatenate([[0], np.rint(steps).astype(np.int64).cumsum()])
        if indices[-1] == expected - 1:
            return centers, indices
    return None


def card_lattice(image_bgr: np.ndarray) -> dict | None:
    """Locate the calibration card's 6x4 patch lattice, in original image pixels.

    Colourful patch centres sit at ``origin + r*A + c*B`` for ``r`` in 0..5 and
    ``c`` in 0..2; the card's neutral column is one B step outside that range
    (which side is resolved later by colour order). Returns None when no
    regular 6x3 colourful grid is found -- the caller then reports that no
    chart was detected rather than guessing.
    """
    if image_bgr is None or image_bgr.ndim != 3 or min(image_bgr.shape[:2]) < 8:
        return None
    if image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
    height, width = image_bgr.shape[:2]
    scale = max(height, width) / 1200.0
    small = cv2.resize(
        image_bgr,
        (max(1, round(width / scale)), max(1, round(height / scale))),
        interpolation=cv2.INTER_AREA,
    )
    saturated = cv2.inRange(
        cv2.cvtColor(small, cv2.COLOR_BGR2HSV), (0, 55, 60), (179, 255, 255)
    )
    # Closing turns the 18 colourful patches into one blob while scattered
    # colourful objects (beans!) stay separate. Only inside such a cluster is
    # the saturated mask trusted to be patch pixels.
    merged = cv2.morphologyEx(
        saturated, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)
    )
    _, labels, merged_stats, _ = cv2.connectedComponentsWithStats(merged)
    candidates = []
    for index in range(1, len(merged_stats)):
        x, y, w, h, area = merged_stats[index]
        if area < 1000 or not 0.3 < w / h < 3.4:
            continue
        candidates.append((area, index, x, y, w, h))
    candidates.sort(reverse=True)
    for _, label, x, y, w, h in candidates[:4]:
        region = (labels[y : y + h, x : x + w] == label) & (saturated[y : y + h, x : x + w] > 0)
        ys, xs = np.nonzero(region)
        if len(xs) < 400:
            continue
        points = np.column_stack([xs, ys]).astype(np.float64)
        mean = points.mean(axis=0)
        origin2d = mean + np.array([x, y], dtype=np.float64)
        _, _, vt = np.linalg.svd(points - mean, full_matrices=False)
        major, minor = vt[0], vt[1]
        along = (points - mean) @ major
        across = (points - mean) @ minor
        rows = _projection_centers(along, 6)
        if rows is None:
            continue
        row_centers, row_index = rows
        row_pitch = float(np.min(np.diff(row_centers)))
        # Per-row projections give each patch its own centre, which is what a
        # sheared or slightly rotated card needs -- a global column projection
        # would smear neighbouring columns into one another.
        observations = []
        for r, centre in zip(row_index, row_centers):
            band = np.abs(along - centre) < 0.4 * row_pitch
            if band.sum() < 20:
                continue
            cols = _projection_centers(across[band], 3)
            if cols is None or len(cols[0]) != 3:
                continue
            for c, across_centre in zip(cols[1], cols[0]):
                observations.append(
                    (r, c, origin2d + centre * major + across_centre * minor)
                )
        if len(observations) < 9:
            continue
        r_idx = np.array([obs[0] for obs in observations])
        c_idx = np.array([obs[1] for obs in observations])
        if (np.bincount(c_idx, minlength=3) < 2).any() or len(set(r_idx.tolist())) < 4:
            continue
        positions = np.array([obs[2] for obs in observations])
        design = np.column_stack([np.ones(len(observations)), r_idx, c_idx])
        sol_x, *_ = np.linalg.lstsq(design, positions[:, 0], rcond=None)
        sol_y, *_ = np.linalg.lstsq(design, positions[:, 1], rcond=None)
        origin = np.array([sol_x[0], sol_y[0]])
        row_vec = np.array([sol_x[1], sol_y[1]])
        col_vec = np.array([sol_x[2], sol_y[2]])
        row_len, col_len = np.linalg.norm(row_vec), np.linalg.norm(col_vec)
        if min(row_len, col_len) <= 0 or not 0.55 < row_len / col_len < 1.8:
            continue
        predicted = origin + np.outer(r_idx, row_vec) + np.outer(c_idx, col_vec)
        if np.median(np.linalg.norm(positions - predicted, axis=1)) > 0.3 * min(row_len, col_len):
            continue
        return {
            "origin": origin * scale,
            "A": row_vec * scale,
            "B": col_vec * scale,
        }
    return None


class _GridChecker:
    def __init__(self, values, box):
        self.values = values
        self.box = box

    def getChartsRGB(self):
        chart = np.zeros((72, 2), dtype=np.float32)
        chart[:, 1] = (self.values * 255).reshape(-1)
        return chart

    def getBox(self):
        return self.box


def _patch_mean(image_bgr: np.ndarray, center: np.ndarray, radius: float):
    x, y = int(round(center[0])), int(round(center[1]))
    r = max(2, int(round(radius)))
    x1, x2 = max(0, x - r), min(image_bgr.shape[1], x + r + 1)
    y1, y2 = max(0, y - r), min(image_bgr.shape[0], y + r + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return image_bgr[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0)[::-1] / 255.0


def _card_candidates(image_bgr: np.ndarray):
    """Yield ``(checker, centers)`` candidates from a detected patch lattice.

    Only the neutral column's side and the card's orientation remain unknown
    after the lattice fit, so the four consistent orderings are evaluated and
    the residual picks the winner -- no window or pose search is involved.
    """
    grid = card_lattice(image_bgr)
    if grid is None:
        return
    origin, row_vec, col_vec = grid["origin"], grid["A"], grid["B"]
    pitch = min(np.linalg.norm(row_vec), np.linalg.norm(col_vec))
    if pitch < 4:
        return
    columns = (-1, 0, 1, 2, 3)
    samples = np.empty((6, len(columns), 3), dtype=np.float64)
    for row in range(6):
        for index, col in enumerate(columns):
            value = _patch_mean(
                image_bgr, origin + row * row_vec + col * col_vec, 0.3 * pitch
            )
            if value is None:
                return
            samples[row, index] = value
    for side in ((0, 1, 2, 3), (-1, 0, 1, 2)):
        idx = [columns.index(col) for col in side]
        grid_values = samples[:, idx, :]
        for flip in (False, True):
            flipped = grid_values[::-1] if flip else grid_values
            for turns in (1, 3):
                values = np.rot90(flipped, turns).reshape(-1, 3).astype(np.float32)
                centers = []
                for i in range(4):
                    for j in range(6):
                        p, q = (5 - j, i) if turns == 3 else (j, 3 - i)
                        row = 5 - p if flip else p
                        point = origin + row * row_vec + side[q] * col_vec
                        centers.append([int(point[0]), int(point[1])])
                yield _GridChecker(values, None), centers


def _process_chart(detector, image, max_charts):
    try:
        return detector.process(image, cv2.mcc.MCC24, max_charts)
    except cv2.error as exc:
        if "takes at most 2 arguments" not in str(exc):
            raise
        return detector.process(image, max_charts)


def detect_charts(image_bgr: np.ndarray, max_charts: int = MAX_CHARTS) -> list:
    """Every MCC24 chart the detector can find, best-fitting one first.

    Fast path first: the OpenCV detector on a long-side-capped copy. When that
    misses (the chart occupies too few pixels after downsampling) the fallback
    locates the patch lattice itself on a 1200 px view and samples at native
    resolution -- both paths produce candidates in original image coordinates.
    """
    if cv2 is None or not hasattr(cv2, "mcc"):
        raise CalibrationError("OpenCV MCC color checker support is unavailable.")
    height, width = image_bgr.shape[:2]
    coord_scale = 1.0
    detect_image = image_bgr
    long_side = max(height, width)
    if long_side > DETECT_LONG_SIDE:
        coord_scale = long_side / float(DETECT_LONG_SIDE)
        detect_image = cv2.resize(
            image_bgr,
            (int(round(width / coord_scale)), int(round(height / coord_scale))),
            interpolation=cv2.INTER_AREA,
        )
    scored = []

    def score(checker, scale, offset, centers=None):
        values = _patch_values(checker)
        if values is None:
            return
        matrix = fit_matrix(values)
        corrected = np.clip(
            np.hstack([values, np.ones((len(values), 1), values.dtype)]) @ matrix, 0.0, 1.0
        )
        errors = delta_e(corrected, MCC24_REF_SRGB)
        scored.append(
            {
                "checker": checker,
                "values": values,
                "matrix": matrix,
                "mean_delta_e": float(np.mean(errors)),
                "max_delta_e": float(np.max(errors)),
                "coord_scale": scale,
                "offset": offset,
                "centers": centers,
            }
        )

    detector = cv2.mcc.CCheckerDetector_create()
    if _process_chart(detector, detect_image, max_charts):
        for checker in detector.getListColorChecker():
            score(checker, coord_scale, (0, 0))

    if not scored or min(item["mean_delta_e"] for item in scored) > MAX_MEAN_DELTA_E:
        for checker, centers in _card_candidates(image_bgr):
            score(checker, 1.0, (0, 0), centers)

    if not scored:
        raise CalibrationError("No MCC24 color checker was detected.")
    # A grey card or a mis-paired chart fits far worse than the real thing, so the
    # residual is what picks the winner rather than detector order.
    scored.sort(key=lambda item: item["mean_delta_e"])
    return scored


def correct_colors(image_bgr: np.ndarray) -> dict:
    """Detect a chart and colour-correct the image.

    Returns a dict with ``corrected`` (BGR uint8), ``centers``, ``matrix`` (4x3),
    ``mean_delta_e``, ``max_delta_e``, ``raw_delta_e`` and ``charts`` (how many
    candidates were considered). Raises CalibrationError when no chart is found or
    the best fit is too poor to trust.
    """
    if cv2 is None:
        raise CalibrationError("OpenCV is required for color calibration.")
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise CalibrationError("Color calibration requires a BGR color image.")
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    candidates = detect_charts(image_bgr)
    best = candidates[0]
    if best["mean_delta_e"] > MAX_MEAN_DELTA_E:
        raise CalibrationError(
            "The detected chart does not match a color checker "
            f"(mean dE {best['mean_delta_e']:.1f}); calibration was skipped."
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    corrected = apply_matrix(image_rgb, best["matrix"])
    corrected_bgr = cv2.cvtColor((corrected * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    return {
        "corrected": corrected_bgr,
        "centers": best.get("centers")
        or patch_centers(best["checker"], best.get("coord_scale", 1.0), best["offset"]),
        "matrix": best["matrix"],
        "mean_delta_e": best["mean_delta_e"],
        "max_delta_e": best["max_delta_e"],
        "raw_delta_e": float(np.mean(delta_e(best["values"], MCC24_REF_SRGB))),
        "charts": len(candidates),
    }
