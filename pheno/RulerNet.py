"""RulerNet: find the tick marks of a ruler and the pixels-per-centimetre scale.

The ONNX model regresses one tick point, a spacing, a growth ratio and a
direction, from which the whole tick sequence is reconstructed. It runs on a
768x768 letterboxed image, so on a 6000 px photograph the millimetre marks are
almost gone -- which is why a single pass often finds nothing.

This module therefore runs the model over several *hypotheses* (whole frame, the
ruler neighbourhood at native resolution, contrast-boosted crops proposed by a
classical high-frequency search, a rotated frame) and scores every answer against
the image itself before trusting it. The score includes an FFT of the intensity
profile along the tick direction: a ruler is a periodic structure, so the real
pitch is measurable without the network, and a prediction that disagrees with it
is either rejected or snapped onto it.

`get_tickpoints_and_scale` keeps its original contract:
    (tickpoints Nx2 in original image pixels, pixels_per_cm)  or  (None, None)
`get_tickpoints_and_scale_legacy` is the previous two-stage implementation, kept
for A/B measurement (`python -m tools.eval_ruler`).
"""

import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

INPUT_SIZE = 768

# A tick sequence is only believable when it has this many marks and covers a
# noticeable part of the frame; the failures observed in the DUS sets were 3-point
# sequences spanning under 100 px.
MIN_TICKS = 5
MIN_SPAN_RATIO = 0.12
# Reconstructed spacings must be nearly uniform (the model also returns a growth
# ratio for perspective, so a little drift is expected).
MAX_SPACING_CV = 0.12
# Accepted relations between the predicted centimetre spacing and the measured
# pitch: millimetre ticks, centimetre ticks, half-centimetre, two-millimetre.
PITCH_RATIOS = (10.0, 1.0, 2.0, 5.0)
PITCH_TOLERANCE = 0.06
# Real rulers in the DUS sets score 26-235 here; the strongest false positive on
# a tray with no ruler at all scored 10, so the gate sits between them.
MIN_PITCH_STRENGTH = 15.0
# A centimetre cannot be a handful of pixels (the millimetre marks would not be
# resolvable) nor a third of the frame.
MIN_CM_PIXELS = 8.0
MAX_CM_FRACTION = 0.34
# Without a periodicity confirmation the answer is only kept when the geometry is
# exceptionally clean. Every real ruler in the 50-image DUS sample was confirmed by
# the FFT (strength 26-235), so this path exists only for rulers whose marks are
# too coarse to resolve -- and it is deliberately too strict for a row of pods,
# which is what used to slip through (12 ticks, strength 8).
UNVERIFIED_MIN_TICKS = 20
UNVERIFIED_MAX_CV = 0.03
UNVERIFIED_MIN_EDGE_RATIO = 6.0


class RulerNet:
    def __init__(self, model_path="./models/rulernet.onnx"):
        # 加载ONNX模型
        self.model_path = model_path
        # Prefer CUDA when onnxruntime-gpu is present; ORT falls back to CPU
        # with a warning when the provider cannot be initialized. The provider
        # DLL needs cudart/cuDNN — torch's wheel bundles them and registers
        # torch/lib in the DLL search path on import, so make sure torch is
        # loaded first (no-op when torch is absent or already imported).
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.ort_session = ort.InferenceSession(model_path, providers=providers)

    # ------------------------------------------------------------------
    # model plumbing
    # ------------------------------------------------------------------

    def outward_cumsum(self, initial_point, line_direction, directions, n):
        left_directions = directions[n < 0][::-1]
        right_directions = directions[n >= 0]
        left_increments = -np.expand_dims(left_directions, axis=1) * line_direction
        right_increments = np.expand_dims(right_directions, axis=1) * line_direction
        zero = np.zeros((1, 2), dtype=initial_point.dtype)
        left_cumulative = np.cumsum(np.vstack([zero, left_increments]), axis=0)
        right_cumulative = np.cumsum(np.vstack([zero, right_increments]), axis=0)
        left_points = initial_point + left_cumulative
        right_points = initial_point + right_cumulative
        extended_points = np.vstack([left_points[::-1], right_points[1:]])
        return extended_points

    def preprocess_image(self, pil_img):
        image = np.array(pil_img).astype(np.float32) / 255.0
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        resized = np.zeros((768, 768, 3), dtype=np.float32)
        h, w = image.shape[:2]
        scale = min(768 / w, 768 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize((new_w, new_h))).astype(np.float32) / 255.0
        top = (768 - new_h) // 2
        left = (768 - new_w) // 2
        resized[top:top + new_h, left:left + new_w] = image_resized
        input_tensor = np.transpose(resized, (2, 0, 1))[np.newaxis, ...].copy()
        return input_tensor, (top, left, new_h, new_w)

    def infer(self, image_pil):
        image_tensor, _ = self.preprocess_image(image_pil)
        ort_inputs = {"input": image_tensor}
        ort_outs = self.ort_session.run(None, ort_inputs)
        left_point_2d_reconstructed = ort_outs[0][0]
        dist = ort_outs[1][0][0]
        ratio = ort_outs[2][0][0]
        direction = ort_outs[3][0]
        points_info = ort_outs[4][0]
        min_x, min_y, max_x, max_y = points_info[1:].tolist()
        num_points = int(points_info[0])
        n = np.arange(-num_points, num_points + 1)
        directions = (ratio ** n) * dist
        extended_points = self.outward_cumsum(left_point_2d_reconstructed, direction, directions, n)
        within_bounds = (
            (extended_points[:, 0] >= min_x) & 
            (extended_points[:, 0] <= max_x) & 
            (extended_points[:, 1] >= min_y) & 
            (extended_points[:, 1] <= max_y)
        )
        best_generated_points = extended_points[within_bounds]
        if len(best_generated_points) > 1:
            diffs = np.linalg.norm(best_generated_points[:-1] - best_generated_points[1:], axis=1)
            pred_pix_cm = np.nanmedian(diffs)
        else:
            pred_pix_cm = 0.0
        return best_generated_points, pred_pix_cm

    # ------------------------------------------------------------------
    # previous implementation, kept for comparison
    # ------------------------------------------------------------------

    def get_tickpoints_and_scale_legacy(self, img_cv):
        rawImg = Image.fromarray(img_cv)
        best_generated_points, _ = self.infer(rawImg)
        rawImg = np.array(rawImg).astype(np.uint8)
        scale = rawImg.shape[1] / 768
        bias = (rawImg.shape[0] - 768 * scale) / 2
        best_generated_points = best_generated_points * scale
        best_generated_points[:, 1] += bias
        if len(best_generated_points) > 1:
            midpoint = (best_generated_points[0] + best_generated_points[-1]) / 2
            midpoint = midpoint.astype(int)
            center_x, center_y = midpoint
            crop_size = 768 * round(scale / 4)
            half_crop = crop_size // 2
            y1 = max(0, center_y - half_crop)
            y2 = min(rawImg.shape[0], center_y + half_crop)
            x1 = max(0, center_x - half_crop)
            x2 = min(rawImg.shape[1], center_x + half_crop)
            cropped_img = rawImg[y1:y2, x1:x2]
            cropped_img = Image.fromarray(cropped_img)
            tickpoints, pred_pix_cm = self.infer(cropped_img)
            tickpoints = tickpoints * round(scale / 4)
            tickpoints = tickpoints + np.array([x1, y1])
            if x2 - x1 < crop_size:
                tickpoints[:, 0] -= (crop_size - (x2 - x1)) // 2
            if y2 - y1 < crop_size:
                tickpoints[:, 1] -= (crop_size - (y2 - y1)) // 2
            return tickpoints, pred_pix_cm * round(scale / 4)
        else:
            return None, None

    # ------------------------------------------------------------------
    # improved pipeline
    # ------------------------------------------------------------------

    def _infer_region(self, image_rgb, origin=(0, 0)):
        """Run the model on one region and map the ticks back to image pixels.

        `image_rgb` is the region as RGB; `origin` is its top-left corner in the
        full image. Returns (points Nx2, spacing_px) in full-image coordinates.
        The letterbox is undone properly here, for any aspect ratio -- the old
        code assumed a landscape frame and mapped portrait images onto the wrong
        place entirely.
        """
        height, width = image_rgb.shape[:2]
        if height < 16 or width < 16:
            return None, 0.0

        scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
        new_w, new_h = int(width * scale), int(height * scale)
        top = (INPUT_SIZE - new_h) // 2
        left = (INPUT_SIZE - new_w) // 2

        points, spacing = self.infer(Image.fromarray(image_rgb))
        if points is None or len(points) < 2:
            return None, 0.0

        mapped = points.astype(np.float64).copy()
        mapped[:, 0] = (mapped[:, 0] - left) / scale + origin[0]
        mapped[:, 1] = (mapped[:, 1] - top) / scale + origin[1]
        return mapped, float(spacing) / scale

    def _propose_regions(self, image_rgb):
        """Regions worth a second look, ordered cheapest and most likely first."""
        height, width = image_rgb.shape[:2]
        regions = [("full", image_rgb, (0, 0))]

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        # Tick marks are dense fine detail: a ruler lights up in a local
        # high-frequency energy map far more than beans or a colour chart do.
        small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        energy = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        energy = cv2.GaussianBlur(energy, (0, 0), 3)
        threshold = float(np.percentile(energy, 98))
        mask = (energy >= max(threshold, 8)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if max(w, h) < 24:
                continue
            boxes.append((w * h, x * 4, y * 4, w * 4, h * 4))
        boxes.sort(reverse=True)

        for index, (_, x, y, w, h) in enumerate(boxes[:2]):
            pad = int(0.12 * max(w, h)) + 16
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(width, x + w + pad)
            y2 = min(height, y + h + pad)
            if x2 - x1 < 64 or y2 - y1 < 64:
                continue
            crop = image_rgb[y1:y2, x1:x2]
            regions.append((f"energy{index}", crop, (x1, y1)))
            regions.append((f"energy{index}-clahe", self._boost(crop), (x1, y1)))

        return regions

    @staticmethod
    def _boost(image_rgb):
        """Local contrast boost so faint engraved ticks survive the downscale."""
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _band_region(self, image_rgb, points, margin_ratio=0.35):
        """The ruler's own neighbourhood, at native resolution.

        Cropping tight around the detected ticks means the 768 px input is spent
        on the ruler instead of on the whole tray, which is what recovers the
        millimetre marks on a 6000 px photograph.
        """
        height, width = image_rgb.shape[:2]
        xs, ys = points[:, 0], points[:, 1]
        span = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
        if span < 16:
            return None
        margin = max(48.0, span * margin_ratio)
        x1 = int(max(0, xs.min() - margin))
        y1 = int(max(0, ys.min() - margin))
        x2 = int(min(width, xs.max() + margin))
        y2 = int(min(height, ys.max() + margin))
        if x2 - x1 < 64 or y2 - y1 < 64:
            return None
        return image_rgb[y1:y2, x1:x2], (x1, y1)

    # ------------------------------------------------------------------
    # validation against the image itself
    # ------------------------------------------------------------------

    @staticmethod
    def sample_profile(gray, start, end, half_width=12):
        """Mean intensity along the tick line, averaged across a narrow band."""
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        span = end - start
        length = float(np.hypot(*span))
        if length < 32:
            return None, 0.0
        direction = span / length
        normal = np.array([-direction[1], direction[0]])

        count = int(length)
        t = np.linspace(0.0, length, count)
        offsets = np.arange(-half_width, half_width + 1, 2.0)
        grid = start[None, None, :] + t[:, None, None] * direction + offsets[None, :, None] * normal
        xs = np.clip(grid[..., 0], 0, gray.shape[1] - 1).astype(np.int32)
        ys = np.clip(grid[..., 1], 0, gray.shape[0] - 1).astype(np.int32)
        return gray[ys, xs].astype(np.float64).mean(axis=1), length / max(count - 1, 1)

    @staticmethod
    def dominant_period(profile, step, min_period=4.0, max_period=400.0):
        """(period_px, strength) of the strongest periodic component, or (None, 0).

        Deliberately just the strongest peak. Two refinements were tried on the
        50-image DUS sample and both made things worse, so they are recorded here
        instead of shipped:

        * preferring the fundamental over its harmonics (walking towards lower
          frequencies) cost one true detection and added one false positive -- the
          accepted PITCH_RATIOS already absorb a harmonic, because a centimetre
          prediction matches a millimetre, two-millimetre or half-centimetre
          period equally well;
        * widening the detrending window to keep centimetre-only rulers let uneven
          lighting through and had the same cost.

        A 65-tap high-pass therefore stays, which resolves pitches up to roughly
        60 px -- every ruler in the sample is finer than that.
        """
        if profile is None or len(profile) < 64:
            return None, 0.0
        return RulerNet._period_for_window(profile, step, 65, min_period, max_period)

    @staticmethod
    def _period_for_window(profile, step, window, min_period, max_period):
        detrended = profile - cv2.GaussianBlur(profile.reshape(-1, 1), (1, window), 0).ravel()
        windowed = detrended * np.hanning(len(detrended))
        spectrum = np.abs(np.fft.rfft(windowed))
        spectrum[0] = 0.0
        freqs = np.fft.rfftfreq(len(windowed), d=step)
        with np.errstate(divide="ignore"):
            periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-12), np.inf)
        usable = (periods >= min_period) & (periods <= max_period)
        if not usable.any():
            return None, 0.0
        index = int(np.argmax(np.where(usable, spectrum, 0.0)))
        baseline = float(np.median(spectrum[usable])) or 1e-9
        return float(periods[index]), float(spectrum[index] / baseline)

    def _score(self, gray, points, spacing):
        """Grade one candidate answer. Returns (score, refined_spacing, notes)."""
        height, width = gray.shape[:2]
        notes = {}
        if points is None or len(points) < MIN_TICKS or spacing <= 0:
            return 0.0, spacing, {"reject": "too few ticks"}

        span = float(np.hypot(*(points[-1] - points[0])))
        notes["span"] = span
        if span < MIN_SPAN_RATIO * min(height, width):
            return 0.0, spacing, {"reject": "span too short"}

        inside = (
            (points[:, 0] >= -2) & (points[:, 0] <= width + 2)
            & (points[:, 1] >= -2) & (points[:, 1] <= height + 2)
        )
        if inside.mean() < 0.8:
            return 0.0, spacing, {"reject": "ticks outside the frame"}

        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        median = float(np.median(diffs))
        if median <= 0:
            return 0.0, spacing, {"reject": "degenerate spacing"}
        spread = float(np.std(diffs) / median)
        notes["spacing_cv"] = spread
        if spread > MAX_SPACING_CV:
            return 0.0, spacing, {"reject": "irregular spacing"}

        if spacing < MIN_CM_PIXELS or spacing > MAX_CM_FRACTION * max(height, width):
            return 0.0, spacing, {"reject": "implausible centimetre size"}

        score = 1.0 + min(len(points), 40) / 40.0

        # Do the ticks sit on real edges?
        gradient = cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.0), cv2.CV_32F)
        gradient = np.abs(gradient)
        xs = np.clip(points[:, 0].astype(np.int32), 0, width - 1)
        ys = np.clip(points[:, 1].astype(np.int32), 0, height - 1)
        on_tick = float(np.median(gradient[ys, xs]))
        reference = float(np.median(gradient)) or 1e-6
        edge_ratio = on_tick / reference
        notes["edge_ratio"] = edge_ratio
        score += min(edge_ratio, 4.0) / 4.0

        # Does the image actually repeat at this pitch? This is the decisive test:
        # a ruler is periodic, a row of pods is not.
        profile, step = self.sample_profile(gray, points[0], points[-1])
        period, strength = self.dominant_period(profile, step)
        refined = spacing
        notes["fft_period"] = period
        notes["fft_strength"] = strength
        if period and strength >= MIN_PITCH_STRENGTH:
            error, ratio = min((abs(spacing / (period * r) - 1.0), r) for r in PITCH_RATIOS)
            notes["pitch_ratio"] = ratio
            notes["pitch_error"] = error
            if error > PITCH_TOLERANCE:
                # Something periodic is there, but not at the predicted pitch.
                return 0.0, spacing, {**notes, "reject": "pitch disagrees"}
            # The measured period is the more precise of the two, so adopt it.
            refined = period * ratio
            score += 3.0 * (1.0 - error / PITCH_TOLERANCE) + 1.0
        else:
            if (
                len(points) < UNVERIFIED_MIN_TICKS
                or spread > UNVERIFIED_MAX_CV
                or edge_ratio < UNVERIFIED_MIN_EDGE_RATIO
            ):
                return 0.0, spacing, {**notes, "reject": "no periodicity confirmation"}
            score += 0.5
        return score, refined, notes

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    def measure_scale_uniformity(self, image_bgr, points):
        """How much the tick pitch drifts from one end of the rule to the other.

        Purely a diagnostic -- nothing here changes a reported scale. It exists
        because the DUS photographs turned out to carry a real 4.5% drift over a
        40 cm rule (median of 18 images, measured two independent ways), and a
        single pixels-per-centimetre number cannot express that. Knowing the drift
        tells the operator how much position-dependent error to expect: a sample
        spanning a tenth of the rule inherits roughly a tenth of it.

        Radial lens distortion was ruled out as the cause: fitting a k1 that makes
        the ticks equidistant barely reduces the drift (4.54% -> 4.21%) and the
        fitted k1 is inconsistent between photographs, so a chessboard calibration
        would not help. What is left is the camera not being exactly square to the
        table, or the rule itself not lying in the sample plane.

        Returns ``{drift, head_pitch, tail_pitch, span_cm}`` or None.
        """
        if points is None or len(points) < 9:
            return None
        points = np.asarray(points, dtype=np.float64)
        image = np.asarray(image_bgr)
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        )

        start, end = points[0], points[-1]
        first_third = start + (end - start) / 3.0
        last_third = start + 2.0 * (end - start) / 3.0
        pitches = []
        for segment_start, segment_end in ((start, first_third), (last_third, end)):
            profile, step = self.sample_profile(gray, segment_start, segment_end)
            period, strength = self.dominant_period(profile, step)
            pitches.append(period if period and strength >= MIN_PITCH_STRENGTH / 2 else None)

        if not all(pitches):
            # Fall back to the reconstructed spacings, which carry the model's own
            # perspective term but agreed with the FFT to within 0.1% on the sample.
            diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
            third = max(2, len(diffs) // 3)
            pitches = [float(np.median(diffs[:third])), float(np.median(diffs[-third:]))]
            if not all(pitches):
                return None

        head, tail = pitches
        mean_pitch = (head + tail) / 2.0
        return {
            "drift": abs(tail - head) / mean_pitch,
            "head_pitch": head,
            "tail_pitch": tail,
        }

    @staticmethod
    def _merge_peaks(profile, floor, min_dist):
        """Local maxima above ``floor``, with sub-tick duplicates merged.

        A blurry tick can top out across two or three bins, which would count
        as several jittery ticks; candidates closer than ``min_dist`` are
        collapsed into their height-weighted centroid first.
        """
        if len(profile) < 5:
            return np.empty(0)
        centre = (profile[1:-1] >= profile[:-2]) & (profile[1:-1] > profile[2:])
        index = np.where(centre)[0] + 1
        index = index[profile[index] > floor]
        if len(index) == 0:
            return np.empty(0)
        groups = [[index[0]]]
        for i in index[1:]:
            if i - groups[-1][-1] <= min_dist:
                groups[-1].append(i)
            else:
                groups.append([i])
        return np.array([np.average(g, weights=profile[g]) for g in groups])

    def _card_band(self, image_rgb, image_bgr):
        """Crop the card's own scale band so the model can resolve its ticks.

        The 768x768 letterbox shrinks millimetre marks on a full photograph to
        ~1 px, which is why the model says "no ruler". Once the patch lattice
        localises the card, the ruler band is simply the region outside the
        last patch row; a native-resolution crop makes the same model able to
        read it. Returns ``(crop_rgb, (x, y) origin)`` or ``None``.
        """
        from pheno.color import card_lattice

        grid = card_lattice(image_bgr)
        if grid is None:
            return None
        origin, row_vec, col_vec = grid["origin"], grid["A"], grid["B"]
        if float(np.linalg.norm(col_vec)) < 8.0:
            return None

        def point(a, c):
            return origin + a * row_vec + c * col_vec

        # Cover both ends of the card in lattice coordinates: the printed
        # scale sits just below a patch row, and which end that is depends on
        # the card's orientation. A slightly generous quad is fine -- the crop
        # only needs to contain the band, not bound it tightly.
        corners = np.float32(
            [
                point(-3.0, -1.2),
                point(8.6, -1.2),
                point(8.6, 4.2),
                point(-3.0, 4.2),
            ]
        )
        height, width = image_rgb.shape[:2]
        x0 = max(0, int(np.floor(corners[:, 0].min())))
        x1 = min(width, int(np.ceil(corners[:, 0].max())))
        y0 = max(0, int(np.floor(corners[:, 1].min())))
        y1 = min(height, int(np.ceil(corners[:, 1].max())))
        if x1 - x0 < 64 or y1 - y0 < 64:
            return None
        return image_rgb[y0:y1, x0:x1], (x0, y0)

    def _card_ruler(self, image_rgb, image_bgr):
        """Measure the millimetre scale printed on the calibration card itself.

        The patch lattice from ``pheno.color.card_lattice`` fixes the card's
        geometry, so the ruler band is simply the region just outside the last
        patch row -- no detection search is needed. The band is rectified
        through the lattice, the bright tick columns form a periodic profile,
        and the FFT pitch plus the peak spacing must agree. Returns
        ``(tickpoints, pixels_per_cm)`` in original image pixels or
        ``(None, 0.0)`` when the card or its scale cannot be confirmed.
        """
        from pheno.color import card_lattice

        grid = card_lattice(image_bgr)
        if grid is None:
            return None, 0.0
        origin, row_vec, col_vec = grid["origin"], grid["A"], grid["B"]
        unit_pitch = float(np.linalg.norm(col_vec))
        if unit_pitch < 8.0:
            return None, 0.0
        rect_unit = min(unit_pitch, 96.0)
        height, width = image_rgb.shape[:2]
        columns = (-1.0, 4.0)

        def point(a, c):
            return origin + a * row_vec + c * col_vec

        # The printed scale sits below the last patch row in card coordinates;
        # the lattice's row direction is arbitrary, so try both ends. Millimetre
        # ticks are only ~7 px apart, so single peaks jitter too much to score
        # directly -- instead a sliding strip looks for the first strong
        # autocorrelation peak of the column-gradient profile, which is both
        # the periodicity proof and the sub-pixel pitch measurement.
        for start, stop in ((5.55, 7.6), (-2.6, -0.55)):
            w = int(round((columns[1] - columns[0]) * rect_unit))
            h = int(round((stop - start) * rect_unit))
            if w < 64 or h < 16:
                continue
            src = np.float32(
                [point(start, columns[0]), point(start, columns[1]), point(stop, columns[0])]
            )
            dst = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
            band = cv2.warpAffine(image_rgb, cv2.getAffineTransform(src, dst), (w, h))
            gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY).astype(np.float64)
            strip_h = max(10, int(round(0.35 * rect_unit)))
            # Millimetre ticks sit far below the patch-edge period, so the scan
            # stops well under one lattice unit -- otherwise a periodic
            # structure near the patch row can outvote the real scale.
            lag_limit = min(30, w // 4)
            best = None
            for y0 in range(0, h - strip_h + 1, max(4, strip_h // 3)):
                profile = gray[y0 : y0 + strip_h].mean(axis=0)
                detrended = profile - cv2.GaussianBlur(
                    profile.reshape(-1, 1), (1, 33), 0
                ).ravel()
                autocorr = np.concatenate(
                    [
                        [1.0],
                        np.array(
                            [
                                detrended[:-lag] @ detrended[lag:]
                                / (np.linalg.norm(detrended[:-lag]) * np.linalg.norm(detrended[lag:]) + 1e-9)
                                for lag in range(1, lag_limit + 1)
                            ]
                        ),
                    ]
                )
                # first local maximum above the noise floor = the mm pitch
                for lag in range(3, lag_limit):
                    if (
                        autocorr[lag] > 0.45
                        and autocorr[lag] >= autocorr[lag - 1]
                        and autocorr[lag] > autocorr[lag + 1]
                    ):
                        if best is None or autocorr[lag] > best[0]:
                            best = (float(autocorr[lag]), lag, y0, profile)
                        break
            if best is None:
                continue
            corr, lag, y0, profile = best
            detrended = profile - cv2.GaussianBlur(profile.reshape(-1, 1), (1, 33), 0).ravel()
            autocorr = np.concatenate(
                [
                    [1.0],
                    np.array(
                        [
                            detrended[:-l] @ detrended[l:]
                            / (np.linalg.norm(detrended[:-l]) * np.linalg.norm(detrended[l:]) + 1e-9)
                            for l in range(1, lag_limit + 1)
                        ]
                    ),
                ]
            )
            y_minus, y_peak, y_plus = autocorr[lag - 1], autocorr[lag], autocorr[lag + 1]
            denom = y_minus - 2.0 * y_peak + y_plus
            refined_lag = lag + (0.5 * (y_minus - y_plus) / denom if denom else 0.0)
            # Ticks may print dark on a light strip or vice versa; score both
            # polarities of the detrended profile and keep the stronger one.
            # The strip can be taller than the winning window, so re-measure
            # over a slightly taller neighbourhood for fuller tick coverage.
            mid = y0 + strip_h // 2
            wide = gray[
                max(0, mid - 3 * strip_h // 4) : min(h, mid + 3 * strip_h // 4)
            ].mean(axis=0)
            wide_d = wide - cv2.GaussianBlur(wide.reshape(-1, 1), (1, 33), 0).ravel()
            floor = 0.35 * wide_d.std()
            above = self._merge_peaks(wide_d, floor, 0.6 * lag)
            below = self._merge_peaks(-wide_d, floor, 0.6 * lag)
            ticks = above if len(above) >= len(below) else below
            # Keep the longest run of peaks spaced like ticks: the card edge,
            # its corner marks and stray text also peak in the profile, but
            # only the printed scale chains together at ~one lag per step.
            runs = np.split(
                ticks,
                np.where(np.diff(ticks) > 1.8 * lag)[0] + 1,
            )
            ticks = max(runs, key=len) if len(runs) else ticks
            if len(ticks) < 8:
                continue
            pixels_per_cm = 10.0 * refined_lag * unit_pitch / rect_unit
            if not MIN_CM_PIXELS < pixels_per_cm < MAX_CM_FRACTION * max(height, width):
                continue
            middle = start + (y0 + strip_h / 2) / rect_unit
            points = np.array(
                [point(middle, columns[0] + tick / rect_unit) for tick in ticks]
            )
            return points, pixels_per_cm
        return None, 0.0

    def get_tickpoints_and_scale(self, img_cv, debug=False):
        """Best-effort ruler detection over several hypotheses.

        Returns (tickpoints, pixels_per_cm) in original image pixels, or
        (None, None). Set `debug=True` to also get the per-candidate report.
        """
        if img_cv is None or getattr(img_cv, "size", 0) == 0:
            return (None, None, []) if debug else (None, None)

        image = np.asarray(img_cv)
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            # Callers hand over OpenCV BGR; the model was trained on RGB.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        attempts = []
        best = None

        def consider(name, region, origin, local=False):
            nonlocal best
            try:
                points, spacing = self._infer_region(region, origin)
            except Exception as exc:  # noqa: BLE001 - a bad crop must not kill the run
                attempts.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
                return None
            if points is None:
                attempts.append({"name": name, "error": "no ticks"})
                return None
            if local:
                local_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
                score, refined, notes = self._score(local_gray, points - origin, spacing)
            else:
                score, refined, notes = self._score(gray, points, spacing)
            attempts.append({"name": name, "score": score, "spacing": refined, **notes})
            if score > 0 and (best is None or score > best[0]):
                best = (score, points, refined)
            return points

        # 1. whole frame, then the ruler's own neighbourhood at native resolution
        points = consider("full", image_rgb, (0, 0))
        for source, tag in ((points, "band"), (best[1] if best else None, "band-best")):
            if source is None:
                continue
            band = self._band_region(image_rgb, source)
            if band is None:
                continue
            crop, origin = band
            consider(tag, crop, origin)
            if tag == "band":
                consider("band-clahe", self._boost(crop), origin)
            break

        # 2. classical proposals when nothing convincing came out of the frame
        if best is None or best[0] < 3.0:
            for name, crop, origin in self._propose_regions(image_rgb)[1:]:
                consider(name, crop, origin)
                if best is not None and best[0] >= 3.5:
                    break

        # When the model has not found a usable ruler anywhere, localise the
        # colour card geometrically and let it retry on a crop where the
        # printed millimetre ticks are actually resolvable.
        if best is None:
            band = self._card_band(image_rgb, image)
            if band is not None:
                crop, origin = band
                consider("card-band", crop, origin, local=True)
                consider("card-band-clahe", self._boost(crop), origin, local=True)

        # Last resort: measure the printed scale geometrically (autocorrelation
        # of the rectified band profile) without the model at all.
        if best is None:
            points, spacing = self._card_ruler(image_rgb, image)
            if points is not None:
                attempts.append(
                    {"name": "card-ruler", "ticks": len(points), "spacing": spacing}
                )
                best = (4.0, points, spacing)

        # 3. last resort: the model may simply dislike this orientation
        if best is None:
            rotated = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
            try:
                points, spacing = self._infer_region(rotated, (0, 0))
            except Exception as exc:  # noqa: BLE001
                attempts.append({"name": "rot90", "error": f"{type(exc).__name__}: {exc}"})
                points = None
            if points is not None:
                # Undo the rotation: (x, y) in the rotated frame maps back to
                # (y, H_rot - 1 - x) ... expressed in the original frame.
                height = image_rgb.shape[0]
                unrotated = np.stack([points[:, 1], height - 1 - points[:, 0]], axis=1)
                score, refined, notes = self._score(gray, unrotated, spacing)
                attempts.append({"name": "rot90", "score": score, "spacing": refined, **notes})
                if score > 0:
                    best = (score, unrotated, refined)

        if best is None:
            return (None, None, attempts) if debug else (None, None)

        score, points, spacing = best
        return (points, spacing, attempts) if debug else (points, spacing)


if __name__ == '__main__':
    import sys

    img = cv2.imread(sys.argv[1])
    model_processor = RulerNet()
    tickpoints, pred_pix_cm = model_processor.get_tickpoints_and_scale(img)
    print(tickpoints, pred_pix_cm)
