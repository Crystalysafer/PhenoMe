"""Per-session analysis state for the web service.

Everything here is pure CPU work: object bookkeeping, naming, scale, trait
computation and annotation export. GPU inference lives in ``web/engine.py``.

Desktop code is reused rather than reimplemented:

* ``pheno.lib.compute_polygon_from_mask`` for contours,
* ``pheno.canvas.next_label_for_prefix`` for object names,
* ``pheno.calcPheno.PhenomicsCalculator`` for traits,
* ``pheno.annotation`` for the ``.pheno.json`` sidecar contract.

``PhenomicsCalculator.calcPhenomics`` itself is skipped because it takes a
``QPixmap``; the registered algorithms and headers are driven directly with the
decoded BGR array instead, which is exactly what ``qimage_to_cv2`` would have
produced on the desktop.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from pheno import annotation
from pheno.calcPheno import PhenomicsCalculator
from pheno.canvas import (
    DEFAULT_LABEL_PREFIX,
    _filter_text_mask_candidates,
    _text_mask_traits,
    next_label_for_prefix,
)
from pheno.lib import compute_polygon_from_mask, sanitize_label_prefix


# Decoded 4624x3472 BGR frames are ~48 MB each, so only the few most recently
# used images stay unpacked; everything else is re-decoded on demand.
MAX_DECODED_IMAGES = 3
# Color-corrected frames are a second full-size copy per image, so they get a
# tighter cap of their own: 24 analyses each holding one would be ~1.2 GB.
MAX_CORRECTED_IMAGES = 2
MAX_ANALYSIS_ENTRIES = 24
MAX_OBJECTS_PER_IMAGE = 200

# Phones record the sensor orientation in an EXIF tag instead of rotating the
# pixels. Two things then have to agree: the pixels the analyser works on, and
# the size the client is told about, because the browser lays out and converts
# clicks with the size it was given.
#
# OpenCV's imdecode honours the tag by itself (as browsers do), while Pillow's
# Image.size reports the stored dimensions and ignores it. Reading the tag here
# lets the upload endpoint swap width and height so both sides describe the same
# upright frame. Before this, a sideways phone photo was analysed upright but
# announced sideways, and every mask landed rotated and stretched against the
# organ the user had clicked.
EXIF_ORIENTATION_TAG = 0x0112
# Orientations that exchange width and height once applied.
EXIF_SWAPPED_ORIENTATIONS = (5, 6, 7, 8)


def read_exif_orientation(data: bytes) -> int:
    """EXIF orientation of an uploaded image, or 1 when absent/unreadable."""
    try:
        with Image.open(BytesIO(data)) as image:
            value = image.getexif().get(EXIF_ORIENTATION_TAG, 1)
        orientation = int(value or 1)
    except Exception:  # noqa: BLE001 - a broken EXIF block must not fail an upload
        return 1
    return orientation if 1 <= orientation <= 8 else 1


class AnalysisError(RuntimeError):
    """Invalid request against analysis state (bad name, unknown object, ...)."""


def _json_number(value):
    """Make a numeric result JSON-safe (NaN/Inf are not valid JSON)."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return round(number, 6)
    if isinstance(value, (int, bool, str)):
        return value
    return str(value)


class AnalysisObject:
    """One segmented object.

    Field names mirror ``pheno.shape.Shape`` so ``pheno.annotation`` can
    serialise it unchanged, but nothing here touches Qt.
    """

    def __init__(self):
        self.points: list[list[float]] = []
        self.mask = None
        self.label = None
        self.uid = None
        self.name_prefix = None
        self.source_prompt = None
        self.detection_score = None
        self.prompt_points: list[list[float]] = []
        self.prompt_labels: list[int] = []

    def set(self, points=None, mask=None, label=None):
        if points is not None:
            self.points = [
                [round(float(point[0]), 1), round(float(point[1]), 1)]
                for point in points
            ]
        if mask is not None:
            self.mask = (np.asarray(mask) > 0).astype(np.uint8)
        if label is not None:
            self.label = label
        return self

    def add_prompt_point(self, x, y, label):
        self.prompt_points.append([float(x), float(y)])
        self.prompt_labels.append(int(label))

    @property
    def area_pixels(self) -> int:
        if self.mask is None:
            return 0
        return int(np.count_nonzero(self.mask))

    @property
    def bbox(self):
        if self.mask is None:
            return None
        columns = np.flatnonzero(self.mask.any(axis=0))
        rows = np.flatnonzero(self.mask.any(axis=1))
        if columns.size == 0 or rows.size == 0:
            return None
        return [int(columns[0]), int(rows[0]), int(columns[-1]), int(rows[-1])]

    def to_public(self) -> dict:
        return {
            "uid": self.uid,
            "label": self.label,
            "name_prefix": self.name_prefix,
            "source_prompt": self.source_prompt,
            "detection_score": _json_number(self.detection_score),
            "area_pixels": self.area_pixels,
            "bbox": self.bbox,
            "polygon": self.points,
            "prompt_points": [
                {
                    "x": int(round(point[0])),
                    "y": int(round(point[1])),
                    "label": int(self.prompt_labels[index])
                    if index < len(self.prompt_labels)
                    else 1,
                }
                for index, point in enumerate(self.prompt_points)
            ],
        }


def polygon_from_mask(mask) -> list[list[float]]:
    """``compute_polygon_from_mask`` output as a JSON-ready ``[[x, y], ...]``."""
    polygon = compute_polygon_from_mask(np.asarray(mask))
    return [[round(float(x), 1), round(float(y), 1)] for x, y in polygon]


# The next two wrap private desktop helpers on purpose. Semantic detection has
# to reject the same outliers as the desktop, and re-deriving the median-based
# statistics here would be a second implementation free to drift; importing the
# originals keeps one definition of "obvious outlier". They are named without the
# underscore so the routes do not reach into ``pheno.canvas`` internals directly.


def mask_traits(mask):
    """Area / aspect ratio / extent / solidity of one mask, or ``None``."""
    return _text_mask_traits(mask)


def filter_mask_candidates(candidates, minimum_group_size=4):
    """Split semantic-detection candidates into ``(accepted, filtered)``."""
    return _filter_text_mask_candidates(candidates, minimum_group_size)


def trait_headers(unit=None) -> list[str]:
    """Column names of the trait table, identical to the desktop export.

    ``unit`` is the current scale unit, so the headers carry it: a column of bare
    numbers is unreadable without knowing whether they are pixels or millimetres,
    and that was exactly the complaint about the exported table.
    """
    return list(PhenomicsCalculator.getHeaders(unit))


def compute_traits(image_bgr, objects, scale_per_pixel=1.0) -> list[list]:
    """Run every registered algorithm over ``objects``.

    A headless copy of ``PhenomicsCalculator.calcPhenomics``: same algorithm
    registry, same per-algorithm error padding, but the image arrives as a BGR
    array instead of a ``QPixmap``.
    """
    rows = []
    for obj in objects:
        if obj.mask is None:
            continue
        mask = (np.asarray(obj.mask) > 0).astype(np.uint8)
        row = [obj.label]
        for name, func in PhenomicsCalculator.custom_algorithms.items():
            headers = PhenomicsCalculator.custom_headers.get(name)
            if headers is None:
                continue
            width = len(headers) if isinstance(headers, list) else 1
            try:
                result = func(image_bgr, mask, scale_per_pixel)
            except Exception:
                row.extend([None] * width)
                continue
            if isinstance(result, list):
                row.extend(_json_number(item) for item in result)
            else:
                row.append(_json_number(result))
        rows.append(row)
    return rows


class ImageAnalysis:
    """Objects, scale and pending click state for one image of one session."""

    def __init__(self, session_id, image_id, filename, width, height, digest=None):
        self.session_id = session_id
        self.image_id = image_id
        self.filename = filename
        self.width = int(width)
        self.height = int(height)
        self.digest = digest

        self.objects: list[AnalysisObject] = []
        self.total_objects = 0
        self.scale_per_pixel = 1.0
        self.scale_unit = "pixel"
        self.color_correction = False
        self.ruler_detection = False
        self.color_centers: list[list[int]] = []
        # How well the chart fitted (mean/max dE and the error before correction).
        # Reported to the client so a poor calibration is visible rather than silent.
        self.color_quality: dict | None = None
        self.ruler_ticks: list[list[float]] = []
        self.corrected_image = None
        # Small JPEG of the corrected frame for the browser preview. Kept even
        # when the full-size array is evicted, since it is ~0.3 MB rather than
        # ~48 MB and re-encoding 16 MPx per request is wasteful.
        self.corrected_preview = None

        self.lock = threading.RLock()
        self._pending = None

    # ------------------------------------------------------------------
    # Objects
    # ------------------------------------------------------------------

    @property
    def key(self) -> str:
        return "%s:%s" % (self.session_id, self.image_id)

    def clamp(self, x, y):
        cx = min(max(float(x), 0.0), float(self.width - 1))
        cy = min(max(float(y), 0.0), float(self.height - 1))
        return cx, cy

    def register(self, obj: AnalysisObject, prefix=None) -> AnalysisObject:
        """Assign uid and display name, mirroring ``Canvas.registerShape``."""
        with self.lock:
            if len(self.objects) >= MAX_OBJECTS_PER_IMAGE:
                raise AnalysisError(
                    "This image already holds %d objects." % MAX_OBJECTS_PER_IMAGE
                )
            self.total_objects += 1
            obj.uid = self.total_objects
            if prefix is not None:
                obj.name_prefix = prefix
            if not obj.label:
                obj.label = next_label_for_prefix(
                    self.objects, obj.name_prefix or DEFAULT_LABEL_PREFIX
                )
            self.objects.append(obj)
            return obj

    def find(self, uid) -> AnalysisObject | None:
        with self.lock:
            for obj in self.objects:
                if obj.uid == int(uid):
                    return obj
            return None

    def rename(self, uid, label) -> AnalysisObject:
        text = str(label or "").strip()
        if not text:
            raise AnalysisError("The object name must not be empty.")
        with self.lock:
            target = self.find(uid)
            if target is None:
                raise AnalysisError("Unknown object.")
            for obj in self.objects:
                if obj is not target and obj.label == text:
                    raise AnalysisError("Another object is already named %s." % text)
            target.label = text
            return target

    def remove(self, uid) -> AnalysisObject:
        with self.lock:
            target = self.find(uid)
            if target is None:
                raise AnalysisError("Unknown object.")
            self.objects.remove(target)
            return target

    def clear_objects(self) -> int:
        with self.lock:
            count = len(self.objects)
            self.objects = []
            return count

    def selected(self, uids=None) -> list[AnalysisObject]:
        with self.lock:
            if not uids:
                return list(self.objects)
            wanted = [int(uid) for uid in uids]
            found = []
            for uid in wanted:
                obj = self.find(uid)
                if obj is None:
                    raise AnalysisError("Unknown object %s." % uid)
                found.append(obj)
            return found

    # ------------------------------------------------------------------
    # Click preview cache
    # ------------------------------------------------------------------

    @staticmethod
    def signature(points) -> tuple:
        return tuple((int(x), int(y), int(label)) for x, y, label in points)

    def remember_preview(self, points, mask, polygon) -> None:
        with self.lock:
            self._pending = {
                "signature": self.signature(points),
                "mask": mask,
                "polygon": polygon,
            }

    def take_preview(self, points):
        """Reuse the mask from the matching preview click, if any."""
        with self.lock:
            pending = self._pending
            if pending is None:
                return None
            if pending["signature"] != self.signature(points):
                return None
            return pending

    def clear_preview(self) -> None:
        with self.lock:
            self._pending = None

    # ------------------------------------------------------------------
    # Scale, traits, annotation
    # ------------------------------------------------------------------

    def set_scale(self, per_pixel, unit) -> dict:
        value = float(per_pixel)
        if not math.isfinite(value) or value <= 0:
            raise AnalysisError("The scale must be a positive number.")
        text = str(unit or "pixel").strip() or "pixel"
        if len(text) > 16:
            raise AnalysisError("The unit name is too long.")
        with self.lock:
            self.scale_per_pixel = value
            self.scale_unit = text
            return self.scale_public()

    def set_color_correction(self, corrected_image, centers) -> None:
        with self.lock:
            self.corrected_image = np.asarray(corrected_image).copy()
            self.corrected_preview = None
            self.color_centers = [
                [int(round(point[0])), int(round(point[1]))] for point in centers
            ]
            self.color_correction = True

    def release_corrected_image(self) -> bool:
        """Drop the cached corrected frame, keeping the fact that it was made.

        The patch centers and the ``color_correction`` flag stay: they are what
        the overlay and the annotation document need. Only the ~48 MB pixel copy
        goes, and it can be recomputed from the original on demand.
        """
        with self.lock:
            had_image = self.corrected_image is not None
            self.corrected_image = None
            return had_image

    def set_ruler_detection(self, tickpoints, per_pixel) -> dict:
        with self.lock:
            self.ruler_ticks = [
                [round(float(point[0]), 1), round(float(point[1]), 1)]
                for point in tickpoints
            ]
            self.ruler_detection = True
            self.scale_per_pixel = float(per_pixel)
            self.scale_unit = "mm"
            return self.scale_public()

    def scale_public(self) -> dict:
        return {"per_pixel": self.scale_per_pixel, "unit": self.scale_unit}

    def traits(self, image_bgr, uids=None) -> dict:
        objects = self.selected(uids)
        rows = compute_traits(image_bgr, objects, self.scale_per_pixel)
        return {
            "headers": trait_headers(self.scale_unit),
            "rows": rows,
            "scale": self.scale_public(),
        }

    def document(self) -> dict:
        with self.lock:
            return annotation.build_document(
                self.filename,
                list(self.objects),
                (self.width, self.height),
                scale_per_pixel=self.scale_per_pixel,
                scale_unit=self.scale_unit,
                color_correction=self.color_correction,
                ruler_detection=self.ruler_detection,
                digest=self.digest,
            )

    def to_public(self) -> dict:
        with self.lock:
            return {
                "session_id": self.session_id,
                "image_id": self.image_id,
                "filename": self.filename,
                "width": self.width,
                "height": self.height,
                "scale": self.scale_public(),
                "color_correction": self.color_correction,
                "color_centers": list(self.color_centers),
                "color_quality": self.color_quality,
                "ruler_detection": self.ruler_detection,
                "ruler_ticks": list(self.ruler_ticks),
                "objects": [obj.to_public() for obj in self.objects],
                "object_count": len(self.objects),
            }


class AnalysisStore:
    """Bounded registry of :class:`ImageAnalysis` plus a decoded-image cache."""

    def __init__(self, max_entries=MAX_ANALYSIS_ENTRIES, max_decoded=MAX_DECODED_IMAGES,
                 max_corrected=MAX_CORRECTED_IMAGES):
        self._entries: "OrderedDict[tuple[str, str], ImageAnalysis]" = OrderedDict()
        self._decoded: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._corrected: "OrderedDict[tuple[str, str], ImageAnalysis]" = OrderedDict()
        self._max_entries = max_entries
        self._max_decoded = max_decoded
        self._max_corrected = max(1, int(max_corrected))
        self._lock = threading.RLock()

    def get_or_create(self, session_id, image_id, record) -> ImageAnalysis:
        key = (session_id, image_id)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = ImageAnalysis(
                    session_id=session_id,
                    image_id=image_id,
                    filename=record["filename"],
                    width=record["width"],
                    height=record["height"],
                    digest=hashlib.md5(record["data"]).hexdigest(),
                )
                self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return entry

    def get(self, session_id, image_id) -> ImageAnalysis | None:
        with self._lock:
            return self._entries.get((session_id, image_id))

    def drop_image(self, image_id) -> list[str]:
        """Forget every session's analysis of ``image_id``; returns engine keys."""
        dropped = []
        with self._lock:
            for key in [k for k in self._entries if k[1] == image_id]:
                entry = self._entries.pop(key)
                self._corrected.pop(key, None)
                dropped.append(entry.key)
            self._decoded.pop(image_id, None)
        return dropped

    def drop_session(self, session_id) -> list[str]:
        dropped = []
        with self._lock:
            for key in [k for k in self._entries if k[0] == session_id]:
                entry = self._entries.pop(key)
                self._corrected.pop(key, None)
                dropped.append(entry.key)
        return dropped

    def note_corrected(self, analysis: ImageAnalysis) -> int:
        """Register a freshly corrected frame and evict the oldest ones.

        Returns how many corrected frames were released.
        """
        key = (analysis.session_id, analysis.image_id)
        released = 0
        with self._lock:
            self._corrected[key] = analysis
            self._corrected.move_to_end(key)
            while len(self._corrected) > self._max_corrected:
                _old_key, old_analysis = self._corrected.popitem(last=False)
                if old_analysis.release_corrected_image():
                    released += 1
        return released

    def decode(self, image_id, data) -> np.ndarray:
        """Decoded BGR array for an uploaded image, cached with an LRU bound."""
        with self._lock:
            cached = self._decoded.get(image_id)
            if cached is not None:
                self._decoded.move_to_end(image_id)
                return cached

        # cv2 applies the EXIF orientation on its own (OpenCV >= 3.1), exactly as
        # a browser does when it displays the image, so the pixels below are
        # already upright. Do not rotate a second time here: the fix belongs in
        # the size reported to the client, which is why _inspect_image swaps
        # width and height for the rotated orientations.
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise AnalysisError("The stored image could not be decoded.")

        with self._lock:
            self._decoded[image_id] = image
            self._decoded.move_to_end(image_id)
            while len(self._decoded) > self._max_decoded:
                self._decoded.popitem(last=False)
        return image

    def stats(self) -> dict:
        with self._lock:
            return {
                "analyses": len(self._entries),
                "decoded_images": len(self._decoded),
                "corrected_images": len(self._corrected),
                "analysis_limit": self._max_entries,
                "corrected_limit": self._max_corrected,
            }


def sanitize_prefix(prefix) -> str:
    """Public wrapper so routes do not import from ``pheno.lib`` directly."""
    return sanitize_label_prefix(prefix)
