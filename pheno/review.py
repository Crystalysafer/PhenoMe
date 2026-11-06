"""Batch result review: scan a folder of sidecars, summarise and flag outliers.

This backs the standalone review window (``pheno/reviewDialog.py``). It is kept
free of Qt widgets so the scanning and statistics can be tested headlessly and
reused by the web front-end if needed.

The point of this module is that a finished batch run leaves behind one
``.pheno.json`` per image; those already contain the exact masks, so traits can
be recomputed and cross-image summaries built without re-running SAM.
"""

import csv
import os

import numpy as np

from pheno import annotation
from pheno.calcPheno import PhenomicsCalculator
from pheno.outliers import (
    MIN_GROUP_FOR_OUTLIERS,
    area_is_outlier,
    axis_is_outlier,
    contract_mask,
    mask_major_axis,
    outlier_reason,
)

IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
)


def find_sidecars(folder):
    """Every ``*.pheno.json`` directly inside ``folder``, sorted by name."""
    if not folder or not os.path.isdir(folder):
        return []
    suffix = annotation.SIDECAR_SUFFIX
    found = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.endswith(suffix)
    ]
    found.sort(key=lambda path: os.path.basename(path).lower())
    return found


def locate_image(sidecar_path, document, extra_dirs=()):
    """Best effort lookup of the image a sidecar belongs to.

    Batch output normally lives in a different folder than the images, so the
    sidecar's own directory is tried first and then any directories the caller
    knows about. Returns ``None`` when nothing matches.
    """
    image_info = document.get("image") or {}
    recorded = image_info.get("path")
    recorded_dir = image_info.get("dir") or ""
    stem = os.path.basename(sidecar_path)[: -len(annotation.SIDECAR_SUFFIX)]

    candidates = []
    # 新 sidecar 记了原图的完整目录 —— 之后的会话直接按它回找，
    # 不依赖当时打开过的文件夹。
    if recorded and recorded_dir:
        candidates.append(os.path.join(recorded_dir, recorded))
    search_dirs = [os.path.dirname(sidecar_path)]
    search_dirs.extend(directory for directory in extra_dirs if directory)

    for directory in search_dirs:
        if recorded:
            candidates.append(os.path.join(directory, recorded))
        for extension in IMAGE_EXTENSIONS:
            candidates.append(os.path.join(directory, stem + extension))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def load_entry(sidecar_path, extra_dirs=()):
    """Read one sidecar into a review entry (no traits computed yet)."""
    document = annotation.load_document(sidecar_path)
    if document is None:
        return None

    stem = os.path.basename(sidecar_path)[: -len(annotation.SIDECAR_SUFFIX)]
    size = annotation.document_image_size(document)
    image_info = document.get("image") or {}
    recorded_name = image_info.get("path") or ""
    recorded_dir = image_info.get("dir") or ""
    recorded_path = (os.path.join(recorded_dir, recorded_name)
                     if recorded_dir and recorded_name else "")
    digest = image_info.get("md5") or ""
    image_path = locate_image(sidecar_path, document, extra_dirs)

    # 原图完整性：找不到任何候选 -> missing；找到的文件字节对不上记录 ->
    # 被同名文件覆盖（replaced）；记录位置空了、从备选目录找到同内容的 ->
    # moved（可用但记警告）。哈希只读一遍文件，图多时秒级开销换完整性。
    if image_path is None:
        image_state = "missing"
    elif digest and annotation.image_digest(image_path) != digest:
        image_state = "replaced"
    elif (recorded_path
          and os.path.normcase(os.path.normpath(image_path))
          != os.path.normcase(os.path.normpath(recorded_path))):
        image_state = "moved"
    else:
        image_state = ""
    return {
        "sidecar": sidecar_path,
        "name": stem,
        "document": document,
        "image_path": image_path,
        "recorded_path": recorded_path,
        "image_digest": digest,
        "image_state": image_state,
        "image_size": size,
        "object_count": len(document.get("objects") or []),
        "scale_per_pixel": float((document.get("scale") or {}).get("per_pixel", 1.0)),
        "scale_unit": str((document.get("scale") or {}).get("unit", "pixel")),
    }


def scan_folder(folder, extra_dirs=()):
    """Load every sidecar in ``folder`` into review entries."""
    entries = []
    for path in find_sidecars(folder):
        entry = load_entry(path, extra_dirs=extra_dirs)
        if entry is not None:
            entries.append(entry)
    return entries


def trait_image_path(entry):
    """Which file the traits/preview should be measured from.

    When the batch run colour-corrected an image it wrote the corrected pixels
    to ``<stem>_color_corrected.jpg`` next to the sidecar. Colour traits must be
    measured from those pixels — recomputing MeanRGB on the raw image would
    report pre-calibration colours the pipeline never actually used.
    """
    if not (entry.get("document", {}).get("processing") or {}).get(
            "color_correction"):
        return entry.get("image_path")
    candidate = os.path.join(
        os.path.dirname(entry["sidecar"]),
        entry["name"] + "_color_corrected.jpg")
    if os.path.isfile(candidate):
        return candidate
    return entry.get("image_path")


def decode_objects(entry):
    """Decode the masks of one entry. Returns ``[(object dict, mask), ...]``."""
    document = entry["document"]
    size = entry.get("image_size")
    decoded = []
    for data in document.get("objects") or []:
        mask = None
        payload = data.get("mask")
        if payload:
            try:
                mask = annotation.decode_mask(payload, image_size=size)
            except (KeyError, ValueError):
                mask = None
        decoded.append((data, mask))
    return decoded


def _mask_area(mask):
    # contract first so the flagged area matches the reported trait convention
    return int((contract_mask(mask) > 0).sum()) if mask is not None else 0


def mark_outliers(rows):
    """Flag rows whose area or major axis deviates from their group's median.

    Grouping is by ``name_prefix`` so pods are only compared with pods. Groups
    smaller than :data:`MIN_GROUP_FOR_OUTLIERS` are left untouched, since a
    median over two or three objects says nothing.
    """
    groups = {}
    for row in rows:
        key = (row.get("name_prefix") or "").strip().lower()
        groups.setdefault(key, []).append(row)

    for group in groups.values():
        areas = [row["area_px"] for row in group if row["area_px"] > 0]
        if len(areas) < MIN_GROUP_FOR_OUTLIERS:
            for row in group:
                row["outlier"] = False
                row["outlier_reason"] = ""
            continue

        median = float(np.median(areas))
        axes = [row["axis_px"] for row in group if row["axis_px"] > 0]
        median_axis = (
            float(np.median(axes))
            if len(axes) >= MIN_GROUP_FOR_OUTLIERS
            else None
        )
        for row in group:
            area = row["area_px"]
            axis = row["axis_px"]
            axis_bad = (
                median_axis is not None and axis_is_outlier(axis, median_axis)
            )
            if area_is_outlier(area, median) or axis_bad:
                row["outlier"] = True
                row["outlier_reason"] = outlier_reason(
                    area, median, axis, median_axis)
            else:
                row["outlier"] = False
                row["outlier_reason"] = ""
    return rows


def build_rows(entries, progress=None):
    """Recompute traits for every object of every entry.

    ``progress(index, total, name)`` is called before each entry so a dialog can
    show where it is. Entries whose image cannot be found still contribute rows,
    but only geometry traits: the colour algorithms need pixels.
    """
    from collections import Counter
    from qtpy import QtGui  # Imported lazily; this module stays Qt-free otherwise.

    # Headers carry the measurement unit ("Area (mm²)" vs "Area (px²)"), and the
    # batch that wrote the sidecars already recorded one per file. Use the unit
    # most entries share rather than the pixel default, or a millimetre batch
    # reads back in pixels.
    units = [entry["scale_unit"] for entry in entries if entry.get("scale_unit")]
    unit = Counter(units).most_common(1)[0][0] if units else None
    headers = PhenomicsCalculator.getHeaders(unit)
    rows = []
    total = len(entries)

    for index, entry in enumerate(entries):
        if progress is not None:
            progress(index, total, entry["name"])

        pixmap = None
        source = trait_image_path(entry)
        if source:
            image = QtGui.QImage(source)
            if not image.isNull():
                pixmap = QtGui.QPixmap.fromImage(image)

        # 原图被挪动/覆盖的行级提示 —— 性状仍照常算，错误列注明出处问题；
        # 校色图是管线自有产物，不参与原图哈希校验。
        measuring_replaced = (source is not None
                              and source == entry.get("image_path")
                              and entry.get("image_state") == "replaced")

        scale = entry["scale_per_pixel"]
        for data, mask in decode_objects(entry):
            row = {
                "image": entry["name"],
                "label": data.get("label") or "",
                "name_prefix": data.get("name_prefix") or "",
                "source_prompt": data.get("source_prompt") or "",
                "detection_score": data.get("detection_score"),
                "area_px": _mask_area(mask),
                "axis_px": mask_major_axis(mask) if mask is not None else 0.0,
                "values": None,
                "error": "",
                "outlier": False,
                "outlier_reason": "",
            }

            if mask is None:
                row["error"] = "mask missing"
            elif pixmap is None:
                row["error"] = ("image moved or deleted" if source is None
                                else "image unreadable")
            else:
                shape = _TraitShape(data.get("label") or "", mask)
                try:
                    row["values"] = PhenomicsCalculator.calcPhenomics(
                        shape, pixmap, scale)
                except Exception as error:  # noqa: BLE001 - reported per row
                    row["error"] = str(error)
                if not row["error"] and measuring_replaced:
                    row["error"] = ("warning: image replaced since "
                                    "sidecar was written")
            rows.append(row)

        entry["object_count"] = len(entry["document"].get("objects") or [])

    mark_outliers(rows)
    if progress is not None:
        progress(total, total, "")
    return headers, rows


class _TraitShape:
    """Minimal stand-in for ``Shape`` so calcPhenomics can be reused as-is."""

    def __init__(self, label, mask):
        self.label = label
        self.mask = mask


def summary_headers(trait_headers):
    """Column names of the exported wide table."""
    return (
        ["image"]
        + list(trait_headers)
        + ["name_prefix", "source_prompt", "detection_score",
           "area_px", "outlier", "outlier_reason", "error"]
    )


def summary_rows(trait_headers, rows):
    """Flatten review rows into exportable lists, one per object."""
    blank = [""] * len(trait_headers)
    flat = []
    for row in rows:
        values = row["values"]
        if values is None:
            # Keep the Name column meaningful even when traits failed.
            values = [row["label"]] + blank[1:]
        flat.append(
            [row["image"]]
            + list(values)
            + [
                row["name_prefix"],
                row["source_prompt"],
                "" if row["detection_score"] is None else row["detection_score"],
                row["area_px"],
                "yes" if row["outlier"] else "",
                row["outlier_reason"],
                row["error"],
            ]
        )
    return flat


def export_summary(path, trait_headers, rows):
    """Write the merged wide table (one row per object)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(summary_headers(trait_headers))
        writer.writerows(summary_rows(trait_headers, rows))
    return path
