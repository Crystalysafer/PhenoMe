"""Sidecar annotation persistence for PhenoMe.

One image gets one companion file: ``<image stem>.pheno.json``. It stores every
object of that image, including its binary mask, so a session can be reopened
without re-running SAM.

Masks are the primary data here, not the polygons: the polygon in a ``Shape`` is
only a simplified contour of the mask, so persisting polygons alone would make
recomputed traits drift. Each mask is therefore cropped to its bounding box,
run-length encoded column-first, and stored as zlib-compressed base64. On real
4624x3472 pod masks that is about 1.0 KB per object versus 19.9 KB for the 8-bit
PNG currently exported.

The wire format is documented in ``docs/ANNOTATION_FORMAT.md`` and is also
consumed by the web front-end, so changes here must stay in sync with that file.
"""

import base64
import hashlib
import json
import os
import zlib

import numpy as np


FORMAT_NAME = "phenome-annotation"
FORMAT_VERSION = 1
SIDECAR_SUFFIX = ".pheno.json"
# 批处理导出的校色图后缀（``<stem>_color_corrected.jpg``）。它没有自己的
# sidecar —— 从属于去掉该后缀的同名原图那一份。
CORRECTED_SUFFIX = "_color_corrected"

# Mask payload encodings understood by :func:`decode_mask`.
ENCODING_RLE_Z85 = "rle-zlib-base64"

_COUNTS_DTYPE = np.uint32


def sidecar_path(image_path):
    """Return the annotation path that belongs to ``image_path``."""
    stem, _ext = os.path.splitext(str(image_path))
    return stem + SIDECAR_SUFFIX


def image_digest(image_path):
    """MD5 of the image bytes, used to detect a stale sidecar."""
    digest = hashlib.md5()
    try:
        with open(image_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Mask codec
# ---------------------------------------------------------------------------


def rle_encode(mask):
    """Column-first run-length encode a binary mask.

    Returns a ``uint32`` array of run lengths that alternates background /
    foreground and always *starts* with a background run, so ``counts[0]`` is 0
    when the very first pixel is foreground. This matches the COCO convention
    and keeps the decoder branch-free.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    # Column-first (Fortran) order: transpose then flatten C-style.
    flat = np.ascontiguousarray(binary.T).reshape(-1)
    if flat.size == 0:
        return np.zeros(0, dtype=_COUNTS_DTYPE)

    change_points = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change_points, [flat.size]))
    lengths = np.diff(bounds)
    if flat[0] != 0:
        lengths = np.concatenate(([0], lengths))
    return lengths.astype(_COUNTS_DTYPE)


def rle_decode(counts, shape):
    """Inverse of :func:`rle_encode`. ``shape`` is ``(height, width)``."""
    height, width = int(shape[0]), int(shape[1])
    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0 or height <= 0 or width <= 0:
        return np.zeros((max(height, 0), max(width, 0)), dtype=np.uint8)

    total = int(counts.sum())
    if total != height * width:
        raise ValueError(
            "RLE run lengths sum to %d but the shape needs %d pixels"
            % (total, height * width)
        )

    values = np.zeros(counts.size, dtype=np.uint8)
    values[1::2] = 1
    flat = np.repeat(values, counts)
    return flat.reshape(width, height).T


def encode_mask(mask):
    """Encode a full-image mask into a compact, JSON-safe dict.

    ``None`` is returned for an empty mask so callers can skip the field.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or not binary.any():
        return None

    columns = np.flatnonzero(binary.any(axis=0))
    rows = np.flatnonzero(binary.any(axis=1))
    x1, x2 = int(columns[0]), int(columns[-1])
    y1, y2 = int(rows[0]), int(rows[-1])

    crop = np.ascontiguousarray(binary[y1:y2 + 1, x1:x2 + 1])
    counts = rle_encode(crop)
    payload = base64.b64encode(
        zlib.compress(counts.tobytes(), 9)
    ).decode("ascii")

    return {
        "encoding": ENCODING_RLE_Z85,
        "image_size": [int(binary.shape[1]), int(binary.shape[0])],
        "bbox": [x1, y1, x2, y2],
        "counts": payload,
    }


def decode_mask(payload, image_size=None):
    """Rebuild a full-size ``uint8`` mask from :func:`encode_mask` output."""
    if not payload:
        return None

    encoding = payload.get("encoding", ENCODING_RLE_Z85)
    if encoding != ENCODING_RLE_Z85:
        raise ValueError("Unsupported mask encoding: %s" % encoding)

    size = payload.get("image_size") or image_size
    if not size:
        raise ValueError("Mask payload is missing image_size")
    width, height = int(size[0]), int(size[1])

    x1, y1, x2, y2 = (int(value) for value in payload["bbox"])
    counts = np.frombuffer(
        zlib.decompress(base64.b64decode(payload["counts"])),
        dtype=_COUNTS_DTYPE,
    )
    crop = rle_decode(counts, (y2 - y1 + 1, x2 - x1 + 1))

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2 + 1, x1:x2 + 1] = crop
    return mask


# ---------------------------------------------------------------------------
# Shape (de)serialisation
# ---------------------------------------------------------------------------


def _points_to_list(points):
    """``[QPointF, ...]`` -> ``[[x, y], ...]``.

    Coordinates keep 1 decimal: polygons come out of ``find_contours`` on half-
    pixel boundaries, so ``.5`` is meaningful while further digits are noise.
    """
    result = []
    for point in points or []:
        try:
            x, y = point.x(), point.y()
        except AttributeError:
            x, y = point[0], point[1]
        result.append([round(float(x), 1), round(float(y), 1)])
    return result


def shape_to_dict(shape):
    """Serialise one ``Shape``. ``uid`` is internal and intentionally omitted."""
    data = {
        "label": shape.label,
        "name_prefix": shape.name_prefix,
        "source_prompt": shape.source_prompt,
        "polygon": _points_to_list(shape.points),
    }

    score = getattr(shape, "detection_score", None)
    if score is not None:
        data["detection_score"] = round(float(score), 4)

    prompt_points = _points_to_list(getattr(shape, "prompt_points", None))
    if prompt_points:
        labels = list(getattr(shape, "prompt_labels", []) or [])
        data["prompt_points"] = [
            {
                "x": int(round(point[0])),
                "y": int(round(point[1])),
                "label": int(labels[index]) if index < len(labels) else 1,
            }
            for index, point in enumerate(prompt_points)
        ]

    mask = encode_mask(shape.mask) if shape.mask is not None else None
    if mask is not None:
        data["mask"] = mask
    return data


def apply_dict_to_shape(shape, data, image_size=None):
    """Populate an existing ``Shape`` from :func:`shape_to_dict` output."""
    mask = None
    if data.get("mask"):
        mask = decode_mask(data["mask"], image_size=image_size)

    polygon = data.get("polygon") or []
    shape.set(points=polygon, mask=mask, label=data.get("label"))
    shape.name_prefix = data.get("name_prefix")
    shape.source_prompt = data.get("source_prompt")
    shape.detection_score = data.get("detection_score")

    for point in data.get("prompt_points") or []:
        shape.add_prompt_point(
            float(point.get("x", 0)),
            float(point.get("y", 0)),
            int(point.get("label", 1)),
        )
    return shape


# ---------------------------------------------------------------------------
# Document level helpers
# ---------------------------------------------------------------------------


def _jsonable(value):
    """Recursively convert numpy containers/scalars to JSON-safe natives."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def build_document(
    image_path,
    shapes,
    image_size,
    scale_per_pixel=1.0,
    scale_unit="pixel",
    color_correction=False,
    ruler_detection=False,
    digest=None,
    color_checker_centers=None,
    ruler_tickpoints=None,
    correction_matrix=None,
):
    """Assemble the complete sidecar document for one image."""
    width, height = int(image_size[0]), int(image_size[1])
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "image": {
            "path": os.path.basename(str(image_path)),
            # 完整目录 —— 审查窗之后随时打开都能按记录位置找回原图，
            # 不依赖当时会话里的搜索目录。老 sidecar 没有此字段时
            # 仍退回按文件名在 sidecar 旁/搜索目录里找。
            "dir": os.path.dirname(os.path.abspath(str(image_path))),
            "width": width,
            "height": height,
            "md5": digest if digest is not None else image_digest(image_path),
        },
        "scale": {
            "per_pixel": float(scale_per_pixel),
            "unit": str(scale_unit),
        },
        "processing": {
            "color_correction": bool(color_correction),
            "ruler_detection": bool(ruler_detection),
            # 检测点位也落盘：审查/回编辑时能把色卡中心和标尺刻度画回来，
            # 不必为重画标记而重跑检测模型。老 sidecar 没有这两个字段时跳过。
            # 注意用 is not None 判空：调用方传的是 numpy 数组，`or`/`if x`
            # 会触发 ndarray 真值歧义 ValueError。
            **({"color_checker_centers": _jsonable(color_checker_centers)}
               if color_checker_centers is not None else {}),
            **({"ruler_tickpoints": _jsonable(ruler_tickpoints)}
               if ruler_tickpoints is not None else {}),
            # The fitted 4x3 matrix lets a reopen restore the corrected view
            # in ~100 ms instead of re-running chart detection -- and is far
            # smaller than storing a second image.
            **({"correction_matrix": _jsonable(correction_matrix)}
               if correction_matrix is not None else {}),
        },
        "objects": [shape_to_dict(shape) for shape in shapes],
    }


def save_document(path, document):
    """Write ``document`` as compact UTF-8 JSON, atomically.

    Compact separators matter here: the polygon arrays dominate the file, and
    pretty-printing them one number per line inflates a 3-object sidecar from
    ~25 KB to ~61 KB for no practical benefit.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temp_path, path)
    return path


def load_document(path):
    """Read a sidecar file, or return ``None`` when it is absent/unreadable."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return None

    if not isinstance(document, dict):
        return None
    if document.get("format") != FORMAT_NAME:
        return None
    return document


def document_image_size(document):
    """``(width, height)`` recorded in the document, or ``None``."""
    image = (document or {}).get("image") or {}
    width, height = image.get("width"), image.get("height")
    if width and height:
        return int(width), int(height)
    return None
