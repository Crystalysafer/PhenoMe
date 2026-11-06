import csv
import json
import logging
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Literal
from urllib.parse import quote
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi import Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from web.analysis import (
    EXIF_ORIENTATION_TAG,
    EXIF_SWAPPED_ORIENTATIONS,
    AnalysisError,
    AnalysisObject,
    AnalysisStore,
    filter_mask_candidates,
    mask_traits,
    polygon_from_mask,
    sanitize_prefix,
    trait_headers,
)
from web.calibration import CalibrationError, color_correct_with_report
from web.engine import (
    MAX_PROMPT_POINTS,
    EngineBusy,
    EngineError,
    EngineNotReady,
    EngineUnavailable,
    SamEngine,
    model_label,
)
from web.ruler import RulerDetectionError, detect_ruler
from pheno.TextDetector import TextDetector


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ICON_DIR = BASE_DIR.parent / "pheno" / "icons"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_SESSION_IMAGES = 20
# Sized for a shared LAN/behind-nginx deployment: one phone browser holds one
# session, and each session is only a bookkeeping dict plus its image bytes,
# which are bounded separately by MAX_SESSION_IMAGES and the image cache.
MAX_SESSIONS = 64
# The image cache used to be capped at MAX_SESSION_IMAGES *globally*, which meant
# the 21st visitor evicted the first one's photo and silently broke their analysis.
# With the service open to the public it is bounded by bytes instead: the host has
# 64 GB, so this is generous while still refusing to grow without limit.
MAX_CACHED_IMAGE_BYTES = 1536 * 1024 * 1024
SESSION_IDLE_SECONDS = 30 * 60
# Corrected frames are only ever displayed inside the canvas, so the browser copy
# is capped instead of shipping a re-encode of the full 16 MPx frame.
PREVIEW_MAX_SIDE = 2048
PREVIEW_JPEG_QUALITY = 88
RULER_MODEL_PATH = BASE_DIR.parent / "models" / "rulernet.onnx"
SEMANTIC_MODEL_DIR = BASE_DIR.parent / "models" / "grounding-dino-tiny"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


# The Chinese dictionaries are the fast, deterministic path for detection prompts.
# This plugs a model in behind them for the terms they do not know, and writes each
# answer to a cache so a term is only ever translated once -- see web/translate.py.
# Unset by default if there is no API key, and switchable with
# PHENOME_TRANSLATE_FALLBACK=off.
from pheno import prompt_zh as _prompt_zh
from web import translate as _translate

_prompt_zh.set_fallback_translator(_translate.translate)

app = FastAPI(title="PhenoMe Web", version="0.3.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Reuse the desktop application's logo instead of maintaining a web copy.
app.mount("/assets", StaticFiles(directory=ICON_DIR), name="assets")

_images: OrderedDict[str, dict] = OrderedDict()
_sessions: OrderedDict[str, dict] = OrderedDict()
_images_lock = Lock()

_store = AnalysisStore()
_engine = None
_engine_lock = Lock()
_ruler_lock = Lock()
_semantic_lock = Lock()
_semantic_detector = None
_cleanup_stop = Event()
_cleanup_thread = None


def get_engine():
    """The process-wide SAM engine, created on first use (model still lazy)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = SamEngine()
        return _engine


def set_engine(engine) -> None:
    """Test seam: replace the inference engine without touching the routes."""
    global _engine
    with _engine_lock:
        _engine = engine


def get_store() -> AnalysisStore:
    return _store


def semantic_model_available() -> bool:
    return (
        (SEMANTIC_MODEL_DIR / "config.json").is_file()
        and (SEMANTIC_MODEL_DIR / "preprocessor_config.json").is_file()
        and any((SEMANTIC_MODEL_DIR / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
    )


# Free GPU memory needed before the Grounding DINO detector may live on CUDA.
# On the 8 GB Wuhan box vit_h + embeddings left ~700 MB free, so it had to stay
# on CPU; the SAM2-Tiny backend frees enough that GPU inference is safe again.
SEMANTIC_MIN_FREE_GPU_GB = 2.0


def _semantic_device() -> str:
    forced = os.getenv("PHENOME_DINO_DEVICE")
    if forced:
        return forced
    try:
        import torch  # noqa: PLC0415 - deferred; the detector may never be used

        if not torch.cuda.is_available():
            return "cpu"
        free_bytes, _total = torch.cuda.mem_get_info(0)
        if free_bytes / 1024 ** 3 >= SEMANTIC_MIN_FREE_GPU_GB:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_semantic_detector() -> TextDetector:
    """Lazily create the local Grounding DINO detector.

    It runs on CUDA when there is enough free GPU memory and on CPU otherwise,
    because a CPU DINO is the difference between seconds and tens of seconds
    per semantic-detection pass.
    """
    global _semantic_detector
    with _semantic_lock:
        if _semantic_detector is None:
            _semantic_detector = TextDetector(device=_semantic_device(), model_dir=SEMANTIC_MODEL_DIR)
        return _semantic_detector


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(filename: str | None) -> str:
    return Path(filename or "capture.jpg").name


def _inspect_image(data: bytes) -> tuple[int, int, str]:
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            image_format = (image.format or "unknown").lower()
            orientation = int(image.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid image.",
        ) from exc

    # The analysis path rotates the pixels to match the EXIF tag, so the size
    # reported to the client has to describe the rotated frame, not the stored
    # one -- otherwise every coordinate the browser sends back is off.
    if orientation in EXIF_SWAPPED_ORIENTATIONS:
        width, height = height, width

    if width < 1 or height < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image has invalid dimensions.",
        )
    return width, height, image_format


def _public_image(image_id: str, record: dict) -> dict:
    return {
        "id": image_id,
        "filename": record["filename"],
        "content_type": record["content_type"],
        "width": record["width"],
        "height": record["height"],
        "format": record["format"],
        "size_bytes": len(record["data"]),
        "created_at": record["created_at"],
        "url": f"/api/images/{image_id}",
    }


def _public_session(session_id: str, record: dict) -> dict:
    return {
        "id": session_id,
        "source": record["source"],
        "label": record["label"],
        "notes": record["notes"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "images": [
            _public_image(image_id, _images[image_id])
            for image_id in record["image_ids"]
            if image_id in _images
        ],
    }


def _forget_analysis(keys) -> None:
    """Release cached embeddings for analyses that no longer exist."""
    if not keys:
        return
    engine = get_engine()
    for key in keys:
        try:
            engine.drop(key)
        except Exception:
            pass


def _drop_session_state(session_id: str, record: dict) -> None:
    for image_id in record["image_ids"]:
        _images.pop(image_id, None)
    _forget_analysis(_store.drop_session(session_id))


def _expire_idle_sessions() -> None:
    """Discard abandoned browser sessions without persisting uploads to disk."""
    cutoff = time.monotonic() - SESSION_IDLE_SECONDS
    dropped_keys: list[str] = []
    with _images_lock:
        expired = [
            session_id
            for session_id, record in _sessions.items()
            if record.get("last_activity", 0) < cutoff
        ]
        for session_id in expired:
            record = _sessions.pop(session_id)
            for image_id in record["image_ids"]:
                _images.pop(image_id, None)
            dropped_keys.extend(_store.drop_session(session_id))
    _forget_analysis(dropped_keys)


def _busy(exc: Exception) -> HTTPException:
    """503 with Retry-After for a full GPU queue.

    A queue-full condition is explicitly *not* an error in the request: telling
    the client to come back beats making it wait behind eleven other jobs until
    the 300 s proxy timeout kills it.
    """
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(exc),
        headers={"Retry-After": "5"},
    )


def _cached_image_bytes_locked() -> int:
    """Total bytes held in the image cache. Caller must hold ``_images_lock``."""
    return sum(len(record.get("data") or b"") for record in _images.values())


def _evict_surplus_sessions_locked() -> None:
    """Trim to ``MAX_SESSIONS`` by dropping the *most idle* session first.

    Caller must hold ``_images_lock``.

    This used to pop the oldest entry in insertion order, which on a public URL
    is a one-line denial of service: creating ``MAX_SESSIONS`` + 1 sessions
    destroys the state of everyone who started earlier, no matter how actively
    they are working. Ordering by last activity means a flood of fresh, unused
    sessions evicts itself instead of live users.
    """
    dropped_keys: list[str] = []
    while len(_sessions) > MAX_SESSIONS:
        victim = min(
            _sessions.items(), key=lambda item: item[1].get("last_activity", 0.0)
        )[0]
        record = _sessions.pop(victim)
        for image_id in record["image_ids"]:
            _images.pop(image_id, None)
        dropped_keys.extend(_store.drop_session(victim))
    if dropped_keys:
        _forget_analysis(dropped_keys)


def _touch_session(record: dict) -> None:
    record["last_activity"] = time.monotonic()


def _session_cleanup_worker() -> None:
    while not _cleanup_stop.wait(60):
        _expire_idle_sessions()


@app.on_event("startup")
def start_session_cleanup() -> None:
    global _cleanup_thread
    if _cleanup_thread is None or not _cleanup_thread.is_alive():
        _cleanup_stop.clear()
        _cleanup_thread = Thread(
            target=_session_cleanup_worker,
            name="phenome-web-session-cleanup",
            daemon=True,
        )
        _cleanup_thread.start()


@app.on_event("shutdown")
def stop_session_cleanup() -> None:
    _cleanup_stop.set()


class SessionCreate(BaseModel):
    source: str = Field(default="workspace", pattern="^(workspace|capture)$")
    label: str = Field(default="", max_length=120)
    notes: str = Field(default="", max_length=1000)


class ImageRef(BaseModel):
    session_id: str
    image_id: str


class PromptPoint(BaseModel):
    x: float
    y: float
    label: Literal[1] = 1


class ClickRequest(ImageRef):
    points: list[PromptPoint] = Field(min_length=1, max_length=MAX_PROMPT_POINTS)
    commit: bool = False
    name_prefix: str | None = Field(default=None, max_length=60)


class RenameRequest(ImageRef):
    label: str = Field(min_length=1, max_length=120)


class ScaleRequest(ImageRef):
    per_pixel: float = Field(gt=0)
    unit: str = Field(default="pixel", max_length=16)


class TraitsRequest(ImageRef):
    object_ids: list[int] | None = None


class SemanticDetectRequest(ImageRef):
    prompt: str = Field(min_length=1, max_length=240)
    # The desktop shows a "this image does not look like X" dialog with a
    # "continue anyway" button; this is that button.
    skip_semantic_check: bool = False
    # Organs the caller's word could mean, when it names no organ of its own: a crop name
    # like 扁豆 means the pod or the seed, never the flower. The image then picks between
    # them. See TextDetector.resolve_organ.
    candidates: list[str] = Field(default_factory=list, max_length=4)


def _require_analysis(session_id: str, image_id: str):
    """Resolve an (analysis, image record) pair or raise the right 4xx."""
    _expire_idle_sessions()
    with _images_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
            )
        record = _images.get(image_id)
        if record is None or image_id not in session["image_ids"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found in this session.",
            )
        _touch_session(session)
    analysis = _store.get_or_create(session_id, image_id, record)
    return analysis, record


def _analysis_payload(analysis, engine) -> dict:
    payload = analysis.to_public()
    payload["embedding"] = engine.embedding_status(analysis.key)
    payload["model"] = engine.status()
    # Headers carry the current unit, so a client that only reads this payload
    # still knows whether the numbers are pixels or millimetres.
    payload["trait_headers"] = trait_headers(analysis.scale_unit)
    return payload


def _processed_image(analysis, record: dict):
    """The pixels analysis should run on: corrected when calibration succeeded.

    The corrected frame is a bounded cache (see ``AnalysisStore.note_corrected``),
    so it can be evicted while ``color_correction`` stays true. Recomputing beats
    silently falling back to the raw pixels, because otherwise traits and the CSV
    would change depending on how many other images the service happened to touch
    in between.
    """
    if analysis.corrected_image is not None:
        return analysis.corrected_image.copy()

    original = _store.decode(analysis.image_id, record["data"])
    if not analysis.color_correction:
        return original

    try:
        report = color_correct_with_report(original)
    except CalibrationError:
        return original
    analysis.set_color_correction(report["corrected"], report["centers"])
    _store.note_corrected(analysis)
    return analysis.corrected_image.copy()


def _csv_text(headers, rows) -> str:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue()


def _corrected_image_url(session_id: str, image_id: str) -> str:
    return "/api/analysis/corrected-image?session_id=%s&image_id=%s" % (
        quote(session_id),
        quote(image_id),
    )


def _corrected_preview(analysis, record: dict) -> bytes:
    """Cached, display-sized JPEG of the corrected frame.

    The browser only ever shows this scaled into the canvas, and overlay
    coordinates come from the recorded image size rather than from this file, so
    capping the long side keeps a phone from pulling a multi-megabyte re-encode
    of a 16 MPx frame on every repaint.
    """
    if analysis.corrected_preview is not None:
        return analysis.corrected_preview

    image = _processed_image(analysis, record)
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > PREVIEW_MAX_SIDE:
        factor = PREVIEW_MAX_SIDE / float(longest)
        image = cv2.resize(
            image,
            (max(1, int(width * factor)), max(1, int(height * factor))),
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY]
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The corrected image could not be encoded.",
        )
    analysis.corrected_preview = encoded.tobytes()
    return analysis.corrected_preview


def _points_tuples(points) -> list[tuple[float, float, int]]:
    return [(point.x, point.y, int(point.label)) for point in points]


def _clamped_points(analysis, points) -> list[tuple[float, float, int]]:
    clamped = []
    for x, y, label in _points_tuples(points):
        cx, cy = analysis.clamp(x, y)
        clamped.append((cx, cy, label))
    return clamped


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    engine = get_engine()
    model = engine.status()
    return {
        "status": "ok",
        "service": "phenome-web",
        "model_backend": model_label(model["model_type"]),
        "model_state": model["state"],
        "device": model["device"],
        "timestamp": _utc_now(),
    }


@app.get("/api/help")
def help_text(lang: str = Query(default="zh")) -> dict:
    """The user guide, from the same source the desktop dialog renders.

    Served rather than duplicated in the front-end bundle so the two front ends
    cannot describe the same buttons differently.
    """
    from pheno.help_content import WEB, help_document

    return help_document(lang, WEB)


@app.get("/api/capabilities")
def capabilities() -> dict:
    engine = get_engine()
    model = engine.status()
    return {
        "service_version": app.version,
        "model_backend": model_label(model["model_type"]),
        "model_state": model["state"],
        "checkpoint_available": model["checkpoint_available"],
        "desktop": {
            "image_session": "available",
            "object_segmentation": "available",
            "semantic_detection": "available" if semantic_model_available() else "unavailable",
            "region_detection": "excluded",
            "color_calibration": "available",
            "ruler_detection": "available" if RULER_MODEL_PATH.is_file() else "unavailable",
            "scale": "available",
            "phenomics": "available",
            "custom_algorithms": "planned",
            "export": "traits_table",
            "batch": "excluded",
        },
        "capture": {
            "camera_input": "available",
            "image_session": "available",
            "object_segmentation": "available",
            "semantic_detection": "available" if semantic_model_available() else "unavailable",
            "color_calibration": "available",
            "ruler_detection": "available" if RULER_MODEL_PATH.is_file() else "unavailable",
            "scale": "available",
            "phenomics": "available",
            "export": "traits_table",
            "batch": "excluded",
        },
        "limits": {
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "session_images": MAX_SESSION_IMAGES,
            "max_sessions": MAX_SESSIONS,
            "session_idle_seconds": SESSION_IDLE_SECONDS,
            "max_prompt_points": MAX_PROMPT_POINTS,
            "accepted_content_types": sorted(ALLOWED_CONTENT_TYPES),
        },
        "store": _store.stats(),
        "data_policy": "ephemeral-memory-only",
    }


@app.post("/api/sessions", status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreate) -> dict:
    session_id = uuid4().hex
    now = _utc_now()
    record = {
        "source": payload.source,
        "label": payload.label.strip(),
        "notes": payload.notes.strip(),
        "created_at": now,
        "updated_at": now,
        "last_activity": time.monotonic(),
        "image_ids": [],
    }
    _expire_idle_sessions()
    with _images_lock:
        _sessions[session_id] = record
        _evict_surplus_sessions_locked()
    return _public_session(session_id, record)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    _expire_idle_sessions()
    with _images_lock:
        record = _sessions.get(session_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        _touch_session(record)
        return _public_session(session_id, record)


@app.post("/api/images", status_code=status.HTTP_201_CREATED)
async def upload_image(
    file: UploadFile = File(...),
    session_id: str | None = Query(default=None),
) -> dict:
    _expire_idle_sessions()
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Supported image types are JPEG, PNG, and WebP.",
        )

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image is empty.",
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The uploaded image exceeds the 25 MB limit.",
        )

    width, height, image_format = _inspect_image(data)
    image_id = uuid4().hex
    record = {
        "filename": _safe_filename(file.filename),
        "content_type": file.content_type,
        "width": width,
        "height": height,
        "format": image_format,
        "created_at": _utc_now(),
        "data": data,
    }

    evicted: list[str] = []
    with _images_lock:
        session = None
        if session_id is not None:
            session = _sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
            if len(session["image_ids"]) >= MAX_SESSION_IMAGES:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="The session image limit has been reached.")
        _images[image_id] = record
        _images.move_to_end(image_id)
        # Evict by total bytes, oldest first, and never evict the image that was
        # just uploaded -- a single 25 MB photo must not push itself out.
        while len(_images) > 1 and _cached_image_bytes_locked() > MAX_CACHED_IMAGE_BYTES:
            old_image_id, _ = _images.popitem(last=False)
            for old_session in _sessions.values():
                if old_image_id in old_session["image_ids"]:
                    old_session["image_ids"].remove(old_image_id)
            evicted.extend(_store.drop_image(old_image_id))
        if session is not None:
            session["image_ids"].append(image_id)
            session["updated_at"] = _utc_now()
            _touch_session(session)

    _forget_analysis(evicted)

    response = _public_image(image_id, record)
    if session_id is not None:
        response["session_id"] = session_id
    return response


@app.get("/api/images/{image_id}")
def get_image(image_id: str) -> Response:
    _expire_idle_sessions()
    with _images_lock:
        record = _images.get(image_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return Response(
        content=record["data"],
        media_type=record["content_type"],
        headers={"Cache-Control": "no-store"},
    )


@app.delete("/api/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(image_id: str) -> Response:
    _expire_idle_sessions()
    with _images_lock:
        record = _images.pop(image_id, None)
        for session in _sessions.values():
            if image_id in session["image_ids"]:
                session["image_ids"].remove(image_id)
                session["updated_at"] = _utc_now()
                _touch_session(session)
        dropped = _store.drop_image(image_id)
    _forget_analysis(dropped)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@app.get("/api/model")
def model_status() -> dict:
    return get_engine().status()


@app.post("/api/model/load", status_code=status.HTTP_202_ACCEPTED)
def model_load() -> JSONResponse:
    engine = get_engine()
    payload = engine.request_load()
    if payload["state"] == "error":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload
        )
    code = status.HTTP_200_OK if payload["state"] == "ready" else status.HTTP_202_ACCEPTED
    return JSONResponse(status_code=code, content=payload)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@app.get("/api/analysis")
def analysis_state(
    session_id: str = Query(...),
    image_id: str = Query(...),
) -> dict:
    engine = get_engine()
    analysis, _record = _require_analysis(session_id, image_id)
    return _analysis_payload(analysis, engine)


@app.post("/api/analysis/embedding")
def analysis_embedding(payload: ImageRef) -> dict:
    engine = get_engine()
    analysis, record = _require_analysis(payload.session_id, payload.image_id)

    def loader():
        return _store.decode(payload.image_id, record["data"])

    try:
        state = engine.prepare(analysis.key, loader)
    except EngineBusy as exc:
        raise _busy(exc) from exc
    except EngineUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return {
        "session_id": payload.session_id,
        "image_id": payload.image_id,
        "embedding": state,
        "model": engine.status(),
    }


@app.post("/api/analysis/click")
def analysis_click(payload: ClickRequest) -> dict:
    engine = get_engine()
    analysis, _record = _require_analysis(payload.session_id, payload.image_id)
    points = _clamped_points(analysis, payload.points)
    if not any(label == 1 for _x, _y, label in points):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one positive point is required.",
        )

    reused = analysis.take_preview(points) if payload.commit else None
    if reused is not None:
        mask, polygon = reused["mask"], reused["polygon"]
    else:
        try:
            mask = engine.segment(analysis.key, points)
        except EngineBusy as exc:
            raise _busy(exc) from exc
        except EngineNotReady as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        except EngineUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
        except EngineError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

        if mask is None:
            analysis.clear_preview()
            return {
                "ok": False,
                "message": "No object was found around that point. Try another one.",
                "objects": analysis.to_public()["objects"],
            }
        polygon = polygon_from_mask(mask)

    if not payload.commit:
        analysis.remember_preview(points, mask, polygon)
        return {
            "ok": True,
            "committed": False,
            "preview": {
                "polygon": polygon,
                "area_pixels": int(mask.sum()),
                "points": [
                    {"x": int(round(x)), "y": int(round(y)), "label": label}
                    for x, y, label in points
                ],
            },
        }

    obj = AnalysisObject()
    obj.set(points=polygon, mask=mask)
    for x, y, label in points:
        obj.add_prompt_point(x, y, label)
    prefix = sanitize_prefix(payload.name_prefix) if payload.name_prefix else None
    try:
        analysis.register(obj, prefix=prefix)
    except AnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    analysis.clear_preview()

    return {
        "ok": True,
        "committed": True,
        "object": obj.to_public(),
        "objects": analysis.to_public()["objects"],
    }


@app.post("/api/analysis/semantic")
def analysis_semantic(payload: SemanticDetectRequest) -> dict:
    """Detect prompt-matched boxes with Grounding DINO, then refine with SAM."""
    if not semantic_model_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local semantic detection model is not available.",
        )
    analysis, record = _require_analysis(payload.session_id, payload.image_id)
    prompt = payload.prompt.strip()
    try:
        detector = get_semantic_detector()
        image_bgr = _processed_image(analysis, record)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with _semantic_lock:
            chosen = None
            # A prompt that names no organ -- a crop name, typically -- is resolved by the
            # image before anything else happens: the four-organ probe already runs for the
            # cross-check below, so asking it which organ this looks like costs nothing
            # extra. The same call is made on the desktop, from the same method.
            if not payload.skip_semantic_check:
                prompt, chosen, summary = detector.resolve_organ(
                    image_rgb, prompt, candidates=payload.candidates
                )
                # Same order as the desktop: probe the base categories first and
                # bail out before any object is created when the prompt clearly does
                # not match the image, so a wrong word cannot silently fill the panel
                # with objects.
                mismatch = detector.cross_check(image_rgb, prompt, summary=summary)
                if mismatch:
                    return {
                        "ok": False,
                        "reason": "semantic_mismatch",
                        "prompt": prompt,
                        "chosen": chosen,
                        "mismatch": {
                            "requested": mismatch["requested"],
                            "requested_count": mismatch["requested_count"],
                            "requested_score": round(float(mismatch["requested_score"]), 4),
                            "suggested": mismatch["suggested"],
                            "suggested_count": mismatch["suggested_count"],
                            "suggested_score": round(float(mismatch["suggested_score"]), 4),
                        },
                        "created": [],
                        "skipped": 0,
                        "detections": 0,
                        "objects": analysis.to_public()["objects"],
                    }
            else:
                chosen = None
            detections = detector.detect(image_rgb, prompt)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    prefix_pairs = TextDetector.split_prompt_pairs(prompt)
    candidates = []
    skipped = 0
    engine = get_engine()
    for detection in detections:
        try:
            mask = engine.segment_box(analysis.key, detection["box"])
        except EngineBusy as exc:
            # Semantic detection segments every box it found, so hitting the queue
            # limit halfway through is possible. Reporting busy beats returning a
            # partial set of objects the visitor did not ask for.
            raise _busy(exc) from exc
        except EngineNotReady as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        except (EngineError, EngineUnavailable):
            skipped += 1
            continue
        if mask is None or not np.any(mask):
            skipped += 1
            continue
        traits = mask_traits(mask)
        if traits is None:
            skipped += 1
            continue
        polygon = polygon_from_mask(mask)
        if len(polygon) < 3:
            skipped += 1
            continue
        obj = AnalysisObject()
        obj.set(points=polygon, mask=mask)
        obj.source_prompt = str(detection.get("label") or prompt)
        obj.detection_score = float(detection.get("score", 0.0))
        candidates.append({
            "shape": obj,
            "label": detection.get("label"),
            "traits": traits,
        })

    # Desktop parity: drop the obvious size/shape outliers within each detected
    # label before anything is registered. Groups smaller than four are left
    # alone, so three pods or a single ruler are never "filtered away".
    accepted, filtered = filter_mask_candidates(candidates)
    created = []
    for candidate in accepted:
        obj = candidate["shape"]
        prefix = TextDetector.term_for_label(obj.source_prompt, prefix_pairs)
        try:
            analysis.register(obj, prefix=sanitize_prefix(prefix))
        except AnalysisError:
            skipped += 1
            continue
        created.append(obj.to_public())

    return {
        "ok": True,
        "prompt": prompt,
        "chosen": chosen,
        "detections": len(detections),
        "created": created,
        "skipped": skipped,
        "filtered": len(filtered),
        # What the model was actually asked, when it differs from what was typed.
        # A Chinese prompt is translated before it reaches Grounding DINO, and a
        # surprising result is only explainable if that rewrite is visible.
        "translated": [
            {"typed": raw, "used": used}
            for raw, used in TextDetector.alias_notes(prompt)
        ],
        "objects": analysis.to_public()["objects"],
    }


@app.patch("/api/analysis/objects/{uid}")
def analysis_rename(uid: int, payload: RenameRequest) -> dict:
    analysis, _record = _require_analysis(payload.session_id, payload.image_id)
    try:
        obj = analysis.rename(uid, payload.label)
    except AnalysisError as exc:
        message = str(exc)
        code = (
            status.HTTP_404_NOT_FOUND
            if message == "Unknown object."
            else status.HTTP_409_CONFLICT
        )
        raise HTTPException(status_code=code, detail=message) from exc
    return {"ok": True, "object": obj.to_public()}


@app.delete("/api/analysis/objects/{uid}")
def analysis_delete(
    uid: int,
    session_id: str = Query(...),
    image_id: str = Query(...),
) -> dict:
    analysis, _record = _require_analysis(session_id, image_id)
    try:
        analysis.remove(uid)
    except AnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return {"ok": True, "objects": analysis.to_public()["objects"]}


@app.post("/api/analysis/scale")
def analysis_scale(payload: ScaleRequest) -> dict:
    analysis, _record = _require_analysis(payload.session_id, payload.image_id)
    try:
        scale = analysis.set_scale(payload.per_pixel, payload.unit)
    except AnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return {"ok": True, "scale": scale}


@app.post("/api/analysis/color")
def analysis_color(payload: ImageRef) -> dict:
    analysis, record = _require_analysis(payload.session_id, payload.image_id)
    # Calibration is idempotent on purpose. It used to read back
    # ``analysis.corrected_image``, so a second press fitted a new CCM on already
    # corrected pixels and pushed the colors further each time; the desktop
    # caches one result and only toggles the view.
    if analysis.color_correction and analysis.corrected_image is not None:
        return {
            "ok": True,
            "color_correction": True,
            "recomputed": False,
            "centers": analysis.color_centers,
            "quality": analysis.color_quality,
            "image_url": _corrected_image_url(payload.session_id, payload.image_id),
        }

    try:
        original = _store.decode(payload.image_id, record["data"])
        report = color_correct_with_report(original)
        analysis.set_color_correction(report["corrected"], report["centers"])
        # Surfaced so the operator can see how well the chart actually fitted: a
        # silent 12 dE calibration is worse than none at all.
        analysis.color_quality = {
            "mean_delta_e": round(report["mean_delta_e"], 2),
            "max_delta_e": round(report["max_delta_e"], 2),
            "before_delta_e": round(report["raw_delta_e"], 2),
            "charts_considered": report["charts"],
        }
        _store.note_corrected(analysis)
    except CalibrationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except AnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return {
        "ok": True,
        "color_correction": True,
        "recomputed": True,
        "centers": analysis.color_centers,
        "quality": analysis.color_quality,
        "image_url": _corrected_image_url(payload.session_id, payload.image_id),
    }


@app.get("/api/analysis/corrected-image")
def corrected_image(
    session_id: str = Query(...),
    image_id: str = Query(...),
) -> Response:
    analysis, record = _require_analysis(session_id, image_id)
    if not analysis.color_correction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Color correction has not been run for this image.",
        )
    return Response(
        content=_corrected_preview(analysis, record),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/analysis/ruler")
def analysis_ruler(payload: ImageRef) -> dict:
    analysis, record = _require_analysis(payload.session_id, payload.image_id)
    try:
        image = _store.decode(payload.image_id, record["data"])
        with _ruler_lock:
            tickpoints, per_pixel, uniformity = detect_ruler(image, RULER_MODEL_PATH)
        scale = analysis.set_ruler_detection(tickpoints, per_pixel)
    except RulerDetectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return {
        "ok": True,
        "ruler_detection": True,
        "tickpoints": analysis.ruler_ticks,
        "scale": scale,
        # Diagnostic only: one number cannot express a scale that changes across
        # the frame, so the drift along the rule is reported instead of hidden.
        "uniformity": uniformity,
    }


@app.post("/api/analysis/traits")
def analysis_traits(payload: TraitsRequest) -> dict:
    analysis, record = _require_analysis(payload.session_id, payload.image_id)
    if not analysis.objects:
        return {
            "headers": trait_headers(analysis.scale_unit),
            "rows": [],
            "scale": analysis.scale_public(),
        }
    try:
        image = _processed_image(analysis, record)
        return analysis.traits(image, payload.object_ids)
    except AnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@app.get("/api/analysis/annotation")
def analysis_annotation(
    session_id: str = Query(...),
    image_id: str = Query(...),
    download: bool = Query(default=True),
) -> Response:
    analysis, _record = _require_analysis(session_id, image_id)
    document = analysis.document()
    if not download:
        return JSONResponse(content=document)

    body = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    name = Path(analysis.filename).stem + ".pheno.json"
    disposition = "attachment; filename*=UTF-8''%s" % quote(name)
    return Response(
        content=body.encode("utf-8"),
        media_type="application/json",
        headers={"Content-Disposition": disposition, "Cache-Control": "no-store"},
    )


@app.get("/api/analysis/results")
def analysis_results(
    session_id: str = Query(...),
    image_id: str = Query(...),
) -> Response:
    """Return the trait table as one CSV download.

    Deliberately only the table: this is the phone deliverable, and packing the
    annotated 16 MPx frame plus per-object mask PNGs turned it into an ~12 MB ZIP
    (the annotated PNG alone was 14.7 MB uncompressed). Segmentation itself is
    still downloadable, losslessly and far more compactly, through
    ``/api/analysis/annotation`` (~1.1 KB of RLE per object).
    """
    analysis, record = _require_analysis(session_id, image_id)
    try:
        image = _processed_image(analysis, record)
        traits = analysis.traits(image)
    except AnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    # The BOM keeps Excel from mangling UTF-8 object names.
    body = ("\ufeff" + _csv_text(traits["headers"], traits["rows"])).encode("utf-8")
    name = (Path(analysis.filename).stem or "capture") + "_traits.csv"
    disposition = "attachment; filename*=UTF-8''%s" % quote(name)
    return Response(
        content=body,
        # Deliberately not text/csv: some mobile browsers ignore the attachment
        # disposition for types they can render and show the table inline instead.
        media_type="application/octet-stream",
        headers={"Content-Disposition": disposition, "Cache-Control": "no-store"},
    )
