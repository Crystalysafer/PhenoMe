"""Headless SAM inference for the web service.

This module is the web-side counterpart of ``pheno/SAM.py``. It exists as a
separate implementation on purpose:

* **No Qt.** ``pheno/SAM.py`` imports ``QRect`` and is fed ``QImage`` data by the
  canvas. Here every image is a plain ``numpy`` BGR array, so the service can run
  without a display, a ``QApplication``, or the Qt event loop.
* **Per-session embeddings.** The desktop predictor holds exactly one image
  embedding and skips re-encoding when the MD5 matches. A server has many
  browsers attached at once, so embeddings are cached per session/image key and
  restored into the predictor on demand.
* **One GPU thread.** Every CUDA touch (model load, image encode, mask predict)
  runs on a single worker thread, which both serialises access and keeps the CUDA
  context on one thread.

The model is loaded lazily: importing this module costs nothing, and the 2.4 GB
``sam_vit_h_4b8939.pth`` checkpoint is only read when the first request asks for
it.

Prediction results are intentionally bit-compatible with the desktop app: the
same 1024-pixel long-side downscale, the same ``int()`` truncation of prompt
coordinates, the same "keep the connected domain that contains a positive point"
post-processing, and the same channel order handed to ``SamPredictor``
(``pheno/canvas.py`` passes ``qimage_to_cv2()`` output, i.e. BGR, while the
predictor's default ``image_format`` is ``"RGB"``). Passing a differently ordered
array here would silently make web masks diverge from desktop masks.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

# Segmenter backends, mirroring the registry in ``pheno/SAM.py``. The desktop
# picks a default itself; the service keeps ``sam_vit_h`` unless
# ``PHENOME_SAM_MODEL`` (or the ``backend`` argument) selects another entry and
# its weights/package are actually deployed. SAM2 weights live under
# ``models/sam2/`` and are not bundled with the source.
BACKENDS = {
    "sam_vit_h": {
        "label": "sam-vit-h",
        "kind": "sam1",
        "registry_key": "vit_h",
        "config": None,
        "weights": DEFAULT_CHECKPOINT,
    },
    "sam2_hiera_t": {
        "label": "sam2-hiera-t",
        "kind": "sam2",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "weights": PROJECT_ROOT / "models" / "sam2" / "sam2.1_hiera_tiny.pt",
    },
    "sam2_hiera_l": {
        "label": "sam2-hiera-l",
        "kind": "sam2",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "weights": PROJECT_ROOT / "models" / "sam2" / "sam2.1_hiera_large.pt",
    },
}
# Accept the bare SAM1 registry key for callers that still pass ``"vit_h"``.
BACKEND_ALIASES = {"vit_h": "sam_vit_h"}
DEFAULT_BACKEND = "sam_vit_h"


def resolve_backend(name: str | None) -> str:
    """Map a user/env backend name onto a ``BACKENDS`` key."""
    resolved = BACKEND_ALIASES.get(name or "", name or DEFAULT_BACKEND)
    if resolved not in BACKENDS:
        raise EngineUnavailable(
            "Unknown SAM backend %r; expected one of %s."
            % (name, ", ".join(sorted(BACKENDS)))
        )
    return resolved


def model_label(model_type: str | None) -> str:
    """Human-readable backend name for status payloads."""
    return BACKENDS.get(BACKEND_ALIASES.get(model_type or "", model_type or ""), {}).get(
        "label", model_type or "unknown"
    )

# Long side the image is scaled to before encoding, matching ``pheno/SAM.py``.
EMBEDDING_MAX_SIDE = 1024
# Cached embeddings live in GPU memory (about 4 MB each for vit_h), so the limit is
# not really about memory -- it bounds how many browsers can stay warm at once.
# With the service open to the public, a small number means visitors keep evicting
# each other and paying the 2-16 s encode again, so this is generous on purpose.
MAX_CACHED_EMBEDDINGS = 24
# Below this much free GPU memory the model is loaded on the CPU instead.
MIN_FREE_GPU_GB = 3.0
MAX_PROMPT_POINTS = 64
# How many GPU jobs may be queued before new ones are refused outright. The worker
# is single-threaded by design, so without a bound a burst of visitors would all
# sit in an unbounded queue and every one of them would time out. Refusing fast
# with "try again in a moment" is far better than making everybody wait 300 s.
MAX_QUEUED_JOBS = 12

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_PENDING = "pending"
STATE_READY = "ready"
STATE_ERROR = "error"


class EngineError(RuntimeError):
    """Any recoverable inference failure the API should report as 4xx/5xx."""


class EngineBusy(EngineError):
    """The GPU queue is full; the caller should retry shortly."""


class EngineNotReady(EngineError):
    """The model or the requested embedding is not available yet."""


class EngineUnavailable(EngineError):
    """The engine can never work in this deployment (missing checkpoint, ...)."""


_torch_module = None


def _torch():
    """Import torch on first use so module import stays cheap."""
    global _torch_module
    if _torch_module is None:
        import torch  # noqa: PLC0415 - deliberately deferred

        _torch_module = torch
    return _torch_module


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Normalise any decoded image to a 3-channel BGR array."""
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.shape[2] == 1:
        return cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR)
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    return array


def resize_for_embedding(image: np.ndarray, max_side: int = EMBEDDING_MAX_SIDE):
    """Downscale so the long side is ``max_side``; returns ``(image, scale)``."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0
    scale = float(max_side) / float(longest)
    resized = cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized, scale


def domain_containing_points(mask: np.ndarray, points) -> np.ndarray | None:
    """Keep the connected component that contains one of ``points``.

    Mirrors ``SAM.get_oneDomain``: components are scanned in label order and the
    first one covering any positive prompt point wins, so a click never returns
    stray blobs elsewhere in the image.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labelled = cv2.connectedComponents(binary)
    height, width = binary.shape[:2]

    for label in range(1, count):
        current = (labelled == label).astype(np.uint8)
        for x, y in points:
            row, column = int(y), int(x)
            if 0 <= row < height and 0 <= column < width and current[row, column] == 1:
                return current
    return None


class SamEngine:
    """Lazily loaded SAM with per-key embeddings and a single GPU worker."""

    def __init__(
        self,
        checkpoint=None,
        model_type: str | None = None,
        device: str | None = None,
        max_embeddings: int = MAX_CACHED_EMBEDDINGS,
        backend: str | None = None,
    ) -> None:
        # ``model_type`` is kept for existing callers/tests; ``backend`` and the
        # PHENOME_SAM_MODEL env var select from BACKENDS.
        self._backend_id = resolve_backend(
            backend
            or os.getenv("PHENOME_SAM_MODEL")
            or model_type
        )
        self._backend = BACKENDS[self._backend_id]
        self._checkpoint = Path(
            checkpoint
            or os.getenv("PHENOME_SAM_CHECKPOINT")
            or self._backend["weights"]
        )
        self._model_type = self._backend_id
        self._requested_device = device or os.getenv("PHENOME_WEB_DEVICE") or None
        self._max_embeddings = max(1, int(max_embeddings))

        self._lock = threading.RLock()
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam-gpu")
        # Everything submitted but not finished. ThreadPoolExecutor's own queue is
        # unbounded, so this is what keeps a burst from turning into a pile of
        # timeouts. Guarded by _lock.
        self._inflight = 0
        self._peak_inflight = 0

        self._model_state = STATE_IDLE
        self._model_error: str | None = None
        self._device: str | None = None
        self._load_seconds: float | None = None
        self._sam = None
        self._predictor = None

        self._embeddings: "OrderedDict[str, dict]" = OrderedDict()
        self._active_key: str | None = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def checkpoint(self) -> Path:
        return self._checkpoint

    def checkpoint_available(self) -> bool:
        return self._checkpoint.is_file()

    def gpu_info(self) -> dict:
        """Driver-level GPU facts, or ``{"available": False}`` without CUDA."""
        try:
            torch = _torch()
        except Exception as exc:  # pragma: no cover - torch is installed here
            return {"available": False, "error": str(exc)}

        if not torch.cuda.is_available():
            return {"available": False}

        info: dict = {"available": True}
        try:
            info["name"] = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            info["free_gb"] = round(free_bytes / 1024 ** 3, 2)
            info["total_gb"] = round(total_bytes / 1024 ** 3, 2)
        except Exception as exc:  # pragma: no cover - driver dependent
            info["error"] = str(exc)
        return info

    def status(self) -> dict:
        with self._lock:
            ready = sum(
                1 for entry in self._embeddings.values() if entry["state"] == STATE_READY
            )
            pending = sum(
                1 for entry in self._embeddings.values() if entry["state"] == STATE_PENDING
            )
            state = self._model_state
            error = self._model_error
            device = self._device
            load_seconds = self._load_seconds
            inflight = self._inflight
            peak = self._peak_inflight

        return {
            "state": state,
            "error": error,
            "device": device,
            "model_type": self._model_type,
            "checkpoint": self._checkpoint.name,
            "checkpoint_available": self.checkpoint_available(),
            "load_seconds": load_seconds,
            "embeddings": {
                "ready": ready,
                "pending": pending,
                "limit": self._max_embeddings,
            },
            # Surfaced so the page can say "3 ahead of you" instead of just
            # spinning, and so operators can see whether the box is saturated.
            "queue": {
                "depth": inflight,
                "limit": MAX_QUEUED_JOBS,
                "peak": peak,
            },
            "gpu": self.gpu_info(),
        }

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _submit(self, function, *args, admission: bool = True):
        """Queue one GPU job, refusing new work when the queue is already full.

        ``admission=False`` is for the no-op probes used to wait for the worker:
        they carry no cost and must not be rejected, or ``load()`` could fail
        merely because visitors are busy.
        """
        with self._lock:
            if admission and self._inflight >= MAX_QUEUED_JOBS:
                raise EngineBusy(
                    "The analysis queue is full (%d waiting). Please try again in a moment."
                    % self._inflight
                )
            self._inflight += 1
            self._peak_inflight = max(self._peak_inflight, self._inflight)

        def done(_future) -> None:
            with self._lock:
                self._inflight = max(0, self._inflight - 1)

        try:
            future = self._gpu.submit(function, *args)
        except Exception:
            done(None)
            raise
        future.add_done_callback(done)
        return future

    def queue_depth(self) -> int:
        """How many GPU jobs are queued or running right now."""
        with self._lock:
            return self._inflight

    def request_load(self) -> dict:
        """Start loading the checkpoint in the background if needed."""
        if not self.checkpoint_available():
            with self._lock:
                self._model_state = STATE_ERROR
                self._model_error = "Checkpoint %s was not found." % self._checkpoint.name
            return self.status()

        with self._lock:
            if self._model_state in (STATE_LOADING, STATE_READY):
                return self.status()
            self._model_state = STATE_LOADING
            self._model_error = None

        self._submit(self._load_model_if_needed)
        return self.status()

    def load(self, timeout: float | None = None) -> dict:
        """Block until the model is loaded (used by scripts and tests)."""
        self.request_load()
        self._submit(lambda: None, admission=False).result(timeout=timeout)
        status = self.status()
        if status["state"] != STATE_READY:
            raise EngineError(status.get("error") or "The SAM model failed to load.")
        return status

    def _load_model_if_needed(self) -> None:
        """Runs on the GPU worker thread only."""
        with self._lock:
            if self._model_state == STATE_READY and self._predictor is not None:
                return
            if not self.checkpoint_available():
                self._model_state = STATE_ERROR
                self._model_error = "Checkpoint %s was not found." % self._checkpoint.name
                raise EngineUnavailable(self._model_error)
            self._model_state = STATE_LOADING
            self._model_error = None

        started = time.time()
        try:
            device = self._resolve_device()
            if self._backend["kind"] == "sam2":
                from sam2.build_sam import build_sam2  # noqa: PLC0415
                from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415

                sam = build_sam2(
                    self._backend["config"],
                    str(self._checkpoint),
                    device=device,
                )
                predictor = SAM2ImagePredictor(sam)
            else:
                from segment_anything import SamPredictor, sam_model_registry  # noqa: PLC0415

                sam = sam_model_registry[self._backend["registry_key"]](
                    checkpoint=str(self._checkpoint)
                )
                sam.to(device=device)
                predictor = SamPredictor(sam)
        except Exception as exc:
            with self._lock:
                self._model_state = STATE_ERROR
                self._model_error = str(exc)
            raise

        with self._lock:
            self._sam = sam
            self._predictor = predictor
            self._device = device
            self._model_state = STATE_READY
            self._model_error = None
            self._load_seconds = round(time.time() - started, 2)

    def _resolve_device(self) -> str:
        if self._requested_device:
            return self._requested_device
        try:
            torch = _torch()
            if not torch.cuda.is_available():
                return "cpu"
            try:
                from pheno.cuda import cuda_supported  # noqa: PLC0415

                ok, why = cuda_supported()
            except Exception:
                ok, why = True, ""
            if not ok:
                print(f"[WARN] CUDA unusable ({why}); using CPU")
                return "cpu"
            free_bytes, _total = torch.cuda.mem_get_info(0)
            if free_bytes / 1024 ** 3 < MIN_FREE_GPU_GB:
                return "cpu"
        except Exception:
            return "cpu"
        return "cuda:0"

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embedding_status(self, key: str) -> dict:
        with self._lock:
            entry = self._embeddings.get(key)
            if entry is None:
                return {"state": STATE_IDLE, "error": None}
            return {
                "state": entry["state"],
                "error": entry.get("error"),
                "seconds": entry.get("seconds"),
                "scale": entry.get("scale"),
            }

    def prepare(self, key: str, loader: Callable[[], np.ndarray]) -> dict:
        """Queue embedding computation for ``key``; returns its current state.

        ``loader`` is called on the GPU worker so decoding a 16 MPx JPEG never
        blocks the HTTP thread, and is skipped entirely when the embedding is
        already cached.
        """
        with self._lock:
            entry = self._embeddings.get(key)
            if entry is not None and entry["state"] in (STATE_PENDING, STATE_READY):
                self._embeddings.move_to_end(key)
                return self.embedding_status(key)
            self._embeddings[key] = {"state": STATE_PENDING, "error": None}
            self._embeddings.move_to_end(key)

        self._submit(self._do_prepare, key, loader)
        return self.embedding_status(key)

    def prepare_blocking(self, key: str, loader: Callable[[], np.ndarray], timeout=None) -> dict:
        self.prepare(key, loader)
        self._submit(lambda: None, admission=False).result(timeout=timeout)
        status = self.embedding_status(key)
        if status["state"] != STATE_READY:
            raise EngineError(status.get("error") or "Embedding preparation failed.")
        return status

    def _do_prepare(self, key: str, loader: Callable[[], np.ndarray]) -> None:
        started = time.time()
        try:
            self._load_model_if_needed()
            image = to_bgr(loader())
            raw_shape = (int(image.shape[0]), int(image.shape[1]))
            scaled, scale = resize_for_embedding(image)

            predictor = self._predictor
            # Desktop-compatible channel order: see the module docstring.
            predictor.set_image(scaled)

            payload = {
                "state": STATE_READY,
                "error": None,
                "predictor_state": self._capture_predictor_state(),
                "scale": scale,
                "raw_shape": raw_shape,
                "seconds": round(time.time() - started, 2),
            }
        except Exception as exc:
            with self._lock:
                entry = self._embeddings.get(key)
                if entry is not None:
                    entry.update(
                        state=STATE_ERROR,
                        error=str(exc),
                        seconds=round(time.time() - started, 2),
                    )
            return

        with self._lock:
            if key not in self._embeddings:
                # Dropped while encoding: release the features instead of caching.
                return
            self._embeddings[key] = payload
            self._embeddings.move_to_end(key)
            self._active_key = key
            self._evict_locked()

    def _evict_locked(self) -> None:
        while len(self._embeddings) > self._max_embeddings:
            victim = None
            for candidate, entry in self._embeddings.items():
                if entry["state"] != STATE_PENDING:
                    victim = candidate
                    break
            if victim is None:
                return
            self._embeddings.pop(victim, None)
            if self._active_key == victim:
                self._active_key = None

    def drop(self, key: str) -> None:
        with self._lock:
            self._embeddings.pop(key, None)
            if self._active_key == key:
                self._active_key = None

    def drop_prefix(self, prefix: str) -> None:
        with self._lock:
            for key in [k for k in self._embeddings if k.startswith(prefix)]:
                self._embeddings.pop(key, None)
                if self._active_key == key:
                    self._active_key = None

    def embedding_scale(self, key: str) -> float | None:
        with self._lock:
            entry = self._embeddings.get(key)
            if entry is None or entry["state"] != STATE_READY:
                return None
            return entry.get("scale")

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment(self, key: str, points, one_domain: bool = True, timeout=None):
        """Predict one full-resolution mask from positive/negative points.

        ``points`` is a sequence of ``(x, y, label)`` in original image pixels.
        Returns ``None`` when SAM produced nothing usable around the positive
        points, which the API surfaces as "pick another point".
        """
        if not points:
            raise EngineError("At least one prompt point is required.")
        if len(points) > MAX_PROMPT_POINTS:
            raise EngineError("At most %d prompt points are supported." % MAX_PROMPT_POINTS)
        if not any(int(label) == 1 for _x, _y, label in points):
            raise EngineError("At least one positive point is required.")

        future = self._submit(self._do_segment, key, list(points), one_domain)
        return future.result(timeout=timeout)

    def segment_box(self, key: str, box, timeout=None):
        """Segment one object from an original-image XYXY box prompt."""
        if len(box) != 4:
            raise EngineError("A semantic detection box must contain four coordinates.")
        future = self._submit(self._do_segment_box, key, tuple(float(value) for value in box))
        return future.result(timeout=timeout)

    def _capture_predictor_state(self) -> dict:
        """Snapshot the predictor's per-image internals for later restore.

        SAM1 keeps flat attributes; SAM2 keeps a feature dict plus the original
        image sizes. Both are GPU tensors -- entries count toward the
        MAX_CACHED_EMBEDDINGS budget either way.
        """
        predictor = self._predictor
        if self._backend["kind"] == "sam2":
            return {
                "features": predictor._features,
                "orig_hw": list(predictor._orig_hw),
            }
        return {
            "features": predictor.features,
            "original_size": predictor.original_size,
            "input_size": predictor.input_size,
        }

    def _restore_locked(self, entry: dict) -> None:
        predictor = self._predictor
        # Entries built before the predictor_state wrapper (and test fixtures)
        # still carry the SAM1 fields at top level.
        state = entry.get("predictor_state", entry)
        if self._backend["kind"] == "sam2":
            predictor.reset_predictor()
            predictor._features = state["features"]
            predictor._orig_hw = list(state["orig_hw"])
            predictor._is_image_set = True
            predictor._is_batch = False
            return
        predictor.reset_image()
        predictor.features = state["features"]
        predictor.original_size = state["original_size"]
        predictor.input_size = state["input_size"]
        predictor.is_image_set = True

    def _do_segment(self, key: str, points, one_domain: bool):
        with self._lock:
            entry = self._embeddings.get(key)
            if entry is None or entry["state"] != STATE_READY:
                raise EngineNotReady("The image embedding is not ready yet.")
            if self._predictor is None:
                raise EngineNotReady("The SAM model is not loaded yet.")
            if self._active_key != key:
                self._restore_locked(entry)
                self._active_key = key
            scale = float(entry["scale"])
            raw_height, raw_width = entry["raw_shape"]
            predictor = self._predictor

        # Truncate exactly like SAM.add_point so a click lands on the same pixel
        # as it would on the desktop.
        scaled_points = [(int(x * scale), int(y * scale)) for x, y, _label in points]
        labels = [int(label) for _x, _y, label in points]

        try:
            masks, _scores, _logits = predictor.predict(
                point_coords=np.array(scaled_points),
                point_labels=np.array(labels),
                multimask_output=False,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self._emergency_cleanup()
                raise EngineError(
                    "The GPU ran out of memory during prediction; try a smaller image."
                ) from exc
            raise

        if len(masks) == 0:
            return None

        mask = masks[0]
        if one_domain:
            positives = [
                point for point, label in zip(scaled_points, labels) if label == 1
            ]
            mask = domain_containing_points(mask, positives)
            if mask is None:
                return None

        return cv2.resize(
            np.asarray(mask).astype(np.uint8),
            (raw_width, raw_height),
            interpolation=cv2.INTER_NEAREST,
        )

    def _do_segment_box(self, key: str, box):
        with self._lock:
            entry = self._embeddings.get(key)
            if entry is None or entry["state"] != STATE_READY:
                raise EngineNotReady("The image embedding is not ready yet.")
            if self._predictor is None:
                raise EngineNotReady("The SAM model is not loaded yet.")
            if self._active_key != key:
                self._restore_locked(entry)
                self._active_key = key
            scale = float(entry["scale"])
            raw_height, raw_width = entry["raw_shape"]
            predictor = self._predictor

        scaled_box = np.asarray(box, dtype=np.float32) * scale
        try:
            masks, _scores, _logits = predictor.predict(
                box=scaled_box,
                multimask_output=False,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self._emergency_cleanup()
                raise EngineError(
                    "The GPU ran out of memory during semantic detection; try a smaller image."
                ) from exc
            raise
        if len(masks) == 0:
            return None
        return cv2.resize(
            np.asarray(masks[0]).astype(np.uint8),
            (raw_width, raw_height),
            interpolation=cv2.INTER_NEAREST,
        )

    def _emergency_cleanup(self) -> None:
        with self._lock:
            self._embeddings.clear()
            self._active_key = None
        try:
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def shutdown(self) -> None:
        self._gpu.shutdown(wait=False)
