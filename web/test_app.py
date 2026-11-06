import json
import logging
import os
import re
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from pheno import annotation
from pheno.calcPheno import PhenomicsCalculator
from web import app as web_app
from web import translate as web_translate
from web.analysis import (
    MAX_CORRECTED_IMAGES,
    AnalysisObject,
    AnalysisStore,
    compute_traits,
    polygon_from_mask,
    read_exif_orientation,
    trait_headers,
)
from web.app import MAX_UPLOAD_BYTES, STATIC_DIR, app
from web.calibration import MCC24_REF_SRGB, CalibrationError, color_correct_bgr
from web.engine import (
    EngineError,
    EngineNotReady,
    SamEngine,
    domain_containing_points,
    resize_for_embedding,
    to_bgr,
)
from web.ruler import RulerDetectionError, detect_ruler


def column_index(headers, bare_name):
    """Index of a trait column, located by its name without the unit suffix.

    Headers carry the current unit ("Area (mm²)"), so tests must not hard-code the
    decorated string or they break every time the wording changes.
    """
    for index, header in enumerate(headers):
        if header == bare_name or header.startswith(bare_name + " ("):
            return index
    raise AssertionError(f"{bare_name!r} not among {headers}")


def _color_report(corrected, centers, mean_delta_e=4.0):
    """What `color_correct_with_report` hands back, for route tests."""
    return {
        "corrected": corrected,
        "centers": centers,
        "matrix": np.eye(4, 3, dtype=np.float32),
        "mean_delta_e": mean_delta_e,
        "max_delta_e": mean_delta_e * 2,
        "raw_delta_e": mean_delta_e * 3,
        "charts": 1,
    }


def make_png(size=(12, 8), color=(45, 132, 96)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def make_sample_png(size=(200, 150)) -> bytes:
    """A dark frame with one bright rectangle, so masks have real contours."""
    array = np.full((size[1], size[0], 3), 30, dtype=np.uint8)
    array[40:110, 60:140] = (210, 190, 120)
    output = BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


class FakeEngine:
    """Deterministic stand-in for :class:`web.engine.SamEngine`.

    It keeps the same contract the routes rely on -- lazy load state, per-key
    embeddings, single mask per click -- without touching torch or the GPU.
    """

    HALF_WIDTH = 25
    HALF_HEIGHT = 18

    def __init__(self):
        self.state = "idle"
        self.embeddings = {}
        self.prepared_keys = []
        self.segment_calls = []
        self.box_calls = []
        self.dropped = []

    # -- model ---------------------------------------------------------
    def status(self):
        return {
            "state": self.state,
            "error": None,
            "device": "cpu",
            "model_type": "vit_h",
            "checkpoint": "sam_vit_h_4b8939.pth",
            "checkpoint_available": True,
            "load_seconds": 0.0 if self.state == "ready" else None,
            "embeddings": {
                "ready": len(self.embeddings),
                "pending": 0,
                "limit": 6,
            },
            "gpu": {"available": False},
        }

    def request_load(self):
        self.state = "ready"
        return self.status()

    # -- embeddings ----------------------------------------------------
    def embedding_status(self, key):
        if key not in self.embeddings:
            return {"state": "idle", "error": None}
        return {"state": "ready", "error": None, "seconds": 0.0, "scale": 1.0}

    def prepare(self, key, loader):
        if key in self.embeddings:
            return self.embedding_status(key)  # cached, like the real engine
        image = loader()
        assert image.ndim == 3 and image.shape[2] == 3
        self.state = "ready"
        self.embeddings[key] = (image.shape[1], image.shape[0])
        self.prepared_keys.append(key)
        return self.embedding_status(key)

    def drop(self, key):
        self.dropped.append(key)
        self.embeddings.pop(key, None)

    def drop_prefix(self, prefix):
        for key in [k for k in self.embeddings if k.startswith(prefix)]:
            self.drop(key)

    # -- segmentation --------------------------------------------------
    def segment(self, key, points, one_domain=True):
        if key not in self.embeddings:
            raise EngineNotReady("The image embedding is not ready yet.")
        self.segment_calls.append((key, tuple(points)))
        width, height = self.embeddings[key]
        positives = [(x, y) for x, y, label in points if label == 1]
        if any(x <= 1 and y <= 1 for x, y in positives):
            return None  # emulates "no valid domain around this point"

        shrink = sum(1 for _x, _y, label in points if label == 0)
        half_w = max(4, self.HALF_WIDTH - 6 * shrink)
        half_h = max(4, self.HALF_HEIGHT - 6 * shrink)
        mask = np.zeros((height, width), dtype=np.uint8)
        for x, y in positives:
            x0 = max(0, int(x) - half_w)
            x1 = min(width, int(x) + half_w)
            y0 = max(0, int(y) - half_h)
            y1 = min(height, int(y) + half_h)
            mask[y0:y1, x0:x1] = 1
        return mask

    def segment_box(self, key, box):
        if key not in self.embeddings:
            raise EngineNotReady("The image embedding is not ready yet.")
        self.box_calls.append((key, tuple(box)))
        width, height = self.embeddings[key]
        x1, y1, x2, y2 = (int(value) for value in box)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 1
        return mask


class WebAppTestCase(unittest.TestCase):
    """Existing surface: static page, health, capabilities, image sessions."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        web_app.set_engine(FakeEngine())

    def test_index_page(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("PhenoMe Web", response.text)
        self.assertIn("canvas-overlay", response.text)
        self.assertIn('/assets/icon.png', response.text)
        self.assertIn('id="semantic-detect"', response.text)
        self.assertNotIn('data-view-target=', response.text)

    def test_single_image_input_without_capture_hints(self):
        """One picker only: the phone's own sheet offers camera vs library."""
        page = self.client.get("/").text

        self.assertEqual(page.count('type="file"'), 1)
        self.assertIn('id="workspace-file"', page)
        self.assertIn('accept="image/*"', page)
        self.assertNotIn('id="camera-input"', page)
        self.assertNotIn('id="gallery-input"', page)
        self.assertNotIn("capture=", page)

    def test_toolbar_is_tiered_and_translatable(self):
        page = self.client.get("/").text

        # Three tiers: lead action, analysis row, collapsible calibration group.
        self.assertIn("command-lead", page)
        self.assertIn('class="tool-group"', page)
        self.assertIn("tool-group-collapsible", page)
        self.assertIn('id="calibrate-group"', page)
        self.assertIn("<summary", page)

        # Language switch plus translation hooks on every static label.
        self.assertIn('id="language-toggle"', page)
        for key in (
            "toolbar.open",
            "pick.start",
            "group.analyze",
            "group.calibrate",
            "group.sample",
            "actions.download",
            "panel.objects",
        ):
            self.assertIn('data-i18n="%s"' % key, page)

    def test_click_flow_is_one_button_and_samples_are_offered(self):
        page = self.client.get("/").text

        # The old pair of buttons is gone: one button starts picking and saves.
        self.assertIn('id="pick-object"', page)
        self.assertNotIn('id="click-mode"', page)
        self.assertNotIn('id="commit-object"', page)
        # Semantic detection now leads the analysis row.
        self.assertLess(page.index('id="semantic-detect"'), page.index('id="pick-object"'))
        self.assertLess(page.index('id="pick-object"'), page.index('id="reset-points"'))
        # Tapping the empty canvas opens the same picker.
        self.assertIn('<label class="stage-empty" for="workspace-file">', page)
        # Two bundled samples with thumbnails.
        self.assertIn('id="sample-pods"', page)
        self.assertIn('id="sample-seeds"', page)
        self.assertIn("/static/samples/pods_thumb.jpg", page)

    def test_sample_buttons_are_wired_by_name(self):
        """Regression: a sample button only showed up if its id was hard-coded.

        The new "mixed pod varieties" button was added to the markup and did
        nothing, because the click handlers were bound one id at a time. Binding
        by data-sample makes adding a sample a markup-only change -- and the
        files it needs must actually be shipped.
        """
        page = self.client.get("/").text
        script = self.client.get("/static/app.js").text

        self.assertIn('querySelectorAll("[data-sample]")', script)
        for name in re.findall(r'data-sample="([^"]+)"', page):
            self.assertNotIn('loadSample("%s")' % name, script,
                             "%s is bound by id again" % name)
            response = self.client.get(f"/static/samples/{name}.jpg")
            self.assertEqual(response.status_code, 200, name)

    def test_progress_strip_exists_and_is_translatable(self):
        page = self.client.get("/").text

        self.assertIn('id="progress"', page)
        self.assertIn('id="progress-fill"', page)
        self.assertIn('id="progress-label"', page)
        self.assertIn('id="progress-percent"', page)

        script = self.client.get("/static/app.js").text
        # Real byte progress for the upload, time-estimated for the GPU steps.
        self.assertIn("function uploadWithProgress", script)
        self.assertIn("xhr.upload.addEventListener", script)
        for key in (
            "progress.model",
            "progress.upload",
            "progress.embedding",
            "progress.semantic",
            "progress.traits",
        ):
            self.assertIn('startProgress("%s"' % key, script)

    def test_upload_sends_the_blob_with_an_explicit_filename(self):
        """Regression: a Blob wrapped in File() serialised as 0 bytes on the phone."""
        script = self.client.get("/static/app.js").text

        # assertTrue/assertFalse rather than assertIn: a failure here should not
        # dump the whole bundle into the test log.
        self.assertTrue(
            'formData.append("file", blob, state.selectedName' in script,
            "the multipart part must be built from the blob plus an explicit name",
        )
        self.assertFalse(
            "new File([" in script,
            "do not wrap the blob in a File: some browsers then send zero bytes",
        )
        # Three layers of defence for the zero-byte upload seen on the phone.
        self.assertTrue("function reencodeToJpeg" in script, "missing the canvas fallback")
        self.assertTrue("function fetchBlobViaXhr" in script, "missing the XHR re-fetch")
        self.assertFalse('cache: "force-cache"' in script, "force-cache was the suspect")

    def test_bundled_sample_bytes_are_accepted_by_the_upload_endpoint(self):
        """The exact files the sample buttons send must survive validation."""
        for name, expected in (("pods", (2048, 1363)), ("seeds", (2048, 1365)),
                               ("leaves", (1024, 682)), ("flowers", (1024, 682))):
            data = (STATIC_DIR / "samples" / f"{name}.jpg").read_bytes()
            response = self.client.post(
                "/api/images",
                files={"file": (f"sample_{name}.jpg", data, "image/jpeg")},
            )

            self.assertEqual(response.status_code, 201, name)
            payload = response.json()
            self.assertEqual((payload["width"], payload["height"]), expected, name)
            self.assertEqual(payload["format"], "jpeg", name)

    def test_sideways_phone_photo_is_analysed_upright(self):
        """The size announced to the client must describe the frame cv2 decodes.

        Phones record the sensor orientation in an EXIF tag instead of rotating
        the pixels. OpenCV honours that tag when decoding, exactly as a browser
        does when displaying, but Pillow's Image.size reports the stored
        dimensions and ignores it. The upload endpoint used to announce a
        sideways frame while the analyser worked on an upright one, so every mask
        landed rotated and stretched against the organ the user had clicked.
        """
        picture = Image.new("RGB", (40, 20), (200, 30, 30))
        exif = picture.getexif()
        exif[0x0112] = 6  # rotate 90 degrees clockwise to display upright
        buffer = BytesIO()
        picture.save(buffer, format="JPEG", exif=exif)
        data = buffer.getvalue()

        response = self.client.post(
            "/api/images", files={"file": ("sideways.jpg", data, "image/jpeg")}
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(read_exif_orientation(data), 6)
        payload = response.json()

        # cv2 is the analyser's decoder, so it defines the frame everything else
        # has to agree with.
        stored = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width = stored.shape[:2]
        self.assertEqual((width, height), (20, 40), "cv2 applies the EXIF tag itself")

        self.assertEqual((payload["width"], payload["height"]), (width, height))
        decoded = AnalysisStore().decode("exif-orientation", data)
        self.assertEqual(decoded.shape[:2], stored.shape[:2])

    def test_sample_images_are_served_and_small(self):
        for name, limit in (("pods", 800_000), ("seeds", 800_000)):
            full = self.client.get(f"/static/samples/{name}.jpg")
            thumb = self.client.get(f"/static/samples/{name}_thumb.jpg")
            self.assertEqual(full.status_code, 200, name)
            self.assertEqual(thumb.status_code, 200, name)
            self.assertEqual(full.headers["content-type"], "image/jpeg")
            # Downscaled on purpose: the public link is a ~0.26 MB/s tunnel.
            self.assertLess(len(full.content), limit, name)
            self.assertLess(len(thumb.content), 60_000, name)

    def test_front_end_defines_both_languages(self):
        script = self.client.get("/static/app.js")

        self.assertEqual(script.status_code, 200)
        text = script.text
        self.assertIn("const STRINGS", text)
        self.assertIn("zh: {", text)
        self.assertIn("en: {", text)
        # Every key used by the Chinese table must exist in the English one.
        chinese = text.split("zh: {", 1)[1].split("\n  },", 1)[0]
        english = text.split("en: {", 1)[1].split("\n  },", 1)[0]
        keys_zh = set(re.findall(r'"([\w.]+)":', chinese))
        keys_en = set(re.findall(r'"([\w.]+)":', english))
        self.assertTrue(keys_zh)
        self.assertEqual(keys_zh - keys_en, set())
        self.assertEqual(keys_en - keys_zh, set())

    def test_existing_desktop_logo_is_available_to_the_web_page(self):
        response = self.client.get("/assets/icon.png")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")

    def test_health_response(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["model_backend"], "sam-vit-h")
        self.assertIn(payload["model_state"], {"idle", "loading", "ready", "error"})

    def test_capability_contract(self):
        response = self.client.get("/api/capabilities")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["desktop"]["object_segmentation"], "available")
        self.assertEqual(payload["desktop"]["phenomics"], "available")
        self.assertEqual(payload["desktop"]["batch"], "excluded")
        self.assertEqual(payload["desktop"]["region_detection"], "excluded")
        self.assertEqual(payload["desktop"]["semantic_detection"], "available")
        self.assertEqual(payload["desktop"]["color_calibration"], "available")
        self.assertEqual(payload["capture"]["semantic_detection"], "available")
        self.assertEqual(payload["capture"]["color_calibration"], "available")
        self.assertEqual(payload["capture"]["batch"], "excluded")
        self.assertEqual(payload["desktop"]["export"], "traits_table")
        self.assertEqual(payload["capture"]["export"], "traits_table")
        self.assertEqual(payload["limits"]["max_upload_bytes"], MAX_UPLOAD_BYTES)

    def test_session_isolates_uploaded_images(self):
        session = self.client.post(
            "/api/sessions",
            json={"source": "capture", "label": "Bean 01", "notes": "Phone sample"},
        )
        self.assertEqual(session.status_code, 201)
        session_payload = session.json()
        self.assertEqual(session_payload["source"], "capture")
        self.assertEqual(session_payload["images"], [])

        upload = self.client.post(
            f"/api/images?session_id={session_payload['id']}",
            files={"file": ("bean.png", make_png(), "image/png")},
        )
        self.assertEqual(upload.status_code, 201)
        self.assertEqual(upload.json()["session_id"], session_payload["id"])

        fetched = self.client.get(f"/api/sessions/{session_payload['id']}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(len(fetched.json()["images"]), 1)

        missing = self.client.post(
            "/api/images?session_id=missing-session",
            files={"file": ("bean.png", make_png(), "image/png")},
        )
        self.assertEqual(missing.status_code, 404)

    def test_upload_read_and_delete_image(self):
        image_data = make_png()
        upload = self.client.post(
            "/api/images",
            files={"file": ("bean.png", image_data, "image/png")},
        )

        self.assertEqual(upload.status_code, 201)
        metadata = upload.json()
        self.assertEqual(metadata["filename"], "bean.png")
        self.assertEqual(metadata["width"], 12)
        self.assertEqual(metadata["height"], 8)

        image = self.client.get(metadata["url"])
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.content, image_data)
        self.assertEqual(image.headers["cache-control"], "no-store")

        deleted = self.client.delete(metadata["url"])
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get(metadata["url"]).status_code, 404)

    def test_rejects_invalid_image_content(self):
        response = self.client.post(
            "/api/images",
            files={"file": ("fake.png", b"not an image", "image/png")},
        )

        self.assertEqual(response.status_code, 400)

    def test_rejects_unsupported_content_type(self):
        response = self.client.post(
            "/api/images",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

        self.assertEqual(response.status_code, 415)

    def test_a_flood_of_new_sessions_cannot_evict_an_active_one(self):
        """Public URL safety: session pressure must not kill live users."""
        active = self.client.post("/api/sessions", json={"source": "capture"}).json()["id"]
        upload = self.client.post(
            f"/api/images?session_id={active}",
            files={"file": ("bean.png", make_png(), "image/png")},
        )
        self.assertEqual(upload.status_code, 201)

        for _ in range(web_app.MAX_SESSIONS + 5):
            self.assertEqual(
                self.client.post("/api/sessions", json={"source": "capture"}).status_code,
                201,
            )
            # Keep working in the original session between the flood requests.
            self.assertEqual(self.client.get(f"/api/sessions/{active}").status_code, 200)

        still_there = self.client.get(f"/api/sessions/{active}")
        self.assertEqual(still_there.status_code, 200)
        self.assertEqual(len(still_there.json()["images"]), 1)


class SegmentationApiTestCase(unittest.TestCase):
    """Click segmentation, object editing, traits and annotation export."""

    def setUp(self):
        self.engine = FakeEngine()
        web_app.set_engine(self.engine)
        self.client = TestClient(app)
        self.session_id = self._new_session()
        self.image = self._upload(self.session_id)
        self.image_id = self.image["id"]

    def _new_session(self, source="workspace"):
        response = self.client.post("/api/sessions", json={"source": source})
        self.assertEqual(response.status_code, 201)
        return response.json()["id"]

    def _upload(self, session_id, name="photo.png"):
        response = self.client.post(
            f"/api/images?session_id={session_id}",
            files={"file": (name, make_sample_png(), "image/png")},
        )
        self.assertEqual(response.status_code, 201)
        return response.json()

    def _ref(self):
        return {"session_id": self.session_id, "image_id": self.image_id}

    def _prepare(self):
        response = self.client.post("/api/analysis/embedding", json=self._ref())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["embedding"]["state"], "ready")
        return response.json()

    def _click(self, points, commit=False, **extra):
        body = dict(self._ref())
        body["points"] = points
        body["commit"] = commit
        body.update(extra)
        return self.client.post("/api/analysis/click", json=body)

    def _commit_object(self, x, y, **extra):
        response = self._click([{"x": x, "y": y, "label": 1}], commit=True, **extra)
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["ok"])
        return payload["object"]

    # -- model ---------------------------------------------------------

    def test_model_status_and_load(self):
        status_response = self.client.get("/api/model")
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["state"], "idle")

        load_response = self.client.post("/api/model/load")
        self.assertEqual(load_response.status_code, 200)
        self.assertEqual(load_response.json()["state"], "ready")

    def test_real_engine_reports_checkpoint_without_loading(self):
        engine = SamEngine()
        status = engine.status()

        self.assertEqual(status["state"], "idle")
        self.assertEqual(status["checkpoint"], "sam_vit_h_4b8939.pth")
        self.assertIsInstance(status["checkpoint_available"], bool)
        self.assertEqual(status["embeddings"]["ready"], 0)

    def test_real_engine_reuses_a_cached_embedding(self):
        """A cached embedding short-circuits before the image is even decoded."""
        engine = SamEngine()
        engine._embeddings["s:i"] = {
            "state": "ready",
            "error": None,
            "scale": 0.25,
            "seconds": 1.0,
        }
        calls = []

        def loader():
            calls.append(1)
            raise AssertionError("the loader must not run for a cached embedding")

        result = engine.prepare("s:i", loader)
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["scale"], 0.25)
        self.assertEqual(calls, [])
        self.assertEqual(engine.embedding_scale("s:i"), 0.25)

        engine.drop("s:i")
        self.assertEqual(engine.embedding_status("s:i")["state"], "idle")
        engine.shutdown()

    # -- embedding -----------------------------------------------------

    def test_embedding_is_prepared_per_session(self):
        self._prepare()
        expected_key = f"{self.session_id}:{self.image_id}"
        self.assertEqual(self.engine.prepared_keys, [expected_key])

        other_session = self._new_session()
        other_image = self._upload(other_session)
        response = self.client.post(
            "/api/analysis/embedding",
            json={"session_id": other_session, "image_id": other_image["id"]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.engine.prepared_keys), 2)
        self.assertEqual(
            self.engine.prepared_keys[1], f"{other_session}:{other_image['id']}"
        )

        # A second request for a cached embedding does not recompute it.
        self._prepare()
        self.assertEqual(len(self.engine.prepared_keys), 2)

    def test_analysis_rejects_foreign_session(self):
        stranger = self._new_session()
        response = self.client.get(
            "/api/analysis",
            params={"session_id": stranger, "image_id": self.image_id},
        )
        self.assertEqual(response.status_code, 404)

    def test_click_requires_embedding(self):
        response = self._click([{"x": 100, "y": 75, "label": 1}])
        self.assertEqual(response.status_code, 409)

    # -- clicking ------------------------------------------------------

    def test_click_preview_then_commit_reuses_the_mask(self):
        self._prepare()
        preview = self._click([{"x": 100, "y": 75, "label": 1}])
        self.assertEqual(preview.status_code, 200)
        body = preview.json()
        self.assertTrue(body["ok"])
        self.assertFalse(body["committed"])
        self.assertGreater(len(body["preview"]["polygon"]), 3)
        self.assertGreater(body["preview"]["area_pixels"], 0)
        self.assertEqual(len(self.engine.segment_calls), 1)

        commit = self._click([{"x": 100, "y": 75, "label": 1}], commit=True)
        self.assertEqual(commit.status_code, 200)
        committed = commit.json()["object"]
        self.assertEqual(committed["label"], "Object01")
        self.assertEqual(committed["polygon"], body["preview"]["polygon"])
        # The preview mask was reused instead of running SAM twice.
        self.assertEqual(len(self.engine.segment_calls), 1)

    def test_background_point_is_rejected(self):
        self._prepare()
        response = self._click(
            [{"x": 100, "y": 75, "label": 1}, {"x": 130, "y": 95, "label": 0}]
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.engine.segment_calls, [])

    @patch("web.app.get_semantic_detector")
    def test_semantic_detection_refines_boxes_with_sam(self, mock_detector):
        class Detector:
            def cross_check(self, _image, _prompt, summary=None):
                return None

            def resolve_organ(self, _image, _prompt, **_kwargs):
                return _prompt, None, None

            def detect(self, _image, _prompt):
                return [{"box": [60, 40, 140, 110], "score": 0.91, "label": "bean pod"}]

        mock_detector.return_value = Detector()
        self._prepare()
        response = self.client.post(
            "/api/analysis/semantic", json=dict(self._ref(), prompt="pod")
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["detections"], 1)
        self.assertEqual(len(payload["created"]), 1)
        self.assertEqual(payload["created"][0]["label"], "pod01")
        self.assertEqual(payload["created"][0]["source_prompt"], "bean pod")
        self.assertEqual(len(self.engine.box_calls), 1)

    def test_click_without_positive_point_is_rejected(self):
        self._prepare()
        response = self._click([{"x": 100, "y": 75, "label": 0}])
        self.assertEqual(response.status_code, 422)

    @patch("web.app.detect_ruler")
    @patch("web.app.color_correct_with_report")
    def test_calibration_and_ruler_routes_return_overlay_geometry(
        self, mock_color_correct, mock_detect_ruler
    ):
        image = np.full((self.image["height"], self.image["width"], 3), 90, dtype=np.uint8)
        mock_color_correct.return_value = _color_report(image, [[12, 16], [24, 32]])
        mock_detect_ruler.return_value = ([[10.0, 20.0], [50.0, 20.0]], 0.25, {"ticks": 2})

        color = self.client.post("/api/analysis/color", json=self._ref())
        self.assertEqual(color.status_code, 200)
        self.assertTrue(color.json()["color_correction"])
        self.assertEqual(color.json()["centers"], [[12, 16], [24, 32]])
        corrected = self.client.get(color.json()["image_url"])
        self.assertEqual(corrected.status_code, 200)
        self.assertEqual(corrected.headers["content-type"], "image/jpeg")

        ruler = self.client.post("/api/analysis/ruler", json=self._ref())
        self.assertEqual(ruler.status_code, 200)
        self.assertEqual(ruler.json()["tickpoints"], [[10.0, 20.0], [50.0, 20.0]])
        self.assertEqual(ruler.json()["scale"], {"per_pixel": 0.25, "unit": "mm"})

    @patch("web.app.color_correct_with_report")
    def test_calibration_is_idempotent(self, mock_color_correct):
        """A second press must not fit a new CCM on already corrected pixels."""
        corrected = np.full(
            (self.image["height"], self.image["width"], 3), 120, dtype=np.uint8
        )
        mock_color_correct.return_value = _color_report(corrected, [[4, 6]])

        first = self.client.post("/api/analysis/color", json=self._ref())
        second = self.client.post("/api/analysis/color", json=self._ref())

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertTrue(first.json()["recomputed"])
        self.assertFalse(second.json()["recomputed"])
        self.assertEqual(mock_color_correct.call_count, 1)
        # The one call it did make saw the original upload, not a corrected frame.
        passed_image = mock_color_correct.call_args[0][0]
        self.assertEqual(int(passed_image[0, 0, 0]), 30)

    @patch("web.app.color_correct_with_report")
    def test_corrected_preview_is_capped_and_cached(self, mock_color_correct):
        big = np.full((3000, 4000, 3), 90, dtype=np.uint8)
        mock_color_correct.return_value = _color_report(big, [[1, 2]])
        self.client.post("/api/analysis/color", json=self._ref())

        first = self.client.get("/api/analysis/corrected-image", params=self._ref())
        second = self.client.get("/api/analysis/corrected-image", params=self._ref())

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.content, second.content)
        with Image.open(BytesIO(first.content)) as preview:
            self.assertEqual(max(preview.size), web_app.PREVIEW_MAX_SIDE)
        # A full-size PNG of this frame would be megabytes.
        self.assertLess(len(first.content), 1_500_000)

    @patch("web.app.color_correct_with_report")
    def test_corrected_frames_are_capped_in_memory(self, mock_color_correct):
        corrected = np.full(
            (self.image["height"], self.image["width"], 3), 77, dtype=np.uint8
        )
        mock_color_correct.return_value = _color_report(corrected, [[1, 1]])

        refs = [self._ref()]
        for _ in range(MAX_CORRECTED_IMAGES):
            image = self._upload(self.session_id, name="extra.png")
            refs.append({"session_id": self.session_id, "image_id": image["id"]})
        for ref in refs:
            self.assertEqual(self.client.post("/api/analysis/color", json=ref).status_code, 200)

        store = web_app.get_store()
        held = [
            store.get(ref["session_id"], ref["image_id"]).corrected_image is not None
            for ref in refs
        ]
        self.assertEqual(sum(held), MAX_CORRECTED_IMAGES)
        # The evicted analysis still knows it was calibrated ...
        evicted = store.get(refs[0]["session_id"], refs[0]["image_id"])
        self.assertTrue(evicted.color_correction)
        self.assertIsNone(evicted.corrected_image)
        # ... and recomputes instead of silently serving raw pixels.
        mock_color_correct.reset_mock()
        traits = self.client.post("/api/analysis/traits", json=refs[0])
        self.assertEqual(traits.status_code, 200)
        self.assertEqual(mock_color_correct.call_count, 0)  # no objects yet
        self.client.post("/api/analysis/embedding", json=refs[0])
        self._click_ref(refs[0], commit=True)
        self.client.post("/api/analysis/traits", json=refs[0])
        self.assertEqual(mock_color_correct.call_count, 1)

    def _click_ref(self, ref, commit=False):
        body = dict(ref)
        body["points"] = [{"x": 100, "y": 75, "label": 1}]
        body["commit"] = commit
        return self.client.post("/api/analysis/click", json=body)

    @patch("web.app.get_semantic_detector")
    def test_semantic_detection_warns_before_creating_objects(self, mock_detector):
        """Prompt/image mismatch mirrors the desktop warning: nothing created."""

        class Detector:
            def __init__(self):
                self.detect_calls = 0

            def resolve_organ(self, _image, _prompt, **_kwargs):
                return _prompt, None, None

            def cross_check(self, _image, _prompt, summary=None):
                return {
                    "requested": ["seed"],
                    "requested_count": 0,
                    "requested_score": 0.0,
                    "suggested": "pod",
                    "suggested_count": 3,
                    "suggested_score": 0.81,
                    "summary": {},
                }

            def detect(self, _image, _prompt):
                self.detect_calls += 1
                return [{"box": [60, 40, 140, 110], "score": 0.9, "label": "bean pod"}]

        detector = Detector()
        mock_detector.return_value = detector
        self._prepare()

        warned = self.client.post(
            "/api/analysis/semantic", json=dict(self._ref(), prompt="seed")
        )
        self.assertEqual(warned.status_code, 200)
        payload = warned.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "semantic_mismatch")
        self.assertEqual(payload["mismatch"]["suggested"], "pod")
        self.assertEqual(payload["objects"], [])
        self.assertEqual(detector.detect_calls, 0)

        forced = self.client.post(
            "/api/analysis/semantic",
            json=dict(self._ref(), prompt="seed", skip_semantic_check=True),
        )
        self.assertEqual(forced.status_code, 200)
        self.assertTrue(forced.json()["ok"])
        self.assertEqual(len(forced.json()["created"]), 1)
        self.assertEqual(detector.detect_calls, 1)

    @patch("web.app.get_semantic_detector")
    def test_semantic_detection_drops_size_outliers(self, mock_detector):
        """Desktop parity: one oversized box among five is filtered away."""
        boxes = [[10 + index * 20, 10, 26 + index * 20, 40] for index in range(4)]
        boxes.append([0, 0, 199, 149])

        class Detector:
            def cross_check(self, _image, _prompt, summary=None):
                return None

            def resolve_organ(self, _image, _prompt, **_kwargs):
                return _prompt, None, None

            def detect(self, _image, _prompt):
                return [
                    {"box": box, "score": 0.8, "label": "bean pod"} for box in boxes
                ]

        mock_detector.return_value = Detector()
        self._prepare()
        response = self.client.post(
            "/api/analysis/semantic", json=dict(self._ref(), prompt="pod")
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["detections"], 5)
        self.assertEqual(len(payload["created"]), 4)
        self.assertEqual(payload["filtered"], 1)
        self.assertEqual(
            [obj["label"] for obj in payload["created"]],
            ["pod01", "pod02", "pod03", "pod04"],
        )

    def test_click_coordinates_are_clamped_to_the_image(self):
        self._prepare()
        response = self._click([{"x": 10_000, "y": -40, "label": 1}], commit=True)
        self.assertEqual(response.status_code, 200)
        prompt = response.json()["object"]["prompt_points"][0]
        self.assertEqual(prompt["x"], self.image["width"] - 1)
        self.assertEqual(prompt["y"], 0)

    def test_failed_prediction_reports_instead_of_raising(self):
        self._prepare()
        response = self._click([{"x": 0, "y": 0, "label": 1}], commit=True)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["objects"], [])

    # -- objects -------------------------------------------------------

    def test_object_naming_follows_desktop_rules(self):
        self._prepare()
        first = self._commit_object(80, 60)
        second = self._commit_object(100, 75)
        third = self._commit_object(120, 90)
        self.assertEqual(
            [first["label"], second["label"], third["label"]],
            ["Object01", "Object02", "Object03"],
        )

        deleted = self.client.delete(
            f"/api/analysis/objects/{second['uid']}", params=self._ref()
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(len(deleted.json()["objects"]), 2)

        # max + 1, so the freed number 02 is not reused.
        fourth = self._commit_object(90, 70)
        self.assertEqual(fourth["label"], "Object04")

        pod = self._commit_object(105, 80, name_prefix="pod")
        self.assertEqual(pod["label"], "pod01")

    def test_rename_object(self):
        self._prepare()
        first = self._commit_object(80, 60)
        second = self._commit_object(120, 90)

        renamed = self.client.patch(
            f"/api/analysis/objects/{first['uid']}",
            json=dict(self._ref(), label="my pod"),
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["object"]["label"], "my pod")

        duplicate = self.client.patch(
            f"/api/analysis/objects/{second['uid']}",
            json=dict(self._ref(), label="my pod"),
        )
        self.assertEqual(duplicate.status_code, 409)

        blank = self.client.patch(
            f"/api/analysis/objects/{second['uid']}",
            json=dict(self._ref(), label="   "),
        )
        self.assertEqual(blank.status_code, 409)

        unknown = self.client.patch(
            "/api/analysis/objects/999", json=dict(self._ref(), label="ghost")
        )
        self.assertEqual(unknown.status_code, 404)

    def test_delete_unknown_object(self):
        self._prepare()
        response = self.client.delete(
            "/api/analysis/objects/42", params=self._ref()
        )
        self.assertEqual(response.status_code, 404)

    def test_state_endpoint_reports_objects_and_headers(self):
        self._prepare()
        self._commit_object(100, 75)
        response = self.client.get("/api/analysis", params=self._ref())

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object_count"], 1)
        self.assertEqual(payload["embedding"]["state"], "ready")
        self.assertEqual(payload["trait_headers"], trait_headers("pixel"))
        self.assertEqual(payload["width"], self.image["width"])

    def test_deleting_the_image_drops_the_embedding(self):
        self._prepare()
        key = f"{self.session_id}:{self.image_id}"
        self.client.delete(f"/api/images/{self.image_id}")
        self.assertIn(key, self.engine.dropped)

    # -- scale and traits ---------------------------------------------

    def test_scale_scales_traits(self):
        self._prepare()
        self._commit_object(100, 75)

        pixel_traits = self.client.post("/api/analysis/traits", json=self._ref()).json()
        self.assertEqual(pixel_traits["headers"], trait_headers("pixel"))
        self.assertEqual(len(pixel_traits["rows"]), 1)
        self.assertEqual(pixel_traits["rows"][0][0], "Object01")

        scaled = self.client.post(
            "/api/analysis/scale", json=dict(self._ref(), per_pixel=0.05, unit="mm")
        )
        self.assertEqual(scaled.status_code, 200)
        self.assertEqual(scaled.json()["scale"], {"per_pixel": 0.05, "unit": "mm"})

        mm_traits = self.client.post("/api/analysis/traits", json=self._ref()).json()
        # The unit rides in the header, so a bare "Area" no longer matches.
        self.assertIn("Area (mm\u00b2)", mm_traits["headers"])
        self.assertIn("Perimeter (mm)", mm_traits["headers"])
        area_index = column_index(mm_traits["headers"], "Area")
        perimeter_index = column_index(mm_traits["headers"], "Perimeter")
        self.assertAlmostEqual(
            mm_traits["rows"][0][area_index],
            pixel_traits["rows"][0][area_index] * 0.05 ** 2,
            places=4,
        )
        self.assertAlmostEqual(
            mm_traits["rows"][0][perimeter_index],
            pixel_traits["rows"][0][perimeter_index] * 0.05,
            places=4,
        )
        self.assertEqual(mm_traits["scale"], {"per_pixel": 0.05, "unit": "mm"})

    def test_traits_for_a_subset_of_objects(self):
        self._prepare()
        first = self._commit_object(80, 60)
        self._commit_object(120, 90)

        response = self.client.post(
            "/api/analysis/traits", json=dict(self._ref(), object_ids=[first["uid"]])
        )
        self.assertEqual(response.status_code, 200)
        rows = response.json()["rows"]
        self.assertEqual([row[0] for row in rows], ["Object01"])

        unknown = self.client.post(
            "/api/analysis/traits", json=dict(self._ref(), object_ids=[999])
        )
        self.assertEqual(unknown.status_code, 400)

    def test_rejects_non_positive_scale(self):
        response = self.client.post(
            "/api/analysis/scale", json=dict(self._ref(), per_pixel=0, unit="mm")
        )
        self.assertEqual(response.status_code, 422)

    def test_traits_without_objects(self):
        self._prepare()
        response = self.client.post("/api/analysis/traits", json=self._ref())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["rows"], [])

    # -- annotation ----------------------------------------------------

    def test_annotation_export_matches_the_documented_format(self):
        self._prepare()
        self._commit_object(100, 75)
        self.client.post(
            "/api/analysis/scale", json=dict(self._ref(), per_pixel=0.0523, unit="mm")
        )

        response = self.client.get("/api/analysis/annotation", params=self._ref())
        self.assertEqual(response.status_code, 200)
        self.assertIn("photo.pheno.json", response.headers["content-disposition"])

        document = json.loads(response.content.decode("utf-8"))
        self.assertEqual(document["format"], "phenome-annotation")
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["image"]["path"], "photo.png")
        self.assertEqual(document["image"]["width"], self.image["width"])
        self.assertEqual(len(document["image"]["md5"]), 32)
        self.assertEqual(document["scale"], {"per_pixel": 0.0523, "unit": "mm"})
        self.assertEqual(
            document["processing"],
            {"color_correction": False, "ruler_detection": False},
        )

        obj = document["objects"][0]
        self.assertEqual(obj["label"], "Object01")
        self.assertEqual(obj["prompt_points"], [{"x": 100, "y": 75, "label": 1}])
        self.assertEqual(obj["mask"]["encoding"], "rle-zlib-base64")

        mask = annotation.decode_mask(obj["mask"])
        self.assertEqual(mask.shape, (self.image["height"], self.image["width"]))
        self.assertEqual(int(mask.sum()), 2 * FakeEngine.HALF_WIDTH * 2 * FakeEngine.HALF_HEIGHT)

    def test_annotation_inline_payload(self):
        self._prepare()
        response = self.client.get(
            "/api/analysis/annotation", params=dict(self._ref(), download=False)
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("content-disposition", response.headers)
        self.assertEqual(response.json()["objects"], [])

    def test_result_download_is_the_trait_table_only(self):
        """The phone deliverable is the table; masks stay in the sidecar."""
        self._prepare()
        self._commit_object(100, 75)
        self.client.post(
            "/api/analysis/scale", json=dict(self._ref(), per_pixel=0.05, unit="mm")
        )
        response = self.client.get("/api/analysis/results", params=self._ref())

        self.assertEqual(response.status_code, 200)
        # octet-stream on purpose: mobile browsers render text/csv inline.
        self.assertEqual(response.headers["content-type"], "application/octet-stream")
        self.assertIn("_traits.csv", response.headers["content-disposition"])


        text = response.content.decode("utf-8-sig")
        rows = [line for line in text.splitlines() if line.strip()]
        # The scale was set to mm above, so the CSV columns must say so: a column of
        # bare numbers is exactly what made the exported table unreadable.
        self.assertEqual(rows[0].split(","), trait_headers("mm"))
        self.assertIn("Area (mm\u00b2)", rows[0])
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[1].startswith("Object01,"))
        # Small enough for a phone: the old ZIP was ~12 MB on a 16 MPx frame.
        self.assertLess(len(response.content), 4096)

    def test_result_download_uses_the_current_scale(self):
        self._prepare()
        self._commit_object(100, 75)
        pixels = self.client.get("/api/analysis/results", params=self._ref()).content
        self.client.post(
            "/api/analysis/scale", json=dict(self._ref(), per_pixel=0.5, unit="mm")
        )
        millimeters = self.client.get("/api/analysis/results", params=self._ref()).content

        area_index = column_index(trait_headers("mm"), "Area")
        pixel_area = float(pixels.decode("utf-8-sig").splitlines()[1].split(",")[area_index])
        mm_area = float(millimeters.decode("utf-8-sig").splitlines()[1].split(",")[area_index])
        self.assertAlmostEqual(mm_area, pixel_area * 0.25, places=3)


class AnalysisUnitTestCase(unittest.TestCase):
    """Pure CPU pieces: polygons, naming, traits, engine helpers."""

    @staticmethod
    def _pod_candidate(area):
        return {
            "label": "pod",
            "score": 0.5,
            "traits": {"area": area, "major_axis": area ** 0.5,
                       "aspect_ratio": 3.0, "extent": 0.5, "solidity": 0.9},
        }

    def test_size_difference_between_varieties_is_not_an_outlier(self):
        """Regression: a plate of four real pods came back as three.

        A plate photographed for variety description holds organs of very
        different sizes on purpose -- one large pod beside a small one is the
        point of the picture. The old 0.5x-2.0x band dropped the largest as an
        "outlier", so the semantic detection reported four detections and
        created three objects.
        """
        from pheno.canvas import _filter_text_mask_candidates

        group = [self._pod_candidate(area)
                 for area in (140_000, 50_000, 44_000, 30_000)]

        accepted, filtered = _filter_text_mask_candidates(group)

        self.assertEqual(len(accepted), 4, "every real pod has to survive")
        self.assertEqual(filtered, [])

    def test_obvious_junk_is_still_filtered_from_a_larger_group(self):
        """Widening the band must not turn the filter off altogether."""
        from pheno.canvas import _filter_text_mask_candidates

        speck = self._pod_candidate(300)
        group = [self._pod_candidate(area)
                 for area in (50_000, 48_000, 46_000, 47_000, 45_000, 49_000)] + [speck]

        accepted, filtered = _filter_text_mask_candidates(group)

        self.assertEqual(len(accepted), 6)
        self.assertEqual(filtered, [speck])

    def test_the_outlier_rule_lives_in_one_place(self):
        """canvas, SAM, review and the cohort tool must agree on the band."""
        from pheno import canvas, outliers, review

        self.assertEqual(canvas.MIN_GROUP_FOR_OUTLIERS, outliers.MIN_GROUP_FOR_OUTLIERS)
        self.assertEqual(review.MIN_GROUP_FOR_OUTLIERS, outliers.MIN_GROUP_FOR_OUTLIERS)
        # The filters used to hard-code their own copy of the band.
        for module in (canvas, review):
            source = Path(module.__file__).read_text(encoding="utf-8")
            self.assertNotIn("2.0 * median", source, module.__name__)

    def test_polygon_from_mask_traces_the_rectangle(self):
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[10:30, 20:50] = 1
        polygon = polygon_from_mask(mask)

        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        # ``compute_polygon_from_mask`` traces contours on a 1px-padded mask and
        # does not subtract that padding, so the desktop polygons sit half a
        # pixel outside the mask plus that one pixel. Reproduced here on purpose.
        self.assertGreaterEqual(len(polygon), 4)
        self.assertAlmostEqual(min(xs), 20.5, places=1)
        self.assertAlmostEqual(max(xs), 50.5, places=1)
        self.assertAlmostEqual(min(ys), 10.5, places=1)
        self.assertAlmostEqual(max(ys), 30.5, places=1)

    def test_object_public_payload(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:15, 8:20] = 1
        obj = AnalysisObject()
        obj.set(points=[[8.0, 5.0], [20.0, 5.0], [20.0, 15.0]], mask=mask, label="pod01")
        obj.add_prompt_point(12, 9, 1)
        obj.uid = 3

        payload = obj.to_public()
        self.assertEqual(payload["label"], "pod01")
        self.assertEqual(payload["area_pixels"], 120)
        self.assertEqual(payload["bbox"], [8, 5, 19, 14])
        self.assertEqual(payload["prompt_points"], [{"x": 12, "y": 9, "label": 1}])

    def test_compute_traits_matches_the_desktop_algorithm(self):
        image = np.full((50, 50, 3), 40, dtype=np.uint8)
        image[10:30, 12:36] = (100, 150, 200)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:30, 12:36] = 1

        obj = AnalysisObject()
        obj.set(points=[], mask=mask, label="Object01")
        rows = compute_traits(image, [obj], 0.5)
        expected = PhenomicsCalculator.defaultPhenomics(image, mask, 0.5)

        self.assertEqual(rows[0][0], "Object01")
        self.assertEqual(len(rows[0]), len(trait_headers("pixel")))
        for index, value in enumerate(expected):
            self.assertAlmostEqual(rows[0][index + 1], round(float(value), 6), places=5)

    def test_trait_headers_mirror_the_desktop_table(self):
        self.assertEqual(trait_headers("mm"), PhenomicsCalculator.getHeaders("mm"))
        self.assertEqual(trait_headers()[0], "Name")

    def test_store_decodes_and_bounds_its_cache(self):
        store = AnalysisStore(max_entries=2, max_decoded=1)
        data = make_sample_png()
        record = {
            "filename": "bean.png",
            "width": 200,
            "height": 150,
            "data": data,
        }
        first = store.get_or_create("s1", "i1", record)
        again = store.get_or_create("s1", "i1", record)
        self.assertIs(first, again)
        self.assertEqual(len(first.digest), 32)

        store.get_or_create("s2", "i2", record)
        store.get_or_create("s3", "i3", record)
        self.assertIsNone(store.get("s1", "i1"))

        image = store.decode("i1", data)
        self.assertEqual(image.shape, (150, 200, 3))
        self.assertIs(store.decode("i1", data), image)
        store.decode("i2", data)
        self.assertEqual(store.stats()["decoded_images"], 1)

    def test_scale_validation(self):
        store = AnalysisStore()
        analysis = store.get_or_create(
            "s", "i", {"filename": "a.png", "width": 10, "height": 10, "data": b"x"}
        )
        self.assertEqual(
            analysis.set_scale(2.5, "mm"), {"per_pixel": 2.5, "unit": "mm"}
        )
        with self.assertRaises(Exception):
            analysis.set_scale(0, "mm")

    def test_engine_helpers(self):
        gray = np.zeros((10, 12), dtype=np.uint8)
        self.assertEqual(to_bgr(gray).shape, (10, 12, 3))
        bgra = np.zeros((10, 12, 4), dtype=np.uint8)
        self.assertEqual(to_bgr(bgra).shape, (10, 12, 3))

        large = np.zeros((2000, 4000, 3), dtype=np.uint8)
        resized, scale = resize_for_embedding(large)
        self.assertEqual(resized.shape[:2], (512, 1024))
        self.assertAlmostEqual(scale, 0.256, places=3)

        small = np.zeros((100, 80, 3), dtype=np.uint8)
        untouched, unit_scale = resize_for_embedding(small)
        self.assertEqual(unit_scale, 1.0)
        self.assertIs(untouched, small)

    def test_domain_containing_points_picks_the_clicked_blob(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        mask[20:30, 20:30] = 1

        picked = domain_containing_points(mask, [(25, 25)])
        self.assertEqual(int(picked.sum()), 100)
        self.assertEqual(picked[3, 3], 0)

        self.assertIsNone(domain_containing_points(mask, [(15, 15)]))


class RulerScaleTestCase(unittest.TestCase):
    """The real ``web/ruler.py`` conversion, with only the ONNX model stubbed.

    Everything above this line mocks ``detect_ruler`` wholesale, which would let
    the pixels-per-cm to millimeters-per-pixel conversion drift unnoticed.
    """

    def setUp(self):
        self.model_path = Path(__file__).resolve().parent.parent / "models" / "rulernet.onnx"

    def _detect(self, tickpoints, pixels_per_cm, uniformity=None):
        class Stub:
            def __init__(self, _path):
                pass

            def get_tickpoints_and_scale(self, _image):
                return tickpoints, pixels_per_cm

            def measure_scale_uniformity(self, _image, _points):
                return uniformity

        image = np.zeros((40, 40, 3), dtype=np.uint8)
        with patch("pheno.RulerNet.RulerNet", Stub):
            return detect_ruler(image, self.model_path)

    def test_conversion_matches_the_desktop_formula(self):
        if not self.model_path.is_file():
            self.skipTest("rulernet.onnx is not present")

        pixels_per_cm = 37.7952
        points, millimeters_per_pixel, report = self._detect(
            [[10.0, 20.0], [47.8, 20.4]], pixels_per_cm
        )

        # Desktop: pheno/app.py does 1 / (pred_cm_pix / 10.0).
        self.assertAlmostEqual(millimeters_per_pixel, 1 / (pixels_per_cm / 10.0), places=12)
        self.assertAlmostEqual(millimeters_per_pixel, 0.2645839, places=6)
        self.assertEqual(points, [[10.0, 20.0], [47.8, 20.4]])
        self.assertEqual(report["ticks"], 2)

    def test_scale_drift_is_reported_but_never_applied(self):
        """A 4.5% end-to-end drift is real in these photographs; report, don't fix."""
        if not self.model_path.is_file():
            self.skipTest("rulernet.onnx is not present")

        pixels_per_cm = 100.0
        _points, millimeters_per_pixel, report = self._detect(
            [[10.0, 20.0], [110.0, 20.0]],
            pixels_per_cm,
            uniformity={"drift": 0.045, "head_pitch": 9.8, "tail_pitch": 10.25},
        )

        self.assertEqual(report["drift_percent"], 4.5)
        self.assertTrue(report["uneven"])
        # The scale itself is untouched by the diagnostic.
        self.assertAlmostEqual(millimeters_per_pixel, 0.1, places=12)

        _points, _mm, quiet = self._detect(
            [[10.0, 20.0], [110.0, 20.0]],
            pixels_per_cm,
            uniformity={"drift": 0.004, "head_pitch": 10.0, "tail_pitch": 10.04},
        )
        self.assertFalse(quiet["uneven"])

    def test_unusable_detections_are_reported(self):
        if not self.model_path.is_file():
            self.skipTest("rulernet.onnx is not present")

        for tickpoints, pixels_per_cm in (
            (None, 37.0),
            ([[1.0, 2.0]], 37.0),
            ([[1.0, 2.0], [3.0, 4.0]], 0.0),
        ):
            with self.assertRaises(RulerDetectionError):
                self._detect(tickpoints, pixels_per_cm)

    def test_missing_model_file_is_reported(self):
        with self.assertRaises(RulerDetectionError):
            detect_ruler(np.zeros((8, 8, 3), dtype=np.uint8), Path("no-such-ruler.onnx"))


class ColorCalibrationTestCase(unittest.TestCase):
    """The real CCM fit in ``web/calibration.py`` with a stubbed MCC detector."""

    @staticmethod
    def _charts_rgb(patch_colors):
        """Shape the values the way ``CChecker.getChartsRGB`` returns them."""
        values = np.asarray(patch_colors, dtype=np.float64).reshape(-1)
        table = np.zeros((values.size, 2), dtype=np.float64)
        table[:, 1] = values
        return table

    def _fake_mcc(
        self,
        patch_colors,
        box=((0, 0), (60, 0), (60, 40), (0, 40)),
        found=True,
        extra_charts=(),
    ):
        charts = self._charts_rgb(patch_colors)
        decoys = [self._charts_rgb(colors) for colors in extra_charts]

        class Checker:
            def __init__(self, table):
                self._table = table

            def getChartsRGB(self):
                return self._table

            def getBox(self):
                return [list(corner) for corner in box]

        class Detector:
            def process(self, _image, _kind, _count):
                return found

            def getListColorChecker(self):
                if not found:
                    return []
                # Decoys first, so ordering cannot be what makes a test pass.
                return [Checker(table) for table in decoys] + [Checker(charts)]

        class FakeMcc:
            MCC24 = 0

            @staticmethod
            def CCheckerDetector_create():
                return Detector()

        return FakeMcc

    def test_identity_fit_leaves_the_image_alone(self):
        reference = (MCC24_REF_SRGB * 255.0).round()
        image = np.full((6, 6, 3), (68, 82, 115), dtype=np.uint8)  # BGR of patch 1

        with patch.object(cv2, "mcc", self._fake_mcc(reference), create=True):
            corrected, centers = color_correct_bgr(image)

        self.assertEqual(corrected.shape, image.shape)
        self.assertTrue(np.all(np.abs(corrected.astype(int) - image.astype(int)) <= 2))
        self.assertEqual(len(centers), 24)

    def test_darkened_chart_is_brightened_back(self):
        reference = (MCC24_REF_SRGB * 255.0).round()
        detected = (reference * 0.5).round()
        # An image painted with a *detected* (darkened) patch color should come
        # back close to the reference color for that patch.
        detected_bgr = tuple(int(value) for value in detected[6][::-1])  # orange patch
        expected_rgb = reference[6]
        image = np.full((4, 4, 3), detected_bgr, dtype=np.uint8)

        with patch.object(cv2, "mcc", self._fake_mcc(detected), create=True):
            corrected, _centers = color_correct_bgr(image)

        corrected_rgb = corrected[0, 0][::-1].astype(float)
        self.assertTrue(
            np.all(np.abs(corrected_rgb - expected_rgb) <= 12),
            (corrected_rgb, expected_rgb),
        )
        self.assertTrue((corrected_rgb > np.asarray(detected_bgr[::-1])).all())

    def test_patch_centers_follow_the_detected_box(self):
        reference = (MCC24_REF_SRGB * 255.0).round()
        image = np.zeros((50, 70, 3), dtype=np.uint8)

        with patch.object(cv2, "mcc", self._fake_mcc(reference), create=True):
            _corrected, centers = color_correct_bgr(image)

        self.assertEqual(len(centers), 24)
        self.assertEqual(centers[0], [5, 5])
        self.assertEqual(centers[-1], [55, 35])

    def test_missing_chart_and_bad_input_are_reported(self):
        reference = (MCC24_REF_SRGB * 255.0).round()
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch.object(cv2, "mcc", self._fake_mcc(reference, found=False), create=True):
            with self.assertRaises(CalibrationError):
                color_correct_bgr(image)

        with patch.object(cv2, "mcc", self._fake_mcc(reference), create=True):
            with self.assertRaises(CalibrationError):
                color_correct_bgr(np.zeros((10, 10), dtype=np.uint8))
            with self.assertRaises(CalibrationError):
                color_correct_bgr(self._charts_rgb(reference))  # wrong shape

    def test_patch_count_mismatch_is_reported(self):
        short = (MCC24_REF_SRGB[:12] * 255.0).round()
        with patch.object(cv2, "mcc", self._fake_mcc(short), create=True):
            with self.assertRaises(CalibrationError):
                color_correct_bgr(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_offset_term_beats_a_plain_3x3_on_a_tone_curve(self):
        """Why the fit is 4x3: a camera's tone curve is not a pure gain.

        Measured on five DUS photographs (`python -m tools.eval_color`), the 4x3
        fit's leave-one-out error is 5.94 dE against 7.24 dE for the old 3x3, and
        linearising first is worse still (9.74). This test reproduces the reason in
        miniature: an S-curve response cannot be undone by a matrix alone.
        """
        from pheno.color import delta_e, fit_matrix

        reference = MCC24_REF_SRGB.astype(np.float32)
        # A gentle S-curve plus a channel gain, i.e. what a JPEG pipeline does.
        curved = np.clip(reference + 0.12 * np.sin(np.pi * reference) - 0.04, 0, 1)
        curved = np.clip(curved * np.array([1.05, 1.0, 0.92], dtype=np.float32), 0, 1)

        with_offset = fit_matrix(curved, reference)
        corrected = np.clip(
            np.hstack([curved, np.ones((len(curved), 1), np.float32)]) @ with_offset, 0, 1
        )
        plain, *_ = np.linalg.lstsq(curved, reference, rcond=None)
        plain_corrected = np.clip(curved @ plain, 0, 1)

        self.assertLess(
            float(np.mean(delta_e(corrected, reference))),
            float(np.mean(delta_e(plain_corrected, reference))),
        )

    def test_a_chart_that_does_not_fit_is_refused(self):
        """A grey card or a mis-paired chart must not silently recolour the photo."""
        from pheno.color import MAX_MEAN_DELTA_E

        # Patch values with no relation to the reference: any matrix fitted here
        # leaves a large residual.
        rng = np.random.default_rng(11)
        nonsense = rng.random((24, 3)).astype(np.float32)

        with patch.object(cv2, "mcc", self._fake_mcc((nonsense * 255).round()), create=True):
            with self.assertRaises(CalibrationError) as caught:
                color_correct_bgr(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertIn("does not match a color checker", str(caught.exception))
        self.assertGreater(MAX_MEAN_DELTA_E, 0)

    def test_block_application_matches_a_single_shot(self):
        """Blocked application is only an allocation trick; results must be identical."""
        from pheno.color import BLOCK_ROWS, apply_matrix

        rng = np.random.default_rng(5)
        image = rng.random((BLOCK_ROWS * 2 + 7, 9, 3)).astype(np.float32)
        matrix = np.array(
            [[1.1, -0.05, 0.0], [0.02, 0.95, 0.03], [-0.01, 0.04, 1.05], [0.01, 0.0, -0.02]],
            dtype=np.float32,
        )

        blocked = apply_matrix(image, matrix)
        flat = np.clip(
            image.reshape(-1, 3) @ matrix[:3] + matrix[3], 0, 1
        ).reshape(image.shape)

        self.assertTrue(np.allclose(blocked, flat, atol=1e-6))

    def test_the_best_fitting_chart_wins(self):
        """With several candidates, the residual decides -- not detector order."""
        reference = (MCC24_REF_SRGB * 255.0).round()
        rng = np.random.default_rng(2)
        decoy = (rng.random((24, 3)) * 255).round()

        # The decoy is returned first; the real chart must still be used.
        with patch.object(
            cv2, "mcc", self._fake_mcc(reference, extra_charts=[decoy]), create=True
        ):
            corrected, centers = color_correct_bgr(
                np.full((6, 6, 3), (68, 82, 115), dtype=np.uint8)
            )

        self.assertEqual(len(centers), 24)
        # Correcting a patch-1 coloured image with the real chart is near-identity.
        expected = np.array([68, 82, 115], dtype=int)
        self.assertTrue(np.all(np.abs(corrected.astype(int) - expected) <= 6))


class EnginePromptScalingTestCase(unittest.TestCase):
    """Prompt geometry really is scaled into embedding space and back."""

    class FakePredictor:
        def __init__(self, scaled_size):
            self.scaled_height, self.scaled_width = scaled_size
            self.calls = []
            self.features = None
            self.original_size = None
            self.input_size = None
            self.is_image_set = False

        def reset_image(self):
            self.is_image_set = False

        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
            self.calls.append(
                {
                    "points": None if point_coords is None else np.asarray(point_coords).tolist(),
                    "labels": None if point_labels is None else np.asarray(point_labels).tolist(),
                    "box": None if box is None else np.asarray(box).tolist(),
                }
            )
            mask = np.zeros((1, self.scaled_height, self.scaled_width), dtype=bool)
            if box is not None:
                x1, y1, x2, y2 = (int(round(value)) for value in np.asarray(box).tolist())
                mask[0, y1:y2, x1:x2] = True
            else:
                for x, y in np.asarray(point_coords).tolist():
                    mask[0, max(0, int(y) - 3):int(y) + 3, max(0, int(x) - 3):int(x) + 3] = True
            return mask, np.array([0.9]), np.zeros((1, 4, 4))

    def _engine(self, scale=0.25, raw_shape=(1200, 1600)):
        engine = SamEngine()
        scaled = (int(raw_shape[0] * scale), int(raw_shape[1] * scale))
        predictor = self.FakePredictor(scaled)
        engine._predictor = predictor
        engine._model_state = "ready"
        engine._embeddings["s:i"] = {
            "state": "ready",
            "error": None,
            "features": object(),
            "original_size": scaled,
            "input_size": scaled,
            "scale": scale,
            "raw_shape": raw_shape,
            "seconds": 0.1,
        }
        return engine, predictor

    def test_box_prompt_is_scaled_and_the_mask_comes_back_full_size(self):
        engine, predictor = self._engine()
        try:
            mask = engine.segment_box("s:i", [400, 200, 800, 600])
        finally:
            engine.shutdown()

        self.assertEqual(predictor.calls[0]["box"], [100.0, 50.0, 200.0, 150.0])
        self.assertEqual(mask.shape, (1200, 1600))
        # The full-size mask lands back on the requested box, within the 4x
        # nearest-neighbour rounding of the embedding downscale.
        columns = np.flatnonzero(mask.any(axis=0))
        rows = np.flatnonzero(mask.any(axis=1))
        self.assertAlmostEqual(int(columns[0]), 400, delta=4)
        self.assertAlmostEqual(int(columns[-1]), 799, delta=4)
        self.assertAlmostEqual(int(rows[0]), 200, delta=4)
        self.assertAlmostEqual(int(rows[-1]), 599, delta=4)

    def test_point_prompts_are_truncated_like_the_desktop(self):
        engine, predictor = self._engine()
        try:
            engine.segment("s:i", [(403.9, 207.9, 1)], one_domain=False)
        finally:
            engine.shutdown()

        # SAM.add_point does int(x * scale): 403.9 * 0.25 -> 100, not 101.
        self.assertEqual(predictor.calls[0]["points"], [[100, 51]])
        self.assertEqual(predictor.calls[0]["labels"], [1])

    def test_box_prompt_needs_four_numbers(self):
        engine, _predictor = self._engine()
        try:
            with self.assertRaises(EngineError):
                engine.segment_box("s:i", [1, 2, 3])
        finally:
            engine.shutdown()

    def test_box_prompt_without_an_embedding_is_rejected(self):
        engine = SamEngine()
        try:
            with self.assertRaises(EngineNotReady):
                engine.segment_box("missing", [1, 2, 3, 4])
        finally:
            engine.shutdown()


class RulerScoringTestCase(unittest.TestCase):
    """The ruler detector's own guardrails, without loading the ONNX model.

    `web/ruler.py` only trusts what RulerNet returns, so the value of the scale
    depends entirely on these checks: they are what turned a 90% / 13%
    false-positive detector into 100% / 0% on the 50-image DUS sample.
    """

    @staticmethod
    def _detector():
        from pheno.RulerNet import RulerNet

        # __new__ skips __init__, so no onnxruntime session is created.
        return RulerNet.__new__(RulerNet)

    @staticmethod
    def _ruler_image(mm_pixels=10.0, size=1200):
        """A steel-rule lookalike: millimetre ticks, every tenth one longer."""
        image = np.full((size, size), 40, np.uint8)
        for index in range(int(size / mm_pixels)):
            x = int(round(index * mm_pixels))
            bottom = 470 if index % 10 == 0 else 425
            image[380:bottom, max(0, x - 1):x + 2] = 235
        return image

    @staticmethod
    def _ticks(centimetre_pixels, count=None, y=420.0, size=1200):
        if count is None:
            # Keep the sequence inside the frame whatever the pitch is.
            count = max(5, int((size - 200) / centimetre_pixels) + 1)
        xs = np.arange(100.0, 100.0 + count * centimetre_pixels, centimetre_pixels)
        xs = xs[xs < size - 20]
        return np.stack([xs, np.full(len(xs), y)], axis=1)

    def test_measured_pitch_refines_the_predicted_scale(self):
        detector = self._detector()
        for mm_pixels in (8.0, 10.0, 18.0):
            image = self._ruler_image(mm_pixels)
            truth = mm_pixels * 10.0
            # Feed a slightly wrong prediction; the FFT should pull it onto truth.
            score, refined, notes = detector._score(image, self._ticks(truth), truth * 1.03)

            self.assertGreater(score, 0.0, notes)
            self.assertAlmostEqual(refined / truth, 1.0, delta=0.01, msg=notes)
            self.assertEqual(notes.get("pitch_ratio"), 10.0)

    def test_short_or_irregular_sequences_are_rejected(self):
        detector = self._detector()
        image = self._ruler_image(10.0)

        cases = {
            "too few ticks": self._ticks(100.0, count=3),
            "irregular spacing": np.stack(
                [np.array([100.0, 180.0, 300.0, 310.0, 500.0, 700.0, 705.0]), np.full(7, 420.0)],
                axis=1,
            ),
        }
        for expected, points in cases.items():
            score, _refined, notes = detector._score(image, points, 100.0)
            self.assertEqual(score, 0.0)
            self.assertEqual(notes.get("reject"), expected)

    def test_a_pitch_that_contradicts_the_image_is_rejected(self):
        detector = self._detector()
        image = self._ruler_image(10.0)

        score, _refined, notes = detector._score(image, self._ticks(37.0, count=25), 37.0)

        self.assertEqual(score, 0.0)
        self.assertEqual(notes.get("reject"), "pitch disagrees")

    def test_a_plain_scene_without_periodicity_is_rejected(self):
        detector = self._detector()
        rng = np.random.default_rng(3)
        image = rng.integers(0, 60, size=(1200, 1200), dtype=np.uint8)

        score, _refined, notes = detector._score(image, self._ticks(100.0), 100.0)

        self.assertEqual(score, 0.0)
        self.assertIn(notes.get("reject"), {"no periodicity confirmation", "pitch disagrees"})

    def test_letterbox_is_undone_for_a_portrait_frame(self):
        """The old code assumed landscape and put portrait ticks in the wrong place."""
        detector = self._detector()
        # 400x800 portrait: scale 0.96, so the letterbox pads 192 px left/right.
        image = np.zeros((800, 400, 3), np.uint8)
        detector.infer = lambda pil: (np.array([[288.0, 384.0], [288.0 + 96.0, 384.0]]), 96.0)

        points, spacing = detector._infer_region(image, origin=(10, 20))

        self.assertAlmostEqual(points[0][0], 10 + 100.0, places=3)
        self.assertAlmostEqual(points[0][1], 20 + 400.0, places=3)
        self.assertAlmostEqual(spacing, 100.0, places=3)


class TraitUnitTestCase(unittest.TestCase):
    """Units belong in the header, and they have to follow the current scale."""

    def test_pixel_units_when_no_scale_is_set(self):
        headers = trait_headers("pixel")

        self.assertIn("Area (px\u00b2)", headers)
        self.assertIn("Perimeter (px)", headers)
        self.assertIn("Major Axis (px)", headers)
        # No scale set is still a unit: leaving it blank invites reading pixels as
        # millimetres, which is the failure mode this whole change is about.
        self.assertEqual(trait_headers(None), headers)
        self.assertEqual(trait_headers(""), headers)

    def test_real_world_units_follow_the_scale(self):
        for unit, area, length in (("mm", "Area (mm\u00b2)", "Perimeter (mm)"),
                                   ("cm", "Area (cm\u00b2)", "Perimeter (cm)")):
            headers = trait_headers(unit)
            self.assertIn(area, headers)
            self.assertIn(length, headers)
            self.assertIn("Convex Area (%s\u00b2)" % unit, headers)
            self.assertIn("Convex Perimeter (%s)" % unit, headers)

    def test_dimensionless_columns_get_no_unit(self):
        """A ratio with a unit would be plain wrong, not merely noisy."""
        headers = trait_headers("mm")

        for ratio in ("Aspect Ratio", "Solidity", "Circularity", "Convexity"):
            self.assertIn(ratio, headers)

    def test_channel_columns_state_their_range(self):
        headers = trait_headers("mm")

        for channel in ("MeanB", "MeanG", "MeanR"):
            self.assertIn("%s (0-255)" % channel, headers)
            self.assertNotIn("%s (mm)" % channel, headers)

    def test_name_column_is_untouched(self):
        self.assertEqual(trait_headers("mm")[0], "Name")

    def test_unknown_columns_are_left_alone(self):
        """Custom (no-code) traits have unknown dimensions, so guessing a unit
        would be worse than printing none."""
        from pheno.calcPheno import PhenomicsCalculator

        original = PhenomicsCalculator.custom_headers
        try:
            PhenomicsCalculator.custom_headers = {
                "DEFAULT": PhenomicsCalculator.defaultHeaders(),
                "CUSTOM": ["My Index", "Area"],
            }
            headers = PhenomicsCalculator.getHeaders("mm")
        finally:
            PhenomicsCalculator.custom_headers = original

        self.assertIn("My Index", headers)
        self.assertNotIn("My Index (mm)", headers)
        # A custom column that reuses a known name does get the unit, which is the
        # right call: same name, same dimension.
        self.assertEqual(headers.count("Area (mm\u00b2)"), 2)

    def test_bare_headers_stay_available_as_identifiers(self):
        from pheno.calcPheno import PhenomicsCalculator

        self.assertEqual(PhenomicsCalculator.getBareHeaders()[:2], ["Name", "Area"])

    def test_desktop_refreshes_headers_on_every_scale_change(self):
        """Source check rather than a Qt run: the point is that no path that
        changes the scale forgets to rebuild the header, because a stale header
        with new numbers is worse than no unit at all."""
        source = (Path(__file__).resolve().parent.parent / "pheno" / "app.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("def traitHeaders(self):", source)
        self.assertIn("def refreshTraitHeaders(self):", source)
        # Manual scale dialog, ruler detection and loading an annotation file.
        self.assertGreaterEqual(source.count("self.refreshTraitHeaders()"), 3)
        # The old static call must be gone from the table-updating paths.
        self.assertNotIn("updateData([], PhenomicsCalculator.getHeaders())", source)

    def test_traits_response_and_csv_agree_on_the_unit(self):
        """The table on screen and the downloaded CSV must not disagree."""
        client = TestClient(app)
        with patch.object(web_app, "get_engine", lambda: FakeEngine()):
            session = client.post("/api/sessions", json={"source": "capture"}).json()
            image = client.post(
                "/api/images",
                params={"session_id": session["id"]},
                files={"file": ("bean.png", make_sample_png(), "image/png")},
            ).json()
            ref = {"session_id": session["id"], "image_id": image["id"]}
            client.post("/api/analysis/embedding", json=ref)
            client.post(
                "/api/analysis/click",
                json=dict(ref, points=[{"x": 100, "y": 75, "label": 1}], commit=True),
            )
            client.post("/api/analysis/scale", json=dict(ref, per_pixel=0.05, unit="mm"))

            table = client.post("/api/analysis/traits", json=ref).json()
            csv_header = (
                client.get("/api/analysis/results", params=ref)
                .content.decode("utf-8-sig")
                .splitlines()[0]
            )

        self.assertEqual(table["headers"], csv_header.split(","))
        self.assertIn("Area (mm\u00b2)", table["headers"])


class ChinesePromptTestCase(unittest.TestCase):
    """Chinese semantic-detection prompts, translated in one shared place.

    Grounding DINO only reads English, so before this a Chinese prompt either hit
    one of a dozen exact aliases or silently detected nothing sensible.
    """

    @staticmethod
    def _normalized(prompt):
        from pheno.TextDetector import TextDetector

        return TextDetector.normalize_prompt(prompt)

    def test_common_targets_translate(self):
        cases = {
            "豆荚": "bean pod.",
            "种子": "bean seed.",
            "叶片": "bean leaf.",
            "花": "bean flower.",
            "尺子": "ruler.",
            "钢尺": "steel ruler.",
            "色卡": "color checker.",
            "标签": "label.",
            "托盘": "tray.",
        }
        for prompt, expected in cases.items():
            self.assertEqual(self._normalized(prompt), expected, prompt)

    def test_modifiers_compose_with_organs(self):
        """A flat alias table could not do this; longest-match scanning can."""
        cases = {
            "绿色豆荚": "green bean pod.",
            "青荚": "green bean pod.",
            "黄色种子": "yellow bean seed.",
            "深绿色叶片": "dark green bean leaf.",
            "破损的种子": "damaged bean seed.",
            "扁平豆荚": "flat bean pod.",
        }
        for prompt, expected in cases.items():
            self.assertEqual(self._normalized(prompt), expected, prompt)

    def test_crop_name_replaces_the_default_bean(self):
        """蚕豆荚 must not become "broad bean bean pod"."""
        self.assertEqual(self._normalized("蚕豆荚"), "broad bean pod.")
        self.assertEqual(self._normalized("大豆种子"), "soybean seed.")
        self.assertEqual(self._normalized("扁豆种子"), "hyacinth bean seed.")
        self.assertEqual(self._normalized("玉米粒"), "maize kernel.")

    def test_cereal_organs_do_not_get_a_bean(self):
        for prompt, expected in (("穗", "rice spike."), ("稻穗", "rice spike."),
                                 ("麦穗", "wheat spike."), ("籽粒", "rice grain.")):
            self.assertEqual(self._normalized(prompt), expected, prompt)

    def test_conjunctions_and_noise_words(self):
        self.assertEqual(self._normalized("豆荚和种子"), "bean pod. bean seed.")
        self.assertEqual(self._normalized("豆荚、叶片"), "bean pod. bean leaf.")
        # Politeness, quantifiers and framing words carry no visual meaning.
        self.assertEqual(self._normalized("请帮我检测所有的豆荚"), "bean pod.")
        self.assertEqual(self._normalized("三个成熟的豆荚"), "ripe bean pod.")

    def test_english_and_mixed_input_still_work(self):
        self.assertEqual(self._normalized("pod"), "bean pod.")
        self.assertEqual(self._normalized("rice spike"), "rice spike.")
        self.assertEqual(self._normalized("green 豆荚"), "green bean pod.")

    def test_unrecognisable_chinese_is_refused_in_chinese(self):
        from pheno.TextDetector import TextDetector

        for prompt in ("这是什么", "阿巴阿巴"):
            with self.assertRaises(ValueError) as caught:
                TextDetector.normalize_prompt(prompt)
            # The message has to be readable by the person who typed Chinese.
            self.assertIn("中文", str(caught.exception))

    def test_alias_notes_explain_the_rewrite(self):
        from pheno.TextDetector import TextDetector

        notes = dict(TextDetector.alias_notes("绿色豆荚"))

        self.assertEqual(notes.get("绿色豆荚"), "green bean pod")

    def test_no_chinese_left_in_the_alias_table(self):
        """One code path for Chinese, not two: a leftover entry here would shadow
        the vocabulary and reintroduce 叶片 -> "leaf" while 叶 -> "bean leaf"."""
        from pheno.TextDetector import TextDetector

        chinese = [
            key for key in TextDetector.PROMPT_ALIASES
            if any("\u4e00" <= char <= "\u9fff" for char in key)
        ]
        self.assertEqual(chinese, [])

    def test_desktop_presets_follow_the_interface_language(self):
        from pheno.i18n import set_language
        from pheno.widgets import TextPromptDialog

        try:
            set_language("zh_CN")
            self.assertEqual(TextPromptDialog.base_targets()[0], "豆荚")
            set_language("en")
            self.assertEqual(TextPromptDialog.base_targets()[0], "pod")
        finally:
            set_language("zh_CN")

    def test_web_prompt_hint_mentions_chinese(self):
        script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

        self.assertIn("豆荚、绿色豆荚", script)
        self.assertIn('state.lang === "zh" ? "豆荚" : "pod"', script)


class PublicServiceLimitsTestCase(unittest.TestCase):
    """What keeps an open, single-GPU service from being trivially overrun."""

    def test_queue_admission_refuses_instead_of_piling_up(self):
        """A full queue must fail fast: the worker is single-threaded, so an
        unbounded backlog would make every visitor wait for the proxy timeout."""
        import threading

        from web.engine import MAX_QUEUED_JOBS, EngineBusy, SamEngine

        engine = SamEngine()
        release = threading.Event()
        try:
            # Occupy the worker, then fill the queue to the limit.
            engine._submit(release.wait)
            for _ in range(MAX_QUEUED_JOBS - 1):
                engine._submit(lambda: None)

            self.assertEqual(engine.queue_depth(), MAX_QUEUED_JOBS)
            with self.assertRaises(EngineBusy):
                engine._submit(lambda: None)

            # The probe used by load() must never be refused, or the model could
            # fail to load merely because visitors are busy.
            engine._submit(lambda: None, admission=False)
        finally:
            release.set()
            engine.shutdown()

    def test_queue_depth_is_reported(self):
        from web.engine import MAX_QUEUED_JOBS, SamEngine

        engine = SamEngine()
        try:
            queue = engine.status()["queue"]
        finally:
            engine.shutdown()

        self.assertEqual(queue["depth"], 0)
        self.assertEqual(queue["limit"], MAX_QUEUED_JOBS)
        self.assertIn("peak", queue)

    def test_image_cache_is_bounded_by_bytes_not_by_count(self):
        """The cap used to be a count shared with the per-session limit, so the
        21st visitor evicted the first one's photo. Bytes are what matter."""
        from web.app import MAX_CACHED_IMAGE_BYTES, MAX_SESSION_IMAGES

        self.assertGreater(MAX_CACHED_IMAGE_BYTES, 512 * 1024 * 1024)
        # A single upload must never be able to evict itself.
        self.assertGreater(MAX_CACHED_IMAGE_BYTES, 25 * 1024 * 1024)
        self.assertEqual(MAX_SESSION_IMAGES, 20)

    def test_more_visitors_can_stay_warm_than_a_handful(self):
        from web.engine import MAX_CACHED_EMBEDDINGS

        # Each vit_h embedding is ~4 MB, so a small number only causes visitors to
        # evict each other and pay the encode again.
        self.assertGreaterEqual(MAX_CACHED_EMBEDDINGS, 16)

    def test_busy_and_throttled_have_their_own_messages(self):
        script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

        self.assertIn('response.status === 503', script)
        self.assertIn('response.status === 429', script)
        self.assertIn('xhr.status === 503', script)
        self.assertTrue(script.count('"error.busy"') >= 3)


class HelpDocumentTestCase(unittest.TestCase):
    """The guide is one source for both front ends, so both must stay in step."""

    def setUp(self):
        self.client = TestClient(app)

    def test_route_serves_both_languages(self):
        for requested, expected in (("zh", "zh"), ("en", "en"), ("zh-CN", "zh"), ("fr", "en")):
            payload = self.client.get("/api/help", params={"lang": requested}).json()

            self.assertEqual(payload["language"], expected, requested)
            self.assertEqual(payload["platform"], "web")
            self.assertTrue(payload["title"])
            self.assertTrue(payload["intro"])
            self.assertGreaterEqual(len(payload["sections"]), 5)
            for section in payload["sections"]:
                self.assertTrue(section["heading"])
                self.assertTrue(section["items"])

    def test_both_languages_have_the_same_shape(self):
        chinese = self.client.get("/api/help", params={"lang": "zh"}).json()
        english = self.client.get("/api/help", params={"lang": "en"}).json()

        self.assertEqual(
            [len(section["items"]) for section in chinese["sections"]],
            [len(section["items"]) for section in english["sections"]],
        )

    def test_platform_specific_entries_are_filtered(self):
        from pheno.help_content import DESKTOP, WEB, help_document

        web = help_document("zh", WEB)
        desktop = help_document("zh", DESKTOP)
        web_text = " ".join(item for s in web["sections"] for item in s["items"])
        desktop_text = " ".join(item for s in desktop["sections"] for item in s["items"])

        # Web-only affordances must not appear in the desktop guide, and vice versa.
        self.assertIn("示例图片", web_text)
        self.assertNotIn("示例图片", desktop_text)
        self.assertIn("编辑掩膜", desktop_text)
        self.assertNotIn("编辑掩膜", web_text)
        # Advice that applies to both stays in both.
        for shared in ("尺度检测", "色卡矫正", "MeanB"):
            self.assertIn(shared, web_text, shared)
            self.assertIn(shared, desktop_text, shared)

    def test_desktop_html_is_self_contained(self):
        from pheno.help_content import DESKTOP, help_html

        html = help_html("en", DESKTOP)

        self.assertTrue(html.startswith("<html>"))
        self.assertIn("</body></html>", html)
        self.assertIn("<h3>", html)
        # No external resources: the dialog must work offline.
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)

    def test_page_opens_the_guide_on_a_first_visit(self):
        page = self.client.get("/").text
        script = self.client.get("/static/app.js").text

        for marker in ('id="help-overlay"', 'id="help-open"', 'id="help-body"'):
            self.assertIn(marker, page, marker)
        self.assertTrue("if (!helpAlreadySeen()) openHelp();" in script)
        self.assertTrue('window.localStorage.setItem("phenome.helpSeen"' in script)
        # Fetched, not bundled, so it follows the language toggle.
        self.assertTrue("/api/help?lang=" in script)



class TranslationFallbackTestCase(unittest.TestCase):
    """The model-backed fallback for Chinese terms the dictionaries do not know.

    The dictionaries stay the fast, deterministic path; the model is only asked
    about terms they cannot fully translate, and its answer is cached so a word is
    only ever translated once.
    """

    def setUp(self):
        from pheno import prompt_zh

        self.prompt_zh = prompt_zh
        self.translate = web_translate
        self.translate.forget_all()
        self.asked: list[str] = []

    def tearDown(self):
        self.prompt_zh.set_fallback_translator(None)
        self.translate.forget_all()

    def install(self, answer):
        """Install a fallback that records what it was asked."""

        def fallback(chinese):
            self.asked.append(chinese)
            return answer(chinese) if callable(answer) else answer

        self.prompt_zh.set_fallback_translator(fallback)

    def test_a_known_term_never_reaches_the_model(self):
        self.install("should not be used")

        self.assertEqual(self.prompt_zh.translate("豆荚"), "bean pod")
        self.assertEqual(self.asked, [], "the dictionary must answer on its own")

    def test_an_unknown_term_is_handed_over_whole(self):
        # 紫藤荚 is only partly known: the dictionary would answer "purple bean pod",
        # which names a different thing. The fallback gets the whole fragment.
        self.install("wisteria pod")

        self.assertEqual(self.prompt_zh.translate("紫藤荚"), "wisteria pod")
        self.assertEqual(self.asked, ["紫藤荚"])

    def test_a_failing_fallback_leaves_the_old_behaviour(self):
        self.install(lambda chinese: None)

        # Nothing usable came back, so the partial dictionary result still applies.
        self.assertEqual(self.prompt_zh.translate("紫藤荚"), "purple bean pod")

    def test_an_exception_in_the_fallback_is_swallowed(self):
        def boom(chinese):
            raise RuntimeError("provider is down")

        self.prompt_zh.set_fallback_translator(boom)

        # A detection must not fail because a translation service did.
        self.assertEqual(self.prompt_zh.translate("紫藤荚"), "purple bean pod")

    def test_no_fallback_means_the_dictionary_alone(self):
        self.assertEqual(self.prompt_zh.translate("豆荚"), "bean pod")
        self.assertEqual(self.prompt_zh.translate("皂角刺"), "")

    # -- qualifiers ------------------------------------------------------------

    def test_qualifiers_are_stripped_from_an_answer(self):
        # The lesson from `red lentil`: measured on one photograph of lentils,
        # `lentil` found 20 seeds while `red lentil` found 3.
        self.assertEqual(self.translate.strip_qualifiers("red lentil"), "lentil")
        self.assertEqual(self.translate.strip_qualifiers("small green pod"), "pod")
        self.assertEqual(self.translate.strip_qualifiers("dried bean seed"), "bean seed")
        self.assertEqual(self.translate.strip_qualifiers("lentil"), "lentil")

    def test_a_phrase_of_only_qualifiers_is_left_alone(self):
        self.assertEqual(self.translate.strip_qualifiers("small"), "small")

    # -- answer cleaning -------------------------------------------------------

    def test_answers_are_cleaned_or_rejected(self):
        cases = {
            '"lentil"': "lentil",
            "wisteria pod": "wisteria pod",
            "  Bean Pod  ": "bean pod",
            "red lentil": "lentil",
            "UNKNOWN": None,
            "unknown": None,
            "": None,
            "紫藤荚": None,  # a Chinese answer is no use to the detector
            "This is a wisteria pod.": None,  # a sentence, not a term
            "- wisteria pod": "wisteria pod",
        }
        for raw, expected in cases.items():
            self.assertEqual(self.translate.clean_answer(raw), expected, raw)

    def test_a_botanical_category_is_used_but_flagged(self):
        # Measured: `lentil` found 20 lentils where `legume` found 3 and `pulse` 1.
        # The answer is still used -- appending a guessed organ would be worse, the
        # term may be a pod -- but it must be visible in the log rather than passed
        # off as a good one.
        with self.assertLogs("phenome.translate", level="INFO") as captured:
            answer = self.translate.clean_answer("legume")

        self.assertEqual(answer, "legume")
        self.assertTrue(any("no visible object" in line for line in captured.output))

    def test_a_qualified_visible_noun_is_not_flagged(self):
        records = []

        class Collector(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        logger = logging.getLogger("phenome.translate")
        collector = Collector()
        logger.addHandler(collector)
        try:
            self.assertEqual(self.translate.clean_answer("wisteria pod"), "wisteria pod")
        finally:
            logger.removeHandler(collector)

        self.assertFalse(any("no visible object" in line for line in records))

    # -- caching ---------------------------------------------------------------

    def test_a_translation_is_asked_for_once_and_then_remembered(self):
        calls = []

        def answer(chinese):
            calls.append(chinese)
            return "wisteria pod"

        with patch.object(self.translate, "_ask_model", answer), patch.dict(
            os.environ, {"PHENOME_AGENT_API_KEY": "test-key", "PHENOME_TRANSLATE_FALLBACK": "on"}
        ):
            first = self.translate.translate("紫藤荚")
            second = self.translate.translate("紫藤荚")

        self.assertEqual(first, "wisteria pod")
        self.assertEqual(second, "wisteria pod")
        # One model call, ever: that is what keeps later runs reproducible.
        self.assertEqual(calls, ["紫藤荚"])

    def test_the_cache_is_persisted_next_to_the_dictionaries(self):
        with patch.object(self.translate, "_ask_model", lambda chinese: "wisteria pod"), patch.dict(
            os.environ, {"PHENOME_AGENT_API_KEY": "test-key", "PHENOME_TRANSLATE_FALLBACK": "on"}
        ):
            self.translate.translate("紫藤荚")

        self.assertTrue(self.translate.CACHE_PATH.is_file())
        written = json.loads(self.translate.CACHE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(written.get("紫藤荚"), "wisteria pod")

    def test_the_fallback_can_be_switched_off(self):
        with patch.dict(os.environ, {"PHENOME_AGENT_API_KEY": "test-key", "PHENOME_TRANSLATE_FALLBACK": "off"}):
            self.assertFalse(self.translate.enabled())

    def test_no_key_means_no_fallback(self):
        with patch.object(self.translate, "_credentials", return_value=("https://x.invalid", "", "m")):
            self.assertFalse(self.translate.enabled())


if __name__ == "__main__":
    unittest.main()
