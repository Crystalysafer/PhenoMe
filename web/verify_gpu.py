"""Real-GPU end-to-end check of the web segmentation pipeline.

Run it from the project root with the project interpreter:

    python -m web.verify_gpu

Unlike ``web/test_app.py`` this loads the actual 2.4 GB ``vit_h`` checkpoint and
drives every route through ``TestClient``, so it needs a free GPU and takes about
a minute. It is a manual verification script, not part of the unittest suite.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from pheno import annotation
from web.app import app


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_image() -> Path:
    """First image under input/ keeps the zero-argument run working locally."""
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        hits = sorted((PROJECT_ROOT / "input").glob(pattern))
        if hits:
            return hits[0]
    return PROJECT_ROOT / "input" / "photo.jpg"


# Pass an image path as the first argument; otherwise the first file in
# input/ is used. POD_POINTS below assume a ~4600x3500 photo with three pods.
IMAGE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_image()

# Points picked from the 4624x3472 original. The pod points are the distance
# transform peaks of the three green blobs, so they sit well inside each pod;
# a point that only grazes a pod makes SAM return the whole dark background.
POD_POINTS = [(2855, 1612), (3585, 1763), (4298, 1704)]
POD_AREA_RANGE = (300_000, 600_000)
TAG_POINT = (3639, 2712)
COLOR_CARD_POINT = (600, 2900)
DARK_BACKGROUND_POINT = (200, 150)


def log(message: str) -> None:
    print(message, flush=True)


def wait_for(client, check, label, attempts=180, delay=1.0):
    started = time.time()
    for _ in range(attempts):
        payload = check()
        if payload is not None:
            return payload, time.time() - started
        time.sleep(delay)
    raise SystemExit("timeout while waiting for %s" % label)


def main() -> int:
    if not IMAGE_PATH.is_file():
        raise SystemExit("missing test image: %s" % IMAGE_PATH)

    client = TestClient(app)

    log("== model ==")
    response = client.post("/api/model/load")
    log("POST /api/model/load -> %s %s" % (response.status_code, response.json()["state"]))

    def model_ready():
        payload = client.get("/api/model").json()
        return payload if payload["state"] in ("ready", "error") else None

    model, seconds = wait_for(client, model_ready, "model load")
    if model["state"] != "ready":
        raise SystemExit("model failed to load: %s" % model["error"])
    log("model ready in %.1fs on %s (reported load_seconds=%s)" % (seconds, model["device"], model["load_seconds"]))
    log("gpu: %s" % model["gpu"])

    log("== session and upload ==")
    session_id = client.post("/api/sessions", json={"source": "workspace"}).json()["id"]
    data = IMAGE_PATH.read_bytes()
    upload = client.post(
        "/api/images?session_id=%s" % session_id,
        files={"file": (IMAGE_PATH.name, data, "image/jpeg")},
    ).json()
    image_id = upload["id"]
    log("uploaded %s %dx%d (%.1f KB)" % (upload["filename"], upload["width"], upload["height"], len(data) / 1024))

    ref = {"session_id": session_id, "image_id": image_id}

    log("== embedding ==")
    started = time.time()
    first = client.post("/api/analysis/embedding", json=ref).json()
    log("POST /api/analysis/embedding -> %s" % first["embedding"]["state"])

    def embedding_ready():
        payload = client.get("/api/analysis", params=ref).json()
        return payload if payload["embedding"]["state"] in ("ready", "error") else None

    analysis, _ = wait_for(client, embedding_ready, "embedding")
    if analysis["embedding"]["state"] != "ready":
        raise SystemExit("embedding failed: %s" % analysis["embedding"]["error"])
    log(
        "embedding ready after %.1fs (engine reported %.2fs, scale %.4f)"
        % (
            time.time() - started,
            analysis["embedding"]["seconds"],
            analysis["embedding"]["scale"],
        )
    )

    cached_started = time.time()
    client.post("/api/analysis/embedding", json=ref)
    log("cached embedding request took %.3fs" % (time.time() - cached_started))

    log("== click segmentation ==")
    x, y = POD_POINTS[0]
    started = time.time()
    preview = client.post(
        "/api/analysis/click", json=dict(ref, points=[{"x": x, "y": y, "label": 1}], commit=False)
    ).json()
    if not preview["ok"]:
        raise SystemExit("preview failed: %s" % preview.get("message"))
    log(
        "one positive point: %d px, polygon %d points, %.2fs"
        % (
            preview["preview"]["area_pixels"],
            len(preview["preview"]["polygon"]),
            time.time() - started,
        )
    )

    started = time.time()
    committed = client.post(
        "/api/analysis/click", json=dict(ref, points=[{"x": x, "y": y, "label": 1}], commit=True)
    ).json()
    first_object = committed["object"]
    log(
        "commit reused the preview mask in %.3fs -> %s (%d px)"
        % (time.time() - started, first_object["label"], first_object["area_pixels"])
    )
    if first_object["area_pixels"] != preview["preview"]["area_pixels"]:
        raise SystemExit("commit changed the mask area")
    if not POD_AREA_RANGE[0] < first_object["area_pixels"] < POD_AREA_RANGE[1]:
        raise SystemExit("pod 1 mask area %d is out of range" % first_object["area_pixels"])

    for index, (px, py) in enumerate(POD_POINTS[1:], start=2):
        payload = client.post(
            "/api/analysis/click",
            json=dict(ref, points=[{"x": px, "y": py, "label": 1}], commit=True, name_prefix="pod"),
        ).json()
        if not payload["ok"]:
            raise SystemExit("pod %d failed: %s" % (index, payload.get("message")))
        obj = payload["object"]
        log("pod click %d -> %s (%d px)" % (index, obj["label"], obj["area_pixels"]))
        if not POD_AREA_RANGE[0] < obj["area_pixels"] < POD_AREA_RANGE[1]:
            raise SystemExit("pod %d mask area %d is out of range" % (index, obj["area_pixels"]))

    log("== negative point ==")
    tx, ty = TAG_POINT
    positive_only = client.post(
        "/api/analysis/click", json=dict(ref, points=[{"x": tx, "y": ty, "label": 1}])
    ).json()
    with_negative = client.post(
        "/api/analysis/click",
        json=dict(
            ref,
            points=[
                {"x": tx, "y": ty, "label": 1},
                {"x": tx, "y": ty - 260, "label": 0},
            ],
        ),
    ).json()
    log(
        "tag: %d px with one point, %d px after a negative point"
        % (positive_only["preview"]["area_pixels"], with_negative["preview"]["area_pixels"])
    )

    for label, (bx, by) in (
        ("color card", COLOR_CARD_POINT),
        ("dark background", DARK_BACKGROUND_POINT),
    ):
        payload = client.post(
            "/api/analysis/click",
            json=dict(ref, points=[{"x": bx, "y": by, "label": 1}]),
        ).json()
        log(
            "%s click ok=%s area=%s"
            % (
                label,
                payload["ok"],
                payload.get("preview", {}).get("area_pixels", payload.get("message")),
            )
        )

    log("== objects ==")
    state = client.get("/api/analysis", params=ref).json()
    log("objects: %s" % [obj["label"] for obj in state["objects"]])

    renamed = client.patch(
        "/api/analysis/objects/%d" % first_object["uid"], json=dict(ref, label="my pod")
    ).json()
    log("renamed uid %d -> %s" % (first_object["uid"], renamed["object"]["label"]))

    duplicate = client.patch(
        "/api/analysis/objects/%d" % state["objects"][1]["uid"], json=dict(ref, label="my pod")
    )
    log("duplicate rename -> %d" % duplicate.status_code)

    removed = client.delete(
        "/api/analysis/objects/%d" % state["objects"][1]["uid"], params=ref
    ).json()
    log("after delete: %s" % [obj["label"] for obj in removed["objects"]])

    again = client.post(
        "/api/analysis/click",
        json=dict(ref, points=[{"x": POD_POINTS[1][0], "y": POD_POINTS[1][1], "label": 1}], commit=True, name_prefix="pod"),
    ).json()
    log("next pod name after deleting pod01: %s" % again["object"]["label"])

    log("== scale and traits ==")
    scale = client.post("/api/analysis/scale", json=dict(ref, per_pixel=0.0523, unit="mm")).json()
    log("scale -> %s" % scale["scale"])

    started = time.time()
    traits = client.post("/api/analysis/traits", json=ref).json()
    log("traits for %d objects in %.2fs" % (len(traits["rows"]), time.time() - started))
    area_index = traits["headers"].index("Area")
    perimeter_index = traits["headers"].index("Perimeter")
    for row in traits["rows"]:
        log(
            "  %-10s area=%.2f mm^2 perimeter=%.2f mm"
            % (row[0], row[area_index], row[perimeter_index])
        )

    log("== annotation ==")
    started = time.time()
    response = client.get("/api/analysis/annotation", params=ref)
    document = json.loads(response.content.decode("utf-8"))
    log(
        "sidecar %.1f KB in %.2fs, disposition=%s"
        % (len(response.content) / 1024, time.time() - started, response.headers.get("content-disposition"))
    )
    log(
        "format=%s version=%s md5=%s scale=%s objects=%d"
        % (
            document["format"],
            document["version"],
            document["image"]["md5"],
            document["scale"],
            len(document["objects"]),
        )
    )

    expected_md5 = hashlib.md5(data).hexdigest()
    if document["image"]["md5"] != expected_md5:
        raise SystemExit("sidecar md5 does not match the uploaded bytes")
    if document["image"]["width"] != upload["width"]:
        raise SystemExit("sidecar image size does not match the upload")

    for obj in document["objects"]:
        mask = annotation.decode_mask(obj["mask"])
        polygon = np.asarray(obj["polygon"], dtype=np.float32)
        log(
            "  %-10s mask=%d px bbox=%s polygon=%d points mask_field=%.1f KB"
            % (
                obj["label"],
                int(mask.sum()),
                obj["mask"]["bbox"],
                len(polygon),
                len(json.dumps(obj["mask"])) / 1024,
            )
        )
        if mask.shape != (upload["height"], upload["width"]):
            raise SystemExit("decoded mask has the wrong shape")

    log("== per-session embedding isolation ==")
    other_session = client.post("/api/sessions", json={"source": "capture"}).json()["id"]
    other_upload = client.post(
        "/api/images?session_id=%s" % other_session,
        files={"file": (IMAGE_PATH.name, data, "image/jpeg")},
    ).json()
    other_ref = {"session_id": other_session, "image_id": other_upload["id"]}
    client.post("/api/analysis/embedding", json=other_ref)

    def other_ready():
        payload = client.get("/api/analysis", params=other_ref).json()
        return payload if payload["embedding"]["state"] in ("ready", "error") else None

    other_state, seconds = wait_for(client, other_ready, "second embedding")
    log("second session embedding ready in %.1fs" % seconds)

    point = {"x": POD_POINTS[2][0], "y": POD_POINTS[2][1], "label": 1}
    area_a = client.post("/api/analysis/click", json=dict(ref, points=[point])).json()["preview"][
        "area_pixels"
    ]
    area_b = client.post("/api/analysis/click", json=dict(other_ref, points=[point])).json()[
        "preview"
    ]["area_pixels"]
    area_a_again = client.post("/api/analysis/click", json=dict(ref, points=[point])).json()[
        "preview"
    ]["area_pixels"]
    log("same point: session A=%d, session B=%d, session A again=%d" % (area_a, area_b, area_a_again))
    if area_a != area_a_again:
        raise SystemExit("switching embeddings changed session A's result")
    if area_a != area_b:
        raise SystemExit("the two sessions disagree on the same image")

    cross = client.get("/api/analysis", params={"session_id": other_session, "image_id": image_id})
    log("cross-session image access -> %d" % cross.status_code)

    log("== final model status ==")
    log(json.dumps(client.get("/api/model").json(), ensure_ascii=False))

    log("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
