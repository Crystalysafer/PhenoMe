"""Lazy wrapper around the existing Qt-free RulerNet implementation."""

from __future__ import annotations

from pathlib import Path

# Above this the scale is noticeably position-dependent: the camera is not square
# to the table, or the rule is not in the sample plane. Measured median on the DUS
# sets is 4.5% over a 40 cm rule, so this is a "look at your setup" hint rather
# than an error.
UNIFORM_SCALE_WARNING = 0.03


class RulerDetectionError(RuntimeError):
    """The ruler model could not produce a usable scale."""


def detect_ruler(image_bgr, model_path: Path) -> tuple[list[list[float]], float, dict]:
    """Return ``(tickpoints, millimeters_per_pixel, uniformity)`` for one image."""
    if not model_path.is_file():
        raise RulerDetectionError("The ruler model was not found.")

    try:
        from pheno.RulerNet import RulerNet

        detector = RulerNet(str(model_path))
        tickpoints, pixels_per_cm = detector.get_tickpoints_and_scale(image_bgr)
        uniformity = (
            detector.measure_scale_uniformity(image_bgr, tickpoints)
            if tickpoints is not None
            else None
        )
    except Exception as exc:
        raise RulerDetectionError(f"Ruler detection failed: {exc}") from exc

    if tickpoints is None or len(tickpoints) < 2 or not pixels_per_cm:
        raise RulerDetectionError("No usable ruler scale was detected.")

    millimeters_per_pixel = 10.0 / float(pixels_per_cm)
    if millimeters_per_pixel <= 0:
        raise RulerDetectionError("The detected ruler scale is invalid.")
    points = [[round(float(x), 1), round(float(y), 1)] for x, y in tickpoints]

    report: dict = {"ticks": len(points)}
    if uniformity:
        drift = float(uniformity["drift"])
        report["drift_percent"] = round(100.0 * drift, 1)
        report["uneven"] = drift > UNIFORM_SCALE_WARNING
    return points, millimeters_per_pixel, report
