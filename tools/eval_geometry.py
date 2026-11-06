"""How much perspective is actually in these photographs?

No chessboard needed, and no camera intrinsics either: the scene already contains
two planar references, and each gives an independent read on how far the camera is
from straight overhead.

1. Colour chart -- a rectangle photographed from an angle has unequal opposite
   sides. The left/right and top/bottom length ratios of the detected quad are
   pure perspective indicators, independent of the chart's true aspect ratio.
2. Steel rule -- if the sensor plane is not parallel to the table, the tick pitch
   drifts along the rule. Comparing the pitch over the first third against the
   last third measures that drift directly, in the unit that matters: how much
   one centimetre changes across the frame.

    python -m tools.eval_geometry --limit 30
"""

from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DIRS = ["input"]
MODEL = Path(__file__).resolve().parent.parent / "models" / "rulernet.onnx"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def sample_images(directory: Path, limit: int, seed: int) -> list[Path]:
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(files) <= limit:
        return files
    return random.Random(seed).sample(files, limit)


def chart_skew(image_bgr):
    """(opposite-side ratio, corner-angle error in degrees) of the colour chart."""
    from pheno.color import CalibrationError, detect_charts

    try:
        candidates = detect_charts(image_bgr)
    except CalibrationError:
        return None
    box = np.asarray(candidates[0]["checker"].getBox(), dtype=np.float64)
    box = box * candidates[0].get("coord_scale", 1.0)
    if box.shape != (4, 2):
        return None

    top_left, top_right, bottom_right, bottom_left = box
    top = np.linalg.norm(top_right - top_left)
    bottom = np.linalg.norm(bottom_right - bottom_left)
    left = np.linalg.norm(bottom_left - top_left)
    right = np.linalg.norm(bottom_right - top_right)
    if min(top, bottom, left, right) <= 0:
        return None

    # 1.0 means the opposite sides are equal, i.e. no perspective on this plane.
    horizontal = max(top, bottom) / min(top, bottom)
    vertical = max(left, right) / min(left, right)

    angles = []
    for index in range(4):
        previous = box[(index - 1) % 4] - box[index]
        following = box[(index + 1) % 4] - box[index]
        cosine = np.dot(previous, following) / (
            np.linalg.norm(previous) * np.linalg.norm(following) + 1e-9
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return {
        "side_ratio": max(horizontal, vertical),
        "angle_error": float(max(abs(angle - 90.0) for angle in angles)),
    }


def rule_pitch_drift(detector, image_bgr):
    """Relative change of the tick pitch between the two ends of the rule.

    Two independent readings, because they can disagree:

    * ``model_drift`` compares the spacing of the reconstructed tick points. The
      network extrapolates them as a geometric series (``ratio ** n``), so this
      number partly reflects the model's own perspective parameter.
    * ``image_drift`` measures the pitch by FFT on the first and last third of the
      intensity profile. Nothing is extrapolated, so this is the honest one.
    """
    points, spacing = detector.get_tickpoints_and_scale(image_bgr)
    if points is None or len(points) < 9:
        return None
    points = np.asarray(points, dtype=np.float64)
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    third = max(2, len(diffs) // 3)
    head = float(np.median(diffs[:third]))
    tail = float(np.median(diffs[-third:]))
    if head <= 0 or tail <= 0:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    start, end = points[0], points[-1]
    first_third = start + (end - start) / 3.0
    last_third = start + 2.0 * (end - start) / 3.0
    measured = []
    for segment_start, segment_end in ((start, first_third), (last_third, end)):
        profile, step = detector.sample_profile(gray, segment_start, segment_end)
        period, strength = detector.dominant_period(profile, step)
        measured.append(period if period and strength >= 8.0 else None)

    image_drift = None
    if all(measured):
        image_drift = abs(measured[1] - measured[0]) / ((measured[0] + measured[1]) / 2.0)

    return {
        "model_drift": abs(tail - head) / ((tail + head) / 2.0),
        "image_drift": image_drift,
        "span_cm": float(np.hypot(*(end - start)) / spacing),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dir", action="append", dest="dirs")
    args = parser.parse_args()

    from pheno.RulerNet import RulerNet

    detector = RulerNet(str(MODEL))
    side_ratios, angle_errors = [], []
    model_drifts, image_drifts = [], []

    for directory in [Path(d) for d in (args.dirs or DEFAULT_DIRS)]:
        if not directory.is_dir():
            continue
        print(f"\n== {directory.name} ==")
        for path in sample_images(directory, args.limit, args.seed):
            image = read_image(path)
            if image is None:
                continue
            chart = chart_skew(image)
            rule = rule_pitch_drift(detector, image)
            if chart:
                side_ratios.append(chart["side_ratio"])
                angle_errors.append(chart["angle_error"])
            if rule:
                model_drifts.append(rule["model_drift"])
                if rule["image_drift"] is not None:
                    image_drifts.append(rule["image_drift"])
            image_drift = rule["image_drift"] if rule else None
            print(
                f"  {path.name:<26}"
                f" chart {('%.4f' % chart['side_ratio']) if chart else '  -   '}"
                f" {('%4.2f deg' % chart['angle_error']) if chart else '  -     '}"
                f" | rule drift: model {('%5.2f%%' % (100 * rule['model_drift'])) if rule else '  -  '}"
                f"  image {('%5.2f%%' % (100 * image_drift)) if image_drift is not None else '  -  '}"
                f"  over {('%4.1f cm' % rule['span_cm']) if rule else '  -  '}"
            )

    def report(name, values, unit=""):
        if not values:
            print(f"  {name:<22} no data")
            return
        ordered = sorted(values)
        p90 = ordered[int(0.9 * (len(ordered) - 1))]
        print(
            f"  {name:<22} median {statistics.median(ordered):7.3f}{unit}"
            f"   p90 {p90:7.3f}{unit}   max {ordered[-1]:7.3f}{unit}   (n={len(ordered)})"
        )

    print("\n==== how much perspective is there ====")
    report("chart side ratio", side_ratios)
    report("chart corner error", angle_errors, " deg")
    report("rule drift (model)", [100 * value for value in model_drifts], " %")
    report("rule drift (image FFT)", [100 * value for value in image_drifts], " %")


if __name__ == "__main__":
    main()
