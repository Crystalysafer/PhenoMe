"""Can the lens be calibrated from the photographs themselves, with no chessboard?

The steel rule is a calibration target: its ticks are equally spaced and collinear
by construction. Radial distortion breaks both properties, so a single parameter
(k1, optionally k2) can be fitted by asking for the spacing to become uniform
again. This is the classic plumb-line calibration, driven by the 40-odd tick
points RulerNet already returns per image.

If one k1 explains every photograph from the same camera, the distortion is a
property of the lens and can be removed once and for all.

    python -m tools.eval_distortion --limit 8
"""

from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DIR = "input"
MODEL = Path(__file__).resolve().parent.parent / "models" / "rulernet.onnx"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def undistort_points(points, shape, k1, k2=0.0):
    """Map observed pixels to ideal pixels for a given radial distortion."""
    height, width = shape[:2]
    focal = max(width, height) / 2.0
    camera = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    )
    distortion = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    source = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(source, camera, distortion, P=camera)
    return ideal.reshape(-1, 2)


def spacing_stats(points):
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    median = float(np.median(diffs))
    third = max(2, len(diffs) // 3)
    head = float(np.median(diffs[:third]))
    tail = float(np.median(diffs[-third:]))
    return {
        "cv": float(np.std(diffs) / median) if median else float("inf"),
        "drift": abs(tail - head) / ((tail + head) / 2.0) if head and tail else float("inf"),
        "median": median,
    }


def fit_k1(points, shape):
    """Grid search then refine the single radial coefficient."""

    def cost(k1):
        return spacing_stats(undistort_points(points, shape, k1))["cv"]

    grid = np.linspace(-0.35, 0.35, 71)
    best = min(grid, key=cost)
    step = grid[1] - grid[0]
    for _ in range(6):
        step /= 4.0
        local = [best - step, best, best + step]
        best = min(local, key=cost)
    return float(best), cost(best)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dir", default=DEFAULT_DIR)
    args = parser.parse_args()

    from pheno.RulerNet import RulerNet

    detector = RulerNet(str(MODEL))
    directory = Path(args.dir)
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(files) > args.limit:
        files = random.Random(args.seed).sample(files, args.limit)

    estimates = []
    print(f"== {directory.name} ==")
    for path in files:
        image = read_image(path)
        if image is None:
            continue
        points, spacing = detector.get_tickpoints_and_scale(image)
        if points is None or len(points) < 12:
            print(f"  {path.name:<24} no usable ruler")
            continue
        points = np.asarray(points, dtype=np.float64)
        before = spacing_stats(points)
        k1, after_cv = fit_k1(points, image.shape)
        after = spacing_stats(undistort_points(points, image.shape, k1))
        estimates.append(k1)
        print(
            f"  {path.name:<24} ticks {len(points):3d}"
            f" | before: cv {100 * before['cv']:5.2f}% drift {100 * before['drift']:5.2f}%"
            f" | k1 {k1:+.4f} -> cv {100 * after['cv']:5.2f}% drift {100 * after['drift']:5.2f}%"
        )

    if estimates:
        print("\n==== k1 across images ====")
        print(
            f"  median {statistics.median(estimates):+.4f}"
            f"   mean {statistics.mean(estimates):+.4f}"
            f"   stdev {statistics.stdev(estimates) if len(estimates) > 1 else 0:.4f}"
            f"   n={len(estimates)}"
        )
        print(
            "  A consistent k1 means the drift is the lens, not the camera angle,"
            " and one undistort fixes every photograph."
        )


if __name__ == "__main__":
    main()
