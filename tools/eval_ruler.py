"""Measure how often the ruler detector finds a usable scale.

Ground truth comes from the image, not from labels: a ruler is a periodic
structure, so an FFT of the intensity profile along the tick line gives the true
pitch. A detection counts as *correct* when the predicted centimetre spacing is
within a few percent of an integer relation to that measured pitch (millimetre
ticks give a factor of 10). Folders without a ruler are scored the other way
round -- every detection there is a false positive.

    python -m tools.eval_ruler --impl legacy --limit 20
    python -m tools.eval_ruler --impl new    --limit 20
    python -m tools.eval_ruler --impl both   --limit 20 --json out.json

Directories default to the local DUS sets; pass --dir/--nodir to evaluate others.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

# (directory, contains a ruler) -- pass --dir/--nodir for more folders
DEFAULT_DIRS = [("input", True)]
MODEL = Path(__file__).resolve().parent.parent / "models" / "rulernet.onnx"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PITCH_RATIOS = (10.0, 1.0, 2.0, 5.0)
PITCH_TOLERANCE = 0.06
MIN_PITCH_STRENGTH = 6.0


def read_image(path: Path):
    """cv2.imread cannot open non-ASCII paths on Windows; decode the bytes."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def sample_images(directory: Path, limit: int, seed: int) -> list[Path]:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(files) <= limit:
        return files
    return random.Random(seed).sample(files, limit)


def verdict(detector, gray, points, pix_cm):
    """'correct' / 'wrong' / 'unverified' for one detection."""
    profile, step = detector.sample_profile(gray, points[0], points[-1])
    period, strength = detector.dominant_period(profile, step)
    if not period or strength < MIN_PITCH_STRENGTH:
        return "unverified", period, strength, None
    error, _ratio = min((abs(pix_cm / (period * r) - 1.0), r) for r in PITCH_RATIOS)
    return ("correct" if error <= PITCH_TOLERANCE else "wrong"), period, strength, error


def run_one(detector, image_bgr, impl: str):
    started = time.perf_counter()
    try:
        if impl == "legacy":
            points, pix_cm = detector.get_tickpoints_and_scale_legacy(image_bgr)
        else:
            points, pix_cm = detector.get_tickpoints_and_scale(image_bgr)
        error = None
    except Exception as exc:  # noqa: BLE001 - counting these is the point
        points, pix_cm, error = None, None, f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started

    detected = points is not None and len(points) > 1 and bool(pix_cm)
    row = {
        "detected": bool(detected),
        "pix_cm": float(pix_cm) if detected else None,
        "ticks": int(len(points)) if points is not None else 0,
        "seconds": round(elapsed, 2),
        "error": error,
        "verdict": "none",
    }
    if detected:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        row["verdict"], period, strength, deviation = verdict(
            detector, gray, np.asarray(points, dtype=np.float64), float(pix_cm)
        )
        row["fft_period"] = round(period, 2) if period else None
        row["fft_strength"] = round(strength, 1)
        row["pitch_error"] = round(deviation, 4) if deviation is not None else None
    return row


def evaluate(impl: str, folders: list, limit: int, seed: int) -> dict:
    from pheno.RulerNet import RulerNet

    detector = RulerNet(str(MODEL))
    report: dict = {"impl": impl, "folders": []}

    for directory, has_ruler in folders:
        if not directory.is_dir():
            print(f"  (skipped, missing: {directory})")
            continue
        images = sample_images(directory, limit, seed)
        rows = []
        kind = "ruler" if has_ruler else "no ruler"
        print(f"\n== {impl} :: {directory.name} [{kind}] ({len(images)} images) ==")
        for path in images:
            image = read_image(path)
            if image is None:
                print(f"  {path.name:<28} unreadable")
                continue
            row = run_one(detector, image, impl)
            row["file"] = path.name
            rows.append(row)
            scale = f"{row['pix_cm']:.2f} px/cm" if row["detected"] else "-"
            extra = f"fft {row.get('fft_period')}px s{row.get('fft_strength')}" if row["detected"] else ""
            print(
                f"  {path.name:<28} {row['verdict']:<10} {scale:>14}"
                f" {row['seconds']:>5.2f}s {extra} {(row['error'] or '')[:46]}"
            )

        counts = {
            key: sum(1 for row in rows if row["verdict"] == key)
            for key in ("correct", "wrong", "unverified", "none")
        }
        report["folders"].append(
            {
                "directory": str(directory),
                "has_ruler": has_ruler,
                "images": len(rows),
                "counts": counts,
                "rows": rows,
            }
        )
        if rows and has_ruler:
            print(
                f"  -> correct {counts['correct']}/{len(rows)}"
                f" ({100 * counts['correct'] / len(rows):.0f}%),"
                f" wrong {counts['wrong']}, unverified {counts['unverified']},"
                f" missed {counts['none']}"
            )
        elif rows:
            false_positive = len(rows) - counts["none"]
            print(
                f"  -> false positives {false_positive}/{len(rows)}"
                f" ({100 * false_positive / len(rows):.0f}%)"
            )
    return report


def summarise(report: dict) -> str:
    ruler = [f for f in report["folders"] if f["has_ruler"]]
    plain = [f for f in report["folders"] if not f["has_ruler"]]
    total = sum(f["images"] for f in ruler)
    correct = sum(f["counts"]["correct"] for f in ruler)
    wrong = sum(f["counts"]["wrong"] for f in ruler)
    unverified = sum(f["counts"]["unverified"] for f in ruler)
    missed = sum(f["counts"]["none"] for f in ruler)
    fp_total = sum(f["images"] for f in plain)
    fp = sum(f["images"] - f["counts"]["none"] for f in plain)
    seconds = [row["seconds"] for f in report["folders"] for row in f["rows"]]
    return (
        f"{report['impl']:>7}: with ruler {correct}/{total} correct"
        f" ({100 * correct / max(total, 1):.0f}%),"
        f" wrong {wrong}, unverified {unverified}, missed {missed}"
        f" | no ruler: {fp}/{fp_total} false positives"
        f" | median {statistics.median(seconds) if seconds else 0:.2f}s/image"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=["legacy", "new", "both"], default="both")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dir", action="append", dest="dirs", help="folder with a ruler")
    parser.add_argument("--nodir", action="append", dest="nodirs", help="folder without a ruler")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.dirs or args.nodirs:
        folders = [(Path(d), True) for d in (args.dirs or [])]
        folders += [(Path(d), False) for d in (args.nodirs or [])]
    else:
        folders = [(Path(d), flag) for d, flag in DEFAULT_DIRS]

    impls = ["legacy", "new"] if args.impl == "both" else [args.impl]
    reports = [evaluate(impl, folders, args.limit, args.seed) for impl in impls]

    print("\n================ summary ================")
    for report in reports:
        print(summarise(report))

    if args.json:
        args.json.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
