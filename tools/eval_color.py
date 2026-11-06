"""Compare colour-correction fits on real photographs, with cross-validation.

Fitting 24 patches with a flexible model always looks better on those same 24
patches, so every candidate is also scored leave-one-out: fit on 23 patches,
measure the one held out. That is what predicts how the correction behaves on the
beans, which are not patches.

Candidates:

  A  gamma3x3      3x3 least squares directly on gamma-encoded sRGB (current code)
  B  linear3x3     3x3 least squares after linearising both sides (gamma 2.2)
  C  ccm_lin       cv2.ccm, CIE2000 loss, gamma linearisation
  D  ccm_id        cv2.ccm, CIE2000 loss, no linearisation
  E  gamma4x3      3x3 plus offset, gamma-encoded
  F  wb_gamma3x3   grey-patch white balance first, then A
  G  poly2         quadratic polynomial per channel (10 terms), gamma-encoded

    python -m tools.eval_color "<image>" [more images...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

MCC24_REF_SRGB = np.array(
    [
        [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
        [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
        [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
        [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
        [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
        [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
    ],
    dtype=np.float64,
) / 255.0
GREY_PATCHES = slice(18, 24)
GAMMA = 2.2


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def linearise(values):
    return np.power(np.clip(values, 0.0, 1.0), GAMMA)


def delinearise(values):
    return np.power(np.clip(values, 0.0, 1.0), 1.0 / GAMMA)


def to_lab(rgb01):
    rgb = np.clip(rgb01, 0, 1).astype(np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def delta_e(a, b):
    return np.linalg.norm(to_lab(a) - to_lab(b), axis=1)


def detect_patches(image_bgr):
    detector = cv2.mcc.CCheckerDetector_create()
    if not detector.process(image_bgr, cv2.mcc.MCC24, 1):
        return None
    checkers = detector.getListColorChecker()
    if not checkers:
        return None
    charts = checkers[0].getChartsRGB()
    if charts is None or len(charts) == 0:
        return None
    return charts[:, 1].copy().reshape(-1, 3).astype(np.float64) / 255.0


# --- candidate models: fit(src, ref) -> apply(values) ----------------------

def make_gamma3x3(src, ref):
    ccm, *_ = np.linalg.lstsq(src, ref, rcond=None)
    return lambda values: np.clip(values @ ccm, 0, 1)


def make_linear3x3(src, ref):
    ccm, *_ = np.linalg.lstsq(linearise(src), linearise(ref), rcond=None)
    return lambda values: delinearise(linearise(values) @ ccm)


def _ccm_model(src, linearisation, ref=None):
    # The built-in COLORCHECKER_Macbeth reference only accepts all 24 patches, so
    # cross-validation has to pass the reference colours explicitly.
    src_mat = src.astype(np.float64).reshape(-1, 1, 3)
    if ref is None or len(ref) == 24:
        model = cv2.ccm.ColorCorrectionModel(src_mat, cv2.ccm.COLORCHECKER_Macbeth)
    else:
        model = cv2.ccm.ColorCorrectionModel(
            src_mat, ref.astype(np.float64).reshape(-1, 1, 3), cv2.ccm.COLOR_SPACE_sRGB
        )
    model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
    model.setCCM_TYPE(cv2.ccm.CCM_3x3)
    model.setDistance(cv2.ccm.DISTANCE_CIE2000)
    model.setLinear(linearisation)
    if linearisation == cv2.ccm.LINEARIZATION_GAMMA:
        model.setLinearGamma(GAMMA)
    model.run()
    return np.asarray(model.getCCM(), dtype=np.float64)


def make_ccm_lin(src, ref):
    ccm = _ccm_model(src, cv2.ccm.LINEARIZATION_GAMMA, ref)
    return lambda values: delinearise(linearise(values) @ ccm)


def make_ccm_id(src, ref):
    ccm = _ccm_model(src, cv2.ccm.LINEARIZATION_IDENTITY, ref)
    return lambda values: np.clip(values @ ccm, 0, 1)


def make_gamma4x3(src, ref):
    design = np.hstack([src, np.ones((len(src), 1))])
    ccm, *_ = np.linalg.lstsq(design, ref, rcond=None)
    return lambda values: np.clip(
        np.hstack([values, np.ones((len(values), 1))]) @ ccm, 0, 1
    )


def _grey_gains(src):
    grey_src = src[GREY_PATCHES]
    grey_ref = MCC24_REF_SRGB[GREY_PATCHES]
    # Match each channel's response on the neutral wedge before mixing channels.
    gains = np.array(
        [
            np.linalg.lstsq(
                grey_src[:, channel : channel + 1], grey_ref[:, channel], rcond=None
            )[0][0]
            for channel in range(3)
        ]
    )
    return gains


def make_wb_gamma3x3(src, ref):
    gains = _grey_gains(src)
    balanced = np.clip(src * gains, 0, 1)
    ccm, *_ = np.linalg.lstsq(balanced, ref, rcond=None)
    return lambda values: np.clip(np.clip(values * gains, 0, 1) @ ccm, 0, 1)


def _poly_terms(values):
    r, g, b = values[:, 0], values[:, 1], values[:, 2]
    return np.stack(
        [r, g, b, r * r, g * g, b * b, r * g, r * b, g * b, np.ones_like(r)], axis=1
    )


def make_poly2(src, ref):
    coefficients, *_ = np.linalg.lstsq(_poly_terms(src), ref, rcond=None)
    return lambda values: np.clip(_poly_terms(values) @ coefficients, 0, 1)


CANDIDATES = {
    "A gamma3x3": make_gamma3x3,
    "B linear3x3": make_linear3x3,
    "C ccm_lin": make_ccm_lin,
    "D ccm_id": make_ccm_id,
    "E gamma4x3": make_gamma4x3,
    "F wb_gamma3x3": make_wb_gamma3x3,
    "G poly2": make_poly2,
}


def score(src, ref, factory):
    fitted = factory(src, ref)
    inside = delta_e(fitted(src), ref)

    held_out = []
    for index in range(len(src)):
        keep = np.ones(len(src), bool)
        keep[index] = False
        try:
            partial = factory(src[keep], ref[keep])
            held_out.append(delta_e(partial(src[index : index + 1]), ref[index : index + 1])[0])
        except Exception:  # noqa: BLE001
            held_out.append(np.nan)
    return {
        "dE_fit": float(np.mean(inside)),
        "dE_cv": float(np.nanmean(held_out)),
        "dE_cv_max": float(np.nanmax(held_out)),
    }


def analyse(path: Path) -> dict | None:
    image = read_image(path)
    if image is None:
        print(f"{path.name}: unreadable")
        return None
    src = detect_patches(image)
    if src is None:
        print(f"{path.name}: no MCC24 chart detected")
        return None

    print(f"\n{path.name}  ({image.shape[1]}x{image.shape[0]})")
    raw = float(np.mean(delta_e(src, MCC24_REF_SRGB)))
    print(f"  {'raw (uncorrected)':<16} dE {raw:6.2f}")
    results = {}
    for name, factory in CANDIDATES.items():
        try:
            stats = score(src, MCC24_REF_SRGB, factory)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<16} failed: {type(exc).__name__}: {exc}")
            continue
        results[name] = stats
        print(
            f"  {name:<16} dE_fit {stats['dE_fit']:6.2f}"
            f"   dE_leave1out {stats['dE_cv']:6.2f} (worst {stats['dE_cv_max']:6.2f})"
        )
    return results


def main() -> None:
    paths = [Path(arg) for arg in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return
    everything: dict[str, list[float]] = {}
    for path in paths:
        results = analyse(path)
        if not results:
            continue
        for name, stats in results.items():
            everything.setdefault(name, []).append(stats["dE_cv"])
    if everything:
        print("\n==== mean leave-one-out dE across images ====")
        for name, values in sorted(everything.items(), key=lambda item: np.mean(item[1])):
            print(f"  {name:<16} {np.mean(values):6.2f}   (n={len(values)})")


if __name__ == "__main__":
    main()
