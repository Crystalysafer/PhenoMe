import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pheno.color import (
    CalibrationError,
    MCC24_REF_SRGB,
    card_lattice,
    correct_colors,
    detect_charts,
    patch_centers,
)
from pheno.RulerNet import RulerNet


class Checker:
    def getBox(self):
        return np.array([[10, 10], [70, 10], [70, 90], [10, 90]])

    def getChartsRGB(self):
        values = np.zeros((72, 2), dtype=np.float32)
        values[:, 1] = (MCC24_REF_SRGB * 255).reshape(-1)
        return values


class Detector:
    def __init__(self, success):
        self.success = success

    def process(self, *args):
        if len(args) == 3:
            raise cv2.error("process() takes at most 2 arguments (3 given)")
        return self.success

    def getListColorChecker(self):
        return [Checker()]


def _synthetic_card(rotated=False, ruler=True):
    """A card photo: 6x4 MCC24 patch grid plus a millimetre scale below it.

    ``rotated`` stands the card on its head (the real 180-degree case, so the
    neutral column and the ruler swap sides), and the mm scale is a 12 px pitch
    tick sequence under the last patch row.
    """
    image = np.full((1200, 900, 3), (60, 55, 50), np.uint8)
    patches = np.rot90(MCC24_REF_SRGB.reshape(4, 6, 3), 1)
    for row in range(6):
        for col in range(4):
            color = tuple(int(v * 255) for v in patches[row, col, ::-1])
            x = 100 + (3 - col if rotated else col) * 40
            y = 300 + (5 - row if rotated else row) * 40
            cv2.rectangle(image, (x, y), (x + 33, y + 33), color, -1)
    if ruler:
        top = 230 if rotated else 545
        for index, x in enumerate(range(90, 270, 12)):
            height = 24 if index % 10 == 0 else (18 if index % 5 == 0 else 12)
            cv2.line(image, (x, top), (x, top + height), (235, 235, 235), 1)
    return image


class CalibrationTest(unittest.TestCase):
    def test_mcc_result_used_when_detector_finds_it(self):
        image = np.zeros((1200, 900, 3), dtype=np.uint8)
        with patch(
            "pheno.color.cv2.mcc.CCheckerDetector_create", return_value=Detector(True)
        ):
            chart = detect_charts(image)[0]
        self.assertLess(chart["mean_delta_e"], 0.01)
        self.assertEqual(patch_centers(chart["checker"])[0], [15, 20])

    def test_lattice_fallback_finds_synthetic_card(self):
        image = _synthetic_card()
        with patch(
            "pheno.color.cv2.mcc.CCheckerDetector_create", return_value=Detector(False)
        ):
            chart = detect_charts(image)[0]
        self.assertLess(chart["mean_delta_e"], 3)
        # Patch order is verified geometrically: the first reference patch
        # (dark skin) sits bottom-left of the upright card.
        np.testing.assert_allclose(chart["centers"][0], [116, 516], atol=8)

    def test_lattice_handles_upside_down_card(self):
        image = _synthetic_card(rotated=True)
        with patch(
            "pheno.color.cv2.mcc.CCheckerDetector_create", return_value=Detector(False)
        ):
            chart = detect_charts(image)[0]
        self.assertLess(chart["mean_delta_e"], 3)
        np.testing.assert_allclose(chart["centers"][0], [236, 316], atol=8)

    def test_no_chart_remains_undetected(self):
        image = np.zeros((1200, 900, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 200), (300, 550), (30, 30, 200), -1)
        with patch(
            "pheno.color.cv2.mcc.CCheckerDetector_create", return_value=Detector(False)
        ):
            with self.assertRaises(CalibrationError):
                correct_colors(image)

    def test_uniform_grid_fails_colour_validation(self):
        # A regular 6x3 cluster of same-coloured blobs still has to pass the
        # 24-patch colour ordering; geometry alone is not enough.
        image = np.zeros((1200, 900, 3), dtype=np.uint8)
        for row in range(6):
            for col in range(3):
                cv2.rectangle(
                    image,
                    (100 + col * 40, 300 + row * 40),
                    (133 + col * 40, 333 + row * 40),
                    (40, 30, 220),
                    -1,
                )
        with patch(
            "pheno.color.cv2.mcc.CCheckerDetector_create", return_value=Detector(False)
        ):
            with self.assertRaises(CalibrationError):
                correct_colors(image)

    def test_irregular_colour_blobs_form_no_lattice(self):
        image = np.zeros((1200, 900, 3), dtype=np.uint8)
        for index, (x, y) in enumerate(
            ((200, 300), (410, 350), (300, 600), (650, 200), (150, 700))
        ):
            cv2.circle(image, (x, y), 25, (20 + index * 40, 40, 220 - index * 20), -1)
        self.assertIsNone(card_lattice(image))


class CardRulerTest(unittest.TestCase):
    def test_card_ruler_reads_printed_ticks(self):
        image = _synthetic_card()
        ruler = RulerNet.__new__(RulerNet)
        points, pixels_per_cm = ruler._card_ruler(image[..., ::-1], image)
        self.assertIsNotNone(points)
        self.assertGreaterEqual(len(points), 9)
        self.assertAlmostEqual(pixels_per_cm, 120.0, delta=6.0)

    def test_card_ruler_handles_upside_down_card(self):
        image = _synthetic_card(rotated=True)
        ruler = RulerNet.__new__(RulerNet)
        points, pixels_per_cm = ruler._card_ruler(image[..., ::-1], image)
        self.assertIsNotNone(points)
        self.assertAlmostEqual(pixels_per_cm, 120.0, delta=6.0)

    def test_card_ruler_ignores_plain_scene(self):
        image = np.zeros((1200, 900, 3), dtype=np.uint8)
        ruler = RulerNet.__new__(RulerNet)
        self.assertEqual(ruler._card_ruler(image[..., ::-1], image), (None, 0.0))

    def test_ruler_falls_back_to_card_scale(self):
        image = _synthetic_card()
        ruler = RulerNet.__new__(RulerNet)
        with patch.object(ruler, "_infer_region", return_value=(None, 0.0)), patch.object(
            ruler, "_propose_regions", return_value=[("full", image, (0, 0))]
        ):
            measured, scale = ruler.get_tickpoints_and_scale(image)
        self.assertIsNotNone(measured)
        self.assertAlmostEqual(scale, 120.0, delta=6.0)


if __name__ == "__main__":
    unittest.main()
