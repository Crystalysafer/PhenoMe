"""Tests for the VRAM readout's arithmetic and thresholds.

The pure part is tested here without a GPU: ``summary()`` takes a plain dict, so the
interesting cases -- a full card, a card with no device, a card whose memory is mostly
held by *other* applications -- need no hardware. The widget itself is thin enough that
the preview is a better check than a test.

    python -m unittest pheno.test_gpuMonitor
"""

from __future__ import annotations

import unittest

from pheno.gpuMonitor import AMBER_PERCENT, RED_PERCENT, summary


def reading(used: float, total: float = 12.0, tensors: float = 0.0, pool: float = 0.0) -> dict:
    """A ``get_memory_info()`` dict, in the shape that function actually returns."""
    free = total - used
    return {
        "device": 0,
        "total_gb": total,
        "allocated_gb": tensors,
        "cached_gb": pool,
        "used_gb": used,
        "free_gb": free,
        "usage_percent": round(used / total * 100, 1) if total else 0.0,
    }


class SummaryTest(unittest.TestCase):
    def test_no_device_is_none_rather_than_zero(self):
        """None is the signal to show nothing at all, so it must not become a dict."""
        self.assertIsNone(summary(None))
        self.assertIsNone(summary({}))

    def test_a_quiet_card_is_not_flagged(self):
        result = summary(reading(2.0))
        self.assertEqual(result["severity"], "ok")
        self.assertAlmostEqual(result["free_gb"], 10.0)

    def test_thresholds_sit_where_they_say(self):
        self.assertEqual(summary(reading(AMBER_PERCENT / 100 * 12))["severity"], "warn")
        self.assertEqual(summary(reading(RED_PERCENT / 100 * 12))["severity"], "danger")
        below = (AMBER_PERCENT - 1) / 100 * 12
        self.assertEqual(summary(reading(below))["severity"], "ok")

    def test_memory_held_elsewhere_is_separated_from_our_own(self):
        """The figure the whole readout exists for.

        A 12 GB card reports 3.5 GB used while the tensors only account for 0.5 GB; the
        rest is the CUDA context and whatever else the machine is running, and that is
        why the next allocation may not fit.
        """
        result = summary(reading(3.5, tensors=0.5, pool=1.0))

        self.assertAlmostEqual(result["tensors_gb"], 0.5)
        self.assertAlmostEqual(result["pool_gb"], 1.0)
        self.assertAlmostEqual(result["elsewhere_gb"], 2.5)

    def test_elsewhere_never_goes_negative(self):
        """PyTorch's pool can exceed the driver figure while memory is being released."""
        result = summary(reading(1.0, pool=2.0))

        self.assertEqual(result["elsewhere_gb"], 0.0)

    def test_missing_fields_do_not_raise(self):
        """A partial dict should degrade to zeros, not take the status bar down."""
        result = summary({"total_gb": 12.0})

        self.assertEqual(result["severity"], "ok")
        self.assertAlmostEqual(result["used_gb"], 0.0)


if __name__ == "__main__":
    unittest.main()
