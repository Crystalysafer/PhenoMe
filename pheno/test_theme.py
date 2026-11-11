"""Tests for the look: the palette, the stylesheet, and the icon recolouring.

Qt is needed for the image work but not a display: an offscreen application is enough to
build pixmaps, so this runs anywhere the rest of the suite does.

    python -m unittest pheno.test_theme
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtGui, QtWidgets  # noqa: E402

from pheno import iconTint  # noqa: E402
from pheno.theme import DARK, LIGHT, apply_theme, mode, palette, stylesheet_for  # noqa: E402

#: Held in a name on purpose. A QApplication with no Python reference is collected and
#: every Qt call after that crashes the process with no message.
_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def off_ink_share(source, ink: str, min_alpha: int = 8) -> float:
    """Share of visible pixels that are not the requested ink.

    The invariant is exact rather than a judgement about greyness: the ink here is a
    blue-grey, so "is it neutral" would be the wrong question. Every visible pixel should
    be that colour with some alpha, and any other colour means the original artwork
    survived.

    ``min_alpha`` exists because Qt scales premultiplied pixmaps and un-premultiplies on
    read, so a rounding error of one in a nearly transparent pixel becomes a large drift
    in RGB. Faint pixels are skipped rather than tolerated.
    """
    wanted = QtGui.QColor(ink)
    image = source if isinstance(source, QtGui.QImage) else source.toImage()
    off = 0
    visible = 0
    for y in range(image.height()):
        for x in range(image.width()):
            colour = image.pixelColor(x, y)
            if colour.alpha() < min_alpha:
                continue
            visible += 1
            drift = max(
                abs(colour.red() - wanted.red()),
                abs(colour.green() - wanted.green()),
                abs(colour.blue() - wanted.blue()),
            )
            if drift > 8:
                off += 1
    return off / max(1, visible)


def painted(name: str, ink: str) -> QtGui.QImage:
    """The tinting step alone, without any later scaling: what the module produces."""
    source = iconTint._load(name)
    assert source is not None, name
    return iconTint._ink_image(iconTint._to_work_box(source), QtGui.QColor(ink))


class PaletteTest(unittest.TestCase):
    def test_modes_define_the_same_colours(self):
        """A key in one mode and not the other is a KeyError waiting for a theme switch."""
        self.assertEqual(sorted(LIGHT), sorted(DARK))

    def test_stylesheet_interpolates_every_value(self):
        for mode in ("light", "dark"):
            sheet = stylesheet_for(mode)
            self.assertIn(palette_hex(mode), sheet, mode)
            self.assertNotIn("{c[", sheet, f"{mode} left a placeholder unformatted")

    def test_stylesheet_covers_the_widgets_the_window_uses(self):
        sheet = stylesheet_for("light")
        for selector in ("QToolBar", "QGroupBox", "QHeaderView", "QScrollBar", "QComboBox"):
            self.assertIn(selector, sheet, selector)


def palette_hex(mode: str) -> str:
    return (DARK if mode == "dark" else LIGHT)["accent"]


class ApplyTest(unittest.TestCase):
    def test_apply_sets_mode_and_font(self):
        apply_theme(_APP, "dark")
        self.assertEqual(palette()["text"], DARK["text"])
        self.assertNotEqual(_APP.styleSheet(), "")
        apply_theme(_APP, "light")
        self.assertEqual(palette()["text"], LIGHT["text"])

    def test_classic_restores_the_native_look(self):
        """Classic is the default: no stylesheet, native palette, original icons."""
        apply_theme(_APP, "light")
        self.assertNotEqual(_APP.styleSheet(), "")
        apply_theme(_APP, "classic")
        self.assertEqual(mode(), "classic")
        self.assertEqual(_APP.styleSheet(), "")
        # palette() still answers a usable table for per-widget styling, read back
        # from the platform palette instead of the fixed theme values.
        self.assertTrue(palette()["text"].startswith("#"))
        apply_theme(_APP, "light")  # leave the app themed for the following tests


class IconTintTest(unittest.TestCase):
    def test_every_icon_is_recoloured(self):
        """The point of the module: one ink, no leftovers.

        Two files are deliberately exempt. ``quit.png`` is an illustration with a painted
        background rather than an icon with a transparent one, so tinting it would
        produce a solid rectangle; ``icon.png`` is the PhenoMe logo and stays coloured,
        because a brand mark flattened to one ink stops being a brand mark.
        """
        names = [
            n
            for n in iconTint.available()
            if n not in iconTint.KEEP_COLOUR and n != "quit"
        ]
        self.assertGreater(len(names), 30)
        for name in names:
            share = off_ink_share(painted(name, LIGHT["icon"]), LIGHT["icon"])
            self.assertLess(share, 0.001, f"{name} still carries its old colours ({share:.1%})")

    def test_logo_keeps_its_colours(self):
        """The application icon is a brand mark, not a UI glyph, and must not be tinted."""
        self.assertIn("icon", iconTint.KEEP_COLOUR)
        for ink in (LIGHT["icon"], DARK["icon"]):
            pixmap = iconTint.tinted_pixmap("icon", ink, 48)
            self.assertIsNotNone(pixmap)
            self.assertGreater(
                off_ink_share(pixmap, ink, min_alpha=64),
                0.1,
                "the logo was recoloured",
            )

    def test_illustration_is_left_alone(self):
        pixmap = iconTint.tinted_pixmap("quit", LIGHT["icon"], 40)
        self.assertIsNotNone(pixmap)
        self.assertGreater(off_ink_share(pixmap, LIGHT["icon"], min_alpha=64), 0.2)

    def test_both_inks_are_used(self):
        """A theme switch has to reach the glyphs, not just the background."""
        for ink, other in ((LIGHT["icon"], DARK["icon"]), (DARK["icon"], LIGHT["icon"])):
            image = painted("area_detect", ink)
            self.assertLess(off_ink_share(image, ink), 0.001, ink)
            # And the two inks really are different colours.
            self.assertGreater(off_ink_share(image, other), 0.9)

    def test_result_is_a_silhouette_not_a_blob(self):
        """The tick has to survive as negative space, or every icon becomes a disc.

        The bound on the maximum is loose on purpose: opacity follows the source
        artwork's luminance, and a mid-blue disc tops out below full ink. What matters is
        that the range is wide and that a real area of the shape stays clear.
        """
        image = painted("blue_check", LIGHT["icon"])
        alphas = []
        for y in range(image.height()):
            for x in range(image.width()):
                alpha = image.pixelColor(x, y).alpha()
                if alpha:
                    alphas.append(alpha)
        self.assertTrue(alphas)
        self.assertGreater(max(alphas), 150, "nothing reaches solid ink")
        # A stroke, not one stray pixel: the tick inside the disc is thin, so the count
        # is the honest measure rather than a share of the whole shape.
        faint = sum(1 for a in alphas if a < 80)
        self.assertGreater(faint, 50, f"transparent detail was lost ({faint} faint pixels)")

    def test_missing_icon_returns_nothing(self):
        self.assertIsNone(iconTint.tinted("no-such-icon", LIGHT["icon"]))

    def test_off_switch(self):
        os.environ["PHENOME_ICON_TINT"] = "off"
        try:
            self.assertFalse(iconTint.enabled())
        finally:
            del os.environ["PHENOME_ICON_TINT"]
        self.assertTrue(iconTint.enabled())


class RefreshTest(unittest.TestCase):
    def test_tagged_icons_are_repainted(self):
        """A theme switch has to reach the pixmaps, which no stylesheet can do."""
        window = QtWidgets.QWidget()
        action = QtWidgets.QAction(window)
        action.setProperty("phenoIcon", "delete")
        untouched = QtWidgets.QAction(window)
        untouched.setProperty("phenoIcon", "no-such-icon")

        self.assertEqual(iconTint.refresh(window, DARK["icon"]), 1)
        self.assertFalse(action.icon().isNull())
        self.assertTrue(untouched.icon().isNull())


if __name__ == "__main__":
    unittest.main()
