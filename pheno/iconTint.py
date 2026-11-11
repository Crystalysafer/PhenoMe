"""Repaint the bundled PNG icons in one ink colour, so the set reads as one set.

**Why this exists.** ``pheno/icons`` grew by addition: 19 of the 46 files are already
dark monochrome, 27 are flat colour, and their source sizes run from 22x22 to 1024x1024.
Placed side by side on one toolbar they look like three different applications, and the
dark ones nearly disappear against a dark toolbar. Redrawing 46 icons is a job for a
designer; recolouring them is a job a program can do.

**How.** An icon carries its shape in three possible ways -- the alpha channel, the
luminance of its pixels, or hue alone. This keeps alpha and luminance and throws hue
away: every visible pixel is painted in the theme's ink colour, with opacity taken from
how dark that pixel was. A blue disc with a white tick therefore becomes a dark disc
with a pale tick, which is exactly what a monochrome version of it should be. Icons
whose shape lives in hue alone lose information, but they keep their silhouette, and the
alternative is what we started with.

**Cost.** Tinting is per-pixel in Python, so it runs once per (icon, colour) and is
cached, and each source is first scaled into a :data:`WORK` box -- which also evens out
the size mismatch as a side effect. Set ``PHENOME_ICON_TINT=off`` to get the originals
back without touching code.
"""

from __future__ import annotations

import os
from pathlib import Path

from qtpy import QtCore, QtGui

ICONS_DIR = Path(__file__).resolve().parent / "icons"

#: Longest side of the working image. Icons are drawn at 22-36 px; 128 is generous
#: enough to stay sharp on a high-DPI screen and small enough that tinting 46 of them
#: costs a fraction of a second.
WORK = 128

#: Below this share of fully transparent pixels the file is an illustration with a
#: painted background rather than an icon with a transparent one (``quit.png`` is the
#: only one), and tinting it would produce a solid rectangle.
MIN_CLEAR = 0.05

#: Icons that keep their own colours. ``icon.png`` is the PhenoMe logo: it is what the
#: taskbar and the title bar show, and a brand mark flattened to one ink stops being a
#: brand mark. Everything else in the set is a UI glyph and is recoloured.
KEEP_COLOUR = frozenset({"icon"})

#: Luminance above which an icon counts as light art -- white glyphs on transparency.
#: Those need the mapping inverted, or they would tint to nothing.
LIGHT_ART = 0.6

_cache: dict[tuple[str, str], QtGui.QIcon] = {}


def enabled() -> bool:
    """Whether tinting is on. Off restores the original artwork."""
    return os.environ.get("PHENOME_ICON_TINT", "on").strip().lower() not in ("off", "0", "no")


def clear_cache() -> None:
    """Drop the cache, for a test or after a theme change."""
    _cache.clear()


def available() -> list[str]:
    """Every bundled icon name, without its extension."""
    return sorted(path.stem for path in ICONS_DIR.glob("*.png"))


def _load(name: str) -> QtGui.QImage | None:
    path = ICONS_DIR / f"{name}.png"
    if not path.is_file():
        return None
    image = QtGui.QImage(str(path))
    return None if image.isNull() else image


def _to_work_box(image: QtGui.QImage) -> QtGui.QImage:
    """Scale into a square WORK box, centred, never upscaling.

    The square matters more than the size: a 298x264 icon padded to a square keeps the
    same optical weight as its neighbours, where a tight-cropped one would look larger.
    """
    side = min(WORK, max(image.width(), image.height()))
    scaled = image.scaled(
        side,
        side,
        QtCore.Qt.KeepAspectRatio,
        QtCore.Qt.SmoothTransformation,
    )
    canvas = QtGui.QImage(side, side, QtGui.QImage.Format_ARGB32_Premultiplied)
    canvas.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(canvas)
    painter.drawImage((side - scaled.width()) // 2, (side - scaled.height()) // 2, scaled)
    painter.end()
    return canvas


def _clear_share(image: QtGui.QImage) -> float:
    clear = 0
    total = image.width() * image.height()
    for y in range(image.height()):
        for x in range(image.width()):
            if image.pixelColor(x, y).alpha() < 8:
                clear += 1
    return clear / max(1, total)


def _is_light_art(image: QtGui.QImage) -> bool:
    total = 0
    luminance = 0.0
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            if pixel.alpha() < 32:
                continue
            total += 1
            luminance += (
                0.2126 * pixel.redF() + 0.7152 * pixel.greenF() + 0.0722 * pixel.blueF()
            )
    return total > 0 and (luminance / total) > LIGHT_ART


def _ink_image(image: QtGui.QImage, ink: QtGui.QColor) -> QtGui.QImage:
    """Paint every visible pixel in ``ink``, opacity from how dark the pixel was.

    Opacity comes from the pixel's *absolute* luminance, deliberately not from its
    luminance relative to the rest of the icon. Normalising per icon was tried and
    rejected: an icon that is a mid-grey shape with a darker outline -- ``delete.png`` is
    exactly that -- has its body stretched down to nearly transparent, and at 22 px all
    that survives is a faint outline. Absolute luminance keeps solid shapes solid.
    """
    light_art = _is_light_art(image)
    # Straight ARGB, not premultiplied: the pixels below are written un-premultiplied,
    # and Qt un-premultiplies on read -- which brightens every partly transparent edge
    # and softens exactly the detail this function exists to keep.
    out = QtGui.QImage(image.size(), QtGui.QImage.Format_ARGB32)
    out.fill(QtCore.Qt.transparent)
    ink_rgb = ink.rgb() & 0x00FFFFFF
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            alpha = pixel.alpha()
            if alpha == 0:
                continue
            luminance = (
                0.2126 * pixel.redF() + 0.7152 * pixel.greenF() + 0.0722 * pixel.blueF()
            )
            coverage = luminance if light_art else 1.0 - luminance
            # Glossy source art is all mid-tones, which would tint to haze; the curve
            # pushes those towards solid ink. There is deliberately no floor: pale
            # interiors have to stay fully transparent, because that is what makes a
            # white tick inside a dark shape read as a tick rather than as a smudge --
            # and it is the dark theme, where the ink is light, that notices first.
            coverage = coverage ** 0.55
            out.setPixel(x, y, (int(coverage * alpha) << 24) | ink_rgb)
    return out


def tinted(name: str, ink: QtGui.QColor | str) -> QtGui.QIcon | None:
    """The icon in one colour, or ``None`` when there is nothing to tint."""
    colour = QtGui.QColor(ink) if not isinstance(ink, QtGui.QColor) else ink
    key = (name, colour.name())
    if key in _cache:
        return _cache[key]

    image = _load(name)
    if image is None:
        return None
    if name in KEEP_COLOUR:
        result = QtGui.QIcon(QtGui.QPixmap.fromImage(image))
    else:
        boxed = _to_work_box(image)
        if _clear_share(boxed) < MIN_CLEAR:
            # An illustration, not an icon: leave it alone rather than make a solid block.
            result = QtGui.QIcon(QtGui.QPixmap.fromImage(image))
        else:
            result = QtGui.QIcon(QtGui.QPixmap.fromImage(_ink_image(boxed, colour)))
    _cache[key] = result
    return result


def tinted_pixmap(name: str, ink: QtGui.QColor | str, size: int) -> QtGui.QPixmap | None:
    """The tinted icon at a specific pixel size, for a preview sheet."""
    icon = tinted(name, ink)
    if icon is None:
        return None
    return icon.pixmap(QtCore.QSize(size, size))


def original(name: str) -> QtGui.QIcon | None:
    """The bundled artwork untouched -- classic mode shows the icons as shipped."""
    path = ICONS_DIR / f"{name}.png"
    if not path.is_file():
        return None
    icon = QtGui.QIcon(str(path))
    return None if icon.isNull() else icon


def refresh(root, ink: QtGui.QColor | str) -> int:
    """Redraw every icon tagged with ``phenoIcon`` under ``root``.

    What ``lib.newIcon`` tags, this repaints. Called on a theme change, because an icon
    is a QPixmap: once set it keeps the colour it was built with, and a stylesheet
    cannot reach it.
    """
    from qtpy import QtWidgets

    count = 0
    for kind in (QtWidgets.QAction, QtWidgets.QAbstractButton):
        for item in root.findChildren(kind):
            name = item.property("phenoIcon")
            if not name:
                continue
            icon = tinted(str(name), ink)
            if icon is not None:
                item.setIcon(icon)
                count += 1
    return count


def restore(root) -> int:
    """Put every tagged icon's original artwork back after a switch to classic."""
    from qtpy import QtWidgets

    count = 0
    for kind in (QtWidgets.QAction, QtWidgets.QAbstractButton):
        for item in root.findChildren(kind):
            name = item.property("phenoIcon")
            if not name:
                continue
            icon = original(str(name))
            if icon is not None:
                item.setIcon(icon)
                count += 1
    return count
