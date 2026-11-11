"""Shared colours and Qt stylesheets for the desktop interface.

The theme is applied before the main window is built. Light and dark stylesheets
are generated from palette tables without changing widget behaviour.
"""

from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets

# Light-theme colour palette.
LIGHT = {
    "ground": "#eef2f3",
    "surface": "#ffffff",
    "soft": "#f4f7f7",
    "hover": "#e9efef",
    "line": "#d8e0e2",
    "line_strong": "#c3cfd3",
    # The canvas sits in a well, on purpose deeper than the ground the panels rest on.
    # Without it the panels are white, the canvas is ground, and a four-percent
    # difference in luminance reads as one undivided window -- the left dock in
    # particular looked like part of the picture area.
    "well": "#d9e0e2",
    "text": "#1d2a31",
    "muted": "#66777f",
    "accent": "#1e7a55",
    "accent_hover": "#249164",
    "accent_soft": "#e7f3ed",
    "accent_text": "#ffffff",
    "danger": "#9d3030",
    "danger_soft": "#fdecec",
    "warn": "#8a6d1f",
    "selection": "#e7f3ed",
    "disabled": "#a9b4b8",
    # Toolbar and menu glyphs are redrawn in this one colour by pheno.iconTint, which is
    # what makes a set assembled from three different icon packs read as one set.
    "icon": "#43535b",
}

DARK = {
    "ground": "#161b1d",
    "surface": "#1e2528",
    "soft": "#232b2f",
    "hover": "#2a3337",
    "line": "#313c41",
    "line_strong": "#3d4a50",
    # Deeper than the ground, for the same reason as in the light palette.
    "well": "#0e1214",
    "text": "#e6ecee",
    "muted": "#93a2a8",
    "accent": "#38a97a",
    "accent_hover": "#45bd8a",
    "accent_soft": "#1d2f28",
    "accent_text": "#0d1512",
    "danger": "#e08a8a",
    "danger_soft": "#3a2222",
    "warn": "#d8b45c",
    "selection": "#1d2f28",
    "disabled": "#5d6a6f",
    "icon": "#c9d4d8",
}

# A Chinese Windows install hands a Qt app SimSun at 9pt, which is why the window read
# as roughly a decade older than the web front end: the font is most of the impression
# before any colour is considered. YaHei is the modern face for the same script; the
# list degrades to whatever exists rather than to a missing family.
_FONT_STACK = (
    '"Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", '
    '"Segoe UI", "Noto Sans CJK SC", sans-serif'
)
_FONT_CANDIDATES = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "PingFang SC",
    "Hiragino Sans GB",
    "Segoe UI",
    "Noto Sans CJK SC",
)


def _ui_font() -> QtGui.QFont:
    """The first installed face from the stack, at a size that matches 13px QSS."""
    available = set(QtGui.QFontDatabase().families())
    family = next((f for f in _FONT_CANDIDATES if f in available), None)
    font = QtGui.QFont()
    if family:
        font.setFamily(family)
    # Delegates and native-drawn pieces take their size from here; the stylesheet's
    # 13px governs the styled ones. 9.75pt is 13px at the usual 96 dpi.
    font.setPointSizeF(9.75)
    return font


def _stylesheet(c: dict) -> str:
    """The whole look, from one palette.

    Qt has no CSS variables, so the values are interpolated once here; keeping them in
    a dict above is what makes a light/dark pair maintainable rather than two
    stylesheets that disagree.
    """
    return f"""
/* ---------------------------------------------------------------- ground */
QWidget {{
    background: {c["ground"]};
    color: {c["text"]};
    font-family: {_FONT_STACK};
    font-size: 13px;
}}
QMainWindow, QDialog {{ background: {c["ground"]}; }}

QToolTip {{
    background: {c["surface"]};
    color: {c["text"]};
    border: 1px solid {c["line"]};
    border-radius: 6px;
    padding: 4px 7px;
}}

/* --------------------------------------------------------------- panels */
/* Docked panels read as cards on the ground rather than as raw sub-windows. The title
   bar is a shade off the body so a panel reads as a panel even when its edges sit
   against a same-coloured neighbour. */
QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    font-weight: 600;
    color: {c["muted"]};
}}
QDockWidget::title {{
    background: {c["soft"]};
    padding: 7px 10px;
    border: 1px solid {c["line_strong"]};
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}
QDockWidget > QWidget {{
    background: {c["surface"]};
    border: 1px solid {c["line_strong"]};
    border-top: none;
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
}}
QSplitter::handle {{ background: {c["line_strong"]}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}

/* The picture area is a well: recessed below the ground the panels sit on, so the
   canvas, the left dock and the right dock are three visibly separate places.
   Selected by object name, not by class: the background is painted by Qt before
   Canvas.paintEvent() runs and Canvas never fills it itself, so a rule that does not
   match this exact widget leaves the generic QWidget ground showing through -- which is
   what a "QGraphicsView" selector did. */
#CanvasView {{ background: {c["well"]}; }}

/* -------------------------------------------------------------- toolbar */
QToolBar {{
    background: {c["surface"]};
    border: none;
    border-bottom: 1px solid {c["line"]};
    padding: 5px 8px;
    spacing: 2px;
}}
QToolBar::separator {{
    background: {c["line"]};
    width: 1px;
    margin: 6px 6px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 5px 7px;
    color: {c["text"]};
}}
QToolButton:hover {{
    background: {c["hover"]};
    border-color: {c["line"]};
}}
QToolButton:pressed, QToolButton:checked {{
    background: {c["accent_soft"]};
    border-color: {c["accent"]};
}}
QToolButton:disabled {{ color: {c["disabled"]}; }}
QToolButton::menu-indicator {{ image: none; }}

QMenuBar {{
    background: {c["surface"]};
    border-bottom: 1px solid {c["line"]};
    padding: 2px 4px;
}}
QMenuBar::item {{ padding: 5px 10px; border-radius: 6px; background: transparent; }}
QMenuBar::item:selected {{ background: {c["hover"]}; }}

QMenu {{
    background: {c["surface"]};
    border: 1px solid {c["line"]};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c["accent_soft"]}; color: {c["accent"]}; }}
QMenu::separator {{ height: 1px; background: {c["line"]}; margin: 5px 8px; }}

/* -------------------------------------------------------------- buttons */
QPushButton {{
    background: {c["surface"]};
    border: 1px solid {c["line_strong"]};
    border-radius: 7px;
    padding: 6px 13px;
    color: {c["text"]};
}}
QPushButton:hover {{ background: {c["hover"]}; }}
QPushButton:pressed {{ background: {c["soft"]}; }}
QPushButton:disabled {{ color: {c["disabled"]}; border-color: {c["line"]}; }}
QPushButton:default, QPushButton[accent="true"] {{
    background: {c["accent"]};
    border-color: {c["accent"]};
    color: {c["accent_text"]};
    font-weight: 600;
}}
QPushButton:default:hover, QPushButton[accent="true"]:hover {{ background: {c["accent_hover"]}; }}

/* ---------------------------------------------------------------- input */
QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox {{
    background: {c["surface"]};
    border: 1px solid {c["line"]};
    border-radius: 7px;
    padding: 6px 9px;
    selection-background-color: {c["accent"]};
    selection-color: {c["accent_text"]};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {c["accent"]};
}}
QLineEdit:disabled {{ color: {c["disabled"]}; background: {c["soft"]}; }}

QComboBox {{
    background: {c["surface"]};
    border: 1px solid {c["line"]};
    border-radius: 7px;
    padding: 5px 9px;
    min-height: 20px;
}}
QComboBox:hover {{ border-color: {c["line_strong"]}; }}
QComboBox:focus {{ border-color: {c["accent"]}; }}
/* The drop-down and its arrow are deliberately left to Fusion: styling either one
   without supplying an arrow image makes Qt draw no arrow at all. */
QComboBox QAbstractItemView {{
    background: {c["surface"]};
    border: 1px solid {c["line"]};
    border-radius: 8px;
    padding: 4px;
    selection-background-color: {c["accent_soft"]};
    selection-color: {c["accent"]};
}}

/* ----------------------------------------------------------- item lists */
QListWidget, QListView, QTreeWidget, QTreeView {{
    background: {c["surface"]};
    border: none;
    padding: 4px;
    outline: none;
}}
QListWidget::item, QListView::item, QTreeWidget::item {{
    padding: 5px 7px;
    border-radius: 6px;
}}
QListWidget::item:hover, QListView::item:hover, QTreeWidget::item:hover {{ background: {c["hover"]}; }}
QListWidget::item:selected, QListView::item:selected, QTreeWidget::item:selected {{
    background: {c["accent_soft"]};
    color: {c["accent"]};
}}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {c["line_strong"]};
    border-radius: 4px;
    background: {c["surface"]};
}}
QCheckBox::indicator:checked {{
    background: {c["accent"]};
    border-color: {c["accent"]};
}}

/* --------------------------------------------------------------- tables */
QTableWidget, QTableView {{
    background: {c["surface"]};
    border: 1px solid {c["line"]};
    border-radius: 8px;
    gridline-color: {c["line"]};
    selection-background-color: {c["accent_soft"]};
    selection-color: {c["accent"]};
    outline: none;
}}
QHeaderView::section {{
    background: {c["soft"]};
    color: {c["muted"]};
    border: none;
    border-right: 1px solid {c["line"]};
    border-bottom: 1px solid {c["line"]};
    padding: 6px 8px;
    font-weight: 600;
}}
QHeaderView::section:last {{ border-right: none; }}
QTableCornerButton::section {{ background: {c["soft"]}; border: none; }}

/* ------------------------------------------------------------ scrollbar */
/* Thin and quiet: a chunky native scrollbar is most of what made the window look
   dated, and it is the cheapest thing to fix. */
QScrollBar:vertical {{
    background: transparent;
    width: 11px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {c["line_strong"]};
    border-radius: 5px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {c["muted"]}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: {c["line_strong"]};
    border-radius: 5px;
    min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{ background: {c["muted"]}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QScrollArea {{ border: none; }}

/* --------------------------------------------------------------- status */
QStatusBar {{
    background: {c["surface"]};
    border-top: 1px solid {c["line"]};
    color: {c["muted"]};
}}
QStatusBar::item {{ border: none; }}

QProgressBar {{
    background: {c["soft"]};
    border: none;
    border-radius: 5px;
    height: 6px;
    text-align: center;
    color: {c["muted"]};
}}
QProgressBar::chunk {{ background: {c["accent"]}; border-radius: 5px; }}

QGroupBox {{
    border: 1px solid {c["line"]};
    border-radius: 8px;
    /* Must clear one line of text, or the title lands on top of the border. */
    margin-top: 18px;
    padding-top: 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 5px;
    color: {c["muted"]};
}}

QTabWidget::pane {{ border: 1px solid {c["line"]}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 13px;
    color: {c["muted"]};
}}
QTabBar::tab:selected {{ color: {c["accent"]}; border-bottom-color: {c["accent"]}; font-weight: 600; }}
QTabBar::tab:hover {{ color: {c["text"]}; }}

QLabel {{ background: transparent; }}
QToolTip, QWhatsThis {{ background: {c["surface"]}; }}

/* Modal progress uses the same card look as the docked panels. */
QProgressDialog, QMessageBox {{ background: {c["surface"]}; }}
"""


#: The appearances on offer. ``classic`` is whatever the platform gave us before
#: this module existed -- the plain LabelMe-era look -- and the other two share
#: the one stylesheet.
MODES = ("classic", "light", "dark")

#: The style and font the application had before any theme touched it, captured on
#: first apply so switching back to ``classic`` restores them exactly rather than
#: approximating them.
_NATIVE_STYLE: str | None = None
_NATIVE_FONT: QtGui.QFont | None = None


def apply_theme(app, mode: str = "classic") -> None:
    """Style the whole application, or restore the native look for ``classic``.

    ``Fusion`` first: it is the one built-in style that draws consistently across
    platforms and obeys a stylesheet predictably. The native Windows style ignores
    parts of one, which is how you get a half-modern window.
    """
    global _ACTIVE_MODE, _NATIVE_STYLE, _NATIVE_FONT
    if _NATIVE_STYLE is None:
        _NATIVE_STYLE = app.style().objectName() or "Fusion"
        _NATIVE_FONT = QtGui.QFont(app.font())
    _ACTIVE_MODE = mode if mode in MODES else "classic"

    if _ACTIVE_MODE == "classic":
        app.setStyle(QtWidgets.QStyleFactory.create(_NATIVE_STYLE))
        app.setPalette(app.style().standardPalette())
        app.setStyleSheet("")
        if _NATIVE_FONT is not None:
            app.setFont(_NATIVE_FONT)
        return

    c = DARK if _ACTIVE_MODE == "dark" else LIGHT

    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    app.setFont(_ui_font())
    app.setStyleSheet(_stylesheet(c))

    # Fusion does not take these from the stylesheet, and a few controls -- the ones
    # drawn by delegates rather than by CSS -- only follow the palette.
    colours = QtGui.QPalette()
    colours.setColor(QtGui.QPalette.Window, QtGui.QColor(c["ground"]))
    colours.setColor(QtGui.QPalette.WindowText, QtGui.QColor(c["text"]))
    colours.setColor(QtGui.QPalette.Base, QtGui.QColor(c["surface"]))
    colours.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(c["soft"]))
    colours.setColor(QtGui.QPalette.Text, QtGui.QColor(c["text"]))
    colours.setColor(QtGui.QPalette.Button, QtGui.QColor(c["surface"]))
    colours.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(c["text"]))
    colours.setColor(QtGui.QPalette.Highlight, QtGui.QColor(c["accent"]))
    colours.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(c["accent_text"]))
    colours.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(c["surface"]))
    colours.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(c["text"]))
    colours.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(c["disabled"]))
    colours.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(c["disabled"]))
    app.setPalette(colours)


def stylesheet_for(mode: str) -> str:
    """The stylesheet alone, for a test or a preview to inspect."""
    if mode == "classic":
        return ""
    return _stylesheet(DARK if mode == "dark" else LIGHT)


#: Which palette is currently applied. The stylesheet covers most of the window, but a
#: handful of widgets carry a colour of their own -- Qt has no CSS variables, and some
#: of that styling predates this module -- so they need to be able to ask.
_ACTIVE_MODE = "classic"


def mode() -> str:
    """The mode currently applied to the application."""
    return _ACTIVE_MODE


def themed() -> bool:
    """Whether a real theme is active, i.e. anything but the native ``classic``."""
    return _ACTIVE_MODE != "classic"


def palette() -> dict:
    """The live colour table.

    Read this when a widget has to set a colour itself rather than through the
    stylesheet, and rebuild that style in the window's theme hook so a mode switch
    reaches it too. Under ``classic`` the table is read back from the platform's own
    palette, so per-widget styling follows the system instead of the fixed theme
    values; the key set is always the same as ``LIGHT``/``DARK``.
    """
    if _ACTIVE_MODE != "classic":
        return DARK if _ACTIVE_MODE == "dark" else LIGHT
    app = QtWidgets.QApplication.instance()
    if app is None:
        return LIGHT
    p = app.palette()

    def hexof(role, fallback="#000000"):
        colour = p.color(role)
        return colour.name() if colour.isValid() else fallback

    return {
        "ground": hexof(QtGui.QPalette.Window),
        "surface": hexof(QtGui.QPalette.Base),
        "soft": hexof(QtGui.QPalette.AlternateBase),
        "hover": hexof(QtGui.QPalette.Light),
        "line": hexof(QtGui.QPalette.Mid),
        "line_strong": hexof(QtGui.QPalette.Dark),
        "well": hexof(QtGui.QPalette.Mid),
        "text": hexof(QtGui.QPalette.WindowText),
        "muted": hexof(QtGui.QPalette.PlaceholderText),
        "accent": hexof(QtGui.QPalette.Highlight),
        "accent_hover": hexof(QtGui.QPalette.Highlight),
        "accent_soft": hexof(QtGui.QPalette.AlternateBase),
        "accent_text": hexof(QtGui.QPalette.HighlightedText),
        "danger": LIGHT["danger"],
        "danger_soft": hexof(QtGui.QPalette.AlternateBase),
        "warn": LIGHT["warn"],
        "selection": hexof(QtGui.QPalette.Highlight),
        "disabled": p.color(QtGui.QPalette.Disabled, QtGui.QPalette.Text).name(),
        "icon": hexof(QtGui.QPalette.ButtonText),
    }
