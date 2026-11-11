"""A quiet status-bar icon that opens a system detail card on hover.

**Why an icon, not a gauge.** A permanent gauge in the status bar competes with the
status text for the eye and tells one number nobody asked about. The person who wants
numbers is already reaching for it: hover the icon and a card rises with the whole
picture -- CPU, RAM, GPU utilisation and the VRAM breakdown.

**Why the driver figure for VRAM, not PyTorch's.** ``get_memory_info()`` reports
``used``/``free`` from ``mem_get_info``, which includes the CUDA context and *other
applications*. That is the number that answers "will the next allocation fit";
``torch.cuda.memory_allocated`` only counts this process's tensors, so a 12 GB card
with a browser open can read 3.5 GB while only 4 GB is really available.

**Why no pin.** A pinned card would have to live somewhere; everything it could cover
-- the canvas, a dock -- is more important than it is. The readout is one hover away.

**Where it goes.** The right-hand side of the status bar, via ``addPermanentWidget``.
``showMessage()`` writes to the left and never overwrites a permanent widget, so the
transient status text and this icon cannot fight.

The numbers were already being computed and thrown away: ``print_memory_status()``
runs on SAM init and on every image change, writing to stdout, which a GUI user never
sees and a packaged .exe does not have at all.
"""

from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets

from pheno import theme
from pheno.i18n import tr

#: Frequent enough to watch a batch's footprint climb while the card is open.
#: Deliberately not ``nvidia-smi``: spawning a process every tick would cost more
#: than everything else in this module put together.
REFRESH_MS = 2000

#: Where a reading stops being quiet. Measured against driver usage, so these are
#: about whether the next allocation fits, not about this application's own tensors.
AMBER_PERCENT = 70.0
RED_PERCENT = 90.0


def _system_stats() -> dict:
    """CPU / RAM / GPU-utilisation readings for the detail card.

    Every probe is optional: psutil is declared in requirements but a packaged
    build could still lack it, and ``torch.cuda.utilization`` exists only where a
    CUDA context is already up. A missing row just reads as blank.
    """
    stats: dict = {}
    try:
        import psutil

        stats["cpu"] = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        stats["ram_used"] = vm.used / float(2 ** 30)
        stats["ram_total"] = vm.total / float(2 ** 30)
    except Exception:  # noqa: BLE001 - an absent probe must not break the readout
        pass
    try:
        import torch

        if torch.cuda.is_available():
            stats["gpu"] = float(torch.cuda.utilization())
    except Exception:  # noqa: BLE001
        pass
    return stats


def summary(info: dict | None) -> dict | None:
    """Reduce a ``get_memory_info()`` reading to the numbers a readout needs.

    Pure, and free of any translation, so it can be tested without a GPU and without a
    language. ``None`` means there is no usable CUDA device -- the card then shows the
    VRAM row blank, since CPU-only machines still have a CPU and RAM worth reporting.
    """
    if not info:
        return None

    used = float(info.get("used_gb") or 0.0)
    total = float(info.get("total_gb") or 0.0)
    free = float(info.get("free_gb") or 0.0)
    percent = float(info.get("usage_percent") or 0.0)
    tensors = float(info.get("allocated_gb") or 0.0)
    pool = float(info.get("cached_gb") or 0.0)

    if percent >= RED_PERCENT:
        severity = "danger"
    elif percent >= AMBER_PERCENT:
        severity = "warn"
    else:
        severity = "ok"

    return {
        "percent": percent,
        "used_gb": used,
        "total_gb": total,
        "free_gb": free,
        "tensors_gb": tensors,
        "pool_gb": pool,
        # Driver usage minus PyTorch's pool: the context, and whatever else on the
        # machine is holding memory. Worth showing, because it is usually the answer to
        # "why is so little free when this application is barely using any".
        "elsewhere_gb": max(0.0, used - pool),
        "severity": severity,
    }


def _vram_reading() -> dict | None:
    from pheno.cuda import memory_manager

    try:
        return summary(memory_manager.get_memory_info())
    except Exception:  # noqa: BLE001 - a driver hiccup must not take the window down
        return None


class GpuMonitor(QtWidgets.QWidget):
    """An 18px monochrome bars icon for the right corner of the status bar.

    The icon is drawn, not loaded: three ascending bars in the palette's muted
    colour, so it stays monochrome on every theme including classic, where the
    toolbar's original coloured icons come back. It shows no number -- the card
    is the readout; the icon is only the way in.
    """

    ICON_SIZE = 18

    def __init__(self, parent=None):
        super().__init__(parent)
        self._card = None

        self._icon = QtWidgets.QLabel()
        self._icon.setFixedSize(self.ICON_SIZE, self.ICON_SIZE)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 8, 0)
        layout.addWidget(self._icon)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.setToolTip(tr("System resources"))

        # Hover raises the card after a short dwell; leaving either widget starts
        # a slightly longer fuse, so the hand can travel between them.
        self._show_timer = QtCore.QTimer(self, singleShot=True)
        self._show_timer.setInterval(350)
        self._show_timer.timeout.connect(self._show_card)
        self._hide_timer = QtCore.QTimer(self, singleShot=True)
        self._hide_timer.setInterval(250)
        self._hide_timer.timeout.connect(self._hide_card)
        self._icon.installEventFilter(self)

        self._busy = 0

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

        self.applyTheme()

    # -- reading ----------------------------------------------------------

    def refresh(self) -> None:
        """Keep an open card current. Cheap enough to run on a timer."""
        if self._card is not None and self._card.isVisible():
            self._card.sync(_vram_reading(), _system_stats())

    # -- card --------------------------------------------------------------

    def eventFilter(self, watched, event):
        """The icon label shares the monitor's hover region for the card."""
        if event.type() == QtCore.QEvent.Enter:
            self.enterEvent(event)
        elif event.type() == QtCore.QEvent.Leave:
            self.leaveEvent(event)
        return super().eventFilter(watched, event)

    def enterEvent(self, event):
        self._hide_timer.stop()
        self._show_timer.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._show_timer.stop()
        self._hide_timer.start()
        super().leaveEvent(event)

    def _show_card(self):
        if self._card is None:
            self._card = _SystemCard(self)
        self._card.sync(_vram_reading(), _system_stats())
        self._card.reposition()
        self._card.show()
        self._card.raise_()

    def _hide_card(self):
        if self._card is not None:
            self._card.hide()

    def card_hover(self, inside: bool) -> None:
        """The card forwards its own Enter/Leave so dwell there keeps it up."""
        if inside:
            self._hide_timer.stop()
        else:
            self._hide_timer.start()

    def set_busy(self, busy: bool) -> None:
        """Long-running work raises the card on its own -- nested callers count.

        A modal progress dialog makes the icon unreachable exactly when the
        numbers matter most, so busy operations show the card themselves.
        """
        self._busy = max(0, self._busy + (1 if busy else -1))
        if self._busy:
            self._hide_timer.stop()
            self._show_timer.stop()
            self._show_card()
        elif (
            self._card is not None and self._card.isVisible()
            and not self._card.underMouse() and not self.underMouse()
        ):
            self._hide_card()

    # -- appearance -------------------------------------------------------

    def applyTheme(self) -> None:
        """Re-colour from the active palette. Called by the window's theme hook."""
        self._draw_icon()
        if self._card is not None and self._card.isVisible():
            self._card.apply_theme()

    def _draw_icon(self) -> None:
        """Three ascending bars -- the least colourful way to say "usage"."""
        size = self.ICON_SIZE
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pm)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        # Clearly lighter than muted text: a quiet black-grey that still tracks
        # each theme's muted tone.
        painter.setBrush(QtGui.QColor(theme.palette()["muted"]).lighter(150))
        for i, frac in enumerate((0.4, 0.7, 1.0)):
            w = max(2.0, size * 0.2)
            x = 1.0 + i * (w + size * 0.09)
            h = size * frac
            painter.drawRoundedRect(
                QtCore.QRectF(x, size - h - 1.0, w, h), 1.0, 1.0
            )
        painter.end()
        self._icon.setPixmap(pm)


class _SystemCard(QtWidgets.QFrame):
    """Hover popup with the wider system picture: CPU / RAM / GPU / VRAM.

    A plain child of the main window, not a top-level popup: no window flags to
    fight, it positions itself above the status bar at the right edge, and it
    stays up only while the pointer is over it or the trigger icon.
    """

    WIDTH = 284

    #: (row key, English source) -- names go through tr() so they follow a
    #: runtime language switch, which this card's one-shot constructor cannot.
    ROWS = (("cpu", "CPU"), ("ram", "RAM"), ("gpu", "GPU"), ("vram", "VRAM"))

    def __init__(self, monitor):
        super().__init__(monitor.window())
        self._monitor = monitor
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setAutoFillBackground(True)
        # Hover affordance: appearing must not pull focus away from the canvas.
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        self._title = QtWidgets.QLabel()
        f = self._title.font()
        f.setBold(True)
        self._title.setFont(f)
        layout.addWidget(self._title)

        self._rows = {}
        for key, _name in self.ROWS:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(6)
            name_label = QtWidgets.QLabel()
            name_label.setFixedWidth(42)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            value = QtWidgets.QLabel("—")
            # Fit the widest plausible reading ("999.9 / 999 GB") in this font,
            # whatever the DPI -- a fixed pixel width clips at larger fonts.
            value.setFixedWidth(
                self.fontMetrics().horizontalAdvance("999.9 / 999 GB") + 8)
            value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            row.addWidget(name_label)
            row.addWidget(bar, 1)
            row.addWidget(value)
            layout.addLayout(row)
            self._rows[key] = (bar, value, name_label)

        self.retranslate()
        self.setMinimumWidth(self.WIDTH)
        self.adjustSize()
        self.apply_theme()

    def retranslate(self):
        """Re-apply tr() to every static label. Runs on every sync, so a
        language switched after the card exists still lands on the next tick."""
        self._title.setText(tr("System"))
        for key, name in self.ROWS:
            self._rows[key][2].setText(tr(name))

    # -- data -------------------------------------------------------------

    def sync(self, reading, stats):
        """Fill the rows. ``reading`` is summary()'s VRAM dict, ``stats`` the rest."""
        self.retranslate()
        cpu = stats.get("cpu")
        self._set_row("cpu", cpu, None if cpu is None else f"{cpu:.0f}%")

        ram_used = stats.get("ram_used")
        ram_total = stats.get("ram_total")
        if ram_used is not None and ram_total:
            self._set_row(
                "ram", 100.0 * ram_used / ram_total,
                f"{ram_used:.1f} / {ram_total:.0f} GB")
        else:
            self._set_row("ram", None, None)

        gpu = stats.get("gpu")
        self._set_row("gpu", gpu, None if gpu is None else f"{gpu:.0f}%")

        if reading:
            self._set_row(
                "vram", reading["percent"],
                f"{reading['used_gb']:.1f} / {reading['total_gb']:.0f} GB")
        else:
            self._set_row("vram", None, None)

    def _set_row(self, key, percent, text):
        bar, value, _name = self._rows[key]
        if percent is None:
            bar.setValue(0)
            value.setText("—")
        else:
            bar.setValue(int(round(percent)))
            value.setText(text if text is not None else f"{percent:.0f}%")

    # -- position / hover --------------------------------------------------

    def reposition(self):
        parent = self.parentWidget()
        if parent is None:
            return
        statusbar_h = 0
        window = self._monitor.window()
        if isinstance(window, QtWidgets.QMainWindow) and window.statusBar() is not None:
            statusbar_h = window.statusBar().height()
        self.move(
            parent.width() - self.width() - 12,
            parent.height() - self.height() - statusbar_h - 8,
        )

    def enterEvent(self, event):
        self._monitor.card_hover(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._monitor.card_hover(False)
        super().leaveEvent(event)

    def apply_theme(self) -> None:
        palette = theme.palette()
        self.setStyleSheet(
            f"_SystemCard {{ background: {palette['surface']};"
            f" border: 1px solid {palette['line_strong']}; border-radius: 6px; }}"
        )
        for _bar, _value, name_label in self._rows.values():
            name_label.setStyleSheet(f"QLabel {{ color: {palette['muted']}; }}")
