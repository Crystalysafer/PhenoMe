import re

from qtpy import QtCore
from qtpy import QtWidgets


# Mnemonic / shortcut suffixes that action texts carry for the menu bar, e.g.
# "打开(&O)", "Open (Ctrl+O)", "Delete (Del)". Menus need them, toolbar buttons
# do not: the caption sits under a 60px icon where the extra "(Ctrl+Shift+S)"
# just widens every button. Only key-looking contents are stripped, so a real
# parenthetical caption would survive.
_SHORTCUT_SUFFIX = re.compile(
    r"\s*\(\s*&?(?:(?:Ctrl|Alt|Shift|Meta)\+)*"
    r"(?:[A-Za-z0-9]|Del|Delete|Esc|Escape|Space|Tab|Ins|Insert|F\d{1,2})"
    r"\s*\)\s*$",
    re.IGNORECASE,
)


def toolbar_label(text):
    """Action text without its mnemonic markers or shortcut hint."""
    label = _SHORTCUT_SUFFIX.sub("", str(text or "")).strip()
    # Drop the Alt mnemonic marker but keep a literal ampersand ("&&" in Qt).
    return label.replace("&&", "\x00").replace("&", "").replace("\x00", "&")


class _LabelSync(QtCore.QObject):
    """Keeps one toolbar button caption free of mnemonics and shortcut hints.

    ``QToolButton::setDefaultAction`` copies the action text verbatim and keeps
    it in sync through ``QAction::changed``, so the caption has to be cleaned
    again after every language switch. This lives in a small QObject parented to
    the button on purpose: a plain lambda would stay connected to the action
    after the button is destroyed and fire on a dead C++ object during shutdown,
    which PyQt turns into a fatal error.
    """

    def __init__(self, button, action):
        super(_LabelSync, self).__init__(button)
        self._button = button
        self._action = action
        action.changed.connect(self.apply)
        self.apply()

    def apply(self):
        try:
            label = toolbar_label(self._action.text())
            if self._button.text() != label:
                self._button.setText(label)
        except RuntimeError:
            # The button was already deleted on the C++ side.
            pass


class ToolBar(QtWidgets.QToolBar):
    def __init__(self, title):
        super(ToolBar, self).__init__(title)
        layout = self.layout()
        m = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*m)
        self.setContentsMargins(*m)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)

    def addAction(self, action):
        if isinstance(action, QtWidgets.QWidgetAction):
            return super(ToolBar, self).addAction(action)
        btn = QtWidgets.QToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        # Connected after setDefaultAction, so this runs last on every change
        # and the cleaned caption wins.
        _LabelSync(btn, action)
        self.addWidget(btn)

        # center align
        for i in range(self.layout().count()):
            if isinstance(self.layout().itemAt(i).widget(), QtWidgets.QToolButton):
                self.layout().itemAt(i).setAlignment(QtCore.Qt.AlignCenter)



class ToolButton(QtWidgets.QToolButton):

    """ToolBar companion class which ensures all buttons have the same size."""

    minSize = (60, 60)

    def minimumSizeHint(self):
        ms = super(ToolButton, self).minimumSizeHint()
        w1, h1 = ms.width(), ms.height()
        w2, h2 = self.minSize
        self.minSize = max(w1, w2), max(h1, h2)
        return QtCore.QSize(*self.minSize)
