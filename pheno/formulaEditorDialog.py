"""Node-graph formula editor dialog.

Left: the canvas (drag ports to wire, right-click to add nodes, drag
nodes to arrange). Right: live compilation - every output node becomes a
trait, with its generated expression and sample value, plus a plain-text
fallback for people who prefer typing.

Saving writes the same .phenoFormula.json as the table editor, with the
graph layout embedded so re-opening restores the arrangement.
"""

import json
import os

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from pheno.formulaCanvas import GraphCanvas
from pheno.formulaGraph import FormulaGraph
from pheno.formulaTraits import (
    BASE_VARIABLES, FORMULA_SUFFIX, FormulaError, FormulaTrait,
    load_traits, save_traits,
)
from pheno.i18n import tr
from pheno import theme


class NodeFormulaEditorDialog(QtWidgets.QDialog):
    """Build formula traits by wiring nodes instead of typing."""

    def __init__(self, parent=None, graph=None, scale_unit=""):
        super().__init__(
            parent,
            Qt.Dialog | Qt.WindowTitleHint | Qt.WindowSystemMenuHint
            | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint,
        )
        self.setWindowTitle(tr("Formula Traits Editor (Graph)"))
        self.resize(1100, 700)
        self.savedPath = None
        self.scaleUnit = scale_unit or ""

        self.graph = graph or FormulaGraph().makeSample()

        layout = QtWidgets.QVBoxLayout(self)

        # --- toolbar ----------------------------------------------------
        bar = QtWidgets.QHBoxLayout()
        help_label = QtWidgets.QLabel(tr(
            "Drag from a node's right port to another node's left port to "
            "wire. Right-click empty space to add nodes; right-click a node "
            "to rename or delete it. Click a wire to detach it."))
        help_label.setWordWrap(True)
        help_label.setStyleSheet(f"color: {theme.palette()['muted']};")
        bar.addWidget(help_label, 1)
        reset_btn = QtWidgets.QPushButton(tr("Reset to Sample"))
        reset_btn.clicked.connect(self.resetSample)
        bar.addWidget(reset_btn)
        layout.addLayout(bar)

        # --- canvas + side panel -----------------------------------------
        split = QtWidgets.QSplitter(Qt.Horizontal)

        self.canvas = GraphCanvas(self.graph, scale_unit=self.scaleUnit)
        self.canvas.graphChanged.connect(self.refreshPreview)
        view = QtWidgets.QGraphicsView(self.canvas)
        view.setRenderHint(QtGui.QPainter.Antialiasing)
        view.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        split.addWidget(view)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)

        side_layout.addWidget(QtWidgets.QLabel(tr("Compiled traits:")))
        self.resultList = QtWidgets.QListWidget()
        self.resultList.setWordWrap(True)
        side_layout.addWidget(self.resultList, 2)

        self.errorLabel = QtWidgets.QLabel("")
        self.errorLabel.setWordWrap(True)
        self.errorLabel.setStyleSheet(f"color: {theme.palette()['danger']};")
        side_layout.addWidget(self.errorLabel)

        side_layout.addStretch()
        self.previewLabel = QtWidgets.QLabel("")
        self.previewLabel.setWordWrap(True)
        self.previewLabel.setStyleSheet(
            f"font-family: monospace; background: {theme.palette()['soft']}; padding: 4px;")
        side_layout.addWidget(self.previewLabel)

        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, 1)

        # --- bottom -------------------------------------------------------
        bottom = QtWidgets.QHBoxLayout()
        cancel_btn = QtWidgets.QPushButton(tr("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        save_btn = QtWidgets.QPushButton(tr("Save As..."))
        save_btn.setDefault(True)
        save_btn.clicked.connect(self.saveFile)
        bottom.addStretch()
        bottom.addWidget(cancel_btn)
        bottom.addWidget(save_btn)
        layout.addLayout(bottom)

        self.refreshPreview()

    # ------------------------------------------------------------------
    def resetSample(self):
        self.graph = FormulaGraph().makeSample()
        self.canvas = GraphCanvas(self.graph, scale_unit=self.scaleUnit)
        self.canvas.graphChanged.connect(self.refreshPreview)
        # Replace the view's scene in place.
        view = self.findChild(QtWidgets.QGraphicsView)
        view.setScene(self.canvas)
        self.refreshPreview()

    def refreshPreview(self):
        """Compile the graph and show traits / errors.

        Values are only shown when they come from a really selected object;
        with no selection the rows show the expression alone plus a hint,
        rather than numbers of mysterious origin.
        """
        self.resultList.clear()
        try:
            traits = self.graph.compile()
        except FormulaError as error:
            self.errorLabel.setText(str(error))
            self.previewLabel.clear()
            self._traits = []
            return
        self.errorLabel.clear()
        self._traits = traits

        values, has_real, source_label = self._previewValues()
        for trait in traits:
            if has_real:
                value = trait.evaluate(values)
                self.resultList.addItem(
                    "%s = %s\n  = %.4g" % (trait.name, trait.expression, value))
            else:
                self.resultList.addItem(
                    "%s = %s" % (trait.name, trait.expression))
        self.previewLabel.setText(source_label)
        if not traits:
            self.errorLabel.setText(tr(
                "Add an Output node and connect something to it."))

    def _previewValues(self):
        """(base values, has_real_values, human label of where they came from)."""
        # Dialogs are top-level widgets, so window() stops at the dialog
        # itself; walk parentWidget() up to the MainWindow that owns canvas.
        window = None
        widget = self
        while widget is not None:
            if hasattr(widget, "canvas") and hasattr(widget, "shapeList"):
                window = widget
                break
            widget = widget.parentWidget()

        shape = None
        if window is not None:
            shapes = window.shapeList.get_selected_shapes()
            shape = shapes[0] if shapes else None

        if shape is not None and shape.mask is not None \
                and getattr(window, "original_image", None) is not None:
            try:
                from pheno.calcPheno import PhenomicsCalculator
                from pheno.lib import qimage_to_cv2
                import numpy as np
                pixmap = window.canvas.pixmap
                scale = window.canvas.scale_per_pixel
                # Base values always come from the built-in measurement set,
                # not the enabled-algorithm registry: unchecking DEFAULT
                # hides its columns but formulas still need its numbers, and
                # previewing another algorithm's columns would mislabel them.
                rgbImg = qimage_to_cv2(pixmap.toImage())
                mask = (shape.mask > 0).astype(np.uint8)
                values = list(PhenomicsCalculator.default_algorithm(
                    rgbImg, mask, scale))
                label = tr('Preview from "{name}"',
                           name=shape.label or "object")
                return values, True, label
            except Exception:
                pass  # fall through to no-value mode

        return None, False, tr(
            "Select an object in the Objects panel to preview real values.")

    def saveFile(self):
        if not getattr(self, "_traits", None):
            QtWidgets.QMessageBox.warning(
                self, tr("Save Formula Traits"),
                tr("Add at least one trait first."))
            return

        default_name = "myTraits" + FORMULA_SUFFIX
        algorithms_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "algorithms")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("Save Formula Traits"),
            os.path.join(algorithms_dir if os.path.isdir(algorithms_dir)
                         else "", default_name),
            "PhenoMe Formula Traits (*%s)" % FORMULA_SUFFIX)
        if not path:
            return
        if not path.endswith(FORMULA_SUFFIX):
            path += FORMULA_SUFFIX

        # Units are derived per-image at evaluation time (they depend on the
        # current scale and the expression's dimension), so the file stores
        # expressions only; the graph layout rides along so reopening
        # restores the arrangement.
        document = {
            "format": "phenome-formula-traits",
            "version": 1,
            "traits": [
                {"name": trait.name, "expression": trait.expression,
                 "unit": "", "description": trait.description}
                for trait in self._traits
            ],
            "graph": self.graph.to_dict(),
        }
        temp_path = path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
        except OSError as error:
            QtWidgets.QMessageBox.critical(
                self, tr("Save Formula Traits"),
                tr("Failed to save: {message}", message=str(error)))
            return
        self.savedPath = path
        self.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) \
                and self.canvas.focusItem() is None:
            self.canvas.deleteSelected()
            return
        super().keyPressEvent(event)
