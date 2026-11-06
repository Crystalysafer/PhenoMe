"""Runtime segmentation-model / compute-device switcher.

The status-bar button in ``app.py`` opens :class:`ModelSelectDialog`. It mirrors
the splash screen's model/device row but resolves against the *running* session:
applying the choice rebuilds ``canvas.sam`` in a worker and re-embeds the open
image, so the file list, objects and traits table all survive the switch.
"""
from qtpy import QtWidgets
from qtpy.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout)

from pheno.i18n import tr


def gpu_status_text():
    """One-line GPU availability summary for the dialog.

    ``cuda_supported`` already folds in the session blacklist
    (``_cuda_disabled_reason``), arch-mismatch probes and plain absence of a
    CUDA device, so its reason string is the honest answer for "what happens
    if I leave this on Auto".
    """
    try:
        from pheno.cuda import cuda_supported, memory_manager
        ok, why = cuda_supported()
        if ok:
            import torch
            name = torch.cuda.get_device_name(0)
            info = memory_manager.get_memory_info()
            tail = "  ({0:.1f} GB free)".format(info['free_gb']) if info else ""
            return "✓ {0}{1}".format(name, tail)
        return tr("✗ Not usable -- Auto will run on CPU ({reason})",
                  reason=why)
    except Exception as exc:
        return tr("✗ Detection failed: {error}", error=exc)


class ModelSelectDialog(QDialog):
    """Pick a SAM backend and compute device; applied without a restart."""

    def __init__(self, parent=None, model_key=None, device_pref="auto"):
        super().__init__(parent)
        self.setWindowTitle(tr("Segmentation Model"))
        self.setModal(True)

        from pheno.SAM import available_segmenters, segmenter_label

        form = QFormLayout()
        form.setSpacing(8)

        self.model_combo = QComboBox()
        for key in available_segmenters():
            self.model_combo.addItem(segmenter_label(key), key)
        idx = self.model_combo.findData(model_key)
        self.model_combo.setCurrentIndex(max(idx, 0))
        form.addRow(tr("Segmentation model:"), self.model_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItem(tr("Auto (GPU when available)"), "auto")
        self.device_combo.addItem(tr("CPU only"), "cpu")
        idx = self.device_combo.findData(str(device_pref))
        self.device_combo.setCurrentIndex(max(idx, 0))
        form.addRow(tr("Compute:"), self.device_combo)

        gpu_label = QLabel(gpu_status_text())
        gpu_label.setWordWrap(True)
        gpu_label.setStyleSheet("color: #666666; font-size: 11px;")
        form.addRow(tr("GPU status:"), gpu_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        note = QLabel(tr(
            "The open image and its objects are kept; the picture is "
            "re-analysed with the new model."))
        note.setWordWrap(True)
        note.setStyleSheet("color: #757575; font-size: 11px;")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    @property
    def model_key(self):
        return self.model_combo.currentData()

    @property
    def device_pref(self):
        return self.device_combo.currentData()
