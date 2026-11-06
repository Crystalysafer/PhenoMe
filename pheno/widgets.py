import sys
import os
import csv
import importlib.util
from pheno.calcPheno import PhenomicsCalculator
from pheno.i18n import tr
from pheno import theme
from qtpy import QtWidgets, QtGui, QtCore
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from qtpy.QtWidgets import *

class ShapeQListWidget(QtWidgets.QListWidget):
    focusOut = Signal()
    deleteRequested = Signal()

    def __init__(self, *args, **kwargs):
        super(ShapeQListWidget, self).__init__(*args, **kwargs)
        self.canvas = None
        self.itemsToShapes = []

        # Highlight selection drives batch delete; the check boxes stay
        # independent and keep driving Calculate / Save.
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.selectAllCheckbox = QtWidgets.QCheckBox("Select All")
        self.selectAllCheckbox.setCheckState(Qt.Checked)
        self.selectAllCheckbox.stateChanged.connect(self.toggle_all)

        self.ignore_focus_event = False

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.deleteRequested.emit()
            return
        super(ShapeQListWidget, self).keyPressEvent(event)

    def check_label(self, name, shape=None):
        """Reject empty names and names already used by another object."""
        text = str(name or "").strip()
        if not text:
            return False
        for item, shape_ in self.itemsToShapes:
            if shape is not None and shape_ is shape:
                continue
            if item.text() == text:
                return False
        return True

    def get_shape_from_item(self, item):
        for index, (item_, shape) in enumerate(self.itemsToShapes):
            if item_ is item:
                return shape

    def get_item_from_shape(self, shape):
        for index, (item, shape_) in enumerate(self.itemsToShapes):
            if shape_ is shape:
                return item

    def remove_shape(self, shape):
        """Drop the row bound to ``shape``; returns True when one was removed."""
        item = self.get_item_from_shape(shape)
        if item is None:
            return False
        self.takeItem(self.row(item))
        self.itemsToShapes = [
            pair for pair in self.itemsToShapes if pair[1] is not shape
        ]
        return True

    def get_selected_shapes(self):
        """Shapes whose rows are highlighted (Ctrl/Shift multi-selection)."""
        shapes = []
        for item in self.selectedItems():
            shape = self.get_shape_from_item(item)
            if shape is not None:
                shapes.append(shape)
        return shapes

    def clear(self):
        super(ShapeQListWidget, self).clear()
        self.itemsToShapes = []

    @property
    def shapes(self):
        shapes = []
        for i in range(self.count()):
            item = self.item(i)
            shape = self.get_shape_from_item(item)
            shapes.append(shape)
        return shapes
    
    def toggle_all(self):
        """ 点击总复选框时控制所有项的选中状态 """
        check_state = self.selectAllCheckbox.isChecked()
        for item, shape in self.itemsToShapes:
            item.setCheckState(Qt.Checked if check_state else Qt.Unchecked)

    def get_checked_shapes(self):
        """ 获取所有选中的形状 """
        checked_shapes = []
        for item, shape in self.itemsToShapes:
            if item.checkState() == Qt.Checked:
                checked_shapes.append(shape)
        return checked_shapes
    
    def focusOutEvent(self, event):
        super(ShapeQListWidget, self).focusOutEvent(event)
        if not self.ignore_focus_event:
            self.focusOut.emit()
    
class FileQListWidget(QtWidgets.QListWidget):
    focusOut = Signal()

    def __init__(self, *args, **kwargs):
        super(FileQListWidget, self).__init__(*args, **kwargs)
        self.canvas = None
        self.itemsToShapes = []

        self.selectAllCheckbox = QtWidgets.QCheckBox("Select All")
        self.selectAllCheckbox.setCheckState(Qt.Unchecked)
        self.selectAllCheckbox.stateChanged.connect(self.toggle_all)

        self.ignore_focus_event = False

    def clear(self):
        super(FileQListWidget, self).clear()

    def toggle_all(self):
        """ 点击总复选框时控制所有项的选中状态 """
        check_state = self.selectAllCheckbox.isChecked()
        for i in range(self.count()):
            item = self.item(i)
            item.setCheckState(Qt.Checked if check_state else Qt.Unchecked)

    def focusOutEvent(self, e):
        return super().focusOutEvent(e)



class PhenomicTableModel(QAbstractTableModel):
    """Traits table.

    Each row optionally carries a hidden key (the ``Shape.uid`` it was computed
    from) so the table can follow renames and deletions instead of silently
    showing data for objects that no longer exist.
    """

    def __init__(self, data=[], header=[]):
        super().__init__()
        # 默认数据
        self._data = data
        self._row_keys = [None] * len(data)
        self._horizontal_headers = header  # 表头内容
        self._stale = False
        # Row currently highlighted because its object is selected elsewhere.
        self._highlight_key = None

    def isEmpty(self):
        return len(self._data) == 0

    def isStale(self):
        """True when the rows no longer match the current set of objects."""
        return self._stale

    def setStale(self, stale=True):
        self._stale = bool(stale) and not self.isEmpty()
        return self._stale

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)  # 行数

    def columnCount(self, parent=QModelIndex()):
        return len(self._horizontal_headers)  # 列数

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return self._data[index.row()][index.column()]  # 返回对应单元格的数据
            if role == Qt.BackgroundRole and self._row_keys:
                # Tint the row whose object is selected (traits <-> objects
                # two-way highlight).
                if self._row_keys[index.row()] == self._highlight_key \
                        and self._highlight_key is not None:
                    return QtGui.QBrush(QtGui.QColor(185, 215, 255))
        return None

    def keyAt(self, row):
        """Hidden uid of a row, or None."""
        if 0 <= row < len(self._row_keys):
            return self._row_keys[row]
        return None

    def rowOfKey(self, key):
        """First row index carrying ``key``, or -1."""
        if key is None:
            return -1
        try:
            return self._row_keys.index(key)
        except ValueError:
            return -1

    def setHighlightKey(self, key):
        """Highlight every row belonging to ``key`` (None clears)."""
        if key == self._highlight_key:
            return
        self._highlight_key = key
        # Whole-table refresh: keys may sit on any row.
        if self._data:
            top_left = self.index(0, 0)
            bottom_right = self.index(len(self._data) - 1,
                                      len(self._horizontal_headers) - 1)
            self.dataChanged.emit(top_left, bottom_right,
                                  [Qt.BackgroundRole])

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._horizontal_headers[section]  # 表头内容
            else:
                return str(section + 1)  # 行号
        return None

    def sort(self, column, order=Qt.AscendingOrder):
        """根据列索引排序数据"""
        # 排序列的数据
        self.layoutAboutToBeChanged.emit()  # 通知视图布局即将改变

        # 行与隐藏键必须一起排序，否则后续的改名/删除会作用到错误的行。
        order_desc = (order == Qt.DescendingOrder)
        paired = sorted(
            zip(self._data, self._row_keys),
            key=lambda pair: pair[0][column],
            reverse=order_desc,
        )
        self._data = [row for row, _key in paired]
        self._row_keys = [key for _row, key in paired]

        self.layoutChanged.emit()  # 通知视图数据已经更新

    def updateData(self, data, header, keys=None):
        self.beginResetModel()
        self._data = list(data)
        self._row_keys = list(keys) if keys is not None else [None] * len(self._data)
        self._horizontal_headers = header
        self._stale = False
        self.endResetModel()

    def renameRow(self, key, name):
        """Follow an object rename. Returns True when a row was updated."""
        if key is None:
            return False
        changed = False
        for row_index, row_key in enumerate(self._row_keys):
            if row_key == key and self._data[row_index]:
                self._data[row_index][0] = name
                index = self.index(row_index, 0)
                self.dataChanged.emit(index, index)
                changed = True
        return changed

    def removeRowsByKeys(self, keys):
        """Drop the rows belonging to deleted objects. Returns removed count."""
        wanted = {key for key in keys if key is not None}
        if not wanted:
            return 0
        keep = [
            (row, key)
            for row, key in zip(self._data, self._row_keys)
            if key not in wanted
        ]
        removed = len(self._data) - len(keep)
        if not removed:
            return 0
        self.beginResetModel()
        self._data = [row for row, _key in keep]
        self._row_keys = [key for _row, key in keep]
        if self.isEmpty():
            self._stale = False
        self.endResetModel()
        return removed

    def to_csv_text(self):
        """The table as CSV text, escaped by the same writer ``export`` uses.

        Kept in one place because the escaping is the part that matters: object names
        are free text and may contain commas or quotes, which a hand-built `','.join`
        would split into extra columns.
        """
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(self._horizontal_headers)
        writer.writerows(self._data)
        return buffer.getvalue()

    def export(self, filename):
        # utf-8-sig 让 Excel 直接双击打开中文名字也不乱码。
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            f.write(self.to_csv_text())

    def getOptimalColumnWidth(self, column_index, max_width=200, min_width=80):
        """计算列的最优宽度"""
        if column_index >= len(self._horizontal_headers):
            return min_width
            
        # 根据标题长度计算基础宽度
        header_text = self._horizontal_headers[column_index]
        base_width = len(header_text) * 8 + 20  # 8像素每字符 + 20像素边距
        
        # 根据数据内容调整宽度
        if self._data:
            max_data_length = 0
            for row in self._data:
                if column_index < len(row):
                    data_text = str(row[column_index])
                    max_data_length = max(max_data_length, len(data_text))
            
            data_width = max_data_length * 8 + 20
            base_width = max(base_width, data_width)
        
        # 限制在最小和最大宽度之间
        return max(min_width, min(base_width, max_width))

    def setupTableView(self, table_view):
        """为表格视图设置合理的列宽模式"""
        if not self._horizontal_headers:
            return
            
        column_count = len(self._horizontal_headers)
        
        # 设置表头高度以确保文字完整显示
        table_view.horizontalHeader().setDefaultSectionSize(120)  # 设置默认列宽
        table_view.horizontalHeader().setMinimumSectionSize(80)   # 设置最小列宽
        table_view.horizontalHeader().setFixedHeight(35)          # 增加表头高度
        
        # 如果列数较少（≤6列），使用拉伸模式
        if column_count <= 6:
            table_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            table_view.horizontalHeader().setStretchLastSection(True)
        else:
            # 列数较多时，使用固定宽度模式并启用水平滚动
            table_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
            table_view.horizontalHeader().setStretchLastSection(False)
            
            # 为每列设置最优宽度
            for i in range(column_count):
                optimal_width = self.getOptimalColumnWidth(i)
                table_view.setColumnWidth(i, optimal_width)
            
            # 确保启用水平滚动条
            table_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
        # 设置垂直滚动条策略
        table_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)


class PhenomicComboBox(QtWidgets.QComboBox):
    def __init__(self, *args, **kwargs):
        super(PhenomicComboBox, self).__init__(*args, **kwargs)
        self.setFixedWidth(120)
        # self.insertItem(0, "Phenomic Pattern")
        # self.setCurrentIndex(0)
        # self.model().item(0).setEnabled(False)

        self.setVisible(False)


class TextPromptDialog(QDialog):
    """Ask the user which object the semantic detector should look for.

    A single editable combo box holds the target: the drop-down offers the
    presets and the same field accepts any custom text.

    Chinese is accepted: pheno/prompt_zh.py translates it before the prompt
    reaches Grounding DINO, so the presets follow the interface language instead
    of forcing everyone to think in English. Whatever is typed goes through the
    same translation, so a custom Chinese target works too.
    """

    BASE_TARGETS_EN = ("pod", "seed", "leaf", "flower", "label", "ruler")
    BASE_TARGETS_ZH = ("豆荚", "种子", "叶片", "花", "标签", "尺子")

    @classmethod
    def base_targets(cls):
        from pheno.i18n import get_language

        return cls.BASE_TARGETS_ZH if get_language() == "zh_CN" else cls.BASE_TARGETS_EN

    def __init__(self, parent=None, initial=None):
        # Build a complete, standard dialog frame instead of subtracting the
        # help-button hint from the default flags: mutating flags after the
        # dialog exists can leave the native title bar laid out for two caption
        # buttons while only the close button is drawn, which makes the "x"
        # look clipped.
        super().__init__(
            parent,
            Qt.Dialog
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowCloseButtonHint,
        )

        self.setWindowTitle(tr("Semantic Detect"))
        # Wide enough that the title text and the close button never crowd.
        self.setMinimumWidth(460)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        self.label = QLabel(tr("Detection target:"))
        main_layout.addWidget(self.label)

        self.promptCombo = QComboBox()
        self.promptCombo.setEditable(True)
        self.promptCombo.setInsertPolicy(QComboBox.NoInsert)
        self.promptCombo.addItems(self.base_targets())
        line_edit = self.promptCombo.lineEdit()
        line_edit.setPlaceholderText(tr("e.g. pod, seed, or any custom target (Chinese works)"))
        line_edit.setClearButtonEnabled(True)

        # A popup completer suggests presets without ever rewriting what the
        # user typed, so custom targets survive as entered.
        completer = self.promptCombo.completer()
        if completer is not None:
            completer.setCompletionMode(QCompleter.PopupCompletion)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
        main_layout.addWidget(self.promptCombo)

        self.hintLabel = QLabel(
            tr(
                "Choose a preset from the list or type any target yourself, "
                "for example ruler or color checker. Separate multiple targets "
                "with commas."
            )
        )
        self.hintLabel.setWordWrap(True)
        self.hintLabel.setStyleSheet(f"color: {theme.palette()['muted']}; font-size: 12px;")
        main_layout.addWidget(self.hintLabel)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.cancel_button = QPushButton(tr("Cancel"))
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button = QPushButton(tr("Confirm"))
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setDefault(True)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        self.promptCombo.setEditText(initial or self.base_targets()[0])
        line_edit.selectAll()
        self.promptCombo.setFocus()
        self.adjustSize()

    def get_prompt(self):
        """Return the prompt typed or selected by the user."""
        return self.promptCombo.currentText().strip()


def describe_batch_step(step):
    """Human-readable, translated one-line summary of a recorded batch step."""
    kind = step.get("type")
    if kind == "text":
        return tr('Semantic Detect: "{prompt}"', prompt=step.get("prompt", ""))
    if kind == "region":
        points = step.get("points") or []
        count = step.get("fix_sample_count")
        count_text = tr("auto") if count is None else str(count)
        return tr(
            "Region Detect: {n}-point region, count {count}",
            n=len(points),
            count=count_text,
        )
    if kind == "click":
        return tr(
            "Click: {count} prompt point(s)", count=len(step.get("points", []))
        )
    if kind == "green":
        return tr("Green Leaf Detect")
    if kind == "color_correction":
        return tr("Color Calibration")
    if kind == "ruler_detection":
        return tr("Ruler Detection")
    return str(kind)


class BatchStepsWidget(QtWidgets.QWidget):
    """Read-only list of the operations recorded for batch replay.

    Entries are never typed in: every row comes from an operation the user
    performed on the first image. Rows can be removed individually or cleared.
    """

    removeRequested = Signal(object)  # list of step indices
    clearRequested = Signal()
    loadRequested = Signal()
    saveRequested = Signal()
    exitRequested = Signal()

    # 批处理回放时，每个步骤的结果在屏幕上停留的默认秒数。分割结果原本一提交
    # 就立刻导出并加载下一张图，界面上几乎看不到，没法肉眼核对。
    DEFAULT_PREVIEW_DWELL_S = 1.0

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.titleLabel = QLabel(tr("Batch Steps"))
        self.titleLabel.setStyleSheet("font-weight: 600;")
        header.addWidget(self.titleLabel)
        header.addStretch()
        self.loadBtn = QToolButton()
        self.loadBtn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.loadBtn.setText(tr("Load Recipe"))
        self.loadBtn.clicked.connect(self.loadRequested.emit)
        self.saveBtn = QToolButton()
        self.saveBtn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.saveBtn.setText(tr("Save Recipe"))
        self.saveBtn.clicked.connect(self.saveRequested.emit)
        header.addWidget(self.loadBtn)
        header.addWidget(self.saveBtn)
        self.removeBtn = QToolButton()
        self.removeBtn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.removeBtn.setText(tr("Remove"))
        self.removeBtn.clicked.connect(self._emitRemove)
        self.clearBtn = QToolButton()
        self.clearBtn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.clearBtn.setText(tr("Clear"))
        self.clearBtn.clicked.connect(self.clearRequested.emit)
        header.addWidget(self.removeBtn)
        header.addWidget(self.clearBtn)
        layout.addLayout(header)

        self.stepList = QListWidget()
        self.stepList.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.stepList.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stepList.setMaximumHeight(140)
        self.stepList.itemSelectionChanged.connect(self._updateButtons)
        layout.addWidget(self.stepList)

        delayRow = QHBoxLayout()
        delayRow.setContentsMargins(0, 2, 0, 0)
        self.delayLabel = QLabel(tr("Preview dwell"))
        self.delaySpin = QDoubleSpinBox()
        self.delaySpin.setRange(0.0, 10.0)
        self.delaySpin.setSingleStep(0.5)
        self.delaySpin.setDecimals(1)
        self.delaySpin.setSuffix(tr(" s"))
        self.delaySpin.setValue(self.DEFAULT_PREVIEW_DWELL_S)
        self.delaySpin.setToolTip(tr(
            "How long each step's result stays on screen during batch replay, so "
            "the segmentation can be checked before the next image loads. "
            "0 continues as fast as possible."
        ))
        delayRow.addWidget(self.delayLabel)
        delayRow.addWidget(self.delaySpin)
        delayRow.addStretch()
        self.exitBtn = QToolButton()
        self.exitBtn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.exitBtn.setText(tr("Exit"))
        self.exitBtn.clicked.connect(self.exitRequested.emit)
        delayRow.addWidget(self.exitBtn)
        layout.addLayout(delayRow)

        self.emptyLabel = QLabel(
            tr("No steps recorded yet. Run an operation on this image to record it.")
        )
        self.emptyLabel.setWordWrap(True)
        self.emptyLabel.setStyleSheet(f"color: {theme.palette()['muted']}; font-size: 12px;")
        layout.addWidget(self.emptyLabel)

        self.setLayout(layout)
        self.refresh([])

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.stepList.hasFocus():
            self._emitRemove()
            return
        super().keyPressEvent(event)

    def _emitRemove(self):
        indices = [
            item.data(Qt.UserRole) for item in self.stepList.selectedItems()
        ]
        if indices:
            self.removeRequested.emit(sorted(indices))

    def _updateButtons(self):
        has_selection = bool(self.stepList.selectedItems())
        self.removeBtn.setEnabled(has_selection)
        self.clearBtn.setEnabled(self.stepList.count() > 0)

    def refresh(self, steps):
        """Rebuild the row list from the recorded steps."""
        self.stepList.clear()
        for index, step in enumerate(steps or []):
            item = QtWidgets.QListWidgetItem(
                "%d. %s" % (index + 1, describe_batch_step(step))
            )
            item.setData(Qt.UserRole, index)
            self.stepList.addItem(item)
        self.emptyLabel.setVisible(not steps)
        self._updateButtons()

    def previewDelayMs(self):
        """每个回放步骤的结果在屏幕上停留多久（毫秒）。

        批处理的工作线程按这个时长休眠，GUI 线程照常处理事件，所以分割结果
        能真正画出来供肉眼核对。0 表示不停留。
        """
        return int(round(self.delaySpin.value() * 1000))

    def retranslateUi(self, steps=None):
        self.titleLabel.setText(tr("Batch Steps"))
        self.loadBtn.setText(tr("Load Recipe"))
        self.saveBtn.setText(tr("Save Recipe"))
        self.removeBtn.setText(tr("Remove"))
        self.clearBtn.setText(tr("Clear"))
        self.exitBtn.setText(tr("Exit"))
        self.delayLabel.setText(tr("Preview dwell"))
        self.delaySpin.setSuffix(tr(" s"))
        self.delaySpin.setToolTip(tr(
            "How long each step's result stays on screen during batch replay, so "
            "the segmentation can be checked before the next image loads. "
            "0 continues as fast as possible."
        ))
        self.emptyLabel.setText(
            tr("No steps recorded yet. Run an operation on this image to record it.")
        )
        if steps is not None:
            self.refresh(steps)


class WaitingDialog(QtWidgets.QProgressDialog):
    def __init__(self, parent=None, text=None):
        super().__init__(text or tr("Processing..."), tr("Cancel"), 0, 0, parent)
        self.setWindowTitle(tr("Please Wait"))
        self.setWindowModality(Qt.ApplicationModal)  # 模态对话框
        self.setCancelButton(None)  # 取消按钮（可选）
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)  # 移除关闭按钮
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setMinimumDuration(500)  # 显示的最短时间，防止闪烁

        self.setFixedSize(300, 100)
        self.center_on_screen()

    # While a modal wait is up, the pointer cannot reach the status-bar monitor
    # icon -- so the monitor raises its detail card itself for the duration.
    def _monitor(self):
        window = self.parent()
        if window is None:
            return None
        from pheno.gpuMonitor import GpuMonitor
        return window.findChild(GpuMonitor)

    def showEvent(self, event):
        monitor = self._monitor()
        if monitor is not None:
            monitor.set_busy(True)
        super().showEvent(event)

    def hideEvent(self, event):
        monitor = self._monitor()
        if monitor is not None:
            monitor.set_busy(False)
        super().hideEvent(event)

    def center_on_screen(self):
        """使对话框居中"""
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        dialog_rect = self.frameGeometry()
        center_x = (screen.width() - dialog_rect.width()) // 2
        center_y = (screen.height() - dialog_rect.height()) // 2
        self.move(center_x, center_y)
    
    def close_dialog(self):

        self.accept()



class ManageAlgorithmDialog(QDialog):
    def __init__(self, custom_algorithms, custom_headers, imported_names, loaded_names, parent=None, on_save_callback=None):
        super().__init__(parent)
        self.on_save_callback = on_save_callback  # 回调函数，用于保存数据
        self.setWindowTitle(tr("Manage Custom Algorithms"))
        self.setFixedSize(900, 400)

        # 储存算法和 header 数据
        self.custom_algorithms = custom_algorithms
        self.custom_headers = custom_headers

        # 主垂直布局
        main_layout = QVBoxLayout()

        # === 左侧：已导入算法 ===
        self.imported_algorithms_list = QListWidget()
        self.imported_algorithms_list.setSelectionMode(QListWidget.MultiSelection)
        self.imported_algorithms_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.imported_algorithms_list.customContextMenuRequested.connect(self.show_imported_context_menu)
        imported_layout = QVBoxLayout()
        imported_layout.addWidget(QLabel(tr("Imported Algorithms")))
        imported_layout.addWidget(self.imported_algorithms_list)
        group_imported = QGroupBox()
        group_imported.setLayout(imported_layout)

        # === 中间：已加载算法 ===
        self.loaded_algorithms_list = QListWidget()
        self.loaded_algorithms_list.setSelectionMode(QListWidget.SingleSelection)
        self.loaded_algorithms_list.itemSelectionChanged.connect(self.update_headers)
        loaded_layout = QVBoxLayout()
        loaded_layout.addWidget(QLabel(tr("Loaded Algorithms")))
        loaded_layout.addWidget(self.loaded_algorithms_list)
        group_loaded = QGroupBox()
        group_loaded.setLayout(loaded_layout)

        # === 右侧：Header 预览 ===
        self.headers_list = QListWidget()
        headers_layout = QVBoxLayout()
        headers_layout.addWidget(QLabel(tr("Header Preview")))
        headers_layout.addWidget(self.headers_list)
        group_headers = QGroupBox()
        group_headers.setLayout(headers_layout)

        # === 中间按钮列 ===
        add_button = QPushButton(tr("Add →"))
        add_button.setToolTip("Move selected algorithms to loaded list")
        add_button.clicked.connect(self.move_to_loaded)

        remove_button = QPushButton(tr("← Remove"))
        remove_button.setToolTip("Remove selected algorithms from loaded list")
        remove_button.clicked.connect(self.remove_from_loaded)

        button_col = QVBoxLayout()
        button_col.addStretch()
        button_col.addWidget(add_button)
        button_col.addWidget(remove_button)
        button_col.addStretch()
        button_widget = QWidget()
        button_widget.setLayout(button_col)

        # === 分割视图 ===
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(group_imported)
        splitter.addWidget(button_widget)
        splitter.addWidget(group_loaded)
        splitter.addWidget(group_headers)

        # 添加分割器到主布局
        main_layout.addWidget(splitter)


        # 添加底部按钮布局
        bottom_buttons_layout = QHBoxLayout()

        # 顶部导入按钮
        import_button = QPushButton(tr("Load Custom Algorithms (.py)"))
        import_button.setToolTip("Import Python files with calcTraits & getHeaders")
        import_button.clicked.connect(self.loadCustom)

        # 无代码入口：节点图公式编辑器（唯一的公式形态，按用户要求移除了表格模式）
        formula_button = QPushButton(tr("New Formula Traits..."))
        formula_button.setToolTip(
            "Build traits by wiring nodes together, no code needed")
        formula_button.clicked.connect(self.openGraphEditor)

        # Save 按钮
        save_button = QPushButton(tr("Save Loaded Algorithms"))
        save_button.setToolTip("Save current loaded algorithms to file")
        save_button.clicked.connect(self.save_loaded_algorithms)

        bottom_buttons_layout.addWidget(import_button)
        bottom_buttons_layout.addWidget(formula_button)
        bottom_buttons_layout.addStretch()
        bottom_buttons_layout.addWidget(save_button)

        main_layout.addLayout(bottom_buttons_layout)
        # 应用主布局
        self.setLayout(main_layout)

        # 初始化已导入算法列表 - 使用实际存在的算法
        self.imported_algorithms_list.addItems(sorted(self.custom_algorithms.keys()))
        for name in loaded_names:
            if name in self.custom_algorithms:  # 只添加存在的算法
                self.loaded_algorithms_list.addItem(name)

    def openGraphEditor(self):
        """Open the node-graph formula editor and register what it saves."""
        from pheno.formulaEditorDialog import NodeFormulaEditorDialog
        # Units are derived from the current image's scale, so pass it in
        # when the dialog is reachable from a window that has one.
        scale_unit = ""
        window = self.window()
        if hasattr(window, "canvas"):
            scale_unit = getattr(window.canvas, "scale_unit", "") or ""
        dialog = NodeFormulaEditorDialog(self, scale_unit=scale_unit)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        path = getattr(dialog, "savedPath", None)
        if not path or not os.path.exists(path):
            return
        self._registerFormulaFile(path)

    def _registerFormulaFile(self, path):
        """Load a .phenoFormula.json and add it to the imported algorithms."""
        from pheno.formulaTraits import (
            FORMULA_SUFFIX, FormulaError, load_traits, make_algorithm,
        )
        algo_name = os.path.basename(path)[: -len(FORMULA_SUFFIX)]
        try:
            traits = load_traits(path)
        except FormulaError as error:
            QMessageBox.warning(self, tr("Failed to Load"), str(error))
            return

        get_headers, calc_traits = make_algorithm(traits)
        self.custom_algorithms[algo_name] = calc_traits
        self.custom_headers[algo_name] = get_headers()

        self.imported_algorithms_list.clear()
        self.imported_algorithms_list.addItems(sorted(self.custom_algorithms.keys()))
        # No modal popup: the imported list itself is the confirmation,
        # and a popup blocks headless/batch use of this dialog.

    def loadCustom(self):
        # Bundled algorithms live in <project>/algorithms so the dialog opens
        # there by default; users can still navigate anywhere.
        from pheno.formulaTraits import FORMULA_SUFFIX
        algorithms_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "algorithms",
        )
        files, _ = QFileDialog.getOpenFileNames(
            self,
            tr("Select Custom Algorithm File(s)"),
            algorithms_dir if os.path.isdir(algorithms_dir) else "",
            "Algorithms and Formula Traits (*.py *%s)" % FORMULA_SUFFIX
        )

        if not files:
            return

        # Formula files are JSON, not importable Python; they register
        # through their own loader.
        formula_files = [f for f in files if f.endswith(FORMULA_SUFFIX)]
        for path in formula_files:
            self._registerFormulaFile(path)
        files = [f for f in files if not f.endswith(FORMULA_SUFFIX)]
        if not files:
            return

        loaded = []
        failed = []
        updated = []

        for file_path in files:
            algo_name = os.path.basename(file_path)[:-3]
            try:
                # 检查是否已存在，如果存在就更新
                is_update = algo_name in self.custom_algorithms
                
                spec = importlib.util.spec_from_file_location(algo_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if not hasattr(module, "calcTraits"):
                    raise AttributeError(f"The {algo_name} is missing 'calcTraits'.")

                if not hasattr(module, "getHeaders"):
                    raise AttributeError(f"The {algo_name} is missing 'getHeaders'.")

                self.custom_algorithms[algo_name] = module.calcTraits
                self.custom_headers[algo_name] = module.getHeaders()
                
                if is_update:
                    updated.append(algo_name)
                else:
                    loaded.append(algo_name)

            except Exception as e:
                failed.append(f"{algo_name}: {e}")

        if loaded or updated:
            self.imported_algorithms_list.clear()
            self.imported_algorithms_list.addItems(sorted(self.custom_algorithms.keys()))
            
            message_parts = []
            if loaded:
                message_parts.append("Loaded:\n" + "\n".join(loaded))
            if updated:
                message_parts.append("Updated:\n" + "\n".join(updated))
            
            QMessageBox.information(
                self, tr("Success"), "\n\n".join(message_parts)
            )

        if failed:
            QMessageBox.warning(
                self, tr("Failed to Load"), "Errors:\n" + "\n".join(failed)
            )

 
    def show_imported_context_menu(self, position):
        """显示已导入算法列表的右键菜单"""
        item = self.imported_algorithms_list.itemAt(position)
        if item is None:
            return
            
        # 检查是否是DEFAULT算法（不允许删除）
        algo_name = item.text()
        if algo_name == "DEFAULT":
            return
            
        menu = QMenu(self)
        remove_action = menu.addAction("🗑️ Remove Algorithm")
        remove_action.triggered.connect(lambda: self.remove_imported_algorithm(algo_name))
        
        # 显示菜单
        global_position = self.imported_algorithms_list.mapToGlobal(position)
        menu.exec_(global_position)
    
    def remove_imported_algorithm(self, algo_name):
        """删除已导入的算法"""
        # 确认对话框
        reply = QMessageBox.question(
            self, 
            "Remove Algorithm", 
            f"Are you sure you want to remove the algorithm '{algo_name}'?\n\n"
            "This will also remove it from the loaded algorithms list if present.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            # 从字典中删除算法
            if algo_name in self.custom_algorithms:
                del self.custom_algorithms[algo_name]
            if algo_name in self.custom_headers:
                del self.custom_headers[algo_name]
            
            # 更新导入算法列表
            self.imported_algorithms_list.clear()
            self.imported_algorithms_list.addItems(sorted(self.custom_algorithms.keys()))
            
            # 从已加载算法列表中删除（如果存在）
            for i in range(self.loaded_algorithms_list.count()):
                item = self.loaded_algorithms_list.item(i)
                if item and item.text() == algo_name:
                    self.loaded_algorithms_list.takeItem(i)
                    break
            
            # 如果删除的是当前选中的算法，更新header预览
            if self.loaded_algorithms_list.count() > 0:
                self.loaded_algorithms_list.setCurrentRow(0)
            else:
                self.headers_list.clear()
                
            QMessageBox.information(self, tr("Success"), f"Algorithm '{algo_name}' has been removed.")
            
        except Exception as e:
            QMessageBox.warning(self, tr("Error"), f"Failed to remove algorithm: {str(e)}")

    def move_to_loaded(self):
        for item in self.imported_algorithms_list.selectedItems():
            name = item.text()
            if not self._is_in_list(self.loaded_algorithms_list, name):
                self.loaded_algorithms_list.addItem(name)

    def remove_from_loaded(self):
        selected_items = self.loaded_algorithms_list.selectedItems()
        if not selected_items:
            return
            
        # 记录被删除项的索引，用于选择下一个项
        removed_rows = []
        for item in selected_items:
            row = self.loaded_algorithms_list.row(item)
            removed_rows.append(row)
            self.loaded_algorithms_list.takeItem(row)
        
        # 选择下一个合适的项
        if self.loaded_algorithms_list.count() > 0:
            # 找到最小的被删除行号，选择该位置或之前的项
            min_removed_row = min(removed_rows)
            # 如果删除的是最后一项，选择新的最后一项
            new_selection_row = min(min_removed_row, self.loaded_algorithms_list.count() - 1)
            self.loaded_algorithms_list.setCurrentRow(new_selection_row)
            # update_headers 会通过 itemSelectionChanged 信号自动调用
        else:
            # 如果没有项了，清空 header 预览
            self.headers_list.clear()

    def update_headers(self):
        self.headers_list.clear()
        selected_items = self.loaded_algorithms_list.selectedItems()
        if selected_items:
            name = selected_items[0].text()
            headers = self.custom_headers.get(name, [])
            if headers is None:
                headers = []
            self.headers_list.addItems(headers)

    def _is_in_list(self, list_widget, text):
        return any(list_widget.item(i).text() == text for i in range(list_widget.count()))
    
    def save_loaded_algorithms(self):
        if not self.on_save_callback:
            QMessageBox.warning(self, tr("Error"), tr("No save callback provided."))
            return
        
        imported_names = [self.imported_algorithms_list.item(i).text() for i in range(self.imported_algorithms_list.count())]
        loaded_names = [self.loaded_algorithms_list.item(i).text() for i in range(self.loaded_algorithms_list.count())]

        self.on_save_callback(self.custom_algorithms, self.custom_headers, imported_names, loaded_names)

        self.accept()  # 关闭对话框

class BatchErrorDialog(QtWidgets.QDialog):
    """批处理结束后的汇总对话框：失败项 + 需要用户过目的警告。

    两类内容合在一个框里，因为批处理结束只该打断用户一次 —— 再弹第二个框，
    用户往往会直接关掉。比例尺借用属于警告而非失败，所以它不进 failures
    （那会把处理成功的图算成整张失败），只在这里单独成节显示。
    """

    def __init__(self, failure_dict, parent=None, scale_warnings=None,
                 scale_report_path=None, review_callback=None):
        super().__init__(parent)
        self.failure_dict = dict(failure_dict or {})
        self.scale_warnings = dict(scale_warnings or {})
        self.scale_report_path = scale_report_path
        self.setWindowTitle(
            tr("Batch Processing Completed - Errors Detected")
            if self.failure_dict
            else tr("Batch Processing Completed - Please Review")
        )
        self.setFixedSize(520, 320)

        layout = QtWidgets.QVBoxLayout(self)
        if self.failure_dict and self.scale_warnings:
            headline = tr("Some files failed and some scales were borrowed:")
        elif self.failure_dict:
            headline = tr("The following files failed to process:")
        else:
            headline = tr("Ruler scale was borrowed on some images:")
        self.headlineLabel = QtWidgets.QLabel(headline)
        self.headlineLabel.setWordWrap(True)
        layout.addWidget(self.headlineLabel)

        # 构建 HTML 格式文本（带样式），失败与警告各成一节
        # Colours come from the theme, not from CSS names: "blue" on a dark surface is
        # barely legible, and this report is exactly where someone needs to read a
        # filename carefully.
        html_sections = []
        plain_sections = []
        ink = theme.palette()

        if self.failure_dict:
            html_lines = []
            plain_lines = []
            for filename, error in self.failure_dict.items():
                html_lines.append(
                    f"<span style='color: {ink['accent']}; font-weight: bold'>{filename}:</span> "
                    f"<span style='color: {ink['danger']}'>{error}</span>"
                )
                plain_lines.append(f"{filename}: {error}")
            html_sections.append("<br><br>".join(html_lines))
            plain_sections.append("\n\n".join(plain_lines))

        if self.scale_warnings:
            summary = tr(
                "Ruler detection failed on {count} image(s). Their scale was "
                "borrowed from another image, so their lengths and areas are not "
                "absolute measurements.", count=len(self.scale_warnings))
            borrow_lines_html = []
            borrow_lines_plain = []
            for filename, source in self.scale_warnings.items():
                source_label = source or tr("(recorded image)")
                name = os.path.basename(filename)
                borrow_lines_html.append(
                    f"<span style='color: {ink['accent']}; font-weight: bold'>{name}</span>"
                    f" <span style='color: {ink['warn']}'>&lt;- {source_label}</span>"
                )
                borrow_lines_plain.append(f"{name}  <-  {source_label}")
            section = (
                f"<span style='color: {ink['warn']}'>{summary}</span><br><br>"
                + "<br>".join(borrow_lines_html)
            )
            if self.scale_report_path:
                section += (f"<br><br><span style='color: {ink['muted']}'>"
                            f"{self.scale_report_path}</span>")
            html_sections.append(section)
            plain = summary + "\n" + "\n".join(borrow_lines_plain)
            if self.scale_report_path:
                plain += "\n" + self.scale_report_path
            plain_sections.append(plain)

        self.html_text = "<hr>".join(html_sections)
        self.plain_text = "\n\n---\n\n".join(plain_sections)

        # 创建可滚动文本框
        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setHtml(self.html_text)
        self.text_edit.setFixedSize(480, 200)
        layout.addWidget(self.text_edit)

        # 按钮区域
        button_layout = QtWidgets.QHBoxLayout()

        copy_button = QtWidgets.QPushButton(tr("Copy to Clipboard"))
        copy_button.clicked.connect(self.copy_to_clipboard)
        button_layout.addWidget(copy_button)

        if review_callback is not None:
            review_button = QtWidgets.QPushButton(tr("Review Batch Results"))
            review_button.clicked.connect(
                lambda _checked=False: (self.accept(), review_callback()))
            button_layout.addWidget(review_button)

        ok_button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        ok_button_box.accepted.connect(self.accept)
        button_layout.addWidget(ok_button_box)

        layout.addLayout(button_layout)

    def copy_to_clipboard(self):
        QtWidgets.QApplication.clipboard().setText(self.plain_text)


class SetScaleDialog(QDialog):
    last_mode_index = 0  # 默认“使用自动设定尺度”
    last_real_value = "10"
    last_pixel_value = ""
    last_unit_index = 1  # 默认单位为 cm

    def __init__(self, pixmap_exists=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Set Scale"))
        self.setFixedSize(400, 200)
        self.draw_callback = None
        self.pixmap_exists = pixmap_exists


        layout = QVBoxLayout()

        # 第一行：是否使用手动设定尺度
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([tr("Automatic"), tr("Manual")])
        self.mode_combo.setCurrentIndex(SetScaleDialog.last_mode_index)
        self.mode_combo.currentIndexChanged.connect(self.toggle_inputs)
        layout.addWidget(self.mode_combo)

        # 输入校验器（仅允许数字和小数点）
        real_validator = QtGui.QRegExpValidator(QtCore.QRegExp(r'^\d*\.?\d*$'))

        # 第二行：真实长度 + 单位
        real_layout = QHBoxLayout()
        real_label = QLabel(tr("Real-world length:"))
        self.real_input = QLineEdit(SetScaleDialog.last_real_value)
        self.real_input.setValidator(real_validator)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["MM", "CM", "M"])
        self.unit_combo.setCurrentIndex(SetScaleDialog.last_unit_index)
        real_layout.addWidget(real_label)
        real_layout.addWidget(self.real_input)
        real_layout.addWidget(self.unit_combo)
        layout.addLayout(real_layout)

        # 第三行：像素长度
        pixel_layout = QHBoxLayout()
        pixel_label = QLabel(tr("Pixel length:"))
        self.pixel_input = QLineEdit(SetScaleDialog.last_pixel_value)
        self.pixel_input.setPlaceholderText("e.g. 150")
        self.pixel_input.setValidator(real_validator)
        pixel_layout.addWidget(pixel_label)
        pixel_layout.addWidget(self.pixel_input)
        layout.addLayout(pixel_layout)

        # 第四行：按钮区域
        btn_layout = QHBoxLayout()
        self.draw_button = QPushButton(tr("Draw on Image"))
        self.draw_button.clicked.connect(self.draw_on_image)
        # draw_button的初始状态将在toggle_inputs中设置
        self.apply_button = QPushButton(tr("Apply"))
        self.apply_button.clicked.connect(self.on_apply)
        self.apply_button.setDefault(True)  # 设置为默认按钮，响应回车键
        self.cancel_button = QPushButton(tr("Cancel"))
        self.cancel_button.clicked.connect(self.reject)
        btn_layout.addWidget(self.draw_button)
        btn_layout.addWidget(self.apply_button)
        btn_layout.addWidget(self.cancel_button)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # 初始启用状态
        self.toggle_inputs()

    def set_draw_callback(self, callback):
        """设置画线回调函数"""
        self.draw_callback = callback

    def is_auto_scale(self):
        """返回是否使用自动设定尺度"""
        return self.mode_combo.currentIndex() == 0

    def toggle_inputs(self):
        use_manual_scale = self.mode_combo.currentIndex() == 1
        # 其他控件根据是否使用手动设定尺度来启用/禁用
        for widget in [self.real_input, self.unit_combo, self.pixel_input]:
            widget.setEnabled(use_manual_scale)
        
        # draw_button需要同时满足：使用手动设定尺度 AND 有图像
        self.draw_button.setEnabled(use_manual_scale and self.pixmap_exists)

    def draw_on_image(self):
        """启动画线模式"""
        if not self.draw_callback:
            QtWidgets.QMessageBox.warning(self, tr("Error"), "Draw function not available")
            return
            
        try:
            # 调用回调函数启动画线模式（非阻塞）
            self.draw_callback()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, tr("Error"), f"Error during drawing: {str(e)}")

    def on_apply(self):
        """记录设定并关闭对话框"""
        # 如果选择使用手动设定尺度，需要验证输入
        if self.mode_combo.currentIndex() == 1:
            try:
                real_value = float(self.real_input.text())
                pixel_value = float(self.pixel_input.text())
                if real_value <= 0 or pixel_value <= 0:
                    QtWidgets.QMessageBox.warning(self, tr("Invalid Input"),
                                                 "Real-world length and pixel length must be positive numbers.")
                    return
            except ValueError:
                QtWidgets.QMessageBox.warning(self, tr("Invalid Input"),
                                             "Please enter valid numbers for real-world length and pixel length.")
                return
        
        # 保存设置
        SetScaleDialog.last_mode_index = self.mode_combo.currentIndex()
        SetScaleDialog.last_real_value = self.real_input.text()
        SetScaleDialog.last_pixel_value = self.pixel_input.text()
        SetScaleDialog.last_unit_index = self.unit_combo.currentIndex()
        self.accept()

    def get_scale(self):
        if self.mode_combo.currentIndex() == 0:
            return None, None
        try:
            real_value = float(self.real_input.text())
            pixel_value = float(self.pixel_input.text())
            unit = self.unit_combo.currentText()
            scale_per_pixel = real_value / pixel_value
            return scale_per_pixel, unit
        except (ValueError, ZeroDivisionError):
            return None, None

