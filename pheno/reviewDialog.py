"""Standalone window for reviewing a finished batch run.

Deliberately separate from the main window: batch review needs a file list, a
preview and a wide table, and cramming those into the already busy main UI would
make both worse. The main window only gains one toolbar button that opens this.

Nothing here needs SAM or the original segmentation session -- everything is
recomputed from the ``.pheno.json`` sidecars a batch run leaves behind.
"""

import os

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from pheno import review
from pheno.i18n import tr
from pheno import theme


class _SummaryTableModel(QtCore.QAbstractTableModel):
    """Read-only view of the merged rows, with outliers tinted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._headers = []
        self._rows = []
        self._flags = []

    def setContent(self, headers, rows):
        self.beginResetModel()
        self._headers = list(headers)
        self._rows = list(rows)
        self._flags = [bool(row and row[-3] == "yes") for row in self._rows]
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            row = self._rows[index.row()]
            if index.column() < len(row):
                value = row[index.column()]
                return "" if value is None else str(value)
            return ""
        if role == Qt.BackgroundRole and self._flags[index.row()]:
            return QtGui.QBrush(QtGui.QColor(255, 235, 205))
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._headers[section] if section < len(self._headers) else ""
        return str(section + 1)


class BatchReviewDialog(QtWidgets.QDialog):
    """Folder picker + per-image preview + merged traits table."""

    THUMBNAIL_MAX = 900

    def __init__(self, parent=None, folder=None, image_dirs=()):
        super().__init__(
            parent,
            Qt.Dialog
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowMaximizeButtonHint,
        )

        self.setWindowTitle(tr('Review Batch Results'))
        self.resize(1180, 760)
        # Non-modal: the user may want to go back to the main window and fix an
        # image while this stays open.
        self.setModal(False)

        self._entries = []
        self._headers = []
        self._rows = []
        self._image_dirs = list(image_dirs or ())
        self._folder = None

        layout = QtWidgets.QVBoxLayout(self)

        # --- folder bar -------------------------------------------------
        folder_bar = QtWidgets.QHBoxLayout()
        self.folderEdit = QtWidgets.QLineEdit()
        self.folderEdit.setReadOnly(True)
        self.folderEdit.setPlaceholderText(tr('Choose a batch output folder...'))
        self.browseBtn = QtWidgets.QPushButton(tr('Browse...'))
        self.browseBtn.clicked.connect(self.chooseFolder)
        self.reloadBtn = QtWidgets.QPushButton(tr('Reload'))
        self.reloadBtn.clicked.connect(lambda: self.loadFolder(self._folder))
        self.reloadBtn.setEnabled(False)
        folder_bar.addWidget(QtWidgets.QLabel(tr('Folder:')))
        folder_bar.addWidget(self.folderEdit, 1)
        folder_bar.addWidget(self.browseBtn)
        folder_bar.addWidget(self.reloadBtn)
        layout.addLayout(folder_bar)

        # --- top half: image list + preview -----------------------------
        top = QtWidgets.QSplitter(Qt.Horizontal)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.imageList = QtWidgets.QTreeWidget()
        self.imageList.setColumnCount(3)
        self.imageList.setRootIsDecorated(False)
        self.imageList.setAlternatingRowColors(True)
        self.imageList.currentItemChanged.connect(self._onImageChanged)
        left_layout.addWidget(self.imageList)
        top.addWidget(left)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.previewLabel = QtWidgets.QLabel()
        self.previewLabel.setAlignment(Qt.AlignCenter)
        self.previewLabel.setMinimumSize(360, 260)
        self.previewLabel.setStyleSheet(
            "background-color: #2b2b2b; color: #cccccc; border: 1px solid #444;")
        right_layout.addWidget(self.previewLabel, 1)
        self.previewInfo = QtWidgets.QLabel()
        self.previewInfo.setWordWrap(True)
        self.previewInfo.setStyleSheet(f"color: {theme.palette()['muted']}; font-size: 12px;")
        right_layout.addWidget(self.previewInfo)
        top.addWidget(right)

        top.setStretchFactor(0, 2)
        top.setStretchFactor(1, 3)
        layout.addWidget(top, 3)

        # --- bottom half: merged table ----------------------------------
        self.tableModel = _SummaryTableModel(self)
        self.tableView = QtWidgets.QTableView()
        self.tableView.setModel(self.tableModel)
        self.tableView.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.tableView.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.tableView, 2)

        # --- status + actions -------------------------------------------
        bottom = QtWidgets.QHBoxLayout()
        self.statusLabel = QtWidgets.QLabel()
        self.statusLabel.setStyleSheet(f"color: {theme.palette()['muted']};")
        bottom.addWidget(self.statusLabel, 1)
        self.outlierOnlyBox = QtWidgets.QCheckBox(tr('Show outliers only'))
        self.outlierOnlyBox.stateChanged.connect(self._refreshTable)
        bottom.addWidget(self.outlierOnlyBox)
        self.editBtn = QtWidgets.QPushButton(tr('Edit in main window'))
        self.editBtn.clicked.connect(self.editInMainWindow)
        self.editBtn.setEnabled(False)
        bottom.addWidget(self.editBtn)
        self.exportBtn = QtWidgets.QPushButton(tr('Export Summary CSV'))
        self.exportBtn.clicked.connect(self.exportSummary)
        self.exportBtn.setEnabled(False)
        bottom.addWidget(self.exportBtn)
        self.closeBtn = QtWidgets.QPushButton(tr('Close'))
        self.closeBtn.clicked.connect(self.close)
        bottom.addWidget(self.closeBtn)
        layout.addLayout(bottom)

        self.retranslateUi()
        if folder:
            self.loadFolder(folder)

    # ------------------------------------------------------------------
    def retranslateUi(self):
        self.imageList.setHeaderLabels(
            [tr('Image'), tr('Objects'), tr('Outliers')])
        self.editBtn.setText(tr('Edit in main window'))

    def chooseFolder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, tr('Choose a batch output folder...'), self._folder or '')
        if folder:
            self.loadFolder(folder)

    def loadFolder(self, folder):
        if not folder:
            return
        self._folder = folder
        self.folderEdit.setText(folder)
        self.reloadBtn.setEnabled(True)

        self._entries = review.scan_folder(folder, extra_dirs=self._image_dirs)
        if not self._entries:
            self.imageList.clear()
            self.tableModel.setContent([], [])
            self.previewLabel.setText(tr('No annotation files found.'))
            self.previewInfo.clear()
            self.statusLabel.setText(tr('No annotation files found.'))
            self.exportBtn.setEnabled(False)
            return

        progress = QtWidgets.QProgressDialog(
            tr('Recomputing traits...'), tr('Cancel'), 0, len(self._entries), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(300)

        cancelled = {'flag': False}

        def report(index, total, name):
            if progress.wasCanceled():
                cancelled['flag'] = True
            progress.setMaximum(total)
            progress.setValue(index)
            if name:
                progress.setLabelText(tr('Recomputing traits: {name}', name=name))
            QtWidgets.QApplication.processEvents()

        self._headers, self._rows = review.build_rows(self._entries, progress=report)
        progress.close()

        if cancelled['flag']:
            self.statusLabel.setText(tr('Loading cancelled; results may be partial.'))

        self._refreshImageList()
        self._refreshTable()
        self.exportBtn.setEnabled(bool(self._rows))
        if self.imageList.topLevelItemCount():
            self.imageList.setCurrentItem(self.imageList.topLevelItem(0))

    # ------------------------------------------------------------------
    def _outlierCountFor(self, name):
        return sum(1 for row in self._rows if row['image'] == name and row['outlier'])

    def _problemCountFor(self, name):
        return sum(1 for row in self._rows if row['image'] == name and row['error'])

    def _refreshImageList(self):
        self.imageList.clear()
        for entry in self._entries:
            outliers = self._outlierCountFor(entry['name'])
            problems = self._problemCountFor(entry['name'])
            flag = str(outliers) if outliers else ''
            if problems:
                flag = (flag + ' ' if flag else '') + tr('{count} error(s)', count=problems)
            item = QtWidgets.QTreeWidgetItem(
                [entry['name'], str(entry['object_count']), flag])
            item.setData(0, Qt.UserRole, entry['sidecar'])
            if outliers or problems:
                for column in range(3):
                    item.setBackground(column, QtGui.QBrush(QtGui.QColor(255, 235, 205)))
            self.imageList.addTopLevelItem(item)
        for column in range(3):
            self.imageList.resizeColumnToContents(column)

    def _refreshTable(self):
        rows = self._rows
        if self.outlierOnlyBox.isChecked():
            rows = [row for row in rows if row['outlier'] or row['error']]
        headers = review.summary_headers(self._headers) if self._headers else []
        self.tableModel.setContent(headers, review.summary_rows(self._headers, rows))
        self.tableView.resizeColumnsToContents()

        total_objects = len(self._rows)
        outliers = sum(1 for row in self._rows if row['outlier'])
        errors = sum(1 for row in self._rows if row['error'])
        self.statusLabel.setText(tr(
            '{images} image(s), {objects} object(s), {outliers} outlier(s), {errors} error(s).',
            images=len(self._entries), objects=total_objects,
            outliers=outliers, errors=errors))

    # ------------------------------------------------------------------
    def _entryFor(self, sidecar):
        for entry in self._entries:
            if entry['sidecar'] == sidecar:
                return entry
        return None

    def editInMainWindow(self):
        """把选中的图送到主窗口打开，方便手动补标漏掉的对象。

        点了之后本窗隐藏、主窗口置顶，画布右上角浮出"Back to Batch Review"
        按钮：按下时把当前对象写回本审查目录的 sidecar，然后回到这里并只对
        这张图重算性状。图片载入时自动读取 sidecar，编辑改的是批处理结果本身。
        """
        item = self.imageList.currentItem()
        if item is None:
            return
        entry = self._entryFor(item.data(0, Qt.UserRole))
        # 找不到原图时用校色图兜底：它就在 sidecar 旁边，几何一致，
        # 主窗口靠 stem 归一找回原 sidecar（见 annotationPath）。
        source = review.trait_image_path(entry) if entry else None
        if not source:
            self.statusLabel.setText(
                tr('Image not found next to the annotation.'))
            return
        parent = self.parent()
        opener = getattr(parent, 'loadFile', None)
        if not callable(opener):
            return
        # 进入编辑前先记下主窗口当前图 —— 返回时要把画布还原回它，
        # 不能把审查编辑的那张留在界面上。
        prior = getattr(parent, 'filename', None)
        opener(source)
        self.hide()
        parent.showNormal()
        parent.raise_()
        parent.activateWindow()
        # Hand the main window the way back plus the batch state of this
        # image: it shows an edit HUD (Back to Batch Review) that writes the
        # sidecar and brings this dialog forward again.
        back = getattr(parent, 'beginReviewReturn', None)
        if callable(back):
            back(self, entry, prior_filename=prior)

    def returnFromMainWindow(self, edited_sidecar=None):
        """从主窗口编辑返回：只重算编辑过的那张图，然后浮到前台。

        ``edited_sidecar`` 是刚写回的 sidecar 路径；为空（写失败/兜底保存）
        时退回整目录重载。
        """
        if edited_sidecar:
            self.reloadEntry(edited_sidecar)
        elif self._folder:
            self.loadFolder(self._folder)
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def reloadEntry(self, sidecar_path):
        """重载并重算单张图的行 —— 一次编辑只动一张，整目录重算没意义。"""
        entry = review.load_entry(sidecar_path, extra_dirs=self._image_dirs)
        if entry is None:
            # sidecar 读不出来就整刷兜底，别静默留下脏数据。
            self.loadFolder(self._folder)
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            _headers, new_rows = review.build_rows([entry])
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        # 表头沿用整批的单位约定，不随单张改写。

        norm = os.path.normcase(os.path.normpath(sidecar_path))
        for index, old in enumerate(self._entries):
            if os.path.normcase(os.path.normpath(old['sidecar'])) == norm:
                self._entries[index] = entry
                break
        else:
            self._entries.append(entry)

        # 换掉该图全部旧行，按 _entries 顺序重排，再全表重标离群
        # （median 变了，重算便宜；图像级计算才是贵的）。
        survivors = [row for row in self._rows if row['image'] != entry['name']]
        by_image = {}
        for row in survivors + new_rows:
            by_image.setdefault(row['image'], []).append(row)
        self._rows = [row for e in self._entries
                      for row in by_image.get(e['name'], [])]
        review.mark_outliers(self._rows)

        self._refreshImageList()
        self._refreshTable()
        self.exportBtn.setEnabled(bool(self._rows))
        for index in range(self.imageList.topLevelItemCount()):
            item = self.imageList.topLevelItem(index)
            if item.data(0, Qt.UserRole) == entry['sidecar']:
                self.imageList.setCurrentItem(item)
                break

    def _onImageChanged(self, current, _previous):
        if current is None:
            return
        entry = self._entryFor(current.data(0, Qt.UserRole))
        if entry is None:
            return

        pixmap = self.renderPreview(entry, self.THUMBNAIL_MAX)
        if pixmap is None:
            self.previewLabel.setText(tr('Image not found next to the annotation.'))
        else:
            self.previewLabel.setPixmap(pixmap.scaled(
                self.previewLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        outliers = self._outlierCountFor(entry['name'])
        scale_text = ('%s = %.4f %s' % (tr('1 pixel'), entry['scale_per_pixel'],
                                        entry['scale_unit']))
        info = [
            tr('{count} object(s)', count=entry['object_count']),
            scale_text,
        ]
        if outliers:
            info.append(tr('{count} outlier(s)', count=outliers))
        if entry['image_size']:
            info.append('%d x %d' % entry['image_size'])
        state = entry.get('image_state', '')
        if state == 'replaced':
            info.append(tr('warning: image file changed since annotation'))
        elif state == 'moved':
            info.append(tr('image moved — showing located copy'))
        elif state == 'missing' and review.trait_image_path(entry):
            info.append(tr('original missing — showing colour-corrected copy'))
        self.previewInfo.setText('    '.join(info))
        self.editBtn.setEnabled(bool(review.trait_image_path(entry)))

    def renderPreview(self, entry, max_side):
        """Draw the image with each object's outline overlaid, downscaled."""
        source = review.trait_image_path(entry)
        if not source:
            return None
        image = QtGui.QImage(source)
        if image.isNull():
            return None

        scale = min(1.0, float(max_side) / max(image.width(), image.height()))
        if scale < 1.0:
            image = image.scaled(
                int(image.width() * scale), int(image.height() * scale),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)

        pixmap = QtGui.QPixmap.fromImage(image)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        outlier_labels = {
            row['label'] for row in self._rows
            if row['image'] == entry['name'] and (row['outlier'] or row['error'])
        }

        for data, _mask in review.decode_objects(entry):
            polygon = data.get('polygon') or []
            if len(polygon) < 3:
                continue
            is_bad = (data.get('label') or '') in outlier_labels
            color = QtGui.QColor(255, 90, 90) if is_bad else QtGui.QColor(60, 220, 130)
            painter.setPen(QtGui.QPen(color, 2))
            painter.setBrush(Qt.NoBrush)
            path = QtGui.QPainterPath()
            first = polygon[0]
            path.moveTo(first[0] * scale, first[1] * scale)
            for point in polygon[1:]:
                path.lineTo(point[0] * scale, point[1] * scale)
            path.closeSubpath()
            painter.drawPath(path)

            label = data.get('label') or ''
            if label:
                painter.setPen(QtGui.QPen(color))
                font = painter.font()
                font.setPointSize(max(8, int(10 / max(scale, 0.2) * scale * 1.4)))
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(
                    QtCore.QPointF(first[0] * scale + 4, first[1] * scale - 4), label)

        # 批处理记录在 sidecar 里的检测点位也画出来：色卡中心是蓝点白圈，
        # 标尺刻度是青色点 —— 和主窗口 canvas 的标注风格保持一致。
        processing = (entry.get('document') or {}).get('processing') or {}
        tickpoints = processing.get('ruler_tickpoints') or []
        centers = processing.get('color_checker_centers') or []
        try:
            for point in tickpoints:
                tx, ty = (point.get('x'), point.get('y')) if isinstance(
                    point, dict) else (point[0], point[1])
                radius = max(3, int(round(6 * scale)))
                painter.setPen(QtGui.QPen(QtGui.QColor('cyan'), 1))
                painter.setBrush(QtGui.QBrush(QtGui.QColor('cyan')))
                painter.drawEllipse(int(tx * scale), int(ty * scale), radius, radius)
            for point in centers:
                cx, cy = (point.get('x'), point.get('y')) if isinstance(
                    point, dict) else (point[0], point[1])
                outer = max(4, int(round(10 * scale)))
                painter.setPen(QtGui.QPen(QtGui.QColor('white'), 2))
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 120, 255, 180)))
                painter.drawEllipse(int(cx * scale), int(cy * scale), outer, outer)
            painter.setBrush(Qt.NoBrush)
        except (TypeError, ValueError, IndexError):
            pass  # 点位格式异常的旧数据不该挡住预览

        painter.end()
        return pixmap

    # ------------------------------------------------------------------
    def exportSummary(self):
        if not self._rows:
            return
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, tr('Export Summary CSV'),
            os.path.join(self._folder or '', 'review_summary.csv'),
            'CSV (*.csv)')
        if not path:
            return

        rows = self._rows
        if self.outlierOnlyBox.isChecked():
            rows = [row for row in rows if row['outlier'] or row['error']]
        try:
            review.export_summary(path, self._headers, rows)
        except OSError as error:
            QtWidgets.QMessageBox.critical(
                self, tr('Export Summary CSV'),
                tr('Failed to export: {message}', message=str(error)))
            return
        self.statusLabel.setText(tr('Exported {count} row(s) to {name}.',
                                    count=len(rows), name=os.path.basename(path)))
