"""
批量处理相关功能模块
"""
import os
import csv
import json
import datetime
import threading
import time
import traceback
from PIL import Image
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QMessageBox, QFileDialog
from pheno.i18n import tr

from pheno.lib import Worker, WorkerSignals, sanitize_label_prefix, qimage_to_cv2
from pheno.calcPheno import PhenomicsCalculator
from pheno.widgets import BatchErrorDialog
from pheno import annotation


class TimerWorker(QtCore.QObject):
    """独立的计时器工作线程"""
    timeUpdate = Signal(str, str)  # elapsed_time, remaining_time
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_time = None
        self.is_running = False
        self.total_files = 0
        self.current_progress = 0
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.update_time)
        
    def start_timer(self, total_files):
        """启动计时器"""
        self.start_time = datetime.datetime.now()
        self.is_running = True
        self.total_files = total_files
        self.current_progress = 0
        self._timer.start(1000)  # 每秒更新一次
        
    def stop_timer(self):
        """停止计时器"""
        self.is_running = False
        self._timer.stop()
        
    def update_progress(self, progress):
        """更新进度"""
        self.current_progress = progress
        
    def update_time(self):
        """更新时间显示"""
        if not self.is_running or not self.start_time:
            return
            
        # 计算已消耗时间
        elapsed = datetime.datetime.now() - self.start_time
        elapsed_seconds = int(elapsed.total_seconds())
        elapsed_formatted = f"{elapsed_seconds // 3600:02d}:{(elapsed_seconds % 3600) // 60:02d}:{elapsed_seconds % 60:02d}"
        
        # 预估剩余时间
        if self.current_progress > 5 and self.current_progress < 98:
            total_estimated_seconds = elapsed_seconds * 100 / self.current_progress
            remaining_seconds = int(total_estimated_seconds - elapsed_seconds)
            if remaining_seconds < 0:
                remaining_seconds = 0
            remaining_formatted = f"{remaining_seconds // 3600:02d}:{(remaining_seconds % 3600) // 60:02d}:{remaining_seconds % 60:02d}"
        elif self.current_progress >= 98:
            remaining_formatted = "00:00:00"
        else:
            remaining_formatted = "Calculating..."
            
        self.timeUpdate.emit(elapsed_formatted, remaining_formatted)


def batchPre(window):
    """批量处理预处理"""
    if window.fileListWidget.count() == 0:
        QMessageBox.information(window, tr("Batch Processing"), tr("No images to process."))
        return

    filenames = [window.fileListWidget.item(i).text()
                    for i in range(window.fileListWidget.count())
                    if window.fileListWidget.item(i).checkState() == Qt.Checked
                ]
    if len(filenames) == 0:
        QMessageBox.information(window, tr("Batch Processing"), tr("No images to process."))
        return

    dir_path = QFileDialog.getExistingDirectory(window, "Select Directory to Save Objects", window.lastOpenDir)
    if not dir_path:
        return
    window.lastOpenDir = dir_path

    window.shape_dock.setEnabled(False)
    window.fileListWidget.setEnabled(False)
    window.fileListWidget.selectAllCheckbox.setEnabled(False)

    window.batch_mode = True
    window.canvas.batch_mode = True
    if hasattr(window, 'modelSelectBtn'):
        window.modelSelectBtn.setEnabled(False)
    window.canvas.BATCH_PARAM = {"save_dir": "",
                               "filenames": "[]",
                               "imagesize": [],
                               "steps": [],
                               "points": [],
                               "failures": {},
                               "processed": set()}
    window.canvas.BATCH_PARAM["save_dir"] = dir_path
    window.canvas.BATCH_PARAM["filenames"] = filenames
    window.canvas.BATCH_PARAM["imagesize"] = [window.canvas.pixmap.width(), window.canvas.pixmap.height()]
    window.canvas.BATCH_PARAM["file_info"] = {filename:{
        "scale_per_pixel": 1.0,
        "scale_unit": "pixel",
        "auto_count": 0,
    } for filename in filenames}

    # 批处理步骤面板只在录制期间显示
    window.actions.colorChecker.setChecked(False)
    window.actions.rulerDetection.setChecked(False)
    window.showBatchStepsPanel(True)

    # 记录批处理开始时间

    window.actions.batchProcessing.triggered.disconnect()
    window.actions.batchProcessing.triggered.connect(lambda: batchStart(window))
    window.actions.batchProcessing.setText(tr("&Start"))


def batchStart(window):
    """开始批量处理"""
    if not window.canvas.BATCH_PARAM.get("steps"):
        QMessageBox.information(
            window,
            tr("Batch Processing"),
            tr("No batch steps recorded. Run an operation on this image first."),
        )
        return

    window.batch_start_time = datetime.datetime.now()

    step_kinds = {step.get("type") for step in window.canvas.BATCH_PARAM["steps"]}
    window._saved_colorChecker_state = "color_correction" in step_kinds
    window._saved_rulerDetection_state = "ruler_detection" in step_kinds
    window.showBatchStepsPanel(False)
    
    window.actions.batchProcessing.triggered.disconnect()
    window.actions.batchProcessing.triggered.connect(lambda: batchPre(window))
    window.actions.batchProcessing.setText(tr("&Batch"))
    window.actions.batchProcessing.setEnabled(False)

    window.progressDialog = QtWidgets.QProgressDialog("Processing images...", "Cancel", 0, 100, window)
    window.progressDialog.setWindowFlags(window.progressDialog.windowFlags() & ~Qt.WindowContextHelpButtonHint) # Hide the help button
    window.progressDialog.resize(450, 150)  # 增加宽度和高度以容纳时间信息
    window.progressDialog.setFixedSize(window.progressDialog.size())

    window.progressDialog.setWindowTitle(tr("Batch Processing"))
    window.progressDialog.setWindowModality(QtCore.Qt.WindowModal)
    window.progressDialog.setAutoClose(True)
    window.progressDialog.setAutoReset(True)
    window.progressDialog.setCancelButtonText("Cancel")

    window.progressDialog.canceled.connect(lambda: cancelBatch(window))
    window.progressDialog.show()
    monitor = getattr(window, 'gpuMonitor', None)
    if monitor is not None:
        monitor.set_busy(True)
    
    # 创建计时器线程
    window.timerWorker = TimerWorker()
    window.timerThread = QtCore.QThread()
    window.timerWorker.moveToThread(window.timerThread)
    
    # 连接计时器信号
    window.timerWorker.timeUpdate.connect(lambda elapsed, remaining: updateProgressWithTime(window, window.progressDialog.value(), window.progressDialog.labelText().split(": ")[-1] if ": " in window.progressDialog.labelText() else "", elapsed, remaining))
    
    # 启动计时器线程
    window.timerThread.start()
    total_files = len(window.canvas.BATCH_PARAM["filenames"])
    window.timerWorker.start_timer(total_files)
            
    window.dont_show_again_back = window.dont_show_again
    window.dont_show_again = True

    window.batchWorker = Worker(
        batchWork, 
        "Batch Processing", 
        window
    )

    # 转发对象必须在 GUI 线程创建 —— 若由 worker 线程创建再 moveToThread，
    # Qt 会拒绝移动（只能由对象所在线程 push），信号就会投递回 worker。
    window._gui_dispatcher = _GuiDispatcher()

    # 连接进度更新信号到新的函数，同时更新计时器进度。
    # 必须 QueuedConnection：信号由工作线程发出，默认直连会让槽函数
    # （progressDialog / 弹窗）在 worker 线程里跑 —— GUI 对象的非法跨线程访问。
    window.batchWorker.signals.progress.connect(
        lambda value, filename: updateProgressWithTimer(window, value, filename),
        QtCore.Qt.QueuedConnection)
    window.batchWorker.signals.result.connect(
        lambda msg: batchEnd(window, msg),
        QtCore.Qt.QueuedConnection)

    window.threadpool.start(window.batchWorker)


def batchEnd(window, msg=None):
    """结束批量处理"""
    if msg:
        window.status(msg)
    monitor = getattr(window, 'gpuMonitor', None)
    if monitor is not None:
        monitor.set_busy(False)
    
    # 停止并清理计时器线程
    if hasattr(window, 'timerWorker'):
        window.timerWorker.stop_timer()
    if hasattr(window, 'timerThread'):
        window.timerThread.quit()
        window.timerThread.wait()
    
    # progressDialog.cancel() 会再发 canceled() 信号 → 重新进入 cancelBatch；
    # batchEnd 是唯一的收尾路径（worker 的 result 信号统一触发），先断开。
    try:
        window.progressDialog.canceled.disconnect()
    except (TypeError, RuntimeError):
        pass
    window.progressDialog.cancel()

    # 记录批处理结束时间
    batch_end_time = datetime.datetime.now()
    # 计算运行时长（秒）并转换为时分秒格式
    if hasattr(window, 'batch_start_time'):
        duration_seconds = (batch_end_time - window.batch_start_time).total_seconds()
        start_time_str = window.batch_start_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 转换为时分秒格式
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        duration_seconds = None
        duration_formatted = "00:00:00"
        start_time_str = None
    end_time_str = batch_end_time.strftime('%Y-%m-%d %H:%M:%S')
    process_date = batch_end_time.strftime('%Y-%m-%d')

    # 计算成功/失败/跳过数
    # failures 的 key 既可能是文件名本身，也可能是 "<文件名>_<步骤>"，
    # 因此按文件名归并，避免同一文件的多个步骤失败被重复计数。
    # processed 记录真正跑完迭代的文件 —— 取消后没轮到的文件既不是成功
    # 也不是失败，单列为 skipped，否则取消一半却报 "Success: N/N"。
    filenames = window.canvas.BATCH_PARAM["filenames"]
    total_files = len(filenames)
    failed_keys = window.canvas.BATCH_PARAM["failures"].keys()
    failed_files = {
        filename for filename in filenames
        if any(key == filename or key.startswith(filename + "_") for key in failed_keys)
    }
    processed_files = set(window.canvas.BATCH_PARAM.get("processed") or set())
    failure_files = len(failed_files)
    success_files = len(processed_files - failed_files)
    skipped_files = total_files - len(processed_files | failed_files)

    # 写入meta.json
    write_batch_meta(window, start_time_str, end_time_str, duration_seconds, process_date, success_files, failure_files, skipped_files)

    # 写入汇总表型数据
    write_summary_pheno(window)

    # 显示批处理完成信息
    completion_msg = (f"Batch processing completed!\n"
                     f"Duration: {duration_formatted}\n"
                     f"Success: {success_files}/{total_files} files")
    if skipped_files:
        completion_msg += f"\nSkipped (cancelled): {skipped_files} files"
    window.status(completion_msg)

    # 每张图的对象都已经由 process_checked_shapes 写成了 sidecar，界面不该再
    # 认为有未保存的改动，否则切到别的图会弹出"要不要保存当前画面"。
    set_clean = getattr(window, "setClean", None)
    if callable(set_clean):
        set_clean()

    # 借用比例尺先静默写成报告文件，内容随后并入下面那个汇总对话框
    report_borrowed_scale(window)

    # 批处理动辄几十分钟，结束时用户多半不在窗口前 —— 闪烁任务栏图标
    # 直到窗口被激活；窗口本来就聚焦时 alert 是无操作。
    QtWidgets.QApplication.alert(window, 0)

    if (window.canvas.BATCH_PARAM["failures"]
            or window.canvas.BATCH_PARAM.get("borrowed_scale")):
        write_batch_failures(window)
    else:
        # 干净的批处理也用同一个“完成 + 检查结果”入口收尾，不打断两次。
        _show_batch_completion(window, duration_formatted, success_files,
                               total_files, skipped_files)
        restore_ui_state(window)


def _show_batch_completion(window, duration_formatted, success_files,
                           total_files, skipped_files=0):
    """结束框：一句话汇总 + 直接打开结果复查。"""
    box = QtWidgets.QMessageBox(window)
    box.setWindowTitle(tr("Batch Processing Completed"))
    text = tr(
        "Duration: {duration}\nSuccess: {success}/{total} files",
        duration=duration_formatted, success=success_files, total=total_files)
    if skipped_files:
        text += "\n" + tr("Skipped (cancelled): {count} file(s)",
                          count=skipped_files)
    box.setText(text)
    review_btn = box.addButton(
        tr("Review Batch Results"), QtWidgets.QMessageBox.ActionRole)
    box.addButton(QtWidgets.QMessageBox.Ok)
    box.exec_()
    if box.clickedButton() is review_btn:
        window.reviewBatchResults(
            folder=window.canvas.BATCH_PARAM["save_dir"])


def cancelBatchSetup(window):
    """放弃批处理录制（还没按 Start 时）：还原界面并清空已录步骤。

    输出目录一旦在 batchPre 的弹窗里选定，界面就再没有反悔入口 ——
    用户只能硬着头皮 Start。这个函数提供干净的退出通道。
    """
    steps = window.canvas.BATCH_PARAM.get("steps") or []
    if steps:
        answer = QMessageBox.question(
            window, tr("Exit Batch Setup"),
            tr("Discard the {count} recorded batch step(s) and exit?",
               count=len(steps)),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return

    window.actions.batchProcessing.triggered.disconnect()
    window.actions.batchProcessing.triggered.connect(lambda: batchPre(window))
    window.actions.batchProcessing.setText(tr("&Batch"))
    window.actions.batchProcessing.setEnabled(True)

    window.batch_mode = False
    window.canvas.batch_mode = False
    if hasattr(window, 'modelSelectBtn'):
        window.modelSelectBtn.setEnabled(True)
    if hasattr(window, 'dont_show_again_back'):
        window.dont_show_again = window.dont_show_again_back
        delattr(window, 'dont_show_again_back')

    window.showBatchStepsPanel(False)
    window.canvas.BATCH_PARAM["steps"] = []

    window.actions.click.setEnabled(True)
    window.shape_dock.setEnabled(True)
    window.fileListWidget.setEnabled(True)
    window.fileListWidget.selectAllCheckbox.setEnabled(True)
    window.status(tr("Batch setup cancelled."))


def restore_ui_state(window):
    """恢复UI界面到批处理前的状态"""
    # 恢复按钮状态和槽连接
    window.actions.batchProcessing.triggered.disconnect()
    window.actions.batchProcessing.triggered.connect(lambda: batchPre(window))
    window.actions.batchProcessing.setText(tr("&Batch"))
    window.actions.batchProcessing.setEnabled(True)
    
    # 恢复批处理模式状态
    window.dont_show_again = window.dont_show_again_back
    window.batch_mode = False
    window.canvas.batch_mode = False
    if hasattr(window, 'modelSelectBtn'):
        window.modelSelectBtn.setEnabled(True)

    # 隐藏批处理步骤面板并清空录制内容
    window.showBatchStepsPanel(False)
    window.canvas.BATCH_PARAM["steps"] = []

    # 恢复colorChecker状态
    if hasattr(window, '_saved_colorChecker_state'):
        window.actions.colorChecker.setChecked(False)
        delattr(window, '_saved_colorChecker_state')  # 清理临时属性

    # 恢复rulerDetection状态
    if hasattr(window, '_saved_rulerDetection_state'):
        window.actions.rulerDetection.setChecked(False)
        delattr(window, '_saved_rulerDetection_state')  # 清理临时属性
    
    # 恢复界面元素的可用状态
    window.actions.click.setEnabled(True)
    window.shape_dock.setEnabled(True)
    window.fileListWidget.setEnabled(True)
    window.fileListWidget.selectAllCheckbox.setEnabled(True)


def write_batch_failures(window):
    """导出批处理失败信息到 batch_failures.txt 并弹窗"""
    try:
        failures_file_path = os.path.join(window.canvas.BATCH_PARAM["save_dir"], f"batch_failures_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(failures_file_path, 'w', encoding='utf-8') as f:
            f.write("Batch Processing Failures Report\n")
            f.write("=" * 50 + "\n\n")
            for failed_item, error_msg in window.canvas.BATCH_PARAM["failures"].items():
                f.write(f"Failed Item: {failed_item}\n")
                f.write(f"Error: {error_msg}\n")
                f.write("-" * 30 + "\n")
        window.status(f"Failure report saved to: {failures_file_path}")
    except Exception as e:
        window.status(f"Error saving failure report: {str(e)}")
    
    # 显示汇总对话框：失败项 + 比例尺借用警告合在一处，只打断用户一次
    dialog = BatchErrorDialog(
        window.canvas.BATCH_PARAM["failures"], window,
        scale_warnings=window.canvas.BATCH_PARAM.get("borrowed_scale"),
        scale_report_path=window.canvas.BATCH_PARAM.get("scale_warnings_report"),
        review_callback=lambda: window.reviewBatchResults(
            folder=window.canvas.BATCH_PARAM["save_dir"]),
    )
    dialog.exec_()

    # 恢复UI状态
    restore_ui_state(window)


def note_borrowed_scale(params, filename, source_filename):
    """记下某张图的比例尺是借用来的，供完成提示与 meta 使用。

    沿用上一张的比例尺是刻意行为（否则该图在整批里根本没法测量），但它必须
    看得见：这件事以前是完全静默的，用户数据里那 7 张比例尺偏差最大 26% 就是
    这么来的 —— 量出来的长度/面积看着正常，其实是别的图的尺度。
    """
    borrowed = params.setdefault("borrowed_scale", {})
    borrowed[filename] = source_filename or ""
    return borrowed


def report_borrowed_scale(window):
    """把“比例尺借用”写成报告文件，并记下路径供汇总对话框使用。

    这里刻意不弹窗：批处理结束只该打断用户一次，这些内容会并入
    BatchErrorDialog（见 write_batch_failures）。本函数只负责把证据落到磁盘。
    """
    borrowed = window.canvas.BATCH_PARAM.get("borrowed_scale") or {}
    if not borrowed:
        return

    def _label(path):
        return os.path.basename(path) if path else tr("(recorded image)")

    try:
        report_path = os.path.join(
            window.canvas.BATCH_PARAM["save_dir"],
            f"scale_warnings_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Borrowed Ruler Scale Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("These images had no ruler of their own, so the scale measured\n"
                    "on another image was reused. Their lengths and areas are NOT\n"
                    "absolute measurements of these images.\n\n")
            for name, source in borrowed.items():
                f.write("%s  <-  %s\n" % (os.path.basename(name), _label(source)))
        window.canvas.BATCH_PARAM["scale_warnings_report"] = report_path
        window.status(f"Scale warning report saved to: {report_path}")
    except Exception as e:
        window.status(f"Error saving scale warning report: {str(e)}")


def loadRecipe(window):
    """Load a recipe file into the current batch, replacing any recorded steps.

    This is the reproducible half of "let something else decide what to do": a
    colleague or a paper's supplementary material writes the file, and
    the file replays the same way every time. Recording steps by hand still works;
    this just means the mouse is no longer the only way to fill the list.
    """
    from pheno import recipe as recipe_module

    if not getattr(window, "batch_mode", False) or not window.canvas.BATCH_PARAM:
        QtWidgets.QMessageBox.information(
            window,
            tr("Load Batch Recipe"),
            tr("Start a batch first -- press Batch and choose a folder -- then load a recipe to fill its steps."),
        )
        return

    path, _selected = QtWidgets.QFileDialog.getOpenFileName(
        window, tr("Load Batch Recipe"), "", tr("Batch recipe (*.json)")
    )
    if not path:
        return

    try:
        document = recipe_module.read_recipe(path)
        steps = recipe_module.to_runtime_steps(document)
    except recipe_module.RecipeError as error:
        QtWidgets.QMessageBox.warning(window, tr("Load Batch Recipe"), str(error))
        return

    window.canvas.BATCH_PARAM["steps"] = steps
    if hasattr(window, "refreshBatchSteps"):
        window.refreshBatchSteps()
    if hasattr(window, "_syncBatchFlagActions"):
        window._syncBatchFlagActions()

    window.status(
        tr("Loaded {count} step(s) from {name}").format(
            count=len(steps), name=os.path.basename(path)
        )
    )


def saveRecipe(window):
    """Write the batch's recorded steps to a recipe file -- the pair of loadRecipe.

    The steps a user clicked into the list are exactly what a recipe file holds, so
    "record once, replay on every new folder" needs nothing more than serializing
    them. Positional steps (region drags, clicks) are pixel coordinates, so the
    current frame's size goes into the file and the loader can refuse images it
    could not replay correctly.
    """
    from pheno import recipe as recipe_module

    if not getattr(window, "batch_mode", False) or not window.batchSteps():
        QtWidgets.QMessageBox.information(
            window,
            tr("Save Batch Recipe"),
            tr("Start a batch and record some steps first -- then save them as a recipe."),
        )
        return

    path, _selected = QtWidgets.QFileDialog.getSaveFileName(
        window, tr("Save Batch Recipe"), "recipe.phenoBatch.json", tr("Batch recipe (*.json)")
    )
    if not path:
        return

    image_size = None
    if not window.image.isNull():
        image_size = (window.image.width(), window.image.height())
    try:
        document = recipe_module.from_runtime_steps(
            window.batchSteps(), image_size=image_size)
        recipe_module.write_recipe(document, path)
    except recipe_module.RecipeError as error:
        QtWidgets.QMessageBox.warning(window, tr("Save Batch Recipe"), str(error))
        return

    window.status(
        tr("Saved {count} step(s) to {name}").format(
            count=len(document["steps"]), name=os.path.basename(path)
        )
    )


def serialize_batch_step(step):
    """把一个批处理步骤转成可 JSON 序列化的字典。"""
    kind = step.get("type")
    if kind == "region":
        return {
            "type": kind,
            "points": [
                {"x": point.x(), "y": point.y()}
                for point in step["points"]
            ],
            "fix_sample_count": (
                "auto" if step.get("fix_sample_count") is None
                else step["fix_sample_count"]
            ),
        }
    if kind == "click":
        return {"type": kind, "points": step.get("points", [])}
    if kind == "text":
        return {"type": kind, "prompt": step.get("prompt", "")}
    return {"type": kind}


def write_batch_meta(window, start_time_str, end_time_str, duration, process_date, success_files, failure_files, skipped_files=0):
    """写入批处理meta.json"""
    # 获取算法名和对应 headers（不包含 'Name'）
    try:
        algo_map = {}
        if hasattr(PhenomicsCalculator, 'custom_headers'):
            for name, headers in PhenomicsCalculator.custom_headers.items():
                # 过滤掉 'Name' 字段
                if isinstance(headers, list):
                    filtered = [h for h in headers if h != 'Name']
                elif isinstance(headers, str):
                    filtered = [headers] if headers != 'Name' else []
                else:
                    filtered = []
                algo_map[name] = filtered
        else:
            # 兼容性兜底
            algo_map = {}
    except Exception:
        algo_map = {}

    # 转换持续时间为时分秒格式
    duration_formatted = "00:00:00"
    if duration is not None:
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    meta_data = {
        "start_time": start_time_str,
        "end_time": end_time_str,
        "duration_seconds": duration,
        "duration_formatted": duration_formatted,
        "process_date": process_date,
        "algorithm": algo_map,
        "segmenter": getattr(getattr(window.canvas, "sam", None),
                             "model_key", None),
        "device": getattr(getattr(window.canvas, "sam", None),
                          "device", None),
        "save_dir": window.canvas.BATCH_PARAM["save_dir"],
        "filenames": window.canvas.BATCH_PARAM["filenames"],
        "imagesize": window.canvas.BATCH_PARAM["imagesize"],
        "steps": [serialize_batch_step(step) for step in window.canvas.BATCH_PARAM.get("steps", [])],
        "points": window.canvas.BATCH_PARAM["points"],
        "success_files": success_files,
        "failure_files": failure_files,
        "skipped_files": skipped_files,
        "color_correction_enabled": hasattr(window, '_saved_colorChecker_state') and window._saved_colorChecker_state,
        "ruler_detection_enabled": hasattr(window, '_saved_rulerDetection_state') and window._saved_rulerDetection_state,
        "file_info": window.canvas.BATCH_PARAM["file_info"],
        "borrowed_ruler_scale": window.canvas.BATCH_PARAM.get("borrowed_scale", {}),
    }

    meta_file_path = os.path.join(window.canvas.BATCH_PARAM["save_dir"], f"meta_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(meta_file_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        window.status(f"Metadata saved to: {meta_file_path}")
    except Exception as e:
        window.status(f"Error saving metadata: {str(e)}")


def write_summary_pheno(window):
    """把每张图的 *_phenomics.csv 合并成 summary_pheno.csv，首列为图像名。

    全程走 csv 模块：对象名是用户自由文本，可能含逗号或引号，用 split(',') /
    ','.join 会把一个名字拆成多列。
    """
    filenames = window.canvas.BATCH_PARAM["filenames"]
    saveDir = window.canvas.BATCH_PARAM["save_dir"]

    summary_path = os.path.join(saveDir, "summary_pheno.csv")
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as summary_file:
        writer = csv.writer(summary_file)
        header_written = False

        for filename in filenames:
            base_name = os.path.splitext(os.path.basename(filename))[0]
            csv_path = os.path.join(saveDir, f"{base_name}_phenomics.csv")

            if not os.path.exists(csv_path):
                continue  # 如果该文件不存在，跳过

            try:
                with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
                    rows = list(csv.reader(f))

                if not rows:
                    continue  # 如果文件为空，跳过

                if not header_written:
                    writer.writerow(['filename'] + rows[0])
                    header_written = True

                for row in rows[1:]:
                    writer.writerow([base_name] + row)
            except Exception:
                continue


def updateProgressSimple(window, value, filename):
    """简单的进度更新，不显示时间信息"""
    window.progressDialog.setLabelText(f"Processing: {filename}")
    window.progressDialog.setValue(value)


def updateProgressWithTimer(window, value, filename):
    """带计时器的进度更新"""
    # 用户已取消：继续 setValue/setLabelText 会把已关闭的弹窗重新拉出来
    dialog = getattr(window, "progressDialog", None)
    if dialog is None or dialog.wasCanceled():
        return

    # 更新计时器的进度
    if hasattr(window, 'timerWorker'):
        window.timerWorker.update_progress(value)
    
    # 暂时只更新进度值，时间信息由计时器线程处理
    window.progressDialog.setValue(value)
    # 存储当前文件名供计时器使用。信号里已经是带 (n/total) 的显示串，
    # 不能再过 basename —— 里面的 "/" 会被当路径分隔符，切得只剩 "50)"。
    window._current_processing_file = filename


def updateProgressWithTime(window, value, filename, elapsed_time, remaining_time):
    """更新进度条显示，包含时间信息"""
    # 同上：取消之后 QProgressDialog.setValue() 会自动重新 show()
    dialog = getattr(window, "progressDialog", None)
    if dialog is None or dialog.wasCanceled():
        return

    if hasattr(window, '_current_processing_file'):
        filename = window._current_processing_file
    
    progress_text = (f"Processing: {filename}\n"
                    f"Elapsed: {elapsed_time} | Remaining: {remaining_time}")
    
    window.progressDialog.setLabelText(progress_text)
    window.progressDialog.setValue(value)


def updateProgress(window, value, filename):
    """更新进度，包含已消耗时间和预估剩余时间"""
    # 计算已消耗时间
    if hasattr(window, 'batch_start_time'):
        elapsed = datetime.datetime.now() - window.batch_start_time
        elapsed_seconds = int(elapsed.total_seconds())
        elapsed_formatted = f"{elapsed_seconds // 3600:02d}:{(elapsed_seconds % 3600) // 60:02d}:{elapsed_seconds % 60:02d}"
        
        # 预估剩余时间
        if value > 5 and value < 98:  # 在有意义的进度范围内才计算剩余时间
            total_estimated_seconds = elapsed_seconds * 100 / value
            remaining_seconds = int(total_estimated_seconds - elapsed_seconds)
            if remaining_seconds < 0:  # 避免负数
                remaining_seconds = 0
            remaining_formatted = f"{remaining_seconds // 3600:02d}:{(remaining_seconds % 3600) // 60:02d}:{remaining_seconds % 60:02d}"
        elif value >= 98:
            remaining_formatted = "00:00:00"  # 接近完成
        else:
            remaining_formatted = "Calculating..."  # 进度太少，无法准确估算
            
        progress_text = (f"Processing: {filename}\n"
                        f"Elapsed: {elapsed_formatted} | Remaining: {remaining_formatted}")
    else:
        progress_text = f"Processing: {filename}"
    
    window.progressDialog.setLabelText(progress_text)
    window.progressDialog.setValue(value)


def _update_shape_count(window, filename):
    """轻量进度：只刷新该图的对象计数，不导出。

    步骤循环内调用 —— 重的导出（掩膜 PNG / sidecar / 性状）留给步骤循环后
    的 ``process_checked_shapes`` 统一做一遍。
    """
    info = window.canvas.BATCH_PARAM["file_info"].get(filename)
    if info is not None:
        info["auto_count"] = len(window.shapeList.get_checked_shapes())


def process_checked_shapes(window, filename):
    """保存当前勾选的对象、更新计数并导出该图像的表型 CSV。

    ``get_checked_shapes()`` 返回的是当前图像累计的全部勾选对象，因此每次调用
    都会用累计结果重写同一个 CSV，Region Detect 与 Semantic Detect 的结果可以叠加。

    每图只跑一遍（步骤循环之后）：掩膜 PNG、sidecar、性状计算和性状表刷新
    一次完成。性状用的是步骤全部落定后的像素 —— 校色在中间改图也不影响。
    """
    checked_shapes = window.shapeList.get_checked_shapes()
    info = window.canvas.BATCH_PARAM["file_info"].get(filename)
    if info is not None:
        info["auto_count"] = len(checked_shapes)
    if not checked_shapes:
        window.results, window.result_keys = [], []
        window.headers = window.traitHeaders()
        window.update_phenomics_table()
        return

    # 自动保存掩膜图像
    save_selected_masks_worker(checked_shapes, window.canvas.BATCH_PARAM["save_dir"], filename)

    # 同步写出 sidecar 标注，便于事后在 GUI 里复查这张图的分割结果
    try:
        save_dir = window.canvas.BATCH_PARAM["save_dir"]
        base_name = os.path.splitext(os.path.basename(filename))[0]
        annotation.save_document(
            os.path.join(save_dir, base_name + annotation.SIDECAR_SUFFIX),
            annotation.build_document(
                filename,
                checked_shapes,
                (window.canvas.pixmap.width(), window.canvas.pixmap.height()),
                scale_per_pixel=window.canvas.scale_per_pixel,
                scale_unit=window.canvas.scale_unit,
                color_correction=bool(getattr(window, "_saved_colorChecker_state", False)),
                ruler_detection=bool(getattr(window, "_saved_rulerDetection_state", False)),
                color_checker_centers=(getattr(window, "color_correction_data", None) or {}).get("centers"),
                ruler_tickpoints=(getattr(window, "ruler_detection_data", None) or {}).get("tickpoints"),
            ),
        )
    except Exception as annotation_error:
        window.canvas.BATCH_PARAM["failures"][f"{filename}_annotation"] = str(annotation_error)

    # 执行phenomics计算：整帧只转换一次，21 个对象共享 —— 之前每个对象
    # 各转一遍 24MP pixmap，一张图单转换就花掉几秒。
    results, keys = [], []
    headers = PhenomicsCalculator.getHeaders(
        getattr(window.canvas, "scale_unit", None))
    rgb_img = qimage_to_cv2(window.canvas.pixmap.toImage())

    for shape in checked_shapes:
        try:
            result = PhenomicsCalculator.calcPhenomics(
                shape, window.canvas.pixmap, window.canvas.scale_per_pixel,
                image=rgb_img)
            results.append(result)
            keys.append(shape.uid)
        except Exception as calc_error:
            window.canvas.BATCH_PARAM["failures"][f"{filename}_calc"] = str(calc_error)
            continue

    # 性状表与 CSV 同源刷新 —— 原先由单独的 _computeBatchTraits 又算一遍。
    window.results, window.result_keys = results, keys
    window.headers = window.traitHeaders()
    window.update_phenomics_table()

    # 导出计算结果到CSV文件
    if not results:
        return

    base_name = os.path.splitext(os.path.basename(filename))[0]
    csv_path = os.path.join(window.canvas.BATCH_PARAM["save_dir"], f"{base_name}_phenomics.csv")
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(results)
    except Exception as export_error:
        window.canvas.BATCH_PARAM["failures"][f"{filename}_export"] = str(export_error)


# 只有这两类步骤把录制时的像素坐标带进了处理过程：region 记录圈选的
# 多边形顶点，click 记录点选的 points。它们绑死在录制图的坐标系上，
# 换一张尺寸不同的图就会把坐标落到别处，所以只有它们必须要求批处理图与录制图
# 尺寸完全一致。text（语义检测）、ruler_detection、color_correction 都是整图
# 操作，与坐标无关，尺寸或方向不同也能正确执行 —— 之前一律拦截，导致横竖混拍
# 的一组照片（4096x3072 与 3072x4096）除了与录制图同向的那张之外全部被跳过。
COORDINATE_STEP_TYPES = ("region", "click")


def batch_steps_need_matching_size(steps):
    """批处理是否必须要求每张图与录制图尺寸一致。"""
    return any(
        str(step.get("type")) in COORDINATE_STEP_TYPES for step in (steps or [])
    )


# 每个回放步骤的结果在屏幕上停留多久的兜底值（毫秒）。面板上的“结果预览停留”
# （widgets.BatchStepsWidget.DEFAULT_PREVIEW_DWELL_S）说的同一件事，这里只是
# 窗口上没有该控件时的后路。
DEFAULT_PREVIEW_DELAY_MS = 1000


class _GuiDispatcher(QtCore.QObject):
    """驻留在 GUI 线程的转发对象：worker 把 callable 发给它，由它在 GUI 线程执行。

    invokeMethod 只接受槽名字符串，接受不了任意 callable，所以需要这个信号。
    """

    invoke = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.invoke.connect(self._run, QtCore.Qt.QueuedConnection)

    def _run(self, fn):
        fn()


def _gui_call(window, fn):
    """在 GUI 线程执行 ``fn`` 并阻塞当前工作线程，返回 ``fn`` 的结果。

    QPixmap / 控件读写只能在 GUI 线程进行 —— 批处理的工作线程以前直接调
    ``loadQImage``、读 ``shapeList``、用 ``canvas.pixmap`` 算性状，这些就是
    ``QBackingStore::endPaint() called with active painter`` 崩溃的来源。
    计算（SAM / cv2 / DINO）留在工作线程，只有碰 GUI 对象的片段交给这里。
    抛出的异常会原样传回，让步骤级 try/except 照常记入 failures。
    """
    # 判断"我是不是 GUI 线程"用 Python 层的主线程身份：QThread.currentThread()
    # 对非 Qt 创建的 python 线程可能误报主线程（adopted-thread 行为）。
    if threading.current_thread() is threading.main_thread():
        return fn()

    dispatcher = getattr(window, "_gui_dispatcher", None)
    if dispatcher is None:
        dispatcher = _GuiDispatcher()
        # dispatcher 可能是在 worker 线程里创建的 —— moveToThread 后信号投递
        # 才落到 GUI 线程。
        dispatcher.moveToThread(window.thread())
        window._gui_dispatcher = dispatcher

    result = {}
    done = QtCore.QEventLoop()

    def _run():
        try:
            result["v"] = fn()
        except Exception as error:
            result["e"] = error
        done.quit()

    dispatcher.invoke.emit(_run)
    done.exec_()
    if "e" in result:
        raise result["e"]
    return result.get("v")


def preview_delay_ms(window):
    """从批处理面板读“结果预览停留”（毫秒），读不到就用兜底值。"""
    getter = getattr(getattr(window, "batchStepsWidget", None),
                     "previewDelayMs", None)
    if callable(getter):
        try:
            return max(0, int(getter()))
        except Exception:
            return DEFAULT_PREVIEW_DELAY_MS
    return DEFAULT_PREVIEW_DELAY_MS


def _replay_color_correction(window, filename) -> bool:
    """Run the calibration step on the loaded frame. Returns True when it ran."""
    try:
        result = window.perform_color_correction(window.image)
        if not result:
            window.canvas.BATCH_PARAM["failures"][f"{filename}_color_correction"] = \
                "Color correction failed: No valid result returned"
            return False
        corrected_image = result['corrected_image']
        height, width, channel = corrected_image.shape
        bytes_per_line = 3 * width
        from qtpy import QtGui
        corrected_qimage = QtGui.QImage(
            corrected_image.data, width, height, bytes_per_line,
            QtGui.QImage.Format_RGB888).rgbSwapped()
        # loadQImage 写 canvas.pixmap 并触发重绘 —— GUI 对象，必须 GUI 线程做
        _gui_call(window, lambda: window.canvas.loadQImage(
            corrected_qimage, update_sam=False))
        # 在批量处理中减少强制重绘频率，使用毫秒延迟
        QtCore.QThread.msleep(200)
        return True
    except Exception as error:  # 校正失败，记录错误但继续处理
        window.canvas.BATCH_PARAM["failures"][f"{filename}_color_correction"] = str(error)
        return False


def _replay_ruler_detection(window, filename, ruler_source):
    """Run the ruler step on the loaded frame; returns the current scale source.

    A file whose ruler is not found borrows the last good scale -- deliberate, or
    the image could not be measured at all -- and the borrow is annotated so the
    provenance is visible rather than silent.
    """
    try:
        result = window.perform_ruler_detection(window.image)
        if result:
            # set_ruler_ticks 里的 repaint、scaleLabel 更新都是 GUI 操作
            _gui_call(window, lambda: (
                window.show_ruler_tags(result),
                window.set_ruler_scale(result)))
            QtCore.QThread.msleep(200)
            return filename
        QtCore.QThread.msleep(200)
    except Exception as error:
        window.canvas.BATCH_PARAM["failures"][f"{filename}_ruler_detection"] = str(error)

    note_borrowed_scale(window.canvas.BATCH_PARAM, filename, ruler_source)
    borrowed_from = (os.path.basename(ruler_source) if ruler_source
                     else tr("(recorded image)"))
    window.canvas.BATCH_PARAM["file_info"][filename]["scale_borrowed_from"] = borrowed_from
    _gui_call(window, lambda: window.status(tr(
        "Ruler not detected in {image}; its scale is borrowed from {source}.",
        image=os.path.basename(filename), source=borrowed_from)))
    return ruler_source


def batchWork(window):
    """批量处理工作函数"""
    filenames = window.canvas.BATCH_PARAM["filenames"]
    total = len(filenames)
    coordinate_steps = batch_steps_need_matching_size(
        window.canvas.BATCH_PARAM.get("steps")
    )
    # 最近一次真正检出标尺的图：后来检测失败的图会沿用它的比例尺
    ruler_source = None
    # 每个步骤回放结束后停留多久，供肉眼核对分割结果（面板上可调）
    preview_ms = preview_delay_ms(window)

    loadFinished_event = QtCore.QEventLoop()
    load_state = {"loaded": False}

    def _on_load_finished(_):
        load_state["loaded"] = True
        loadFinished_event.quit()

    window.loadFinished.connect(_on_load_finished)

    # 看门狗：加载超时或用户取消时也要退出等待。没有它，一个损坏的文件
    # （loadFinished 永远不来）或一次取消会让工作线程永远停在 exec_()，
    # 前台只能看到"卡死"，而 failures 里什么都不会留下。
    LOAD_TIMEOUT_MS = 120000
    watchdog = QtCore.QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(loadFinished_event.quit)
    cancel_poll = QtCore.QTimer()
    cancel_poll.setInterval(500)
    cancel_poll.timeout.connect(
        lambda: loadFinished_event.quit()
        if not getattr(window.batchWorker, "is_running", True) else None)
    cancel_poll.start()

    for idx, filename in enumerate(filenames):
        if not window.batchWorker.is_running:
            break

        # 发出进度信号
        percent = int(idx / total * 100)
        window.batchWorker.signals.progress.emit(
            percent, f"{os.path.basename(filename)} ({idx + 1}/{total})")

        # 每图分段计时：perf_counter 单次调用约百纳秒，开销可忽略。
        # 结果写进 file_info，随 meta_*.json 一并落盘。
        timing = {
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "load_s": 0.0, "segment_s": 0.0, "scale_detect_s": 0.0,
            "color_correct_s": 0.0, "export_s": 0.0,
            "preview_pause_s": 0.0, "total_s": 0.0,
        }
        _t_img = time.perf_counter()

        # precheck image size - 在检查前先应用 EXIF 旋转
        try:
            _t0 = time.perf_counter()
            with Image.open(filename) as img:
                # 处理 EXIF 方向信息，确保图像方向一致
                try:
                    exif = img._getexif()
                    if exif is not None:
                        orientation_key = 274  # EXIF 方向标签的键值
                        if orientation_key in exif:
                            orientation = exif[orientation_key]
                            # 根据方向信息旋转图像
                            if orientation == 2:
                                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            elif orientation == 3:
                                img = img.rotate(180, expand=True)
                            elif orientation == 4:
                                img = img.rotate(180, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                            elif orientation == 5:
                                img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                            elif orientation == 6:
                                img = img.rotate(-90, expand=True)
                            elif orientation == 7:
                                img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                            elif orientation == 8:
                                img = img.rotate(90, expand=True)
                except (AttributeError, KeyError, TypeError):
                    # 如果没有 EXIF 数据或无法读取方向信息，使用原始图像
                    pass
                
                w, h = img.size  # 获取旋转后的尺寸

            expected_w, expected_h = window.canvas.BATCH_PARAM["imagesize"]

            if (w, h) != (expected_w, expected_h) and coordinate_steps:
                if (w, h) == (expected_h, expected_w):
                    raise Exception(
                        "Image orientation differs from the recorded image: "
                        "expected {}, got {}. The recorded steps contain region/click "
                        "operations that carry pixel coordinates, so orientation must "
                        "match. Record the steps again on an image with this "
                        "orientation, or split the batch by orientation.".format(
                            (expected_w, expected_h), (w, h)))
                raise Exception("Image size inconsistency: expected {}, got {}".format(
                    (expected_w, expected_h), (w, h)))

            # 与坐标无关的步骤（语义检测 / 尺子检测 / 颜色校正）尺寸不同也放行：
            # 它们按整图处理，录制的参数里没有任何像素坐标。

            load_state["loaded"] = False
            window.loadFileInMainThread.emit(filename)
            watchdog.start(LOAD_TIMEOUT_MS)
            loadFinished_event.exec_()  # 等待图像加载完成，超时/取消由看门狗兜底
            watchdog.stop()
            timing["load_s"] += time.perf_counter() - _t0
            if not getattr(window.batchWorker, "is_running", True):
                break
            if not load_state["loaded"]:
                window.canvas.BATCH_PARAM["failures"][filename] = \
                    "Image load timed out after %ds" % (LOAD_TIMEOUT_MS // 1000)
                continue

            # 颜色校正 / 标尺检测也是步骤：在它们被录制的位置回放，而不是提前到
            # 列表之前——否则录在分割之后的校正，回放时却会先改图再让分割跑在
            # 校正后的像素上，结果和录制时不是一回事。
            correction_applied = False

            # 按录制顺序复刻分割类步骤
            for step in window.canvas.BATCH_PARAM.get("steps", []):
                if not window.batchWorker.is_running:
                    break
                kind = step.get("type")
                _t0 = time.perf_counter()
                try:
                    if kind == "region":
                        window.canvas.autoPredict(
                            step["points"],
                            step["fix_sample_count"],
                        )
                    elif kind == "text":
                        window.detectTextPromptSync(step["prompt"])
                    elif kind == "green":
                        window.canvas.segmentGreenLeafs(window.original_image)
                    elif kind == "click":
                        window.canvas.segmentClickPoints(step["points"])
                    elif kind == "color_correction":
                        correction_applied = _replay_color_correction(window, filename)
                    elif kind == "ruler_detection":
                        ruler_source = _replay_ruler_detection(
                            window, filename, ruler_source)
                    else:
                        continue
                except Exception as step_error:
                    window.canvas.BATCH_PARAM["failures"][f"{filename}_{kind}"] = str(step_error)
                    continue
                finally:
                    _bucket = {
                        "region": "segment_s", "text": "segment_s",
                        "green": "segment_s", "click": "segment_s",
                        "color_correction": "color_correct_s",
                        "ruler_detection": "scale_detect_s",
                    }.get(kind)
                    if _bucket:
                        timing[_bucket] += time.perf_counter() - _t0

                # 每步只刷新对象计数：掩膜 PNG、sidecar 和性状都在步骤循环后
                # 统一导出一遍 —— 校色步骤会改像素，循环内导出的前几遍反正都
                # 要被最后一遍覆盖，纯属重复劳动。shapeList 是控件，走 GUI。
                _gui_call(window, lambda: _update_shape_count(window, filename))

                if preview_ms:
                    # 工作线程在这里休眠，GUI 线程照常处理事件，分割结果因而能
                    # 真正画到界面上供核对 —— 这一步以前没有等待，结果一闪而过。
                    _tp = time.perf_counter()
                    QtCore.QThread.msleep(preview_ms)
                    timing["preview_pause_s"] += time.perf_counter() - _tp

            # 步骤中途被取消：这张图的步骤没跑完，不导出也不计入 processed，
            # 按 skipped 结算，免得残缺结果被当成成功文件写出去。
            if not window.batchWorker.is_running:
                break

            # 比例字段写在步骤循环之后：录在列表中间的标尺检测也必须被记下。
            window.canvas.BATCH_PARAM["file_info"][filename]["scale_per_pixel"] = window.canvas.scale_per_pixel
            window.canvas.BATCH_PARAM["file_info"][filename]["scale_unit"] = window.canvas.scale_unit

            # 全量导出一遍：掩膜 PNG、sidecar、性状 + 性状表刷新，都在
            # 所有步骤（含校色/标尺这类改像素的变换）落定后进行 —— 这是唯一
            # 需要的一遍，之前的逐步导出已被移除。GUI 线程执行并阻塞工作线程，
            # 因为 pixmap / 控件是 GUI 对象。
            _t0 = time.perf_counter()
            try:
                _gui_call(window, lambda: process_checked_shapes(window, filename))
            except Exception as export_error:
                window.canvas.BATCH_PARAM["failures"][f"{filename}_export"] = str(export_error)
            timing["export_s"] += time.perf_counter() - _t0

            if preview_ms:
                # 表更新后再停一拍：让用户先看到性状变化，再进入下一张。
                _tp = time.perf_counter()
                QtCore.QThread.msleep(preview_ms)
                timing["preview_pause_s"] += time.perf_counter() - _tp

            # 如果启用了颜色校正，保存校正后的完整图像（放在处理完所有步骤之后）
            if correction_applied:
                _t0 = time.perf_counter()
                try:
                    base_name = os.path.splitext(os.path.basename(filename))[0]
                    corrected_image_filename = f"{base_name}_color_corrected.jpg"
                    corrected_image_path = os.path.join(window.canvas.BATCH_PARAM["save_dir"], corrected_image_filename)

                    # 保存当前显示的图像（已经是校正后的）；QPixmap 读写只能在 GUI 线程
                    _gui_call(window, lambda: window.canvas.pixmap.save(corrected_image_path))
                except Exception as save_error:
                    window.canvas.BATCH_PARAM["failures"][f"{filename}_save_corrected"] = str(save_error)
                timing["export_s"] += time.perf_counter() - _t0

            window.canvas.BATCH_PARAM.setdefault("processed", set()).add(filename)

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            err_file = tb.filename
            err_line = tb.lineno
            err_func = tb.name
            err_msg = f"{type(e).__name__}: {e} (at {err_file}:{err_line} in {err_func}())"
            window.canvas.BATCH_PARAM["failures"][filename] = err_msg
            continue
        finally:
            timing["total_s"] = time.perf_counter() - _t_img
            _info = window.canvas.BATCH_PARAM["file_info"].get(filename)
            if _info is not None:
                _info["timing_s"] = timing

    # 发出完成进度信号
    cancel_poll.stop()
    window.loadFinished.disconnect(_on_load_finished)
    window.batchWorker.signals.progress.emit(100, "Completed")
    return "Batch processing completed."


def cancelBatch(window):
    """取消批量处理。

    只置停止标志：worker 在文件/步骤边界检查后自行退出，再由 result 信号
    统一走 batchEnd 结算。若这里直接调 batchEnd，worker 返回后还会结算
    第二次，而且未处理的文件会被错计为成功。"""
    window.batchWorker.stop()
    window.status("Batch processing canceled. Finishing the current step...")


def save_selected_masks_worker(shapes, save_dir, filename):
    """后台线程：保存选中的 mask"""
    for idx, shape in enumerate(shapes):
        mask = shape.mask
        # 对象名可以被用户改成任意文本，落盘前必须清掉非法文件名字符。
        label = sanitize_label_prefix(shape.label)

        if mask is None or mask.size == 0:
            continue  # 跳过无效 mask

        # 生成文件路径
        file_name = f"{os.path.splitext(os.path.basename(filename))[0]}_{label}.png"
        file_path = os.path.join(save_dir, file_name)
        try:
            # 掩膜是二值的，用 1bit PNG 存；同样的内容比 8bit 灰度小约 4.6 倍。
            Image.fromarray((mask > 0)).save(file_path, bits=1, optimize=True)
        except Exception as e:
            return f"Error saving {file_name}: {str(e)}"

    return "Selected objects extracted and saved."
