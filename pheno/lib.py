import cv2
import numpy as np
from math import sqrt
import os.path as osp
import skimage
import imgviz

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import Qt, QRunnable, Signal, QObject
from qtpy.QtGui import QImage


here = osp.dirname(osp.abspath(__file__))


def write_image_file(path, image):
    """把图像写到任意路径，包括文件名或目录含中文的路径。

    cv2.imwrite 在 Windows 上按 ANSI 打开路径，遇到非 ASCII 的文件名或目录会
    直接失败 —— 而且它只返回 False、不抛异常，调用方于是提示"保存成功"却什么
    也没写出来（颜色校正图的保存就是这样把图静默丢掉的）。这里用 imencode 编码、
    tofile 落盘，走的是 Python 的文件 API，不受该限制；编码失败和写盘失败都会
    抛异常，交给调用方如实报错。
    """
    extension = osp.splitext(str(path))[1] or ".jpg"
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"Could not encode the image as {extension!r}")
    buffer.tofile(str(path))


class WorkerSignals(QObject):
    """Worker 线程的信号"""
    finished = Signal(str)
    error = Signal(str)
    result = Signal(object)
    progress = Signal(int, str)

class Worker(QRunnable):
    """使用 QRunnable 实现可取消的 Worker"""

    def __init__(self, func, func_name, *args, **kwargs):
        super().__init__()
        self.func = func
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.is_running = True  # 线程运行标志

    def run(self):
        """运行传递的封闭函数"""
        try:
            result = self.func(*self.args, **self.kwargs)
            self.signals.result.emit(result)
            self.signals.finished.emit(self.func_name)
        except Exception as e:
            self.signals.error.emit(str(e))

    def stop(self):
        """终止任务"""
        self.is_running = False  # 让任务在下次循环检查时终止


def newIcon(icon):
    """An icon in the current theme's ink colour.

    The bundled PNGs are a mixed set -- some dark monochrome, some flat colour, source
    sizes from 22 px to 1024 px. Redrawing them is a designer's job; recolouring them is
    this module's, so every glyph in the window comes from one palette and the dark ones
    stop vanishing against a dark toolbar. See ``pheno/iconTint.py``.
    """
    from pheno import iconTint, theme

    if iconTint.enabled() and theme.themed():
        tinted = iconTint.tinted(icon, theme.palette()["icon"])
        if tinted is not None:
            return tinted
    icons_dir = osp.join(here, 'icons')
    return QtGui.QIcon(osp.join(':/', icons_dir, '%s.png' % icon))


def newButton(text, icon=None, slot=None):
    b = QtWidgets.QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
        b.setProperty('phenoIcon', icon)
    if slot is not None:
        b.clicked.connect(slot)
    return b


def newAction(parent, text, slot=None, shortcut=None, icon=None,
              tip=None, checkable=False, enabled=True):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QtWidgets.QAction(text, parent)
    if icon is not None:
        a.setIcon(newIcon(icon))
        # Remembered so the icon can be redrawn when the theme changes; without it the
        # source name is lost and the glyph keeps the colour it was built with.
        a.setProperty('phenoIcon', icon)
    if shortcut is not None and shortcut != "None":
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a


def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QtWidgets.QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def labelValidator():
    return QtGui.QRegExpValidator(QtCore.QRegExp(r'^[^ \t].+'), None)


DEFAULT_LABEL_PREFIX = "Object"
_INVALID_LABEL_CHARS = '\\/:*?"<>|'


def sanitize_label_prefix(prefix):
    """Make an object name safe to embed in an exported file name.

    Object names are user-editable free text, so they must be scrubbed before
    they reach the filesystem. Lives in ``lib`` because both the canvas (naming)
    and batch/export code (file names) need it.
    """
    text = str(prefix or "").strip()
    for char in _INVALID_LABEL_CHARS:
        text = text.replace(char, " ")
    text = " ".join(text.split()).strip(". ")
    return text or DEFAULT_LABEL_PREFIX


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def fmtShortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)


def qimage_to_cv2(qimage):
    # 获取原始格式信息
    format_map = {
        QImage.Format_Grayscale8: ("gray", 1),
        QImage.Format_RGB888: ("rgb", 3),
        QImage.Format_RGBA8888: ("rgba", 4)
    }
    
    # 转换为标准格式（支持1/3/4通道）
    if qimage.format() not in format_map:
        qimage = qimage.convertToFormat(QImage.Format_RGBA8888)  # 默认转4通道[2][4]
    else:
        qimage = qimage.convertToFormat(qimage.format())
    
    # 获取通道数和图像尺寸
    mode, channels = format_map.get(qimage.format(), ("unknown", 4))
    height, width = qimage.height(), qimage.width()
    
    # 处理内存对齐（bytesPerLine可能不等于 width*channels）
    ptr = qimage.constBits()
    ptr.setsize(qimage.byteCount())
    bytes_per_line = qimage.bytesPerLine()
    
    # 转换为numpy数组
    arr = np.frombuffer(ptr, dtype=np.uint8)
    if bytes_per_line == width * channels:  # 无填充字节
        arr = arr.reshape(height, width, channels)
    else:  # 处理对齐填充
        arr = arr.reshape(height, bytes_per_line)
        arr = arr[:, :width*channels]
        arr = arr.reshape(height, width, channels)
    
    # 根据通道数转换颜色空间
    if channels == 1:  # 灰度图直接返回[1][3]
        return arr
    elif channels == 3:  # RGB转BGR
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif channels == 4:  # RGBA转BGRA（保留透明度）[2][4]
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    

def _get_contour_length(contour):
    contour_start = contour
    contour_end = np.r_[contour[1:], contour[0:1]]
    return np.linalg.norm(contour_end - contour_start, axis=1).sum()

def compute_polygon_from_mask(mask):
    contours = skimage.measure.find_contours(np.pad(mask, pad_width=1))
    if len(contours) == 0:
        return np.empty((0, 2), dtype=np.float32)

    contour = max(contours, key=_get_contour_length)
    POLYGON_APPROX_TOLERANCE = 0.001
    polygon = skimage.measure.approximate_polygon(
        coords=contour,
        tolerance=np.ptp(contour, axis=0).max() * POLYGON_APPROX_TOLERANCE,
    )
    polygon = np.clip(polygon, (0, 0), (mask.shape[0] - 1, mask.shape[1] - 1))
    polygon = polygon[:-1]  # drop last point that is duplicate of first point

    if 0:
        import PIL.Image

        image_pil = PIL.Image.fromarray(imgviz.gray2rgb(imgviz.bool2ubyte(mask)))
        imgviz.draw.line_(image_pil, yx=polygon, fill=(0, 255, 0))
        for point in polygon:
            imgviz.draw.circle_(image_pil, center=point, diameter=10, fill=(0, 255, 0))
        imgviz.io.imsave("contour.jpg", np.asarray(image_pil))

    return polygon[:, ::-1]  # yx -> xy