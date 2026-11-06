from qtpy import QT_VERSION
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

import numpy as np
import cv2
import math

QT5 = QT_VERSION[0] == '5'  # NOQA

from pheno.lib import qimage_to_cv2, compute_polygon_from_mask, sanitize_label_prefix
from pheno.outliers import (
    MIN_GROUP_FOR_OUTLIERS, area_is_outlier, axis_is_outlier,
    mask_major_axis)
from pheno.shape import Shape
from pheno.SAM import SAM, compute_iou, mask_overlap_ratio
from pheno.RulerNet import RulerNet
from pheno.TextDetector import TextDetector


# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor


DEFAULT_LABEL_PREFIX = "Object"


def parse_label_index(label, prefix):
    """Return the trailing number of ``label`` when it belongs to ``prefix``.

    ``("pod07", "pod")`` -> ``7``; a different prefix or no trailing digits
    yields ``None``.
    """
    if not label:
        return None
    text = str(label)
    if len(text) <= len(prefix) or not text.startswith(prefix):
        return None
    suffix = text[len(prefix):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def next_label_for_prefix(shapes, prefix):
    """Next display name for ``prefix``: one past the highest number in use.

    This is deliberately "max + 1" rather than "smallest free number":

    * ``Object01..Object05`` with ``Object03`` deleted -> ``Object06``
    * ``Object01..Object05`` with ``Object05`` deleted -> ``Object05``
    * everything deleted -> ``Object01``
    """
    prefix = sanitize_label_prefix(prefix)
    highest = 0
    for shape in shapes:
        index = parse_label_index(getattr(shape, "label", None), prefix)
        if index is not None and index > highest:
            highest = index
    return "%s%02d" % (prefix, highest + 1)


# 绿叶分割生成的对象名前缀。对象名会进导出文件名，保持 ASCII 安全。
GREEN_LEAF_PREFIX = "Leaf"


def _text_mask_traits(mask):
    binary = np.asarray(mask, dtype=np.uint8)
    y_coords, x_coords = np.where(binary > 0)
    if len(x_coords) == 0:
        return None

    width = int(x_coords.max() - x_coords.min() + 1)
    height = int(y_coords.max() - y_coords.min() + 1)
    area = int(len(x_coords))
    bbox_area = width * height

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour = max(contours, key=cv2.contourArea) if contours else None
    if contour is not None and len(contour) >= 3:
        rotated_width, rotated_height = cv2.minAreaRect(contour)[1]
        minor_axis = min(rotated_width, rotated_height)
        major_axis = float(max(rotated_width, rotated_height))
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else float("inf")
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0.0
    else:
        major_axis = float(max(width, height))
        aspect_ratio = major_axis / max(1, min(width, height))
        solidity = 0.0

    return {
        "area": float(area),
        "major_axis": major_axis,
        "aspect_ratio": float(aspect_ratio),
        "extent": float(area / bbox_area) if bbox_area else 0.0,
        "solidity": float(min(solidity, 1.0)),
    }


def _box_containment(inner, outer):
    """Fraction of ``inner`` box that lies inside ``outer`` box (xyxy)."""
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in inner[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in outer[:4]]
    except (TypeError, ValueError, IndexError):
        return 0.0
    inner_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    if inner_area == 0:
        return 0.0
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return (iw * ih) / inner_area


def _dedupe_overlapping_masks(candidates, max_overlap=0.30, max_containment=0.70):
    """Drop candidates whose mask overlaps a kept, higher-scoring one.

    Grounding DINO's box-level NMS lets boxes through whose IoU is under its
    threshold, yet SAM can expand both into nearly the same mask (a rice
    spike detected twice, slightly offset). This pass compares the actual
    masks and keeps the best-scoring detection of each overlapping cluster.

    A second check catches the escape hatch of the first: a weak box nested
    inside a strong one can yield a *different* mask (SAM picks a sub-part),
    so mask overlap stays low while the boxes still say "same object". A
    lower-scoring box that is mostly inside a kept one is dropped too.
    """
    ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept = []
    dropped = []
    for candidate in ordered:
        overlap = False
        for keeper in kept:
            if mask_overlap_ratio(candidate["mask"], keeper["mask"]) > max_overlap:
                overlap = True
                break
            if _box_containment(candidate.get("box"), keeper.get("box")) > max_containment:
                overlap = True
                break
        (dropped if overlap else kept).append(candidate)
    return kept, dropped


def _filter_text_mask_candidates(candidates, minimum_group_size=MIN_GROUP_FOR_OUTLIERS):
    """Remove only clear size or shape outliers within each detected label.

    The size band and the minimum group size live in :mod:`pheno.outliers`, so
    this filter, the region filter, the review window and the cohort analysis
    all agree; see that module for why the band is as wide as it is.
    """
    groups = {}
    for candidate in candidates:
        label = str(candidate.get("label") or "").strip().lower()
        groups.setdefault(label, []).append(candidate)

    accepted = []
    filtered = []
    for group in groups.values():
        if len(group) < minimum_group_size:
            accepted.extend(group)
            continue

        median_area = float(np.median([item["traits"]["area"] for item in group]))
        median_axis = float(
            np.median([item["traits"]["major_axis"] for item in group])
        )
        median_aspect = float(
            np.median([item["traits"]["aspect_ratio"] for item in group])
        )
        median_extent = float(np.median([item["traits"]["extent"] for item in group]))
        median_solidity = float(
            np.median([item["traits"]["solidity"] for item in group])
        )

        for candidate in group:
            traits = candidate["traits"]
            area_ok = not area_is_outlier(traits["area"], median_area)
            axis_ok = not axis_is_outlier(traits["major_axis"], median_axis)

            aspect_ratio = traits["aspect_ratio"] / max(median_aspect, 1e-6)
            aspect_ok = 0.4 <= aspect_ratio <= 2.5
            extent_ok = traits["extent"] >= max(0.10, 0.45 * median_extent)
            solidity_ok = traits["solidity"] >= max(0.25, 0.55 * median_solidity)
            shape_ok = sum((aspect_ok, extent_ok, solidity_ok)) >= 2

            if area_ok and axis_ok and shape_ok:
                accepted.append(candidate)
            else:
                filtered.append(candidate)

    return accepted, filtered


def _box_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _box_center_in_mask(box, mask):
    cx = int(round((box[0] + box[2]) / 2.0))
    cy = int(round((box[1] + box[3]) / 2.0))
    h, w = mask.shape[:2]
    return 0 <= cx < w and 0 <= cy < h and bool(mask[cy, cx])


def _dedupe_region_masks(masks):
    """Region 提案去重：小 mask 先入库；IoU/overlap 之外再卡"候选质心
    落在已入库 mask 内" —— IoU<0.3 放过的平移重复框全靠这一条拦。
    收尾丢掉面积 < 中位数 1/4 的碎片（只在样本量足够时启用）。
    外接框不相交的配对不可能重叠/包含，先剪枝再碰像素。"""
    kept = []  # (mask, bbox, area)
    for mask in sorted(masks, key=lambda m: int(np.count_nonzero(m))):
        m = np.asarray(mask) > 0
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        bbox = (xs.min(), ys.min(), xs.max(), ys.max())
        cy, cx = int(ys.mean()), int(xs.mean())
        duplicate = False
        for k, kb, _ka in kept:
            if (bbox[0] > kb[2] or kb[0] > bbox[2]
                    or bbox[1] > kb[3] or kb[1] > bbox[3]):
                continue  # 框不相交 => 无重叠，质心也必不在 k 内
            if (compute_iou(m, k) >= 0.3
                    or mask_overlap_ratio(m, k) >= 0.5
                    or k[cy, cx]):
                duplicate = True
                break
        if not duplicate:
            kept.append((m.astype(np.uint8), bbox, len(xs)))
    if len(kept) >= MIN_GROUP_FOR_OUTLIERS:
        median_area = float(np.median([a for _m, _b, a in kept]))
        kept = [t for t in kept if t[2] >= 0.25 * median_area]
    return [t[0] for t in kept]


def _region_edge_band(region_mask, width=5):
    inner = cv2.erode(
        np.asarray(region_mask, dtype=np.uint8),
        np.ones((width, width), np.uint8)).astype(bool)
    return (np.asarray(region_mask) > 0) & ~inner


def _split_mask_components(mask, max_components=64):
    """把多连通域 mask 拆成每个显著连通域一个 mask。

    检测框把一群个体合成一个框时，SAM 的解码结果是一份含 N 个连通域的
    mask——当作一个对象既错计数也污染面积统计。拆开后每个连通域走各自
    的过滤判据。碎片（<100px 或 <最大连通域 5%）不计对象、也不回填到
    主 mask 里。
    """
    mask = np.asarray(mask) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return [mask]
    areas = stats[1:, cv2.CC_STAT_AREA]
    floor = max(100, areas.max() * 0.05)
    keep = [i + 1 for i, a in enumerate(areas) if a >= floor]
    if len(keep) <= 1 or len(keep) > max_components:
        return [mask]
    return [labels == i for i in keep]


def _drop_edge_artifacts(masks, edge_band, region_mask):
    """丢弃圈选时沿多边形边缘切出来的伪对象。

    独立判据（任一命中即丢）：
    - fill < 0.25：稀疏碎片——被圈边斜切出的板面/背景残片散占整个外接
      框但像素稀疏（正常器官填充率通常 >0.4）；
    - edge_share > 0.5：贴边条带——mask 边界过半骑在多边形边上；
    - edge_share > 0.3 且细长（短边 < 区域 4% 或长宽比 > 4）：偏细长的
      贴边物——被圈边切掉一侧的正常对象 edge_share 通常 <0.3；
    - 外接框贴住 Region 包围盒的 >=3 条边：贯穿区域的通长条带——正常
      对象最多贴 1 边。
    """
    rys, rxs = np.where(np.asarray(region_mask) > 0)
    if len(rxs) == 0:
        return list(masks)
    rx1, ry1, rx2, ry2 = rxs.min(), rys.min(), rxs.max(), rys.max()
    span = max(1, max(rx2 - rx1, ry2 - ry1))
    kept = []
    kernel = np.ones((3, 3), np.uint8)
    for m in masks:
        m = np.asarray(m) > 0
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        area = float(len(xs))
        bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max(), ys.max()
        bw, bh = float(bx2 - bx1 + 1), float(by2 - by1 + 1)
        fill = area / (bw * bh)
        boundary = m & ~cv2.erode(m.astype(np.uint8), kernel).astype(bool)
        edge_share = float((boundary & edge_band).sum()) / max(1.0, boundary.sum())
        near = 8  # 外接框与 Region 包围盒视为"贴上"的容差（像素）
        touched = (int(bx1 - rx1 <= near) + int(rx2 - bx2 <= near)
                   + int(by1 - ry1 <= near) + int(ry2 - by2 <= near))
        slender = min(bw, bh) < 0.04 * span or max(bw, bh) / min(bw, bh) > 4.0
        # 外接框沿某一轴贯通 >=90% 区域且细长：被圈边切出的"内条带"——
        # 它未必骑在多边形边上（条带边界跟着被切物走），靠贴边规则拦不住。
        span_v = bh >= 0.9 * float(ry2 - ry1 + 1)
        span_h = bw >= 0.9 * float(rx2 - rx1 + 1)
        elongated = max(bw, bh) / min(bw, bh) > 3.0
        if (fill < 0.25 or edge_share > 0.5
                or (edge_share > 0.3 and slender) or touched >= 3
                or ((span_v or span_h) and elongated)
                or (slender and touched >= 2)):
            print(f"[Region] dropped edge artifact (fill={fill:.2f}, "
                  f"edge={edge_share:.2f}, touched={touched})")
            continue
        kept.append(m.astype(np.uint8))
    return kept


def _filter_region_outliers(masks):
    """与 auto_predict_with_centers 相同的面积/长轴离群过滤。"""
    if len(masks) < MIN_GROUP_FOR_OUTLIERS:
        return list(masks)
    areas = [float(np.count_nonzero(m)) for m in masks]
    median_area = float(np.median(areas))
    kept = [m for m, a in zip(masks, areas)
            if not area_is_outlier(a, median_area)]
    if len(kept) >= MIN_GROUP_FOR_OUTLIERS:
        axes = [mask_major_axis(m) for m in kept]
        median_axis = float(np.median(axes))
        kept = [m for m, a in zip(kept, axes)
                if not axis_is_outlier(a, median_axis)]
    return kept


class Canvas(QtWidgets.QWidget):
    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    selectionChanged = QtCore.Signal(bool)
    newShape = QtCore.Signal(object)
    autoShapesReady = QtCore.Signal(object)
    maskPolygon = QtCore.Signal(object)  # list[QPoint] lasso outline
    line_drawn = QtCore.Signal(float)
    mousePositionChanged = QtCore.Signal(int, int)  # 鼠标位置变化信号
    brushStrokeStarted = QtCore.Signal()  # 笔刷落笔，供撤销记录快照
    brushStrokeFinished = QtCore.Signal(object)  # 笔刷抬笔，载荷为被修改的 Shape
    brushRadiusChanged = QtCore.Signal(int)  # 滚轮改完半径，HUD 实时刷新读数
    shapePicked = QtCore.Signal(object)  # NORMAL 模式点中已分割对象（None=空白）


    NORMAL, DRAW, MASK, AUTO, MEASURE, BRUSH = 0, 1, 2, 3, 4, 5
    BATCH_PARAM = {}
    

    def __init__(self, *args, **kwargs):
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.scale = 1.0
        self.shapes = []
        # Monotonic counter behind Shape.uid: never reused, never displayed.
        self.totalShapes = 0
        self.current = None
        self.pixmap = QtGui.QPixmap()
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT

        self.selectedShapes = []
        self.visible = {}

        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

        # SAM
        self.mode = self.NORMAL
        self.sam = SAM()
        self.show_bgmask = False
        self.region_points = []  # drawn outline for Region Detect
        self.region_mode = 'rect'  # 'rect' drag or 'freehand' lasso
        self.dragging = False

        # Batch
        self.batch_mode = False

        # Brush mask editing
        self.brush_radius = 30
        self.brush_target = None      # Shape being edited
        self.brush_painting = False   # True while a stroke is in progress
        self.brush_erasing = False    # Right button removes instead of adds
        self.brush_dirty = False      # Stroke actually changed something
        # Image-space points of the stroke in progress. Painted as a translucent
        # ribbon so the user sees the dab immediately: the object outline is only
        # recomputed on mouse release (compute_polygon_from_mask costs ~240 ms).
        self.brush_stroke = []

        # Drop
        parent = kwargs.get('parent', None)
        self.parent = parent
        self.setAcceptDrops(True)

        # Real-world Scale
        self.scale_per_pixel = 1.0
        self.scale_unit = "pixel"
        self.measure_line = None
        self.current_mouse_pos = QtCore.QPoint()
        self.last_valid_mouse_pos = QtCore.QPoint()  # 存储最后有效的图像内坐标
        self.ruler_net = RulerNet()

        self.autoShapesReady.connect(
            self._commitAutoShapes,
            QtCore.Qt.BlockingQueuedConnection,
        )
        
        # Color Checker markers
        self.color_checker_centers = []  # 存储颜色校正标记的中心点
        self.ruler_ticks = []  # 存储标尺刻度信息


    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isdrawing(self):
        return self.mode == self.DRAW
    
    def isMasking(self):
        return self.mode == self.MASK
    
    def isMeasuring(self):
        return self.mode == self.MEASURE

    def isBrushing(self):
        return self.mode == self.BRUSH

    # region Brush mask editing
    def setBrushTarget(self, shape):
        """Choose which object the brush edits."""
        self.brush_target = shape
        self.update()

    def adjustBrushRadius(self, delta):
        """Grow/shrink the brush; returns the new radius."""
        previous = self.brush_radius
        step = max(1, int(round(self.brush_radius * 0.2)))
        self.brush_radius = int(min(400, max(2, self.brush_radius + (step if delta > 0 else -step))))
        if self.brush_radius != previous:
            # Only the cursor ring changed, so repaint the larger of the two
            # rings instead of the whole (possibly 16 MPx) canvas.
            self.update(self._brushDirtyRect(
                [self.current_mouse_pos], radius=max(previous, self.brush_radius)
            ))
            self.brushRadiusChanged.emit(self.brush_radius)
        return self.brush_radius

    def _brushDirtyRect(self, positions, radius=None):
        """Widget-space rect covering the brush ring at the given image points.

        ``paintEvent`` always redraws everything, but Qt clips it to the region
        passed to ``update()``, so a small rect keeps the cost of a stroke
        proportional to the dab rather than to the image.
        """
        radius = self.brush_radius if radius is None else radius
        points = [point for point in positions if point is not None]
        if not points:
            return self.rect()

        offset = self.offsetToCenter()
        xs = [point.x() for point in points]
        ys = [point.y() for point in points]
        # Image space -> widget space: (image + offset) * scale.
        left = (min(xs) - radius + offset.x()) * self.scale
        top = (min(ys) - radius + offset.y()) * self.scale
        right = (max(xs) + radius + offset.x()) * self.scale
        bottom = (max(ys) + radius + offset.y()) * self.scale
        # Slack for the ring pen width, antialiasing and int truncation.
        margin = 6
        return QtCore.QRect(
            int(left) - margin,
            int(top) - margin,
            int(right - left) + 2 * margin,
            int(bottom - top) + 2 * margin,
        )

    def _paintBrushAt(self, pos, erase, previous=None):
        """Stamp one dab into the target mask, sweeping from ``previous``.

        Stamping only a circle per mouse-move event leaves gaps whenever the
        pointer moves faster than the event rate, which reads as a stuttering
        brush; drawing the connecting segment as a thick line makes the stroke
        continuous no matter how fast the pointer travels.
        """
        shape = self.brush_target
        if shape is None or shape.mask is None:
            return False

        mask = shape.mask
        if mask.dtype != np.uint8:
            mask = (np.asarray(mask) > 0).astype(np.uint8)
            shape.mask = mask

        height, width = mask.shape[:2]
        center = (int(round(pos.x())), int(round(pos.y())))
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            return False

        value = 0 if erase else 1
        radius = int(self.brush_radius)
        # cv2 writes in place, which is what makes a stroke cheap even on a
        # 16 MPx mask: only the touched pixels are rewritten.
        if previous is not None:
            start = (int(round(previous.x())), int(round(previous.y())))
            if start != center:
                cv2.line(mask, start, center, value, radius * 2, cv2.LINE_8)
        cv2.circle(mask, center, radius, value, -1)
        return True

    def _refreshBrushTargetOutline(self):
        """Recompute the polygon after a stroke so the outline follows the mask."""
        shape = self.brush_target
        if shape is None or shape.mask is None:
            return
        if not np.any(shape.mask):
            # An emptied mask has no contour; leave the points untouched so the
            # object stays selectable and the user can undo.
            return
        polygon = compute_polygon_from_mask(shape.mask)
        if len(polygon):
            shape.set(points=polygon)
    # endregion
    
    def setMode(self, mode):
        if self.mode == self.DRAW and mode == self.NORMAL:
            if not self.current:
                self.mode = self.NORMAL
                self.show_bgmask = False
                self.repaint()
                return
            self.sam.reset_points()
            if self.batch_mode:
                # While recording a batch, clicks only contribute their prompt
                # points; no object is committed to the current image.
                self.current = None
                self.mode = self.NORMAL
                self.show_bgmask = False
                self.repaint()
                return
            self.shapes.append(self.current)
            shape = self.current
            self.current = None
            self.mode = self.NORMAL
            self.show_bgmask = False
            self.registerShape(shape)
            self.repaint()

        if self.mode == self.MASK and mode == self.NORMAL:
            self.mode = self.NORMAL
            self.show_bgmask = False
            self.repaint()
        
        if self.mode == self.NORMAL and mode == self.DRAW:
            self.mode = self.DRAW
            self.show_bgmask = True
            self.current = Shape()
            self.repaint()

        if self.mode == self.NORMAL and mode == self.MASK:
            self.mode = self.MASK
            self.region_points = []
            self.show_bgmask = True
            self.current = Shape()
            self.repaint()

        if self.mode == self.NORMAL and mode == self.MEASURE:
            self.mode = self.MEASURE
            self.measure_line = []
            self.overrideCursor(CURSOR_DRAW)  # 使用 overrideCursor 而不是 setCursor
            self.show_bgmask = False
            self.repaint()

        if mode == self.BRUSH and self.mode != self.BRUSH:
            self.mode = self.BRUSH
            self.show_bgmask = False
            self.overrideCursor(QtCore.Qt.BlankCursor)  # 用绘制的圆圈代替系统光标
            self.repaint()

        if self.mode == self.BRUSH and mode == self.NORMAL:
            self.mode = self.NORMAL
            self.brush_target = None
            self.brush_painting = False
            self.brush_stroke = []
            self.restoreCursor()
            self.overrideCursor(CURSOR_DEFAULT)
            self.repaint()

        # if self.mode == self.NORMAL and mode == self.AUTO:
        #     self.mode = self.AUTO
        #     self.sam.auto_predict(self.pixmap, num_mask=3)

        # if self.mode == self.AUTO and mode == self.NORMAL:
        #     self.mode = self.NORMAL
        #     self.show_bgmask = False
      

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.pos())
        else:
            pos = self.transformPos(ev.posF())
        pos = self.clampToCanvas(pos)
        if self.isBrushing():
            if ev.button() == QtCore.Qt.LeftButton:
                self.brush_erasing = False
            elif ev.button() == QtCore.Qt.RightButton:
                self.brush_erasing = True
            else:
                return
            if self.brush_target is None:
                return
            # One snapshot per stroke, not per dab.
            self.brushStrokeStarted.emit()
            self.brush_painting = True
            self.brush_stroke = [pos]
            self.brush_dirty = self._paintBrushAt(pos, self.brush_erasing)
            self.update(self._brushDirtyRect([pos]))
            return

        if self.isdrawing():
            if ev.button() == QtCore.Qt.LeftButton:
                label = 1
            elif ev.button() == QtCore.Qt.RightButton:
                label = 0
            else:
                return
            self.sam.add_point(pos.x(), pos.y(), label)
            self.current.add_prompt_point(pos.x(), pos.y(), label)
            try:
                mask = self.sam.predict(oneDomain=True)
                polygon = compute_polygon_from_mask(mask)
                self.current.set(points=polygon, mask=mask)
            except:
                QtWidgets.QMessageBox.warning(self, 'Error', 'Invalid prediction point.\n Please try again.')
                self.sam.pop_point()
                self.current.pop_prompt_point()
            self.update()

        elif self.isMasking():
            if ev.button() == QtCore.Qt.RightButton:
                # Right click clears the drawn region so it can be redrawn,
                # without leaving the mode.
                if self.region_points:
                    self.region_points = []
                    self.update()
                return
            if ev.button() != QtCore.Qt.LeftButton:
                return
            self.region_points = [pos, pos] if self.region_mode == 'rect' else [pos]
            self.dragging = True
            self.update()
            
        elif self.isMeasuring():
            if ev.button() == QtCore.Qt.LeftButton:
                if len(self.measure_line) == 0:
                    self.measure_line.append(pos)
                    self.update()
                elif len(self.measure_line) == 1:
                    self.measure_line.append(pos)
                    self.update()
                    self.overrideCursor(CURSOR_DEFAULT)  # 使用 overrideCursor 而不是 setCursor
                    self.mode = self.NORMAL

                    # 计算像素距离（不考虑缩放因子）
                    p1 = self.measure_line[0]
                    p2 = self.measure_line[1]
                    dx = p2.x() - p1.x()
                    dy = p2.y() - p1.y()
                    pixel_distance = (dx ** 2 + dy ** 2) ** 0.5  # 不乘以缩放因子

                    # 发出信号，传递测量的像素距离
                    self.line_drawn.emit(pixel_distance)

                    self.measure_line = []
            elif ev.button() == QtCore.Qt.RightButton:
                # 右键取消测量
                self.measure_line = []
                self.overrideCursor(CURSOR_DEFAULT)  # 使用 overrideCursor 而不是 setCursor
                self.mode = self.NORMAL
                # 发出信号表示取消了测量
                self.line_drawn.emit(0.0)  # 发送0表示取消
                self.update()

        elif self.mode == self.NORMAL and ev.button() == QtCore.Qt.LeftButton:
            self._pickShapeAt(pos)
    def _pickShapeAt(self, pos):
        """命中测试 ``pos`` 处已提交的对象并发出 ``shapePicked``。

        掩膜逐像素判中；无掩膜的对象退回多边形包含测试（先过一遍缓存的
        外接框再建 QPainterPath）。重叠时取面积最小者 —— 大包围掩膜会
        "吞掉"内部小对象的点击，用户指的几乎总是上面那个小东西。
        空白处 emit None，由主窗口取消选择。
        """
        x, y = int(pos.x()), int(pos.y())
        hits = []
        for shape in self.shapes:
            if not self.isVisible(shape):
                continue
            if shape.mask is not None:
                height, width = shape.mask.shape[:2]
                if 0 <= x < width and 0 <= y < height and shape.mask[y, x]:
                    hits.append(shape)
            elif shape.points and shape.boundingRect().contains(pos):
                path = QtGui.QPainterPath(shape.points[0])
                for point in shape.points[1:]:
                    path.lineTo(point)
                path.closeSubpath()
                if path.contains(pos):
                    hits.append(shape)
        if len(hits) > 1:
            def _hit_area(shape):
                if shape.mask is not None:
                    return float(np.count_nonzero(shape.mask))
                rect = shape.boundingRect()
                return rect.width() * rect.height()
            hits.sort(key=_hit_area)
        self.shapePicked.emit(hits[0] if hits else None)

    def mouseMoveEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.pos())
        else:
            pos = self.transformPos(ev.posF())
        pos = self.clampToCanvas(pos)

        previous_pos = self.current_mouse_pos
        # 更新当前鼠标位置
        self.current_mouse_pos = pos
        
        # 如果鼠标在图像范围内，更新最后有效位置
        if (self.pixmap and 
            0 <= pos.x() < self.pixmap.width() and 
            0 <= pos.y() < self.pixmap.height()):
            self.last_valid_mouse_pos = pos
            # 发送鼠标位置信号
            self.mousePositionChanged.emit(int(pos.x()), int(pos.y()))
        else:
            # 鼠标在图像外时，发送最后有效位置
            if not self.last_valid_mouse_pos.isNull():
                self.mousePositionChanged.emit(int(self.last_valid_mouse_pos.x()), int(self.last_valid_mouse_pos.y()))
            else:
                self.mousePositionChanged.emit(-1, -1)  # 使用-1表示无效坐标
        
        if self.isBrushing():
            if self.brush_painting:
                if self._paintBrushAt(pos, self.brush_erasing, previous=previous_pos):
                    self.brush_dirty = True
                    self.brush_stroke.append(pos)
            # Both the cursor ring and the new stroke segment live between the
            # previous and the current position.
            self.update(self._brushDirtyRect([previous_pos, pos]))
        elif self.isMasking() and self.dragging:
            if self.region_mode == 'rect':
                self.region_points[-1] = pos
                self.update()
            else:
                # Decimate the lasso trail: a point every ~3 px keeps the
                # outline accurate without thousands of polygon vertices.
                last = self.region_points[-1]
                if abs(pos.x() - last.x()) + abs(pos.y() - last.y()) >= 3:
                    self.region_points.append(pos)
                    self.update()
        elif self.isMeasuring() and len(self.measure_line) == 1:
            # 在测量模式下，如果已经有第一个点，实时更新预览
            self.update()
        # Other modes draw nothing that follows the cursor, so no repaint is
        # needed here: a full repaint of a 16 MPx pixmap on every mouse-move
        # event is what used to make the canvas feel sluggish. The coordinate
        # readout is a label and updates through mousePositionChanged.

    def mouseReleaseEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.pos())
        else:
            pos = self.transformPos(ev.posF())
        pos = self.clampToCanvas(pos)
        if self.isBrushing():
            if self.brush_painting:
                self.brush_painting = False
                self.brush_stroke = []
                if self.brush_dirty:
                    self._refreshBrushTargetOutline()
                    self.brushStrokeFinished.emit(self.brush_target)
                self.brush_dirty = False
                self.update()
            return
        if self.isMasking():
            self.dragging = False
            if self.region_mode == 'rect':
                if len(self.region_points) == 2:
                    p0, p1 = self.region_points
                    x1, x2 = sorted((p0.x(), p1.x()))
                    y1, y2 = sorted((p0.y(), p1.y()))
                    if x2 - x1 >= 2 and y2 - y1 >= 2:
                        # A rectangle is just a 4-vertex polygon downstream.
                        self.region_points = [
                            QtCore.QPoint(int(vx), int(vy))
                            for vx, vy in (
                                (x1, y1), (x2, y1), (x2, y2), (x1, y2)
                            )
                        ]
                        self.maskPolygon.emit(list(self.region_points))
                    else:
                        self.region_points = []
                else:
                    self.region_points = []
            elif len(self.region_points) >= 3:
                self.maskPolygon.emit(list(self.region_points))
            else:
                self.region_points = []
            self.update()


    def _visibleImageRect(self, event):
        """Image-space rect that this paint event actually has to cover.

        ``None`` means "everything", which is what a full repaint gets. Widget
        space maps back with ``image = widget / scale - offset``; the margin
        absorbs pen widths so an outline just outside the rect is not clipped
        away visibly.
        """
        if event is None or not self.scale:
            return None
        rect = event.rect()
        if rect.isNull() or rect.contains(self.rect()):
            return None

        offset = self.offsetToCenter()
        margin = 8.0 / self.scale
        return QtCore.QRectF(
            rect.left() / self.scale - offset.x() - margin,
            rect.top() / self.scale - offset.y() - margin,
            rect.width() / self.scale + 2 * margin,
            rect.height() / self.scale + 2 * margin,
        )

    def clampToCanvas(self, pos):
        if not self.pixmap:
            return pos
        
        width = self.pixmap.width()
        height = self.pixmap.height()

        clamped_x = max(0, min(pos.x(), width - 1))
        clamped_y = max(0, min(pos.y(), height - 1))

        return QtCore.QPoint(clamped_x, clamped_y)


    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)

        if self.region_points:
            pen = QtGui.QPen(QtCore.Qt.magenta, max(2, int(round(2.0 / self.scale))), QtCore.Qt.DashLine)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush)
            if len(self.region_points) == 2:
                # Live rectangle preview while the drag is still open.
                p.drawRect(QtCore.QRect(
                    self.region_points[0], self.region_points[1]))
            else:
                path = QtGui.QPainterPath(self.region_points[0])
                for point in self.region_points[1:]:
                    path.lineTo(point)
                if not self.dragging:
                    path.closeSubpath()
                p.drawPath(path)

        Shape.scale = self.scale
        # A partial repaint (brush strokes) only needs the objects it overlaps.
        # Rebuilding a 460-point QPainterPath per object is the fixed cost of a
        # frame, so culling is what keeps a small dab cheap on a busy image.
        visible_area = self._visibleImageRect(event)
        for shape in self.shapes:
            if not self.isVisible(shape):
                continue
            if visible_area is not None:
                bounds = shape.boundingRect()
                if not bounds.isNull() and not visible_area.intersects(bounds):
                    continue
            shape.paint(p)

        if self.current:
            self.current.paint(p, prompt=True)

        if self.isBrushing():
            # Highlight the object being edited so it is obvious what the brush
            # will change, then draw the brush outline at the cursor.
            if self.brush_target is not None and self.brush_target.points:
                pen = QtGui.QPen(QtGui.QColor(0, 200, 255),
                                 max(2, int(round(3.0 / self.scale))))
                p.setPen(pen)
                p.setBrush(QtCore.Qt.NoBrush)
                path = QtGui.QPainterPath()
                path.moveTo(self.brush_target.points[0])
                for point in self.brush_target.points:
                    path.lineTo(point)
                p.drawPath(path)

            color = QtGui.QColor(255, 80, 80) if self.brush_erasing else QtGui.QColor(80, 255, 120)

            # Live stroke feedback. The mask itself is edited immediately but is
            # not what gets drawn (rendering a 16 MPx mask per frame is far too
            # slow) and the outline is only recomputed on release, so without
            # this ribbon a stroke would look like nothing was happening.
            if self.brush_stroke:
                ribbon = QtGui.QColor(color)
                ribbon.setAlpha(110)
                stroke_pen = QtGui.QPen(
                    ribbon, self.brush_radius * 2,
                    QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin,
                )
                p.setPen(stroke_pen)
                p.setBrush(QtCore.Qt.NoBrush)
                if len(self.brush_stroke) == 1:
                    p.drawPoint(self.brush_stroke[0])
                else:
                    stroke_path = QtGui.QPainterPath()
                    stroke_path.moveTo(self.brush_stroke[0])
                    for point in self.brush_stroke[1:]:
                        stroke_path.lineTo(point)
                    p.drawPath(stroke_path)

            p.setPen(QtGui.QPen(color, max(1, int(round(2.0 / self.scale))), QtCore.Qt.SolidLine))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawEllipse(self.current_mouse_pos, self.brush_radius, self.brush_radius)

        if self.show_bgmask:
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(0, 0, 0, 100))  # 黑色半透明
            p.drawRect(0, 0, self.pixmap.width(), self.pixmap.height())

        # 保存当前的painter状态
        p.save()
        
        if self.isMeasuring():
            # 显示测量模式的提示 (文字不受缩放影响)
            if len(self.measure_line) == 0:
                # 还没有点击任何点
                # 重置变换，让文字不受缩放影响
                p.resetTransform()
                p.setPen(QtGui.QPen(QtGui.QColor("yellow")))
                font = QtGui.QFont()
                font.setPointSize(18)  # 放大字体
                font.setBold(True)
                p.setFont(font)
                p.drawText(15, 40, "Measuring Mode: Click to set first point")
                p.drawText(15, 70, "Right-click to cancel measurement")
                # 恢复变换用于绘制其他元素
                p.scale(self.scale, self.scale)
                p.translate(self.offsetToCenter())
            elif len(self.measure_line) == 1:
                # 显示正在绘制的线（从第一个点到当前鼠标位置）
                pen = QtGui.QPen(QtGui.QColor("cyan"), max(2, int(round(2.0 / self.scale))), QtCore.Qt.DashLine)
                p.setPen(pen)
                p.drawLine(self.measure_line[0], self.current_mouse_pos)
                
                # 显示第一个点
                pen_point = QtGui.QPen(QtGui.QColor("red"), max(4, int(round(4.0 / self.scale))))
                p.setPen(pen_point)
                p.drawEllipse(self.measure_line[0], 3, 3)
                
                # 显示实时长度（不考虑缩放因子）
                dx = self.current_mouse_pos.x() - self.measure_line[0].x()
                dy = self.current_mouse_pos.y() - self.measure_line[0].y()
                distance = math.hypot(dx, dy)  # 不乘以缩放因子
                midpoint = QtCore.QPointF(
                    (self.measure_line[0].x() + self.current_mouse_pos.x()) / 2,
                    (self.measure_line[0].y() + self.current_mouse_pos.y()) / 2
                )
                
                # 保存当前变换，重置变换让文字不受缩放影响
                p.save()
                p.resetTransform()
                # 将midpoint转换到屏幕坐标
                screen_midpoint = QtCore.QPointF(
                    midpoint.x() * self.scale + self.offsetToCenter().x() * self.scale,
                    midpoint.y() * self.scale + self.offsetToCenter().y() * self.scale
                )
                p.setPen(QtGui.QPen(QtGui.QColor("white")))
                font = QtGui.QFont()
                font.setPointSize(20)  # 进一步放大字体到20pt
                font.setBold(True)
                p.setFont(font)
                p.drawText(screen_midpoint, f"{distance:.1f} px")
                p.restore()  # 恢复变换
                
                # 显示提示 (文字不受缩放影响)
                p.resetTransform()
                p.setPen(QtGui.QPen(QtGui.QColor("yellow")))
                font = QtGui.QFont()
                font.setPointSize(18)  # 放大字体
                font.setBold(True)
                p.setFont(font)
                p.drawText(15, 40, "Click to set second point to complete")
                p.drawText(15, 70, "Right-click to cancel measurement")
                # 恢复变换
                p.scale(self.scale, self.scale)
                p.translate(self.offsetToCenter())
                
            elif len(self.measure_line) == 2:
                # 显示完整的测量线
                pen = QtGui.QPen(QtGui.QColor("cyan"), 2, QtCore.Qt.SolidLine)
                p.setPen(pen)
                p.drawLine(self.measure_line[0], self.measure_line[1])

                # 显示测量长度（不考虑缩放因子）
                dx = self.measure_line[1].x() - self.measure_line[0].x()
                dy = self.measure_line[1].y() - self.measure_line[0].y()
                distance = math.hypot(dx, dy)  # 不乘以缩放因子
                midpoint = QtCore.QPointF(
                    (self.measure_line[0].x() + self.measure_line[1].x()) / 2,
                    (self.measure_line[0].y() + self.measure_line[1].y()) / 2
                )
                
                # 保存当前变换，重置变换让文字不受缩放影响
                p.save()
                p.resetTransform()
                # 将midpoint转换到屏幕坐标
                screen_midpoint = QtCore.QPointF(
                    midpoint.x() * self.scale + self.offsetToCenter().x() * self.scale,
                    midpoint.y() * self.scale + self.offsetToCenter().y() * self.scale
                )
                p.setPen(QtGui.QPen(QtGui.QColor("white")))
                font = QtGui.QFont()
                font.setPointSize(20)  # 进一步放大字体到20pt
                font.setBold(True)
                p.setFont(font)
                p.drawText(screen_midpoint, f"{distance:.1f} px")
                p.restore()  # 恢复变换
        
        # 绘制颜色校正标记
        if self.color_checker_centers:
            for i, (cx, cy) in enumerate(self.color_checker_centers):
                # 绘制简约的圆形标记
                # 外圈：白色边框，提供对比度
                pen = QtGui.QPen(QtCore.Qt.white, max(2, int(round(3.0 / self.scale))))
                p.setPen(pen)
                radius = max(6, int(round(12.0 / self.scale)))
                p.drawEllipse(QtCore.QPoint(cx, cy), radius, radius)
                
                # 内圈：半透明蓝色填充
                p.setBrush(QtGui.QBrush(QtGui.QColor(0, 120, 255, 180)))  # 蓝色，透明度70%
                p.setPen(QtCore.Qt.NoPen)
                inner_radius = max(4, int(round(8.0 / self.scale)))
                p.drawEllipse(QtCore.QPoint(cx, cy), inner_radius, inner_radius)
                
                # 中心点：白色小点
                p.setBrush(QtGui.QBrush(QtCore.Qt.white))
                center_radius = max(1, int(round(2.0 / self.scale)))
                p.drawEllipse(QtCore.QPoint(cx, cy), center_radius, center_radius)
                
                # 重置画刷
                p.setBrush(QtCore.Qt.NoBrush)
        
        if len(self.ruler_ticks) > 0:
            pen = QtGui.QPen(QtGui.QColor("cyan"), max(2, int(round(2.0 / self.scale))), QtCore.Qt.SolidLine)
            p.setPen(pen)
            
            # 设置实心填充
            brush = QtGui.QBrush(QtGui.QColor("cyan"))
            p.setBrush(brush)  # 设置填充颜色为青色

            for tick in self.ruler_ticks:
                x, y = tick
                # 原本的半径大小为 max(2, int(round(4.0 / self.scale)))
                # 乘以2以加大一倍
                size = max(2, int(round(4.0 / self.scale))) * 2
                p.drawEllipse(int(x), int(y), size, size)  # 画更大的圆


        # 恢复painter状态
        p.restore()

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        if QT5:
            return QtCore.QPoint(int(x), int(y))
        else:
            return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w and 0 <= p.y() <= h)

    # These two, along with a call to adjustSize are required for the scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if self.isBrushing():
            # In brush mode a bare wheel resizes the brush, which is the gesture
            # needed constantly while retouching; Ctrl+wheel still zooms the
            # canvas, so the same modifier means "zoom" in every mode.
            mods = int(ev.modifiers())
            if not (mods & int(QtCore.Qt.ControlModifier)):
                delta = ev.angleDelta().y() if QT5 else ev.delta()
                if delta:
                    self.adjustBrushRadius(delta)
                ev.accept()
                return
            self.zoomRequest.emit(
                ev.angleDelta().y() if QT5 else ev.delta(), ev.pos()
            )
            ev.accept()
            return

        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical)
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()


    def loadQImage(self, QImage, update_sam=True):
        self.pixmap = QtGui.QPixmap.fromImage(QImage)
        if update_sam:
            self.sam.set_image(qimage_to_cv2(QImage))
        # 发送图像尺寸变化信号
        self.mousePositionChanged.emit(-1, -1)
        self.repaint()


    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.shapes = []
        self.totalShapes = 0
        self.current = None
        self.pixmap = QtGui.QPixmap()
        self.selectedShapes = []
        self.visible = {}
        self.region_points = []
        # 重置测量状态
        self.measure_line = []
        self.measure_callback = None
        self.mode = self.NORMAL
        self.last_valid_mouse_pos = QtCore.QPoint()  # 重置最后有效坐标
        # 重置颜色校正标记
        self.color_checker_centers = []
        self.ruler_ticks = []
        self.brush_target = None
        self.brush_painting = False
        self.brush_stroke = []
        self.repaint()

    def registerShape(self, shape, prefix=None):
        """Assign a uid plus a display name, then announce the new shape.

        ``prefix`` overrides the naming term; otherwise ``shape.name_prefix``
        is used, falling back to the generic "Object".
        """
        self.totalShapes += 1
        shape.uid = self.totalShapes
        if prefix is not None:
            shape.name_prefix = prefix
        if not shape.label:
            shape.label = next_label_for_prefix(
                self.shapes, shape.name_prefix or DEFAULT_LABEL_PREFIX
            )
        self.newShape.emit(shape)
        return shape

    def _commitAutoShapes(self, shapes):
        """Commit worker-generated shapes on the Qt UI thread."""
        for shape in shapes:
            self.shapes.append(shape)
            self.registerShape(shape)
        self.update()

    def addLoadedShapes(self, shapes):
        """Add shapes restored from a sidecar file.

        Their names come from the file and must not be regenerated, so each one
        keeps whatever ``label`` it was loaded with; only the internal uid is
        assigned fresh.
        """
        for shape in shapes:
            self.shapes.append(shape)
            self.registerShape(shape)
        self.update()
        return len(shapes)

    def selectShape(self, shape):
        self.deSelectShape()
        shape.selected = True
        self.selectedShapes = [shape]
        # self.selectionChanged.emit(True)
        self.repaint()

    def setSelectedShapes(self, shapes):
        """Highlight exactly ``shapes`` (possibly empty) on the canvas."""
        for shape in self.shapes:
            shape.selected = False
        selected = [shape for shape in shapes if shape is not None]
        for shape in selected:
            shape.selected = True
        self.selectedShapes = selected
        self.repaint()

    def deSelectShape(self):
        if self.selectedShapes:
            for shape in self.selectedShapes:
                shape.selected = False
            self.selectedShapes = []
            # self.selectionChanged.emit(False)
            self.repaint()

    @property
    def selectedShape(self):
        """The most recently highlighted shape, or ``None``."""
        return self.selectedShapes[-1] if self.selectedShapes else None

    def deleteShapes(self, shapes):
        """Remove the given shapes and return the ones actually removed."""
        removed = []
        for shape in shapes:
            if shape in self.shapes:
                self.shapes.remove(shape)
                self.visible.pop(shape, None)
                shape.selected = False
                removed.append(shape)
        self.selectedShapes = [
            shape for shape in self.selectedShapes if shape not in removed
        ]
        if removed:
            self.repaint()
        return removed

    def deleteSelected(self):
        removed = self.deleteShapes(list(self.selectedShapes))
        return removed[-1] if removed else None

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.repaint()

    def autoCount_centers(self, img, mask=None, bounds=None):
        return [(b[0], b[1])
                for b in self.autoCount_blobs(img, mask=mask, bounds=bounds)]

    def autoCount_blobs(self, img, mask=None, bounds=None):
        """KMeans 二分出前景连通域，返回 (cx, cy, x1, y1, x2, y2) 列表（原图坐标）。"""
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        if bounds is not None:
            xmin, ymin, xmax, ymax = bounds
            crop_img = img[ymin:ymax + 1, xmin:xmax + 1, :]
            crop_mask = np.ones(crop_img.shape[:2], dtype=np.uint8)
        elif mask is not None:
            mask = mask.astype(np.uint8)
            mask_y, mask_x = np.where(mask)
            if len(mask_y) == 0:
                return []
            ymin, ymax = np.min(mask_y), np.max(mask_y)
            xmin, xmax = np.min(mask_x), np.max(mask_x)
            crop_img = img[ymin:ymax + 1, xmin:xmax + 1, :]
            crop_mask = mask[ymin:ymax + 1, xmin:xmax + 1]
        else:
            xmin, ymin = 0, 0
            crop_img = img
            crop_mask = np.ones_like(img[:, :, 0], dtype=np.uint8)

        if crop_img.size == 0:
            return []

        h_raw, w_raw, _ = crop_img.shape

        # 进行resize
        resize_factor = 1024.0 / max(h_raw, w_raw) if max(h_raw, w_raw) > 1024 else 1.0
        small_img = cv2.resize(crop_img, (0, 0), fx=resize_factor, fy=resize_factor)  # 经过resize后的图像
        small_mask = cv2.resize(crop_mask, (small_img.shape[1], small_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        small_img = small_img * small_mask[:, :, np.newaxis]  # 应用mask

        h, w, _ = small_img.shape

        # 只考虑mask为1的有效区域的像素
        pixels = small_img[small_mask == 1]  # 提取mask为1的像素点
        pixels = pixels.reshape(-1, 3)  # 将有效像素展平为一维数组

        # 获取边缘像素作为背景样本
        edge_width = 3
        top = small_img[:edge_width, :, :]
        bottom = small_img[-edge_width:, :, :]
        left = small_img[:, :edge_width, :]
        right = small_img[:, -edge_width:, :]
        edge_pixels = np.vstack([top.reshape(-1, 3), bottom.reshape(-1, 3),
                                left.reshape(-1, 3), right.reshape(-1, 3)])
        edge_pixels = edge_pixels[np.all(edge_pixels != 0, axis=1)]  # 去除黑色像素点

        if len(pixels) < 2:
            return []

        bg_color = edge_pixels.mean(axis=0) if len(edge_pixels) else pixels.mean(axis=0)

        # A small deterministic sample is sufficient to estimate the two color
        # centers. Assign all pixels only after the centers have been fitted.
        max_samples = 20000
        if len(pixels) > max_samples:
            sample_indices = np.linspace(0, len(pixels) - 1, max_samples, dtype=np.int64)
            fit_pixels = pixels[sample_indices]
        else:
            fit_pixels = pixels

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.2,
        )
        cv2.setRNGSeed(0)
        _, _, centers = cv2.kmeans(
            fit_pixels.astype(np.float32),
            2,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )

        pixels_float = pixels.astype(np.float32)
        distance_0 = np.sum((pixels_float - centers[0]) ** 2, axis=1)
        distance_1 = np.sum((pixels_float - centers[1]) ** 2, axis=1)
        all_labels = (distance_1 < distance_0).astype(np.uint8)

        # 判断哪个类别是背景
        dists = np.linalg.norm(centers - bg_color, axis=1)
        bg_label = np.argmin(dists)
        fg_label = 1 - bg_label

        # 生成前景mask并限制在mask区域
        binary_mask = np.zeros_like(small_mask, dtype=np.uint8)  # 创建一个大小为 small_mask 的全零数组
        binary_mask[small_mask == 1] = (all_labels == fg_label).astype(np.uint8)  # 仅在mask区域内赋值

        # import time
        # cv2.imwrite(f"debug_auto_count_binary_mask_{int(time.time())}.png", binary_mask * 255)

        # 连通域分析 + 面积过滤
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=4)

        blobs = []  # (cx, cy, x1, y1, x2, y2)，原图坐标

        for i in range(1, num_labels):  # 忽略背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= h * w * 0.001:
                bw = (labels == i).astype(np.uint8) * 255
                dt = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
                _, _, _, maxLoc = cv2.minMaxLoc(dt)
                cx, cy = maxLoc
                blobs.append((
                    float(cx) / resize_factor + xmin,
                    float(cy) / resize_factor + ymin,
                    stats[i, cv2.CC_STAT_LEFT] / resize_factor + xmin,
                    stats[i, cv2.CC_STAT_TOP] / resize_factor + ymin,
                    (stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH])
                        / resize_factor + xmin,
                    (stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT])
                        / resize_factor + ymin,
                ))

        return blobs

    def rmColorCheckerAndRulerRegions(self, mask):
        if self.parent.color_correction_data is not None:
            centers = self.parent.color_correction_data['centers']
            self.sam.add_points(centers, 1)
            color_checker_mask = self.sam.predict(oneDomain=False)
            mask[color_checker_mask > 0] = 0  # 去除颜色校正区域
            self.sam.reset_points()
        if self.parent.ruler_detection_data is not None:
            tickpoints = self.parent.ruler_detection_data['tickpoints']
            self.sam.add_points(tickpoints, 1)
            ruler_mask = self.sam.predict(oneDomain=False)
            mask[ruler_mask > 0] = 0  # 去除标尺区域
            self.sam.reset_points()
        return mask

    def _text_candidates_from_detections(self, detections, prompt_pairs):
        candidates = []
        boxes = [detection["box"] for detection in detections]
        masks = self.sam.predict_boxes(boxes)
        for detection, mask in zip(detections, masks):
            if mask is None or not np.any(mask):
                continue

            # 一个框解出多连通域 mask = 检测把一群个体合成了一框——
            # 逐个连通域各出一个候选（灰板种子阵实测就是这个形态）。
            parts = _split_mask_components(mask)
            for part in parts:
                # 拆出的 part 得用自己的外接框代替共享的检测框——否则
                # 去重环节的 box 包含检查会把同框兄弟全判成重复。
                cand_box = detection.get("box")
                if len(parts) > 1:
                    pys, pxs = np.where(part)
                    cand_box = np.array(
                        [pxs.min(), pys.min(), pxs.max() + 1, pys.max() + 1],
                        dtype=np.float32)
                traits = _text_mask_traits(part)
                if traits is None:
                    continue

                polygon = compute_polygon_from_mask(part)
                if len(polygon) == 0:
                    continue

                shape = Shape()
                shape.set(points=polygon, mask=part)
                shape.source_prompt = detection.get("label")
                shape.detection_score = detection.get("score")
                if prompt_pairs:
                    shape.name_prefix = sanitize_label_prefix(
                        TextDetector.term_for_label(detection.get("label"), prompt_pairs)
                    )
                candidates.append({
                    "shape": shape,
                    "label": detection.get("label"),
                    "score": float(detection.get("score") or 0.0),
                    "mask": part,
                    "box": cand_box,
                    "traits": traits,
                })
        return candidates

    def _retry_merged_detection(self, candidate, image_rgb, detector, prompt,
                                prompt_pairs):
        """Re-detect inside a lone blob's bounding box.

        When the text detector merges a whole pile of objects into one box,
        SAM expands it into a single mask covering the cluster. Inside a crop
        of that box the objects fill the detector's input and resolve as
        individuals -- the same effect that rescues region detection.
        Returns the retry's candidate list when it finds 2+ objects, else None.
        """
        mask = np.asarray(candidate["mask"])
        coverage = float(np.count_nonzero(mask)) / float(mask.size)
        if coverage <= 0.05:
            return None
        ys, xs = np.where(mask > 0)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        margin_x = int(round((x2 - x1 + 1) * 0.12))
        margin_y = int(round((y2 - y1 + 1) * 0.12))
        height, width = image_rgb.shape[:2]
        cx1 = max(0, x1 - margin_x)
        cy1 = max(0, y1 - margin_y)
        cx2 = min(width - 1, x2 + margin_x)
        cy2 = min(height - 1, y2 + margin_y)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop_rgb = image_rgb[cy1:cy2 + 1, cx1:cx2 + 1]
        try:
            if hasattr(detector, "detect_with_report"):
                retry_detections, _ = detector.detect_with_report(crop_rgb, prompt)
            else:
                retry_detections = detector.detect(crop_rgb, prompt)
        except Exception:
            return None
        retry_detections = [
            detection for detection in retry_detections or []
            if detection.get("box") is not None
        ]
        if len(retry_detections) < 2:
            return None
        offset = np.array([cx1, cy1, cx1, cy1], dtype=np.float32)
        shifted = []
        for detection in retry_detections:
            shifted_detection = dict(detection)
            shifted_detection["box"] = (
                np.asarray(detection["box"], dtype=np.float32) + offset)
            shifted.append(shifted_detection)
        return self._text_candidates_from_detections(shifted, prompt_pairs)

    def segmentTextDetections(self, detections, request_id=None, prompt=None,
                              image_rgb=None, detector=None):
        # Objects detected from text are named after the concept the user
        # typed, so ``pod`` yields pod01, pod02, ... instead of Object01.
        prompt_pairs = TextDetector.split_prompt_pairs(prompt) if prompt else []

        candidates = self._text_candidates_from_detections(
            detections, prompt_pairs)
        if not candidates:
            return {"created": 0, "filtered": 0}

        # Two-stage filter: first drop masks that overlap a better-scoring
        # detection (same object found twice), then the size/shape outliers.
        deduped, overlap_dropped = _dedupe_overlapping_masks(candidates)
        accepted, filtered = _filter_text_mask_candidates(deduped)
        filtered = list(filtered) + list(overlap_dropped)

        # A lone surviving mask over a large share of the frame is usually a
        # merged cluster, not one real object. One retry inside its bounding
        # box resolves the individuals; 2+ objects replaces it, otherwise the
        # original blob stands (it may genuinely be a single organ).
        if len(accepted) == 1 and image_rgb is not None and detector is not None:
            retry = self._retry_merged_detection(
                accepted[0], image_rgb, detector, prompt, prompt_pairs)
            if retry:
                retry_deduped, retry_dropped = _dedupe_overlapping_masks(retry)
                retry_accepted, retry_filtered = _filter_text_mask_candidates(
                    retry_deduped)
                if len(retry_accepted) >= 2:
                    accepted = retry_accepted
                    filtered = list(retry_filtered) + list(retry_dropped)
        shapes = [candidate["shape"] for candidate in accepted]

        if request_id is not None and request_id != getattr(self.parent, '_sam_request_id', None):
            return {"created": 0, "filtered": 0}

        if shapes:
            if QtCore.QThread.currentThread() == self.thread():
                self._commitAutoShapes(shapes)
            else:
                self.autoShapesReady.emit(shapes)

        return {"created": len(shapes), "filtered": len(filtered)}


    def segmentGreenLeafs(self, qimage=None):
        """按颜色分割绿色叶片并提交为对象。

        与语义检测并行的一条路（原理与适用范围见 pheno/greenSeg.py 的说明）：
        剪下的叶片摆在灯箱上时，颜色比大模型更靠得住 —— 不受"整图被压到千像素级"
        的影响，也不需要提示词。批量回放走同一个入口，录制与回放行为一致。
        """
        from pheno import greenSeg
        from pheno.lib import qimage_to_cv2

        source = qimage if qimage is not None else getattr(
            self.parent, 'original_image', None)
        if source is None:
            return {'created': 0, 'filtered': 0}

        image = qimage_to_cv2(source)
        if image is None:
            return {'created': 0, 'filtered': 0}

        candidates = greenSeg.segment_leaves(image)
        shapes = []
        for candidate in candidates:
            polygon = compute_polygon_from_mask(candidate['mask'])
            if len(polygon) == 0:
                continue
            shape = Shape()
            shape.set(points=polygon, mask=candidate['mask'])
            shape.name_prefix = GREEN_LEAF_PREFIX
            shapes.append(shape)

        if not shapes:
            return {'created': 0, 'filtered': len(candidates)}

        if QtCore.QThread.currentThread() == self.thread():
            self._commitAutoShapes(shapes)
        else:
            self.autoShapesReady.emit(shapes)

        return {'created': len(shapes), 'filtered': 0}

    def segmentClickPoints(self, points):
        """Replay recorded click prompt points as one object.

        ``points`` is a list of ``{"x", "y", "label"}`` dicts captured while
        recording a batch. Runs on the batch worker thread, so the resulting
        shape is committed through the same queued signal as the other
        worker-generated shapes.
        """
        if not points:
            return {"created": 0, "filtered": 0}

        request_id = getattr(self.parent, '_sam_request_id', None)
        self.sam.reset_points()
        try:
            for point in points:
                self.sam.add_point(point["x"], point["y"], point["label"])
            mask = self.sam.predict(oneDomain=True)
        finally:
            self.sam.reset_points()

        if mask is None or not np.any(mask):
            return {"created": 0, "filtered": 0}

        polygon = compute_polygon_from_mask(mask)
        if len(polygon) == 0:
            return {"created": 0, "filtered": 0}

        if request_id is not None and request_id != getattr(self.parent, '_sam_request_id', None):
            return {"created": 0, "filtered": 0}

        shape = Shape()
        shape.set(points=polygon, mask=mask)
        for point in points:
            shape.add_prompt_point(point["x"], point["y"], point["label"])

        if QtCore.QThread.currentThread() == self.thread():
            self._commitAutoShapes([shape])
        else:
            self.autoShapesReady.emit([shape])

        return {"created": 1, "filtered": 0}

    def autoPredict(self, points, samples_count):
        request_id = getattr(self.parent, '_sam_request_id', None)
        if not getattr(self.parent, '_sam_ready', True):
            raise RuntimeError("Segmentation model is still preparing the current image.")

        original_image = qimage_to_cv2(self.parent.original_image)
        if original_image.shape[-1] == 4:  # RGBA -> RGB
            original_image = original_image[:, :, :3]
        height, width = original_image.shape[:2]
        # The drawn region is a freehand polygon; everything downstream crops
        # to its bounding box and clips masks against the polygon itself.
        pts = np.array(
            [[p.x(), p.y()] for p in points], dtype=np.int32
        ).reshape(-1, 1, 2)
        if len(pts) < 3:
            return 0
        pts[:, :, 0] = np.clip(pts[:, :, 0], 0, width - 1)
        pts[:, :, 1] = np.clip(pts[:, :, 1], 0, height - 1)
        x_min, y_min = pts[:, 0, :].min(axis=0)
        x_max, y_max = pts[:, 0, :].max(axis=0)
        if x_min >= x_max or y_min >= y_max:
            return 0
        region_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(region_mask, [pts], 1)

        # 提案统一走 box prompt：经典连通域的外接框天然带着对象范围，比
        # 点 prompt 更不容易把相邻个体粘成一个 mask。原来的 everything-mode
        # fallback（auto_predict 网格采样）比这套贵一个数量级，密集小目标
        # 上还尽切碎片 —— 已由语义补充路径取代。
        blobs = self.autoCount_blobs(original_image, mask=region_mask)
        pad = max(4, int(min(width, height) * 0.002))
        boxes = [
            (max(0.0, bx1 - pad), max(0.0, by1 - pad),
             min(float(width), bx2 + pad), min(float(height), by2 + pad))
            for _cx, _cy, bx1, by1, bx2, by2 in blobs
        ]

        masks = []
        if boxes:
            print(f"Region proposals: {len(boxes)} blob boxes")
            masks = [np.asarray(m) > 0 for m in self.sam.predict_boxes(boxes)
                     if m is not None]
        if samples_count is not None and len(masks) != samples_count:
            print(f"[Warning] Region produced {len(masks)} masks, expected {samples_count}")

        # 经典提案欠检（计数不符）或未指定期望数时，用语义检测补充个体
        if samples_count is None or len(masks) != samples_count:
            extra = self._semantic_region_masks(
                original_image, x_min, y_min, x_max, y_max, region_mask,
                expected_count=samples_count or len(masks),
                existing=masks)
            masks = masks + list(extra)

        # Point-prompt masks are bounded by the polygon's bounding box; clip
        # them to the drawn polygon so nothing outside the lasso survives.
        masks = [
            clipped
            for mask in masks
            if np.any(clipped := (np.asarray(mask) > 0) & (region_mask > 0))
        ]
        # 一个 box prompt 也可能解出多连通域 mask——拆开后再过边缘/去重
        # 过滤，主对象保住、附带的碎片被单独判死。
        masks = [
            part
            for mask in masks
            for part in _split_mask_components(mask)
        ]
        masks = _drop_edge_artifacts(
            masks, _region_edge_band(region_mask), region_mask)
        masks = _dedupe_region_masks(masks)
        masks = _filter_region_outliers(masks)

        shapes = []
        for mask in masks:
            polygon = compute_polygon_from_mask(mask)
            shape = Shape()
            shape.set(points=polygon, mask=mask)
            shapes.append(shape)

        if not shapes:
            return 0

        if request_id != getattr(self.parent, '_sam_request_id', None):
            return 0

        if QtCore.QThread.currentThread() == self.thread():
            self._commitAutoShapes(shapes)
        else:
            self.autoShapesReady.emit(shapes)

        return len(shapes)

    def _semantic_region_masks(self, image_bgr, x_min, y_min, x_max, y_max,
                               region_mask, expected_count=None,
                               existing=None):
        detector = getattr(self.parent, 'text_detector', None)
        if detector is None:
            try:
                detector = TextDetector(device=self.sam.device)
            except Exception:
                return []
            self.parent.text_detector = detector

        # 整图检测按图像指纹缓存：同一图上的后续 Region 调用零成本复用；
        # 批处理里每张图也只付一次 DINO（原来是 crop + 2x2 切块共 5 次）。
        rgb = image_bgr[:, :, ::-1]
        boxes = self._cached_detect_boxes(detector, rgb)
        region_boxes = [b for b in boxes
                        if _box_center_in_mask(b, region_mask)]

        if (not region_boxes) or (expected_count
                and len(region_boxes) < max(2, expected_count // 2)):
            # 整图检测容易把一簇密集小目标合并成单框。只在明显欠检时才退回
            # crop + 2x2 切块找回个体 —— 无条件跑是每图 5 次 DINO。
            region_boxes = self._tiled_region_boxes(
                detector, image_bgr, x_min, y_min, x_max, y_max, region_boxes)

        # 框心落在已有 mask 内、且该 mask 面积与框同量级 => 同一对象的二次
        # 提案，提前在 box 层面跳过，省一轮 SAM decode。注意不能把"框心被
        # 覆盖"单独当判据：经典提案失败时整簇会是一个巨 mask，DINO 的个体
        # 框框心全落在簇内——它们是补漏的唯一来源，必须放行去拆簇。
        if existing:
            areas = [np.count_nonzero(m) for m in existing]
            novel = []
            for b in region_boxes:
                box_area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                duplicate = any(
                    a <= 1.6 * box_area and _box_center_in_mask(b, m)
                    for m, a in zip(existing, areas))
                if not duplicate:
                    novel.append(b)
            region_boxes = novel
        if not region_boxes:
            return []
        out = []
        for mask in self.sam.predict_boxes(region_boxes) or []:
            if mask is None:
                continue
            mask = (np.asarray(mask) > 0) & (region_mask > 0)
            if np.any(mask):
                out.append(mask.astype(np.uint8))
        return out

    def _cached_detect_boxes(self, detector, rgb):
        """整图 "seed" 检测框，按 (图像指纹) 缓存到本 canvas。"""
        key = TextDetector._image_fingerprint(rgb)
        if getattr(self, "_sem_detect_key", None) != key:
            try:
                detections = detector.detect(rgb, "seed")
            except Exception:
                detections = []
            self._sem_detect_key = key
            self._sem_detect_boxes = [
                np.asarray(d["box"], dtype=np.float32)
                for d in detections or [] if d.get("box") is not None]
        return self._sem_detect_boxes

    def _tiled_region_boxes(self, detector, image_bgr,
                            x_min, y_min, x_max, y_max, seed_boxes):
        """Region crop + 2x2 重叠切块检测，返回与 seed_boxes 合并去重后的框。"""
        crop_rgb = image_bgr[y_min:y_max + 1, x_min:x_max + 1][:, :, ::-1]
        detect_jobs = [(crop_rgb, 0, 0)]
        tile_h, tile_w = crop_rgb.shape[:2]
        overlap_x, overlap_y = int(tile_w * 0.15), int(tile_h * 0.15)
        for ty in range(2):
            for tx in range(2):
                tx1 = max(0, tx * tile_w // 2 - overlap_x)
                ty1 = max(0, ty * tile_h // 2 - overlap_y)
                tx2 = min(tile_w, (tx + 1) * tile_w // 2 + overlap_x)
                ty2 = min(tile_h, (ty + 1) * tile_h // 2 + overlap_y)
                if tx2 - tx1 < 300 or ty2 - ty1 < 300:
                    continue  # region too small to tile usefully
                detect_jobs.append((crop_rgb[ty1:ty2, tx1:tx2], tx1, ty1))
        boxes = []
        for tile_rgb, off_x, off_y in detect_jobs:
            try:
                tile_detections = detector.detect(tile_rgb, "seed")
            except Exception:
                continue
            offset = np.array(
                [x_min + off_x, y_min + off_y,
                 x_min + off_x, y_min + off_y], dtype=np.float32)
            boxes.extend(
                np.asarray(detection["box"], dtype=np.float32) + offset
                for detection in tile_detections or []
                if detection.get("box") is not None)
        merged = list(seed_boxes)
        for box in sorted(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1])):
            if all(_box_iou(box, kept) < 0.8 for kept in merged):
                merged.append(box)
        return merged
            
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            # print("Drag Enter")
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')):
                if self.parent.mayContinue():
                    self.parent.loadFile(file_path)
                    break
    
    def start_drawing_measure_line(self):
        """启动测量模式"""
        self.setMode(self.MEASURE)
    
    def set_color_checker_markers(self, centers):
        """设置颜色校正标记的中心点"""
        # centers 可能是 numpy 数组 —— 真值判断必须走 is None，
        # `if centers` 会触发 ndarray 歧义报错。
        self.color_checker_centers = list(centers) if centers is not None else []
        self.repaint()
    
    def clear_color_checker_markers(self):
        """清除颜色校正标记"""
        self.color_checker_centers = []
        self.repaint()

    def set_ruler_ticks(self, ticks):
        """设置标尺刻度信息"""
        self.ruler_ticks = ticks
        self.repaint()


    def clear_ruler_ticks(self):
        """清除标尺刻度信息"""
        self.ruler_ticks = []
        self.repaint()
