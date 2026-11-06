from qtpy import QtGui, QtCore


# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


DEFAULT_LINE_COLOR = QtGui.QColor(0, 255, 0, 128)
DEFAULT_FILL_COLOR = QtGui.QColor(255, 0, 0, 128)
DEFAULT_SELECT_LINE_COLOR = QtGui.QColor(255, 255, 255)
DEFAULT_SELECT_FILL_COLOR = QtGui.QColor(0, 128, 255, 155)



class Shape(object):

    # The following class variables influence the drawing of all shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    scale = 1.0

    def __init__(self):
        self.points = []
        # Cache behind boundingRect(); invalidated whenever points change.
        self._bounds = None
        self.mask = None
        # ``label`` is the display name shown in the Objects panel and exported
        # as the "Name" column. It is user-editable and may be renamed freely,
        # so it must never be used as an identity key.
        self.label = None
        # ``uid`` is the internal, never-reused identity of this object. It is
        # assigned once by the canvas, is not shown anywhere in the UI, and is
        # what code should rely on when it needs a stable unique key.
        self.uid = None
        # Naming hint: the user-facing term this object came from, e.g. the
        # text-detection prompt. ``None`` means the generic "Object" prefix.
        self.name_prefix = None
        # Text-detection provenance (unused by the generic pipeline).
        self.source_prompt = None
        self.detection_score = None
        self.prompt_points = []
        self.prompt_labels = []
        self.fill = True
        self.selected = False


    def paint(self, painter, prompt=False):
        if len(self.points) == 0:
            return
        color = self.select_line_color \
            if self.selected else self.line_color
        pen = QtGui.QPen(color)
        # Try using integer sizes for smoother drawing(?)
        pen.setWidth(max(2, int(round(2.0 / self.scale))))
        painter.setPen(pen)

        line_path = QtGui.QPainterPath()
        line_path.moveTo(self.points[0])

        for p in self.points:
            line_path.lineTo(p)

        painter.drawPath(line_path)

        if self.fill:
            color = self.select_fill_color \
                if self.selected else self.fill_color
            painter.fillPath(line_path, color)

        if prompt:
            radius = min(20, int(round(20 / self.scale)))
            for i, p in enumerate(self.prompt_points):
                if self.prompt_labels[i] == 0:
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0)))
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
                else:
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                painter.drawEllipse(p, radius, radius)


    def set(self, points=None, mask=None, label=None, line_color=None):
        if points is not None:
            points = [QtCore.QPointF(point[0], point[1]) for point in points]
            self.points = points
            self._bounds = None
        if mask is not None:
            self.mask = mask
        if label is not None:
            self.label = label
        if line_color is not None:
            self.line_color = line_color

    def boundingRect(self):
        """Cached outline bounds, used by the canvas to skip offscreen shapes.

        Outlines carry a few hundred points each, so rebuilding a QPainterPath
        for every object on every repaint is the fixed cost of a frame; this lets
        the canvas drop the objects a partial repaint cannot touch. The cache is
        dropped whenever the points change.
        """
        if self._bounds is None:
            if not self.points:
                self._bounds = QtCore.QRectF()
            else:
                xs = [point.x() for point in self.points]
                ys = [point.y() for point in self.points]
                left, top = min(xs), min(ys)
                self._bounds = QtCore.QRectF(
                    left, top, max(xs) - left, max(ys) - top
                )
        return self._bounds
    
    def add_prompt_point(self, posx, posy, label):
        point = QtCore.QPointF(posx, posy)
        self.prompt_points.append(point)
        self.prompt_labels.append(label)

    def pop_prompt_point(self):
        self.prompt_points.pop(-1)
        self.prompt_labels.pop(-1)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value
        self._bounds = None
