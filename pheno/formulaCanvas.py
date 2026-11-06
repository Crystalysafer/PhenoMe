"""Interactive node-graph canvas for building formula traits.

Drawn with a plain QGraphicsScene/QGraphicsView (no third-party graph
library - that would drag in PySide alongside the project's PyQt5):

* NodeItem      - a rounded box with title, input ports (left) and one
                  output port (right). Draggable, selectable, deletable.
* EdgeItem      - a cubic bezier from an output port to an input port.
* GraphCanvas   - the scene; left-drag from a port draws a live wire,
                  dropping it on a compatible port connects; releasing
                  on empty space cancels. Right-click a node to delete,
                  right-click empty space for the add menu.

The canvas is a thin view over :class:`pheno.formulaGraph.FormulaGraph`;
all semantics (single wire per port, cycle rejection, compilation) live
in the model.
"""

import math

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QPointF, QRectF, Qt, Signal

from pheno.formulaGraph import FormulaGraph
from pheno.formulaTraits import BASE_VARIABLES, FormulaError
from pheno.i18n import tr

NODE_W = 130.0
PORT_R = 7.0
HEADER_H = 26.0


class PortItem(QtWidgets.QGraphicsItem):
    """A connection point on a node: circle + hover highlight."""

    def __init__(self, node_item, index, is_output, label=""):
        super().__init__(node_item)
        self.nodeItem = node_item
        self.index = index
        self.isOutput = is_output
        self.label = label
        self.setAcceptHoverEvents(True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsScenePositionChanges)
        self._hover = False
        self._compatible = False   # highlighted as a valid drop target

    def scenePos(self):
        return self.mapToScene(QPointF(0, 0))

    def boundingRect(self):
        return QRectF(-PORT_R - 5, -PORT_R - 5, 2 * PORT_R + 10, 2 * PORT_R + 10)

    def paint(self, painter, _option, _widget):
        # Filled port = something is connected; hollow = free.
        connected = False
        scene = self.scene()
        if scene is not None and hasattr(scene, "graph") and not self.isOutput:
            connected = scene.graph.incoming(
                self.nodeItem.graphNode.id, self.index) is not None

        if self._compatible:
            color = QtGui.QColor(80, 200, 120)     # drop target: green
            radius = PORT_R + 2.5
        elif self._hover:
            color = QtGui.QColor(240, 160, 30)     # plain hover: amber
            radius = PORT_R + 1
        else:
            color = QtGui.QColor(70, 130, 180)
            radius = PORT_R

        pen = QtGui.QPen(QtGui.QColor(40, 40, 40), 1.5)
        if self._compatible:
            pen = QtGui.QPen(color, 2.5)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(color) if connected or self.isOutput
                         else QtGui.QBrush(QtGui.QColor(245, 245, 245)))
        painter.drawEllipse(QPointF(0, 0), radius, radius)
        if self._compatible:
            # Halo so the snap target is visible from a distance.
            painter.setPen(QtGui.QPen(color, 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(0, 0), radius + 5, radius + 5)

    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()
        if self.scene():
            self.scene().views()[0].setCursor(Qt.CrossCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()
        if self.scene():
            self.scene().views()[0].unsetCursor()
        super().hoverLeaveEvent(event)


class NodeItem(QtWidgets.QGraphicsObject):
    """A node: header (kind-coloured) + body + ports."""

    deleted = Signal(str)          # node id
    renameRequested = Signal(str)  # node id (output nodes only)
    positionMoved = Signal(str, float, float)

    COLORS = {
        "variable": QtGui.QColor(70, 130, 180),
        "op": QtGui.QColor(150, 110, 190),
        "output": QtGui.QColor(70, 160, 110),
    }

    def __init__(self, graph_node):
        super().__init__()
        self.graphNode = graph_node
        self.setPos(graph_node.x, graph_node.y)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(2)

        from pheno.formulaGraph import FormulaGraph
        self._ops = FormulaGraph.OPERATIONS
        self.inPorts = []
        self.outPort = None
        self._buildPorts()

        self._height = self._computeHeight()

    # -- construction ---------------------------------------------------
    def title(self):
        node = self.graphNode
        if node.kind == "variable":
            return node.variable
        if node.kind == "op":
            return self._ops[node.op][0]
        return node.name or tr("Trait")

    def subtitle(self):
        node = self.graphNode
        if node.kind == "output":
            # Derived unit from the graph's dimension analysis; '' when the
            # combination has no meaningful unit.
            scene = self.scene()
            if scene is not None and hasattr(scene, "graph"):
                dimension = scene.graph._dimensionFor(
                    scene.graph.incoming(node.id, 0)
                ) if scene.graph.incoming(node.id, 0) else None
                unit = FormulaGraph.unitForScale(dimension, scene.scaleUnit)
                return unit
            return ""
        return ""

    def _inputCount(self):
        if self.graphNode.kind != "op":
            return 0
        return self._ops[self.graphNode.op][1]

    def _buildPorts(self):
        count = self._inputCount()
        labels = ["a", "b"]
        for index in range(count):
            port = PortItem(self, index, False, labels[index])
            self.inPorts.append(port)
        if self.graphNode.kind != "output":
            self.outPort = PortItem(self, 0, True)
        # Output nodes are consumers; they also get a single input port.
        if self.graphNode.kind == "output":
            self.inPorts = [PortItem(self, 0, False, "value")]
        self._layoutPorts()

    def _layoutPorts(self):
        # Inputs along the left edge, evenly spaced below the header.
        row_h = 24.0
        top = HEADER_H + 8.0
        for index, port in enumerate(self.inPorts):
            port.setPos(0.0, top + index * row_h + PORT_R)
        if self.outPort is not None:
            # Output on the right edge, vertically centred on the body.
            self.outPort.setPos(NODE_W, HEADER_H + 8.0 + row_h / 2.0)

    def _computeHeight(self):
        rows = max(len(self.inPorts), 1 if self.outPort else 0) + 1
        return HEADER_H + rows * 24.0 + 10.0

    # -- geometry ---------------------------------------------------------
    def portScenePos(self, port):
        return port.scenePos()

    def inputPos(self, index):
        return self.inPorts[index].scenePos()

    def outputPos(self):
        return self.outPort.scenePos()

    def boundingRect(self):
        return QRectF(0, 0, NODE_W, self._height)

    def paint(self, painter, option, _widget):
        painter.setPen(QtGui.QPen(QtGui.QColor(40, 40, 40), 1))
        body = QRectF(0, 0, NODE_W, self._height)
        if self.isSelected():
            painter.setPen(QtGui.QPen(QtGui.QColor(240, 160, 30), 2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(250, 250, 250)))
        painter.drawRoundedRect(body, 6, 6)

        # Header band
        header = QRectF(0, 0, NODE_W, HEADER_H)
        path = QtGui.QPainterPath()
        path.addRoundedRect(header, 6, 6)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QtGui.QBrush(self.COLORS[self.graphNode.kind]))
        painter.drawPath(path)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        font = painter.font()
        font.setBold(True)
        font.setPointSizeF(8.5)
        painter.setFont(font)
        painter.drawText(header.adjusted(8, 0, -4, 0),
                         Qt.AlignVCenter | Qt.AlignLeft, self.title())

        # Body text
        painter.setPen(QtGui.QPen(QtGui.QColor(90, 90, 90)))
        font.setBold(False)
        font.setPointSizeF(8)
        painter.setFont(font)
        if self.subtitle():
            painter.drawText(QRectF(8, HEADER_H + 4, NODE_W - 16, 18),
                             Qt.AlignLeft | Qt.AlignVCenter, self.subtitle())

        # Ports already painted as children; draw their labels here so they
        # sit inside the body rather than next to the circles.
        painter.setPen(QtGui.QPen(QtGui.QColor(110, 110, 110)))
        for index, port in enumerate(self.inPorts):
            pos = port.pos()
            painter.drawText(
                QRectF(pos.x() + PORT_R + 6, pos.y() - 8, 60, 14),
                Qt.AlignVCenter | Qt.AlignLeft, port.label)
        if self.outPort:
            painter.drawText(
                QRectF(self.outPort.pos().x() - PORT_R - 40,
                       self.outPort.pos().y() - 8, 34, 14),
                Qt.AlignVCenter | Qt.AlignRight, "out")

    # -- interaction ------------------------------------------------------
    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self.positionMoved.emit(self.graphNode.id, self.x(), self.y())
            scene = self.scene()
            if scene and hasattr(scene, "edgesOf"):
                for edge in scene.edgesOf(self.graphNode.id):
                    edge.reposition()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            # Right-click opens a context menu instead of deleting outright.
            menu = QtWidgets.QMenu()
            node = self.graphNode
            if node.kind == "output":
                rename = menu.addAction(tr("Rename..."))
            delete = menu.addAction(tr("Delete node"))
            chosen = menu.exec_(event.screenPos())
            if chosen is None:
                event.accept()
                return
            if chosen is delete:
                self.deleted.emit(node.id)
            elif node.kind == "output" and chosen is rename:
                self.renameRequested.emit(node.id)
            event.accept()
            return
        super().mousePressEvent(event)


class EdgeItem(QtWidgets.QGraphicsPathItem):
    """Cubic bezier wire between an output port and an input port.

    Hover thickens the wire (so thin wires stay clickable) and a click
    detaches it: the wire's source starts a new drag, letting users
    re-route by grabbing instead of deleting and redrawing.
    """

    def __init__(self, from_node, to_node, in_port, canvas):
        super().__init__()
        self.fromNode = from_node
        self.toNode = to_node
        self.inPort = in_port
        self.canvas = canvas
        self.setZValue(1)
        self.setAcceptHoverEvents(True)
        self.setPen(QtGui.QPen(QtGui.QColor(90, 90, 110), 2))
        self._basePen = QtGui.QPen(QtGui.QColor(90, 90, 110), 2)
        self._hoverPen = QtGui.QPen(QtGui.QColor(240, 120, 30), 3)
        self.reposition()

    def hoverEnterEvent(self, event):
        self.setPen(self._hoverPen)
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(self._basePen)
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Detach: remove from the model and start dragging from source.
            self.canvas.disconnectEdge(self.toNode.graphNode.id, self.inPort)
            event.accept()
            return
        super().mousePressEvent(event)

    def reposition(self):
        p1 = self.fromNode.outputPos()
        p2 = self.toNode.inputPos(self.inPort)
        path = QtGui.QPainterPath(p1)
        dx = max(40.0, abs(p2.x() - p1.x()) * 0.5)
        path.cubicTo(p1 + QPointF(dx, 0), p2 - QPointF(dx, 0), p2)
        self.setPath(path)


class GraphCanvas(QtWidgets.QGraphicsScene):
    """Scene holding nodes and wires; wires are drawn by dragging ports."""

    graphChanged = Signal()
    scaleUnitChanged = Signal(str)   # the unit the output dims are shown in

    def __init__(self, graph, parent=None, scale_unit=""):
        super().__init__(parent)
        self.graph = graph
        self.scaleUnit = scale_unit or ""
        self.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
        self.setSceneRect(-200, -200, 1600, 1000)

        self.nodeItems = {}
        self.edgeItems = []
        self._wire = None            # live wire being dragged
        self._wireSource = None      # NodeItem it started from

        self._rebuild()

    def setScaleUnit(self, unit):
        self.scaleUnit = unit or ""
        # Output node subtitles show the derived unit; refresh them all.
        for item in self.nodeItems.values():
            if item.graphNode.kind == "output":
                item.update()

    # -- scene sync -------------------------------------------------------
    def _rebuild(self):
        self.clear()
        self.nodeItems = {}
        self.edgeItems = []
        for node in self.graph.nodes.values():
            self._addNodeItem(node)
        for from_id, _out, to_id, in_port in self.graph.edges:
            self._addEdgeItem(from_id, to_id, in_port)

    def _addNodeItem(self, graph_node):
        item = NodeItem(graph_node)
        item.deleted.connect(self._removeNode)
        item.renameRequested.connect(self._renameNode)
        item.positionMoved.connect(self._moveNode)
        self.addItem(item)
        self.nodeItems[graph_node.id] = item
        return item

    def _addEdgeItem(self, from_id, to_id, in_port):
        from_item = self.nodeItems.get(from_id)
        to_item = self.nodeItems.get(to_id)
        if from_item is None or to_item is None:
            return
        edge = EdgeItem(from_item, to_item, in_port, self)
        self.addItem(edge)
        self.edgeItems.append(edge)
        return edge

    def disconnectEdge(self, to_id, in_port):
        """Remove a wire (used by edge click-to-detach)."""
        self.graph.disconnect(to_id, in_port)
        self._rebuild()
        self.graphChanged.emit()

    def edgesOf(self, node_id):
        return [edge for edge in self.edgeItems
                if edge.fromNode.graphNode.id == node_id
                or edge.toNode.graphNode.id == node_id]

    def _removeNode(self, node_id):
        self.graph.removeNode(node_id)
        self._rebuild()
        self.graphChanged.emit()

    def _renameNode(self, node_id):
        """Prompt for a new trait name on an output node."""
        node = self.graph.nodes.get(node_id)
        if node is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self.views()[0] if self.views() else None,
            tr("Rename trait"), tr("New name:"),
            text=node.name or "Trait")
        if not ok:
            return
        from pheno.formulaTraits import validate_name, FormulaError
        try:
            node.name = validate_name(name)
        except FormulaError as error:
            QtWidgets.QMessageBox.warning(
                self.views()[0] if self.views() else None,
                tr("Rename trait"), str(error))
            return
        self._rebuild()
        self.graphChanged.emit()

    def _moveNode(self, node_id, x, y):
        node = self.graph.nodes.get(node_id)
        if node is not None:
            node.x, node.y = float(x), float(y)

    # -- wire dragging ------------------------------------------------------
    def _portAt(self, scene_pos):
        for item in self.items(scene_pos):
            if isinstance(item, PortItem):
                return item
        return None

    def _snapPort(self, scene_pos, max_distance=40.0):
        """Nearest compatible input port within ``max_distance`` pixels."""
        best, best_dist = None, max_distance
        for item in self.nodeItems.values():
            for port in item.inPorts:
                distance = (port.scenePos() - scene_pos).manhattanLength()
                if distance < best_dist:
                    best, best_dist = port, distance
        return best

    def _refreshDropHighlights(self, target_port):
        """Green-highlight the port the wire would land on."""
        for item in self.nodeItems.values():
            for port in item.inPorts:
                port._compatible = (target_port is not None
                                    and port is target_port)
            for port in ([item.outPort] if item.outPort else []):
                port._compatible = False
            item.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            port = self._portAt(event.scenePos())
            if port is not None and port.isOutput and port.nodeItem.outPort is port:
                self._wireSource = port.nodeItem
                line = QtGui.QPen(QtGui.QColor(240, 160, 30), 2.5, Qt.DashLine)
                self._wire = QtWidgets.QGraphicsPathItem()
                self._wire.setPen(line)
                self._wire.setZValue(3)
                self.addItem(self._wire)
                self._updateWire(event.scenePos())
                event.accept()
                return
        if event.button() == Qt.RightButton:
            pos = event.scenePos()
            if self._portAt(pos) is None and \
                    not any(isinstance(i, NodeItem) for i in self.items(pos)):
                self._openAddMenu(event.screenPos(), pos)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._wire is not None:
            # Snap to the nearest compatible port so a slightly-off release
            # still connects; the highlight shows where it will land.
            target = self._snapPort(event.scenePos())
            self._refreshDropHighlights(target)
            self._updateWire(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._wire is not None:
            self._refreshDropHighlights(None)
            port = self._portAt(event.scenePos()) or self._snapPort(
                event.scenePos())
            connected = False
            if port is not None and not port.isOutput:
                connected = self.graph.connect(
                    self._wireSource.graphNode.id, port.index,
                    port.nodeItem.graphNode.id)
            self.removeItem(self._wire)
            self._wire = None
            self._wireSource = None
            if connected:
                self._rebuild()
                self.graphChanged.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _updateWire(self, to_pos):
        p1 = self._wireSource.outputPos()
        path = QtGui.QPainterPath(p1)
        dx = max(40.0, abs(to_pos.x() - p1.x()) * 0.5)
        path.cubicTo(p1 + QPointF(dx, 0), to_pos - QPointF(dx, 0), to_pos)
        self._wire.setPath(path)

    # -- node creation -------------------------------------------------------
    def _openAddMenu(self, screen_pos, scene_pos):
        menu = QtWidgets.QMenu()
        variables = menu.addMenu(tr("Measurement"))
        for name in BASE_VARIABLES:
            variables.addAction(name)
        ops = menu.addMenu(tr("Operation"))
        for op_key, (label, _arity, _symbol, _pretty) in \
                FormulaGraph.OPERATIONS.items():
            ops.addAction(label)
        out_action = menu.addAction(tr("Output (trait)"))

        chosen = menu.exec_(screen_pos)
        if chosen is None:
            return
        if chosen is out_action:
            node = self.graph.addNode("output", name="Trait", x=scene_pos.x(),
                                      y=scene_pos.y())
        elif chosen.parent() is variables:
            node = self.graph.addNode("variable", variable=chosen.text(),
                                      x=scene_pos.x(), y=scene_pos.y())
        else:
            op_key = next(k for k, v in FormulaGraph.OPERATIONS.items()
                          if v[0] == chosen.text())
            node = self.graph.addNode("op", op=op_key,
                                      x=scene_pos.x(), y=scene_pos.y())
        self._addNodeItem(node)
        self.graphChanged.emit()

    def deleteSelected(self):
        ids = [item.graphNode.id for item in self.selectedItems()
               if isinstance(item, NodeItem)]
        for node_id in ids:
            self.graph.removeNode(node_id)
        if ids:
            self._rebuild()
            self.graphChanged.emit()
