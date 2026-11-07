"""Node-graph formula editor: the visual layer over formulaTraits.

A graph is made of three node kinds:

* VariableNode  - a source: one of the 13 base measurements (no inputs).
* OpNode        - an operation: + - * / ** % sqrt abs min max round
                  (1-2 inputs depending on the op).
* OutputNode    - a trait: name/unit plus the value flowing in.

The graph compiles down to plain expressions with intermediate variables,
which are then fed through :mod:`pheno.formulaTraits` validation - so a
node-built trait and a hand-typed one are the same thing underneath, and
either editor can open what the other saved.

Graph state round-trips through the ``graph`` field of the formula file,
so re-editing preserves layout.
"""

import json
import math
import os

from pheno.formulaTraits import (
    BASE_VARIABLES, FORMULA_SUFFIX, FormulaError, FormulaTrait,
    load_traits, make_algorithm, save_traits,
)


# ---------------------------------------------------------------------------
# Node/graph model (pure data, no Qt - testable headless)
# ---------------------------------------------------------------------------


class GraphNode:
    def __init__(self, node_id, kind, op=None, variable=None, name="", unit=""):
        self.id = str(node_id)
        self.kind = kind            # 'variable' | 'op' | 'output'
        self.op = op                # for op nodes
        self.variable = variable    # for variable nodes
        self.name = name            # for output nodes
        self.unit = unit            # for output nodes
        self.x = 0.0
        self.y = 0.0

    def to_dict(self):
        data = {"id": self.id, "kind": self.kind, "x": self.x, "y": self.y}
        if self.op is not None:
            data["op"] = self.op
        if self.variable is not None:
            data["variable"] = self.variable
        if self.name:
            data["name"] = self.name
        if self.unit:
            data["unit"] = self.unit
        return data

    @classmethod
    def from_dict(cls, data):
        node = cls(
            data["id"], data["kind"],
            op=data.get("op"), variable=data.get("variable"),
            name=data.get("name", ""), unit=data.get("unit", ""))
        node.x = float(data.get("x", 0.0))
        node.y = float(data.get("y", 0.0))
        return node


class FormulaGraph:
    """Nodes plus edges ``[(from_id, out_port, to_id, in_port), ...]``.

    ``out_port`` is always 0 (every node has one output); ``in_port`` is
    0 or 1 for op nodes, 0 for outputs.
    """

    # op -> (label, arity, python symbol, pretty)
    OPERATIONS = {
        "add":      ("a + b", 2, "+", "+"),
        "sub":      ("a - b", 2, "-", "-"),
        "mul":      ("a x b", 2, "*", "*"),
        "div":      ("a / b", 2, "/", "/"),
        "pow":      ("a ^ b", 2, "**", "^"),
        "mod":      ("a mod b", 2, "%", "%"),
        "sqrt":     ("sqrt(a)", 1, "sqrt", "sqrt"),
        "abs":      ("abs(a)", 1, "abs", "abs"),
        "neg":      ("-a", 1, "-", "-"),
        "round":    ("round(a)", 1, "round", "round"),
    }

    # ------------------------------------------------------------------
    # Unit (dimension) inference.
    #
    # Length is the only base dimension (L); everything else a formula can
    # reach is either dimensionless or a power of L:
    #
    #   Area, ConvexArea          -> 2     (L^2)
    #   Perimeter, MajorAxis, ... -> 1     (L)
    #   ratios, MeanB/G/R         -> 0
    #
    # Operations combine dimensions arithmetically; the result is only
    # meaningful when it lands on an integer power of L. sqrt of an area
    # gives L (a length); sqrt of a length gives L^0.5, which is no known
    # physical unit, so the trait becomes unit-less - exactly the user's
    # "weird computation" case.
    # ------------------------------------------------------------------

    VARIABLE_DIMENSIONS = {
        "Area": 2, "ConvexArea": 2,
        "Perimeter": 1, "MajorAxis": 1, "MinorAxis": 1,
        "ConvexPerimeter": 1,
        "AspectRatio": 0, "Solidity": 0, "Circularity": 0, "Convexity": 0,
        "MeanB": 0, "MeanG": 0, "MeanR": 0,
    }

    # How each op maps (dim_a, dim_b) -> result dimension; None = unit-less.
    @staticmethod
    def _opDimension(op, da, db):
        # An unwired input (None dimension) makes the result unit-less.
        if da is None:
            return None
        if op in ("abs", "neg", "round"):
            return da
        if op == "sqrt":
            # Half-integer results (e.g. sqrt of a length) are meaningless.
            return da / 2.0 if da % 2 == 0 else None
        if db is None:
            return None
        if op in ("add", "sub", "min", "max", "mod"):
            # Same-dimension operands keep the dimension; mismatched inputs
            # make the result meaningless.
            return da if da == db else None
        if op == "mul":
            return da + db
        if op == "div":
            return da - db
        # pow and anything else have no meaningful unit.
        return None

    def _dimensionFor(self, node):
        """Dimension of a node's value: 0=scalar, 1=length, 2=area, None=?"""
        if node.kind == "variable":
            return self.VARIABLE_DIMENSIONS.get(node.variable, 0)
        if node.kind == "op":
            source_a = self.incoming(node.id, 0)
            da = self._dimensionFor(source_a) if source_a else None
            if node.op in ("sqrt", "abs", "neg", "round"):
                return self._opDimension(node.op, da, None)
            source_b = self.incoming(node.id, 1)
            db = self._dimensionFor(source_b) if source_b else None
            return self._opDimension(node.op, da, db)
        return None

    @staticmethod
    def unitForScale(dimension, scale_unit):
        """Human unit for a dimension under the current scale, '' if none."""
        if dimension is None:
            return ""
        unit = (scale_unit or "").strip()
        if not unit or unit == "pixel":
            # Un-calibrated: show nothing rather than px^2-style noise.
            return ""
        if dimension == 0:
            return ""
        if dimension == 1:
            return unit
        if dimension == 2:
            return "%s^2" % unit
        return ""

    def __init__(self):
        self.nodes = {}
        self.edges = []
        self._next_id = 1

    # -- construction --------------------------------------------------
    def newId(self):
        while str(self._next_id) in self.nodes:
            self._next_id += 1
        identifier = str(self._next_id)
        self._next_id += 1
        return identifier

    def addNode(self, kind, op=None, variable=None, name="", unit="",
                x=0.0, y=0.0):
        node = GraphNode(self.newId(), kind, op=op, variable=variable,
                         name=name, unit=unit)
        node.x, node.y = float(x), float(y)
        self.nodes[node.id] = node
        return node

    def removeNode(self, node_id):
        self.nodes.pop(str(node_id), None)
        self.edges = [edge for edge in self.edges
                      if edge[0] != str(node_id) and edge[2] != str(node_id)]

    def connect(self, from_id, in_port, to_id):
        """Wire ``from_id``'s output into ``to_id``'s ``in_port``.

        A port holds at most one wire: connecting replaces whatever was
        there, which is what users expect from a dataflow graph.
        """
        from_id, to_id = str(from_id), str(to_id)
        if from_id not in self.nodes or to_id not in self.nodes:
            return False
        target = self.nodes[to_id]
        if target.kind == "variable":
            return False
        max_port = 1 if target.kind == "op" and self.OPERATIONS[target.op][1] == 2 else 0
        if in_port < 0 or in_port > max_port:
            return False
        # No cycles: drop the edge if it would create one.
        if self._wouldCycle(from_id, to_id):
            return False
        self.edges = [edge for edge in self.edges
                      if not (edge[2] == to_id and edge[3] == in_port)]
        self.edges.append((from_id, 0, to_id, int(in_port)))
        return True

    def disconnect(self, to_id, in_port):
        self.edges = [edge for edge in self.edges
                      if not (edge[2] == str(to_id) and edge[3] == int(in_port))]

    def incoming(self, node_id, in_port):
        for edge in self.edges:
            if edge[2] == str(node_id) and edge[3] == int(in_port):
                return self.nodes.get(edge[0])
        return None

    def _wouldCycle(self, from_id, to_id):
        seen = set()
        stack = [to_id]
        while stack:
            current = stack.pop()
            if current == from_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            for edge in self.edges:
                if edge[0] == current:
                    stack.append(edge[2])
        return False

    # -- compile to expressions -----------------------------------------
    def compile(self):
        """Build one :class:`FormulaTrait` per output node.

        The expression is produced by recursively inlining each node's
        inputs, so the result is a single flat expression exactly like one
        typed by hand - validated by the same FormulaTrait machinery.
        """
        traits = []
        for node in sorted(self.nodes.values(), key=lambda n: n.name):
            if node.kind == "output":
                traits.append(self._compileOutput(node))
        return traits

    def _expressionFor(self, node):
        """Recursive inline expansion of a node's expression."""
        if node.kind == "variable":
            if node.variable not in BASE_VARIABLES:
                raise FormulaError(
                    "Unknown measurement '%s'." % node.variable)
            return node.variable

        if node.kind == "op":
            label, arity, symbol, _pretty = self.OPERATIONS[node.op]
            a = self._connectedExpression(node, 0)
            if a is None:
                raise FormulaError(
                    "The '%s' node has an empty input." % node.op)
            if arity == 1:
                if node.op == "neg":
                    return "(-%s)" % a
                return "%s(%s)" % (symbol, a)
            b = self._connectedExpression(node, 1)
            if b is None:
                raise FormulaError(
                    "The '%s' node has an empty second input." % node.op)
            return "(%s %s %s)" % (a, symbol, b)

        raise FormulaError(
            "An output node cannot feed another node.")

    def _connectedExpression(self, node, port):
        source = self.incoming(node.id, port)
        if source is None:
            return None
        return self._expressionFor(source)

    def _compileOutput(self, node):
        source = self.incoming(node.id, 0)
        if source is None:
            raise FormulaError(
                "The output '%s' has nothing connected."
                % (node.name or "?"))
        return FormulaTrait(
            node.name or "Trait%s" % node.id,
            self._expressionFor(source),
            unit=node.unit)

    # -- persistence -----------------------------------------------------
    def to_dict(self):
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [
                {"from": f, "out": o, "to": t, "in": i}
                for f, o, t, i in self.edges],
        }

    @classmethod
    def from_dict(cls, data):
        graph = cls()
        for entry in data.get("nodes", []):
            try:
                node = GraphNode.from_dict(entry)
            except (KeyError, TypeError):
                continue
            graph.nodes[node.id] = node
            try:
                graph._next_id = max(graph._next_id, int(node.id) + 1)
            except ValueError:
                pass
        for entry in data.get("edges", []):
            try:
                graph.connect(entry["from"], int(entry["in"]), entry["to"])
            except (KeyError, TypeError, ValueError):
                continue
        return graph

    def makeSample(self):
        """A ready-made FruitShapeIndex graph, the editor's starting point.

        FruitShapeIndex = (MajorAxis * MinorAxis) / Area
        """
        area = self.addNode("variable", variable="Area", x=40, y=40)
        major = self.addNode("variable", variable="MajorAxis", x=40, y=150)
        minor = self.addNode("variable", variable="MinorAxis", x=40, y=260)
        mul1 = self.addNode("op", op="mul", x=300, y=150)
        div = self.addNode("op", op="div", x=530, y=130)
        out = self.addNode("output", name="FruitShapeIndex",
                           unit="mm2/mm2", x=760, y=130)
        self.connect(major.id, 0, mul1.id)
        self.connect(minor.id, 1, mul1.id)
        self.connect(mul1.id, 0, div.id)
        self.connect(area.id, 1, div.id)
        self.connect(div.id, 0, out.id)
        return self
