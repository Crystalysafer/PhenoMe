"""Formula-based custom traits: parse, validate and safely evaluate.

This is the no-code layer on top of the existing Python-algorithm contract.
A "formula trait" is one line of arithmetic over the base measurements that
:mod:`pheno.calcPheno` already computes (Area, Perimeter, MeanR, ...), e.g.::

    FruitShapeIndex = (MajorAxis * MinorAxis) / Area

Files with the ``.phenoFormula.json`` suffix hold a list of such traits and can
be loaded through Manage Algorithms exactly like ``.py`` algorithm files. The
module is Qt-free so the web front-end (or scripts) can reuse the evaluation.

Evaluation safety: expressions are compiled with :func:`ast.parse` and walked
against a whitelist of node types and names - never :func:`eval`. Functions
beyond arithmetic are limited to ``abs``, ``min``, ``max``, ``round`` and a few
constants, because that is all a derived measurement should ever need.
"""

import ast
import json
import math
import os
import re

import numpy as np


FORMULA_SUFFIX = ".phenoFormula.json"

# Base measurements available to formulas, in DEFAULT-algorithm header order.
# Keys are the names users type in expressions.
BASE_VARIABLES = (
    "Area",            # px^2 (real units once a scale is set)
    "Perimeter",
    "MajorAxis",
    "MinorAxis",
    "AspectRatio",
    "Solidity",
    "Circularity",
    "ConvexArea",
    "ConvexPerimeter",
    "Convexity",
    "MeanB",
    "MeanG",
    "MeanR",
)

# The header of the DEFAULT algorithm, in the same order as BASE_VARIABLES.
DEFAULT_HEADERS = (
    "Area", "Perimeter", "Major Axis", "Minor Axis", "Aspect Ratio",
    "Solidity", "Circularity", "Convex Area", "Convex Perimeter", "Convexity",
    "MeanB", "MeanG", "MeanR",
)

_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
}
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
)

_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class FormulaError(ValueError):
    """A user-facing problem with a formula: message is shown as-is."""


# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------


def validate_name(name):
    """Trait names must be usable as a CSV header and as an identifier."""
    text = str(name or "").strip()
    if not text:
        raise FormulaError("The trait name is empty.")
    if not _NAME_PATTERN.match(text):
        raise FormulaError(
            "The trait name '%s' may only contain letters, digits and "
            "underscores, and must not start with a digit." % text)
    if text in BASE_VARIABLES:
        raise FormulaError(
            "'%s' is the name of a base measurement and cannot be reused."
            % text)
    return text


def validate_expression(expression, known=BASE_VARIABLES):
    """Compile ``expression`` and translate failures into plain language."""
    text = str(expression or "").strip()
    if not text:
        raise FormulaError("The expression is empty.")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise FormulaError(
            "Cannot understand the expression '%s': %s."
            % (text, _friendly_syntax_error(error)))

    known_set = set(known)
    used = set()

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise FormulaError(
                "The expression '%s' uses something that is not allowed. "
                "Only arithmetic (+ - * / ** %% //), numbers and the base "
                "measurements are supported." % text)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaError(
                    "Only plain function calls like sqrt(...) are allowed.")
            function = node.func.id
            if function not in _FUNCTIONS:
                raise FormulaError(
                    "The function '%s' is not available. Allowed: %s."
                    % (function, ", ".join(sorted(_FUNCTIONS))))
        if isinstance(node, ast.Name):
            name = node.id
            if name in _FUNCTIONS or name in _CONSTANTS:
                continue
            if name not in known_set:
                suggestion = _closest_name(name, known_set)
                hint = (" Did you mean '%s'?" % suggestion) if suggestion else ""
                raise FormulaError(
                    "'%s' is not a known measurement.%s Available: %s."
                    % (name, hint, ", ".join(known)))
            used.add(name)

    return text, used


def _friendly_syntax_error(error):
    text = str(error.msg)
    if text == "unexpected EOF while parsing":
        return "it seems to be incomplete"
    if text == "invalid syntax":
        return "check the operators and parentheses"
    return text


def _closest_name(name, candidates):
    """Nearest known variable by prefix containment, else None."""
    lowered = name.lower()
    for candidate in sorted(candidates):
        if candidate.lower().startswith(lowered) or lowered.startswith(candidate.lower()):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _compile_expression(expression):
    """Pre-compile a validated expression into a code object."""
    return compile(ast.parse(expression, mode="eval"), "<formula>", "eval")


def _eval_code(code, variables):
    return eval(code, {"__builtins__": {}}, {**_FUNCTIONS, **_CONSTANTS, **variables})


class FormulaTrait:
    """One derived measurement, e.g. ``FruitShapeIndex = Length * Width / Area``.

    ``base_values`` fed to :meth:`evaluate` is the list returned by the DEFAULT
    algorithm, so index ``i`` of :data:`DEFAULT_HEADERS` matches index ``i`` of
    the value list.
    """

    def __init__(self, name, expression, unit="", description=""):
        self.name = validate_name(name)
        self.expression, _used = validate_expression(expression)
        self.unit = str(unit or "").strip()
        self.description = str(description or "").strip()
        self._code = _compile_expression(self.expression)

    def evaluate(self, base_values):
        """Compute the trait from DEFAULT-algorithm results.

        ``nan`` propagates naturally through arithmetic, which is what a
        missing measurement should produce; division by zero yields ``nan``
        as well rather than crashing a whole batch run.
        """
        if base_values is None or len(base_values) < len(BASE_VARIABLES):
            return float("nan")
        variables = dict(zip(BASE_VARIABLES, base_values))
        with np.errstate(divide="ignore", invalid="ignore"):
            try:
                result = _eval_code(self._code, variables)
            except ZeroDivisionError:
                return float("nan")
        if isinstance(result, (int, float, np.integer, np.floating)):
            return float(result)
        return float("nan")

    def header(self):
        """The column header for this trait."""
        return self.name if not self.unit else "%s (%s)" % (self.name, self.unit)

    def to_dict(self):
        return {
            "name": self.name,
            "expression": self.expression,
            "unit": self.unit,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data.get("name", ""),
            expression=data.get("expression", ""),
            unit=data.get("unit", ""),
            description=data.get("description", ""),
        )

    def __repr__(self):
        return "FormulaTrait(%s = %s)" % (self.name, self.expression)


# ---------------------------------------------------------------------------
# A formula file becomes a synthetic "algorithm" for PhenomicsCalculator
# ---------------------------------------------------------------------------


def make_algorithm(traits):
    """Build a ``(getHeaders, calcTraits)`` pair from a list of FormulaTrait.

    The returned callable matches the custom-algorithm contract exactly, so a
    formula file can be registered next to Python algorithms without the
    calculator knowing the difference.
    """
    def get_headers():
        return [trait.header() for trait in traits]

    def calc_traits(rgb_img, mask, scale_per_pixel=1.0):
        # Base values come from the DEFAULT algorithm; formulas reuse that
        # work instead of recomputing contours.
        from pheno.calcPheno import PhenomicsCalculator
        base = PhenomicsCalculator.custom_algorithms.get("DEFAULT")
        base_values = base(rgb_img, mask, scale_per_pixel) if base else None
        return [trait.evaluate(base_values) for trait in traits]

    return get_headers, calc_traits


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def save_traits(path, traits):
    """Write traits as ``{"format": ..., "traits": [...]}`` JSON, atomically."""
    document = {
        "format": "phenome-formula-traits",
        "version": 1,
        "traits": [trait.to_dict() for trait in traits],
    }
    temp_path = str(path) + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, str(path))
    return path


def filename_for(traits):
    """A file name for a set of traits, taken from the first trait's name.

    For when the caller did not say where to write -- a name derived from the
    traits is more predictable than anything an invented path could produce.
    Trait names are already constrained to letters, digits and underscores by
    :func:`validate_name`, so no escaping is needed here.
    """
    stem = str(getattr(traits[0], "name", "") or "").strip() if traits else ""
    return (stem or "formula") + FORMULA_SUFFIX


def algorithm_name_for(path):
    """The algorithm name a file registers under, as the dialog computes it."""
    base = os.path.basename(str(path))
    return base[: -len(FORMULA_SUFFIX)] if base.endswith(FORMULA_SUFFIX) else base


def load_traits(path):
    """Read a formula file. Raises FormulaError with a friendly message."""
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except OSError as error:
        raise FormulaError("Cannot read the file: %s" % error)
    except ValueError as error:
        raise FormulaError("This file is not valid JSON: %s" % error)

    if not isinstance(document, dict) or document.get("format") != "phenome-formula-traits":
        raise FormulaError(
            "This is not a PhenoMe formula file (missing format tag).")

    entries = document.get("traits")
    if not isinstance(entries, list) or not entries:
        raise FormulaError("The formula file contains no traits.")

    traits = []
    for index, entry in enumerate(entries, start=1):
        try:
            traits.append(FormulaTrait.from_dict(entry))
        except FormulaError as error:
            raise FormulaError("Trait %d: %s" % (index, error))
    return traits
