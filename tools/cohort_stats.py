"""Summarize a PhenoMe cohort for the application section of the manuscript.

Reads the trait tables that batch processing writes (one ``*_phenomics.csv`` per
image/accession, or a merged wide table from the batch review window), groups the
objects by accession, and reports the statistics a breeding or variety-testing
audience asks for first:

* per-accession mean, standard deviation and coefficient of variation per trait
  (the CV sets the smallest difference that can be resolved between accessions);
* a one-way ANOVA across accessions per trait, with the effect size;
* object-level outliers under the shared median rule that the review window and
  the detection filters also use (:mod:`pheno.outliers`);
* optional agreement with manual reference measurements: Pearson r plus a
  Bland-Altman summary (bias and 95% limits of agreement), which quantifies
  agreement beyond what a correlation coefficient alone conveys.

The output is a Markdown report plus a tidy CSV, both meant to be pasted into the
manuscript (Section 3.6) without further processing.

    python -m tools.cohort_stats --dir output --out paper/cohort_report.md
    python -m tools.cohort_stats --dir output --summary summary_pheno.csv
    python -m tools.cohort_stats --dir output --manual manual_pods.csv \
        --manual-trait Width --manual-value-col width_mm

Manual-measurement CSV: one row per organ, with the automatic object name in the
``name`` column (default; override with ``--manual-name-col``) and the caliper or
ruler value in the value column. Matching is by name, so the automatic table and
the manual sheet must use the same organ identifiers.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from pheno.outliers import (
    MIN_GROUP_FOR_OUTLIERS,
    area_is_outlier,
    axis_is_outlier,
    outlier_reason,
)

MIN_GROUP_FOR_ANOVA = 2
MIN_GROUP_FOR_CV = 3

CSV_SUFFIX = "_phenomics.csv"
UNIT_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def read_rows(path: Path):
    """Read one trait table; returns ``(headers, rows)`` with unit suffixes gone."""
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            raw_headers = next(reader)
        except StopIteration:
            return [], []
        headers = [UNIT_SUFFIX.sub("", name).strip() for name in raw_headers]
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    return headers, rows


def to_float(text):
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def accession_from(stem: str, split_on: str | None) -> str:
    """``accession_01_phenomics`` -> ``accession_01`` (when splitting) or the whole stem."""
    if split_on and split_on in stem:
        return stem.split(split_on, 1)[0]
    return stem


def collect(folder: Path, summary: str | None, split_on: str | None):
    """Gather every object of every table into one list of records.

    Each record is ``{"accession", "image", "label", traits...}``. The merged
    wide table (``--summary``) carries the image name in its first column and the
    accession is derived from that name.
    """
    tables = []
    explicit = folder / summary if summary else None
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit("summary table not found: %s" % explicit)
        tables.append((accession_from(explicit.stem, split_on) if split_on else explicit.stem,
                       explicit))
    else:
        for path in sorted(folder.glob("*" + CSV_SUFFIX)):
            tables.append((accession_from(path.name[: -len(CSV_SUFFIX)], split_on), path))
    if not tables:
        raise SystemExit("no trait tables found in %s (expected *%s)" % (folder, CSV_SUFFIX))

    records = []
    traits = []
    for accession, path in tables:
        headers, rows = read_rows(path)
        if not headers:
            continue
        name_index = 0
        for row in rows:
            record = {
                "accession": accession,
                "image": path.stem,
                "label": row[name_index] if len(row) > name_index else "",
            }
            for index, header in enumerate(headers):
                if index == name_index or header.lower() in {"name", "image"}:
                    continue
                value = to_float(row[index]) if index < len(row) else None
                record[header] = value
                if header not in traits:
                    traits.append(header)
            records.append(record)
    return records, traits


def numeric_column(records, trait):
    return [record[trait] for record in records if record.get(trait) is not None]


def coefficient_of_variation(values):
    if len(values) < MIN_GROUP_FOR_CV:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return None
    return statistics.stdev(values) / abs(mean) * 100.0


def anova_across_accessions(records, trait):
    """One-way ANOVA over accessions; returns ``(F, p, eta_squared, k, n)``."""
    groups = []
    seen = {}
    for record in records:
        value = record.get(trait)
        if value is None:
            continue
        seen.setdefault(record["accession"], []).append(value)
    groups = [values for values in seen.values() if len(values) >= 2]
    if len(groups) < MIN_GROUP_FOR_ANOVA:
        return None
    total_n = sum(len(group) for group in groups)
    if total_n <= len(groups):
        return None
    f_stat, p_value = stats.f_oneway(*groups)
    grand = statistics.fmean([value for group in groups for value in group])
    ss_between = sum(len(group) * (statistics.fmean(group) - grand) ** 2 for group in groups)
    ss_total = sum((value - grand) ** 2 for group in groups for value in group)
    eta_squared = ss_between / ss_total if ss_total else float("nan")
    return f_stat, p_value, eta_squared, len(groups), total_n


def outliers(records):
    """Flag objects whose area or major axis departs from their group's median.

    The axis criterion only applies when the trait table carries a Major Axis
    column; older exports without one fall back to the area rule alone.
    """
    by_accession = {}
    for record in records:
        by_accession.setdefault(record["accession"], []).append(record)

    flagged = []
    for accession, group in by_accession.items():
        areas = [record["Area"] for record in group if record.get("Area")]
        if len(areas) < MIN_GROUP_FOR_OUTLIERS:
            for record in group:
                record["outlier"] = ""
            continue
        median = statistics.median(areas)
        axis_key = next(
            (key for key in group[0]
             if isinstance(key, str) and "major axis" in key.lower()),
            None,
        )
        axes = ([record[axis_key] for record in group if record.get(axis_key)]
                if axis_key else [])
        median_axis = (
            statistics.median(axes)
            if len(axes) >= MIN_GROUP_FOR_OUTLIERS
            else None
        )
        for record in group:
            area = record.get("Area")
            axis = record.get(axis_key) if axis_key else None
            axis_bad = (median_axis is not None
                        and axis_is_outlier(axis or 0.0, median_axis))
            if area_is_outlier(area, median) or axis_bad:
                record["outlier"] = "yes" if area else "empty mask"
                flagged.append((
                    accession, record.get("label", ""), area, median,
                    outlier_reason(area, median, axis, median_axis)))
            else:
                record["outlier"] = ""
    return flagged


def load_manual(path: Path, name_column: str, value_column: str, accession_column: str):
    """Read the manual sheet into lookup tables.

    Organ names repeat across accessions far more often than not (every accession
    has a "pod 01"), so matching is attempted first on the accession/name pair and
    only then on the bare name. Returns ``(by_pair, by_name, ambiguous)``.
    """
    by_pair, by_name, ambiguous = {}, {}, 0
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        for row in reader:
            cleaned = {(key or "").strip(): value for key, value in row.items()}
            name = (cleaned.get(name_column) or "").strip()
            value = to_float(cleaned.get(value_column))
            if not name or value is None:
                continue
            accession = (cleaned.get(accession_column) or "").strip()
            if accession:
                by_pair[(accession.lower(), name.lower())] = value
            if name.lower() in by_name:
                ambiguous += 1
            else:
                by_name[name.lower()] = value
    if not by_pair and not by_name:
        raise SystemExit(
            "no usable rows in %s; expected a '%s' column and a '%s' column (found: %s)"
            % (path, name_column, value_column, ", ".join(fieldnames))
        )
    return by_pair, by_name, ambiguous


def agreement(records, trait, manual):
    """Pearson r and Bland-Altman summary between automatic and manual values."""
    by_pair, by_name, ambiguous = manual
    auto, ref, unmatched = [], [], 0
    for record in records:
        value = record.get(trait)
        if value is None:
            continue
        label = (record.get("label") or "").strip().lower()
        accession = (record.get("accession") or "").strip().lower()
        reference = by_pair.get((accession, label))
        if reference is None:
            reference = by_name.get(label)
        if reference is None:
            unmatched += 1
            continue
        auto.append(value)
        ref.append(reference)
    if len(auto) < 3:
        return {
            "n": len(auto),
            "unmatched": unmatched,
            "ambiguous_manual_rows": ambiguous,
            "too_few": True,
        }
    auto_array, ref_array = np.asarray(auto), np.asarray(ref)
    r_value = float(np.corrcoef(auto_array, ref_array)[0, 1])
    differences = auto_array - ref_array
    bias = float(differences.mean())
    sd = float(differences.std(ddof=1))
    slope, intercept = np.polyfit(ref_array, auto_array, 1)
    return {
        "n": len(auto),
        "unmatched": unmatched,
        "ambiguous_manual_rows": ambiguous,
        "too_few": False,
        "r": r_value,
        "p": float(stats.pearsonr(auto_array, ref_array)[1]),
        "bias": bias,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "slope": float(slope),
        "intercept": float(intercept),
        "auto_mean": float(auto_array.mean()),
        "manual_mean": float(ref_array.mean()),
    }


def write_csv(path: Path, records, traits):
    headers = ["accession", "image", "label"] + traits + ["outlier"]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for record in records:
            writer.writerow(
                [record.get("accession", ""), record.get("image", ""), record.get("label", "")]
                + ["" if record.get(trait) is None else record[trait] for trait in traits]
                + [record.get("outlier", "")]
            )
    return path


def format_report(folder, records, traits, flagged, stats_rows, comparison):
    accessions = sorted({record["accession"] for record in records})
    lines = []
    lines.append("# Cohort summary")
    lines.append("")
    lines.append("Source folder: `%s`" % folder)
    lines.append("")
    lines.append("- Objects: **%d**" % len(records))
    lines.append("- Accessions: **%d** (%s)" % (len(accessions), ", ".join(accessions)))
    lines.append("- Traits summarised: **%d**" % len(traits))
    lines.append("- Objects failing the median area/axis rule: **%d**" % len(flagged))
    lines.append("")

    lines.append("## Between-accession variation")
    lines.append("")
    lines.append("| Trait | Accessions | n | F | p | eta^2 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for trait, row in stats_rows.items():
        if row is None:
            lines.append("| %s | - | - | - | - | - |" % trait)
            continue
        f_stat, p_value, eta_squared, k, n = row
        p_text = "< 0.001" if p_value < 0.001 else "%.3f" % p_value
        lines.append(
            "| %s | %d | %d | %.3f | %s | %.3f |" % (trait, k, n, f_stat, p_text, eta_squared)
        )
    lines.append("")

    lines.append("## Per-accession summary")
    lines.append("")
    lines.append("| Accession | Trait | n | Mean | SD | CV (%) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for accession in accessions:
        group = [record for record in records if record["accession"] == accession]
        for trait in traits:
            values = [record[trait] for record in group if record.get(trait) is not None]
            if not values:
                continue
            cv = coefficient_of_variation(values)
            lines.append(
                "| %s | %s | %d | %.4g | %s | %s |"
                % (
                    accession,
                    trait,
                    len(values),
                    statistics.fmean(values),
                    ("%.4g" % statistics.stdev(values)) if len(values) > 1 else "-",
                    ("%.2f" % cv) if cv is not None else "-",
                )
            )
    lines.append("")

    lines.append("## Objects flagged by the area/axis rule")
    lines.append("")
    if not flagged:
        lines.append("None. (Accessions with fewer than %d objects are exempt by design.)"
                     % MIN_GROUP_FOR_OUTLIERS)
    else:
        lines.append("| Accession | Object | Area (px) | Group median (px) | Reason |")
        lines.append("| --- | --- | --- | --- | --- |")
        for accession, label, area, median, reason in flagged:
            lines.append(
                "| %s | %s | %s | %.0f | %s |"
                % (accession, label, "-" if area is None else "%.0f" % area,
                   median, reason)
            )
    lines.append("")

    if comparison is not None:
        lines.append("## Agreement with manual measurements")
        lines.append("")
        comp = comparison["result"]
        if comp.get("too_few"):
            lines.append(
                "Only %d organs could be matched to the manual sheet (at least 3 are needed); "
                "%d automatic objects had no manual counterpart."
                % (comp["n"], comp["unmatched"])
            )
        else:
            lines.append(
                "Trait `%s`, n = %d matched organs." % (comparison["trait"], comp["n"])
            )
            lines.append("")
            lines.append("- Pearson r = **%.3f** (p = %.3g)" % (comp["r"], comp["p"]))
            lines.append(
                "- Automatic mean %.3f vs manual mean %.3f (bias %.3f, 95%% limits of agreement "
                "%.3f to %.3f)" % (comp["auto_mean"], comp["manual_mean"], comp["bias"],
                                   comp["loa_low"], comp["loa_high"])
            )
            lines.append(
                "- Regression automatic = %.3f x manual + %.3f" % (comp["slope"], comp["intercept"])
            )
        if comp.get("unmatched"):
            lines.append(
                "- %d automatic objects had no manual counterpart; check that organ names in "
                "the manual sheet match the object labels, and pass --manual-accession-col if "
                "the sheet carries an accession column." % comp["unmatched"]
            )
        if comp.get("ambiguous_manual_rows"):
            lines.append(
                "- %d manual rows shared an organ name with an earlier row; matching fell back "
                "to the accession/name pair where possible." % comp["ambiguous_manual_rows"]
            )
        lines.append("")

    lines.append("## Notes for the manuscript")
    lines.append("")
    lines.append("- CV is computed within an accession, i.e. across organs; it is not a")
    lines.append("  repeat-imaging repeatability unless the organs were imaged more than once.")
    lines.append("- ANOVA assumes independent organs; if several organs share a plant, use a")
    lines.append("  mixed model with plant as a random effect instead.")
    lines.append("- Report the object count that the review window flagged, and say what the")
    lines.append("  flagged objects were (merged organs, partial masks, border objects).")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", required=True, type=Path,
                        help="folder holding the trait tables")
    parser.add_argument("--summary", default=None,
                        help="merged wide table filename inside --dir (from the review window)")
    parser.add_argument("--split-on", default="_",
                        help="split the file stem on this to get the accession id "
                             "(default '_'; pass '' to keep the whole stem)")
    parser.add_argument("--traits", default=None,
                        help="comma-separated trait names to report (default: all found)")
    parser.add_argument("--manual", type=Path, default=None,
                        help="CSV of manual reference measurements")
    parser.add_argument("--manual-trait", default=None,
                        help="automatic trait to compare against the manual values")
    parser.add_argument("--manual-name-col", default="name",
                        help="column in --manual holding the organ name (default 'name')")
    parser.add_argument("--manual-value-col", default="value",
                        help="column in --manual holding the measured value (default 'value')")
    parser.add_argument("--manual-accession-col", default="accession",
                        help="column in --manual holding the accession id (default 'accession'); "
                             "matching then uses accession and organ name together, which is "
                             "needed whenever organ names repeat across accessions")
    parser.add_argument("--out", type=Path, default=None, help="Markdown report path")
    parser.add_argument("--csv", type=Path, default=None, help="tidy CSV path")
    args = parser.parse_args(argv)

    split_on = args.split_on or None
    records, traits = collect(args.dir, args.summary, split_on)
    if args.traits:
        wanted = [name.strip() for name in args.traits.split(",") if name.strip()]
        traits = [trait for trait in traits if trait in wanted]
    if not traits:
        raise SystemExit("no traits to report")

    flagged = outliers(records)
    stats_rows = {trait: anova_across_accessions(records, trait) for trait in traits}

    comparison = None
    if args.manual is not None:
        if not args.manual_trait:
            raise SystemExit("--manual requires --manual-trait")
        manual = load_manual(args.manual, args.manual_name_col, args.manual_value_col,
                             args.manual_accession_col)
        comparison = {
            "trait": args.manual_trait,
            "result": agreement(records, args.manual_trait, manual),
        }

    report = format_report(args.dir, records, traits, flagged, stats_rows, comparison)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print("report written to %s" % args.out)
    else:
        print(report)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv, records, traits)
        print("table written to %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
