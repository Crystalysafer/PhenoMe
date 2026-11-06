"""Export a curated public-release snapshot of the PhenoMe repository.

The working repository contains operational material (deploy/, model weights,
input/output data, paper drafts) that must not be published. This script copies
a whitelist into ``release_public/`` — the exact tree that should go into the
public repository and the Zenodo source archive, so the two always match.

Usage:
    python tools/export_release.py [dest]     # default: release_public/
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Top-level files copied verbatim.
FILES = [
    "main.py",
    "PhenoMe.spec",
    "PhenoMe.iss",
    "build_v2.bat",
    "test_build.py",
    "LICENSE",
]

# Top-level files copied after stripping private-only material:
#   README.md         — blocks between <!-- PUBLIC-OMIT-START/END --> markers
#   requirements.txt  — lines carrying a "# public:omit" comment (web deps)
FILTERED_FILES = ["README.md", "requirements.txt"]

# Directories copied in full (minus __pycache__ / *.pyc). The web front end
# is private-only: deployment is hard to reproduce and the hosted service at
# webpheno.me is what reviewers should use.
DIRS = ["pheno", "algorithms"]

# Curated subsets.
TOOLS = [
    "__init__.py",
    "export_release.py",
    "eval_color.py",
    "eval_distortion.py",
    "eval_geometry.py",
    "eval_ruler.py",
    "cohort_stats.py",
    "probe_blackwell.py",
]

# ANNOTATION_FORMAT.md documents the private pheno<->web contract and stays
# out of the public tree with the web front end itself.
DOCS = [
    "BATCH_RECIPES.md",
    "PACKAGING.md",
]

# Paths excluded from the full-directory copy: private demo art only.
DIR_EXCLUDES = {"pheno/icons/demo.png"}

GENERATED = {"docs/TUTORIAL.md", ".gitignore"}  # rendered below, not repo files

GITIGNORE = """__pycache__/
*.pyc
*.pyo
dist/
build/
models/
*.egg-info/
.idea/
.vscode/
*.log
"""


_OMIT_BLOCK_RE = re.compile(
    r"^[ \t]*<!--[ \t]*PUBLIC-OMIT-START[ \t]*-->.*?^[ \t]*<!--[ \t]*PUBLIC-OMIT-END[ \t]*-->[ \t]*\n?",
    re.DOTALL | re.MULTILINE,
)


def strip_for_public(text: str) -> str:
    """Remove material that only makes sense in the private repo."""
    text = _OMIT_BLOCK_RE.sub("", text)
    lines = [line for line in text.splitlines() if "# public:omit" not in line]
    return "\n".join(lines) + "\n"


def render_tutorial() -> str:
    """Render the shared in-app help into a standalone markdown guide."""
    sys.path.insert(0, str(ROOT))
    from pheno.help_content import DESKTOP, help_document

    parts = ["# PhenoMe user guide\n"]
    for platform, heading in ((DESKTOP, "Desktop application"),):
        doc = help_document("en", platform)
        parts.append(f"\n## {heading}\n\n{doc['intro']}\n")
        for section in doc["sections"]:
            parts.append(f"\n### {section['heading']}\n")
            parts.extend(f"- {item}\n" for item in section["items"])
    return "".join(parts)


def copy_tree(src: Path, dst: Path) -> int:
    count = 0
    for path in src.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if (path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc"
                or any(rel == ex or rel.startswith(ex + "/") for ex in DIR_EXCLUDES)):
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


def _force_remove(_fn, path, _exc):
    """rmtree callback: git object files are read-only on Windows."""
    os.chmod(path, 0o666)
    os.unlink(path)


def main() -> int:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "release_public"
    dest.mkdir(parents=True, exist_ok=True)
    for entry in dest.iterdir():
        if entry.name == ".git":  # keep the pushed repo's history across re-exports
            continue
        if entry.is_dir():
            shutil.rmtree(entry, onerror=_force_remove)
        else:
            entry.unlink()

    count = 0
    for name in FILES:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, dest / name)
            count += 1
    for name in FILTERED_FILES:
        src = ROOT / name
        if src.exists():
            (dest / name).write_text(
                strip_for_public(src.read_text(encoding="utf-8")), encoding="utf-8"
            )
            count += 1
    for name in DIRS:
        count += copy_tree(ROOT / name, dest / name)
    for name in TOOLS:
        src = ROOT / "tools" / name
        if src.exists():
            (dest / "tools").mkdir(exist_ok=True)
            shutil.copy2(src, dest / "tools" / name)
            count += 1
    for name in DOCS:
        src = ROOT / "docs" / name
        if src.exists():
            (dest / "docs").mkdir(exist_ok=True)
            shutil.copy2(src, dest / "docs" / name)
            count += 1

    (dest / "docs").mkdir(exist_ok=True)
    (dest / "docs" / "TUTORIAL.md").write_text(render_tutorial(), encoding="utf-8")
    (dest / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

    print(f"Exported {count} files + generated docs/TUTORIAL.md -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
