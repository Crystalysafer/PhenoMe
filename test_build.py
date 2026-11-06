"""Pre-build dependency smoke test.

Run by ``build_v2.bat`` before PyInstaller so a broken environment fails in
seconds instead of after a long packaging run. Exits non-zero when a required
import fails; optional components only warn (missing optional pieces simply
disable the corresponding feature at runtime).
"""

from __future__ import annotations

import importlib
import sys

REQUIRED = [
    # GUI and app framework
    "PyQt5", "qtpy",
    # deep learning stack
    "torch", "torchvision", "transformers", "safetensors", "tokenizers",
    "huggingface_hub", "onnxruntime", "segment_anything",
    # scientific stack
    "numpy", "scipy", "sklearn", "skimage", "shapely", "pandas",
    "matplotlib", "PIL", "cv2",
    # misc pinned deps
    "yaml", "requests", "urllib3", "certifi", "charset_normalizer", "idna",
    # application package
    "pheno", "algorithms",
]

OPTIONAL = [
    ("sam2", "SAM2 segmenter backends (sam_vit_h still works without it)"),
    ("fastapi", "web service (desktop app does not need it)"),
    ("uvicorn", "web service (desktop app does not need it)"),
]


def _try_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception as exc:  # noqa: BLE001 - report the actual cause
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    failures = 0

    print("\nRequired modules:")
    for name in REQUIRED:
        if not _try_import(name):
            failures += 1

    print("\nOptional modules:")
    for name, why in OPTIONAL:
        status = "ok" if _try_import(name) else f"missing ({why})"
        print(f"  {name}: {status}")

    try:
        import torch
        arches = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
        cuda = torch.version.cuda or "?"
        print(f"\ntorch {torch.__version__} (cuda {cuda})")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if arches:
            print(f"arch list: {', '.join(arches)}")
    except Exception as exc:  # noqa: BLE001
        print(f"\ntorch probe failed: {exc}")
        failures += 1

    if failures:
        print(f"\n{failures} required import(s) failed — fix the environment "
              "before building.")
        return 1
    print("\nAll required dependencies import cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
