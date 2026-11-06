"""Verify that the shipped torch build actually covers the installed GPU.

Runs fine over SSH on a headless Linux box — no display, no Qt required for
steps 1-4. An older wheel (e.g. cu124, which tops out at sm_90) reproduces
the crash on RTX 50-series cards; torch>=2.7+cu128 confirms the GPU path
works once the build actually covers the card.

    python tools/probe_blackwell.py \
        --sam-checkpoint models/sam_vit_h_4b8939.pth \
        --image input/photo.jpg

Steps: (1) print build vs device capability; (2) launch one real CUDA kernel
in a *subprocess* to learn whether the failure is a catchable RuntimeError
or a hard abort; (3) check the new device guard refuses the GPU; (4) force a
GPU pick and let a real kernel failure hit SAM.set_image -> CPU fallback.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def step1_report(torch) -> None:
    print("== step 1: build vs device ==")
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"compiled archs: {torch.cuda.get_arch_list()}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            print(f"  dev{i}: {torch.cuda.get_device_name(i)}  capability sm_{cap[0]}{cap[1]}")


def step2_bare_kernel(torch) -> bool:
    """Launch a real kernel in a subprocess; True if the error is catchable."""
    print("\n== step 2: bare kernel launch (subprocess) ==")
    code = (
        "import torch\n"
        "try:\n"
        "    x = torch.randn(64, 64, device='cuda')\n"
        "    y = torch.mm(x, x)\n"
        "    torch.cuda.synchronize()\n"
        "    print('KERNEL_OK')\n"
        "except Exception as e:\n"
        "    print('CATCHABLE:', type(e).__name__, str(e)[:200])\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    out = (proc.stdout + proc.stderr).strip()
    print(f"  exit={proc.returncode}  {out.splitlines()[-1] if out else '(no output)'}")
    if proc.returncode != 0 and "CATCHABLE" not in out:
        print("  -> HARD CRASH: python-level fallback cannot catch this; the")
        print("     device guard (step 3) is the only defence. Report this.")
        return False
    return True


def step3_device_guard() -> None:
    print("\n== step 3: device guard ==")
    from pheno.cuda import cuda_supported, get_optimal_device

    ok, why = cuda_supported()
    print(f"  cuda_supported() -> ({ok}, {why or 'ok'})")
    print(f"  get_optimal_device() -> {get_optimal_device()}")
    if not ok and get_optimal_device() == "cpu":
        print("  PASS: unsupported GPU refused, app will run on CPU")


def step4_real_fallback(checkpoint: str, image_path: str) -> None:
    print("\n== step 4: real kernel failure -> CPU fallback ==")
    import cv2

    import pheno.SAM as sam_mod

    # Force a GPU pick the way the un-patched build would: the guard in step 3
    # is what normally prevents ever getting here.
    sam_mod.get_optimal_device = lambda: "cuda:0"
    sam_mod.check_memory = lambda required_gb=0: True

    sam = sam_mod.SAM()
    img = cv2.imread(image_path)
    assert img is not None, f"cannot read {image_path}"
    sam.set_image(img)
    print(f"  device after set_image: {sam.device}")
    # Two acceptable outcomes: the GPU is covered so the image embedded on
    # cuda, or a real kernel failure tripped the fallback and we are on cpu.
    sam.add_point(img.shape[1] // 2, img.shape[0] // 2, 1)
    mask = sam.predict(oneDomain=False)
    assert mask is not None and mask.sum() > 0
    if sam.device == "cpu":
        print("  PASS: real kernel failure caught, CPU prediction produced a mask")
    else:
        print("  PASS: GPU is covered by this build; set_image succeeded on cuda")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sam-checkpoint", default="models/sam_vit_h_4b8939.pth")
    ap.add_argument("--image", default="input/photo.jpg")
    args = ap.parse_args()

    import torch

    step1_report(torch)
    catchable = step2_bare_kernel(torch)
    step3_device_guard()
    if catchable and os.path.exists(args.sam_checkpoint):
        step4_real_fallback(args.sam_checkpoint, args.image)
    elif not os.path.exists(args.sam_checkpoint):
        print(f"\n(step 4 skipped: {args.sam_checkpoint} not found)")
    print("\nPROBE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
