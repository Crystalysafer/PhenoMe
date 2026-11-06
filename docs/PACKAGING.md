# Packaging PhenoMe for Windows

The Windows build is produced in two steps:

## 1. PyInstaller → `dist/PhenoMe/`

```bat
build_v2.bat
```

`build_v2.bat` runs `python -m PyInstaller --clean PhenoMe.spec`.
`PhenoMe.spec` bundles the application, the Qt resources and the model
directories into `dist/PhenoMe/PhenoMe.exe`. The bundled PyTorch build
provides CUDA support; the SAM checkpoint and RulerNet weights are *not*
embedded — they ship beside the exe (see `README.md` for the download links).

Note: `build_v2.bat` first runs `python test_build.py`, a dependency smoke
test that fails fast on a broken environment; a failure only warns — the
build continues so optional-component gaps do not block packaging.

The packaged app runs windowed (no console). Crash reports are written to
`%LOCALAPPDATA%/PhenoMe/logs/` by the faulthandler installed in `main.py`.

## 2. Installer → `PhenoMeInstaller.exe`

The `dist/PhenoMe/` folder is wrapped into the single-file installer
`PhenoMeInstaller.exe` that is deposited on Zenodo. This step is currently
performed outside the repository.

## GPU compatibility

The build inherits CUDA-architecture coverage from the bundled PyTorch
(`torch.cuda.get_arch_list()`): a build on `torch 2.6.0+cu124` covers up to
sm_90 and cannot execute kernels on Blackwell cards (RTX 50 series, sm_120).
Since v2.0.0 the application detects this at startup and falls back to CPU
inference instead of crashing (`pheno/cuda.py`, `pheno/SAM.py`); for native
GPU speed on Blackwell, rebuild against `torch >= 2.7` with CUDA 12.8
(see `requirements.txt`). `tools/probe_blackwell.py` reproduces and verifies
both behaviours on a physical card.
