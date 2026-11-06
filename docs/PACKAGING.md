# Packaging PhenoMe for Windows

The Windows build is produced in two steps:

## 1. PyInstaller → `dist/PhenoMe/`

```bat
build_v2.bat
```

`build_v2.bat` runs `python -m PyInstaller --clean PhenoMe.spec`.
`PhenoMe.spec` bundles the application, the Qt resources and the model
directories into `dist/PhenoMe/PhenoMe.exe`. Model weights under `models/`
are bundled when present, so a full staging produces an all-in-one build;
the bundled PyTorch build provides CUDA support (see `README.md` for the
model download table).

Note: `build_v2.bat` first runs `python test_build.py`, a dependency smoke
test that fails fast on a broken environment; a failure only warns — the
build continues so optional-component gaps do not block packaging.

The packaged app runs windowed (no console). Crash reports are written to
`%LOCALAPPDATA%/PhenoMe/logs/` by the faulthandler installed in `main.py`.

## 2. Installer → `dist/PhenoMeInstaller.exe`

`PhenoMe.iss` (Inno Setup 6) wraps `dist/PhenoMe/` into the installer set
that is deposited on Zenodo. The payload exceeds Windows' ~4.2 GB
single-executable limit, so `DiskSpanning` is enabled: compilation yields
`PhenoMeInstaller.exe` plus `PhenoMeInstaller-*.bin` slices (2.1 GB
each). All files must sit in the same folder; running the .exe installs
the app.

```bat
:: generate the wizard icon once (needs PIL in the build env)
python -c "from PIL import Image; Image.open('pheno/icons/icon.png').save('build/icon.ico', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])"

"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" PhenoMe.iss
```

The installer uses a stable `AppId`, so later releases upgrade the same
installation. It installs to Program Files, creates Start-menu and
(optional) desktop shortcuts, and registers a standard uninstall entry.
The installer is all-in-one (includes `models/`); a standalone
`models.zip` of the same weights is attached to the GitHub release for
source-tree users.

## GPU compatibility

The build inherits CUDA-architecture coverage from the bundled PyTorch
(`torch.cuda.get_arch_list()`): a build on `torch 2.6.0+cu124` covers up to
sm_90 and cannot execute kernels on Blackwell cards (RTX 50 series, sm_120).
Since v2.0.0 the application detects this at startup and falls back to CPU
inference instead of crashing (`pheno/cuda.py`, `pheno/SAM.py`); for native
GPU speed on Blackwell, rebuild against `torch >= 2.7` with CUDA 12.8
(see `requirements.txt`). `tools/probe_blackwell.py` reproduces and verifies
both behaviours on a physical card.
