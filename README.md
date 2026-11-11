# <img src="pheno/icons/icon.png" width="32"> PhenoMe

**English** | [简体中文](README.zh-CN.md)

A user-friendly software for standardized high-throughput phenotyping of
plant organs. One photograph in — calibrated, per-object trait tables out.

PhenoMe segments organs from plain RGB photographs, calibrates colour and
scale against a colour chart and ruler in the frame, and computes a
configurable trait table per object — without training any model. It is
organ-agnostic by design (seeds, pods, leaves, flowers, ...); the bundled
descriptor sets target legume seeds and pods as a case study, and further
species/organs plug in without retraining.

## Features

- <img src="pheno/icons/text_detect.png" width="18"> **Zero-shot organ segmentation** —
  Grounding DINO resolves free-text prompts
  ("pod", "seed", "leaf", "flower") to boxes; SAM / SAM2 turn boxes (or user
  clicks/brush strokes) into masks. No per-species training.
- <img src="pheno/icons/color_checker.png" width="18"> **Calibration you can trust** —
  MCC24 colour-chart correction reports ΔE
  before/after and *refuses* a bad fit; RulerNet reads the ruler and
  self-verifies its own tick pattern, so a wrong scale is rejected instead of
  silently corrupting every measurement.
- <img src="pheno/icons/calc.png" width="18"> **Quantitative traits, traceable** —
  shape, size and colour per object, plus
  organ-specific traits (pod curvedness and tip sharpness, seed caruncle area
  and curvature, ...). Every number traces back to a mask, a prompt and a
  calibration stored in a `.pheno.json` sidecar.
- <img src="pheno/icons/file.png" width="18"> **Offline desktop app** —
  a PyQt5 application with batch processing, mask editing and a review window.
- <img src="pheno/icons/manageAlgorithm.png" width="18"> **Extensible without code** —
  define new traits in a node-graph editor with
  automatic dimensional analysis, or drop a two-function Python file into
  `algorithms/` (see `algorithms/README.md` for the authoring guide).

## Getting the app

The previous v1 installer is archived at
<https://doi.org/10.5281/zenodo.19936151>.

The current v2 release artifacts — the all-in-one Windows installer
(`PhenoMe-2.0.0-setup.zip`) and a standalone weights archive for running
from source (`models_v2.zip`) — are hosted on Hugging Face:
<https://huggingface.co/datasets/Crystalysafer/phenome-artifacts>

> Model weights are not part of this repository — the installer bundles all
> of them; for source runs grab `models_v2.zip` from the Hugging Face
> record above (or fetch them individually per the table below).

A live demo of the web front end runs at <https://webpheno.me>.

## Running from source

Requires Python 3.10. Dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
# plus: models/sam_vit_h_4b8939.pth and models/grounding-dino-tiny/ (see below)

python main.py                      # desktop application
```


Model assets expected by the source tree:

| Path | What | Source |
| --- | --- | --- |
| `models/sam_vit_h_4b8939.pth` | SAM ViT-H checkpoint | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) |
| `models/sam2/sam2.1_hiera_tiny.pt` | SAM2.1 Hiera-Tiny checkpoint | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| `models/sam2/sam2.1_hiera_large.pt` | SAM2.1 Hiera-Large checkpoint | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| `models/grounding-dino-tiny/` | Grounding DINO Tiny (config + weights + tokenizer) | [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) |
| `models/rulernet.onnx` | RulerNet tick-mark detector | [ymp5078/RulerNet](https://huggingface.co/ymp5078/RulerNet) (CC BY-NC 4.0) |

The Windows installer artifacts are distributed via the Hugging Face dataset
above; packaging scripts are maintained privately and are not part of this
source release.

## Layout

| Path | Contents |
| --- | --- |
| `main.py` | desktop entry point |
| `pheno/` | application + shared analysis core (SAM, calibration, traits, UI) |
| `algorithms/` | trait-algorithm plugins + authoring guide |

## Documentation

- `algorithms/README.md` — plugin authoring guide

## Citation

If you use PhenoMe in your research, please cite the software release
archived at <https://doi.org/10.57967/hf/11486> — and the accompanying
paper in *Plant Phenomics* once the revised version is published.

## License

GPL-3.0 — see `LICENSE`. PhenoMe is based on [labelme](https://github.com/wkentaro/labelme) (GPL-3.0); attribution details in `NOTICE`.
