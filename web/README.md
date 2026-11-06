# PhenoMe Web

The web application is one responsive single-image workflow. Its toolbar offers
local file input, phone camera capture, and gallery input through a single
`打开图片` control: the input is a plain `type="file" accept="image/*"` with no
`capture` attribute, so the phone's own sheet offers camera, photo library and
files. Two separate buttons for camera and gallery were removed because they
duplicated what the OS picker already does.

Tapping the empty canvas opens the same picker: the placeholder is a
`<label for="workspace-file">`, because the middle of the canvas is what people
try first.

Two of the user's own DUS photographs ship as one-tap samples
(`/static/samples/pods.jpg` and `seeds.jpg`, downscaled to a 2048 px long side,
300 KB and 163 KB). They are fetched and then pushed through exactly the same
upload path as a camera shot, so nothing about the analysis is special-cased.

The toolbar is deliberately tiered rather than one flat row of buttons:

1. **lead** -- `打开图片` (and `清除`), the only thing that matters before an image exists;
2. **samples** -- the two thumbnails;
3. **analyse** -- `语义检测` first (it needs no interaction), then one `点选目标`
   button and `清除提示点`;
4. **calibration & scale** -- `色卡矫正` / `尺度检测` / `设置比例` inside a `<details>`
   that starts folded on phones (it is a once-per-image detour) and open above 900 px.

`点击分割` and `确认目标` were merged into one button: two buttons whose names both
read like "start segmenting" is exactly the ambiguity to avoid. The single button
carries the whole flow and relabels itself:

| state | label | press does |
| --- | --- | --- |
| not picking | 点选目标 | enter picking mode |
| picking, nothing previewed | 退出点选 | leave picking mode |
| picking, a preview exists | **保存目标** (filled) | save the object, stay in picking mode |

Choosing a new image clears the previous analysis **immediately**, before the
upload starts, and releases the old image on the server. On a phone the upload
takes seconds to a minute, and the old outlines used to sit on top of the new
photo for that whole time. The `<img>` source is cleared as well: an image
element keeps displaying the last picture that decoded, so a failed load looked
like a successful one.

Every slow step reports progress in one shared strip under the toolbar
(`#progress`). The upload has real byte progress (that is why it goes through
`XMLHttpRequest` rather than `fetch`); the model load, embedding, click, semantic
detection, calibration, ruler and trait steps have no server-side progress, so
they ease asymptotically towards 92 % over the duration they usually take and
jump to 100 % when the answer arrives. It never claims to be finished early.

The bundled samples once failed with *"The uploaded image is empty"* on the phone
while the canvas still showed a picture. Three defences now cover that:

1. the sample is fetched without a `force-cache` hint, and if the body is
   zero-length it is re-fetched through `XMLHttpRequest` with `responseType =
   "blob"`;
2. the multipart part is built as `FormData.append("file", blob, name)` -- the
   blob itself, never `new File([blob], ...)`, which some browsers serialise as
   zero bytes;
3. if the server still reports an empty or unsupported upload, the picture is
   redrawn into a canvas and sent again as a fresh JPEG (this also converts an
   iPhone HEIC).

The trait table download is fetched into a Blob and handed to an `<a download>`:
some mobile browsers (Quark, older Safari) ignore `Content-Disposition` for types
they can render, and showed the CSV inline instead of saving it. The endpoint also
returns `application/octet-stream` rather than `text/csv` for the same reason.

Result actions (`计算` / `下载形状表` / `导出标注` / `删除全部`) live in the右侧 panel,
not in the toolbar, because they act on the object list rather than on the image.

The interface is bilingual: a header toggle switches between 中文 and English, the
choice is stored in `localStorage` (`phenome.lang`) and defaults to
`navigator.language`. Static markup carries `data-i18n` keys, dynamic strings go
through `t()` in `app.js`, and both tables in `STRINGS` are asserted to have the
same key set by `web/test_app.py`. Trait table headers stay English on purpose:
they are the CSV column names and must match the desktop export.

Those headers carry their unit -- `Area (mm²)`, `Perimeter (mm)`, or `(px²)`/`(px)`
before calibration -- and it is generated from the current scale rather than written
into the column name, so the table and the CSV can never disagree with the scale in
force. Dimensionless columns (Aspect Ratio, Solidity, Circularity, Convexity) get no
unit because giving them one would be wrong, the channel means are marked `(0-255)`,
and columns produced by custom no-code traits are left alone since their dimension is
unknown. `PhenomicsCalculator.getBareHeaders()` still returns undecorated names for
anything that needs a stable identifier.

A user guide opens by itself on a browser's first visit and afterwards sits behind
the `?` button in the header (`phenome.helpSeen` in `localStorage` remembers the
difference). Its content is fetched from `GET /api/help?lang=zh|en` rather than
bundled into `app.js`, because `pheno/help_content.py` is also what the desktop
`Help -> User Guide` dialog renders -- one source, so the two front ends cannot
describe the same buttons differently. Entries are tagged per platform, so the web
guide never mentions brush editing or batch mode and the desktop guide never
mentions the sample images.

Click segmentation is now connected to the desktop SAM checkpoint, so the
service covers: responsive UI, camera/gallery input, validated upload,
per-session images, lazy model loading, per-session image embeddings,
positive-click segmentation, object create/rename/delete, scale,
phenomics calculation, and `.pheno.json` annotation export.

Region detection and batch processing are excluded from the web app. Semantic
detection uses the local Grounding DINO Tiny model to find prompt-matched boxes,
then refines each box with the already prepared SAM embedding.
Color calibration and ruler detection are available when the local OpenCV MCC
module and `rulernet.onnx` model are installed; their points, ticks, and ruler
line are drawn in the SVG overlay.

Ruler detection runs the improved `pheno/RulerNet.py` pipeline: several
hypotheses (whole frame, the ruler's neighbourhood at native resolution,
contrast-boosted proposals, a rotated frame), each scored against the image and
confirmed by an FFT of the intensity profile along the ticks. It answers in about
1.8 s and, measured on 80 real photographs, finds the correct scale on all 50
that contain a ruler and reports nothing on all 30 that do not. When the
periodicity cannot be confirmed the endpoint deliberately returns 422 rather than
a guess -- a wrong scale would silently corrupt every trait in millimetres, so
`设置比例` by hand is the better failure mode.

The service is intentionally ephemeral: there are no accounts, database, or
server-side history. Uploaded images and analysis stay in process memory and
are reclaimed after 30 minutes of inactivity (or on restart).

Two separate downloads, on purpose:

- `下载形状表` / `GET /api/analysis/results` — the trait table as one CSV. This is
  the phone deliverable and stays around 2 KB. It carries no segmentation.
- `导出标注` / `GET /api/analysis/annotation` — the `.pheno.json` sidecar, which is
  where segmentation lives: RLE masks at about 1.1 KB per object, plus polygons,
  prompt points, scale, and the processing flags.

There is deliberately no result ZIP any more: packing the annotated 16 MPx frame
plus one mask PNG per object produced an ~12 MB download (the annotated PNG alone
was 14.7 MB uncompressed), which is the wrong thing to hand a phone on Wi-Fi.

## Start on the LAN

Run from the project root with the Python executable from the `bean` environment:

```powershell
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

On a phone connected to the same Wi-Fi, open:

```text
http://<host-ip>:8000
```

## Test

```powershell
python -m unittest web.test_app -v
```

The suite never touches the GPU: routes run against a deterministic fake engine.
It does exercise the real CPU-side maths rather than mocking it away — the
`web/ruler.py` pixels-per-cm to millimetres-per-pixel conversion (with only the
ONNX session stubbed), the `web/calibration.py` CCM fit and patch-center
interpolation (with only the MCC detector stubbed), and the prompt-geometry
scaling in `web/engine.py` (with only `SamPredictor` stubbed).

For a real GPU run — pass an image path, or drop one into `input/`:

```powershell
python -m web.verify_gpu
```

## Model API

| Route | Purpose |
| --- | --- |
| `GET /api/model` | model state (`idle`/`loading`/`ready`/`error`), device, checkpoint, cached embeddings, GPU memory |
| `POST /api/model/load` | start the lazy load; returns 202 while loading, 200 once ready |
| `POST /api/analysis/embedding` | queue the image embedding for one session/image pair |
| `GET /api/analysis` | objects, scale, embedding state, model state, trait headers |
| `POST /api/analysis/click` | segment from positive points only; `commit: false` previews, `commit: true` creates the object |
| `PATCH /api/analysis/objects/{uid}` | rename (non-empty, unique within the image) |
| `DELETE /api/analysis/objects/{uid}` | delete one object |
| `POST /api/analysis/scale` | set `per_pixel` and `unit` |
| `POST /api/analysis/semantic` | detect a prompt such as `pod`, cross-check it against the image, then refine boxes with SAM and create objects |
| `POST /api/analysis/color` | detect an MCC24 color card, correct the image once, and return overlay centers |
| `GET /api/analysis/corrected-image` | display-sized JPEG of the corrected frame (long side capped at 2048, cached) |
| `POST /api/analysis/ruler` | detect ruler ticks, set millimetres-per-pixel, and return overlay points |
| `GET /api/analysis/results` | download the trait table as one CSV (no masks, no annotated image) |
| `POST /api/analysis/traits` | trait table for all or selected objects |
| `GET /api/analysis/annotation` | `.pheno.json` sidecar, `download=false` for inline JSON |

Embeddings are keyed by `"<session_id>:<image_id>"`, so two browsers never share
predictor state, and an image is only encoded once per session. Deleting an
image or evicting a session releases its embedding.

### Semantic detection behaviour

`POST /api/analysis/semantic` mirrors the desktop in two ways that matter:

- It runs `TextDetector.cross_check()` **before** creating anything. When the
  prompt clearly disagrees with the image the response is `200` with
  `ok: false`, `reason: "semantic_mismatch"` and no new objects; resend with
  `skip_semantic_check: true` to proceed anyway (the desktop's "continue anyway"
  button). The front-end asks with a confirm dialog.
- It then drops obvious size/shape outliers per detected label through the same
  `pheno.canvas` filter the desktop uses, and reports how many in `filtered`.
  Groups smaller than four candidates are never filtered.

Grounding DINO runs on the **CPU** so it cannot compete with SAM for VRAM. The
cost is latency: on a 16 MPx photo one semantic call takes roughly 13-17 s
(cross-check probe plus the prompt pass), versus ~1 s for a click.

## Color calibration

Calibration is idempotent: the CCM is always fitted on the original upload, and a
second request returns the cached result with `recomputed: false`. Fitting a new
matrix on already corrected pixels would push the colors further every press.

Only `MAX_CORRECTED_IMAGES` corrected frames (2) are held in memory at a time.
An evicted analysis keeps `color_correction: true` and its patch centers, and the
corrected pixels are recomputed on demand, so traits never silently fall back to
the raw image.

## Inference engine

`web/engine.py` is a Qt-free reimplementation of `pheno/SAM.py`:

- lazy load of `sam_vit_h_4b8939.pth` on first request,
- one worker thread for every CUDA operation, so GPU work is serialised,
- per-key embedding cache (six by default) restored into the predictor on demand,
- `cuda:0` when at least 3 GB is free, otherwise CPU.

Masks are bit-compatible with the desktop app: same 1024 px long-side downscale,
same `int()` truncation of prompt coordinates, same "keep the connected domain
containing a positive point" filter, and the same channel order handed to
`SamPredictor`.

`web/analysis.py` reuses the desktop code rather than copying it:
`pheno.lib.compute_polygon_from_mask`, `pheno.canvas.next_label_for_prefix`,
the `pheno.canvas` mask-outlier statistics used by semantic detection,
`pheno.calcPheno.PhenomicsCalculator` and `pheno.annotation`. Only
`calcPhenomics` is bypassed, because it takes a `QPixmap`; the registered
algorithms run on the decoded BGR array instead. `web/calibration.py` and
`web/ruler.py` are the same two exceptions in the other direction: calibration is
a Qt-free copy of the desktop's MCC24 fit (the desktop version takes a `QImage`),
while ruler detection calls `pheno.RulerNet` directly.

## Current data policy

- JPEG, PNG, and WebP are accepted.
- Each upload is limited to 25 MB.
- At most 20 images are retained in process memory.
- At most 3 decoded frames (~48 MB each at 16 MPx) stay unpacked.
- At most 2 color-corrected frames are cached; the rest are recomputed on demand.
- Sessions idle for 30 minutes are discarded together with their images.
- Images are not written to the project directory.
- Restarting the service clears all uploaded images and embeddings.

## No authentication

Every route is unauthenticated: anyone who can reach the port can upload an
image, trigger GPU inference, and read back any session whose `session_id` and
`image_id` they hold (both are `uuid4`, so they cannot be enumerated, but they are
not bound to a client either). This is meant for a laboratory LAN — do not expose
the port to the internet as it stands.

## Integration boundary

Model routes accept image bytes and JSON-compatible coordinates, masks, and
settings. Qt types such as `QImage`, `QPixmap`, and `QPoint` remain on the
desktop side of the boundary.
