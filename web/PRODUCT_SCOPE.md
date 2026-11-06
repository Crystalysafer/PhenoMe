# PhenoMe Web Product Scope

## Interface

The web app is one responsive single-image workflow. It combines local file,
phone camera, and gallery input with positive-click segmentation, semantic
detection, color calibration, ruler detection, scale management, phenomics, and
annotation/table export. Batch and region detection are intentionally excluded.

The web service is ephemeral by design: no accounts, database, browser history,
or cross-device storage. Uploads and analysis are held in process memory only,
expire after inactivity, and leave as downloads.

## Deliverables

Decided: the phone download is the **shape/trait table only** (one CSV). The
annotated image and per-object mask PNGs are not packaged, because on a 16 MPx
photo that made the download ~12 MB. Segmentation is still exportable on its own
through the `.pheno.json` sidecar (~1.1 KB of RLE per object), which is also the
desktop interchange format.

## Product Decisions Still Needed

1. Whether captured originals should be retained, and for how long.
2. Whether one phone may keep multiple named samples in a session.
3. Required photo metadata: sample ID, plant/organ type, operator, location,
   treatment, date, and notes.
4. Whether offline capture and later upload are required.
5. Whether login is needed inside the laboratory Wi-Fi.
6. Whether the phone ever needs the annotated image back (currently no; the CSV
   plus the sidecar cover measurement and re-import).
7. Whether mobile analysis should run automatically after upload or only after
   an explicit command.

## Mobile Capture Constraints

The current `capture="environment"` file input opens the rear camera on common
mobile browsers and works over local HTTP. Advanced live-camera controls based
on `getUserMedia`, such as overlays, continuous preview, and camera switching,
normally require HTTPS even on a LAN.

Photo quality should later be guarded with configurable checks for minimum
resolution, blur, exposure, framing, scale reference visibility, and duplicate
captures before expensive model inference starts.

## Backend Boundary

The web backend should expose model-independent operations:

- create an image session — done
- prepare image embeddings — done, cached per session/image
- segment by positive clicks — done
- calibrate color — done when OpenCV MCC support is available; idempotent
- detect ruler and set scale — done when `rulernet.onnx` is available
- semantic detection — local Grounding DINO Tiny (CPU) + SAM box refinement,
  including the desktop prompt/image cross-check and outlier filtering
- calculate selected object traits — done
- export one result — trait-table CSV plus the `.pheno.json` sidecar; no ZIP
- create, monitor, cancel, and download a batch task — excluded from the web app

Long-running work needs progress, cancellation, per-session isolation, GPU
serialization, bounded memory, and cleanup after abandoned browser sessions.

Currently satisfied: per-session isolation (embeddings keyed by session and
image), GPU serialization (one worker thread owns every CUDA call and Grounding
DINO stays on the CPU), bounded memory (20 images, 6 embeddings, 3 decoded
frames, 2 corrected frames), and cleanup on image deletion, session eviction, or
30 minutes of inactivity. Still missing: progress reporting and cancellation for
long operations (a semantic call blocks for 13-17 s), and any authentication.
