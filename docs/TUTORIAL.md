# PhenoMe user guide

## Desktop application

PhenoMe measures the shape and colour of plant organs from a single photograph: segment the objects from the background, calibrate with the ruler or colour chart in the frame, then export one row of traits per object.

### 📂 Basic workflow
- 📂 Open an image with File -> Open, or a whole folder with Open Dir.
- Segment the objects -- see the next section.
- 📏 Calibrate: use ruler detection, or Set scale to type how many millimetres one pixel is. Without calibration the results are in pixels.
- 💾 Save: File -> Save Annotation (Ctrl+S) writes the objects and their masks to a .pheno.json beside the image, ready to reopen later.
- 📤 Export: the toolbar's Export writes the trait table as CSV; Export Masks writes the ticked objects as PNG files, for tools that do not read .pheno.json. For a whole folder, use batch mode.

### 🔤 Segmenting objects
- 🔤 Semantic detection: type a direct description of the object (pod, green pod, seed, leaf, label, ruler, colour chart) -- English and Chinese both work. It is the least work, so try it first.
- If the wording is not recognised, try a more common one (pod, seed, leaf, flower).
- If the description clearly disagrees with the picture you are warned first and nothing is created until you confirm.
- 👆 Click segment: left-click inside the object, right-click on background to exclude it, then confirm to create the object.
- 🖌️ Edit mask: brush the result -- left button adds, right button removes; the wheel changes the brush size and Ctrl+wheel zooms the canvas.
- 📋 Objects can be renamed and deleted in the list. The name becomes the Name column of the trait table.

### 🎨 Calibration and colour
- 📏 Ruler detection finds the tick marks and converts them to a scale. The ticks must be resolvable -- one centimetre needs at least a dozen pixels.
- 🎨 Colour calibration detects a 24-patch MCC24 chart and corrects the whole image; colour traits are then computed on the corrected pixels.
- Calibration reports the colour error (ΔE) before and after. Lower is better, and a fit that is too poor is refused rather than quietly wrecking the picture.

### 📷 Photography tips
- Keep the camera square to the table -- tilt makes the scale inconsistent across the frame.
- Put the ruler next to the samples, in the same part of the frame -- not along the very edge. The calibration then represents the samples' position and reduces perspective error.
- Leave gaps between samples so they do not touch; segmentation is much cleaner.
- The bigger the brightness difference between background and sample the better; avoid strong reflections and shadows.
- Keep the chart and the ruler in the same plane as the samples.

### 📋 Trait table columns
- Name: object label. Area / Perimeter: area and outline length.
- Major Axis / Minor Axis: axes of the fitted ellipse; Aspect Ratio is their quotient.
- Solidity: area over convex-hull area, lower means a more concave outline. Convexity: hull perimeter over outline perimeter.
- Circularity: 1 for a perfect circle. Convex Area / Convex Perimeter: the hull's area and perimeter.
- MeanB / MeanG / MeanR: mean blue, green and red value inside the object (0-255).
- The unit is printed in the header: (px)/(px²) before calibration, (mm)/(mm²) after, following the scale automatically. Ratio columns carry no unit.
- Changing the scale clears the computed values: the old numbers were produced under the old scale, and pairing them with a new unit would look right and be wrong. Press Calculate again.

### ❓ Troubleshooting
- Ruler detection failed: there is no resolvable tick pattern in the frame. A failure is deliberate -- a wrong scale would corrupt every measurement -- so type the scale by hand instead.
- Colour calibration failed: no standard 24-patch chart, or it is too small or too reflective. Home-made paper charts are usually not recognised.
- “No object found there”: tap closer to the middle of the object.

### 💾 About your data
- 💾 On the desktop the annotation is saved as .pheno.json next to the image and can be reopened to continue editing.
