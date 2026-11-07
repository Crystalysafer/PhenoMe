# Custom trait algorithms

This directory holds PhenoMe's custom trait algorithms. On the desktop,
`Manage Algorithms → Load Custom Algorithms (.py)` opens this folder by default.

## Algorithm file format

Each `.py` file is a self-contained algorithm that must expose two functions:

```python
def getHeaders():
    """Return the trait names, in the same order as calcTraits' values."""
    return ["Area", "Perimeter", ...]


def calcTraits(rgb_img, mask, scale_per_pixel=1.0):
    """
    Parameters:
    - rgb_img: RGB image (numpy array, HxWx3)
    - mask: binary mask (HxW, >0 is foreground)
    - scale_per_pixel: real-world size of one pixel

    Returns: the trait values, in the same order as getHeaders().
    """
    return [area, perimeter, ...]
```

`scale_per_pixel` is `1.0` when the image is uncalibrated (pixel units); after
ruler detection or Set Scale it is millimetres per pixel. Remember to multiply
length-like traits by it and area-like traits by its square.

## Bundled algorithms

| File | Object | Traits |
| --- | --- | --- |
| `beanSeedPheno.py` | Bean seeds | 10 shape traits + MeanB/G/R + caruncle area and curvature, 15 in total |
| `beanPodPheno.py` | Bean pods | area, curvedness, width, length, sharpness of both tips, edge length, skeleton length + colour, 12 in total |
| `gardeniaPheno.py` | Gardenia fruit | area, minimum-area-rectangle sides + CIELAB L/a/b means (IQR outlier rejection), 6 in total |

## Dependencies

- `beanSeedPheno.py`: cv2, numpy, scikit-learn, scikit-image
- `beanPodPheno.py`: cv2, numpy, scipy, scikit-image, shapely
- `gardeniaPheno.py`: cv2, numpy

All are already listed in `requirements.txt`.

## Writing your own

Copy any existing file and edit it; the minimal template is:

```python
import cv2

def getHeaders():
    return ["Area"]

def calcTraits(rgb_img, mask, scale_per_pixel=1.0):
    area = cv2.countNonZero(mask) * (scale_per_pixel ** 2)
    return [area]
```

## No-code alternative: formula traits (node graph)

If all you need is arithmetic over the base measurements (a shape index, a
colour ratio, ...), you do not have to write Python at all.
`Manage Algorithms → New Formula Traits...` opens the node-graph editor:

- Drag from a node's right-hand port to another node's left-hand port to wire
  them; right-click empty canvas to add a node, right-click a node to rename or
  delete it; click a wire to disconnect it.
- Three node kinds: measurements (blue), operations (purple: `+ - * / ^ %`,
  `sqrt`, `abs`, negate, `round`), and outputs/traits (green).
- **Units are derived automatically** by dimensional analysis — length×length
  gives an area unit, the square root of an area gives a length unit, ratios
  are dimensionless; the unit follows the image's measurement scale (ruler
  detection or Set Scale), nothing to type. Meaningless combinations (e.g. the
  square root of a length) are shown without a unit.
- The right-hand panel shows the compiled expression and a trial value in real
  time; saving produces a `.phenoFormula.json` (the node layout is stored too,
  so reopening restores the graph), which is fully equivalent to a `.py`
  algorithm — it can be imported, loaded, used in batch mode and in the review
  window.

Variables: `Area, Perimeter, MajorAxis, MinorAxis, AspectRatio, Solidity,
Circularity, ConvexArea, ConvexPerimeter, Convexity, MeanB, MeanG, MeanR`.
These base values come from the built-in measurement set and are **always
available** — unchecking DEFAULT in Manage Algorithms only hides its columns;
it never starves formulas of their inputs. Division by zero and missing
values yield `nan` without interrupting a batch.

Formula files are plain JSON and can also be edited by hand. Implementation:
`pheno/formulaTraits.py` (evaluation), `pheno/formulaGraph.py` (graph model and
dimensional analysis), `pheno/formulaCanvas.py` (canvas),
`pheno/formulaEditorDialog.py` (editor).
