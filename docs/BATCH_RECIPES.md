# Batch recipes: reproducible batch pipelines

A batch recorded by clicking is easy to replay once and hard to defend later: the
step list lives in the dialog and dies with the session. A **recipe** is the same
step list as a file — something you can read, diff, keep next to the data, and
hand to anyone who wants the same numbers.

## The format

`pheno/recipe.py`. A recipe is JSON, deliberately boring:

```json
{
  "format": "phenome-batch-recipe",
  "version": 1,
  "created": "2026-09-14T21:40:11+00:00",
  "note": "three steps, no coordinates",
  "portable": true,
  "steps": [
    {"type": "text", "prompt": "pod"},
    {"type": "color_correction"},
    {"type": "ruler_detection"}
  ]
}
```

Two kinds of step, and the difference is the one thing worth understanding:

| | steps | replays on |
| --- | --- | --- |
| **portable** | `text`, `color_correction`, `ruler_detection` | **any** image size |
| **positional** | `region`, `click` | only the frame size they were recorded on |

A `text` step says *what to look for*, which does not depend on where anything sits.
A `region` step says *look in this rectangle*, which is pixel coordinates and tied to
the frame — `batch.py` already refuses to replay those on other sizes, so a recipe
containing one must declare its `image_size` and is marked `portable: false`.

**Write portable recipes when the batch is not all the same size.** That is what
makes a recipe reusable across a season of photographs rather than one afternoon.

## Using one

**Produce it** — write the JSON yourself, or record a batch interactively and save
the step list from the desktop batch dialog:

```json
{"type": "text", "prompt": "pod"}
```

is one step. Order the steps the way you would have clicked them.

**Replay it** — in the desktop app: press **Batch** and choose a folder, then
**File → Load Batch Recipe**. The recipe's steps replace the recorded ones and
appear in the step panel exactly as if you had recorded them by hand; press Start
Batch.

Steps are still recorded by mouse as before. A recipe is simply another way to fill
the list — one that happens to be a file.

**Validate it** — `read_recipe()` rejects anything unusable with a message that says
what is wrong, so a hand-edited recipe fails loudly instead of measuring the wrong
things:

```
steps[0]: A text step needs a non-empty 'prompt', e.g. {'type': 'text', 'prompt': 'pod'}.
steps[0]: Unknown step type 'nonsense'; expected one of: text, color_correction, ruler_detection, region, click.
region/click steps are pixel coordinates, so the recipe needs 'image_size': [w, h].
```

## Formula traits

The same idea for a derived measurement. A `.phenoFormula.json` file the desktop
already loads (Manage Algorithms → Load Custom Algorithms) turns a quantity the
built-in table lacks — `MajorAxis * MinorAxis / Area` — into a definition rather
than a calculation someone did once.

Expressions are validated by `pheno/formulaTraits.py`: arithmetic, `sqrt/min/max/abs/round`,
`pi` and `e` over the base measurements, `ast`-based with no `eval`. An unknown
variable comes back with the closest correct name (`Area2 -> Did you mean 'Area'?`).

## Verifying

`read_recipe()` and the step validators live in `pheno/recipe.py`; the batch
dialog exercises them through **File → Load Batch Recipe**, so an invalid recipe
is rejected by the same code that would run it.
