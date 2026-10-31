# SceneLens

Analyze a still image with YOLO26s-seg detection and instance masks, and Depth
Anything V2 Metric Indoor Small. Results include an annotated image with a
all-object mask gallery, transparent object cutouts, a depth visualization, and a
JSON report. See the [architecture](docs/architecture.md) and
[project plan](docs/project_plan.md). The [V2 plan](docs/v2_project_plan.md) breaks
agent-ready perception into small, verifiable steps.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

YOLO downloads `yolo26s-seg.pt` into `weights/` on first use. The first depth
run downloads [the metric indoor Small checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)
into `.cache/huggingface/`. Offline use requires these assets already present.
The runtime loads two networks: YOLO instance segmentation and indoor metric depth.
For a CUDA 12.4 environment, install matching PyTorch packages:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

## Analyze an image

```bash
python -m scenelens --image data/images/indoor67/livingroom/room502.jpg
```

All models prefer `cuda:0`; no device argument is needed. If CUDA is
unavailable, they automatically fall back to CPU. Each model prints its selected
device at startup. An explicit
`--device` override remains available for diagnostics. The result window stays open until `q`, Escape, or window close.
For saving results without a graphical display:

```bash
python -m scenelens --image data/images/indoor67/livingroom/room502.jpg --headless \
  --output data/outputs/result.png
```

Dataset outputs preserve the category: `data/outputs/<category>/<input-stem>_scenelens.png`. Other explicit input paths use `data/outputs/<input-stem>_scenelens.png`:

- Annotated image with boxes, labels, confidence, estimated depth, and sidebar.
- JSON report alongside the image with image-local object numbers, boxes,
  confidence, distances, crop paths, dimensions, device, and per-stage timings.
- Transparent PNG cutouts in `<output-stem>_objects/`.
- A colorized metric depth image at `<output-stem>_depth.png`.

Bounding boxes use the analyzed image dimensions before the sidebar is appended;
original and analyzed sizes are included in the report. Input images are preserved.
Generated files with the same output names are replaced on subsequent runs.

## Configuration

All defaults live in `scenelens/config.py`; `python -m scenelens --help` lists overrides.

| Option | Purpose |
| --- | --- |
| `--image` | Input image; defaults to `data/images/indoor67/livingroom/room502.jpg` |
| `--output` | Annotated PNG, JPEG, or WebP destination |
| `--weights` | Segmentation weights; defaults to YOLO26s-seg |
| `--detector-size` | YOLO inference size; default 960 |
| `--width` | Maximum analyzed width, preserving aspect ratio |
| `--depth-size` | Depth model input size, multiple of 14; default 518 |
| `--device` | Optional override; defaults to `cuda:0` for all models |
| `--threads` | Optional CPU thread limit |
| `--fp16` | Optional CUDA YOLO FP16 |
| `--headless` | Save without opening a window |

`DETECTOR_CONFIDENCE` defaults to 50%. Every returned object receives its own YOLO
mask at the analyzed image resolution. There is no separate segmentation budget.
Set `DEPTH_ENABLED`, `SAVE_DEPTH_PREVIEW`, or `SIDEBAR_ENABLED` to control outputs.
The object gallery shows every detection above the confidence threshold in object-ID
order, including unknown-depth objects. It expands into additional rows and columns
rather than selecting only the nearest objects. The saved result may be taller or
wider than the input; the main image remains at the top left at its analyzed size.
`SIDEBAR_WIDTH` sets card width and `SIDEBAR_MIN_CARD_HEIGHT` controls minimum card
height. The main image overlays every returned mask as well.

Distance is the median of finite positive depth values inside the segmentation
mask, requiring at least `OBJECT_DEPTH_MIN_PIXELS` samples. It estimates surface
depth, not Euclidean distance to an object's center. The indoor checkpoint and its
metric estimates still require sensor validation. Timings are milliseconds per
image, with input/model loading reported separately from inference stages.

## Depth evaluation and tests

`scenelens/evaluation/depth.py` evaluates registered TUM RGB-D image pairs without scale
alignment. Supply an association file with four columns: RGB timestamp, RGB path,
depth timestamp, depth path. Sensor PNG values are divided by 5000. Example:

```bash
python -m scenelens.evaluation.depth --tum-root data/tum/sequence \
  --associations data/tum/sequence/associations.txt \
  --output data/outputs/depth-evaluation.json
python -m unittest discover -s tests -v
```

The evaluator reports MAE/RMSE in meters, AbsRel, delta-1, and prediction coverage
in JSON and per-pair CSV. Real sensor-depth accuracy remains unmeasured; tests use
known synthetic arrays and simulated inference for pipeline checks.

## Project layout

```text
scenelens/
├── scenelens/               # Application package
│   ├── __main__.py          # python -m scenelens
│   ├── cli.py               # Command-line options
│   ├── config.py            # Paths and settings
│   ├── pipeline.py          # Image analysis and exports
│   ├── models/              # Detection, segmentation, metric depth
│   ├── analysis/            # Object data and distance calculation
│   ├── rendering/           # Overlays, crops, sidebar
│   └── evaluation/          # Sensor-depth evaluation
├── weights/                 # Downloaded model checkpoints (ignored)
├── data/
│   ├── images/indoor67/     # Indoor images grouped by category
│   ├── splits/indoor67/     # Development/regression lists and selection manifest
│   └── outputs/            # Generated images, crops, reports
├── tests/                   # Components and image workflow checks
├── scripts/                 # Dataset selection and maintenance
├── docs/                    # Architecture and project plan
├── requirements.txt
└── README.md
```

The application package owns executable code; `weights/` contains model files.
Run `python -m scenelens` from the project root; default asset paths resolve relative to the repository rather than
the current working directory.

## Compare detection quality

```bash
python -m scenelens.evaluation.detection
```

This compares YOLO26s-seg at 640/960 and YOLO26m-seg at 960 on the configured indoor image. It does not automatically run the entire dataset.
Use `--images data/images/indoor67/livingroom/room502.jpg data/images/indoor67/office/o5.jpg` to select inputs. It saves detection overlays,
side-by-side images, per-detection JSON, and a Markdown latency/memory report to
`data/outputs/detection-comparison/`. Each input shape is warmed up before three
measured detection runs; decoding, model loading, and rendering are excluded.
Confidence is held at the configured threshold. A failed model is recorded without
discarding successful results, and an incomplete comparison exits with code 1.
More detections do not by themselves establish higher precision/recall or mAP.

The default is YOLO26s-seg/960 with confidence 0.50. Use `--detector-size 640` for
lower inference cost. Quantitative detection quality on the indoor dataset still
requires a labeled evaluation set; counts alone do not measure accuracy.

## Indoor dataset

A selected 200-image MIT Indoor67 subset is stored under `data/images/indoor67/`
(20 images across each of 10 categories). The other 15,420 images were deleted.
Use 170 development images for iteration and 30 fixed regression images for repeatable checks.
Lists and SHA256 selection records are in `data/splits/indoor67/`. Run the
configured living-room example with `python -m scenelens`, or choose a category image
using `--image`. See [dataset organization](docs/dataset.md).

## Segmentation migration status

YOLO26s-seg replaces the previous separate detector and MobileSAM. Timings now
report a combined `detection_segmentation_ms` stage plus depth and rendering.
Use `--weights weights/yolo26s-seg.pt`; detection-only weights are rejected by the
image pipeline because they cannot provide the required masks. YOLO's
`retina_masks=True` requests original-image mask geometry, which is checked before
fusion. Segmentation and depth need fresh quality and latency comparisons on the
regression subset. The pretrained segmentation checkpoint download was blocked in
the agent environment; local execution will download it automatically when online.
