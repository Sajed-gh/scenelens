from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVICE = "cuda:0"
DATASET_IMAGES_DIR = PROJECT_ROOT / "data" / "images" / "indoor67"
DATASET_SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "indoor67"
IMAGE_PATH = DATASET_IMAGES_DIR / "livingroom" / "room502.jpg"
OUTPUT_IMAGE = None  # Defaults to data/outputs/<image-stem>_scenelens.png.
SAVE_DEPTH_PREVIEW = True
IMAGE_MAX_WIDTH = 1280  # None keeps original size. Smaller images are not enlarged.
WINDOW_NAME = "SceneLens - Image analysis"
QUIT_KEY = "q"
HEADLESS = False
TORCH_THREADS = None

DETECTOR_WEIGHTS = PROJECT_ROOT / "weights" / "yolo26s-seg.pt"
DETECTOR_CONFIDENCE = 0.5
DETECTOR_IMAGE_SIZE = 960
DETECTOR_DEVICE = DEFAULT_DEVICE
DETECTOR_FP16 = False

MASK_ALPHA = 0.35
OBJECT_CROP_SIZE = (160, 160)  # Width, height

DEPTH_ENABLED = True
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
DEPTH_DEVICE = DEFAULT_DEVICE
DEPTH_INPUT_SIZE = 518  # Positive multiple of 14.
DEPTH_PREVIEW_MAX_METERS = 20.0  # Visualization scale only.
OBJECT_DEPTH_MIN_PIXELS = 20

OVERLAY_FONT_SCALE = 0.5
OVERLAY_LINE_HEIGHT = 22
OVERLAY_PADDING = 7
OVERLAY_COLORS = ((255, 180, 0), (80, 220, 100), (80, 160, 255),
                  (220, 120, 230), (0, 220, 230), (230, 170, 120))  # BGR
SIDEBAR_ENABLED = True
SIDEBAR_WIDTH = 280
SIDEBAR_MIN_CARD_HEIGHT = 220
