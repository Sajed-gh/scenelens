from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "test.jpg"
OUTPUT_IMAGE = None  # Defaults to data/outputs/<image-stem>_scenelens.png.
SAVE_DEPTH_PREVIEW = True
IMAGE_MAX_WIDTH = 1280  # None keeps original size. Smaller images are not enlarged.
WINDOW_NAME = "SceneLens - Image analysis"
QUIT_KEY = "q"
HEADLESS = False
TORCH_THREADS = None

DETECTOR_WEIGHTS = PROJECT_ROOT / "weights" / "yolo26s.pt"
DETECTOR_CONFIDENCE = 0.50
DETECTOR_IMAGE_SIZE = 640
DETECTOR_DEVICE = None  # Auto-select CUDA or CPU; override with "cuda:0" or "cpu".
DETECTOR_FP16 = False

SAM_ENABLED = True
SAM_WEIGHTS = PROJECT_ROOT / "weights" / "mobile_sam.pt"
SAM_DEVICE = None
SAM_MAX_OBJECTS = 6  # Segment the highest-confidence objects; raise for detailed images.
SAM_BATCH_PROMPTS = True  # Decode box prompts together on CUDA.
MASK_ALPHA = 0.35
OBJECT_CROP_SIZE = (160, 160)  # Width, height

DEPTH_ENABLED = True
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
DEPTH_DEVICE = None
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
SIDEBAR_MAX_OBJECTS = 6
SIDEBAR_MIN_CARD_HEIGHT = 140
