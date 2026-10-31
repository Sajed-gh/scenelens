"""Analyze one image with YOLO, MobileSAM and metric depth."""

import json
from pathlib import Path
from time import perf_counter

import cv2
import torch

from scenelens import config
from scenelens.models.detector import Detector
from scenelens.models.depth import DepthEstimator
from scenelens.models.segmenter import Segmenter
from scenelens.analysis.objects import SceneObject
from scenelens.analysis.distance import update_object_distances
from scenelens.rendering.renderer import draw_scene_objects, extract_object_crop, compose_sidebar, render_depth_preview


def resize_image(image):
    height, width = image.shape[:2]
    if config.IMAGE_MAX_WIDTH is None or width <= config.IMAGE_MAX_WIDTH:
        return image
    scale = config.IMAGE_MAX_WIDTH / width
    return cv2.resize(image, (config.IMAGE_MAX_WIDTH, max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA)


def save_image(path, image, source_path=None):
    path = Path(path)
    if source_path is not None and path.resolve() == Path(source_path).resolve():
        raise ValueError('An output image must not overwrite the source image.')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f'Cannot save image: {path}')


def run():
    start = perf_counter()
    if config.TORCH_THREADS is not None:
        torch.set_num_threads(config.TORCH_THREADS)
    source = Path(config.IMAGE_PATH)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot read image: {source}. Use --image path/to/photo.jpg.')
    original_height, original_width = image.shape[:2]
    image = resize_image(image)
    detector = Detector(config.DETECTOR_WEIGHTS, config.DETECTOR_CONFIDENCE,
                        config.DETECTOR_IMAGE_SIZE, config.DETECTOR_DEVICE, config.DETECTOR_FP16)
    segmenter = Segmenter(config.SAM_WEIGHTS, config.SAM_DEVICE,
                          config.SAM_BATCH_PROMPTS) if config.SAM_ENABLED else None
    depth_estimator = DepthEstimator(config.DEPTH_MODEL, config.DEPTH_CACHE_DIR,
                                     config.DEPTH_DEVICE, config.DEPTH_INPUT_SIZE) if config.DEPTH_ENABLED else None
    model_load_ms = (perf_counter() - start) * 1000
    timings = {}

    def measure(name, operation):
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        begin = perf_counter()
        value = operation()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        timings[name + '_ms'] = (perf_counter() - begin) * 1000
        return value

    detections = measure('detection', lambda: detector.detect(image))
    # Assign image-local object numbers.
    objects = [SceneObject(index, d.label, d.confidence, d.bbox)
               for index, d in enumerate(detections, 1)]
    depth_map = measure('depth', lambda: depth_estimator.estimate(image)) if depth_estimator else None
    if segmenter:
        measure('segmentation', lambda: segmenter.segment(
            image, objects, config.SAM_MAX_OBJECTS, config.DETECTOR_CONFIDENCE))
    update_object_distances(objects, depth_map, config.OBJECT_DEPTH_MIN_PIXELS)
    for obj in objects:
        obj.crop = extract_object_crop(image, obj.mask, config.OBJECT_CROP_SIZE)
    if config.OUTPUT_IMAGE:
        output = Path(config.OUTPUT_IMAGE)
    else:
        try:
            relative = source.resolve().relative_to(config.DATASET_IMAGES_DIR.resolve())
        except ValueError:
            relative = Path(source.name)
        output = config.PROJECT_ROOT / 'data' / 'outputs' / relative.parent / f'{source.stem}_scenelens.png'
    if output.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
        raise ValueError('Output must be a PNG, JPEG or WebP image.')
    if output.resolve() == source.resolve():
        raise ValueError('Output must differ from the source image.')
    crops_dir = output.parent / (output.stem + '_objects')
    records = []
    for obj in objects:
        crop_path = None
        if obj.crop is not None:
            crop_path = crops_dir / f'object_{obj.object_id}.png'
            save_image(crop_path, obj.crop, source)
        records.append(dict(object_id=obj.object_id, label=obj.label, confidence=obj.confidence,
                            bbox=list(obj.bbox), distance_m=obj.distance,
                            crop=str(crop_path) if crop_path else None))
    def render():
        draw_scene_objects(image, objects)
        return compose_sidebar(image, objects)
    display = measure('rendering', render)
    save_image(output, display, source)
    depth_preview = None
    if config.SAVE_DEPTH_PREVIEW and depth_map is not None:
        depth_preview = output.with_name(output.stem + '_depth.png')
        save_image(depth_preview, render_depth_preview(depth_map, config.DEPTH_PREVIEW_MAX_METERS), source)
    report = dict(source=str(source), output=str(output), model=str(config.DETECTOR_WEIGHTS),
                  device=detector.device, original_size=[original_width, original_height],
                  analyzed_size=[image.shape[1], image.shape[0]],
                  coordinate_system='bounding boxes refer to analyzed_size, before the sidebar',
                  model_load_and_input_ms=model_load_ms, timings=timings,
                  total_ms=(perf_counter() - start) * 1000, objects=records,
                  depth_preview=str(depth_preview) if depth_preview else None)
    report_path = output.with_suffix('.json')
    if report_path.resolve() == source.resolve():
        raise ValueError('The JSON report must not overwrite the source image.')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {output} ({len(objects)} objects, {report["total_ms"] / 1000:.2f} seconds including model loading).')
    if not config.HEADLESS:
        try:
            cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(config.WINDOW_NAME, display)
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key in (ord(config.QUIT_KEY), 27) or cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()
    return report

