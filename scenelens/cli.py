"""Analyze a still image with object detection, segmentation, and metric depth."""
import argparse
from pathlib import Path
import cv2
from scenelens import config
from scenelens.pipeline import run


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, key, value_type in [('image','IMAGE_PATH',Path), ('output','OUTPUT_IMAGE',Path),
                                  ('weights','DETECTOR_WEIGHTS',Path), ('width','IMAGE_MAX_WIDTH',int),
                                  ('depth-size','DEPTH_INPUT_SIZE',int), ('max-objects','SAM_MAX_OBJECTS',int),
                                  ('threads','TORCH_THREADS',int)]:
        parser.add_argument('--' + name, dest=key, type=value_type)
    parser.add_argument('--device', help='Apply cpu or cuda:0 to all models.')
    parser.add_argument('--headless', action='store_true', default=None)
    parser.add_argument('--fp16', action='store_true', default=None, help='CUDA YOLO FP16 only.')
    args = vars(parser.parse_args())
    device = args.pop('device')
    for name, key in [('headless','HEADLESS'), ('fp16','DETECTOR_FP16')]:
        value = args.pop(name)
        if value is not None:
            setattr(config, key, value)
    for key, value in args.items():
        if value is not None:
            if isinstance(value, int) and value < 1:
                parser.error(f'{key} must be positive')
            setattr(config, key, value)
    if device:
        config.DETECTOR_DEVICE = config.SAM_DEVICE = config.DEPTH_DEVICE = device


def main():
    try:
        parse_options()
        run()
    except (RuntimeError, ValueError, cv2.error) as error:
        raise SystemExit(str(error)) from None
    except KeyboardInterrupt:
        pass
