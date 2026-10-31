"""YOLO boxes, labels, and image-aligned instance masks from one inference."""

from dataclasses import dataclass

import torch
import numpy as np
from ultralytics import YOLO
from scenelens.runtime import resolve_device


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    class_id: int
    label: str
    confidence: float
    mask: np.ndarray | None = None


class Detector:
    def __init__(self, weights, confidence=0.25, image_size=640, device=None, half=False,
                 require_masks=False):
        self.device = resolve_device(device)
        self.confidence = confidence
        self.image_size = image_size
        self.half = half and self.device != "cpu"
        self.require_masks = require_masks
        try:
            self.model = YOLO(str(weights))
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"Cannot load detector weights: {weights}. "
                "The first run requires internet access to download the selected weights, "
                "or set DETECTOR_WEIGHTS to a local weights file."
            ) from error
        if require_masks and self.model.task != "segment":
            raise ValueError("SceneLens requires segmentation weights, such as yolo26s-seg.pt.")
        print(f"YOLO device: {self.device}")

    def detect(self, image) -> list[Detection]:
        result = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
            retina_masks=True,
            **({"half": True} if self.half else {}),
        )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.cpu()
        masks = result.masks.data.to(dtype=torch.bool).cpu().numpy() if result.masks is not None else None
        if self.require_masks and len(boxes) and masks is None:
            raise RuntimeError("YOLO returned boxes without instance masks.")
        if masks is not None and (len(masks) != len(boxes) or masks.shape[1:] != image.shape[:2]):
            raise RuntimeError("YOLO masks must match the input image dimensions and detection order.")
        return [
            Detection(
                bbox=tuple(float(value) for value in bbox),
                class_id=int(class_id),
                label=result.names[int(class_id)],
                confidence=float(confidence),
                mask=masks[index] if masks is not None else None,
            )
            for index, (bbox, class_id, confidence) in enumerate(zip(
                boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
            ))
        ]
