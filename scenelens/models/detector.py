"""YOLO detection with a small, model-independent output representation."""

from dataclasses import dataclass

import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    class_id: int
    label: str
    confidence: float


class Detector:
    def __init__(self, weights, confidence=0.25, image_size=640, device=None, half=False):
        self.device = device if device is not None else (
            "0" if torch.cuda.is_available() else "cpu"
        )
        self.confidence = confidence
        self.image_size = image_size
        self.half = half and self.device != "cpu"
        try:
            self.model = YOLO(str(weights))
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"Cannot load detector weights: {weights}. "
                "The first run requires internet access to download the selected weights, "
                "or set DETECTOR_WEIGHTS to a local weights file."
            ) from error
        print(f"Detector device: {self.device}")

    def detect(self, image) -> list[Detection]:
        result = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
            **({"half": True} if self.half else {}),
        )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.cpu()
        return [
            Detection(
                bbox=tuple(float(value) for value in bbox),
                class_id=int(class_id),
                label=result.names[int(class_id)],
                confidence=float(confidence),
            )
            for bbox, class_id, confidence in zip(
                boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
            )
        ]
