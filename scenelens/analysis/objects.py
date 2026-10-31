"""Objects detected in a single image, with image-local numbering."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SceneObject:
    object_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    crop: np.ndarray | None = None
    distance: float | None = None
