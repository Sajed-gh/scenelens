"""Estimate object distance from metric depth pixels inside its segmentation mask."""

import numpy as np

from scenelens.analysis.objects import SceneObject


def masked_distance(depth_map, mask, min_pixels: int) -> float | None:
    if min_pixels < 1:
        raise ValueError("Minimum depth pixel count must be positive.")
    if depth_map is None or mask is None:
        return None
    if depth_map.ndim != 2 or mask.shape != depth_map.shape:
        raise ValueError("Depth map and object mask must have matching 2D shapes.")
    values = depth_map[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < min_pixels:
        return None
    return float(np.median(values))


def update_object_distances(objects: list[SceneObject], depth_map, min_pixels: int):
    for obj in objects:
        obj.distance = masked_distance(depth_map, obj.mask, min_pixels)
