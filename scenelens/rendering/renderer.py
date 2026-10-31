"""Image helpers for object visualization."""

import cv2
import numpy as np

from scenelens import config


def object_color(object_id):
    return config.OVERLAY_COLORS[object_id % len(config.OVERLAY_COLORS)]


def sidebar_objects(objects, limit):
    return sorted(
        (obj for obj in objects if obj.confidence >= config.DETECTOR_CONFIDENCE),
        key=lambda obj: (obj.distance is None, obj.distance if obj.distance is not None else float('inf'), obj.object_id),
    )[:max(0, limit)]


def compose_sidebar(image, objects):
    if not config.SIDEBAR_ENABLED:
        return image
    height = image.shape[0]
    sidebar = np.full((height, config.SIDEBAR_WIDTH, 3), (22, 22, 26), np.uint8)
    cv2.putText(sidebar, "NEAREST OBJECTS", (14, 27), cv2.FONT_HERSHEY_SIMPLEX,
                .6, (240, 240, 240), 1, cv2.LINE_AA)
    capacity = min(config.SIDEBAR_MAX_OBJECTS, max(0, (height - 44) // config.SIDEBAR_MIN_CARD_HEIGHT))
    selected = sidebar_objects(objects, capacity)
    if not selected:
        cv2.putText(sidebar, "No visible objects" if capacity else "Increase image height", (14, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (160, 160, 170), 1, cv2.LINE_AA)
    if selected:
        card_height = (height - 44) // len(selected)
        for index, obj in enumerate(selected):
            top = 44 + index * card_height
            cv2.rectangle(sidebar, (8, top), (config.SIDEBAR_WIDTH - 8, top + card_height - 5), (34, 34, 40), -1)
            cv2.rectangle(sidebar, (8, top), (11, top + card_height - 5), object_color(obj.object_id), -1)
            lines = [f"{obj.label.capitalize()} #{obj.object_id}  {obj.confidence:.0%}",
                     "Depth unknown" if obj.distance is None else f"{obj.distance:.1f} m"]
            for line_index, line in enumerate(lines):
                cv2.putText(sidebar, line, (20, top + 24 + line_index * 21), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (235, 235, 240), 1, cv2.LINE_AA)
            if obj.crop is not None:
                available_height = card_height - 90
                scale = min((config.SIDEBAR_WIDTH - 40) / obj.crop.shape[1], available_height / obj.crop.shape[0])
                size = (max(1, round(obj.crop.shape[1] * scale)), max(1, round(obj.crop.shape[0] * scale)))
                crop = cv2.resize(obj.crop, size, interpolation=cv2.INTER_NEAREST)
                left = (config.SIDEBAR_WIDTH - size[0]) // 2
                region = sidebar[top + 83:top + 83 + size[1], left:left + size[0]]
                alpha = crop[:, :, 3:4].astype(np.float32) / 255
                region[:] = (crop[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)
            else:
                cv2.putText(sidebar, "Mask unavailable", (20, top + 105), cv2.FONT_HERSHEY_SIMPLEX,
                            .4, (150, 150, 160), 1, cv2.LINE_AA)
    return np.concatenate((image, sidebar), axis=1)


def draw_scene_objects(image, objects):
    """Draw masks below boxes and compact metadata panels, using stable ID colors."""
    visible = [obj for obj in objects if obj.confidence >= config.DETECTOR_CONFIDENCE]
    height, width = image.shape[:2]
    for obj in visible:
        if obj.mask is not None:
            image[obj.mask] = (
                (1 - config.MASK_ALPHA) * image[obj.mask]
                + config.MASK_ALPHA * np.asarray(object_color(obj.object_id))
            ).astype(image.dtype)

    panels = []
    occupied = []
    padding, line_height = config.OVERLAY_PADDING, config.OVERLAY_LINE_HEIGHT
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, config.OVERLAY_FONT_SCALE
    for obj in visible:
        color = object_color(obj.object_id)
        x1, y1, x2, y2 = (round(value) for value in obj.bbox)
        x1, x2 = np.clip([x1, x2], 0, width - 1).tolist()
        y1, y2 = np.clip([y1, y2], 0, height - 1).tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        distance = "Depth unknown" if obj.distance is None else f"{obj.distance:.1f} m"
        lines = [f"{obj.label.capitalize()} #{obj.object_id}  {obj.confidence:.0%}", distance]
        panel_width = min(width, max(cv2.getTextSize(line, font, scale, 1)[0][0]
                                     for line in lines) + 2 * padding)
        panel_height = min(height, len(lines) * line_height + 2 * padding)
        left = max(0, min(x1, width - panel_width))
        candidates = [y1 - panel_height - 4, y2 + 4, y1]
        candidates += list(range(44, max(45, height - panel_height + 1), panel_height + 4))
        positions = [max(0, min(top, height - panel_height)) for top in candidates]

        def overlap(top):
            right, bottom = left + panel_width, top + panel_height
            return sum(max(0, min(right, r) - max(left, l))
                       * max(0, min(bottom, b) - max(top, t)) for l, t, r, b in occupied)

        top = min(positions, key=overlap)
        occupied.append((left, top, left + panel_width, top + panel_height))
        panels.append((left, top, panel_width, panel_height, lines, color))

    # Draw all panels last so later masks/boxes never obscure metadata.
    for left, top, panel_width, panel_height, lines, color in panels:
        cv2.rectangle(image, (left, top), (left + panel_width - 1, top + panel_height - 1),
                      (24, 24, 24), -1)
        cv2.rectangle(image, (left, top), (left + 3, top + panel_height - 1), color, -1)
        for index, line in enumerate(lines):
            # Trim long metadata to stay inside the image at the configured font size.
            while line and cv2.getTextSize(line, font, scale, 1)[0][0] > panel_width - 2 * padding:
                line = line[:-1]
            cv2.putText(image, line, (left + padding, top + padding + (index + 1) * line_height - 5),
                        font, scale, (245, 245, 245), 1, cv2.LINE_AA)


def render_depth_preview(depth_map, max_depth: float):
    """Fixed metric color scale: bright near pixels, dark far pixels, black invalid pixels."""
    if max_depth <= 0:
        raise ValueError("Depth preview maximum must be positive.")
    valid = np.isfinite(depth_map) & (depth_map > 0)
    normalized = np.zeros(depth_map.shape, dtype=np.uint8)
    normalized[valid] = ((1 - np.clip(depth_map[valid] / max_depth, 0, 1)) * 255).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    preview[~valid] = 0
    text = f"Near: 0 m (bright) | Far: {max_depth:g} m (dark)"
    cv2.putText(preview, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(preview, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def extract_object_crop(image, mask, size: tuple[int, int]) -> np.ndarray | None:
    """Return a centered BGRA cutout on a transparent (width, height) canvas."""
    width, height = size
    if width < 1 or height < 1:
        raise ValueError("Crop dimensions must be positive.")
    if mask is None:
        return None
    if mask.shape != image.shape[:2]:
        raise ValueError("Object mask must match the image dimensions.")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    source = image[y1:y2, x1:x2]
    source_mask = mask[y1:y2, x1:x2].astype(np.uint8)
    scale = min(width / source.shape[1], height / source.shape[0])
    resized_width = min(width, max(1, round(source.shape[1] * scale)))
    resized_height = min(height, max(1, round(source.shape[0] * scale)))
    dimensions = (resized_width, resized_height)
    # Resize colors separately so removed background cannot bleed into the alpha.
    colors = cv2.resize(source, dimensions, interpolation=cv2.INTER_NEAREST)
    alpha = cv2.resize(source_mask, dimensions, interpolation=cv2.INTER_NEAREST) * 255
    colors[alpha == 0] = 0
    crop = np.zeros((height, width, 4), dtype=np.uint8)
    left, top = (width - resized_width) // 2, (height - resized_height) // 2
    crop[top:top + resized_height, left:left + resized_width, :3] = colors
    crop[top:top + resized_height, left:left + resized_width, 3] = alpha
    return crop
