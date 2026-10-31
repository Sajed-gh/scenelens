"""MobileSAM masks prompted by detected bounding boxes in one image."""

from pathlib import Path

import cv2
import numpy as np
import torch

from scenelens.analysis.objects import SceneObject


class Segmenter:
    def __init__(self, weights, device=None, batch_prompts=True):
        try:
            from mobile_sam import SamPredictor, sam_model_registry
        except ImportError as error:
            raise RuntimeError(
                "MobileSAM dependencies are missing. Run: "
                "python -m pip install -r requirements.txt"
            ) from error
        if not Path(weights).is_file():
            raise RuntimeError(
                f"MobileSAM checkpoint not found: {weights}. Download mobile_sam.pt "
                "from https://github.com/ChaoningZhang/MobileSAM/tree/master/weights "
                "and set SAM_WEIGHTS in scenelens/config.py."
            )
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = sam_model_registry["vit_t"](checkpoint=str(weights))
        model.to(device=self.device)
        model.eval()
        self.predictor = SamPredictor(model)
        self.batch_prompts = batch_prompts and torch.device(self.device).type == "cuda"
        print(f"Segmenter device: {self.device}")

    @torch.inference_mode()
    def segment(self, image, objects: list[SceneObject], max_objects: int,
                min_confidence: float):
        for obj in objects:
            obj.mask = None
        candidates = sorted(
            (obj for obj in objects if obj.confidence >= min_confidence),
            key=lambda obj: (-obj.confidence, obj.object_id),
        )
        height, width = image.shape[:2]
        prompts = []
        for obj in candidates if max_objects > 0 else []:
            box = np.asarray(obj.bbox, dtype=np.float32).copy()
            if not np.isfinite(box).all():
                continue
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if box[2] > box[0] and box[3] > box[1]:
                prompts.append((obj, box))
            if len(prompts) >= max_objects:
                break
        if not prompts:
            return
        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.batch_prompts and len(prompts) > 1:
            boxes = torch.as_tensor(np.stack([box for _, box in prompts]),
                                    dtype=torch.float32, device=self.predictor.device)
            boxes = self.predictor.transform.apply_boxes_torch(boxes, (height, width))
            masks, _, _ = self.predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=boxes, multimask_output=False
            )
            predicted = masks[:, 0].cpu().numpy()
        else:
            predicted = [self.predictor.predict(box=box, multimask_output=False)[0][0]
                         for _, box in prompts]
        for (obj, _), mask in zip(prompts, predicted):
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (height, width):
                raise RuntimeError("MobileSAM mask does not match the image dimensions.")
            obj.mask = mask
