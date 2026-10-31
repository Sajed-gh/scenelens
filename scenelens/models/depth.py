"""Single-image metric depth inference using the indoor Depth Anything V2 Small model."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class DepthEstimator:
    def __init__(self, model_name, cache_dir, device=None, input_size=518):
        if input_size < 14 or input_size % 14:
            raise ValueError("Depth input size must be a positive multiple of 14.")
        self.input_size = input_size
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as error:
            raise RuntimeError(
                "Depth dependencies are missing. Run: python -m pip install -r requirements.txt"
            ) from error
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=str(cache_dir))
            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_name, cache_dir=str(cache_dir)
            )
        except OSError as error:
            raise RuntimeError(
                f"Cannot load depth model: {model_name}. First use requires internet access. "
                "DEPTH_MODEL can also point to a local Transformers model directory."
            ) from error
        if getattr(self.model.config, "depth_estimation_type", None) != "metric":
            raise RuntimeError("DEPTH_MODEL must be a metric depth checkpoint, not a relative model.")
        self.model.to(self.device).eval()
        print(f"Depth device: {self.device}")

    @torch.inference_mode()
    def estimate(self, image) -> np.ndarray:
        """Return an image-aligned float32 depth map in meters."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt",
                                size={"height": self.input_size, "width": self.input_size})
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        prediction = self.model(**inputs).predicted_depth
        prediction = F.interpolate(
            prediction.unsqueeze(1), size=image.shape[:2], mode="bilinear", align_corners=False
        )[0, 0]
        return prediction.float().cpu().numpy()
