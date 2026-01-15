"""
Explainability Utilities for AegisVerity
----------------------------------------
Responsibilities:
- Provide Grad-CAM visualization for CNN-based models.
- Provide attention map visualization for transformer-based models (e.g., TimeSformer).
- Export heatmaps overlayed on input images for forensic reporting.

Compatibility:
- PyTorch 2.0.1 + CUDA 11.8
- pytorch-grad-cam 1.5.0
- loguru 0.7.2
"""

from __future__ import annotations

from typing import Optional, Any, List
import numpy as np
import cv2
import torch
from loguru import logger

try:
    from pytorch_grad_cam import GradCAM, ScoreCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    logger.warning("pytorch-grad-cam not available. Explainability disabled.")


class ExplainabilityConfig:
    """
    Configuration for explainability utilities.
    """
    def __init__(
        self,
        use_cuda: bool = True,
        method: str = "gradcam"  # options: gradcam, scorecam, eigencam
    ):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.method = method.lower()


class ExplainabilityEngine:
    """
    Explainability engine for generating heatmaps.
    """

    def __init__(self, config: Optional[ExplainabilityConfig] = None):
        self.config = config or ExplainabilityConfig()

    def _select_cam_method(self, model: torch.nn.Module, target_layers: List[Any]):
        """
        Select CAM method based on config.
        """
        if not GRAD_CAM_AVAILABLE:
            raise RuntimeError("Grad-CAM library not available.")

        if self.config.method == "gradcam":
            return GradCAM(model=model, target_layers=target_layers, use_cuda=self.config.use_cuda)
        elif self.config.method == "scorecam":
            return ScoreCAM(model=model, target_layers=target_layers, use_cuda=self.config.use_cuda)
        elif self.config.method == "eigencam":
            return EigenCAM(model=model, target_layers=target_layers, use_cuda=self.config.use_cuda)
        else:
            raise ValueError(f"Unsupported CAM method: {self.config.method}")

    def generate_heatmap(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        target_layers: List[Any],
        target_class: Optional[int] = None,
        original_image_bgr: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Generate CAM heatmap overlay.
        Args:
            model: PyTorch model.
            input_tensor: Preprocessed input tensor.
            target_layers: Layers to target for CAM.
            target_class: Optional class index for targeted CAM.
            original_image_bgr: Original image (BGR) for overlay.
        Returns:
            Heatmap overlay (BGR) or None.
        """
        if not GRAD_CAM_AVAILABLE:
            logger.warning("Grad-CAM not available.")
            return None

        try:
            cam = self._select_cam_method(model, target_layers)
            targets = None
            if target_class is not None:
                targets = [ClassifierOutputTarget(target_class)]

            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0]  # (H, W)

            if original_image_bgr is not None:
                img_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
                img_float = img_rgb.astype(np.float32) / 255.0
                visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
                vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                return vis_bgr
            else:
                # Return raw grayscale CAM
                heatmap = (grayscale_cam * 255).astype(np.uint8)
                return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            return None

    def visualize_attention(
        self,
        attn_map: np.ndarray,
        original_image_bgr: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Visualize transformer attention map overlay.
        Args:
            attn_map: Attention map (H x W).
            original_image_bgr: Original image (BGR).
        Returns:
            Overlay visualization (BGR).
        """
        try:
            heatmap = cv2.resize(attn_map, (original_image_bgr.shape[1], original_image_bgr.shape[0]))
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original_image_bgr, 0.6, heatmap_color, 0.4, 0)
            return overlay
        except Exception as e:
            logger.error(f"Attention visualization failed: {e}")
            return None
