"""
SpatialCNNModel for AegisVerity
--------------------------------
Responsibilities:
- Load CNN backbone (EfficientNet-B4 or Xception) via timm/efficientnet-pytorch.
- Preprocess face crops (BGR -> RGB, resize, normalize).
- Perform inference to produce confidence scores.
- Optionally export Grad-CAM heatmaps for explainability.

Compatibility:
- PyTorch 2.0.1 + CUDA 11.8
- torchvision 0.15.2
- timm 0.9.12
- efficientnet-pytorch 0.7.1
- pytorch-grad-cam 1.5.0
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not available. Xception/EfficientNet via timm will be disabled.")

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_PYTORCH_AVAILABLE = True
except ImportError:
    EFFICIENTNET_PYTORCH_AVAILABLE = False
    logger.warning("efficientnet-pytorch not available. EfficientNet-B4 via this package will be disabled.")

# Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    logger.warning("pytorch-grad-cam not available. Heatmap export will be disabled.")

# Torchvision normalization constants (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SpatialCNNConfig:
    """
    Configuration for SpatialCNNModel.
    """
    def __init__(
        self,
        backbone: str = "efficientnet_b4",  # options: "efficientnet_b4", "xception"
        num_classes: int = 2,               # binary: real vs fake
        input_size: Tuple[int, int] = (380, 380),  # EfficientNet-B4 default
        use_cuda: bool = True,
        checkpoint_path: Optional[str] = None,
        grad_cam: bool = True
    ):
        """
        Args:
            backbone: CNN backbone choice.
            num_classes: Output classes (default binary).
            input_size: Model input resolution (WxH).
            use_cuda: Use CUDA if available.
            checkpoint_path: Optional path to model weights.
            grad_cam: Enable Grad-CAM support.
        """
        self.backbone = backbone.lower()
        self.num_classes = num_classes
        self.input_size = input_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.checkpoint_path = checkpoint_path
        self.grad_cam = grad_cam and GRAD_CAM_AVAILABLE


class SpatialCNNModel(nn.Module):
    """
    Spatial CNN for visual artifact detection.
    """

    def __init__(self, config: Optional[SpatialCNNConfig] = None):
        super().__init__()
        self.config = config or SpatialCNNConfig()
        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")
        self.model: Optional[nn.Module] = None
        self._target_layers: Optional[list] = None  # for Grad-CAM

        self._build_model()
        self.to(self.device)
        self.eval()

    # ---------------------------
    # Model construction
    # ---------------------------
    def _build_model(self) -> None:
        """
        Build backbone and classification head.
        """
        try:
            if self.config.backbone == "efficientnet_b4":
                if TIMM_AVAILABLE:
                    self.model = timm.create_model(
                        "efficientnet_b4",
                        pretrained=True,
                        num_classes=self.config.num_classes
                    )
                    # Target layer for Grad-CAM: last conv
                    self._target_layers = [self.model.blocks[-1]]
                    logger.info("Loaded EfficientNet-B4 via timm.")
                elif EFFICIENTNET_PYTORCH_AVAILABLE:
                    base = EfficientNet.from_pretrained("efficientnet-b4")
                    # Replace classifier
                    in_features = base._fc.in_features
                    base._fc = nn.Linear(in_features, self.config.num_classes)
                    self.model = base
                    # Grad-CAM target: last MBConv block
                    self._target_layers = [self.model._blocks[-1]]
                    logger.info("Loaded EfficientNet-B4 via efficientnet-pytorch.")
                else:
                    raise RuntimeError("No EfficientNet-B4 implementation available.")
                # Default input size for B4 is 380x380
                self.config.input_size = (380, 380)

            elif self.config.backbone == "xception":
                if TIMM_AVAILABLE:
                    self.model = timm.create_model(
                        "xception",
                        pretrained=True,
                        num_classes=self.config.num_classes
                    )
                    # Target layer for Grad-CAM: last conv
                    # timm xception has 'conv4' blocks; pick a deep feature layer
                    try:
                        self._target_layers = [self.model.conv4]
                    except Exception:
                        self._target_layers = [self.model]
                    logger.info("Loaded Xception via timm.")
                    # Typical Xception input size ~299x299
                    self.config.input_size = (299, 299)
                else:
                    raise RuntimeError("timm required for Xception backbone.")
            else:
                raise ValueError(f"Unsupported backbone: {self.config.backbone}")

            # Load checkpoint if provided
            if self.config.checkpoint_path and os.path.exists(self.config.checkpoint_path):
                state = torch.load(self.config.checkpoint_path, map_location="cpu")
                self.model.load_state_dict(state, strict=False)
                logger.info(f"Loaded checkpoint: {self.config.checkpoint_path}")

        except Exception as e:
            logger.error(f"Model build failed: {e}")
            raise

    # ---------------------------
    # Preprocessing
    # ---------------------------
    def preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        """
        Preprocess BGR face