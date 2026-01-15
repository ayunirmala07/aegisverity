"""
TemporalModel for AegisVerity
------------------------------
Responsibilities:
- Detect temporal inconsistencies (flicker, warping) across frame sequences.
- Provide two backbones:
  1) 3D-CNN (lightweight custom) for spatio-temporal features.
  2) TimeSformer (via timm) for transformer-based temporal modeling.
- Preprocess sequences of frames (BGR -> RGB, resize, normalize).
- Inference returns confidence score for 'fake' class and probabilities.

Compatibility:
- PyTorch 2.0.1 + CUDA 11.8
- timm 0.9.12 (for TimeSformer)
- loguru 0.7.2
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any, List

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
    logger.warning("timm not available. TimeSformer backbone will be disabled.")

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TemporalConfig:
    """
    Configuration for TemporalModel.
    """
    def __init__(
        self,
        backbone: str = "3d_cnn",  # options: "3d_cnn", "timesformer"
        num_classes: int = 2,
        input_size: Tuple[int, int] = (224, 224),
        sequence_length: int = 16,
        use_cuda: bool = True,
        checkpoint_path: Optional[str] = None
    ):
        """
        Args:
            backbone: Temporal backbone choice.
            num_classes: Output classes (default binary).
            input_size: (width, height) for frame resizing.
            sequence_length: Number of frames per clip.
            use_cuda: Use CUDA if available.
            checkpoint_path: Optional path to model weights.
        """
        self.backbone = backbone.lower()
        self.num_classes = num_classes
        self.input_size = input_size
        self.sequence_length = max(4, sequence_length)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.checkpoint_path = checkpoint_path


# ---------------------------
# 3D-CNN backbone (lightweight)
# ---------------------------
class Simple3DCNN(nn.Module):
    """
    Lightweight 3D-CNN for spatio-temporal feature extraction.
    Input shape: (B, C, T, H, W)
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class TemporalModel(nn.Module):
    """
    Temporal model for detecting motion inconsistencies.
    """

    def __init__(self, config: Optional[TemporalConfig] = None):
        super().__init__()
        self.config = config or TemporalConfig()
        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")
        self.model: Optional[nn.Module] = None

        self._build_model()
        self.to(self.device)
        self.eval()

    # ---------------------------
    # Model construction
    # ---------------------------
    def _build_model(self) -> None:
        try:
            if self.config.backbone == "3d_cnn":
                self.model = Simple3DCNN(num_classes=self.config.num_classes)
                logger.info("Loaded Simple3DCNN backbone.")
            elif self.config.backbone == "timesformer":
                if not TIMM_AVAILABLE:
                    raise RuntimeError("timm required for TimeSformer backbone.")
                # Use a small TimeSformer variant for video
                # timm model expects (B, T, C, H, W) or (B, C, T, H, W) depending on implementation.
                # We'll adapt input in forward.
                self.model = timm.create_model(
                    "timesformer_base_patch16_224",
                    pretrained=True,
                    num_classes=self.config.num_classes
                )
                # TimeSformer default input size 224x224
                self.config.input_size = (224, 224)
                logger.info("Loaded TimeSformer backbone via timm.")
            else:
                raise ValueError(f"Unsupported backbone: {self.config.backbone}")

            # Load checkpoint if provided
            if self.config.checkpoint_path and os.path.exists(self.config.checkpoint_path):
                state = torch.load(self.config.checkpoint_path, map_location="cpu")
                self.model.load_state_dict(state, strict=False)
                logger.info(f"Loaded checkpoint: {self.config.checkpoint_path}")

        except Exception as e:
            logger.error(f"Temporal model build failed: {e}")
            raise

    # ---------------------------
    # Preprocessing
    # ---------------------------
    def preprocess_sequence(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a list of BGR frames to a model tensor.
        - Convert BGR -> RGB
        - Resize to input_size
        - Normalize with ImageNet mean/std
        Returns:
            Tensor of shape:
              3D-CNN: (1, C, T, H, W)
              TimeSformer: (1, T, C, H, W)
        """
        try:
            w, h = self.config.input_size
            seq = []
            for f in frames_bgr:
                img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (w, h))
                img_float = img_resized.astype(np.float32) / 255.0
                img_norm = (img_float - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
                # HWC -> CHW
                img_chw = np.transpose(img_norm, (2, 0, 1))  # (C, H, W)
                seq.append(img_chw)

            # Stack along time dimension
            seq_np = np.stack(seq, axis=1)  # (C, T, H, W)
            tensor = torch.from_numpy(seq_np).unsqueeze(0).float()  # (1, C, T, H, W)

            if self.config.backbone == "timesformer":
                # TimeSformer expects (B, T, C, H, W)
                tensor = tensor.permute(0, 2, 1, 3, 4)  # (1, T, C, H, W)

            return tensor.to(self.device)
        except Exception as e:
            logger.error(f"Temporal preprocessing failed: {e}")
            # Return a dummy tensor to avoid crashing downstream
            if self.config.backbone == "timesformer":
                dummy = torch.zeros((1, self.config.sequence_length, 3, self.config.input_size[1], self.config.input_size[0]), dtype=torch.float32)
            else:
                dummy = torch.zeros((1, 3, self.config.sequence_length, self.config.input_size[1], self.config.input_size[0]), dtype=torch.float32)
            return dummy.to(self.device)

    # ---------------------------
    # Inference
    # ---------------------------
    @torch.inference_mode()
    def infer(self, frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
        """
        Perform inference on a sequence of frames.
        Args:
            frames_bgr: List of BGR frames (length should be >= sequence_length).
        Returns:
            dict with logits, probabilities, and confidence score for 'fake' class.
        """
        try:
            # Ensure sequence length
            if len(frames_bgr) < self.config.sequence_length:
                logger.warning(f"Insufficient frames: got {len(frames_bgr)}, required {self.config.sequence_length}. Padding by repeating last frame.")
                if len(frames_bgr) == 0:
                    return {"logits": None, "probs": None, "confidence_fake": 0.0}
                last = frames_bgr[-1]
                while len(frames_bgr) < self.config.sequence_length:
                    frames_bgr.append(last)

            x = self.preprocess_sequence(frames_bgr)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            probs_np = probs.detach().cpu().numpy()[0]

            # Assume class index 1 = 'fake'
            fake_conf = float(probs_np[1]) if self.config.num_classes >= 2 else float(probs_np.max())

            return {
                "logits": logits.detach().cpu().numpy()[0],
                "probs": probs_np,
                "confidence_fake": fake_conf
            }
        except Exception as e:
            logger.error(f"Temporal inference failed: {e}")
            return {
                "logits": None,
                "probs": None,
                "confidence_fake": 0.0
            }

    # ---------------------------
    # Utility
    # ---------------------------
    def summary(self) -> str:
        """
        Return a short summary of the temporal model configuration.
        """
        try:
            params = sum(p.numel() for p in self.model.parameters())
            return f"Backbone={self.config.backbone}, SeqLen={self.config.sequence_length}, Classes={self.config.num_classes}, Params={params}, Device={self.device}"
        except Exception as e:
            logger.error(f"Summary failed: {e}")
            return f"Backbone={self.config.backbone}, SeqLen={self.config.sequence_length}, Device={self.device}"
