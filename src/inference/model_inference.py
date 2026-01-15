"""
ModelInference Module for AegisVerity
-------------------------------------
Responsibilities:
- Unified wrapper for SpatialCNN, TemporalModel, and AVSyncModel.
- Run inference across visual, temporal, and audio-visual domains.
- Aggregate confidence scores and produce final verdict.
- Provide structured output for downstream VerityEngine.

Compatibility:
- PyTorch 2.0.1 + CUDA 11.8
- loguru 0.7.2
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from loguru import logger

# Import models
from models.spatial_cnn import SpatialCNNModel, SpatialCNNConfig
from models.temporal_model import TemporalModel, TemporalConfig
from models.av_sync_model import AVSyncModel, AVSyncConfig


class ModelInferenceConfig:
    """
    Configuration for unified model inference.
    """
    def __init__(
        self,
        spatial_backbone: str = "efficientnet_b4",
        temporal_backbone: str = "3d_cnn",
        av_n_mfcc: int = 20,
        sequence_length: int = 16,
        use_cuda: bool = True
    ):
        self.spatial_backbone = spatial_backbone
        self.temporal_backbone = temporal_backbone
        self.av_n_mfcc = av_n_mfcc
        self.sequence_length = sequence_length
        self.use_cuda = use_cuda


class ModelInference:
    """
    Unified inference wrapper for AegisVerity.
    """

    def __init__(self, config: Optional[ModelInferenceConfig] = None):
        self.config = config or ModelInferenceConfig()

        # Initialize sub-models
        try:
            self.spatial_model = SpatialCNNModel(
                SpatialCNNConfig(
                    backbone=self.config.spatial_backbone,
                    num_classes=2,
                    use_cuda=self.config.use_cuda
                )
            )
            logger.info("SpatialCNNModel initialized.")
        except Exception as e:
            logger.error(f"Failed to init SpatialCNNModel: {e}")
            self.spatial_model = None

        try:
            self.temporal_model = TemporalModel(
                TemporalConfig(
                    backbone=self.config.temporal_backbone,
                    num_classes=2,
                    sequence_length=self.config.sequence_length,
                    use_cuda=self.config.use_cuda
                )
            )
            logger.info("TemporalModel initialized.")
        except Exception as e:
            logger.error(f"Failed to init TemporalModel: {e}")
            self.temporal_model = None

        try:
            self.av_model = AVSyncModel(
                AVSyncConfig(
                    n_mfcc=self.config.av_n_mfcc,
                    sequence_length=self.config.sequence_length,
                    use_cuda=self.config.use_cuda
                )
            )
            logger.info("AVSyncModel initialized.")
        except Exception as e:
            logger.error(f"Failed to init AVSyncModel: {e}")
            self.av_model = None

    # ---------------------------
    # Inference methods
    # ---------------------------
    def infer_spatial(self, face_crop: np.ndarray) -> Dict[str, Any]:
        if not self.spatial_model:
            return {"confidence_fake": 0.0}
        return self.spatial_model.infer(face_crop)

    def infer_temporal(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        if not self.temporal_model:
            return {"confidence_fake": 0.0}
        return self.temporal_model.infer(frames)

    def infer_av(self, mfcc: np.ndarray, mouth_frames: List[np.ndarray]) -> Dict[str, Any]:
        if not self.av_model:
            return {"confidence_fake": 0.0}
        return self.av_model.infer(mfcc, mouth_frames)

    # ---------------------------
    # Unified inference
    # ---------------------------
    def infer_unified(
        self,
        face_crop: Optional[np.ndarray] = None,
        frames: Optional[List[np.ndarray]] = None,
        mfcc: Optional[np.ndarray] = None,
        mouth_frames: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Run inference across all domains and aggregate.
        Args:
            face_crop: Single face crop (BGR).
            frames: Sequence of frames (BGR).
            mfcc: MFCC feature matrix (n_mfcc x T).
            mouth_frames: Sequence of mouth crops (BGR).
        Returns:
            dict with domain scores and aggregated verdict.
        """
        results = {}

        try:
            if face_crop is not None:
                results["spatial"] = self.infer_spatial(face_crop)
            if frames is not None:
                results["temporal"] = self.infer_temporal(frames)
            if mfcc is not None and mouth_frames is not None:
                results["av_sync"] = self.infer_av(mfcc, mouth_frames)
        except Exception as e:
            logger.error(f"Unified inference failed: {e}")

        # Aggregate confidence
        confidences = []
        for domain in ["spatial", "temporal", "av_sync"]:
            if domain in results and results[domain].get("confidence_fake") is not None:
                confidences.append(results[domain]["confidence_fake"])

        if confidences:
            avg_conf = float(np.mean(confidences))
        else:
            avg_conf = 0.0

        verdict = "fake" if avg_conf >= 0.5 else "real"

        results["aggregate"] = {
            "avg_confidence_fake": avg_conf,
            "verdict": verdict
        }

        return results
