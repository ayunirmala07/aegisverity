"""
VerityEngine for AegisVerity
----------------------------
Responsibilities:
- Orchestrate end-to-end deepfake analysis:
  - Ingest video via VideoLoader (FFmpeg/OpenCV)
  - Detect & extract faces via FaceExtractor
  - Extract audio MFCC via AudioProcessor
  - Run unified inference (Spatial, Temporal, AVSync)
  - Generate explainability artifacts (Grad-CAM)
  - Apply basic adversarial checks
  - Produce forensic report with domain scores and signals

Compatibility:
- PyTorch 2.0.1 + CUDA 11.8
- ffmpeg-python, OpenCV, librosa
- loguru 0.7.2
"""

from __future__ import annotations

import os
import uuid
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from loguru import logger

# Pipeline modules
from pipeline.video_loader import VideoLoader, VideoLoaderConfig
from pipeline.face_extractor import FaceExtractor
from pipeline.audio_processor import AudioProcessor, AudioProcessorConfig

# Inference modules
from inference.model_inference import ModelInference, ModelInferenceConfig

# Explainability
from inference.explainability import ExplainabilityEngine, ExplainabilityConfig

# Defense checks
from defense.adversarial_checks import AdversarialChecks


@dataclass
class VerityConfig:
    """
    Configuration for VerityEngine.
    """
    # Video ingestion
    target_size: tuple[int, int] = (640, 360)
    fixed_stride: int = 5
    event_driven: bool = True
    face_threshold: int = 1
    max_frames: Optional[int] = 128
    use_ffmpeg: bool = True

    # Audio
    sample_rate: int = 16000
    n_mfcc: int = 20
    audio_duration: Optional[float] = None

    # Models
    spatial_backbone: str = "efficientnet_b4"
    temporal_backbone: str = "3d_cnn"
    sequence_length: int = 16
    use_cuda: bool = True

    # Explainability
    enable_gradcam: bool = True
    gradcam_method: str = "gradcam"

    # Output
    output_dir: str = "data/outputs"
    save_heatmaps: bool = True
    save_report_json: bool = True


@dataclass
class VerityResult:
    """
    Structured result for forensic output.
    """
    task_id: str
    metadata: Dict[str, Any]
    domain_scores: Dict[str, Any]
    aggregate: Dict[str, Any]
    signals: List[str]
    artifacts: Dict[str, Any]


class VerityEngine:
    """
    Orchestrator for AegisVerity deepfake detection.
    """

    def __init__(self, config: Optional[VerityConfig] = None):
        self.config = config or VerityConfig()
        self.task_id = str(uuid.uuid4())

        # Ensure output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Initialize pipeline
        try:
            self.video_loader = VideoLoader(
                VideoLoaderConfig(
                    target_size=self.config.target_size,
                    fixed_stride=self.config.fixed_stride,
                    event_driven=self.config.event_driven,
                    face_threshold=self.config.face_threshold,
                    max_frames=self.config.max_frames,
                    use_ffmpeg=self.config.use_ffmpeg
                )
            )
            logger.info("VideoLoader initialized.")
        except Exception as e:
            logger.error(f"Failed to init VideoLoader: {e}")
            raise

        try:
            self.face_extractor = FaceExtractor(target_size=(224, 224))
            logger.info("FaceExtractor initialized.")
        except Exception as e:
            logger.error(f"Failed to init FaceExtractor: {e}")
            raise

        try:
            self.audio_processor = AudioProcessor(
                AudioProcessorConfig(
                    sample_rate=self.config.sample_rate,
                    n_mfcc=self.config.n_mfcc,
                    duration=self.config.audio_duration
                )
            )
            logger.info("AudioProcessor initialized.")
        except Exception as e:
            logger.error(f"Failed to init AudioProcessor: {e}")
            raise

        # Initialize inference
        try:
            self.inference = ModelInference(
                ModelInferenceConfig(
                    spatial_backbone=self.config.spatial_backbone,
                    temporal_backbone=self.config.temporal_backbone,
                    av_n_mfcc=self.config.n_mfcc,
                    sequence_length=self.config.sequence_length,
                    use_cuda=self.config.use_cuda
                )
            )
            logger.info("ModelInference initialized.")
        except Exception as e:
            logger.error(f"Failed to init ModelInference: {e}")
            raise

        # Explainability
        try:
            self.explain = ExplainabilityEngine(
                ExplainabilityConfig(
                    use_cuda=self.config.use_cuda,
                    method=self.config.gradcam_method
                )
            )
            logger.info("ExplainabilityEngine initialized.")
        except Exception as e:
            logger.error(f"Failed to init ExplainabilityEngine: {e}")
            self.explain = None

        # Defense checks
        try:
            self.defense = AdversarialChecks()
            logger.info("AdversarialChecks initialized.")
        except Exception as e:
            logger.error(f"Failed to init AdversarialChecks: {e}")
            self.defense = None

    # ---------------------------
    # Core orchestration
    # ---------------------------
    def analyze_video(self, video_path: str, audio_path: Optional[str] = None) -> VerityResult:
        """
        Run full analysis on a video:
        - Probe metadata
        - Sample frames
        - Extract faces & mouth crops
        - Extract MFCC (from audio file or video audio track)
        - Run unified inference
        - Generate Grad-CAM heatmaps
        - Apply defense checks
        - Save artifacts & report
        """
        metadata = {}
        domain_scores = {}
        signals: List[str] = []
        artifacts: Dict[str, Any] = {}

        # Probe metadata
        try:
            meta = self.video_loader.probe(video_path)
            metadata.update(meta)
        except Exception as e:
            logger.error(f"Metadata probe failed: {e}")

        # Collect frames
        frames: List[np.ndarray] = []
        try:
            for frame in self.video_loader.frames(video_path):
                frames.append(frame)
            logger.info(f"Collected {len(frames)} sampled frames.")
        except Exception as e:
            logger.error(f"Frame sampling failed: {e}")

        # Extract faces (first face per frame for simplicity)
        face_crops: List[np.ndarray] = []
        mouth_crops: List[np.ndarray] = []
        try:
            for f in frames:
                faces = self.face_extractor.process_frame(f)
                if faces:
                    face_crops.append(faces[0])
                    # Derive mouth crop from face (simple heuristic: lower third)
                    h, w, _ = faces[0].shape
                    y1 = int(h * 2 / 3)
                    mouth = faces[0][y1:h, 0:w]
                    mouth_crops.append(mouth)
            logger.info(f"Extracted {len(face_crops)} face crops and {len(mouth_crops)} mouth crops.")
        except Exception as e:
            logger.error(f"Face extraction failed: {e}")

        # Prepare temporal sequence (use face crops or original frames)
        temporal_seq = face_crops[:self.config.sequence_length] if face_crops else frames[:self.config.sequence_length]

        # Audio MFCC
        mfcc = None
        try:
            if audio_path and os.path.exists(audio_path):
                audio_result = self.audio_processor.process(audio_path)
                mfcc = audio_result.get("mfcc")
            else:
                # Optional: extract audio from video via ffmpeg to temp file (not implemented here)
                logger.warning("No audio_path provided; AVSync will be skipped unless MFCC is available.")
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")

        # Inference per domain
        try:
            # Spatial: use first face crop if available
            if face_crops:
                spatial_res = self.inference.infer_spatial(face_crops[0])
                domain_scores["spatial"] = spatial_res
            else:
                logger.warning("No face crops available for spatial inference.")

            # Temporal: use sequence
            if temporal_seq and len(temporal_seq) > 0:
                temporal_res = self.inference.infer_temporal(temporal_seq)
                domain_scores["temporal"] = temporal_res
            else:
                logger.warning("No frames available for temporal inference.")

            # AVSync: requires mfcc + mouth frames
            if mfcc is not None and mouth_crops:
                av_res = self.inference.infer_av(mfcc, mouth_crops[:self.config.sequence_length])
                domain_scores["av_sync"] = av_res
            else:
                logger.warning("AVSync inference skipped (missing MFCC or mouth frames).")
        except Exception as e:
            logger.error(f"Model inference failed: {e}")

        # Aggregate verdict
        try:
            aggregate = self.inference.infer_unified(
                face_crop=face_crops[0] if face_crops else None,
                frames=temporal_seq if temporal_seq else None,
                mfcc=mfcc,
                mouth_frames=mouth_crops[:self.config.sequence_length] if mouth_crops else None
            ).get("aggregate", {"avg_confidence_fake": 0.0, "verdict": "real"})
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            aggregate = {"avg_confidence_fake": 0.0, "verdict": "real"}

        # Explainability (Grad-CAM) for spatial domain
        try:
            if self.config.enable_gradcam and self.explain and face_crops and hasattr(self.inference.spatial_model, "_target_layers"):
                # Prepare tensor via spatial model's preprocess
                tensor = self.inference.spatial_model.preprocess(face_crops[0])
                target_layers = self.inference.spatial_model._target_layers
                heatmap = self.explain.generate_heatmap(
                    model=self.inference.spatial_model.model,
                    input_tensor=tensor,
                    target_layers=target_layers,
                    target_class=1,  # 'fake' class
                    original_image_bgr=face_crops[0]
                )
                if heatmap is not None and self.config.save_heatmaps:
                    heatmap_path = os.path.join(self.config.output_dir, f"{self.task_id}_gradcam.jpg")
                    cv2.imwrite(heatmap_path, heatmap)
                    artifacts["gradcam_spatial"] = heatmap_path
            else:
                logger.info("Grad-CAM skipped or unavailable.")
        except Exception as e:
            logger.error(f"Explainability failed: {e}")

        # Defense checks
        try:
            if self.defense and frames:
                defense_signals = self.defense.evaluate_frames(frames)
                signals.extend(defense_signals)
        except Exception as e:
            logger.error(f"Defense checks failed: {e}")

        # Build result
        result = VerityResult(
            task_id=self.task_id,
            metadata=metadata,
            domain_scores=domain_scores,
            aggregate=aggregate,
            signals=signals,
            artifacts=artifacts
        )

        # Save report
        try:
            if self.config.save_report_json:
                report_path = os.path.join(self.config.output_dir, f"{self.task_id}_report.json")
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(result), f, indent=2)
                artifacts["report_json"] = report_path
        except Exception as e:
            logger.error(f"Failed to save report JSON: {e}")

        return result

    # ---------------------------
    # Convenience
    # ---------------------------
    def summary(self) -> Dict[str, Any]:
        """
        Return a summary of engine configuration and model status.
        """
        try:
            return {
                "task_id": self.task_id,
                "config": asdict(self.config),
                "spatial_model": self.inference.spatial_model.summary() if self.inference.spatial_model else "N/A",
                "temporal_model": self.inference.temporal_model.summary() if self.inference.temporal_model else "N/A",
                "av_model": self.inference.av_model.summary() if self.inference.av_model else "N/A"
            }
        except Exception as e:
            logger.error(f"Summary failed: {e}")
            return {"task_id": self.task_id, "config": asdict(self.config)}
