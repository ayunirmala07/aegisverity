"""
AegisVerity - Advanced Deepfake Detection Framework

A comprehensive multi-modal deepfake detection system featuring:
- Spatial CNN analysis (EfficientNet-B4/Xception)
- Temporal analysis (ConvLSTM/3D-CNN/TimeSformer)
- Audio-visual synchronization detection
- Explainability (Grad-CAM, attention maps)
- Adversarial defense mechanisms
"""

__version__ = "1.0.0"
__author__ = "AegisVerity Team"

from src.engine.verity_engine import VerityEngine

__all__ = ['VerityEngine']