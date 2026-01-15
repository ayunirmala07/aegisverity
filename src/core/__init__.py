"""
AegisVerity Core Module
Abstract base classes and data types for layered forensic analysis
"""

from .base_layer import BaseDetectionLayer
from .data_types import (
    ForensicResult,
    DetectionConfig,
    LayerOutput,
    FusionResult
)
from .pipeline import ForensicPipeline

__all__ = [
    "BaseDetectionLayer",
    "ForensicResult", 
    "DetectionConfig",
    "LayerOutput",
    "FusionResult",
    "ForensicPipeline"
]
