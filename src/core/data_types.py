"""
AegisVerity Data Types
Core data structures for forensic analysis pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import numpy as np


class DetectionStatus(Enum):
    """Detection result status enumeration"""
    AUTHENTIC = "AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"
    MANIPULATED = "MANIPULATED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class ForensicResult:
    """Base forensic analysis result"""
    status: DetectionStatus
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    layer_name: str = ""
    analysis_id: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "status": self.status.value,
            "confidence_score": self.confidence,
            "evidence": self.evidence,
            "metadata": self.metadata,
            "layer_name": self.layer_name,
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp
        }


@dataclass
class DetectionConfig:
    """Configuration for detection layers"""
    model_path: str = ""
    confidence_threshold: float = 0.5
    enable_gpu: bool = True
    batch_size: int = 1
    max_frames: int = 100
    sample_rate: int = 5
    indonesian_optimized: bool = True
    debug_mode: bool = False
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        return (
            0.0 <= self.confidence_threshold <= 1.0 and
            self.batch_size > 0 and
            self.max_frames > 0 and
            self.sample_rate > 0
        )


@dataclass
class LayerOutput:
    """Output from individual detection layer"""
    layer_id: str
    layer_name: str
    results: List[ForensicResult]
    processing_time: float
    confidence_scores: List[float]
    aggregated_confidence: float
    anomalies: List[str] = field(default_factory=list)
    
    def get_best_result(self) -> Optional[ForensicResult]:
        """Get result with highest confidence"""
        if not self.results:
            return None
        return max(self.results, key=lambda x: x.confidence)


@dataclass
class FusionResult:
    """Fused result from multiple layers"""
    final_status: DetectionStatus
    final_confidence: float
    layer_outputs: Dict[str, LayerOutput]
    fusion_method: str
    explanation: str
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    consensus_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "final_status": self.final_status.value,
            "final_confidence": self.final_confidence,
            "fusion_method": self.fusion_method,
            "explanation": self.explanation,
            "supporting_evidence": self.supporting_evidence,
            "consensus_score": self.consensus_score,
            "layer_outputs": {
                layer_id: {
                    "layer_name": output.layer_name,
                    "aggregated_confidence": output.aggregated_confidence,
                    "anomalies": output.anomalies,
                    "processing_time": output.processing_time
                }
                for layer_id, output in self.layer_outputs.items()
            }
        }


@dataclass
class MediaMetadata:
    """Media file metadata"""
    file_path: str
    file_type: str
    duration: Optional[float] = None
    fps: Optional[float] = None
    resolution: Optional[tuple] = None
    audio_channels: Optional[int] = None
    sample_rate: Optional[int] = None
    file_size: int = 0
    format: str = ""
    codec: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "duration": self.duration,
            "fps": self.fps,
            "resolution": self.resolution,
            "audio_channels": self.audio_channels,
            "sample_rate": self.sample_rate,
            "file_size": self.file_size,
            "format": self.format,
            "codec": self.codec
        }


@dataclass
class AnomalyReport:
    """Individual anomaly report"""
    anomaly_type: str
    severity: str  # "low", "medium", "high", "critical"
    confidence: float
    description: str
    location: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "description": self.description,
            "location": self.location,
            "timestamp": self.timestamp
        }
