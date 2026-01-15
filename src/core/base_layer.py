"""
AegisVerity Base Detection Layer
Abstract base class for all forensic analysis layers
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import time
import uuid
from .data_types import ForensicResult, DetectionConfig, LayerOutput, MediaMetadata


class BaseDetectionLayer(ABC):
    """
    Abstract base class for forensic detection layers
    
    Each layer implements specific detection algorithms:
    - L1: Forensic Analysis Layer
    - L2: Visual Analysis Layer  
    - L3: Audio-Visual Analysis Layer
    - L4: Audio Analysis Layer
    - L5: Explainability Layer
    - L6: Fusion Layer
    """
    
    def __init__(self, config: DetectionConfig, layer_name: str):
        self.config = config
        self.layer_name = layer_name
        self.layer_id = f"{layer_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        
        # Validate configuration
        if not config.validate():
            raise ValueError(f"Invalid configuration for {layer_name}")
    
    @abstractmethod
    def analyze(self, media_path: str, metadata: MediaMetadata) -> LayerOutput:
        """
        Perform forensic analysis on media file
        
        Args:
            media_path: Path to media file
            metadata: Media metadata information
            
        Returns:
            LayerOutput with analysis results
        """
        pass
    
    @abstractmethod
    def load_models(self) -> bool:
        """
        Load required models for this layer
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up resources and models
        """
        pass
    
    def preprocess(self, media_path: str, metadata: MediaMetadata) -> Any:
        """
        Preprocess media file for analysis
        
        Args:
            media_path: Path to media file
            metadata: Media metadata
            
        Returns:
            Preprocessed data ready for analysis
        """
        # Default implementation returns the path
        return media_path
    
    def postprocess(self, raw_results: List[Any], metadata: MediaMetadata) -> List[ForensicResult]:
        """
        Postprocess raw detection results
        
        Args:
            raw_results: Raw results from detection algorithms
            metadata: Media metadata
            
        Returns:
            List of formatted ForensicResult objects
        """
        # Default implementation converts to basic results
        processed_results = []
        
        for i, result in enumerate(raw_results):
            forensic_result = ForensicResult(
                status=self._determine_status(result.get("confidence", 0.0)),
                confidence=result.get("confidence", 0.0),
                evidence=result.get("evidence", {}),
                metadata={
                    "layer": self.layer_name,
                    "processing_time": result.get("processing_time", 0.0),
                    "model_version": getattr(self.config, 'model_version', 'unknown')
                },
                layer_name=self.layer_name,
                analysis_id=f"{self.layer_id}_{i}",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            processed_results.append(forensic_result)
        
        return processed_results
    
    def _determine_status(self, confidence: float) -> str:
        """
        Determine detection status based on confidence
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Status string
        """
        if confidence >= 0.8:
            return "MANIPULATED"
        elif confidence >= 0.6:
            return "SUSPICIOUS"
        elif confidence >= 0.4:
            return "AUTHENTIC"
        else:
            return "UNKNOWN"
    
    def create_layer_output(
        self, 
        results: List[ForensicResult], 
        processing_time: float,
        anomalies: List[str] = None
    ) -> LayerOutput:
        """
        Create standardized layer output
        
        Args:
            results: List of forensic results
            processing_time: Total processing time
            anomalies: List of detected anomalies
            
        Returns:
            LayerOutput object
        """
        if anomalies is None:
            anomalies = []
        
        confidence_scores = [r.confidence for r in results]
        aggregated_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return LayerOutput(
            layer_id=self.layer_id,
            layer_name=self.layer_name,
            results=results,
            processing_time=processing_time,
            confidence_scores=confidence_scores,
            aggregated_confidence=aggregated_confidence,
            anomalies=anomalies
        )
    
    def get_layer_info(self) -> Dict[str, Any]:
        """
        Get layer information for debugging and monitoring
        
        Returns:
            Dictionary with layer information
        """
        return {
            "layer_id": self.layer_id,
            "layer_name": self.layer_name,
            "config": self.config.__dict__,
            "models_loaded": hasattr(self, '_models_loaded'),
            "supported_formats": self._get_supported_formats()
        }
    
    @abstractmethod
    def _get_supported_formats(self) -> List[str]:
        """
        Get list of supported media formats
        
        Returns:
            List of supported file extensions
        """
        pass
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.layer_name} (ID: {self.layer_id})"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"{self.__class__.__name__}(name='{self.layer_name}', id='{self.layer_id}')"
