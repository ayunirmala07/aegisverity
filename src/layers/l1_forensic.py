"""
AegisVerity L1 Forensic Analysis Layer
Base forensic analysis focusing on metadata and file integrity
"""

import os
import hashlib
import json
from typing import Dict, List, Any, Optional
import time
from pathlib import Path

from ..core.base_layer import BaseDetectionLayer
from ..core.data_types import (
    ForensicResult, 
    DetectionConfig, 
    LayerOutput, 
    MediaMetadata,
    AnomalyReport
)


class L1ForensicLayer(BaseDetectionLayer):
    """
    Layer 1: Forensic Analysis Layer
    
    Focuses on:
    - File metadata analysis
    - Hash verification and integrity checking
    - Format and codec analysis
    - Editing software detection
    - Timestamp consistency analysis
    """
    
    def __init__(self, config: DetectionConfig):
        super().__init__(config, "L1 Forensic Analysis")
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wav', '.mp3', '.m4a', '.flac', '.aac']
        self._models_loaded = False
    
    def load_models(self) -> bool:
        """Load forensic analysis models"""
        try:
            # L1 primarily uses built-in analysis, no external models needed
            self._models_loaded = True
            print("  ✅ L1 Forensic models loaded (built-in analysis)")
            return True
        except Exception as e:
            print(f"  ❌ L1 Forensic model loading failed: {str(e)}")
            return False
    
    def analyze(self, media_path: str, metadata: MediaMetadata) -> LayerOutput:
        """
        Perform comprehensive forensic analysis
        
        Args:
            media_path: Path to media file
            metadata: Media metadata
            
        Returns:
            LayerOutput with forensic analysis results
        """
        start_time = time.time()
        
        try:
            # Preprocess - extract file information
            processed_data = self.preprocess(media_path, metadata)
            
            # Perform forensic analysis
            raw_results = self._perform_forensic_analysis(processed_data, metadata)
            
            # Postprocess results
            forensic_results = self.postprocess(raw_results, metadata)
            
            # Extract anomalies
            anomalies = self._extract_forensic_anomalies(forensic_results, metadata)
            
            processing_time = time.time() - start_time
            
            return self.create_layer_output(
                results=forensic_results,
                processing_time=processing_time,
                anomalies=anomalies
            )
            
        except Exception as e:
            print(f"L1 Forensic analysis error: {str(e)}")
            error_result = ForensicResult(
                status="ERROR",
                confidence=0.0,
                evidence={"error": str(e)},
                layer_name=self.layer_name,
                analysis_id=f"{self.layer_id}_error",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            return self.create_layer_output(
                results=[error_result],
                processing_time=time.time() - start_time,
                anomalies=[f"Analysis error: {str(e)}"]
            )
    
    def _perform_forensic_analysis(self, media_path: str, metadata: MediaMetadata) -> List[Dict[str, Any]]:
        """Perform core forensic analysis"""
        results = []
        
        # 1. File Hash Analysis
        hash_analysis = self._analyze_file_hashes(media_path)
        results.append(hash_analysis)
        
        # 2. Metadata Analysis
        metadata_analysis = self._analyze_file_metadata(media_path, metadata)
        results.append(metadata_analysis)
        
        # 3. Format and Codec Analysis
        format_analysis = self._analyze_format_consistency(media_path, metadata)
        results.append(format_analysis)
        
        # 4. Editing Software Detection
        editing_analysis = self._detect_editing_software(media_path, metadata)
        results.append(editing_analysis)
        
        # 5. Timestamp Analysis
        timestamp_analysis = self._analyze_timestamp_consistency(media_path, metadata)
        results.append(timestamp_analysis)
        
        return results
    
    def _analyze_file_hashes(self, file_path: str) -> Dict[str, Any]:
        """Analyze file hashes for integrity verification"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Calculate multiple hash algorithms
            md5_hash = hashlib.md5(content).hexdigest()
            sha256_hash = hashlib.sha256(content).hexdigest()
            
            return {
                "confidence": 0.8,  # Hash analysis is deterministic
                "evidence": {
                    "md5": md5_hash,
                    "sha256": sha256_hash,
                    "file_size": len(content),
                    "analysis_type": "hash_verification"
                },
                "processing_time": 0.1
            }
        except Exception as e:
            return {
                "confidence": 0.0,
                "evidence": {"error": str(e)},
                "processing_time": 0.0
            }
    
    def _analyze_file_metadata(self, file_path: str, metadata: MediaMetadata) -> Dict[str, Any]:
        """Analyze file metadata for inconsistencies"""
        try:
            # Check for metadata manipulation
            anomalies = []
            
            # File extension vs actual format mismatch
            file_ext = Path(file_path).suffix.lower()
            if metadata.format and file_ext != f".{metadata.format.lower()}":
                anomalies.append("Extension mismatch detected")
            
            # Unusual metadata patterns
            if metadata.duration and metadata.duration < 0.1:
                anomalies.append("Suspiciously short duration")
            
            if metadata.fps and (metadata.fps < 10 or metadata.fps > 120):
                anomalies.append("Unusual frame rate detected")
            
            confidence = 0.9 if not anomalies else 0.6
            
            return {
                "confidence": confidence,
                "evidence": {
                    "metadata": metadata.to_dict(),
                    "anomalies": anomalies,
                    "analysis_type": "metadata_verification"
                },
                "processing_time": 0.2
            }
        except Exception as e:
            return {
                "confidence": 0.0,
                "evidence": {"error": str(e)},
                "processing_time": 0.0
            }
    
    def _analyze_format_consistency(self, file_path: str, metadata: MediaMetadata) -> Dict[str, Any]:
        """Analyze format and codec consistency"""
        try:
            anomalies = []
            
            # Check for suspicious codecs
            suspicious_codecs = ['divx', 'xvid', 'wmv2', 'mpeg4']
            if metadata.codec and metadata.codec.lower() in suspicious_codecs:
                anomalies.append(f"Suspicious codec: {metadata.codec}")
            
            # Check for unusual aspect ratios
            if metadata.resolution:
                width, height = metadata.resolution
                aspect_ratio = width / height
                if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                    anomalies.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
            
            confidence = 0.85 if not anomalies else 0.7
            
            return {
                "confidence": confidence,
                "evidence": {
                    "format_analysis": {
                        "codec": metadata.codec,
                        "aspect_ratio": aspect_ratio if metadata.resolution else None,
                        "anomalies": anomalies
                    },
                    "analysis_type": "format_consistency"
                },
                "processing_time": 0.15
            }
        except Exception as e:
            return {
                "confidence": 0.0,
                "evidence": {"error": str(e)},
                "processing_time": 0.0
            }
    
    def _detect_editing_software(self, file_path: str, metadata: MediaMetadata) -> Dict[str, Any]:
        """Detect traces of video/audio editing software"""
        try:
            # This is a simplified implementation
            # In production, would use more sophisticated techniques
            editing_indicators = []
            
            # Check for common editing software signatures
            file_path_lower = file_path.lower()
            software_signatures = {
                'adobe': ['premiere', 'after effects', 'audition'],
                'final cut': ['final cut', 'motion'],
                'davinci': ['davinci', 'resolve'],
                'filmora': ['filmora', 'wondershare']
            }
            
            detected_software = []
            for company, products in software_signatures.items():
                for product in products:
                    if product in file_path_lower:
                        detected_software.append(f"{company} {product}")
                        editing_indicators.append(f"Software signature: {company} {product}")
            
            confidence = 0.6 if editing_indicators else 0.9
            
            return {
                "confidence": confidence,
                "evidence": {
                    "editing_indicators": editing_indicators,
                    "detected_software": detected_software,
                    "analysis_type": "editing_detection"
                },
                "processing_time": 0.25
            }
        except Exception as e:
            return {
                "confidence": 0.0,
                "evidence": {"error": str(e)},
                "processing_time": 0.0
            }
    
    def _analyze_timestamp_consistency(self, file_path: str, metadata: MediaMetadata) -> Dict[str, Any]:
        """Analyze timestamp consistency across metadata"""
        try:
            timestamp_anomalies = []
            
            # Check creation vs modification timestamps
            if os.path.exists(file_path):
                stat_info = os.stat(file_path)
                creation_time = stat_info.st_ctime
                modification_time = stat_info.st_mtime
                
                # Large gap might indicate tampering
                time_diff = abs(modification_time - creation_time)
                if time_diff > 86400:  # More than 24 hours
                    timestamp_anomalies.append("Large creation-modification time gap")
            
            # Check for future timestamps
            current_time = time.time()
            if creation_time > current_time or modification_time > current_time:
                timestamp_anomalies.append("Future timestamp detected")
            
            confidence = 0.8 if not timestamp_anomalies else 0.9
            
            return {
                "confidence": confidence,
                "evidence": {
                    "timestamp_analysis": {
                        "creation_time": creation_time,
                        "modification_time": modification_time,
                        "anomalies": timestamp_anomalies
                    },
                    "analysis_type": "timestamp_consistency"
                },
                "processing_time": 0.1
            }
        except Exception as e:
            return {
                "confidence": 0.0,
                "evidence": {"error": str(e)},
                "processing_time": 0.0
            }
    
    def _extract_forensic_anomalies(self, results: List[ForensicResult], metadata: MediaMetadata) -> List[str]:
        """Extract anomalies from forensic analysis results"""
        anomalies = []
        
        for result in results:
            if result.evidence and "anomalies" in result.evidence:
                anomalies.extend(result.evidence["anomalies"])
            
            # Check for low confidence results
            if result.confidence < 0.5:
                anomalies.append(f"Low confidence in {result.evidence.get('analysis_type', 'unknown')}")
        
        return list(set(anomalies))  # Remove duplicates
    
    def cleanup(self) -> None:
        """Clean up forensic analysis resources"""
        # L1 doesn't use external resources that need cleanup
        self._models_loaded = False
        print("  ✅ L1 Forensic layer cleaned up")
    
    def _get_supported_formats(self) -> List[str]:
        """Get supported media formats"""
        return self.supported_formats
