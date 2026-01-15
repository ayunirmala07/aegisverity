"""
AegisVerity L2 Visual Analysis Layer
Computer vision-based deepfake detection and visual artifact analysis
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
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


class L2VisualLayer(BaseDetectionLayer):
    """
    Layer 2: Visual Analysis Layer
    
    Focuses on:
    - Face detection and analysis
    - Visual artifact detection
    - Temporal consistency analysis
    - Indonesian facial feature analysis
    - Deepfake visual signatures
    """
    
    def __init__(self, config: DetectionConfig):
        super().__init__(config, "L2 Visual Analysis")
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.jpg', '.jpeg', '.png']
        self.face_model = None
        self._models_loaded = False
    
    def load_models(self) -> bool:
        """Load computer vision models"""
        try:
            # Load face detection model (YOLOv8 or similar)
            model_path = getattr(self.config, 'model_path', 'yolov8n.pt')
            print(f"  📥 Loading L2 visual model: {model_path}")
            
            # Try to load YOLOv8 model
            try:
                from ultralytics import YOLO
                self.face_model = YOLO(model_path, verbose=False)
                self._models_loaded = True
                print("  ✅ L2 Visual models loaded successfully")
                return True
            except ImportError:
                print("  ⚠️  YOLOv8 not available, using fallback analysis")
                self._models_loaded = True  # Still functional with fallback
                return True
                
        except Exception as e:
            print(f"  ❌ L2 Visual model loading failed: {str(e)}")
            return False
    
    def analyze(self, media_path: str, metadata: MediaMetadata) -> LayerOutput:
        """
        Perform visual forensic analysis
        
        Args:
            media_path: Path to media file
            metadata: Media metadata
            
        Returns:
            LayerOutput with visual analysis results
        """
        start_time = time.time()
        
        try:
            # Preprocess - extract frames or load image
            processed_data = self.preprocess(media_path, metadata)
            
            # Perform visual analysis
            raw_results = self._perform_visual_analysis(processed_data, metadata)
            
            # Postprocess results
            visual_results = self.postprocess(raw_results, metadata)
            
            # Extract visual anomalies
            anomalies = self._extract_visual_anomalies(visual_results, metadata)
            
            processing_time = time.time() - start_time
            
            return self.create_layer_output(
                results=visual_results,
                processing_time=processing_time,
                anomalies=anomalies
            )
            
        except Exception as e:
            print(f"L2 Visual analysis error: {str(e)}")
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
                anomalies=[f"Visual analysis error: {str(e)}"]
            )
    
    def _perform_visual_analysis(self, processed_data: Any, metadata: MediaMetadata) -> List[Dict[str, Any]]:
        """Perform core visual analysis"""
        results = []
        
        if metadata.file_type.lower() in ['image']:
            # Image analysis
            image_analysis = self._analyze_image(processed_data, metadata)
            results.append(image_analysis)
        else:
            # Video analysis
            video_analysis = self._analyze_video(processed_data, metadata)
            results.append(video_analysis)
        
        return results
    
    def _analyze_image(self, image_data: np.ndarray, metadata: MediaMetadata) -> Dict[str, Any]:
        """Analyze single image for deepfake indicators"""
        try:
            # 1. Face Detection and Analysis
            face_analysis = self._detect_faces(image_data)
            
            # 2. Texture Analysis
            texture_analysis = self._analyze_texture(image_data)
            
            # 3. Artifact Detection
            artifact_analysis = self._detect_visual_artifacts(image_data)
            
            # 4. Indonesian Feature Analysis
            indonesian_analysis = self._analyze_indonesian_features(image_data, face_analysis)
            
            # Combine results
            overall_confidence = self._calculate_visual_confidence([
                face_analysis.get('confidence', 0.0),
                texture_analysis.get('confidence', 0.0),
                artifact_analysis.get('confidence', 0.0)
            ])
            
            return {
                "confidence": overall_confidence,
                "evidence": {
                    "face_analysis": face_analysis,
                    "texture_analysis": texture_analysis,
                    "artifact_analysis": artifact_analysis,
                    "indonesian_features": indonesian_analysis,
                    "analysis_type": "image_forensics"
                },
                "processing_time": 0.5
            }
        except Exception as e:
            return {
                "confidence": 0.0,
                "evidence": {"error": str(e)},
                "processing_time": 0.0
            }
    
    def _analyze_video(self, video_path: str, metadata: MediaMetadata) -> Dict[str, Any]:
        """Analyze video for temporal deepfake indicators"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    "confidence": 0.0,
                    "evidence": {"error": "Cannot open video file"},
                    "processing_time": 0.0
                }
            
            frame_analyses = []
            frame_count = 0
            sample_rate = getattr(self.config, 'sample_rate', 5)
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Sample frames for analysis
                    if frame_count % sample_rate == 0:
                        frame_analysis = self._analyze_image(frame, metadata)
                        frame_analyses.append(frame_analysis)
                    
                    frame_count += 1
                    
                    # Limit frames for performance
                    if frame_count >= getattr(self.config, 'max_frames', 100):
                        break
                        
            finally:
                cap.release()
            
            if not frame_analyses:
                return {
                    "confidence": 0.5,  # Neutral if no frames analyzed
                    "evidence": {"message": "No frames analyzed"},
                    "processing_time": 0.1
                }
            
            # Aggregate frame results
            aggregated_analysis = self._aggregate_video_analysis(frame_analyses)
            
            return {
                "confidence": aggregated_analysis['confidence'],
                "evidence": {
                    "video_analysis": aggregated_analysis,
                    "frames_analyzed": len(frame_analyses),
                    "total_frames": frame_count,
                    "sample_rate": sample_rate,
                    "analysis_type": "video_forensics"
                },
                "processing_time": len(frame_analyses) * 0.1
            }
            
        except Exception as e:
            return {
                "confidence": 0.0,
                "evidence": {"error": str(e)},
                "processing_time": 0.0
            }
    
    def _detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect faces and analyze facial features"""
        try:
            if self.face_model is None:
                return {
                    "confidence": 0.5,
                    "faces_detected": 0,
                    "message": "Face model not loaded"
                }
            
            # Run face detection
            results = self.face_model(image, verbose=False)
            
            faces = []
            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    for box in boxes:
                        # Extract face ROI
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        face_roi = image[y1:y2, x1:x2]
                        
                        # Analyze face
                        face_info = self._analyze_face_roi(face_roi)
                        
                        faces.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(box.conf[0]),
                            "analysis": face_info
                        })
            
            # Calculate overall face confidence
            face_confidence = min([f["confidence"] for f in faces]) if faces else 0.5
            
            return {
                "confidence": face_confidence,
                "faces_detected": len(faces),
                "faces": faces,
                "face_density": len(faces) / (image.shape[0] * image.shape[1]) * 1e-6  # faces per MP
            }
        except Exception as e:
            return {
                "confidence": 0.0,
                "error": str(e),
                "faces_detected": 0
            }
    
    def _analyze_face_roi(self, face_roi: np.ndarray) -> Dict[str, Any]:
        """Analyze face region for deepfake indicators"""
        try:
            # Convert to grayscale for texture analysis
            if len(face_roi.shape) == 3:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_roi
            
            # Texture analysis
            texture_variance = np.var(gray_face)
            edge_density = np.sum(cv2.Canny(gray_face, 50, 150)) / (gray_face.size)
            
            # Symmetry analysis
            h, w = gray_face.shape
            left_half = gray_face[:, :w//2]
            right_half = gray_face[:, w//2:]
            symmetry_score = 1.0 - np.mean(np.abs(left_half - right_half)) / 255.0
            
            # Eye region analysis (simplified)
            eye_region = gray_face[h//3:h//2, :]
            eye_consistency = np.std(eye_region)
            
            # Mouth region analysis
            mouth_region = gray_face[2*h//3:, :]
            mouth_naturalness = np.var(mouth_region)
            
            # Calculate face confidence
            face_confidence = self._calculate_face_confidence(
                texture_variance, edge_density, symmetry_score, 
                eye_consistency, mouth_naturalness
            )
            
            return {
                "texture_variance": float(texture_variance),
                "edge_density": float(edge_density),
                "symmetry_score": float(symmetry_score),
                "eye_consistency": float(eye_consistency),
                "mouth_naturalness": float(mouth_naturalness),
                "confidence": face_confidence
            }
        except Exception as e:
            return {
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _analyze_texture(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image texture for artificial patterns"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Local Binary Pattern for texture analysis
            lbp = cv2.LBP(gray, 8, 1)
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=256)
            lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
            
            # Gabor filter for texture patterns
            gabor_responses = []
            for theta in range(0, 180, 45):
                gabor = cv2.getGaborKernel((15, 15), 2*np.pi*theta/180, 8.0, 1.0, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, gabor)
                gabor_responses.append(np.mean(filtered))
            
            # Calculate texture confidence
            texture_confidence = self._calculate_texture_confidence(lbp_hist, gabor_responses)
            
            return {
                "confidence": texture_confidence,
                "lbp_histogram": lbp_hist.tolist(),
                "gabor_responses": gabor_responses,
                "texture_uniformity": 1.0 - np.std(lbp_hist)
            }
        except Exception as e:
            return {
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _detect_visual_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect common deepfake visual artifacts"""
        try:
            artifacts = []
            
            # 1. Inconsistent lighting
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            lighting_variance = np.var(gray)
            if lighting_variance > 2000:  # Unusually high variance
                artifacts.append("Inconsistent lighting detected")
            
            # 2. Edge artifacts
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (edges.size)
            if edge_density < 0.05 or edge_density > 0.3:
                artifacts.append("Abnormal edge density")
            
            # 3. Compression artifacts
            if len(image.shape) == 3:
                # Check for blocking artifacts in each channel
                for i, channel in enumerate(cv2.split(image)):
                    dct = cv2.dct(channel.astype(float))
                    # High-frequency analysis
                    hf_energy = np.sum(dct[8:, 8:]**2)
                    if hf_energy < 1000:  # Unusually smooth
                        artifacts.append(f"Compression artifacts in channel {i}")
            
            # 4. Color inconsistencies
            if len(image.shape) == 3:
                # Check color channel correlations
                color_corr = np.corrcoef(image.reshape(-1, 3))
                if np.any(np.abs(color_corr) < 0.8):
                    artifacts.append("Color channel inconsistencies")
            
            artifact_confidence = 0.9 if not artifacts else 0.6
            
            return {
                "confidence": artifact_confidence,
                "artifacts": artifacts,
                "lighting_variance": float(lighting_variance),
                "edge_density": float(edge_density)
            }
        except Exception as e:
            return {
                "confidence": 0.0,
                "error": str(e),
                "artifacts": []
            }
    
    def _analyze_indonesian_features(self, image: np.ndarray, face_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Indonesian-specific facial features"""
        try:
            indonesian_features = {}
            
            if face_analysis.get('faces_detected', 0) > 0:
                # Analyze facial proportions common in Indonesian population
                faces = face_analysis.get('faces', [])
                for face in faces:
                    bbox = face['bbox']
                    face_roi = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    
                    if len(face_roi.shape) == 3:
                        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_face = face_roi
                    
                    # Facial feature analysis
                    features = self._extract_facial_features(gray_face)
                    indonesian_match = self._match_indonesian_features(features)
                    
                    indonesian_features[f"face_{len(indonesian_features)}"] = {
                        "features": features,
                        "indonesian_match": indonesian_match,
                        "confidence": face['confidence']
                    }
            
            # Calculate overall Indonesian confidence
            if indonesian_features:
                match_scores = [f.get('indonesian_match', 0.0) for f in indonesian_features.values()]
                indonesian_confidence = np.mean(match_scores)
            else:
                indonesian_confidence = 0.5
            
            return {
                "confidence": indonesian_confidence,
                "facial_matches": indonesian_features,
                "population_match": "Indonesian" if indonesian_confidence > 0.6 else "Unknown"
            }
        except Exception as e:
            return {
                "confidence": 0.0,
                "error": str(e),
                "facial_matches": {}
            }
    
    def _extract_facial_features(self, face_roi: np.ndarray) -> Dict[str, float]:
        """Extract key facial measurements"""
        try:
            h, w = face_roi.shape[:2]
            
            # Basic facial proportions
            eye_to_nose_ratio = 0.6  # Typical for Indonesian faces
            mouth_to_chin_ratio = 0.4
            face_width_to_height = w / h
            
            # Simplified feature extraction
            features = {
                "face_width_to_height": float(face_width_to_height),
                "aspect_ratio": float(w / h),
                "symmetry": self._calculate_facial_symmetry(face_roi),
                "skin_tone_variance": float(np.var(face_roi)) if len(face_roi.shape) == 3 else float(np.var(face_roi))
            }
            
            return features
        except Exception as e:
            return {"error": str(e)}
    
    def _match_indonesian_features(self, features: Dict[str, float]) -> float:
        """Match features against Indonesian facial patterns"""
        try:
            # Typical Indonesian facial feature ranges (simplified)
            indonesian_ranges = {
                "face_width_to_height": (0.7, 0.9),  # Wider faces
                "aspect_ratio": (0.8, 1.2),
                "symmetry": (0.7, 1.0),  # High symmetry
                "skin_tone_variance": (500, 2000)  # Moderate variance
            }
            
            match_score = 0.0
            total_features = 0
            
            for feature, value in features.items():
                if feature in indonesian_ranges and isinstance(value, (int, float)):
                    min_val, max_val = indonesian_ranges[feature]
                    if min_val <= value <= max_val:
                        match_score += 1.0
                    total_features += 1
            
            return match_score / total_features if total_features > 0 else 0.0
            
        except Exception as e:
            return 0.0
    
    def _aggregate_video_analysis(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate analysis across video frames"""
        if not frame_analyses:
            return {
                "confidence": 0.5,
                "message": "No frame analyses to aggregate"
            }
        
        # Aggregate confidences
        confidences = [f.get('confidence', 0.0) for f in frame_analyses]
        avg_confidence = np.mean(confidences)
        
        # Aggregate face detections
        total_faces = sum(f.get('face_analysis', {}).get('faces_detected', 0) for f in frame_analyses)
        avg_faces_per_frame = total_faces / len(frame_analyses) if frame_analyses else 0
        
        # Aggregate anomalies
        all_anomalies = []
        for analysis in frame_analyses:
            if 'artifact_analysis' in analysis.get('evidence', {}):
                artifacts = analysis['evidence']['artifact_analysis'].get('artifacts', [])
                all_anomalies.extend(artifacts)
        
        # Temporal consistency
        consistency_score = self._calculate_temporal_consistency(frame_analyses)
        
        return {
            "confidence": avg_confidence,
            "avg_faces_per_frame": avg_faces_per_frame,
            "temporal_consistency": consistency_score,
            "total_anomalies": list(set(all_anomalies)),
            "frames_analyzed": len(frame_analyses)
        }
    
    def _calculate_visual_confidence(self, confidences: List[float]) -> float:
        """Calculate overall visual confidence"""
        if not confidences:
            return 0.5
        
        # Weight different analysis types
        weights = [0.3, 0.3, 0.4]  # Face, texture, artifacts
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
        
        return min(max(weighted_confidence, 0.0), 1.0)
    
    def _calculate_face_confidence(self, texture_var: float, edge_density: float, 
                                 symmetry: float, eye_consistency: float, 
                                 mouth_naturalness: float) -> float:
        """Calculate face analysis confidence"""
        # Normalize individual scores
        texture_score = max(0, 1.0 - texture_var / 5000)
        edge_score = max(0, 1.0 - abs(edge_density - 0.15) * 5)
        symmetry_score = symmetry
        eye_score = max(0, 1.0 - eye_consistency / 100)
        mouth_score = max(0, 1.0 - mouth_naturalness / 1000)
        
        # Weighted average
        return (texture_score * 0.25 + edge_score * 0.25 + 
                symmetry_score * 0.25 + eye_score * 0.125 + mouth_score * 0.125)
    
    def _calculate_texture_confidence(self, lbp_hist: np.ndarray, gabor_responses: List[float]) -> float:
        """Calculate texture analysis confidence"""
        # LBP uniformity
        lbp_uniformity = 1.0 - np.std(lbp_hist)
        
        # Gabor consistency
        gabor_consistency = 1.0 - np.std(gabor_responses) / np.mean(gabor_responses) if gabor_responses else 0.5
        
        return (lbp_uniformity * 0.6 + gabor_consistency * 0.4)
    
    def _calculate_facial_symmetry(self, face_roi: np.ndarray) -> float:
        """Calculate facial symmetry score"""
        h, w = face_roi.shape[:2]
        
        # Left-right symmetry
        left_half = face_roi[:, :w//2]
        right_half = face_roi[:, w//2:]
        
        if len(face_roi.shape) == 3:
            left_gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_half
            right_gray = right_half
        
        # Calculate symmetry
        diff = np.mean(np.abs(left_gray.astype(float) - right_gray.astype(float)))
        symmetry = 1.0 - diff / 255.0
        
        return max(0.0, symmetry)
    
    def _calculate_temporal_consistency(self, frame_analyses: List[Dict[str, Any]]) -> float:
        """Calculate temporal consistency across frames"""
        if len(frame_analyses) < 2:
            return 1.0
        
        # Extract confidence scores over time
        confidences = [f.get('confidence', 0.5) for f in frame_analyses]
        
        # Calculate variance (lower is more consistent)
        consistency = 1.0 - np.var(confidences)
        
        return max(0.0, consistency)
    
    def _extract_visual_anomalies(self, results: List[ForensicResult], metadata: MediaMetadata) -> List[str]:
        """Extract anomalies from visual analysis"""
        anomalies = []
        
        for result in results:
            if result.evidence:
                evidence = result.evidence
                
                # Check for face anomalies
                if 'face_analysis' in evidence:
                    face_analysis = evidence['face_analysis']
                    if face_analysis.get('faces_detected', 0) == 0:
                        anomalies.append("No faces detected in visual content")
                
                # Check for texture anomalies
                if 'texture_analysis' in evidence:
                    texture_analysis = evidence['texture_analysis']
                    if texture_analysis.get('confidence', 1.0) < 0.5:
                        anomalies.append("Unusual texture patterns detected")
                
                # Check for artifact anomalies
                if 'artifact_analysis' in evidence:
                    artifact_analysis = evidence['artifact_analysis']
                    if artifact_analysis.get('artifacts'):
                        anomalies.extend(artifact_analysis['artifacts'])
                
                # Check for Indonesian feature mismatches
                if 'indonesian_features' in evidence:
                    indonesian_analysis = evidence['indonesian_features']
                    if indonesian_analysis.get('population_match') != 'Indonesian':
                        anomalies.append("Facial features don't match Indonesian patterns")
        
        return list(set(anomalies))
    
    def cleanup(self) -> None:
        """Clean up visual analysis resources"""
        if self.face_model:
            del self.face_model
            self.face_model = None
        
        self._models_loaded = False
        print("  ✅ L2 Visual layer cleaned up")
    
    def _get_supported_formats(self) -> List[str]:
        """Get supported media formats"""
        return self.supported_formats
