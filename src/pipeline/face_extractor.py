"""
FaceExtractor Module for AegisVerity
------------------------------------
Responsibilities:
- Detect faces using RetinaFace (primary) or MTCNN (fallback).
- Align faces using landmarks (dlib/MediaPipe).
- Normalize (resize, crop) for downstream models.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from loguru import logger

# RetinaFace
try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False
    logger.warning("RetinaFace not available. Will fallback to MTCNN.")

# MTCNN fallback
try:
    from mtcnn.mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logger.warning("MTCNN not available. Face detection may fail.")


class FaceDetection:
    """Container for face detection results."""
    
    def __init__(self, box: List[int], confidence: float, 
                 landmarks: Optional[Dict[str, Tuple[int, int]]] = None):
        """
        Initialize face detection.
        
        Args:
            box: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence score
            landmarks: Facial landmarks dictionary
        """
        self.box = box
        self.confidence = confidence
        self.landmarks = landmarks
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box as tuple."""
        return tuple(self.box)
    
    @property
    def width(self) -> int:
        """Get face width."""
        return self.box[2] - self.box[0]
    
    @property
    def height(self) -> int:
        """Get face height."""
        return self.box[3] - self.box[1]
    
    @property
    def area(self) -> int:
        """Get face area."""
        return self.width * self.height
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'box': self.box,
            'confidence': self.confidence,
            'landmarks': self.landmarks,
            'width': self.width,
            'height': self.height,
            'area': self.area
        }


class FaceExtractor:
    """Face detection and alignment using RetinaFace or MTCNN."""
    
    def __init__(self, config: dict):
        """
        Initialize FaceExtractor.
        
        Args:
            config: Configuration dictionary with face detection parameters
        """
        self.config = config
        face_config = config.get('face_detection', {})
        self.detector_type = face_config.get('detector', 'retinaface')
        self.confidence_threshold = face_config.get('confidence_threshold', 0.9)
        self.min_face_size = face_config.get('min_face_size', 80)
        self.alignment = face_config.get('alignment', True)
        
        self.detector = None
        self.retinaface_available = RETINAFACE_AVAILABLE
        self.mtcnn_available = MTCNN_AVAILABLE
        
        self._load_detector()
        
        logger.info(f"FaceExtractor initialized with detector={self.detector_type}, "
                   f"confidence={self.confidence_threshold}, min_size={self.min_face_size}")
    
    def _load_detector(self):
        """Load the face detection model."""
        try:
            if self.detector_type == 'retinaface' and RETINAFACE_AVAILABLE:
                # RetinaFace doesn't need explicit initialization
                self.detector = 'retinaface'
                logger.info("RetinaFace detector loaded successfully")
            elif self.detector_type == 'mtcnn' and MTCNN_AVAILABLE:
                self.detector = MTCNN()
                logger.info("MTCNN detector loaded successfully")
            elif MTCNN_AVAILABLE:
                # Fallback to MTCNN if RetinaFace not available
                logger.warning(f"Requested detector '{self.detector_type}' not available, falling back to MTCNN")
                self.detector = MTCNN()
                self.detector_type = 'mtcnn'
            elif RETINAFACE_AVAILABLE:
                # Fallback to RetinaFace if MTCNN not available
                logger.warning(f"Requested detector '{self.detector_type}' not available, falling back to RetinaFace")
                self.detector = 'retinaface'
                self.detector_type = 'retinaface'
            else:
                logger.warning("No face detection library available, using OpenCV Haar Cascade fallback")
                self.detector = None
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
            self.detector = None
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        if self.detector is None:
            # Fallback to OpenCV Haar Cascade
            return self._detect_faces_opencv(frame)
        
        # Use actual detector implementation
        if self.detector_type == 'retinaface':
            return self._detect_faces_retinaface(frame)
        elif self.detector_type == 'mtcnn':
            return self._detect_faces_mtcnn(frame)
        
        return []
    
    def _detect_faces_opencv(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Fallback face detection using OpenCV Haar Cascade.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            detections = []
            for (x, y, w, h) in faces:
                detection = FaceDetection(
                    box=[int(x), int(y), int(x + w), int(y + h)],
                    confidence=0.95,  # OpenCV doesn't provide confidence
                    landmarks=None
                )
                detections.append(detection)
            
            logger.debug(f"OpenCV detected {len(detections)} faces")
            return detections
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return []
    
    def _detect_faces_retinaface(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces using RetinaFace.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        try:
            # RetinaFace expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = RetinaFace.detect_faces(rgb_frame)
            
            detections = []
            if isinstance(faces, dict):
                for key, face_data in faces.items():
                    # Extract bounding box
                    facial_area = face_data.get('facial_area', [])
                    if len(facial_area) == 4:
                        x1, y1, x2, y2 = facial_area
                        
                        # Extract landmarks
                        landmarks_data = face_data.get('landmarks', {})
                        landmarks = {
                            'left_eye': tuple(landmarks_data.get('left_eye', (0, 0))),
                            'right_eye': tuple(landmarks_data.get('right_eye', (0, 0))),
                            'nose': tuple(landmarks_data.get('nose', (0, 0))),
                            'mouth_left': tuple(landmarks_data.get('mouth_left', (0, 0))),
                            'mouth_right': tuple(landmarks_data.get('mouth_right', (0, 0)))
                        } if landmarks_data else None
                        
                        detection = FaceDetection(
                            box=[int(x1), int(y1), int(x2), int(y2)],
                            confidence=float(face_data.get('score', 0.99)),
                            landmarks=landmarks
                        )
                        detections.append(detection)
            
            logger.debug(f"RetinaFace detected {len(detections)} faces")
            return detections
        except Exception as e:
            logger.error(f"RetinaFace detection failed: {e}")
            return []
    
    def _detect_faces_mtcnn(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces using MTCNN.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        try:
            # MTCNN expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.detector.detect_faces(rgb_frame)
            
            detections = []
            for face in faces:
                # Extract bounding box
                x, y, w, h = face['box']
                
                # Extract landmarks
                keypoints = face.get('keypoints', {})
                landmarks = {
                    'left_eye': tuple(keypoints.get('left_eye', (0, 0))),
                    'right_eye': tuple(keypoints.get('right_eye', (0, 0))),
                    'nose': tuple(keypoints.get('nose', (0, 0))),
                    'mouth_left': tuple(keypoints.get('mouth_left', (0, 0))),
                    'mouth_right': tuple(keypoints.get('mouth_right', (0, 0)))
                } if keypoints else None
                
                detection = FaceDetection(
                    box=[int(x), int(y), int(x + w), int(y + h)],
                    confidence=float(face.get('confidence', 0.99)),
                    landmarks=landmarks
                )
                detections.append(detection)
            
            logger.debug(f"MTCNN detected {len(detections)} faces")
            return detections
        except Exception as e:
            logger.error(f"MTCNN detection failed: {e}")
            return []
    
    def align_face(self, frame: np.ndarray, landmarks: Optional[Dict[str, Tuple[int, int]]], 
                   output_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Align face using facial landmarks.
        
        Args:
            frame: Input frame containing the face
            landmarks: Facial landmarks dictionary with keys: left_eye, right_eye, nose, mouth_left, mouth_right
            output_size: Desired output size
            
        Returns:
            Aligned face image
        """
        if landmarks is None or not isinstance(landmarks, dict):
            # Return center crop if no landmarks
            h, w = frame.shape[:2]
            size = min(h, w)
            start_h = (h - size) // 2
            start_w = (w - size) // 2
            crop = frame[start_h:start_h + size, start_w:start_w + size]
            return cv2.resize(crop, output_size)
        
        # Convert landmarks dict to array
        try:
            landmarks_array = np.array([
                landmarks.get('left_eye', (0, 0)),
                landmarks.get('right_eye', (0, 0)),
                landmarks.get('nose', (0, 0)),
                landmarks.get('mouth_left', (0, 0)),
                landmarks.get('mouth_right', (0, 0))
            ], dtype=np.float32)
            
            # Check if landmarks are valid
            if np.all(landmarks_array == 0):
                # No valid landmarks, return center crop
                h, w = frame.shape[:2]
                size = min(h, w)
                start_h = (h - size) // 2
                start_w = (w - size) // 2
                crop = frame[start_h:start_h + size, start_w:start_w + size]
                return cv2.resize(crop, output_size)
            
        except Exception as e:
            logger.warning(f"Failed to process landmarks: {e}")
            h, w = frame.shape[:2]
            size = min(h, w)
            start_h = (h - size) // 2
            start_w = (w - size) // 2
            crop = frame[start_h:start_h + size, start_w:start_w + size]
            return cv2.resize(crop, output_size)
        
        # Standard face template (normalized coordinates)
        template = np.array([
            [0.34, 0.46],  # left eye
            [0.66, 0.46],  # right eye
            [0.50, 0.64],  # nose
            [0.37, 0.82],  # left mouth
            [0.63, 0.82]   # right mouth
        ], dtype=np.float32)
        
        template[:, 0] *= output_size[0]
        template[:, 1] *= output_size[1]
        
        # Compute similarity transform
        try:
            tform = cv2.estimateAffinePartial2D(landmarks_array, template)[0]
            
            if tform is None:
                raise ValueError("Failed to estimate transform")
            
            # Apply transformation
            aligned = cv2.warpAffine(frame, tform, output_size, flags=cv2.INTER_LINEAR)
            
            return aligned
        except Exception as e:
            logger.warning(f"Face alignment failed: {e}, returning resized crop")
            h, w = frame.shape[:2]
            size = min(h, w)
            start_h = (h - size) // 2
            start_w = (w - size) // 2
            crop = frame[start_h:start_h + size, start_w:start_w + size]
            return cv2.resize(crop, output_size)
    
    def extract_faces(self, frame: np.ndarray, 
                     align: Optional[bool] = None) -> List[Tuple[np.ndarray, FaceDetection]]:
        """
        Extract all faces from a frame.
        
        Args:
            frame: Input frame (BGR format)
            align: Whether to align faces (overrides config if specified)
            
        Returns:
            List of tuples (face_image, FaceDetection)
        """
        detections = self.detect_faces(frame)
        
        if align is None:
            align = self.alignment
        
        faces = []
        for det in detections:
            if det.confidence < self.confidence_threshold:
                continue
            
            x1, y1, x2, y2 = det.bbox
            
            # Check minimum size
            if det.width < self.min_face_size or det.height < self.min_face_size:
                continue
            
            # Ensure bounding box is within frame
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract face region
            face_roi = frame[y1:y2, x1:x2]
            
            # Align if enabled and landmarks available
            if align and det.landmarks is not None:
                try:
                    face_img = self.align_face(frame, det.landmarks)
                except Exception as e:
                    logger.warning(f"Face alignment failed, using resize: {e}")
                    face_img = cv2.resize(face_roi, (224, 224))
            else:
                face_img = cv2.resize(face_roi, (224, 224))
            
            faces.append((face_img, det))
        
        logger.debug(f"Extracted {len(faces)} faces from frame")
        return faces
    
    def extract_faces_batch(self, frames: List[np.ndarray]) -> List[List[Tuple[np.ndarray, FaceDetection]]]:
        """
        Extract faces from a batch of frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of face extractions for each frame
        """
        batch_results = []
        for frame in frames:
            faces = self.extract_faces(frame)
            batch_results.append(faces)
        
        return batch_results
    
    def get_largest_face(self, frame: np.ndarray, 
                        align: Optional[bool] = None) -> Optional[Tuple[np.ndarray, FaceDetection]]:
        """
        Extract only the largest face from a frame.
        
        Args:
            frame: Input frame (BGR format)
            align: Whether to align face
            
        Returns:
            Tuple of (face_image, FaceDetection) or None if no face found
        """
        faces = self.extract_faces(frame, align=align)
        
        if not faces:
            return None
        
        # Find largest face by area
        largest_face = max(faces, key=lambda x: x[1].area)
        
        return largest_face
    
    def visualize_detections(self, frame: np.ndarray, 
                            detections: List[FaceDetection],
                            draw_landmarks: bool = True) -> np.ndarray:
        """
        Visualize face detections on frame.
        
        Args:
            frame: Input frame
            detections: List of face detections
            draw_landmarks: Whether to draw facial landmarks
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        for det in detections:
            # Draw bounding box
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0) if det.confidence >= self.confidence_threshold else (0, 165, 255)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            label = f"{det.confidence:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw landmarks
            if draw_landmarks and det.landmarks:
                for landmark_name, (lx, ly) in det.landmarks.items():
                    if lx > 0 and ly > 0:  # Valid landmark
                        cv2.circle(vis_frame, (int(lx), int(ly)), 3, (255, 0, 0), -1)
        
        return vis_frame
