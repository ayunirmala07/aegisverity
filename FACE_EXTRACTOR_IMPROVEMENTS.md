# FaceExtractor Module Improvements

## Overview
The `face_extractor.py` module has been significantly enhanced with proper library handling, improved data structures, and robust error handling.

## Key Improvements

### 1. **Proper Import Handling**
```python
# RetinaFace with availability flag
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
```

**Benefits:**
- Graceful degradation when libraries are missing
- Clear warning messages via loguru
- No runtime crashes from missing dependencies

### 2. **FaceDetection Data Class**
```python
class FaceDetection:
    """Container for face detection results."""
    
    def __init__(self, box: List[int], confidence: float, 
                 landmarks: Optional[Dict[str, Tuple[int, int]]] = None):
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
```

**Benefits:**
- Type-safe face detection results
- Convenient property accessors
- Easy conversion to dictionary
- Self-documenting API

### 3. **Enhanced Detector Loading**
```python
def _load_detector(self):
    """Load the face detection model."""
    try:
        if self.detector_type == 'retinaface' and RETINAFACE_AVAILABLE:
            self.detector = 'retinaface'
            logger.info("RetinaFace detector loaded successfully")
        elif self.detector_type == 'mtcnn' and MTCNN_AVAILABLE:
            self.detector = MTCNN()
            logger.info("MTCNN detector loaded successfully")
        elif MTCNN_AVAILABLE:
            logger.warning(f"Requested detector '{self.detector_type}' not available, falling back to MTCNN")
            self.detector = MTCNN()
            self.detector_type = 'mtcnn'
        elif RETINAFACE_AVAILABLE:
            logger.warning(f"Requested detector '{self.detector_type}' not available, falling back to RetinaFace")
            self.detector = 'retinaface'
            self.detector_type = 'retinaface'
        else:
            logger.warning("No face detection library available, using OpenCV Haar Cascade fallback")
            self.detector = None
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        self.detector = None
```

**Benefits:**
- Intelligent fallback mechanism
- Clear logging of detector selection
- No crashes from missing libraries

### 4. **Complete RetinaFace Implementation**
```python
def _detect_faces_retinaface(self, frame: np.ndarray) -> List[FaceDetection]:
    """Detect faces using RetinaFace."""
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
        
        return detections
    except Exception as e:
        logger.error(f"RetinaFace detection failed: {e}")
        return []
```

**Benefits:**
- Full RetinaFace integration
- Proper color space conversion (BGR → RGB)
- Landmark extraction
- Comprehensive error handling

### 5. **Complete MTCNN Implementation**
```python
def _detect_faces_mtcnn(self, frame: np.ndarray) -> List[FaceDetection]:
    """Detect faces using MTCNN."""
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
        
        return detections
    except Exception as e:
        logger.error(f"MTCNN detection failed: {e}")
        return []
```

**Benefits:**
- Full MTCNN integration
- Keypoint extraction
- Proper bounding box conversion

### 6. **Improved Face Alignment**
```python
def align_face(self, frame: np.ndarray, landmarks: Optional[Dict[str, Tuple[int, int]]], 
               output_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Align face using facial landmarks."""
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
        # Fallback to center crop
        ...
```

**Benefits:**
- Works with dictionary-based landmarks
- Robust error handling
- Fallback to center crop when alignment fails

### 7. **New Helper Methods**

#### Get Largest Face
```python
def get_largest_face(self, frame: np.ndarray, 
                    align: Optional[bool] = None) -> Optional[Tuple[np.ndarray, FaceDetection]]:
    """Extract only the largest face from a frame."""
    faces = self.extract_faces(frame, align=align)
    
    if not faces:
        return None
    
    # Find largest face by area
    largest_face = max(faces, key=lambda x: x[1].area)
    
    return largest_face
```

#### Visualize Detections
```python
def visualize_detections(self, frame: np.ndarray, 
                        detections: List[FaceDetection],
                        draw_landmarks: bool = True) -> np.ndarray:
    """Visualize face detections on frame."""
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
                if lx > 0 and ly > 0:
                    cv2.circle(vis_frame, (int(lx), int(ly)), 3, (255, 0, 0), -1)
    
    return vis_frame
```

### 8. **Enhanced Error Handling**
- Boundary checking for bounding boxes
- Validation of landmarks
- Graceful fallbacks at every step
- Comprehensive try-except blocks

### 9. **Loguru Integration**
```python
from loguru import logger

# Usage throughout the code
logger.info("RetinaFace detector loaded successfully")
logger.warning("MTCNN not available. Face detection may fail.")
logger.error(f"RetinaFace detection failed: {e}")
logger.debug(f"RetinaFace detected {len(detections)} faces")
```

**Benefits:**
- Better logging with context
- Color-coded log levels
- Easy debugging

## Migration Guide

### Old Code
```python
faces = extractor.extract_faces(frame)
for face_img, detection_dict in faces:
    box = detection_dict['box']
    confidence = detection_dict['confidence']
```

### New Code
```python
faces = extractor.extract_faces(frame)
for face_img, detection in faces:
    box = detection.bbox  # or detection.box
    confidence = detection.confidence
    width = detection.width
    area = detection.area
```

## Testing

Run the comprehensive test suite:
```bash
python test_face_extractor_improved.py
```

Tests cover:
1. Initialization with different configurations
2. FaceDetection data class
3. Face detection on images
4. Face extraction pipeline
5. Largest face extraction
6. Batch processing
7. Visualization
8. Error handling

## Dependencies

Updated `requirements.txt` includes:
```
retinaface-pytorch==0.0.7
mtcnn==0.1.1
loguru==0.7.2
opencv-python==4.8.1.78
```

## Summary

The improved FaceExtractor module is now:
- ✅ Production-ready with full RetinaFace/MTCNN support
- ✅ Robust with comprehensive error handling
- ✅ Type-safe with FaceDetection class
- ✅ Well-documented with loguru logging
- ✅ Feature-rich with visualization and batch support
- ✅ Maintainable with clean, organized code
