"""
Test suite for AegisVerity face extractor module.
"""

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.face_extractor import FaceExtractor


class TestFaceExtractor(unittest.TestCase):
    """Test cases for FaceExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'face_detection': {
                'detector': 'retinaface',
                'confidence_threshold': 0.9,
                'min_face_size': 80,
                'alignment': True
            }
        }
        self.extractor = FaceExtractor(self.config)
    
    def test_initialization(self):
        """Test FaceExtractor initialization."""
        self.assertEqual(self.extractor.detector_type, 'retinaface')
        self.assertEqual(self.extractor.confidence_threshold, 0.9)
        self.assertEqual(self.extractor.min_face_size, 80)
        self.assertTrue(self.extractor.alignment)
    
    def test_detect_faces_opencv(self):
        """Test OpenCV face detection fallback."""
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detections = self.extractor._detect_faces_opencv(test_image)
        
        self.assertIsInstance(detections, list)
    
    def test_align_face(self):
        """Test face alignment."""
        # Create a test face image
        face_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Create dummy landmarks
        landmarks = np.array([
            [80, 100],   # left eye
            [120, 100],  # right eye
            [100, 130],  # nose
            [85, 160],   # left mouth
            [115, 160]   # right mouth
        ], dtype=np.float32)
        
        aligned = self.extractor.align_face(face_image, landmarks, output_size=(224, 224))
        
        self.assertEqual(aligned.shape, (224, 224, 3))
    
    def test_extract_faces(self):
        """Test face extraction from frame."""
        # Create a test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        faces = self.extractor.extract_faces(test_frame)
        
        self.assertIsInstance(faces, list)


if __name__ == '__main__':
    unittest.main()
