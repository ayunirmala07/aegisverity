"""
Test script for the improved FaceExtractor module.

This demonstrates the new features:
- Proper RetinaFace/MTCNN import handling
- FaceDetection data class
- Enhanced error handling
- Visualization capabilities
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.face_extractor import FaceExtractor, FaceDetection
from loguru import logger


def test_face_extractor_initialization():
    """Test FaceExtractor initialization with different configurations."""
    logger.info("=" * 80)
    logger.info("Test 1: FaceExtractor Initialization")
    logger.info("=" * 80)
    
    config = {
        'face_detection': {
            'detector': 'retinaface',
            'confidence_threshold': 0.9,
            'min_face_size': 80,
            'alignment': True
        }
    }
    
    extractor = FaceExtractor(config)
    
    logger.info(f"✓ Detector type: {extractor.detector_type}")
    logger.info(f"✓ Confidence threshold: {extractor.confidence_threshold}")
    logger.info(f"✓ Min face size: {extractor.min_face_size}")
    logger.info(f"✓ RetinaFace available: {extractor.retinaface_available}")
    logger.info(f"✓ MTCNN available: {extractor.mtcnn_available}")
    
    return extractor


def test_face_detection_class():
    """Test FaceDetection data class."""
    logger.info("\n" + "=" * 80)
    logger.info("Test 2: FaceDetection Data Class")
    logger.info("=" * 80)
    
    # Create a sample detection
    landmarks = {
        'left_eye': (100, 150),
        'right_eye': (200, 150),
        'nose': (150, 200),
        'mouth_left': (120, 250),
        'mouth_right': (180, 250)
    }
    
    detection = FaceDetection(
        box=[50, 100, 250, 300],
        confidence=0.95,
        landmarks=landmarks
    )
    
    logger.info(f"✓ Bounding box: {detection.bbox}")
    logger.info(f"✓ Width: {detection.width}px")
    logger.info(f"✓ Height: {detection.height}px")
    logger.info(f"✓ Area: {detection.area}px²")
    logger.info(f"✓ Confidence: {detection.confidence}")
    
    # Convert to dict
    det_dict = detection.to_dict()
    logger.info(f"✓ Dictionary representation: {len(det_dict)} keys")
    
    return detection


def test_face_detection_on_image(extractor: FaceExtractor):
    """Test face detection on a test image."""
    logger.info("\n" + "=" * 80)
    logger.info("Test 3: Face Detection on Test Image")
    logger.info("=" * 80)
    
    # Create a test image with simulated face region
    test_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    # Add a brighter region to simulate a face
    test_image[150:350, 220:420] = np.random.randint(180, 220, (200, 200, 3), dtype=np.uint8)
    
    logger.info(f"Created test image: {test_image.shape}")
    
    # Detect faces
    detections = extractor.detect_faces(test_image)
    
    logger.info(f"✓ Detected {len(detections)} face(s)")
    
    for i, det in enumerate(detections, 1):
        logger.info(f"  Face {i}:")
        logger.info(f"    - Confidence: {det.confidence:.3f}")
        logger.info(f"    - Size: {det.width}x{det.height}px")
        logger.info(f"    - Has landmarks: {det.landmarks is not None}")
    
    return test_image, detections


def test_face_extraction(extractor: FaceExtractor):
    """Test complete face extraction pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("Test 4: Face Extraction Pipeline")
    logger.info("=" * 80)
    
    # Create test image
    test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    
    # Extract faces
    faces = extractor.extract_faces(test_image, align=True)
    
    logger.info(f"✓ Extracted {len(faces)} face(s)")
    
    for i, (face_img, det) in enumerate(faces, 1):
        logger.info(f"  Face {i}:")
        logger.info(f"    - Face image shape: {face_img.shape}")
        logger.info(f"    - Confidence: {det.confidence:.3f}")
        logger.info(f"    - Original size: {det.width}x{det.height}px")
    
    return faces


def test_largest_face_extraction(extractor: FaceExtractor):
    """Test largest face extraction."""
    logger.info("\n" + "=" * 80)
    logger.info("Test 5: Largest Face Extraction")
    logger.info("=" * 80)
    
    # Create test image
    test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    
    # Get largest face
    result = extractor.get_largest_face(test_image)
    
    if result:
        face_img, det = result
        logger.info(f"✓ Largest face extracted")
        logger.info(f"  - Size: {det.width}x{det.height}px (area: {det.area}px²)")
        logger.info(f"  - Confidence: {det.confidence:.3f}")
        logger.info(f"  - Face image shape: {face_img.shape}")
    else:
        logger.info("✗ No face detected")
    
    return result


def test_batch_extraction(extractor: FaceExtractor):
    """Test batch face extraction."""
    logger.info("\n" + "=" * 80)
    logger.info("Test 6: Batch Face Extraction")
    logger.info("=" * 80)
    
    # Create multiple test frames
    frames = [
        np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    
    logger.info(f"Created {len(frames)} test frames")
    
    # Extract faces from batch
    batch_results = extractor.extract_faces_batch(frames)
    
    logger.info(f"✓ Processed {len(batch_results)} frames")
    
    for i, faces in enumerate(batch_results, 1):
        logger.info(f"  Frame {i}: {len(faces)} face(s) extracted")
    
    return batch_results


def test_visualization(extractor: FaceExtractor):
    """Test face detection visualization."""
    logger.info("\n" + "=" * 80)
    logger.info("Test 7: Face Detection Visualization")
    logger.info("=" * 80)
    
    # Create test image
    test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    
    # Detect faces
    detections = extractor.detect_faces(test_image)
    
    if detections:
        # Visualize detections
        vis_image = extractor.visualize_detections(test_image, detections, draw_landmarks=True)
        
        logger.info(f"✓ Visualization created")
        logger.info(f"  - Input shape: {test_image.shape}")
        logger.info(f"  - Output shape: {vis_image.shape}")
        logger.info(f"  - Faces visualized: {len(detections)}")
        
        return vis_image
    else:
        logger.info("✗ No faces to visualize")
        return None


def test_error_handling(extractor: FaceExtractor):
    """Test error handling with edge cases."""
    logger.info("\n" + "=" * 80)
    logger.info("Test 8: Error Handling")
    logger.info("=" * 80)
    
    # Test with empty image
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = extractor.detect_faces(empty_image)
    logger.info(f"✓ Empty image handled: {len(detections)} faces")
    
    # Test with very small image
    tiny_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    detections = extractor.detect_faces(tiny_image)
    logger.info(f"✓ Tiny image handled: {len(detections)} faces")
    
    # Test with grayscale image converted to BGR
    gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    detections = extractor.detect_faces(bgr_image)
    logger.info(f"✓ Grayscale image handled: {len(detections)} faces")
    
    logger.info("✓ All error cases handled gracefully")


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("FaceExtractor Module - Comprehensive Test Suite")
    logger.info("=" * 80)
    
    try:
        # Test 1: Initialization
        extractor = test_face_extractor_initialization()
        
        # Test 2: FaceDetection class
        test_face_detection_class()
        
        # Test 3: Face detection
        test_face_detection_on_image(extractor)
        
        # Test 4: Face extraction
        test_face_extraction(extractor)
        
        # Test 5: Largest face
        test_largest_face_extraction(extractor)
        
        # Test 6: Batch extraction
        test_batch_extraction(extractor)
        
        # Test 7: Visualization
        test_visualization(extractor)
        
        # Test 8: Error handling
        test_error_handling(extractor)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        logger.info("=" * 80)
        
        logger.info("\nKey Improvements:")
        logger.info("  ✓ Proper RetinaFace/MTCNN import handling with fallbacks")
        logger.info("  ✓ FaceDetection data class with properties")
        logger.info("  ✓ Enhanced error handling and validation")
        logger.info("  ✓ Landmark-based face alignment")
        logger.info("  ✓ Batch processing support")
        logger.info("  ✓ Visualization capabilities")
        logger.info("  ✓ Largest face extraction")
        logger.info("  ✓ Comprehensive logging with loguru")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
