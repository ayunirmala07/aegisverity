"""
Test suite for AegisVerity video loader module.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.video_loader import VideoLoader


class TestVideoLoader(unittest.TestCase):
    """Test cases for VideoLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'video': {
                'sampling_rate': 30,
                'frame_size': [224, 224],
                'adaptive_sampling': True
            }
        }
        self.loader = VideoLoader(self.config)
    
    def test_initialization(self):
        """Test VideoLoader initialization."""
        self.assertEqual(self.loader.sampling_rate, 30)
        self.assertEqual(self.loader.frame_size, (224, 224))
        self.assertTrue(self.loader.adaptive_sampling)
    
    def test_compute_motion_score(self):
        """Test motion score computation."""
        # Create two similar frames
        frame1 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        frame2 = frame1.copy()
        
        motion_score = self.loader.compute_motion_score(frame1, frame2)
        
        # Motion should be low for identical frames
        self.assertIsInstance(motion_score, float)
        self.assertGreaterEqual(motion_score, 0.0)
    
    def test_get_video_metadata(self):
        """Test video metadata extraction."""
        # This would require an actual video file
        # Placeholder test
        metadata = self.loader.get_video_metadata('nonexistent.mp4')
        self.assertIsInstance(metadata, dict)


class TestVideoLoaderIntegration(unittest.TestCase):
    """Integration tests for VideoLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'video': {
                'sampling_rate': 1,
                'frame_size': [224, 224],
                'adaptive_sampling': False
            }
        }
        self.loader = VideoLoader(self.config)
    
    def test_frame_extraction_workflow(self):
        """Test complete frame extraction workflow."""
        # This would require an actual video file
        # Placeholder test structure
        pass


if __name__ == '__main__':
    unittest.main()
