"""
Test suite for AegisVerity Engine.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.verity_engine import VerityEngine


class TestVerityEngine(unittest.TestCase):
    """Test cases for VerityEngine."""
    
    @patch('src.engine.verity_engine.VideoLoader')
    @patch('src.engine.verity_engine.FaceExtractor')
    @patch('src.engine.verity_engine.AudioProcessor')
    def setUp(self, mock_audio, mock_face, mock_video):
        """Set up test fixtures."""
        # Create a minimal config for testing
        self.test_config = {
            'video': {
                'sampling_rate': 30,
                'frame_size': [224, 224],
                'adaptive_sampling': True
            },
            'face_detection': {
                'detector': 'retinaface',
                'confidence_threshold': 0.9,
                'min_face_size': 80,
                'alignment': True
            },
            'audio': {
                'sampling_rate': 16000,
                'n_mfcc': 40,
                'hop_length': 512,
                'n_fft': 2048
            },
            'performance': {
                'device': 'cpu',
                'batch_size': 16,
                'num_workers': 4
            },
            'thresholds': {
                'fake_probability': 0.5,
                'confidence_low': 0.3,
                'confidence_high': 0.7
            },
            'output': {
                'save_frames': False,
                'generate_report': True,
                'report_format': 'json'
            }
        }
        
        # Mock the config loading
        with patch.object(VerityEngine, '_load_config', return_value=self.test_config):
            with patch.object(VerityEngine, '_setup_logging'):
                self.engine = VerityEngine()
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine.config)
        self.assertFalse(self.engine.models_loaded)
    
    def test_compute_final_verdict_majority_fake(self):
        """Test final verdict computation with majority fake predictions."""
        results = {
            'spatial_results': [
                {'fake_probability': 0.8, 'confidence': 0.9},
                {'fake_probability': 0.7, 'confidence': 0.85}
            ],
            'temporal_results': {
                'prediction': 1,
                'confidence': 0.8
            },
            'av_sync_results': {
                'prediction': 1,
                'confidence': 0.75
            }
        }
        
        verdict, confidence = self.engine._compute_final_verdict(results)
        
        self.assertEqual(verdict, 1)  # Should predict fake
        self.assertGreater(confidence, 0.0)
    
    def test_compute_final_verdict_majority_real(self):
        """Test final verdict computation with majority real predictions."""
        results = {
            'spatial_results': [
                {'fake_probability': 0.2, 'confidence': 0.9},
                {'fake_probability': 0.3, 'confidence': 0.85}
            ],
            'temporal_results': {
                'prediction': 0,
                'confidence': 0.8
            },
            'av_sync_results': {
                'prediction': 0,
                'confidence': 0.75
            }
        }
        
        verdict, confidence = self.engine._compute_final_verdict(results)
        
        self.assertEqual(verdict, 0)  # Should predict real
        self.assertGreater(confidence, 0.0)
    
    def test_preprocess_face(self):
        """Test face preprocessing."""
        import numpy as np
        
        face_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        face_tensor = self.engine._preprocess_face(face_img)
        
        self.assertEqual(face_tensor.shape, (1, 3, 224, 224))
        self.assertGreaterEqual(face_tensor.min().item(), 0.0)
        self.assertLessEqual(face_tensor.max().item(), 1.0)
    
    def test_preprocess_audio(self):
        """Test audio preprocessing."""
        import numpy as np
        
        mfcc = np.random.randn(40, 100)
        audio_tensor = self.engine._preprocess_audio(mfcc)
        
        self.assertEqual(audio_tensor.ndim, 3)
        self.assertEqual(audio_tensor.shape[0], 1)


class TestVerityEngineIntegration(unittest.TestCase):
    """Integration tests for VerityEngine."""
    
    def test_engine_workflow(self):
        """Test complete engine workflow."""
        # This would require actual video files and trained models
        # Placeholder for integration test structure
        pass


if __name__ == '__main__':
    unittest.main()
