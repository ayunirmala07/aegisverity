"""
Test suite for AegisVerity models.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.spatial_cnn import SpatialCNN, AttentionSpatialCNN
from src.models.temporal_model import TemporalModel, C3D
from src.models.av_sync_model import AVSyncModel, CrossModalAttention


class TestSpatialCNN(unittest.TestCase):
    """Test cases for SpatialCNN."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SpatialCNN(backbone='efficientnet-b4', num_classes=2, pretrained=False)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.num_classes, 2)
        self.assertEqual(self.model.backbone_name, 'efficientnet-b4')
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        logits, features = self.model(input_tensor)
        
        self.assertEqual(logits.shape, (batch_size, 2))
        self.assertIsNotNone(features)
    
    def test_predict(self):
        """Test prediction."""
        input_tensor = torch.randn(1, 3, 224, 224)
        
        probs = self.model.predict(input_tensor)
        
        self.assertEqual(probs.shape, (1, 2))
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)


class TestTemporalModel(unittest.TestCase):
    """Test cases for TemporalModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TemporalModel(
            model_type='convlstm',
            input_dim=512,
            hidden_dim=256,
            num_classes=2
        )
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_type, 'convlstm')
        self.assertEqual(self.model.input_dim, 512)
        self.assertEqual(self.model.hidden_dim, 256)
    
    def test_forward_pass_convlstm(self):
        """Test ConvLSTM forward pass."""
        batch_size = 2
        seq_length = 16
        input_tensor = torch.randn(batch_size, seq_length, 512)
        
        logits, features = self.model(input_tensor)
        
        self.assertEqual(logits.shape, (batch_size, 2))
        self.assertIsNotNone(features)


class TestAVSyncModel(unittest.TestCase):
    """Test cases for AVSyncModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = AVSyncModel(
            visual_dim=512,
            audio_dim=128,
            hidden_dim=256,
            num_classes=2
        )
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.visual_dim, 512)
        self.assertEqual(self.model.audio_dim, 128)
        self.assertEqual(self.model.hidden_dim, 256)
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        visual_features = torch.randn(batch_size, 512)
        audio_features = torch.randn(batch_size, 128)
        
        logits, sync_score, fused = self.model(visual_features, audio_features)
        
        self.assertEqual(logits.shape, (batch_size, 2))
        self.assertEqual(sync_score.shape, (batch_size, 1))
        self.assertIsNotNone(fused)
    
    def test_predict(self):
        """Test prediction."""
        visual_features = torch.randn(1, 512)
        audio_features = torch.randn(1, 128)
        
        probs, sync_score = self.model.predict(visual_features, audio_features)
        
        self.assertEqual(probs.shape, (1, 2))
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)


class TestC3D(unittest.TestCase):
    """Test cases for C3D model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = C3D(num_classes=2)
    
    def test_forward_pass(self):
        """Test C3D forward pass."""
        batch_size = 1
        # Input: (B, C, T, H, W) - expects 16 frames of 224x224
        input_tensor = torch.randn(batch_size, 3, 16, 224, 224)
        
        logits = self.model(input_tensor)
        
        self.assertEqual(logits.shape, (batch_size, 2))


if __name__ == '__main__':
    unittest.main()
