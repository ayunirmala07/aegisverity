"""
AegisVerity Demo Script

Demonstrates basic usage of the AegisVerity deepfake detection framework.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.engine.verity_engine import VerityEngine
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_single_video():
    """Demonstrate processing a single video."""
    logger.info("=" * 80)
    logger.info("AegisVerity Demo - Single Video Processing")
    logger.info("=" * 80)
    
    # Initialize engine
    logger.info("Initializing AegisVerity Engine...")
    engine = VerityEngine('config/settings.yaml')
    
    # Load models (in production, provide actual checkpoint paths)
    logger.info("Loading detection models...")
    try:
        engine.load_models()
        logger.info("Models loaded successfully (using default/random weights for demo)")
    except Exception as e:
        logger.warning(f"Could not load model checkpoints: {e}")
        logger.info("Continuing with uninitialized models for demo purposes...")
    
    # Example video path (replace with actual video)
    video_path = "data/samples/example_video.mp4"
    
    if not Path(video_path).exists():
        logger.warning(f"Video not found: {video_path}")
        logger.info("Please place a video file at: data/samples/example_video.mp4")
        logger.info("Supported formats: .mp4, .avi, .mov, .mkv")
        return
    
    # Process video
    logger.info(f"Processing video: {video_path}")
    results = engine.process_video(video_path, output_dir='data/outputs')
    
    # Display results
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Video: {results['video_path']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Frames Analyzed: {results['frames_analyzed']}")
    print("-" * 80)
    print(f"VERDICT: {results['verdict_text']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print("-" * 80)
    
    if results.get('warnings'):
        print("Warnings:")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    print("=" * 80)
    logger.info("Demo completed successfully!")


def demo_batch_processing():
    """Demonstrate batch processing of multiple videos."""
    logger.info("=" * 80)
    logger.info("AegisVerity Demo - Batch Processing")
    logger.info("=" * 80)
    
    # Initialize engine
    engine = VerityEngine('config/settings.yaml')
    engine.load_models()
    
    # Example video paths
    video_paths = [
        "data/samples/video1.mp4",
        "data/samples/video2.mp4",
        "data/samples/video3.mp4"
    ]
    
    # Filter existing videos
    existing_videos = [v for v in video_paths if Path(v).exists()]
    
    if not existing_videos:
        logger.warning("No videos found in data/samples/")
        logger.info("Please add video files to data/samples/ directory")
        return
    
    # Process batch
    logger.info(f"Processing {len(existing_videos)} videos...")
    results = engine.process_batch(existing_videos, output_dir='data/outputs')
    
    # Display summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING RESULTS")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\nVideo {i}: {Path(result['video_path']).name}")
        print(f"  Verdict: {result['verdict_text']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Frames: {result['frames_analyzed']}")
    
    print("=" * 80)
    logger.info("Batch demo completed!")


def demo_model_components():
    """Demonstrate individual model components."""
    logger.info("=" * 80)
    logger.info("AegisVerity Demo - Model Components")
    logger.info("=" * 80)
    
    import torch
    from src.models.spatial_cnn import SpatialCNN
    from src.models.temporal_model import TemporalModel
    from src.models.av_sync_model import AVSyncModel
    
    # Spatial CNN
    logger.info("\n1. Testing Spatial CNN (EfficientNet-B4)...")
    spatial_model = SpatialCNN(backbone='efficientnet-b4', num_classes=2, pretrained=False)
    dummy_frame = torch.randn(1, 3, 224, 224)
    logits, features = spatial_model(dummy_frame)
    logger.info(f"   Input shape: {dummy_frame.shape}")
    logger.info(f"   Output logits shape: {logits.shape}")
    logger.info(f"   ✓ Spatial CNN working correctly")
    
    # Temporal Model
    logger.info("\n2. Testing Temporal Model (ConvLSTM)...")
    temporal_model = TemporalModel(model_type='convlstm', input_dim=512, hidden_dim=256)
    dummy_sequence = torch.randn(1, 16, 512)  # Batch, Time, Features
    logits, features = temporal_model(dummy_sequence)
    logger.info(f"   Input shape: {dummy_sequence.shape}")
    logger.info(f"   Output logits shape: {logits.shape}")
    logger.info(f"   ✓ Temporal Model working correctly")
    
    # AV Sync Model
    logger.info("\n3. Testing AV Sync Model...")
    av_model = AVSyncModel(visual_dim=512, audio_dim=128, hidden_dim=256)
    dummy_visual = torch.randn(1, 512)
    dummy_audio = torch.randn(1, 128)
    logits, sync_score, fused = av_model(dummy_visual, dummy_audio)
    logger.info(f"   Visual input: {dummy_visual.shape}")
    logger.info(f"   Audio input: {dummy_audio.shape}")
    logger.info(f"   Output logits: {logits.shape}")
    logger.info(f"   Sync score: {sync_score.shape}")
    logger.info(f"   ✓ AV Sync Model working correctly")
    
    print("\n" + "=" * 80)
    print("All model components tested successfully!")
    print("=" * 80)


def demo_pipeline_components():
    """Demonstrate pipeline components."""
    logger.info("=" * 80)
    logger.info("AegisVerity Demo - Pipeline Components")
    logger.info("=" * 80)
    
    import numpy as np
    from src.pipeline.video_loader import VideoLoader
    from src.pipeline.face_extractor import FaceExtractor
    from src.pipeline.audio_processor import AudioProcessor
    
    config = {
        'video': {'sampling_rate': 30, 'frame_size': [224, 224], 'adaptive_sampling': True},
        'face_detection': {'detector': 'retinaface', 'confidence_threshold': 0.9, 'min_face_size': 80, 'alignment': True},
        'audio': {'sampling_rate': 16000, 'n_mfcc': 40, 'hop_length': 512, 'n_fft': 2048}
    }
    
    # Video Loader
    logger.info("\n1. Testing Video Loader...")
    video_loader = VideoLoader(config)
    logger.info(f"   Sampling rate: {video_loader.sampling_rate} fps")
    logger.info(f"   Frame size: {video_loader.frame_size}")
    logger.info(f"   ✓ Video Loader initialized")
    
    # Face Extractor
    logger.info("\n2. Testing Face Extractor...")
    face_extractor = FaceExtractor(config)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    faces = face_extractor.extract_faces(test_frame)
    logger.info(f"   Detector: {face_extractor.detector_type}")
    logger.info(f"   Test frame shape: {test_frame.shape}")
    logger.info(f"   Faces detected: {len(faces)}")
    logger.info(f"   ✓ Face Extractor working")
    
    # Audio Processor
    logger.info("\n3. Testing Audio Processor...")
    audio_processor = AudioProcessor(config)
    logger.info(f"   Sampling rate: {audio_processor.sampling_rate} Hz")
    logger.info(f"   MFCC coefficients: {audio_processor.n_mfcc}")
    logger.info(f"   ✓ Audio Processor initialized")
    
    print("\n" + "=" * 80)
    print("All pipeline components tested successfully!")
    print("=" * 80)


def main():
    """Main demo function."""
    print("\n" + "=" * 80)
    print("        AEGISVERITY - DEEPFAKE DETECTION FRAMEWORK")
    print("=" * 80)
    print("\nAvailable Demos:")
    print("  1. Single Video Processing")
    print("  2. Batch Video Processing")
    print("  3. Test Model Components")
    print("  4. Test Pipeline Components")
    print("  0. Exit")
    
    while True:
        try:
            choice = input("\nSelect demo (0-4): ").strip()
            
            if choice == '0':
                print("\nExiting demo. Thank you!")
                break
            elif choice == '1':
                demo_single_video()
            elif choice == '2':
                demo_batch_processing()
            elif choice == '3':
                demo_model_components()
            elif choice == '4':
                demo_pipeline_components()
            else:
                print("Invalid choice. Please select 0-4.")
        except KeyboardInterrupt:
            print("\n\nDemo interrupted. Exiting...")
            break
        except Exception as e:
            logger.error(f"Demo error: {e}", exc_info=True)
            print(f"\nError occurred: {e}")
            print("Please check the logs for details.")


if __name__ == "__main__":
    main()
