# AegisVerity Setup Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aegisverity.git
cd aegisverity

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/settings.yaml` to customize:
- Video sampling rate and frame size
- Audio processing parameters
- Face detection settings
- Model paths
- Detection thresholds
- Output formats

### 3. Basic Usage

#### Python API

```python
from src.engine.verity_engine import VerityEngine

# Initialize engine
engine = VerityEngine('config/settings.yaml')

# Load models (optional: provide custom checkpoint paths)
engine.load_models(
    spatial_path='models/spatial_efficientnet_b4.pth',
    temporal_path='models/temporal_timesformer.pth',
    av_sync_path='models/av_sync_detector.pth'
)

# Process video
results = engine.process_video('data/samples/video.mp4', output_dir='data/outputs')

# Check results
print(f"Verdict: {results['verdict_text']}")
print(f"Confidence: {results['confidence']:.2%}")
```

#### Command Line

```bash
# Process single video
python src/engine/verity_engine.py video.mp4 --output data/outputs

# With custom models
python src/engine/verity_engine.py video.mp4 \
    --spatial-model models/spatial.pth \
    --temporal-model models/temporal.pth \
    --av-model models/av_sync.pth
```

### 4. Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=src tests/
```

## Project Structure

```
AegisVerity/
│
├── config/                    # Configuration files
│   ├── settings.yaml         # Global settings
│   └── logging.conf          # Logging configuration
│
├── data/                      # Data directories
│   ├── samples/              # Sample videos/audio
│   └── outputs/              # Analysis results
│
├── src/                       # Source code
│   ├── pipeline/             # Data ingestion
│   │   ├── video_loader.py   # Video processing
│   │   ├── face_extractor.py # Face detection
│   │   └── audio_processor.py # Audio features
│   │
│   ├── models/               # Detection models
│   │   ├── spatial_cnn.py    # Spatial CNN
│   │   ├── temporal_model.py # Temporal models
│   │   └── av_sync_model.py  # AV sync detection
│   │
│   ├── inference/            # Inference engine
│   │   ├── model_inference.py # Unified inference
│   │   └── explainability.py  # Grad-CAM, attention
│   │
│   ├── defense/              # Adversarial defense
│   │   └── adversarial_checks.py
│   │
│   └── engine/               # Main orchestrator
│       └── verity_engine.py  # VerityEngine class
│
├── tests/                     # Unit tests
│   ├── test_video_loader.py
│   ├── test_face_extractor.py
│   ├── test_models.py
│   └── test_engine.py
│
├── logs/                      # Runtime logs
│   └── aegisverity.log
│
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
└── README.md                 # Documentation
```

## Components

### Pipeline Layer

**VideoLoader**: Video ingestion with adaptive frame sampling
- FFmpeg integration for metadata and audio extraction
- OpenCV for frame processing
- Motion-based adaptive sampling

**FaceExtractor**: Face detection and alignment
- RetinaFace/MTCNN support
- Facial landmark detection
- Face alignment and normalization

**AudioProcessor**: Audio feature extraction
- MFCC extraction with Librosa
- Mel spectrogram computation
- Spectral feature analysis

### Model Layer

**SpatialCNN**: Frame-level analysis
- EfficientNet-B4 or Xception backbone
- Attention mechanisms (optional)
- Feature extraction for temporal models

**TemporalModel**: Sequential analysis
- ConvLSTM for temporal patterns
- 3D-CNN for spatiotemporal features
- TimeSformer for transformer-based analysis

**AVSyncModel**: Audio-visual synchronization
- Cross-modal attention
- Lip-sync detection
- Multimodal fusion

### Inference Layer

**ModelInference**: Unified inference wrapper
- Batch processing support
- Ensemble methods
- Confidence scoring

**ExplainabilityEngine**: Model interpretation
- Grad-CAM heatmaps
- Attention visualizations
- Feature importance analysis

### Defense Layer

**AdversarialDefense**: Robustness mechanisms
- Perturbation detection
- Input transformations (JPEG, median, Gaussian)
- Randomized smoothing
- Ensemble defense

### Engine Layer

**VerityEngine**: Main orchestrator
- End-to-end pipeline coordination
- Multi-format report generation
- Batch processing
- Comprehensive error handling

## Advanced Usage

### Custom Model Training

```python
from src.models.spatial_cnn import SpatialCNN
import torch

# Initialize model
model = SpatialCNN(backbone='efficientnet-b4', num_classes=2)

# Training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        images, labels = batch
        logits, _ = model(images)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save checkpoint
model.save_weights('models/spatial_custom.pth', epoch=epoch)
```

### Batch Processing

```python
# Process multiple videos
video_paths = [
    'data/samples/video1.mp4',
    'data/samples/video2.mp4',
    'data/samples/video3.mp4'
]

results = engine.process_batch(video_paths, output_dir='data/outputs')

# Analyze results
for result in results:
    print(f"{result['video_path']}: {result['verdict_text']}")
```

### Custom Configuration

```python
import yaml

# Load and modify config
with open('config/settings.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Customize settings
config['video']['sampling_rate'] = 60
config['thresholds']['fake_probability'] = 0.6

# Initialize with custom config
engine = VerityEngine()
engine.config = config
```

## Performance Optimization

### GPU Acceleration

Enable CUDA in `config/settings.yaml`:
```yaml
performance:
  device: "cuda"  # or "cpu", "mps" for Apple Silicon
  batch_size: 32
  num_workers: 8
```

### Memory Optimization

For large videos, adjust batch size and frame sampling:
```yaml
video:
  sampling_rate: 15  # Lower rate for memory efficiency
performance:
  batch_size: 8  # Reduce batch size
```

## Troubleshooting

### FFmpeg Not Found
```bash
# Windows (using Chocolatey)
choco install ffmpeg

# Linux
sudo apt-get install ffmpeg

# Mac
brew install ffmpeg
```

### CUDA Out of Memory
- Reduce batch size in config
- Lower video sampling rate
- Process shorter video segments

### Face Detection Issues
- Adjust `confidence_threshold` in config
- Try different detector: 'retinaface' vs 'mtcnn'
- Check minimum face size requirements

## Citation

If you use AegisVerity in your research, please cite:

```bibtex
@software{aegisverity2026,
  title={AegisVerity: Advanced Deepfake Detection Framework},
  author={AegisVerity Team},
  year={2026},
  url={https://github.com/yourusername/aegisverity}
}
```

## License

[Add your license here]

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
