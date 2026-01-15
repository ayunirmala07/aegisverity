# AegisVerity

AegisVerity adalah sistem **forensik deteksi deepfake** berlapis tinggi yang dirancang dengan pendekatan multi‑modal (visual, temporal, audio‑visual). Sistem ini mengutamakan **robustness, accuracy, dan explainability** untuk menghadapi ancaman manipulasi media modern.

---

## 🚀 Arsitektur Sistem

### 1. Data Pipeline
- **VideoLoader**  
  - Ingest video via `ffmpeg-python` (primary) atau OpenCV (fallback).  
  - Adaptive frame sampling: fixed stride + event-driven (prioritas frame dengan wajah).  
- **FaceExtractor**  
  - Deteksi wajah dengan **RetinaFace** (MTCNN fallback).  
  - Landmark alignment & normalisasi ke ukuran target.  
- **AudioProcessor**  
  - Ekstraksi audio → MFCC dengan `librosa`.  
  - Konfigurasi sample rate, hop length, FFT size.

### 2. Model Tri‑Shield
- **SpatialCNN**  
  - Backbone: EfficientNet‑B4 atau Xception.  
  - Deteksi artefak visual (noise residuals, blending boundaries).  
- **TemporalModel**  
  - Backbone: 3D‑CNN atau TimeSformer.  
  - Deteksi inkonsistensi temporal (flicker, warping).  
- **AVSyncModel**  
  - Audio encoder (MFCC → 1D‑CNN + GRU).  
  - Visual encoder (lip‑region → 2D‑CNN + GRU).  
  - Fusion untuk mendeteksi mismatch phoneme‑viseme.

### 3. Inference & Explainability
- **ModelInference**  
  - Wrapper unified untuk Spatial, Temporal, AVSync.  
  - Menggabungkan confidence score → verdict (`real` / `fake`).  
- **ExplainabilityEngine**  
  - Grad‑CAM / Attention Map untuk heatmap manipulasi.  
  - Artefak visual disimpan sebagai bukti forensik.

### 4. Defense Layer
- **AdversarialChecks**  
  - Noise variance, blur detection, compression blockiness.  
  - Histogram drift, temporal jitter, FFT energy spikes.  
  - Menghasilkan sinyal heuristik untuk memperingatkan serangan adversarial.

### 5. Orchestrator
- **VerityEngine**  
  - Menyatukan pipeline, inference, explainability, defense.  
  - Output: `VerityResult` berisi metadata, domain scores, aggregate verdict, signals, artifacts.  
  - Menyimpan laporan JSON + heatmap Grad‑CAM.

---

## 📦 Instalasi

```bash
git clone https://github.com/your-org/aegisverity.git
cd aegisverity

# Buat virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
