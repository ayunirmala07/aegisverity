"""
AVSyncModel for AegisVerity
---------------------------
Responsibilities:
- Detect audio-visual lip-sync mismatch (phoneme-viseme mismatch).
- Encode audio MFCC features and visual lip-region frames.
- Fuse embeddings and classify (real vs fake).

Architecture:
- Audio encoder: 1D-CNN + BiGRU over MFCC time axis.
- Visual encoder: 2D-CNN (lightweight) + BiGRU over frame sequence.
- Fusion: concatenation + MLP classifier.

Inputs:
- MFCC: numpy array (n_mfcc x frames)
- Mouth frames: list of BGR images (lip-region crops)

Compatibility:
- PyTorch 2.0.1 + CUDA 11.8
- loguru 0.7.2
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AVSyncConfig:
    """
    Configuration for AVSyncModel.
    """
    def __init__(
        self,
        num_classes: int = 2,
        mouth_size: Tuple[int, int] = (112, 112),
        n_mfcc: int = 20,
        audio_hidden: int = 128,
        visual_hidden: int = 128,
        fusion_hidden: int = 128,
        sequence_length: int = 16,
        use_cuda: bool = True,
        checkpoint_path: Optional[str] = None
    ):
        """
        Args:
            num_classes: Output classes (default binary).
            mouth_size: Resize for lip-region crops (WxH).
            n_mfcc: Expected MFCC coefficient count.
            audio_hidden: Hidden size for audio GRU.
            visual_hidden: Hidden size for visual GRU.
            fusion_hidden: Hidden size for fusion MLP.
            sequence_length: Number of mouth frames per clip.
            use_cuda: Use CUDA if available.
            checkpoint_path: Optional path to model weights.
        """
        self.num_classes = num_classes
        self.mouth_size = mouth_size
        self.n_mfcc = n_mfcc
        self.audio_hidden = audio_hidden
        self.visual_hidden = visual_hidden
        self.fusion_hidden = fusion_hidden
        self.sequence_length = max(4, sequence_length)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.checkpoint_path = checkpoint_path


# ---------------------------
# Audio encoder: 1D-CNN + BiGRU
# ---------------------------
class AudioEncoder(nn.Module):
    def __init__(self, n_mfcc: int, hidden: int):
        super().__init__()
        # Input shape: (B, n_mfcc, T)
        self.conv = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out_dim = hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mfcc, T)
        x = self.conv(x)              # (B, 128, T')
        x = x.permute(0, 2, 1)        # (B, T', 128)
        out, _ = self.gru(x)          # (B, T', 2*hidden)
        # Temporal pooling
        emb = out.mean(dim=1)         # (B, 2*hidden)
        return emb


# ---------------------------
# Visual encoder: 2D-CNN + BiGRU
# ---------------------------
class VisualEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        # Per-frame CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out_dim = hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)           # (B*T, 128, 1, 1)
        feats = feats.view(B, T, 128) # (B, T, 128)
        out, _ = self.gru(feats)      # (B, T, 2*hidden)
        emb = out.mean(dim=1)         # (B, 2*hidden)
        return emb


class AVSyncModel(nn.Module):
    """
    Audio-Visual lip-sync mismatch detection model.
    """

    def __init__(self, config: Optional[AVSyncConfig] = None):
        super().__init__()
        self.config = config or AVSyncConfig()
        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")

        self.audio_encoder = AudioEncoder(n_mfcc=self.config.n_mfcc, hidden=self.config.audio_hidden)
        self.visual_encoder = VisualEncoder(hidden=self.config.visual_hidden)

        fusion_in = self.audio_encoder.out_dim + self.visual_encoder.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, self.config.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.config.fusion_hidden, self.config.num_classes)
        )

        self.to(self.device)
        self.eval()

        # Load checkpoint if provided
        if self.config.checkpoint_path:
            self._load_checkpoint(self.config.checkpoint_path)

    # ---------------------------
    # Checkpoint loading
    # ---------------------------
    def _load_checkpoint(self, path: str) -> None:
        try:
            state = torch.load(path, map_location="cpu")
            self.load_state_dict(state, strict=False)
            logger.info(f"Loaded AVSync checkpoint: {path}")
        except Exception as e:
            logger.error(f"Failed to load AVSync checkpoint: {e}")

    # ---------------------------
    # Preprocessing
    # ---------------------------
    def _preprocess_mfcc(self, mfcc: np.ndarray) -> torch.Tensor:
        """
        MFCC: (n_mfcc, T) -> tensor (1, n_mfcc, T)
        """
        try:
            if mfcc.ndim != 2:
                raise ValueError(f"MFCC must be 2D (n_mfcc x T), got shape {mfcc.shape}")
            # Pad/truncate time axis to reasonable length (optional)
            tensor = torch.from_numpy(mfcc).unsqueeze(0).float()  # (1, n_mfcc, T)
            return tensor.to(self.device)
        except Exception as e:
            logger.error(f"MFCC preprocessing failed: {e}")
            # Fallback dummy
            return torch.zeros((1, self.config.n_mfcc, 100), dtype=torch.float32).to(self.device)

    def _preprocess_mouth_frames(self, mouth_frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """
        Mouth frames: list of BGR images -> tensor (1, T, C, H, W)
        """
        try:
            w, h = self.config.mouth_size
            frames = mouth_frames_bgr.copy()

            # Ensure sequence length
            if len(frames) < self.config.sequence_length:
                logger.warning(f"Mouth frames insufficient: {len(frames)} < {self.config.sequence_length}. Padding by repeating last frame.")
                if len(frames) == 0:
                    # Dummy sequence
                    dummy = np.zeros((h, w, 3), dtype=np.uint8)
                    frames = [dummy] * self.config.sequence_length
                else:
                    last = frames[-1]
                    while len(frames) < self.config.sequence_length:
                        frames.append(last)
            elif len(frames) > self.config.sequence_length:
                frames = frames[:self.config.sequence_length]

            seq = []
            for f in frames:
                img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (w, h))
                img_float = img_resized.astype(np.float32) / 255.0
                img_norm = (img_float - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
                img_chw = np.transpose(img_norm, (2, 0, 1))  # (C, H, W)
                seq.append(img_chw)

            seq_np = np.stack(seq, axis=0)  # (T, C, H, W)
            tensor = torch.from_numpy(seq_np).unsqueeze(0).float()  # (1, T, C, H, W)
            return tensor.to(self.device)
        except Exception as e:
            logger.error(f"Mouth frames preprocessing failed: {e}")
            # Fallback dummy
            return torch.zeros((1, self.config.sequence_length, 3, self.config.mouth_size[1], self.config.mouth_size[0]), dtype=torch.float32).to(self.device)

    # ---------------------------
    # Inference
    # ---------------------------
    @torch.inference_mode()
    def infer(self, mfcc: np.ndarray, mouth_frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
        """
        Perform AV lip-sync mismatch inference.
        Args:
            mfcc: numpy array (n_mfcc x T)
            mouth_frames_bgr: list of BGR mouth crops (length ~ sequence_length)
        Returns:
            dict with logits, probabilities, and confidence score for 'fake' class.
        """
        try:
            a = self._preprocess_mfcc(mfcc)                 # (B, n_mfcc, T)
            v = self._preprocess_mouth_frames(mouth_frames_bgr)  # (B, T, C, H, W)

            a_emb = self.audio_encoder(a)                   # (B, A_dim)
            v_emb = self.visual_encoder(v)                  # (B, V_dim)

            fused = torch.cat([a_emb, v_emb], dim=1)        # (B, A_dim + V_dim)
            logits = self.classifier(fused)                 # (B, num_classes)
            probs = torch.softmax(logits, dim=1)
            probs_np = probs.detach().cpu().numpy()[0]

            # Assume class index 1 = 'fake'
            fake_conf = float(probs_np[1]) if self.config.num_classes >= 2 else float(probs_np.max())

            return {
                "logits": logits.detach().cpu().numpy()[0],
                "probs": probs_np,
                "confidence_fake": fake_conf
            }
        except Exception as e:
            logger.error(f"AVSync inference failed: {e}")
            return {
                "logits": None,
                "probs": None,
                "confidence_fake": 0.0
            }

    # ---------------------------
    # Utility
    # ---------------------------
    def summary(self) -> str:
        try:
            params = sum(p.numel() for p in self.parameters())
            return f"AVSyncModel: Classes={self.config.num_classes}, SeqLen={self.config.sequence_length}, Params={params}, Device={self.device}"
        except Exception as e:
            logger.error(f"Summary failed: {e}")
            return f"AVSyncModel: SeqLen={self.config.sequence_length}, Device={self.device}"
