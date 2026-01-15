"""
AudioProcessor Module for AegisVerity
-------------------------------------
Responsibilities:
- Load audio stream from video or audio file.
- Extract MFCC features using librosa.
- Normalize and package features for downstream models.
- Handle errors gracefully with logging.

Notes:
- Uses librosa for audio decoding and feature extraction.
- Supports direct audio files (.wav, .mp3) or audio track from video.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import librosa
import soundfile as sf
from loguru import logger


class AudioProcessorConfig:
    """
    Configuration for AudioProcessor.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        hop_length: int = 512,
        n_fft: int = 2048,
        duration: Optional[float] = None
    ):
        """
        Args:
            sample_rate: Target sampling rate for audio.
            n_mfcc: Number of MFCC coefficients.
            hop_length: Hop length for STFT.
            n_fft: FFT window size.
            duration: Optional max duration (seconds) to load.
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.duration = duration


class AudioProcessor:
    """
    AudioProcessor for extracting MFCC features.
    """

    def __init__(self, config: Optional[AudioProcessorConfig] = None):
        self.config = config or AudioProcessorConfig()

    def load_audio(self, path: str) -> Optional[np.ndarray]:
        """
        Load audio file.
        Args:
            path: Path to audio file.
        Returns:
            Audio time series (numpy array).
        """
        if not os.path.exists(path):
            logger.error(f"Audio file not found: {path}")
            return None

        try:
            y, sr = librosa.load(
                path,
                sr=self.config.sample_rate,
                duration=self.config.duration
            )
            logger.info(f"Loaded audio: {path}, sr={sr}, length={len(y)} samples")
            return y
        except Exception as e:
            logger.error(f"Failed to load audio {path}: {e}")
            return None

    def extract_mfcc(self, y: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract MFCC features