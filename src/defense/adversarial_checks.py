"""
AdversarialChecks Module for AegisVerity
----------------------------------------
Responsibilities:
- Run lightweight heuristics to flag potential adversarial manipulations:
  - Noise injection (variance spikes)
  - Blur/defocus (Laplacian variance)
  - Compression artifacts (blockiness)
  - Color histogram shifts (distribution drift)
  - Temporal jitter (frame-to-frame instability)
  - Frequency anomalies (FFT energy spikes)
- Return human-readable signals for forensic context.

Compatibility:
- OpenCV 4.8+
- NumPy
- loguru 0.7.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
from loguru import logger


@dataclass
class DefenseConfig:
    """
    Configuration for adversarial heuristics thresholds.
    """
    # Noise
    noise_var_threshold: float = 0.015  # normalized variance threshold

    # Blur
    laplacian_var_threshold: float = 60.0  # lower => blur

    # Compression blockiness
    blockiness_threshold: float = 12.0

    # Color histogram drift
    hist_drift_threshold: float = 0.15  # Bhattacharyya distance

    # Temporal jitter
    psnr_low_threshold: float = 24.0  # dB; lower => unstable
    temporal_jitter_ratio: float = 0.35  # fraction of pairs below PSNR threshold

    # Frequency anomalies
    fft_energy_spike_ratio: float = 0.25  # fraction of energy in high bands


class AdversarialChecks:
    """
    Basic adversarial defense heuristics.
    """

    def __init__(self, config: Optional[DefenseConfig] = None):
        self.config = config or DefenseConfig()

    # ---------------------------
    # Public API
    # ---------------------------
    def evaluate_frames(self, frames_bgr: List[np.ndarray]) -> List[str]:
        """
        Evaluate a sequence of frames and return signals.
        Args:
            frames_bgr: List of BGR frames.
        Returns:
            List of human-readable signals (warnings).
        """
        signals: List[str] = []
        if not frames_bgr or len(frames_bgr) < 2:
            logger.warning("Insufficient frames for adversarial checks.")
            return ["[DEFENSE] Insufficient frames for adversarial checks"]

        try:
            # Per-frame checks
            noise_flags = self._check_noise_variance(frames_bgr)
            blur_flags = self._check_blur_laplacian(frames_bgr)
            block_flags = self._check_blockiness(frames_bgr)
            hist_flags = self._check_histogram_drift(frames_bgr)

            # Temporal checks
            jitter_flag = self._check_temporal_jitter(frames_bgr)
            freq_flags = self._check_frequency_anomalies(frames_bgr)

            signals.extend(noise_flags)
            signals.extend(blur_flags)
            signals.extend(block_flags)
            signals.extend(hist_flags)
            if jitter_flag:
                signals.append(jitter_flag)
            signals.extend(freq_flags)

        except Exception as e:
            logger.error(f"Adversarial checks failed: {e}")
            signals.append(f"[DEFENSE] Adversarial checks failed: {e}")

        return signals

    # ---------------------------
    # Noise variance
    # ---------------------------
    def _check_noise_variance(self, frames_bgr: List[np.ndarray]) -> List[str]:
        """
        Estimate noise via local variance on grayscale.
        """
        signals: List[str] = []
        try:
            for idx, f in enumerate(frames_bgr):
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                gray_norm = gray.astype(np.float32) / 255.0
                # Local variance via Laplacian magnitude proxy
                lap = cv2.Laplacian(gray_norm, cv2.CV_32F)
                var = float(np.var(lap))
                if var > self.config.noise_var_threshold:
                    signals.append(f"[DEFENSE] High noise variance detected (frame {idx}, var={var:.4f})")
            return signals
        except Exception as e:
            logger.error(f"Noise variance check failed: {e}")
            return [f"[DEFENSE] Noise variance check failed: {e}"]

    # ---------------------------
    # Blur via Laplacian variance
    # ---------------------------
    def _check_blur_laplacian(self, frames_bgr: List[np.ndarray]) -> List[str]:
        """
        Detect blur/defocus using Laplacian variance.
        """
        signals: List[str] = []
        try:
            for idx, f in enumerate(frames_bgr):
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                var = float(lap.var())
                if var < self.config.laplacian_var_threshold:
                    signals.append(f"[DEFENSE] Possible blur/defocus (frame {idx}, lap_var={var:.2f})")
            return signals
        except Exception as e:
            logger.error(f"Blur check failed: {e}")
            return [f"[DEFENSE] Blur check failed: {e}"]

    # ---------------------------
    # Compression blockiness
    # ---------------------------
    def _check_blockiness(self, frames_bgr: List[np.ndarray]) -> List[str]:
        """
        Estimate blockiness (compression artifacts) via DCT grid energy.
        """
        signals: List[str] = []
        try:
            for idx, f in enumerate(frames_bgr):
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                # Downscale for speed
                ds = cv2.resize(gray, (w // 2, h // 2))
                # Horizontal/vertical gradients
                gx = cv2.Sobel(ds, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(ds, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(gx**2 + gy**2)
                # Sample grid lines every 8 pixels (JPEG block size)
                grid_energy = []
                for step in (8, 16):
                    grid_energy.append(float(np.mean(mag[:, ::step]) + np.mean(mag[::step, :])))
                score = float(np.mean(grid_energy))
                if score > self.config.blockiness_threshold:
                    signals.append(f"[DEFENSE] Compression blockiness suspected (frame {idx}, score={score:.2f})")
            return signals
        except Exception as e:
            logger.error(f"Blockiness check failed: {e}")
            return [f"[DEFENSE] Blockiness check failed: {e}"]

    # ---------------------------
    # Color histogram drift
    # ---------------------------
    def _check_histogram_drift(self, frames_bgr: List[np.ndarray]) -> List[str]:
        """
        Detect distribution drift via Bhattacharyya distance between consecutive histograms.
        """
        signals: List[str] = []
        try:
            prev_hist = None
            for idx, f in enumerate(frames_bgr):
                hist = []
                for ch in range(3):
                    h = cv2.calcHist([f], [ch], None, [32], [0, 256])
                    h = cv2.normalize(h, h).flatten()
                    hist.append(h)
                hist = np.concatenate(hist, axis=0)  # (96,)

                if prev_hist is not None:
                    # Bhattacharyya distance
                    bc = np.sum(np.sqrt(prev_hist * hist))
                    dist = float(np.clip(1.0 - bc, 0.0, 1.0))
                    if dist > self.config.hist_drift_threshold:
                        signals.append(f"[DEFENSE] Color histogram drift (pair {idx-1}->{idx}, dist={dist:.3f})")
                prev_hist = hist
            return signals
        except Exception as e:
            logger.error(f"Histogram drift check failed: {e}")
            return [f"[DEFENSE] Histogram drift check failed: {e}"]

    # ---------------------------
    # Temporal jitter via PSNR
    # ---------------------------
    def _check_temporal_jitter(self, frames_bgr: List[np.ndarray]) -> Optional[str]:
        """
        Estimate instability via frame-to-frame PSNR.
        """
        try:
            low_count = 0
            total = 0
            for i in range(1, len(frames_bgr)):
                a = cv2.cvtColor(frames_bgr[i - 1], cv2.COLOR_BGR2GRAY)
                b = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
                a = cv2.resize(a, (256, 256))
                b = cv2.resize(b, (256, 256))
                mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
                if mse == 0:
                    psnr = 99.0
                else:
                    psnr = 10.0 * np.log10((255.0 ** 2) / mse)
                total += 1
                if psnr < self.config.psnr_low_threshold:
                    low_count += 1
            if total == 0:
                return None
            ratio = low_count / total
            if ratio > self.config.temporal_jitter_ratio:
                return f"[DEFENSE] Temporal jitter detected (low-PSNR ratio={ratio:.2f})"
            return None
        except Exception as e:
            logger.error(f"Temporal jitter check failed: {e}")
            return f"[DEFENSE] Temporal jitter check failed: {e}"

    # ---------------------------
    # Frequency anomalies via FFT
    # ---------------------------
    def _check_frequency_anomalies(self, frames_bgr: List[np.ndarray]) -> List[str]:
        """
        Detect high-frequency energy spikes indicative of adversarial noise.
        """
        signals: List[str] = []
        try:
            for idx, f in enumerate(frames_bgr):
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (256, 256))
                # 2D FFT
                fft = np.fft.fft2(gray.astype(np.float32))
                fft_shift = np.fft.fftshift(fft)
                mag = np.abs(fft_shift)

                # Define center low-frequency mask
                h, w = mag.shape
                cy, cx = h // 2, w // 2
                r = 32
                mask = np.zeros_like(mag, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), r, 1, -1)

                low_energy = float(np.sum(mag * mask))
                high_energy = float(np.sum(mag * (1 - mask)))
                total_energy = low_energy + high_energy
                if total_energy == 0:
                    continue
                high_ratio = high_energy / total_energy

                if high_ratio > self.config.fft_energy_spike_ratio:
                    signals.append(f"[DEFENSE] High-frequency energy spike (frame {idx}, ratio={high_ratio:.2f})")
            return signals
        except Exception as e:
            logger.error(f"Frequency anomaly check failed: {e}")
            return [f"[DEFENSE] Frequency anomaly check failed: {e}"]
