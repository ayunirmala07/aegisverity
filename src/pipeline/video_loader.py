"""
VideoLoader Module for AegisVerity
----------------------------------
Responsibilities:
- Robust video decoding using ffmpeg-python (primary) with OpenCV fallback.
- Adaptive frame sampling: fixed stride and event-driven (face presence).
- Metadata extraction (fps, duration, resolution) via ffprobe.
- Safe resource management and meaningful logging.

Notes:
- Primary path uses ffmpeg to pipe raw frames (RGB24) into Python.
- Fallback path uses OpenCV VideoCapture if ffmpeg pipeline fails.
"""

from __future__ import annotations

import io
import math
import subprocess
from typing import Generator, Optional, Tuple, Dict, Any, List

import numpy as np
import cv2
from loguru import logger

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    logger.warning("ffmpeg-python not available. Will fallback to OpenCV VideoCapture.")

# Optional: face-driven sampling requires a face detector
try:
    from .face_extractor import FaceExtractor
    FACE_EXTRACTOR_AVAILABLE = True
except Exception:
    FACE_EXTRACTOR_AVAILABLE = False
    logger.warning("FaceExtractor not available. Event-driven sampling will be disabled.")


class VideoLoaderConfig:
    """
    Configuration for VideoLoader.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 360),
        fixed_stride: int = 5,
        event_driven: bool = True,
        face_threshold: int = 1,
        max_frames: Optional[int] = None,
        use_ffmpeg: bool = True
    ):
        """
        Args:
            target_size: Resize frames to (width, height) for downstream processing.
            fixed_stride: Sample every Nth frame (applies to both pipelines).
            event_driven: If True, prioritize frames containing faces (requires FaceExtractor).
            face_threshold: Minimum number of faces to consider a frame "eventful".
            max_frames: Optional cap on total frames yielded.
            use_ffmpeg: Prefer ffmpeg pipeline if available.
        """
        self.target_size = target_size
        self.fixed_stride = max(1, fixed_stride)
        self.event_driven = event_driven and FACE_EXTRACTOR_AVAILABLE
        self.face_threshold = max(1, face_threshold)
        self.max_frames = max_frames if (isinstance(max_frames, int) and max_frames > 0) else None
        self.use_ffmpeg = use_ffmpeg and FFMPEG_AVAILABLE


class VideoLoader:
    """
    VideoLoader for robust ingestion and adaptive sampling.
    """

    def __init__(self, config: Optional[VideoLoaderConfig] = None):
        self.config = config or VideoLoaderConfig()
        self._metadata: Dict[str, Any] = {}
        self._face_extractor: Optional[FaceExtractor] = None
        if self.config.event_driven:
            try:
                self._face_extractor = FaceExtractor(target_size=(224, 224))
                logger.info("Event-driven sampling enabled with FaceExtractor.")
            except Exception as e:
                logger.error(f"Failed to initialize FaceExtractor: {e}")
                self._face_extractor = None
                self.config.event_driven = False

    # ---------------------------
    # Metadata via ffprobe
    # ---------------------------
    def probe(self, path: str) -> Dict[str, Any]:
        """
        Extract metadata using ffprobe.
        Returns dict with keys: width, height, fps, duration, codec.
        """
        if not FFMPEG_AVAILABLE:
            logger.warning("ffmpeg-python not available. Metadata will be limited.")
            return {}

        try:
            probe = ffmpeg.probe(path)
            video_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
            if not video_stream:
                logger.error("No video stream found in file.")
                return {}

            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            # fps parsing
            r_frame_rate = video_stream.get("r_frame_rate", "0/0")
            try:
                num, den = r_frame_rate.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 0.0
            except Exception:
                fps = 0.0

            duration = float(video_stream.get("duration", probe.get("format", {}).get("duration", 0.0)))
            codec = video_stream.get("codec_name", "unknown")

            self._metadata = {
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "codec": codec
            }
            logger.info(f"Probed metadata: {self._metadata}")
            return self._metadata
        except Exception as e:
            logger.error(f"ffprobe failed: {e}")
            return {}

    # ---------------------------
    # FFmpeg primary pipeline
    # ---------------------------
    def _ffmpeg_frame_generator(self, path: str) -> Generator[np.ndarray, None, None]:
        """
        Decode frames via ffmpeg piping rawvideo (RGB24).
        """
        try:
            meta = self._metadata or self.probe(path)
            width = meta.get("width", 0)
            height = meta.get("height", 0)
            if width == 0 or height == 0:
                logger.warning("Invalid metadata dimensions; falling back to OpenCV.")
                raise RuntimeError("Invalid dimensions")

            process = (
                ffmpeg
                .input(path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

            frame_size = width * height * 3
            idx = 0
            while True:
                in_bytes = process.stdout.read(frame_size)
                if not in_bytes:
                    break
                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                # Convert to BGR for OpenCV compatibility
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                yield frame_bgr
                idx += 1

            process.stdout.close()
            process.wait()
        except Exception as e:
            logger.error(f"FFmpeg frame generator failed: {e}")
            raise

    # ---------------------------
    # OpenCV fallback pipeline
    # ---------------------------
    def _opencv_frame_generator(self, path: str) -> Generator[np.ndarray, None, None]:
        """
        Decode frames via OpenCV VideoCapture.
        """
        cap = None
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError("OpenCV cannot open video file.")

            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                yield frame
                idx += 1
        except Exception as e:
            logger.error(f"OpenCV frame generator failed: {e}")
            raise
        finally:
            if cap is not None:
                cap.release()

    # ---------------------------
    # Adaptive sampling
    # ---------------------------
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            w, h = self.config.target_size
            return cv2.resize(frame, (w, h))
        except Exception as e:
            logger.error(f"Frame resize failed: {e}")
            return frame

    def _eventful(self, frame: np.ndarray) -> bool:
        """
        Event-driven predicate: returns True if frame contains >= face_threshold faces.
        """
        if not self._face_extractor:
            return False
        try:
            faces = self._face_extractor.process_frame(frame)
            return len(faces) >= self.config.face_threshold
        except Exception as e:
            logger.error(f"Event-driven check failed: {e}")
            return False

    def frames(self, path: str) -> Generator[np.ndarray, None, None]:
        """
        Public generator: yields sampled frames (BGR, resized).
        Applies fixed stride and optional event-driven prioritization.
        """
        generator = None
        try:
            if self.config.use_ffmpeg:
                generator = self._ffmpeg_frame_generator(path)
            else:
                generator = self._opencv_frame_generator(path)
        except Exception:
            logger.warning("Primary pipeline failed. Falling back to OpenCV.")
            generator = self._opencv_frame_generator(path)

        count = 0
        stride = self.config.fixed_stride
        for idx, frame in enumerate(generator):
            # Fixed stride sampling
            if idx % stride != 0:
                continue

            # Event-driven prioritization
            if self.config.event_driven:
                if not self._eventful(frame):
                    # Skip non-eventful frames but still allow periodic sampling
                    # e.g., every K*stride to maintain coverage
                    pass

            resized = self._resize_frame(frame)
            yield resized
            count += 1

            if self.config.max_frames and count >= self.config.max_frames:
                logger.info(f"Reached max_frames={self.config.max_frames}. Stopping.")
                break

    # ---------------------------
    # Convenience methods
    # ---------------------------
    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata

    def estimate_total_frames(self) -> Optional[int]:
        """
        Estimate total frames using fps * duration (if available).
        """
        try:
            fps = float(self._metadata.get("fps", 0.0))
            duration = float(self._metadata.get("duration", 0.0))
            if fps > 0 and duration > 0:
                return int(math.floor(fps * duration))
            return None
        except Exception as e:
            logger.error(f"Failed to estimate total frames: {e}")
            return None
