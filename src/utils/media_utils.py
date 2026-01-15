"""
AegisVerity Media Utilities
Media file processing and metadata extraction
"""

import os
import cv2
import librosa
from pathlib import Path
from typing import Optional, Tuple

from ..core.data_types import MediaMetadata


def extract_media_metadata(file_path: str) -> MediaMetadata:
    """
    Extract comprehensive metadata from media file
    
    Args:
        file_path: Path to media file
        
    Returns:
        MediaMetadata object with extracted information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Basic file information
    file_stat = file_path.stat()
    file_size = file_stat.st_size
    file_ext = file_path.suffix.lower()
    
    # Determine file type
    if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return _extract_video_metadata(file_path, file_size)
    elif file_ext in ['.wav', '.mp3', '.m4a', '.flac', '.aac']:
        return _extract_audio_metadata(file_path, file_size)
    elif file_ext in ['.jpg', '.jpeg', '.png']:
        return _extract_image_metadata(file_path, file_size)
    else:
        # Unknown format
        return MediaMetadata(
            file_path=str(file_path),
            file_type="unknown",
            file_size=file_size,
            format=file_ext.lstrip('.'),
            codec="unknown"
        )


def _extract_video_metadata(file_path: str, file_size: int) -> MediaMetadata:
    """Extract video-specific metadata"""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return MediaMetadata(
                file_path=file_path,
                file_type="video",
                file_size=file_size,
                format=Path(file_path).suffix.lstrip('.'),
                codec="unknown"
            )
        
        # Extract video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps and frame_count else None
        
        # Get codec information
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return MediaMetadata(
            file_path=file_path,
            file_type="video",
            duration=duration,
            fps=fps,
            resolution=(width, height),
            file_size=file_size,
            format=Path(file_path).suffix.lstrip('.'),
            codec=codec
        )
        
    except Exception as e:
        print(f"Warning: Could not extract video metadata: {str(e)}")
        return MediaMetadata(
            file_path=file_path,
            file_type="video",
            file_size=file_size,
            format=Path(file_path).suffix.lstrip('.'),
            codec="extraction_error"
        )


def _extract_audio_metadata(file_path: str, file_size: int) -> MediaMetadata:
    """Extract audio-specific metadata"""
    try:
        # Use librosa for audio analysis
        y, sr = librosa.load(file_path, sr=None)
        
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Determine format from file extension
        format_ext = Path(file_path).suffix.lstrip('.')
        
        return MediaMetadata(
            file_path=file_path,
            file_type="audio",
            duration=duration,
            sample_rate=sr,
            audio_channels=y.shape[0] if len(y.shape) > 1 else 1,
            file_size=file_size,
            format=format_ext,
            codec="librosa"
        )
        
    except Exception as e:
        print(f"Warning: Could not extract audio metadata: {str(e)}")
        return MediaMetadata(
            file_path=file_path,
            file_type="audio",
            file_size=file_size,
            format=Path(file_path).suffix.lstrip('.'),
            codec="extraction_error"
        )


def _extract_image_metadata(file_path: str, file_size: int) -> MediaMetadata:
    """Extract image-specific metadata"""
    try:
        # Use OpenCV to read image
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read image: {file_path}")
        
        height, width = img.shape[:2]
        
        # Determine format from file extension
        format_ext = Path(file_path).suffix.lstrip('.')
        
        return MediaMetadata(
            file_path=file_path,
            file_type="image",
            resolution=(width, height),
            file_size=file_size,
            format=format_ext,
            codec="opencv"
        )
        
    except Exception as e:
        print(f"Warning: Could not extract image metadata: {str(e)}")
        return MediaMetadata(
            file_path=file_path,
            file_type="image",
            file_size=file_size,
            format=Path(file_path).suffix.lstrip('.'),
            codec="extraction_error"
        )


def validate_media_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate if media file is supported and accessible
    
    Args:
        file_path: Path to media file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            return False, f"File not found: {file_path}"
        
        # Check file extension
        supported_extensions = {
            '.mp4', '.avi', '.mov', '.mkv', '.webm',  # Video
            '.wav', '.mp3', '.m4a', '.flac', '.aac',  # Audio
            '.jpg', '.jpeg', '.png'  # Image
        }
        
        if path.suffix.lower() not in supported_extensions:
            return False, f"Unsupported file format: {path.suffix}"
        
        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            return False, f"File not readable: {file_path}"
        
        # Try to read file header
        file_size = path.stat().st_size
        if file_size == 0:
            return False, f"Empty file: {file_path}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_media_info_string(metadata: MediaMetadata) -> str:
    """
    Generate human-readable media information string
    
    Args:
        metadata: MediaMetadata object
        
    Returns:
        Formatted string with media information
    """
    info_parts = []
    
    # Basic info
    info_parts.append(f"Type: {metadata.file_type.upper()}")
    info_parts.append(f"Format: {metadata.format}")
    info_parts.append(f"Size: {metadata.file_size / (1024*1024):.1f} MB")
    
    # Type-specific info
    if metadata.file_type == "video":
        if metadata.duration:
            info_parts.append(f"Duration: {metadata.duration:.1f}s")
        if metadata.fps:
            info_parts.append(f"FPS: {metadata.fps:.1f}")
        if metadata.resolution:
            w, h = metadata.resolution
            info_parts.append(f"Resolution: {w}x{h}")
        if metadata.codec:
            info_parts.append(f"Codec: {metadata.codec}")
    
    elif metadata.file_type == "audio":
        if metadata.duration:
            info_parts.append(f"Duration: {metadata.duration:.1f}s")
        if metadata.sample_rate:
            info_parts.append(f"Sample Rate: {metadata.sample_rate} Hz")
        if metadata.audio_channels:
            info_parts.append(f"Channels: {metadata.audio_channels}")
    
    elif metadata.file_type == "image":
        if metadata.resolution:
            w, h = metadata.resolution
            info_parts.append(f"Resolution: {w}x{h}")
    
    return " | ".join(info_parts)
