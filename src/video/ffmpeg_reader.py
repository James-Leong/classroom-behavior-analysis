"""FFmpeg-based video reader with hardware acceleration support.

Provides faster video decoding compared to OpenCV, especially with GPU acceleration.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import List, Optional

import cv2
import numpy as np

from src.utils.log import get_logger

logger = get_logger(__name__)


def _has_hwaccel(hwaccel: str) -> bool:
    """Check if ffmpeg supports the given hardware acceleration."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return hwaccel in output
    except Exception:
        return False


class FFmpegFrameReader:
    """Read video frames using ffmpeg with optional hardware acceleration.

    Significantly faster than OpenCV for reading multiple frames, especially with GPU decode.

    Example:
        >>> reader = FFmpegFrameReader("video.mp4", hwaccel="cuda")
        >>> frames = reader.read_frames([0, 10, 20, 30])  # Read specific frames
        >>> reader.close()
    """

    def __init__(
        self,
        video_path: str,
        hwaccel: Optional[str] = "auto",
    ):
        """Initialize ffmpeg reader.

        Args:
            video_path: Path to video file
            hwaccel: Hardware acceleration: 'auto', 'cuda', 'none', or None
                    'auto' tries cuda -> falls back to none
        """
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found")

        self.video_path = str(video_path)

        # Determine hardware acceleration
        if hwaccel == "auto":
            if _has_hwaccel("cuda"):
                self.hwaccel = "cuda"
                logger.debug("Using CUDA hardware acceleration for video decode")
            else:
                self.hwaccel = None
                logger.debug("Hardware acceleration not available, using CPU decode")
        elif hwaccel in ["cuda", "cuvid"]:
            if _has_hwaccel("cuda"):
                self.hwaccel = "cuda"
            else:
                logger.warning("CUDA hwaccel requested but not available, falling back to CPU")
                self.hwaccel = None
        else:
            self.hwaccel = None

        # Get video info using OpenCV (lightweight)
        cap = cv2.VideoCapture(self.video_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def read_frames(self, frame_indices: List[int]) -> List[np.ndarray]:
        """Read multiple frames efficiently using ffmpeg.

        Args:
            frame_indices: List of frame indices to read (0-based)

        Returns:
            List of BGR frames (H, W, 3) uint8 arrays

        Notes:
            - Automatically sorts and deduplicates frame indices
            - Uses select filter for efficient frame extraction
            - Much faster than seeking+reading with OpenCV
        """
        if not frame_indices:
            return []

        # Sort and deduplicate
        indices = sorted(set(frame_indices))

        # Build select filter expression: select='eq(n,0)+eq(n,10)+eq(n,20)...'
        select_expr = "+".join(f"eq(n,{idx})" for idx in indices)

        # Build ffmpeg command
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        # Add hardware acceleration if available
        if self.hwaccel:
            cmd += ["-hwaccel", self.hwaccel]

        # Input and filtering
        cmd += [
            "-i",
            self.video_path,
            "-vf",
            f"select='{select_expr}'",
            "-vsync",
            "0",  # Don't duplicate/drop frames
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            frames = []
            frame_size = self.height * self.width * 3

            while len(frames) < len(indices):
                raw_data = proc.stdout.read(frame_size)
                if len(raw_data) != frame_size:
                    break

                frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width, 3))
                frames.append(frame)

            proc.stdout.close()
            proc.wait(timeout=30)

            if len(frames) != len(indices):
                logger.warning(f"Expected {len(indices)} frames, got {len(frames)}")

            return frames

        except Exception as e:
            logger.error(f"FFmpeg frame reading failed: {e}")
            raise

    def read_frame_range(self, start_frame: int, num_frames: int) -> List[np.ndarray]:
        """Read a contiguous range of frames efficiently using ffmpeg seeking.

        Args:
            start_frame: Start frame index (0-based)
            num_frames: Number of frames to read

        Returns:
            List of BGR frames
        """
        if num_frames <= 0:
            return []

        # Calculate start time
        # Note: ffmpeg -ss before -i is fast and accurate (decodes and discards to reach timestamp)
        start_time = max(0.0, start_frame / self.fps)

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        # Add hardware acceleration if available
        if self.hwaccel:
            cmd += ["-hwaccel", self.hwaccel]

        # Fast seek to start time
        cmd += ["-ss", f"{start_time:.6f}"]

        # Input
        cmd += ["-i", self.video_path]

        # Limit number of frames
        cmd += ["-vframes", str(num_frames)]

        # Output format
        cmd += [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            frames = []
            frame_size = self.height * self.width * 3

            while len(frames) < num_frames:
                raw_data = proc.stdout.read(frame_size)
                if len(raw_data) != frame_size:
                    break

                frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width, 3))
                frames.append(frame)

            proc.stdout.close()
            proc.wait(timeout=30)

            return frames

        except Exception as e:
            logger.error(f"FFmpeg frame range reading failed: {e}")
            raise

    def close(self):
        """Cleanup (no-op for this implementation)."""
        pass


def read_frames_with_ffmpeg(
    video_path: str,
    frame_indices: List[int],
    hwaccel: Optional[str] = "auto",
) -> List[np.ndarray]:
    """Convenience function to read frames using ffmpeg.

    Args:
        video_path: Path to video file
        frame_indices: Frame indices to read
        hwaccel: Hardware acceleration ('auto', 'cuda', 'none')

    Returns:
        List of BGR frames
    """
    reader = FFmpegFrameReader(video_path, hwaccel=hwaccel)
    try:
        return reader.read_frames(frame_indices)
    finally:
        reader.close()
