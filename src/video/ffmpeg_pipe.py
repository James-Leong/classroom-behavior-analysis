from __future__ import annotations

import shutil
import subprocess

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _ffmpeg_has_encoder(encoder: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return False
    encoders_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return str(encoder) in encoders_text


def _pick_ffmpeg_codec(requested: Optional[str]) -> str:
    if requested:
        return str(requested)
    # Prefer NVENC if available; otherwise fall back to software.
    if _ffmpeg_has_encoder("h264_nvenc"):
        return "h264_nvenc"
    if _ffmpeg_has_encoder("libx264"):
        return "libx264"
    return "mpeg4"


class FFmpegPipeWriter:
    """Write frames to a video file by piping raw BGR frames into ffmpeg.

    This is used to avoid OpenCV VideoWriter and avoid a second ffmpeg transcode pass.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: Optional[str],
        preset: str = "p4",
        crf_or_cq: int = 23,
    ):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("未找到 ffmpeg，无法输出视频")

        self.output_path = str(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec = _pick_ffmpeg_codec(codec)
        self.preset = str(preset)
        self.crf_or_cq = int(crf_or_cq)

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
        ]

        if self.codec == "h264_nvenc":
            cmd += ["-c:v", "h264_nvenc", "-preset", self.preset, "-cq", str(self.crf_or_cq)]
        elif self.codec == "libx264":
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.crf_or_cq)]
        else:
            # Maximum compatibility fallback (CPU).
            cmd += ["-c:v", "mpeg4", "-q:v", "5"]

        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", self.output_path]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin 管道创建失败")

    def write(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8, copy=False)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("frame must be HxWx3 BGR")
            h, w = int(frame.shape[0]), int(frame.shape[1])
            if h != self.height or w != self.width:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError as e:
            raise RuntimeError("ffmpeg 管道已断开（编码失败）") from e

    def close(self) -> None:
        if getattr(self, "_proc", None) is None:
            return
        try:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.flush()
                except Exception:
                    pass
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            rc = self._proc.wait(timeout=30)
            if rc != 0:
                raise RuntimeError(f"ffmpeg 编码失败: returncode={rc}, codec={self.codec}")
        finally:
            self._proc = None


# Backward-compatible alias (internal name used previously)
_FFmpegPipeWriter = FFmpegPipeWriter
