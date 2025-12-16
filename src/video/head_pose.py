"""Head pose estimation interface (reserved for future implementation).

This module defines the interface for head pose estimators that can be used
to detect head orientation (e.g., looking down, looking up, looking sideways).

Currently, the system uses body bbox tracking to maintain track continuity when
face detection fails. In the future, explicit head pose estimation (e.g., via
MediaPipe Pose, 6DoF head pose, or landmark-based methods) can be integrated
by implementing this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class HeadPose:
    """Head pose estimation result."""

    pitch: float  # Head rotation around X-axis (nodding): positive = looking down, negative = looking up
    yaw: float  # Head rotation around Y-axis (shaking): positive = looking right, negative = looking left
    roll: float  # Head rotation around Z-axis (tilting): positive = tilt right, negative = tilt left
    confidence: float  # Confidence score [0, 1]

    def is_looking_down(self, threshold_degrees: float = 20.0) -> bool:
        """Check if the person is looking down (e.g., reading, writing)."""
        return self.pitch > threshold_degrees

    def is_looking_up(self, threshold_degrees: float = 20.0) -> bool:
        """Check if the person is looking up."""
        return self.pitch < -threshold_degrees

    def is_looking_sideways(self, threshold_degrees: float = 30.0) -> bool:
        """Check if the person is looking to the side (profile view)."""
        return abs(self.yaw) > threshold_degrees

    def is_frontal(self, pitch_threshold: float = 15.0, yaw_threshold: float = 20.0) -> bool:
        """Check if the person is looking roughly toward the camera."""
        return abs(self.pitch) < pitch_threshold and abs(self.yaw) < yaw_threshold


class HeadPoseEstimator(ABC):
    """Abstract interface for head pose estimators.

    Implementations can use various methods:
    - MediaPipe Face Mesh landmarks
    - 6DoF head pose estimation
    - Deep learning-based models (e.g., HopeNet, FSA-Net)
    - InsightFace landmark-based estimation
    """

    @abstractmethod
    def estimate(self, frame: np.ndarray, face_bbox: Optional[List[int]] = None) -> Optional[HeadPose]:
        """Estimate head pose from a video frame.

        Args:
            frame: BGR image (H, W, 3)
            face_bbox: Optional face bounding box [x1, y1, x2, y2] to crop the region of interest

        Returns:
            HeadPose if successful, None if estimation fails
        """
        pass

    @abstractmethod
    def estimate_batch(
        self, frames: List[np.ndarray], face_bboxes: Optional[List[Optional[List[int]]]] = None
    ) -> List[Optional[HeadPose]]:
        """Estimate head poses for a batch of frames (for efficiency).

        Args:
            frames: List of BGR images
            face_bboxes: Optional list of face bounding boxes corresponding to each frame

        Returns:
            List of HeadPose (or None for failed estimations)
        """
        pass


class DummyHeadPoseEstimator(HeadPoseEstimator):
    """Dummy implementation that always returns None (no pose estimation).

    This is the default implementation used when head pose estimation is not enabled.
    """

    def estimate(self, frame: np.ndarray, face_bbox: Optional[List[int]] = None) -> Optional[HeadPose]:
        return None

    def estimate_batch(
        self, frames: List[np.ndarray], face_bboxes: Optional[List[Optional[List[int]]]] = None
    ) -> List[Optional[HeadPose]]:
        return [None] * len(frames)


# Future implementations can be added here:
#
# class MediaPipeHeadPoseEstimator(HeadPoseEstimator):
#     """Head pose estimation using MediaPipe Face Mesh."""
#     ...
#
# class InsightFaceHeadPoseEstimator(HeadPoseEstimator):
#     """Head pose estimation using InsightFace landmarks (5-point or 106-point)."""
#     ...
