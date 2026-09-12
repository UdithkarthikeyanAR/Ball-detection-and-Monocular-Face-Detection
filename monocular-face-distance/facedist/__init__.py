"""Monocular face distance and deviation-angle estimation."""

from .bearing import Bearing, angular_separation_deg
from .camera import CameraIntrinsics, resolve_intrinsics
from .face_model import CanonicalFaceModel, DEFAULT_IPD_MM
from .estimator import DistanceEstimator, DistanceEstimate, PoseAngles
from .tracking import FaceTracker, TrackState
from .baseline import BaselineWidthEstimator, BaselineEstimate
from .pipeline import FaceDistancePipeline, FaceResult
from .quality import QualityMonitor, QualityReport

__version__ = "1.1.0"
__all__ = [
    "Bearing", "angular_separation_deg",
    "CameraIntrinsics", "resolve_intrinsics", "CanonicalFaceModel",
    "DEFAULT_IPD_MM", "DistanceEstimator", "DistanceEstimate", "PoseAngles",
    "FaceTracker", "TrackState", "BaselineWidthEstimator", "BaselineEstimate",
    "FaceDistancePipeline", "FaceResult", "QualityMonitor", "QualityReport",
]
