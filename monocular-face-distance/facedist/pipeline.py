"""End-to-end pipeline: frame in, tracked distance estimates out."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import bearing as bearing_mod
from .bearing import Bearing
from .camera import CameraIntrinsics, resolve_intrinsics
from .estimator import DistanceEstimate, DistanceEstimator
from .face_model import CanonicalFaceModel, DEFAULT_IPD_MM
from .quality import QualityMonitor, QualityReport
from .tracking import FaceTracker, TrackState


@dataclass
class FaceResult:
    """One face, as reported to the caller."""

    track_id: int
    bbox: tuple[float, float, float, float]
    distance_mm: float          # smoothed
    sigma_mm: float
    raw_distance_mm: float      # this frame only, unsmoothed
    velocity_mm_s: float
    yaw: float
    pitch: float
    roll: float
    confirmed: bool
    theta_deg: float = float("nan")
    """Deviation angle from the optical axis, smoothed. +ve = image right."""
    phi_deg: float = float("nan")
    """Elevation angle, smoothed. +ve = up. Beyond spec; free from the same fit."""
    sigma_theta_deg: float = float("nan")
    raw_theta_deg: float = float("nan")
    """Unsmoothed theta for this frame only."""
    bearing_source: str = "none"
    quality: QualityReport | None = None
    """Confidence breakdown for this reading. See facedist/quality.py."""
    ref_x_px: float = 0.0
    """Smoothed image position of the tracked anatomical point (eye midpoint)."""
    ref_y_px: float = 0.0
    face_span_px: float = 0.0
    reproj_rms_px: float = 0.0
    pnp_distance_mm: float | None = None
    ipd_distance_mm: float | None = None

    @property
    def distance_cm(self) -> float:
        return self.distance_mm / 10.0

    @property
    def approaching(self) -> bool:
        return self.velocity_mm_s < -50.0

    @property
    def theta_rad(self) -> float:
        return math.radians(self.theta_deg)

    def as_spec_output(self) -> tuple[float, float]:
        """``(Z_metres, theta_radians)`` -- the problem statement's expected output."""
        return self.distance_mm / 1000.0, self.theta_rad


class FaceDistancePipeline:
    """Detection -> per-frame geometry -> temporal filtering."""

    def __init__(
        self,
        width: int,
        height: int,
        calibration_path: str | None = None,
        ipd_mm: float = DEFAULT_IPD_MM,
        hfov_deg: float = 60.0,
        max_faces: int = 2,
        detector=None,
    ):
        self.intrinsics: CameraIntrinsics = resolve_intrinsics(
            width, height, calibration_path, hfov_deg
        )
        self.model = CanonicalFaceModel(ipd_mm=ipd_mm)
        self.estimator = DistanceEstimator(self.intrinsics, self.model)
        self.tracker = FaceTracker()
        self.quality = QualityMonitor()

        if detector is None:
            from .landmarks import LandmarkDetector
            detector = LandmarkDetector(max_faces=max_faces)
        self.detector = detector

        self._last_time: float | None = None
        self._fps_ema: float | None = None

    @property
    def fps(self) -> float:
        return self._fps_ema or 0.0

    def process(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        now = time.perf_counter()
        dt = (now - self._last_time) if self._last_time else 1.0 / 30.0
        self._last_time = now
        instantaneous = 1.0 / max(dt, 1e-6)
        self._fps_ema = (
            instantaneous if self._fps_ema is None
            else 0.9 * self._fps_ema + 0.1 * instantaneous
        )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        observations = self.detector.detect(frame_rgb)

        detections, per_face = [], []
        for obs in observations:
            est: DistanceEstimate = self.estimator.estimate(obs.landmarks_px)
            if not est.valid:
                continue
            # Track the eye midpoint, not the bbox centre. The box edge is
            # defined by the visible silhouette and slides toward the near
            # cheek as the head yaws, which would inject pose-dependent error
            # straight into the reported deviation angle.
            centre = est.ref_pixel if est.ref_pixel is not None else obs.centre
            detections.append({
                "bbox": obs.bbox,
                "centre": centre,
                "distance_mm": est.distance_mm,
                "sigma_mm": est.sigma_mm,
            })
            per_face.append((obs, est))

        tracks: list[TrackState] = self.tracker.update(detections, dt)

        results: list[FaceResult] = []
        for track in tracks:
            obs, est = self._nearest_observation(track, per_face)

            # Re-derive the bearing from the *smoothed* image position, so the
            # angle inherits the Kalman stage's noise rejection for free and
            # stays consistent with the smoothed depth.
            smoothed = bearing_mod.from_pixel(
                self.intrinsics, track.x_px, track.y_px,
                sigma_px=max(track.sigma_x_px, 0.5),
            )
            raw_bearing: Bearing = est.bearing if est else Bearing(
                float("nan"), float("nan"), source="none"
            )
            report = self.quality.update(
                track_id=track.track_id,
                distance_mm=track.distance_mm,
                face_span_px=est.face_span_px if est else 0.0,
                reproj_rms_px=est.reproj_rms_px if est else float("nan"),
                pnp_distance_mm=est.pnp_distance_mm if est else None,
                ipd_distance_mm=est.ipd_distance_mm if est else None,
                yaw_deg=est.pose.yaw if est else 0.0,
                pitch_deg=est.pose.pitch if est else 0.0,
                confirmed=track.confirmed,
            )
            results.append(
                FaceResult(
                    track_id=track.track_id,
                    bbox=obs.bbox if obs else (0.0, 0.0, 0.0, 0.0),
                    distance_mm=track.distance_mm,
                    sigma_mm=track.sigma_mm,
                    raw_distance_mm=est.distance_mm if est else float("nan"),
                    velocity_mm_s=track.velocity_mm_s,
                    yaw=est.pose.yaw if est else 0.0,
                    pitch=est.pose.pitch if est else 0.0,
                    roll=est.pose.roll if est else 0.0,
                    confirmed=track.confirmed,
                    theta_deg=smoothed.theta_deg,
                    phi_deg=smoothed.phi_deg,
                    sigma_theta_deg=smoothed.sigma_theta_deg,
                    raw_theta_deg=raw_bearing.theta_deg,
                    bearing_source=raw_bearing.source,
                    quality=report,
                    ref_x_px=track.x_px,
                    ref_y_px=track.y_px,
                    face_span_px=est.face_span_px if est else 0.0,
                    reproj_rms_px=est.reproj_rms_px if est else 0.0,
                    pnp_distance_mm=est.pnp_distance_mm if est else None,
                    ipd_distance_mm=est.ipd_distance_mm if est else None,
                )
            )
        return results

    @staticmethod
    def _nearest_observation(track: TrackState, per_face):
        best, best_d = (None, None), float("inf")
        for obs, est in per_face:
            d = (obs.centre[0] - track.x_px) ** 2 + (obs.centre[1] - track.y_px) ** 2
            if d < best_d:
                best, best_d = (obs, est), d
        return best

    def close(self) -> None:
        close = getattr(self.detector, "close", None)
        if callable(close):
            close()
