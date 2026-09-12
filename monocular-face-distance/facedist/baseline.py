"""The baseline this project is meant to beat.

The conventional approach -- and what the reference implementation does -- is
the similar-triangles rule applied to the face bounding box:

    Z = f * W_real / w_pixels

It has two structural weaknesses that the fused estimator does not share:

1. **No pose compensation.** Turn your head 30 degrees and the box narrows by
   roughly cos(30) = 13%, which the baseline reads as having moved 13% closer.
2. **Detector-box noise.** The box edge is defined by whatever the detector
   thinks the face boundary is; it jitters by several pixels frame to frame and
   shifts systematically with expression and hair.

It is kept here as an honest reference point, not as a straw man: it is given
the same intrinsics and the same landmarks-derived box, so the comparison
isolates the estimation method rather than the detector quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .camera import CameraIntrinsics

#: Mean adult bizygomatic (cheekbone-to-cheekbone) breadth, in mm.
#: The problem statement quotes 0.14-0.16 m; we take the low end of that band
#: as the point estimate. Note the band itself is +/-7%, which propagates
#: one-for-one into depth error -- this is the baseline's dominant error term
#: and the reason per-subject scale calibration exists in this project.
DEFAULT_FACE_WIDTH_MM = 140.0


@dataclass(frozen=True)
class BaselineEstimate:
    """The pair the problem statement asks for: ``(depth, theta)``."""

    depth_mm: float
    theta_deg: float

    @property
    def depth_m(self) -> float:
        return self.depth_mm / 1000.0

    @property
    def theta_rad(self) -> float:
        return math.radians(self.theta_deg)

    def as_tuple(self) -> tuple[float, float]:
        """``(Z_metres, theta_radians)`` -- the literal expected output."""
        return self.depth_m, self.theta_rad


class BaselineWidthEstimator:
    """Reference implementation of the problem statement's model.

    Implements, verbatim::

        Depth: Z     = (f * W) / w_px
        Angle: theta = arctan((x - c_x) / f)

    Kept exact -- no undistortion, no pose correction, bounding-box centre for
    ``x`` -- because its purpose is to be an honest measuring stick, not to win.
    """

    def __init__(self, intrinsics: CameraIntrinsics,
                 face_width_mm: float = DEFAULT_FACE_WIDTH_MM):
        self.intrinsics = intrinsics
        self.face_width_mm = float(face_width_mm)

    # -- depth --------------------------------------------------------------

    def estimate(self, bbox: tuple[float, float, float, float]) -> float:
        """Return depth in mm, or NaN if the box is degenerate."""
        x1, _, x2, _ = bbox
        width_px = abs(x2 - x1)
        if width_px < 1.0:
            return float("nan")
        return self.intrinsics.fx * self.face_width_mm / width_px

    def estimate_from_landmarks(self, landmarks_px: np.ndarray) -> float:
        x1, y1 = landmarks_px.min(axis=0)[:2]
        x2, y2 = landmarks_px.max(axis=0)[:2]
        return self.estimate((float(x1), float(y1), float(x2), float(y2)))

    # -- angle --------------------------------------------------------------

    def angle(self, bbox: tuple[float, float, float, float]) -> float:
        """``theta = arctan((x - c_x) / f)`` in degrees, x = bbox centre."""
        x1, _, x2, _ = bbox
        centre_x = 0.5 * (x1 + x2)
        return math.degrees(
            math.atan((centre_x - self.intrinsics.cx) / self.intrinsics.fx)
        )

    # -- the pair the spec asks for ----------------------------------------

    def estimate_full(
        self, bbox: tuple[float, float, float, float]
    ) -> BaselineEstimate:
        """Return ``(depth, theta)`` exactly as the problem statement defines."""
        return BaselineEstimate(depth_mm=self.estimate(bbox),
                                theta_deg=self.angle(bbox))

    def estimate_full_from_landmarks(
        self, landmarks_px: np.ndarray
    ) -> BaselineEstimate:
        x1, y1 = landmarks_px.min(axis=0)[:2]
        x2, y2 = landmarks_px.max(axis=0)[:2]
        return self.estimate_full((float(x1), float(y1), float(x2), float(y2)))
