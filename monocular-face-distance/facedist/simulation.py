"""Synthetic evaluation harness.

Live webcam footage has no ground-truth depth, so accuracy claims made from it
are unfalsifiable. Instead we render the canonical face model at *known* poses
and distances through the exact camera model the estimator assumes, add
realistic landmark noise, and measure the recovered depth against truth.

This isolates the estimator's geometry from the landmark detector's accuracy,
which is what we want when comparing estimation methods. Two deliberate
realism controls are included:

* ``shape_jitter_mm`` perturbs the *subject's* 3D face away from the canonical
  model, so the estimator never sees the exact geometry it assumes;
* ``true_ipd_mm`` can differ from the assumed IPD, reproducing the dominant
  real-world error term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import CameraIntrinsics
from .face_model import (
    IDX_R_EYE_OUTER, IDX_R_EYE_INNER, IDX_L_EYE_OUTER, IDX_L_EYE_INNER,
    IDX_R_IRIS_CENTRE, IDX_R_IRIS_RING, IDX_L_IRIS_CENTRE, IDX_L_IRIS_RING,
    IRIS_DIAMETER_MM, IRIS_DIAMETER_SIGMA_MM,
)
from .face_model import CanonicalFaceModel, PNP_LANDMARK_IDS

MAX_LANDMARK_ID = 468


@dataclass
class SyntheticFrame:
    landmarks_px: np.ndarray
    true_distance_mm: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    true_theta_deg: float = 0.0
    """Ground-truth deviation angle (azimuth) of the eye midpoint."""
    true_phi_deg: float = 0.0
    """Ground-truth elevation of the eye midpoint."""


class FaceSimulator:
    """Projects a perturbed 3D face through a known camera."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        true_ipd_mm: float = 63.0,
        shape_jitter_mm: float = 3.0,
        landmark_noise_px: float = 1.5,
        seed: int = 0,
    ):
        self.intrinsics = intrinsics
        self.landmark_noise_px = landmark_noise_px
        self.rng = np.random.default_rng(seed)

        subject = CanonicalFaceModel(ipd_mm=true_ipd_mm)
        jitter = self.rng.normal(0.0, shape_jitter_mm, subject.points_mm.shape)
        self.subject_points = subject.points_mm + jitter
        self.reference_point = subject.reference_point_mm

        # This subject's own iris diameter. Drawing it from the real population
        # spread is the whole point of the test: an iris cue that is only ever
        # rendered at the population mean would look perfect for reasons that
        # do not survive contact with an actual face.
        self.true_iris_mm = float(
            self.rng.normal(IRIS_DIAMETER_MM, IRIS_DIAMETER_SIGMA_MM)
        )
        # Pupils sit at the midpoint of each eye's corner pair.
        ids = list(PNP_LANDMARK_IDS)
        def _mid(a, b):
            return 0.5 * (self.subject_points[ids.index(a)]
                          + self.subject_points[ids.index(b)])
        r = self.true_iris_mm / 2.0
        self.iris_points = {}
        for centre_id, ring_ids, outer, inner in (
            (IDX_R_IRIS_CENTRE, IDX_R_IRIS_RING, IDX_R_EYE_OUTER, IDX_R_EYE_INNER),
            (IDX_L_IRIS_CENTRE, IDX_L_IRIS_RING, IDX_L_EYE_OUTER, IDX_L_EYE_INNER),
        ):
            c = _mid(outer, inner)
            # MediaPipe ring order is right, top, left, bottom; +y is down.
            self.iris_points[centre_id] = c
            for rid, off in zip(ring_ids, ([r, 0, 0], [0, -r, 0],
                                           [-r, 0, 0], [0, r, 0])):
                self.iris_points[rid] = c + np.asarray(off, dtype=np.float64)

    def render(self, distance_mm: float, yaw_deg: float = 0.0,
               pitch_deg: float = 0.0, roll_deg: float = 0.0,
               theta_deg: float = 0.0, phi_deg: float = 0.0) -> SyntheticFrame:
        """Return a full-length landmark array with the PnP points filled in.

        ``theta_deg`` / ``phi_deg`` place the face off the optical axis at a
        known bearing while keeping its *depth* exactly ``distance_mm``. The
        placement is exact rather than approximate: writing the reference point
        as

            X = Z * tan(theta)
            Y = -Z * sin(phi) / (cos(theta) * cos(phi))
            Z = distance_mm

        gives back atan2(X, Z) == theta and atan2(-Y, hypot(X, Z)) == phi
        identically, so the harness can score angular error without the ground
        truth itself carrying an approximation.
        """
        rot = cv2.Rodrigues(
            np.radians([pitch_deg, yaw_deg, roll_deg]).astype(np.float64)
        )[0]

        th, ph = math.radians(theta_deg), math.radians(phi_deg)
        target = np.array([
            distance_mm * math.tan(th),
            -distance_mm * math.sin(ph) / (math.cos(th) * math.cos(ph)),
            distance_mm,
        ], dtype=np.float64)

        # Place the face so the reference point sits at exactly `target`.
        ref_rotated = rot @ self.reference_point
        tvec = target - ref_rotated

        iris_ids = list(self.iris_points.keys())
        iris_arr = np.array([self.iris_points[i] for i in iris_ids])
        all_model = np.vstack([self.subject_points, iris_arr])

        cam_points = (rot @ all_model.T).T + tvec
        projected, _ = cv2.projectPoints(
            all_model,
            cv2.Rodrigues(rot)[0],
            tvec.reshape(3, 1),
            self.intrinsics.matrix,
            self.intrinsics.distortion,
        )
        pts = projected.reshape(-1, 2)
        pts = pts + self.rng.normal(0.0, self.landmark_noise_px, pts.shape)

        n_pnp = len(PNP_LANDMARK_IDS)
        full = np.zeros((max(MAX_LANDMARK_ID, 478), 2), dtype=np.float64)
        for row, lm_id in enumerate(PNP_LANDMARK_IDS):
            full[lm_id] = pts[row]
        for k, lm_id in enumerate(iris_ids):
            full[lm_id] = pts[n_pnp + k]

        # Fill unused slots with the face centroid so bbox-based methods see a
        # sensible extent rather than a box anchored at the origin.
        occupied = np.array(list(PNP_LANDMARK_IDS) + iris_ids)
        centroid = pts[:n_pnp].mean(axis=0)
        mask = np.ones(max(MAX_LANDMARK_ID, 478), dtype=bool)
        mask[occupied] = False
        full[mask] = centroid

        assert cam_points[:, 2].min() > 0, "face rendered behind the camera"
        return SyntheticFrame(
            landmarks_px=full,
            true_distance_mm=distance_mm,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            true_theta_deg=theta_deg,
            true_phi_deg=phi_deg,
        )

    @staticmethod
    def bbox_centre(bbox) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    def bbox_of(self, landmarks_px: np.ndarray):
        pts = landmarks_px[list(PNP_LANDMARK_IDS)]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return float(x1), float(y1), float(x2), float(y2)
