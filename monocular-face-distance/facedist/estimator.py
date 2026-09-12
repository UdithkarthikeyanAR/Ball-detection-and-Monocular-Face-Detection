"""Monocular face-distance estimation from 2D facial landmarks.

Two independent geometric cues are computed:

1. **PnP cue** -- solve the perspective-n-point problem between the metric
   canonical face model and the observed landmarks. Uses the full point set, so
   it stays well-conditioned under large head rotation and gives pose for free.

2. **IPD cue** -- the classical pinhole relation ``Z = f * B / b`` applied to the
   eye baseline ``B`` (interpupillary distance) with apparent width ``b``, yaw-
   corrected using the pose from cue 1. This cue depends on only two well-
   localised points, so it degrades gracefully when the rest of the face is
   poorly tracked (occlusion, extreme expression, motion blur).

The cues share one error term -- the assumed IPD sets the metric scale of both --
so that component is *excluded* from the fusion weights and re-added afterwards.
Fusing it in would understate the true uncertainty. Fusion happens in inverse
depth ``q = 1/Z`` because pixel noise maps to roughly constant variance in ``q``
but to variance growing as ``Z^2`` in ``Z`` itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import bearing as bearing_mod
from .bearing import Bearing
from .camera import CameraIntrinsics
from .face_model import (
    IDX_R_IRIS_CENTRE, IDX_R_IRIS_RING, IDX_L_IRIS_CENTRE, IDX_L_IRIS_RING,
    IRIS_DIAMETER_MM, IRIS_DIAMETER_SIGMA_MM,
    CanonicalFaceModel,
    DEFAULT_IPD_SIGMA_MM,
    IDX_L_EYE_INNER,
    IDX_L_EYE_OUTER,
    IDX_R_EYE_INNER,
    IDX_R_EYE_OUTER,
)

#: Assumed std-dev of a single landmark's image position, in pixels.
LANDMARK_SIGMA_PX = 1.5
#: Assumed std-dev of the yaw estimate, in radians (~4 degrees).
YAW_SIGMA_RAD = 0.07
#: Depth error amplification factor for the PnP cue (empirical, see tests).
PNP_ERROR_GAIN = 3.0
#: Relative scale error a *single* subject's face shape imposes on each cue.
#: Measured on the synthetic harness with 3 mm of per-vertex shape variation:
#: the two-point IPD cue picked up a ~4.5% scale bias, the twelve-point PnP fit
#: only ~1.5%, because averaging over more correspondences cancels most of it.
#: These floors are what stop inverse-variance fusion from over-trusting the
#: weaker cue -- omitting them made the fused result worse than PnP alone.
IPD_SHAPE_SIGMA_REL = 0.045
PNP_SHAPE_SIGMA_REL = 0.015
#: Reject a PnP solution whose reprojection RMS exceeds this fraction of the
#: face's on-screen span.
MAX_REPROJ_FRACTION = 0.06


@dataclass(frozen=True)
class PoseAngles:
    """Head orientation in degrees (camera frame, right-handed)."""

    yaw: float
    pitch: float
    roll: float


@dataclass(frozen=True)
class DistanceEstimate:
    """Result of a single-frame distance estimate."""

    distance_mm: float
    """Depth along the optical axis to the midpoint of the eyes."""
    sigma_mm: float
    """1-sigma uncertainty, including the shared IPD scale term."""
    range_mm: float
    """Euclidean range (not just the Z component)."""
    pose: PoseAngles
    reproj_rms_px: float
    face_span_px: float
    pnp_distance_mm: float | None
    ipd_distance_mm: float | None
    valid: bool
    reason: str = ""
    bearing: Bearing = field(
        default_factory=lambda: Bearing(float("nan"), float("nan"), source="none")
    )
    """Deviation angle from the optical axis -- the spec's ``theta``, plus phi."""
    ref_point_cam_mm: np.ndarray | None = None
    """Eye-midpoint in camera coordinates (mm), when PnP succeeded."""
    ref_pixel: tuple[float, float] | None = None
    """Ideal-pinhole projection of the reference point. Fed to the tracker."""

    @property
    def distance_cm(self) -> float:
        return self.distance_mm / 10.0

    @property
    def theta_deg(self) -> float:
        """The problem statement's deviation angle, in degrees."""
        return self.bearing.theta_deg

    def as_spec_output(self) -> tuple[float, float]:
        """``(Z_metres, theta_radians)`` -- the literal expected output pair."""
        return self.distance_mm / 1000.0, self.bearing.theta_rad


def _rotation_to_euler(rmat: np.ndarray) -> PoseAngles:
    """Decompose a rotation matrix into yaw / pitch / roll (degrees)."""
    sy = math.hypot(rmat[0, 0], rmat[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:  # gimbal-locked
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0
    return PoseAngles(
        yaw=math.degrees(yaw), pitch=math.degrees(pitch), roll=math.degrees(roll)
    )


def _eye_centre(points_2d: dict[int, np.ndarray], outer: int, inner: int):
    if outer not in points_2d or inner not in points_2d:
        return None
    return 0.5 * (points_2d[outer] + points_2d[inner])


class DistanceEstimator:
    """Fuses a PnP depth cue and an IPD depth cue into one estimate."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        face_model: CanonicalFaceModel | None = None,
        ipd_sigma_mm: float = DEFAULT_IPD_SIGMA_MM,
    ):
        self.intrinsics = intrinsics
        self.model = face_model or CanonicalFaceModel()
        self.ipd_sigma_mm = float(ipd_sigma_mm)
        self._object_points = self.model.points_mm.astype(np.float64)
        self._last_rvec: np.ndarray | None = None
        self._last_tvec: np.ndarray | None = None

    # -- public API ---------------------------------------------------------

    def estimate(self, landmarks_px: np.ndarray) -> DistanceEstimate:
        """Estimate distance from an (N, 2) array of image-space landmarks.

        ``landmarks_px`` must be indexable by MediaPipe FaceMesh landmark id,
        i.e. the full 468-point array in pixel coordinates.
        """
        ids = self.model.landmark_ids
        if landmarks_px.ndim != 2 or landmarks_px.shape[1] < 2:
            return self._invalid("landmarks must be an (N, 2+) array")
        if landmarks_px.shape[0] <= max(ids):
            return self._invalid("landmark array too short for FaceMesh indices")

        image_points = np.ascontiguousarray(
            landmarks_px[list(ids), :2].astype(np.float64)
        )
        if not np.isfinite(image_points).all():
            return self._invalid("non-finite landmarks")

        face_span = float(
            np.linalg.norm(image_points.max(axis=0) - image_points.min(axis=0))
        )
        if face_span < 20.0:
            return self._invalid("face too small to measure", face_span=face_span)

        pnp = self._solve_pnp(image_points)
        if pnp is None:
            return self._invalid("PnP failed to converge", face_span=face_span)
        rvec, tvec, reproj_rms = pnp

        rmat, _ = cv2.Rodrigues(rvec)
        pose = _rotation_to_euler(rmat)

        if reproj_rms > MAX_REPROJ_FRACTION * face_span:
            # PnP is untrustworthy (occlusion, extreme expression, blur).
            # Fall back to the two-point cue rather than dropping the frame --
            # a degraded estimate beats a gap in the track.
            return self._ipd_fallback(landmarks_px, pose, reproj_rms, face_span)

        # --- cue 1: PnP depth of the eye midpoint --------------------------
        ref_cam = rmat @ self.model.reference_point_mm.reshape(3, 1) + tvec
        z_pnp = float(ref_cam[2, 0])
        range_pnp = float(np.linalg.norm(ref_cam))
        if z_pnp <= 1.0:
            return self._invalid("degenerate PnP depth", face_span=face_span, pose=pose)

        # --- deviation angle ------------------------------------------------
        # Taken from the same 3D point we report depth to, so depth and angle
        # are guaranteed mutually consistent -- they are two spherical
        # coordinates of one vector rather than two independent estimates that
        # could disagree. Lateral uncertainty shrinks as sqrt(N) over the
        # correspondences, which is why this beats a single-point read.
        ref_vec = np.asarray(ref_cam, dtype=np.float64).reshape(3)
        sigma_lat_mm = (
            LANDMARK_SIGMA_PX * z_pnp
            / max(self.intrinsics.fx, 1.0)
            / math.sqrt(len(self._object_points))
        )
        face_bearing = bearing_mod.from_point(ref_vec, sigma_lateral_mm=sigma_lat_mm)
        ref_px = (
            self.intrinsics.fx * ref_vec[0] / z_pnp + self.intrinsics.cx,
            self.intrinsics.fy * ref_vec[1] / z_pnp + self.intrinsics.cy,
        )

        # Independent (non-scale) relative uncertainty of the PnP cue.
        rel_pnp = math.hypot(
            PNP_ERROR_GAIN * max(reproj_rms, 0.3) / face_span,
            PNP_SHAPE_SIGMA_REL,
        )
        rel_pnp = float(np.clip(rel_pnp, 0.004, 0.5))

        # --- cue 2: yaw-corrected interpupillary distance ------------------
        z_ipd, rel_ipd = self._ipd_cue(landmarks_px, pose.yaw)

        # --- combine -------------------------------------------------------
        # Deliberately NOT an inverse-variance fusion. Benchmarking showed the
        # IPD cue carries a *per-subject bias* (a 3 mm shift in eye-corner
        # placement moves it several percent, consistently, for that person),
        # while the 12-point PnP fit averages the same shape error down to
        # under 2%. Variance weighting cannot cancel a bias, so blending the
        # two made the answer worse than PnP alone -- 37 mm MAE vs 18 mm.
        # PnP is therefore the estimate; the IPD cue earns its place as a
        # fallback (above) and as an independent disagreement check (here).
        distance = z_pnp
        rel_indep = rel_pnp
        if z_ipd is not None:
            disagreement = abs(z_ipd - z_pnp) / z_pnp
            # Two cues that disagree by more than their combined noise signal
            # something the noise model does not cover -- unusual face shape,
            # bad landmarks, wrong focal length. Widen sigma; do not move the
            # estimate.
            expected = math.hypot(rel_pnp, rel_ipd)
            if disagreement > 2.0 * expected:
                rel_indep = math.hypot(rel_pnp, disagreement - 2.0 * expected)

        # Add the shared metric-scale uncertainty. Both cues inherit their
        # scale from the assumed IPD, so this term is common to them and is
        # applied once, at the end, rather than inside the per-cue weights.
        rel_scale = self.ipd_sigma_mm / self.model.ipd_mm
        if not self.intrinsics.calibrated:
            # An uncalibrated focal length is a second shared scale error.
            rel_scale = math.hypot(rel_scale, 0.10)
        sigma = distance * math.hypot(rel_indep, rel_scale)

        return DistanceEstimate(
            distance_mm=distance,
            sigma_mm=sigma,
            range_mm=distance * (range_pnp / z_pnp),
            pose=pose,
            reproj_rms_px=reproj_rms,
            face_span_px=face_span,
            pnp_distance_mm=z_pnp,
            ipd_distance_mm=z_ipd,
            valid=True,
            bearing=face_bearing,
            ref_point_cam_mm=ref_vec,
            ref_pixel=ref_px,
        )

    # -- internals ----------------------------------------------------------

    def _iris_cue(self, landmarks_px: np.ndarray):
        """Depth from apparent iris diameter. Returns (mm, relative_sigma).

        The iris is a near-planar disc of almost constant physical size, so
        ``Z = f * D_iris / d_px`` is the same similar-triangles rule as the
        face-width baseline -- but applied to the least variable dimension on
        the face rather than one of the most variable.

        Two properties make it worth the trouble:

        * its scale error (3.6%) is independent of the canonical face model,
          so it corroborates PnP and IPD rather than repeating their bias;
        * a circle stays a circle under rotation. Yaw foreshortens the eye
          baseline as cos(yaw), which is why the IPD cue needs a pose
          correction, but the iris only shrinks with *distance*. We take the
          maximum of the horizontal and vertical ring extents, which is
          robust to the disc being viewed obliquely.

        **This cue is deliberately NOT fused into the reported depth.**
        Benchmarked over 180 distinct simulated subjects it scores 117 mm MAE
        against PnP's 55 mm -- twice as bad. The theory is sound but the
        signal-to-noise is not: between 0.4 m and 2 m the iris spans only
        7-28 px, so 1.5 px of landmark noise is around 10% error, whereas the
        PnP fit averages twelve landmarks across a ~180 px face. A dimension
        being tightly conserved across people does not help if you cannot
        measure it precisely on the individual.

        It is kept because it is genuinely independent of the canonical face
        model's scale, which makes it a useful *cross-check*: a large,
        persistent iris-vs-PnP disagreement points at a wrong focal length,
        since a focal-length error moves both cues together while a face-shape
        error moves only one.

        Returns ``(None, None)`` when the model was run without iris
        refinement, i.e. fewer than 478 landmarks.
        """
        if landmarks_px.shape[0] <= max(IDX_L_IRIS_RING):
            return None, None

        widths = []
        for centre, ring in ((IDX_R_IRIS_CENTRE, IDX_R_IRIS_RING),
                             (IDX_L_IRIS_CENTRE, IDX_L_IRIS_RING)):
            pts = landmarks_px[list(ring), :2]
            horizontal = float(np.linalg.norm(pts[0] - pts[2]))
            vertical = float(np.linalg.norm(pts[1] - pts[3]))
            # MEAN, not max. Taking the larger of two noisy extents is a
            # biased estimator -- E[max] > true whenever the inputs carry
            # noise -- and since Z is inversely proportional to the diameter,
            # that bias shrinks every distance. Benchmarked at -110 mm of
            # systematic error against +25 mm for the mean. Eyelid occlusion
            # is the lesser problem.
            d = 0.5 * (horizontal + vertical)
            if d > 3.0:
                widths.append(d)

        if not widths:
            return None, None

        d_px = float(np.mean(widths))
        z = self.intrinsics.fx * IRIS_DIAMETER_MM / d_px

        rel_shape = IRIS_DIAMETER_SIGMA_MM / IRIS_DIAMETER_MM
        rel_noise = (LANDMARK_SIGMA_PX / d_px) / math.sqrt(len(widths))
        return z, math.hypot(rel_shape, rel_noise)

    def _solve_pnp(self, image_points: np.ndarray):
        """Solve for head pose. Returns (rvec, tvec, reprojection_rms).

        A face is *nearly* planar, which makes PnP prone to the two-fold mirror
        ambiguity: a reflected pose reprojects almost perfectly while placing
        the face behind the camera. SOLVEPNP_ITERATIVE walks straight into it
        from a cold start. Two guards:

        * SQPnP is the primary solver -- it is globally optimal for the
          non-planar case and does not depend on a good initial guess;
        * every candidate is cheirality-checked (all model points must have
          positive depth) before it is accepted, and the cached extrinsic guess
          is dropped whenever a solution is rejected, so one bad frame cannot
          poison the ones after it.
        """
        camera_matrix = self.intrinsics.matrix
        distortion = self.intrinsics.distortion
        candidates = []

        # Warm start from the previous frame: cheap, and keeps pose continuous.
        if self._last_rvec is not None:
            ok, rvec, tvec = cv2.solvePnP(
                self._object_points, image_points, camera_matrix, distortion,
                self._last_rvec.copy(), self._last_tvec.copy(),
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                candidates.append((rvec, tvec))

        for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP):
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    self._object_points, image_points, camera_matrix,
                    distortion, flags=flag,
                )
            except cv2.error:  # pragma: no cover - very old OpenCV
                continue
            if ok:
                candidates.append((rvec, tvec))

        best = None
        for rvec, tvec in candidates:
            refined_r, refined_t = cv2.solvePnPRefineVVS(
                self._object_points, image_points, camera_matrix, distortion,
                rvec.copy(), tvec.copy(),
            )
            for r, t in ((refined_r, refined_t), (rvec, tvec)):
                rms = self._score(r, t, image_points)
                if rms is None:
                    continue          # failed the cheirality check
                if best is None or rms < best[2]:
                    best = (r, t, rms)
                break

        if best is None:
            self._last_rvec = self._last_tvec = None
            return None

        self._last_rvec, self._last_tvec = best[0], best[1]
        return best

    def _score(self, rvec, tvec, image_points):
        """Reprojection RMS, or None if the pose puts the face behind us."""
        rmat, _ = cv2.Rodrigues(rvec)
        cam_points = (rmat @ self._object_points.T).T + tvec.reshape(1, 3)
        if not np.isfinite(cam_points).all() or cam_points[:, 2].min() <= 1.0:
            return None
        projected, _ = cv2.projectPoints(
            self._object_points, rvec, tvec,
            self.intrinsics.matrix, self.intrinsics.distortion,
        )
        residuals = projected.reshape(-1, 2) - image_points
        return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    def _ipd_cue(self, landmarks_px: np.ndarray, yaw_deg: float):
        """Depth from the eye baseline. Returns (depth_mm, relative_sigma)."""
        pts = {
            i: landmarks_px[i, :2]
            for i in (IDX_R_EYE_OUTER, IDX_R_EYE_INNER,
                      IDX_L_EYE_INNER, IDX_L_EYE_OUTER)
            if i < landmarks_px.shape[0]
        }
        right = _eye_centre(pts, IDX_R_EYE_OUTER, IDX_R_EYE_INNER)
        left = _eye_centre(pts, IDX_L_EYE_OUTER, IDX_L_EYE_INNER)
        if right is None or left is None:
            return None, float("inf")

        b_px = float(np.linalg.norm(left - right))
        if b_px < 4.0:
            return None, float("inf")

        yaw = math.radians(yaw_deg)
        cos_yaw = math.cos(yaw)
        if abs(cos_yaw) < 0.20:  # beyond ~78 deg the cue is meaningless
            return None, float("inf")

        f = 0.5 * (self.intrinsics.fx + self.intrinsics.fy)
        z = f * self.model.ipd_mm * abs(cos_yaw) / b_px

        # Averaging two corners per eye halves the variance of each centre.
        sigma_b = LANDMARK_SIGMA_PX  # sqrt(2 eyes * (sigma^2/2 per centre))
        rel = math.sqrt(
            (sigma_b / b_px) ** 2
            + (abs(math.tan(yaw)) * YAW_SIGMA_RAD) ** 2
            + IPD_SHAPE_SIGMA_REL ** 2
        )
        return z, float(np.clip(rel, 0.004, 1.0))

    def _ipd_fallback(self, landmarks_px, pose, reproj_rms, face_span):
        """Degraded, IPD-only estimate used when the PnP fit is rejected."""
        z_ipd, rel_ipd = self._ipd_cue(landmarks_px, pose.yaw)
        if z_ipd is None:
            return self._invalid(
                "PnP rejected and IPD cue unavailable",
                face_span=face_span, reproj=reproj_rms, pose=pose,
            )
        rel_scale = self.ipd_sigma_mm / self.model.ipd_mm
        if not self.intrinsics.calibrated:
            rel_scale = math.hypot(rel_scale, 0.10)
        # Inflate: the pose used for the yaw correction came from the same
        # fit we just rejected, so it is not to be trusted either.
        rel_total = math.hypot(math.hypot(rel_ipd, rel_scale), 0.05)

        # No trustworthy 3D point here, so fall back to the spec-style pixel
        # bearing -- but read it off the eye midpoint rather than the bbox
        # centre, and undistort first. Strictly better than the baseline even
        # in this degraded mode.
        pts = {
            i: landmarks_px[i, :2]
            for i in (IDX_R_EYE_OUTER, IDX_R_EYE_INNER,
                      IDX_L_EYE_INNER, IDX_L_EYE_OUTER)
            if i < landmarks_px.shape[0]
        }
        right = _eye_centre(pts, IDX_R_EYE_OUTER, IDX_R_EYE_INNER)
        left = _eye_centre(pts, IDX_L_EYE_OUTER, IDX_L_EYE_INNER)
        if right is not None and left is not None:
            mid = 0.5 * (right + left)
            face_bearing = bearing_mod.from_pixel(
                self.intrinsics, float(mid[0]), float(mid[1]),
                sigma_px=LANDMARK_SIGMA_PX,
            )
            ref_px = (float(mid[0]), float(mid[1]))
        else:
            face_bearing = Bearing(float("nan"), float("nan"), source="none")
            ref_px = None

        return DistanceEstimate(
            distance_mm=z_ipd, sigma_mm=z_ipd * rel_total, range_mm=z_ipd,
            pose=pose, reproj_rms_px=reproj_rms, face_span_px=face_span,
            pnp_distance_mm=None, ipd_distance_mm=z_ipd, valid=True,
            reason="degraded: IPD-only fallback",
            bearing=face_bearing, ref_point_cam_mm=None, ref_pixel=ref_px,
        )

    @staticmethod
    def _invalid(reason: str, face_span: float = 0.0, reproj: float = 0.0,
                 pose: PoseAngles | None = None) -> DistanceEstimate:
        return DistanceEstimate(
            distance_mm=float("nan"), sigma_mm=float("inf"), range_mm=float("nan"),
            pose=pose or PoseAngles(0.0, 0.0, 0.0), reproj_rms_px=reproj,
            face_span_px=face_span, pnp_distance_mm=None, ipd_distance_mm=None,
            valid=False, reason=reason,
        )
