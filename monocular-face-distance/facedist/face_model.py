"""Canonical metric 3D face model used for PnP-based depth recovery.

The coordinates below are anthropometric *means* for adult faces, expressed in
millimetres in a face-centred frame:

    origin : nasion (bridge of the nose, between the eyes)
    +X     : towards the image right  (i.e. the subject's LEFT)
    +Y     : down
    +Z     : away from the camera, into the scene (OpenCV convention)

IMPORTANT
---------
These numbers are population means, not your face. The *shape* error they carry
is second-order; the *scale* error is first-order and is what actually dominates
monocular distance estimates. We therefore never use the raw model scale: the
model is rescaled at load time so that its own interpupillary distance matches
the user's IPD (calibrated, or the adult mean of 63 mm). After that rescaling,
scale error in the depth estimate is approximately equal to the relative error
in the assumed IPD -- roughly 5% uncalibrated, <1% calibrated.
"""

from __future__ import annotations

import numpy as np

# --- MediaPipe FaceMesh landmark indices ------------------------------------
# These are the stable, widely-used indices of the 468-point FaceMesh topology.
IDX_NASION = 168
IDX_NOSE_TIP = 1
IDX_R_EYE_OUTER = 33     # subject's right eye, outer canthus
IDX_R_EYE_INNER = 133
IDX_L_EYE_INNER = 362
IDX_L_EYE_OUTER = 263
IDX_MOUTH_R = 61
IDX_MOUTH_L = 291
IDX_CHIN = 152
IDX_FOREHEAD = 10
IDX_FACE_R = 234         # right face silhouette (tragion-ish)
IDX_FACE_L = 454

# Iris landmarks. Present only when MediaPipe runs with refine_landmarks=True
# (or the Tasks FaceLandmarker, which always emits them) -- indices 468-477.
IDX_R_IRIS_CENTRE = 468
IDX_R_IRIS_RING = (469, 470, 471, 472)   # right, top, left, bottom
IDX_L_IRIS_CENTRE = 473
IDX_L_IRIS_RING = (474, 475, 476, 477)

#: Horizontal visible iris diameter (HVID), adult population mean, in mm.
#: This is the single most tightly-conserved dimension on the human face:
#: 11.71 mm with SD 0.42 mm, a coefficient of variation of 3.6%. For
#: comparison the interpupillary distance spreads 5.6% and bizygomatic
#: width about 5%. Because depth from a known length scales linearly with
#: that length, the iris cue therefore carries roughly *half* the
#: systematic error of the IPD cue -- before any per-subject calibration.
#:
#: The trade is resolution: at 60 cm on a 900 px focal length the iris spans
#: only ~18 px against the IPD's ~95 px, so a 1 px landmark error is 5.7%
#: here versus 1.1% there. Low bias, high variance -- which is precisely the
#: profile that makes it worth *fusing* rather than substituting.
IRIS_DIAMETER_MM = 11.71
IRIS_DIAMETER_SIGMA_MM = 0.42

# Ordered list of the landmarks we feed to solvePnP.
PNP_LANDMARK_IDS: tuple[int, ...] = (
    IDX_NASION,
    IDX_NOSE_TIP,
    IDX_R_EYE_OUTER,
    IDX_R_EYE_INNER,
    IDX_L_EYE_INNER,
    IDX_L_EYE_OUTER,
    IDX_MOUTH_R,
    IDX_MOUTH_L,
    IDX_CHIN,
    IDX_FOREHEAD,
    IDX_FACE_R,
    IDX_FACE_L,
)

# Metric model points (mm), same order as PNP_LANDMARK_IDS.
_MODEL_MM = np.array(
    [
        [0.0,    0.0,    0.0],     # nasion
        [0.0,   33.0,   22.0],     # nose tip
        [-45.0, -5.0,  -22.0],     # right eye outer
        [-17.0, -3.0,  -14.0],     # right eye inner
        [17.0,  -3.0,  -14.0],     # left eye inner
        [45.0,  -5.0,  -22.0],     # left eye outer
        [-24.0, 66.0,   -8.0],     # mouth right
        [24.0,  66.0,   -8.0],     # mouth left
        [0.0,  108.0,  -18.0],     # chin
        [0.0,  -75.0,  -12.0],     # forehead
        [-72.0, 12.0,  -60.0],     # right face edge
        [72.0,  12.0,  -60.0],     # left face edge
    ],
    dtype=np.float64,
)

# Row offsets into _MODEL_MM for the four eye-corner points.
_ROW_R_OUTER, _ROW_R_INNER, _ROW_L_INNER, _ROW_L_OUTER = 2, 3, 4, 5

#: Population mean adult interpupillary distance (mm). Dodgson (2004) reports
#: a mean near 63 mm with a standard deviation of about 3.5 mm across adults.
DEFAULT_IPD_MM = 63.0
DEFAULT_IPD_SIGMA_MM = 3.5

#: Row index of the point we report distance to (midpoint of the eyes is
#: computed separately; nasion is the closest single vertex to it).
REFERENCE_ROW = 0


def _model_ipd(points_mm: np.ndarray) -> float:
    """Interpupillary distance implied by a model array (mm)."""
    right_centre = 0.5 * (points_mm[_ROW_R_OUTER] + points_mm[_ROW_R_INNER])
    left_centre = 0.5 * (points_mm[_ROW_L_OUTER] + points_mm[_ROW_L_INNER])
    return float(np.linalg.norm(left_centre - right_centre))


class CanonicalFaceModel:
    """Metric 3D face model, rescaled to a known interpupillary distance."""

    def __init__(self, ipd_mm: float = DEFAULT_IPD_MM):
        if not 40.0 <= ipd_mm <= 85.0:
            raise ValueError(f"implausible IPD: {ipd_mm} mm (expected 40-85)")
        self.ipd_mm = float(ipd_mm)
        base_ipd = _model_ipd(_MODEL_MM)
        self.scale = self.ipd_mm / base_ipd
        self.points_mm: np.ndarray = np.ascontiguousarray(_MODEL_MM * self.scale)
        self.landmark_ids: tuple[int, ...] = PNP_LANDMARK_IDS

    @property
    def eye_centres_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """(right, left) 3D eye-centre positions in the model frame."""
        p = self.points_mm
        right = 0.5 * (p[_ROW_R_OUTER] + p[_ROW_R_INNER])
        left = 0.5 * (p[_ROW_L_OUTER] + p[_ROW_L_INNER])
        return right, left

    @property
    def reference_point_mm(self) -> np.ndarray:
        """Point whose depth we report: the midpoint between the eye centres."""
        right, left = self.eye_centres_mm
        return 0.5 * (right + left)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"CanonicalFaceModel(ipd_mm={self.ipd_mm:.1f}, scale={self.scale:.4f})"
