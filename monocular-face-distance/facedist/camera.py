"""Camera intrinsics: load from calibration, or fall back to an FOV guess."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

#: Typical horizontal field of view for laptop / USB webcams, in degrees.
DEFAULT_HFOV_DEG = 60.0


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics in pixels, plus distortion coefficients."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    #: True when fx/fy came from a real calibration rather than an FOV guess.
    calibrated: bool = False

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        hfov_deg: float = DEFAULT_HFOV_DEG,
    ) -> "CameraIntrinsics":
        """Approximate intrinsics from image size and an assumed horizontal FOV.

        Accurate to roughly the accuracy of the FOV assumption -- typically
        10-20% for an unknown webcam, which maps directly onto 10-20% depth
        error. Run ``tools/calibrate_camera.py`` to remove this term.
        """
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height, calibrated=False)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def distortion(self) -> np.ndarray:
        return np.array(self.dist_coeffs, dtype=np.float64).reshape(-1, 1)

    def rescaled(self, width: int, height: int) -> "CameraIntrinsics":
        """Return intrinsics for the same lens at a different capture size."""
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx, fy=self.fy * sy,
            cx=self.cx * sx, cy=self.cy * sy,
            width=width, height=height,
            dist_coeffs=self.dist_coeffs, calibrated=self.calibrated,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CameraIntrinsics":
        data = json.loads(Path(path).read_text())
        data["dist_coeffs"] = tuple(data.get("dist_coeffs", (0.0,) * 5))
        return cls(**data)

    # -- ray geometry -------------------------------------------------------

    @property
    def has_distortion(self) -> bool:
        return any(abs(c) > 1e-12 for c in self.dist_coeffs)

    def undistort_points(self, pts_px: np.ndarray) -> np.ndarray:
        """Map distorted pixel coords to ideal-pinhole pixel coords.

        The problem statement's angle formula assumes an ideal pinhole. A real
        lens displaces points radially -- on a typical webcam by 5-15 px near
        the frame edge, which is 1-2 degrees of angular error exactly where the
        deviation angle matters most. Undistorting first makes the pinhole
        formula valid again.
        """
        import cv2  # local import keeps camera.py importable without cv2

        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        if not self.has_distortion:
            return pts.reshape(-1, 2)
        out = cv2.undistortPoints(pts, self.matrix, self.distortion,
                                  P=self.matrix)
        return np.asarray(out, dtype=np.float64).reshape(-1, 2)

    def normalised(self, u: float, v: float, undistort: bool = True):
        """Pixel -> normalised image coords (X/Z, Y/Z) on the z=1 plane."""
        if undistort and self.has_distortion:
            u, v = self.undistort_points(np.array([[u, v]], dtype=np.float64))[0]
        return (u - self.cx) / self.fx, (v - self.cy) / self.fy

    def ray(self, u: float, v: float, undistort: bool = True) -> np.ndarray:
        """Unit direction vector through pixel (u, v) in the camera frame.

        Camera frame is OpenCV convention: +x right, +y down, +z into scene.
        """
        x, y = self.normalised(u, v, undistort)
        d = np.array([x, y, 1.0], dtype=np.float64)
        return d / np.linalg.norm(d)

    @property
    def hfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.width / 2.0) / self.fx))

    @property
    def vfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.height / 2.0) / self.fy))


def resolve_intrinsics(
    width: int,
    height: int,
    calibration_path: str | Path | None = None,
    hfov_deg: float = DEFAULT_HFOV_DEG,
) -> CameraIntrinsics:
    """Load calibration if present and usable, else derive from FOV."""
    if calibration_path is not None and Path(calibration_path).exists():
        intr = CameraIntrinsics.load(calibration_path)
        if (intr.width, intr.height) != (width, height):
            intr = intr.rescaled(width, height)
        return intr
    return CameraIntrinsics.from_fov(width, height, hfov_deg)
