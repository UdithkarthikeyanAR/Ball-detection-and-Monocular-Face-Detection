"""Deviation angle of the face from the optical axis.

The problem statement asks for ``theta = arctan((x - c_x) / f)``: the angle
between the optical axis and the ray through the face centre pixel, measured in
the horizontal plane. That formula is exactly ``atan2(X, Z)`` of the ray's
direction in the camera frame, so this module implements the general 3D form and
recovers the stated formula as a special case -- see :func:`from_pixel`, which is
bit-for-bit the spec equation when distortion is zero.

Two improvements are layered on top, both of which keep the same output type:

**1. Undistortion.** The spec formula is only valid for an ideal pinhole. A real
webcam displaces edge pixels radially by 5-15 px, i.e. 1-2 degrees of angular
error, and it is worst exactly where the deviation angle is largest. Points are
undistorted before the arctangent is taken.

**2. An anatomically fixed reference point.** ``x`` in the spec is "face centre
pixel", normally the centre of the detector's bounding box. That box is defined
by the visible silhouette, so when the head yaws by 30 degrees the box centre
slides toward the near cheek -- several pixels of movement with the head
physically stationary. Using the PnP-recovered midpoint of the eyes instead
removes that coupling, because it is the projection of one fixed point on the
skull. :func:`from_point` does this.

Sign conventions (camera frame is OpenCV: +x right, +y down, +z into scene):

    theta (azimuth)   > 0  =>  face is to the RIGHT of the optical axis
                               in the image, i.e. camera should pan right
    phi   (elevation) > 0  =>  face is ABOVE the optical axis
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .camera import CameraIntrinsics

__all__ = ["Bearing", "from_pixel", "from_point", "angular_separation_deg"]


@dataclass(frozen=True)
class Bearing:
    """Direction to the face relative to the optical axis."""

    theta_deg: float
    """Azimuth -- the problem statement's deviation angle. +ve to image right."""
    phi_deg: float
    """Elevation. +ve upward. Not required by the spec; reported as a bonus."""
    sigma_theta_deg: float = float("nan")
    sigma_phi_deg: float = float("nan")
    source: str = "pixel"
    """Which cue produced it: 'pixel' (spec-style) or 'pnp' (pose-invariant)."""

    @property
    def theta_rad(self) -> float:
        return math.radians(self.theta_deg)

    @property
    def phi_rad(self) -> float:
        return math.radians(self.phi_deg)

    @property
    def total_deg(self) -> float:
        """Total off-axis angle, combining azimuth and elevation."""
        return math.degrees(
            math.acos(
                float(np.clip(
                    math.cos(self.theta_rad) * math.cos(self.phi_rad), -1.0, 1.0
                ))
            )
        )

    def unit_vector(self) -> np.ndarray:
        """Unit direction in the camera frame."""
        t, p = self.theta_rad, self.phi_rad
        return np.array([
            math.sin(t) * math.cos(p),
            -math.sin(p),
            math.cos(t) * math.cos(p),
        ], dtype=np.float64)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (f"Bearing(theta={self.theta_deg:+.2f}deg, "
                f"phi={self.phi_deg:+.2f}deg, source={self.source!r})")


def _angles_from_xyz(x: float, y: float, z: float) -> tuple[float, float]:
    """Azimuth and elevation (radians) of a camera-frame direction."""
    theta = math.atan2(x, z)
    phi = math.atan2(-y, math.hypot(x, z))
    return theta, phi


def from_pixel(
    intrinsics: CameraIntrinsics,
    u: float,
    v: float,
    undistort: bool = True,
    sigma_px: float = 1.5,
) -> Bearing:
    """Bearing of the ray through pixel ``(u, v)``.

    With ``undistort=False`` and ``fx == fy`` this is precisely the problem
    statement's ``theta = arctan((x - c_x) / f)``.

    Uncertainty is propagated analytically: d(theta)/du = 1 / (fx * (1 + x^2))
    where x is the normalised coordinate, so the angular precision degrades
    toward the frame edge.
    """
    x, y = intrinsics.normalised(u, v, undistort=undistort)
    theta, phi = _angles_from_xyz(x, y, 1.0)

    d_theta = sigma_px / (intrinsics.fx * (1.0 + x * x))
    r = math.hypot(x, 1.0)
    d_phi = sigma_px / (intrinsics.fy * (1.0 + (y / r) ** 2) * r)

    return Bearing(
        theta_deg=math.degrees(theta),
        phi_deg=math.degrees(phi),
        sigma_theta_deg=math.degrees(d_theta),
        sigma_phi_deg=math.degrees(d_phi),
        source="pixel",
    )


def from_point(
    point_cam_mm: np.ndarray,
    sigma_lateral_mm: float = 3.0,
) -> Bearing:
    """Bearing of a 3D point already expressed in the camera frame.

    This is the preferred path: ``point_cam_mm`` is the PnP-recovered midpoint of
    the eyes, a point fixed to the skull, so the reported angle does not drift
    when the subject turns their head. It also needs no undistortion step --
    distortion was already removed by the PnP solve, which is given the
    distortion coefficients directly.
    """
    p = np.asarray(point_cam_mm, dtype=np.float64).reshape(3)
    if not np.isfinite(p).all() or p[2] <= 1e-6:
        return Bearing(float("nan"), float("nan"), source="pnp")

    theta, phi = _angles_from_xyz(float(p[0]), float(p[1]), float(p[2]))
    depth = float(p[2])
    d_ang = math.degrees(sigma_lateral_mm / max(depth, 1.0))

    return Bearing(
        theta_deg=math.degrees(theta),
        phi_deg=math.degrees(phi),
        sigma_theta_deg=d_ang,
        sigma_phi_deg=d_ang,
        source="pnp",
    )


def angular_separation_deg(a: Bearing, b: Bearing) -> float:
    """Great-circle angle between two bearings, in degrees."""
    dot = float(np.clip(np.dot(a.unit_vector(), b.unit_vector()), -1.0, 1.0))
    return math.degrees(math.acos(dot))
