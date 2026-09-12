"""Temporal filtering of face position and distance.

Filtering is done in **inverse depth** ``q = 1/Z`` rather than in ``Z``. Under a
pinhole camera the measured quantity is an image-space length, and the mapping
to depth is ``Z = fB/b``; propagating pixel noise through it gives
``sigma_Z ~ Z^2``, i.e. wildly heteroscedastic. In ``q`` the same noise is
almost constant, so a single fixed process-noise setting works at 30 cm and at
3 m alike. This is the standard inverse-depth parameterisation from monocular
SLAM, applied here to a much smaller problem.

Measurement noise is *adaptive*: the per-frame sigma produced by the estimator
is fed straight into ``R``, so frames with a small or badly-fitted face are
automatically trusted less instead of yanking the filter around.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Scale used internally: q is stored in 1/metre, so q = 1000 / Z_mm.
_MM_PER_M = 1000.0


@dataclass
class TrackState:
    """Public, smoothed view of one tracked face."""

    track_id: int
    x_px: float
    y_px: float
    distance_mm: float
    sigma_mm: float
    velocity_mm_s: float
    hits: int
    misses: int
    confirmed: bool
    sigma_x_px: float = 0.0
    sigma_y_px: float = 0.0


class _InverseDepthKalman:
    """Constant-velocity Kalman filter over (x_px, y_px, q_per_m)."""

    def __init__(self, x_px: float, y_px: float, distance_mm: float,
                 sigma_mm: float):
        q = _MM_PER_M / max(distance_mm, 1.0)
        sigma_q = self._sigma_q(distance_mm, sigma_mm)

        self.x = np.array([x_px, y_px, q, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag(
            [16.0, 16.0, sigma_q ** 2, 400.0, 400.0, (4.0 * sigma_q) ** 2]
        )
        # Continuous-time process noise densities (per second).
        self._pos_accel_var = 4.0e4   # px^2 / s^3
        self._q_accel_var = 4.0        # (1/m)^2 / s^3

    @staticmethod
    def _sigma_q(distance_mm: float, sigma_mm: float) -> float:
        """Propagate a depth sigma into inverse-depth sigma (1/m)."""
        z_m = max(distance_mm, 1.0) / _MM_PER_M
        s_m = max(sigma_mm, 1.0) / _MM_PER_M
        return s_m / (z_m ** 2)

    def predict(self, dt: float) -> None:
        dt = float(np.clip(dt, 1e-3, 0.5))
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt

        # Piecewise-white-noise acceleration model.
        q_pos = self._pos_accel_var
        q_inv = self._q_accel_var
        t3, t2 = dt ** 3 / 3.0, dt ** 2 / 2.0
        Q = np.zeros((6, 6))
        for i, var in ((0, q_pos), (1, q_pos), (2, q_inv)):
            j = i + 3
            Q[i, i] = var * t3
            Q[i, j] = Q[j, i] = var * t2
            Q[j, j] = var * dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, x_px: float, y_px: float, distance_mm: float,
               sigma_mm: float, gate: float = 9.0) -> bool:
        """Fuse a measurement. Returns False if it was gated out as an outlier."""
        z = np.array(
            [x_px, y_px, _MM_PER_M / max(distance_mm, 1.0)], dtype=np.float64
        )
        sigma_q = self._sigma_q(distance_mm, sigma_mm)
        R = np.diag([4.0, 4.0, max(sigma_q, 1e-4) ** 2])

        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:  # pragma: no cover - numerically rare
            return False

        # Chi-square gate on 3 DOF; rejects teleporting detections.
        if float(y @ S_inv @ y) > gate * 3.0:
            return False

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        # Joseph form: stays positive-definite under adaptive R.
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return True

    @property
    def distance_mm(self) -> float:
        q = self.x[2]
        return _MM_PER_M / q if q > 1e-9 else float("inf")

    @property
    def sigma_mm(self) -> float:
        q = max(self.x[2], 1e-9)
        z_m = 1.0 / q
        return float(np.sqrt(max(self.P[2, 2], 0.0)) * z_m ** 2 * _MM_PER_M)

    @property
    def sigma_x_px(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], 0.0)))

    @property
    def sigma_y_px(self) -> float:
        return float(np.sqrt(max(self.P[1, 1], 0.0)))

    @property
    def velocity_mm_s(self) -> float:
        """Radial speed: dZ/dt = -q_dot / q^2, converted to mm/s."""
        q = max(self.x[2], 1e-9)
        return float(-self.x[5] / (q ** 2) * _MM_PER_M)


@dataclass
class _Track:
    track_id: int
    kf: _InverseDepthKalman
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


class FaceTracker:
    """Greedy IoU-associated multi-face tracker over inverse-depth filters."""

    def __init__(self, max_misses: int = 8, min_hits: int = 3,
                 iou_threshold: float = 0.25):
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[dict], dt: float) -> list[TrackState]:
        """Advance all tracks by ``dt`` and fuse this frame's detections.

        Each detection is a dict with keys ``bbox`` (x1, y1, x2, y2),
        ``centre`` (x, y), ``distance_mm`` and ``sigma_mm``.
        """
        for track in self._tracks:
            track.kf.predict(dt)

        unmatched = list(range(len(detections)))
        for track in self._tracks:
            best_j, best_iou = -1, self.iou_threshold
            for j in unmatched:
                score = _iou(track.bbox, detections[j]["bbox"])
                if score > best_iou:
                    best_j, best_iou = j, score
            if best_j < 0:
                track.misses += 1
                continue

            det = detections[best_j]
            accepted = track.kf.update(
                det["centre"][0], det["centre"][1],
                det["distance_mm"], det["sigma_mm"],
            )
            track.bbox = det["bbox"]
            unmatched.remove(best_j)
            if accepted:
                track.hits += 1
                track.misses = 0
                if track.hits >= self.min_hits:
                    track.confirmed = True
            else:
                track.misses += 1

        for j in unmatched:
            det = detections[j]
            self._tracks.append(
                _Track(
                    track_id=self._next_id,
                    kf=_InverseDepthKalman(
                        det["centre"][0], det["centre"][1],
                        det["distance_mm"], det["sigma_mm"],
                    ),
                    bbox=det["bbox"],
                )
            )
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]
        return [
            TrackState(
                track_id=t.track_id,
                x_px=float(t.kf.x[0]), y_px=float(t.kf.x[1]),
                distance_mm=t.kf.distance_mm, sigma_mm=t.kf.sigma_mm,
                velocity_mm_s=t.kf.velocity_mm_s,
                hits=t.hits, misses=t.misses, confirmed=t.confirmed,
                sigma_x_px=t.kf.sigma_x_px, sigma_y_px=t.kf.sigma_y_px,
            )
            for t in self._tracks
        ]

    def reset(self) -> None:
        self._tracks.clear()
