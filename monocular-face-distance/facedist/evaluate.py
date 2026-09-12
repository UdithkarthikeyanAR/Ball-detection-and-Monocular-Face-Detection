"""Quantitative comparison: fused estimator vs. bounding-box baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bearing as bearing_mod
from .baseline import BaselineWidthEstimator
from .camera import CameraIntrinsics
from .estimator import DistanceEstimator
from .face_model import CanonicalFaceModel
from .simulation import FaceSimulator
from .tracking import FaceTracker


@dataclass
class Metrics:
    mae_mm: float
    rmse_mm: float
    mape_pct: float
    p95_abs_err_mm: float
    n: int

    def __str__(self) -> str:
        return (
            f"MAE {self.mae_mm:7.1f} mm | RMSE {self.rmse_mm:7.1f} mm | "
            f"MAPE {self.mape_pct:5.2f} % | p95 {self.p95_abs_err_mm:7.1f} mm "
            f"| n={self.n}"
        )


def _metrics(pred: np.ndarray, truth: np.ndarray) -> Metrics:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(pred)
    pred, truth = pred[ok], truth[ok]
    err = pred - truth
    return Metrics(
        mae_mm=float(np.mean(np.abs(err))),
        rmse_mm=float(np.sqrt(np.mean(err ** 2))),
        mape_pct=float(np.mean(np.abs(err / truth)) * 100.0),
        p95_abs_err_mm=float(np.percentile(np.abs(err), 95)),
        n=int(pred.size),
    )


@dataclass
class AngleMetrics:
    mae_deg: float
    rmse_deg: float
    p95_abs_err_deg: float
    max_abs_err_deg: float
    n: int

    def __str__(self) -> str:
        return (
            f"MAE {self.mae_deg:6.3f} deg | RMSE {self.rmse_deg:6.3f} deg | "
            f"p95 {self.p95_abs_err_deg:6.3f} deg | max {self.max_abs_err_deg:6.3f} deg "
            f"| n={self.n}"
        )


def _angle_metrics(pred: np.ndarray, truth: np.ndarray) -> AngleMetrics:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(pred)
    pred, truth = pred[ok], truth[ok]
    err = pred - truth
    return AngleMetrics(
        mae_deg=float(np.mean(np.abs(err))),
        rmse_deg=float(np.sqrt(np.mean(err ** 2))),
        p95_abs_err_deg=float(np.percentile(np.abs(err), 95)),
        max_abs_err_deg=float(np.max(np.abs(err))),
        n=int(pred.size),
    )


#: Barrel distortion typical of a consumer webcam. Both methods are rendered
#: through it; only ours is told about it.
WEBCAM_DISTORTION = (-0.25, 0.08, 0.0005, -0.0004, 0.0)


def run_angle_benchmark(
    n_samples: int = 4000,
    max_theta_deg: float = 22.0,
    max_phi_deg: float = 12.0,
    max_yaw_deg: float = 40.0,
    distortion: tuple[float, ...] = WEBCAM_DISTORTION,
    seed: int = 0,
) -> dict[str, AngleMetrics]:
    """Deviation-angle accuracy: spec formula vs. this system.

    Both estimators receive the *same* rendered pixels, from a camera with
    realistic barrel distortion. The baseline does what the problem statement
    specifies -- ``arctan((x - c_x) / f)`` on the bounding-box centre, no
    undistortion. Ours reads the bearing off the PnP-recovered eye midpoint,
    with the distortion coefficients handed to the solver.

    The gap comes from two independent sources, reported separately below:
    lens distortion (a systematic radial bias, worst at the frame edge) and
    bounding-box drift under head yaw (the box centre slides toward the near
    cheek while the head stays put).
    """
    intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
    intr.calibrated = True
    intr.dist_coeffs = tuple(distortion)

    sim = FaceSimulator(intr, seed=seed)
    est = DistanceEstimator(intr, CanonicalFaceModel())
    base = BaselineWidthEstimator(intr)

    rng = np.random.default_rng(seed + 7)
    truth_t, base_t, ours_t = [], [], []
    truth_p, ours_p = [], []
    base_nodist_t = []

    for _ in range(n_samples):
        z = float(rng.uniform(400.0, 2500.0))
        theta = float(rng.uniform(-max_theta_deg, max_theta_deg))
        phi = float(rng.uniform(-max_phi_deg, max_phi_deg))
        yaw = float(rng.uniform(-max_yaw_deg, max_yaw_deg))
        pitch = float(rng.uniform(-15.0, 15.0))
        roll = float(rng.uniform(-15.0, 15.0))

        frame = sim.render(z, yaw, pitch, roll, theta_deg=theta, phi_deg=phi)
        result = est.estimate(frame.landmarks_px)
        if not result.valid or not np.isfinite(result.bearing.theta_deg):
            continue

        bbox = sim.bbox_of(frame.landmarks_px)
        truth_t.append(theta)
        truth_p.append(phi)
        base_t.append(base.angle(bbox))
        ours_t.append(result.bearing.theta_deg)
        ours_p.append(result.bearing.phi_deg)

        # Ablation: the same bbox-centre rule, but undistorted first. Isolates
        # how much of the baseline's error is distortion vs. box drift.
        cx_px, cy_px = sim.bbox_centre(bbox)
        base_nodist_t.append(
            bearing_mod.from_pixel(intr, cx_px, cy_px, undistort=True).theta_deg
        )

    truth_t = np.array(truth_t)
    return {
        "baseline_theta": _angle_metrics(np.array(base_t), truth_t),
        "bbox_undistorted_theta": _angle_metrics(np.array(base_nodist_t), truth_t),
        "proposed_theta": _angle_metrics(np.array(ours_t), truth_t),
        "proposed_phi": _angle_metrics(np.array(ours_p), np.array(truth_p)),
    }


def run_static_benchmark(
    n_samples: int = 4000,
    distances_mm: tuple[float, float] = (300.0, 2500.0),
    max_yaw_deg: float = 40.0,
    true_ipd_mm: float = 63.0,
    assumed_ipd_mm: float = 63.0,
    seed: int = 0,
) -> dict[str, Metrics]:
    """Per-frame accuracy over randomised distance and pose (no tracking)."""
    intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
    intr.calibrated = True
    sim = FaceSimulator(intr, true_ipd_mm=true_ipd_mm, seed=seed)
    est = DistanceEstimator(intr, CanonicalFaceModel(ipd_mm=assumed_ipd_mm))
    base = BaselineWidthEstimator(intr)

    rng = np.random.default_rng(seed + 1)
    truth, fused, pnp_only, ipd_only, baseline = [], [], [], [], []

    for _ in range(n_samples):
        z = float(rng.uniform(*distances_mm))
        yaw = float(rng.uniform(-max_yaw_deg, max_yaw_deg))
        pitch = float(rng.uniform(-15.0, 15.0))
        roll = float(rng.uniform(-15.0, 15.0))

        frame = sim.render(z, yaw, pitch, roll)
        result = est.estimate(frame.landmarks_px)
        if not result.valid:
            continue

        truth.append(z)
        fused.append(result.distance_mm)
        pnp_only.append(result.pnp_distance_mm)
        ipd_only.append(result.ipd_distance_mm if result.ipd_distance_mm else np.nan)
        baseline.append(base.estimate(sim.bbox_of(frame.landmarks_px)))

    truth = np.array(truth)
    return {
        "baseline_bbox": _metrics(np.array(baseline), truth),
        "pnp_only": _metrics(np.array(pnp_only, dtype=float), truth),
        "ipd_only": _metrics(np.array(ipd_only, dtype=float), truth),
        "proposed": _metrics(np.array(fused), truth),
    }


def run_temporal_benchmark(
    n_frames: int = 900,
    fps: float = 30.0,
    seed: int = 0,
) -> dict[str, Metrics]:
    """Accuracy on a moving subject, with and without the Kalman filter."""
    intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
    intr.calibrated = True
    sim = FaceSimulator(intr, seed=seed)
    est = DistanceEstimator(intr, CanonicalFaceModel())
    tracker = FaceTracker(min_hits=1)
    dt = 1.0 / fps

    truth, raw, smoothed = [], [], []
    for i in range(n_frames):
        t = i * dt
        # Subject walks from 2 m to 0.5 m and back, turning their head as they go.
        z = 1250.0 + 750.0 * np.cos(2.0 * np.pi * t / 12.0)
        yaw = 25.0 * np.sin(2.0 * np.pi * t / 3.5)

        frame = sim.render(float(z), float(yaw), 0.0, 0.0)
        result = est.estimate(frame.landmarks_px)
        if not result.valid:
            continue

        bbox = sim.bbox_of(frame.landmarks_px)
        centre = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        tracks = tracker.update(
            [{"bbox": bbox, "centre": centre,
              "distance_mm": result.distance_mm, "sigma_mm": result.sigma_mm}],
            dt,
        )
        if not tracks:
            continue

        truth.append(z)
        raw.append(result.distance_mm)
        smoothed.append(tracks[0].distance_mm)

    truth = np.array(truth)
    return {
        "proposed_unsmoothed": _metrics(np.array(raw), truth),
        "proposed_kalman": _metrics(np.array(smoothed), truth),
    }


def main() -> None:  # pragma: no cover - CLI entry
    print("=" * 78)
    print("STATIC BENCHMARK  (random distance 0.3-2.5 m, yaw +/-40 deg)")
    print("=" * 78)
    for name, m in run_static_benchmark().items():
        print(f"{name:>18}  {m}")

    print()
    print("=" * 78)
    print("STATIC BENCHMARK  (subject IPD 68 mm, estimator assumes 63 mm)")
    print("=" * 78)
    for name, m in run_static_benchmark(true_ipd_mm=68.0).items():
        print(f"{name:>18}  {m}")

    print()
    print("=" * 78)
    print("STATIC BENCHMARK  (same 68 mm subject, after IPD calibration)")
    print("=" * 78)
    for name, m in run_static_benchmark(
        true_ipd_mm=68.0, assumed_ipd_mm=68.0
    ).items():
        print(f"{name:>18}  {m}")

    print()
    print("=" * 78)
    print("ANGLE BENCHMARK  (theta +/-22 deg, yaw +/-40 deg, barrel distortion)")
    print("=" * 78)
    for name, m in run_angle_benchmark().items():
        print(f"{name:>24}  {m}")

    print()
    print("=" * 78)
    print("TEMPORAL BENCHMARK  (30 s of walking + head turning at 30 FPS)")
    print("=" * 78)
    for name, m in run_temporal_benchmark().items():
        print(f"{name:>18}  {m}")


if __name__ == "__main__":  # pragma: no cover
    main()
