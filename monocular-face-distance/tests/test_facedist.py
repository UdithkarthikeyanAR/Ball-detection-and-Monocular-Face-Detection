"""Tests for the geometry, estimation and tracking layers.

None of these need MediaPipe or a camera: the synthetic harness supplies
landmarks with known ground truth, which is the only way to make falsifiable
accuracy claims.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import facedist.bearing as bearing_mod
from facedist.baseline import BaselineWidthEstimator
from facedist.camera import CameraIntrinsics
from facedist.estimator import DistanceEstimator
from facedist.evaluate import run_static_benchmark, run_temporal_benchmark
from facedist.face_model import CanonicalFaceModel, DEFAULT_IPD_MM
from facedist.simulation import FaceSimulator
from facedist.tracking import FaceTracker, _InverseDepthKalman


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
    intr.calibrated = True
    return intr


# --- face model -------------------------------------------------------------

def test_model_rescales_to_requested_ipd():
    for ipd in (55.0, 63.0, 72.0):
        model = CanonicalFaceModel(ipd_mm=ipd)
        right, left = model.eye_centres_mm
        assert np.linalg.norm(left - right) == pytest.approx(ipd, rel=1e-9)


def test_model_rejects_implausible_ipd():
    with pytest.raises(ValueError):
        CanonicalFaceModel(ipd_mm=5.0)


# --- camera -----------------------------------------------------------------

def test_fov_focal_length_matches_geometry():
    intr = CameraIntrinsics.from_fov(1280, 720, 90.0)
    # At exactly 90 deg HFOV the focal length equals half the sensor width.
    assert intr.fx == pytest.approx(640.0, rel=1e-9)


def test_rescale_preserves_field_of_view():
    a = CameraIntrinsics.from_fov(1280, 720, 60.0)
    b = a.rescaled(640, 360)
    assert a.fx / a.width == pytest.approx(b.fx / b.width, rel=1e-12)


def test_calibration_roundtrip(tmp_path, intrinsics):
    path = tmp_path / "cal.json"
    intrinsics.save(path)
    assert CameraIntrinsics.load(path) == intrinsics


# --- estimator --------------------------------------------------------------

def test_recovers_exact_distance_with_no_noise(intrinsics):
    sim = FaceSimulator(intrinsics, shape_jitter_mm=0.0,
                        landmark_noise_px=0.0, seed=1)
    est = DistanceEstimator(intrinsics, CanonicalFaceModel())
    for z in (350.0, 700.0, 1500.0, 2400.0):
        result = est.estimate(sim.render(z, 15.0, -8.0, 5.0).landmarks_px)
        assert result.valid
        assert result.distance_mm == pytest.approx(z, rel=0.005)


def test_pose_recovery_is_sane(intrinsics):
    sim = FaceSimulator(intrinsics, shape_jitter_mm=0.0,
                        landmark_noise_px=0.0, seed=2)
    est = DistanceEstimator(intrinsics, CanonicalFaceModel())
    straight = est.estimate(sim.render(800.0, 0.0, 0.0, 0.0).landmarks_px)
    turned = est.estimate(sim.render(800.0, 30.0, 0.0, 0.0).landmarks_px)
    assert abs(straight.pose.yaw) < 2.0
    assert abs(turned.pose.yaw) > 20.0


def test_scale_error_is_proportional_to_ipd_error(intrinsics):
    """Depth must scale linearly in assumed IPD -- calibrate_ipd relies on it."""
    sim = FaceSimulator(intrinsics, true_ipd_mm=63.0, shape_jitter_mm=0.0,
                        landmark_noise_px=0.0, seed=3)
    frame = sim.render(1000.0, 0.0, 0.0, 0.0)
    wrong = DistanceEstimator(intrinsics, CanonicalFaceModel(ipd_mm=70.0))
    result = wrong.estimate(frame.landmarks_px)
    assert result.distance_mm == pytest.approx(1000.0 * 70.0 / 63.0, rel=0.02)


def test_rejects_degenerate_input(intrinsics):
    est = DistanceEstimator(intrinsics, CanonicalFaceModel())
    assert not est.estimate(np.zeros((468, 2))).valid
    assert not est.estimate(np.zeros((10, 2))).valid
    assert not est.estimate(np.full((468, 2), np.nan)).valid


def test_uncalibrated_camera_reports_larger_sigma(intrinsics):
    sim = FaceSimulator(intrinsics, seed=4)
    frame = sim.render(900.0, 0.0, 0.0, 0.0)
    loose = CameraIntrinsics.from_fov(1280, 720, 60.0)   # calibrated = False
    a = DistanceEstimator(intrinsics, CanonicalFaceModel()).estimate(
        frame.landmarks_px)
    b = DistanceEstimator(loose, CanonicalFaceModel()).estimate(
        frame.landmarks_px)
    assert b.sigma_mm > a.sigma_mm


def test_sigma_is_honest(intrinsics):
    """Reported sigma must actually bracket the truth about 68% of the time."""
    sim = FaceSimulator(intrinsics, shape_jitter_mm=0.0, seed=11)
    est = DistanceEstimator(intrinsics, CanonicalFaceModel(), ipd_sigma_mm=0.01)
    rng = np.random.default_rng(12)
    inside = total = 0
    for _ in range(600):
        z = float(rng.uniform(400.0, 2000.0))
        r = est.estimate(sim.render(z, float(rng.uniform(-30, 30)), 0.0,
                                    0.0).landmarks_px)
        if not r.valid:
            continue
        total += 1
        inside += abs(r.distance_mm - z) <= r.sigma_mm
    # Not over-confident. The shape-error floor makes it conservative on
    # synthetic faces with zero shape variation, which is the safe direction;
    # the second assertion stops it from being conservative to the point of
    # uselessness.
    assert inside / total >= 0.55
    assert est.estimate(
        sim.render(1000.0, 0.0, 0.0, 0.0).landmarks_px
    ).sigma_mm < 100.0


# --- tracking ---------------------------------------------------------------

def test_kalman_converges_to_a_static_target():
    kf = _InverseDepthKalman(100.0, 100.0, 1000.0, 50.0)
    rng = np.random.default_rng(5)
    for _ in range(120):
        kf.predict(1 / 30)
        kf.update(100.0, 100.0, 1000.0 + rng.normal(0, 40), 50.0)
    assert kf.distance_mm == pytest.approx(1000.0, abs=25.0)


def test_kalman_tracks_a_moving_target():
    kf = _InverseDepthKalman(0.0, 0.0, 2000.0, 50.0)
    z = 2000.0
    for _ in range(90):
        z -= 10.0            # 300 mm/s approach at 30 FPS
        kf.predict(1 / 30)
        kf.update(0.0, 0.0, z, 50.0)
    assert kf.distance_mm == pytest.approx(z, rel=0.03)
    assert kf.velocity_mm_s < -150.0     # negative == approaching


def test_kalman_gates_outliers():
    kf = _InverseDepthKalman(0.0, 0.0, 1000.0, 20.0)
    for _ in range(40):
        kf.predict(1 / 30)
        kf.update(0.0, 0.0, 1000.0, 20.0)
    assert kf.update(0.0, 0.0, 12000.0, 20.0) is False
    assert kf.distance_mm == pytest.approx(1000.0, abs=60.0)


def test_covariance_stays_positive_definite():
    kf = _InverseDepthKalman(0.0, 0.0, 800.0, 30.0)
    rng = np.random.default_rng(6)
    for _ in range(400):
        kf.predict(1 / 30)
        kf.update(0.0, 0.0, 800.0 + rng.normal(0, 30),
                  float(rng.uniform(10.0, 200.0)))
    assert np.all(np.linalg.eigvalsh(kf.P) > 0)


def test_tracker_assigns_stable_ids_to_two_faces():
    tracker = FaceTracker(min_hits=2)
    left = (100.0, 100.0, 200.0, 220.0)
    right = (500.0, 110.0, 600.0, 230.0)
    ids = []
    for _ in range(10):
        tracks = tracker.update([
            {"bbox": left, "centre": (150.0, 160.0),
             "distance_mm": 800.0, "sigma_mm": 40.0},
            {"bbox": right, "centre": (550.0, 170.0),
             "distance_mm": 1400.0, "sigma_mm": 60.0},
        ], 1 / 30)
        ids.append(tuple(sorted(t.track_id for t in tracks)))
    assert len(set(ids)) == 1 and ids[-1] == (1, 2)
    assert all(t.confirmed for t in tracks)


def test_tracker_drops_stale_tracks():
    tracker = FaceTracker(max_misses=3)
    tracker.update([{"bbox": (0.0, 0.0, 50.0, 50.0), "centre": (25.0, 25.0),
                     "distance_mm": 900.0, "sigma_mm": 40.0}], 1 / 30)
    for _ in range(6):
        tracks = tracker.update([], 1 / 30)
    assert tracks == []


# --- end-to-end regression --------------------------------------------------

def test_beats_baseline_by_a_wide_margin():
    results = run_static_benchmark(n_samples=1200, seed=21)
    proposed, baseline = results["proposed"], results["baseline_bbox"]
    assert proposed.mape_pct < 2.0
    assert proposed.mae_mm < baseline.mae_mm / 3.0
    assert proposed.p95_abs_err_mm < baseline.p95_abs_err_mm / 2.0


def test_kalman_reduces_jitter_on_a_moving_subject():
    results = run_temporal_benchmark(n_frames=600, seed=22)
    assert (results["proposed_kalman"].rmse_mm
            < results["proposed_unsmoothed"].rmse_mm)


def test_ipd_calibration_recovers_accuracy():
    """A 68 mm subject should go from ~8% error to under 2% once calibrated."""
    before = run_static_benchmark(n_samples=800, true_ipd_mm=68.0,
                                  assumed_ipd_mm=63.0, seed=23)
    after = run_static_benchmark(n_samples=800, true_ipd_mm=68.0,
                                 assumed_ipd_mm=68.0, seed=23)
    assert before["proposed"].mape_pct > 5.0
    assert after["proposed"].mape_pct < 2.0


def test_baseline_is_pose_sensitive_and_ours_is_not(intrinsics):
    """The core qualitative claim: head rotation should not read as movement."""
    sim = FaceSimulator(intrinsics, shape_jitter_mm=0.0,
                        landmark_noise_px=0.0, seed=24)
    est = DistanceEstimator(intrinsics, CanonicalFaceModel())
    base = BaselineWidthEstimator(intrinsics)

    ours, theirs = [], []
    for yaw in (-35.0, -20.0, 0.0, 20.0, 35.0):
        frame = sim.render(1000.0, yaw, 0.0, 0.0)
        ours.append(est.estimate(frame.landmarks_px).distance_mm)
        theirs.append(base.estimate(sim.bbox_of(frame.landmarks_px)))

    spread = lambda v: (max(v) - min(v)) / float(np.mean(v))  # noqa: E731
    assert spread(ours) < 0.03
    assert spread(theirs) > spread(ours) * 3.0


# ---------------------------------------------------------------------------
# Deviation angle (theta) -- the problem statement's second output
# ---------------------------------------------------------------------------

class TestBearing:
    """The spec asks for theta = arctan((x - c_x) / f). Verify we match it,
    then verify we improve on it."""

    def test_matches_spec_formula_exactly(self):
        """With distortion off, from_pixel IS the problem statement's formula."""
        intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
        for u in (10.0, 320.0, 640.0, 900.0, 1270.0):
            spec = math.degrees(math.atan((u - intr.cx) / intr.fx))
            ours = bearing_mod.from_pixel(intr, u, intr.cy,
                                          undistort=False).theta_deg
            assert ours == pytest.approx(spec, abs=1e-9)

    def test_sign_convention(self):
        """+theta means the face is to the image right of the axis."""
        intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
        assert bearing_mod.from_pixel(intr, 900.0, 360.0).theta_deg > 0
        assert bearing_mod.from_pixel(intr, 380.0, 360.0).theta_deg < 0
        # +phi means above the axis; image y grows downward.
        assert bearing_mod.from_pixel(intr, 640.0, 200.0).phi_deg > 0
        assert bearing_mod.from_pixel(intr, 640.0, 520.0).phi_deg < 0

    def test_pixel_and_point_paths_agree(self):
        """A 3D point and its projection must give the same bearing."""
        intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
        p = np.array([137.0, -64.0, 910.0])
        u = intr.fx * p[0] / p[2] + intr.cx
        v = intr.fy * p[1] / p[2] + intr.cy
        a = bearing_mod.from_point(p)
        b = bearing_mod.from_pixel(intr, u, v, undistort=False)
        assert a.theta_deg == pytest.approx(b.theta_deg, abs=1e-9)
        assert a.phi_deg == pytest.approx(b.phi_deg, abs=1e-9)

    def test_unit_vector_roundtrip(self):
        bg = bearing_mod.Bearing(theta_deg=14.0, phi_deg=-7.0)
        v = bg.unit_vector()
        assert np.linalg.norm(v) == pytest.approx(1.0)
        back = bearing_mod.from_point(v * 1500.0)
        assert back.theta_deg == pytest.approx(14.0, abs=1e-9)
        assert back.phi_deg == pytest.approx(-7.0, abs=1e-9)

    def test_recovers_known_angle_noise_free(self):
        intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
        intr.calibrated = True
        sim = FaceSimulator(intr, shape_jitter_mm=0.0,
                            landmark_noise_px=0.0, seed=3)
        est = DistanceEstimator(intr)
        for theta, phi, z, yaw in [(0, 0, 800, 0), (15, -6, 600, 25),
                                   (-19, 9, 1800, -30)]:
            frame = sim.render(z, yaw_deg=yaw, theta_deg=theta, phi_deg=phi)
            r = est.estimate(frame.landmarks_px)
            assert r.valid
            assert r.bearing.theta_deg == pytest.approx(theta, abs=0.01)
            assert r.bearing.phi_deg == pytest.approx(phi, abs=0.01)

    def test_pose_invariance_beats_bbox(self):
        """Head yaw must not move the reported angle. The bbox centre does."""
        intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
        intr.calibrated = True
        sim = FaceSimulator(intr, shape_jitter_mm=0.0,
                            landmark_noise_px=0.0, seed=4)
        est = DistanceEstimator(intr)
        base = BaselineWidthEstimator(intr)

        ours, theirs = [], []
        for yaw in range(-40, 41, 5):
            frame = sim.render(1000.0, yaw_deg=float(yaw), theta_deg=12.0)
            r = est.estimate(frame.landmarks_px)
            assert r.valid
            ours.append(r.bearing.theta_deg)
            theirs.append(base.angle(sim.bbox_of(frame.landmarks_px)))

        spread_ours = max(ours) - min(ours)
        spread_theirs = max(theirs) - min(theirs)
        assert spread_ours < 0.05, f"ours drifted {spread_ours:.3f} deg with yaw"
        assert spread_theirs > 10 * spread_ours

    def test_beats_baseline_on_benchmark(self):
        from facedist.evaluate import run_angle_benchmark
        m = run_angle_benchmark(n_samples=600)
        assert m["proposed_theta"].mae_deg < 0.2
        assert m["proposed_theta"].mae_deg < 0.25 * m["baseline_theta"].mae_deg

    def test_spec_output_tuple_units(self):
        """as_spec_output() must return (metres, radians)."""
        intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
        intr.calibrated = True
        sim = FaceSimulator(intr, shape_jitter_mm=0.0,
                            landmark_noise_px=0.0, seed=5)
        est = DistanceEstimator(intr)
        frame = sim.render(1500.0, theta_deg=10.0)
        r = est.estimate(frame.landmarks_px)
        z_m, theta_rad = r.as_spec_output()
        assert z_m == pytest.approx(1.5, abs=0.02)
        assert math.degrees(theta_rad) == pytest.approx(10.0, abs=0.05)

    def test_invalid_estimate_has_nan_bearing(self):
        intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
        est = DistanceEstimator(intr)
        r = est.estimate(np.zeros((10, 2)))
        assert not r.valid
        assert math.isnan(r.bearing.theta_deg)

    def test_undistortion_is_identity_without_coefficients(self):
        intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
        assert not intr.has_distortion
        pts = np.array([[100.0, 200.0], [640.0, 360.0]])
        assert np.allclose(intr.undistort_points(pts), pts)

    def test_undistortion_moves_edge_points(self):
        intr = CameraIntrinsics.from_fov(1280, 720, 60.0)
        intr.dist_coeffs = (-0.25, 0.08, 0.0, 0.0, 0.0)
        assert intr.has_distortion
        edge = np.array([[60.0, 60.0]])
        moved = intr.undistort_points(edge)
        assert np.linalg.norm(moved - edge) > 3.0
