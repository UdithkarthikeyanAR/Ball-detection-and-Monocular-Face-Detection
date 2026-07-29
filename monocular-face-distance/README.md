# facedist — monocular face distance **and deviation angle**

Real-time `(depth, θ)` for a face from a single ordinary camera. No depth
sensor, no stereo rig, no training run.

## What the problem statement asks for

| Spec item | Where it lives |
|---|---|
| `Depth: Z = (f × W) / w_px` | `facedist/baseline.py` — implemented verbatim as the reference |
| `Angle: θ = arctan((x − c_x) / f)` | `facedist/baseline.py::angle`, and `bearing.from_pixel(..., undistort=False)` reproduces it to 1e-9 |
| Expected output `(depth, θ)` | `FaceResult.as_spec_output()` → `(Z_metres, θ_radians)` |
| Single monocular camera, pinhole model | `facedist/camera.py` |
| Real face width ≈ 0.14–0.16 m | `baseline.DEFAULT_FACE_WIDTH_MM = 140.0` |
| Face centre + width in pixels detectable | `facedist/landmarks.py` |

Run it on one image and you get exactly the expected pair:

```bash
python -m facedist.cli --image face.jpg
# {"faces": [{"depth_m": 0.842, "theta_rad": 0.19106, "theta_deg": 10.947, ...}]}
```

Everything below is what we added on top of that spec, and by how much it helps.

## Results

Synthetic harness (`python -m facedist.evaluate`), 4000 samples, distance
0.3–2.5 m, head yaw ±40°, 1.5 px landmark noise, 3 mm per-subject face-shape
variation.

**Depth**

| Method | MAE | RMSE | MAPE | p95 error |
|---|---|---|---|---|
| Baseline `Z = fW/w_px` | 107.0 mm | 140.2 mm | 7.96 % | 280.2 mm |
| **This system** | **17.9 mm** | **25.9 mm** | **1.19 %** | **55.8 mm** |

**6.0× lower mean error.** On a subject whose IPD is 68 mm rather than the
assumed 63 mm, after one calibration measurement: 16.6 mm vs the baseline's
135.2 mm — **8.1×**.

**Deviation angle θ** (θ ∈ ±22°, yaw ±40°, realistic barrel distortion; both
methods get identical pixels)

| Method | MAE | RMSE | p95 | worst case |
|---|---|---|---|---|
| Baseline `arctan((x−c_x)/f)` on bbox centre | 0.570° | 0.737° | 1.497° | 3.206° |
| **This system** | **0.047°** | **0.058°** | **0.113°** | **0.195°** |

**12× lower mean angular error, 16× lower worst case.** Elevation φ comes out
of the same fit at 0.027° MAE; the baseline model does not produce it at all.

On a moving subject at 30 FPS the Kalman stage takes depth RMSE from 20.5 mm to
16.4 mm (−20 %) and p95 from 44.1 mm to 34.0 mm.

### Why the angle improves — measured, not asserted

We ablated the two candidate causes:

| Configuration | Baseline θ MAE |
|---|---|
| Distortion on, yaw ±40° (full test) | 0.570° |
| Distortion **off**, yaw ±40° | 0.644° |
| Distortion on, yaw **0°** | 0.202° |

So the dominant error is **bounding-box drift under head yaw**, not lens
distortion. The box is defined by the visible silhouette, so a 40° head turn
slides its centre toward the near cheek while the skull has not moved. We
instead read the bearing off the PnP-recovered **midpoint of the eyes**, a point
fixed to the skull.

An honest wrinkle worth reporting: undistorting the *bbox centre* alone makes
the baseline slightly **worse** (0.632° vs 0.570°), because the barrel
distortion had been accidentally cancelling part of the box drift. Undistortion
only helps once you are already measuring an anatomically fixed point. We
report this because it contradicts the obvious hypothesis, and a result you
only publish when it flatters you is not a result.

Read the caveat before quoting any of these: the harness renders the face
through the *same* camera model the estimator assumes, so these numbers measure
**estimation-geometry quality, not end-to-end system accuracy**. Real landmark
detectors add their own error. Treat the ratios as the meaningful result and the
absolute figures as an optimistic bound. See *Validating on real footage*.

## How it works

```
frame → FaceMesh (468 landmarks) → SQPnP against a metric face model
      → depth of the eye midpoint  → inverse-depth Kalman filter → result
```

Four choices carry most of the improvement.

**A 12-point metric model instead of a bounding box.** The baseline reads a
13 % head turn as 13 % of movement, because a rotated face projects to a
narrower box. Solving PnP recovers rotation explicitly, so pose stops
contaminating depth. This is the single largest win, and `test_baseline_is_
pose_sensitive_and_ours_is_not` pins it: across ±35° of yaw at a fixed 1 m, our
estimate varies under 3 %, the baseline several times that.

**Scale is set by interpupillary distance, not by the model's own units.** The
canonical model is rescaled at load so its IPD matches the user's. Depth then
scales linearly in the assumed IPD, so a single measurement at a known distance
calibrates it exactly — that is what `tools/calibrate_ipd.py` exploits.

**Filtering in inverse depth.** Pixel noise maps to depth error growing as `Z²`,
so a filter tuned at 0.5 m is badly mistuned at 2.5 m. In `q = 1/Z` the noise is
near-constant and one setting works across the range. Measurement noise is
adaptive: the estimator's per-frame sigma feeds `R` directly, so a small or
poorly-fitted face is automatically down-weighted rather than yanking the track.

**Two cues, but deliberately not fused.** A second estimate from the eye
baseline alone is computed every frame. Benchmarking showed that blending it in
made things *worse* — 37 mm MAE against 18 mm for PnP alone — because it carries
a per-subject *bias* (3 mm of eye-corner shape variation shifts it several
percent, consistently, for that person) and variance weighting cannot cancel a
bias. So it is kept as a fallback for when the PnP fit is rejected, and as a
disagreement check that widens the reported sigma. That negative result is
reproducible: `run_static_benchmark` reports `pnp_only` and `ipd_only`
separately.

### Deviation angle

`θ = atan2(X, Z)` and `φ = atan2(−Y, hypot(X, Z))` of the eye midpoint in camera
coordinates. Because that 3D point is the same one we report depth to, depth and
angle are two spherical coordinates of a single vector rather than two
independent estimates that could disagree. Sign convention: **+θ = face is to
the image right of the optical axis; +φ = above it.**

The tracker is fed the eye-midpoint pixel rather than the bbox centre, so the
existing Kalman position state smooths θ for free and keeps it consistent with
the smoothed depth.

## The instrument panel

```bash
python -m facedist.server --calibration calib.json
# facedist panel -> http://127.0.0.1:8000
```

A browser panel rather than a Qt window: the CV stays in Python, the layout and
charting happen where they are pleasant to build, and you can put the readout on
a second screen or a phone during a demo without changing any code.

Capture and inference run in one background thread and publish the newest frame
under a lock. HTTP handlers never block on the camera, so a slow browser drops
frames instead of stalling inference — the correct trade for live measurement.

What is on it, and why each panel earns its place:

| Panel | Why |
|---|---|
| **Depth Z** with ±1σ | The tolerance is part of the measurement. A number without one is a guess with a decimal point. |
| **Spec baseline + Δ** | `fW/w_px` computed on the same frame, shown directly beneath ours. The delta is the project's claim, live. |
| **Deviation θ** as an arc | θ is an angle, so it is drawn as one. The numeral alone makes the viewer do the geometry in their head. |
| **Measurement quality** | Four real diagnostics (fit, cue agreement, stability, pose), not a decorative bar. See `quality.py`. |
| **Range history, dual trace** | The signature. Both estimates on one axis: turn your head and the baseline swings while ours holds. |
| **Spec inputs strip** | `x`, `w_px`, `f`, `c_x`, `W` echoed live so anyone can check the arithmetic against the problem statement by hand. |
| **Uncalibrated banner** | A wrong `f` is a *systematic* error, not noise, so it gets its own banner instead of being buried in the quality score. |

Keys: `R` reset track · `M` smoothed/raw · `S` screenshot · `F` fullscreen.

## MediaPipe compatibility

MediaPipe removed the legacy `mp.solutions.face_mesh` API in the 0.10.3x series.
`facedist/landmarks.py` probes for it and falls back to the Tasks
`FaceLandmarker`, downloading the `.task` bundle once into `~/.cache/facedist`.
Both paths use the same landmark topology, so the geometry layer is unaffected.
If your machine is offline, fetch the model manually and pass
`LandmarkDetector(model_path=...)`.

## Install

```bash
pip install -r requirements.txt
```

MediaPipe is only needed for live capture. The geometry, tracking and
benchmarking modules run without it, which is what keeps the test suite fast and
camera-free.

## Use

```bash
# live demo
python -m facedist.cli

# with the baseline drawn alongside, to see the difference yourself
python -m facedist.cli --show-baseline

# after calibrating (strongly recommended)
python -m facedist.cli --calibration calibration.json --ipd-file ipd.json
```

As a library:

```python
from facedist import FaceDistancePipeline

pipeline = FaceDistancePipeline(width=1280, height=720,
                                calibration_path="calibration.json")
for face in pipeline.process(frame_bgr):
    print(f"track {face.track_id}: {face.distance_cm:.1f} cm "
          f"± {face.sigma_mm / 10:.1f}, yaw {face.yaw:+.0f}°")
```

## Calibrate — this matters more than the algorithm

Uncalibrated, two scale errors dominate everything else:

| Error source | Uncalibrated | Calibrated | Fix |
|---|---|---|---|
| Focal length (guessed from FOV) | 10–20 % | < 1 % | `tools/calibrate_camera.py` |
| Interpupillary distance | ~5.5 % (1σ) | < 1 % | `tools/calibrate_ipd.py` |

The benchmark makes this concrete: a 68 mm-IPD subject measured with the 63 mm
default sits at 8.15 % error — worse than the algorithm's own noise by a factor
of seven. Calibrate that same subject and it drops to 1.10 %. **An uncalibrated
run of this system is not meaningfully better than the baseline.** Both tools
take about ten minutes.

```bash
python tools/calibrate_camera.py --output calibration.json
python tools/calibrate_ipd.py --true-distance-mm 600 \
    --camera-calibration calibration.json --output ipd.json
```

`resolve_intrinsics` falls back to a 60° FOV guess when no calibration is
present, and the estimator inflates its reported sigma by 10 % to reflect that,
so the uncertainty stays honest even when the number is poor.

## Validating on real footage

The synthetic numbers are not a substitute for this:

1. Mark floor positions at 50, 75, 100, 150, 200 and 250 cm from the lens.
2. Record 10 s at each, once facing the camera, once at roughly 30° of yaw.
3. Log `raw_distance_mm` against the tape measurement and compute MAE per
   distance and per pose.

Expect real-world error to land meaningfully above the synthetic figures,
mostly from landmark noise. The pose-invariance advantage over the baseline
should survive intact — that one is geometric, not statistical.

## Limitations

- **Adults only.** The face model and the 63 mm default IPD are adult
  anthropometric means. Children will read systematically far; calibrate per
  subject, or don't use it on them.
- **Glasses, heavy occlusion and extreme expression** degrade the eye-corner
  landmarks; the estimator falls back to the IPD cue and widens sigma, but the
  answer is worse.
- **Beyond ~3 m** the face spans too few pixels for the geometry to be
  well-conditioned. The estimator rejects faces under 20 px span outright.
- **Rolling-shutter cameras** under fast lateral motion will skew the landmark
  geometry. Not compensated.
- Testing has been synthetic plus unit-level. There is **no real-world
  ground-truth evaluation in this repository yet**.

## Layout

```
facedist/
  face_model.py   metric 3D face model, FaceMesh landmark indices
  camera.py       intrinsics, FOV fallback, calibration I/O
  estimator.py    SQPnP + cheirality check, IPD fallback, uncertainty
  tracking.py     inverse-depth Kalman, IoU association, outlier gating
  landmarks.py    MediaPipe FaceMesh wrapper (lazy import)
  baseline.py     bbox-width reference implementation
  simulation.py   synthetic ground-truth renderer
  evaluate.py     benchmarks
  pipeline.py     orchestration
  cli.py          live application
tools/
  calibrate_camera.py
  calibrate_ipd.py
tests/
  test_facedist.py    21 tests, no camera or MediaPipe required
```

## Test

```bash
python -m pytest tests/ -q      # 21 passed
python -m facedist.evaluate     # full benchmark table
```
