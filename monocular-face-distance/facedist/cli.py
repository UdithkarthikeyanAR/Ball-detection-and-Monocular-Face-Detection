"""Live webcam application: ``python -m facedist.cli``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .baseline import BaselineWidthEstimator
from .face_model import DEFAULT_IPD_MM
from .pipeline import FaceDistancePipeline, FaceResult

_GREEN = (0, 220, 0)
_AMBER = (0, 170, 255)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)


def _colour_for(result: FaceResult) -> tuple[int, int, int]:
    if not result.confirmed:
        return _AMBER
    relative = result.sigma_mm / max(result.distance_mm, 1.0)
    return _GREEN if relative < 0.05 else _AMBER if relative < 0.12 else _RED


def _draw(frame, result: FaceResult, show_baseline: float | None) -> None:
    x1, y1, x2, y2 = (int(v) for v in result.bbox)
    colour = _colour_for(result)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    lines = [
        f"#{result.track_id}  Z {result.distance_cm:.1f} cm  "
        f"+/- {result.sigma_mm / 10.0:.1f}",
        f"theta {result.theta_deg:+.2f} deg   phi {result.phi_deg:+.2f} deg",
        f"yaw {result.yaw:+.0f}  pitch {result.pitch:+.0f}  "
        f"roll {result.roll:+.0f}",
    ]
    if abs(result.velocity_mm_s) > 50.0:
        arrow = "approaching" if result.approaching else "receding"
        lines.append(f"{arrow} {abs(result.velocity_mm_s) / 10.0:.0f} cm/s")
    if show_baseline is not None:
        lines.append(f"baseline: {show_baseline / 10.0:.1f} cm")

    y = max(y1 - 8 - 18 * (len(lines) - 1), 18)
    for line in lines:
        cv2.putText(frame, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    colour, 2, cv2.LINE_AA)
        y += 18


def _draw_axis_marker(frame, result: FaceResult, intrinsics) -> None:
    """Draw the optical axis and the ray to the face, so theta is visible."""
    h, w = frame.shape[:2]
    cx, cy = int(intrinsics.cx), int(intrinsics.cy)
    cv2.drawMarker(frame, (cx, cy), (90, 90, 90), cv2.MARKER_CROSS, 18, 1)

    x1, y1, x2, y2 = result.bbox
    fx_, fy_ = int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2))
    cv2.line(frame, (cx, cy), (fx_, fy_), (200, 200, 60), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{result.theta_deg:+.1f}d",
                ((cx + fx_) // 2, (cy + fy_) // 2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 60), 1, cv2.LINE_AA)


def run_single_image(path: str, args) -> int:
    """Single 2D image in, ``(depth, theta)`` out -- the spec's literal task."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"error: cannot read image {path!r}", file=sys.stderr)
        return 1
    height, width = frame.shape[:2]

    ipd = args.ipd_mm or DEFAULT_IPD_MM
    if args.ipd_file and Path(args.ipd_file).exists():
        ipd = float(json.loads(Path(args.ipd_file).read_text())["ipd_mm"])

    pipeline = FaceDistancePipeline(
        width=width, height=height, calibration_path=args.calibration,
        ipd_mm=ipd, hfov_deg=args.hfov, max_faces=args.max_faces,
    )
    try:
        results = pipeline.process(frame)
    finally:
        pipeline.close()

    if not results:
        print(json.dumps({"faces": [], "error": "no face detected"}))
        return 2

    payload = []
    for r in results:
        z_m, theta_rad = r.as_spec_output()
        payload.append({
            "depth_m": round(z_m, 4),
            "theta_rad": round(theta_rad, 5),
            "theta_deg": round(r.theta_deg, 3),
            "phi_deg": round(r.phi_deg, 3),
            "depth_sigma_m": round(r.sigma_mm / 1000.0, 4),
            "theta_sigma_deg": round(r.sigma_theta_deg, 3),
            "yaw_deg": round(r.yaw, 2),
            "pitch_deg": round(r.pitch, 2),
            "roll_deg": round(r.roll, 2),
            "bbox": [round(v, 1) for v in r.bbox],
            "calibrated": pipeline.intrinsics.calibrated,
        })
    print(json.dumps({"faces": payload}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Real-time monocular face distance estimation."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source", default=None,
                        help="video file path; overrides --camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--calibration", default=None,
                        help="camera intrinsics JSON from calibrate_camera.py")
    parser.add_argument("--ipd-file", default=None,
                        help="IPD JSON from calibrate_ipd.py")
    parser.add_argument("--ipd-mm", type=float, default=None)
    parser.add_argument("--hfov", type=float, default=60.0,
                        help="assumed horizontal FOV when uncalibrated")
    parser.add_argument("--max-faces", type=int, default=2)
    parser.add_argument("--show-baseline", action="store_true",
                        help="overlay the bbox-width baseline for comparison")
    parser.add_argument("--image", default=None,
                        help="single image path; prints (depth, theta) as JSON")
    parser.add_argument("--mirror", action="store_true", default=True)
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    args = parser.parse_args(argv)

    if args.image:
        return run_single_image(args.image, args)

    ipd = args.ipd_mm or DEFAULT_IPD_MM
    if args.ipd_file and Path(args.ipd_file).exists():
        ipd = float(json.loads(Path(args.ipd_file).read_text())["ipd_mm"])

    source = args.source if args.source is not None else args.camera
    cap = cv2.VideoCapture(source)
    if args.source is None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"error: cannot open video source {source!r}", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height

    pipeline = FaceDistancePipeline(
        width=width, height=height, calibration_path=args.calibration,
        ipd_mm=ipd, hfov_deg=args.hfov, max_faces=args.max_faces,
    )
    baseline = BaselineWidthEstimator(pipeline.intrinsics)

    if not pipeline.intrinsics.calibrated:
        print("warning: no camera calibration; assuming "
              f"{args.hfov:.0f} deg FOV. Expect ~10-20% scale error.",
              file=sys.stderr)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror and args.source is None:
                frame = cv2.flip(frame, 1)

            for result in pipeline.process(frame):
                shown = (baseline.estimate(result.bbox)
                         if args.show_baseline else None)
                _draw(frame, result, shown)
                _draw_axis_marker(frame, result, pipeline.intrinsics)

            cv2.putText(frame, f"{pipeline.fps:.1f} FPS", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2, cv2.LINE_AA)
            cv2.imshow("facedist", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        pipeline.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
