#!/usr/bin/env python3
"""Estimate camera intrinsics from a printed chessboard.

An uncalibrated focal length is the single largest error source in this system:
guessing the field of view is typically 10-20% wrong, and depth error tracks it
one-for-one. Ten minutes with a printed chessboard removes that entire term.

Usage
-----
    python tools/calibrate_camera.py --output calibration.json

Print any chessboard pattern, tape it to something rigid and flat, then hold it
at varied angles and distances. Press SPACE to capture a view when the detected
corners light up; press Q when you have 15-20 spread-out views. Views from many
different angles matter far more than many views from the same angle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facedist.camera import CameraIntrinsics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source", default=None,
                        help="IP-camera URL or video file; overrides --camera. "
                             "Intrinsics are per-camera and never transfer, so "
                             "a phone stream needs its own calibration file.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cols", type=int, default=9,
                        help="inner corners across the board")
    parser.add_argument("--rows", type=int, default=6,
                        help="inner corners down the board")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--min-views", type=int, default=12)
    parser.add_argument("--lock-focus-reminder", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--output", default="calibration.json")
    args = parser.parse_args()

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_mm

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []

    src = args.source if args.source is not None else args.camera
    if isinstance(src, str) and "://" in src:
        print(f"connecting to {src} ... (can take up to 30s if the URL is wrong)")
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"error: cannot open video source {src!r}", file=sys.stderr)
        return 1

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if isinstance(src, str) and "://" in src:
        print("\n  PHONE CAMERA: lock focus AND exposure in the app before you\n"
              "  start. Autofocus changes the focal length as it hunts, which\n"
              "  silently invalidates the calibration you are about to make.\n")
    print("SPACE = capture view, Q = finish")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, pattern, corners, found)

            cv2.putText(display, f"views: {len(obj_points)}/{args.min_views}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow("calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and found:
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           criteria)
                obj_points.append(objp.copy())
                img_points.append(refined)
                print(f"captured view {len(obj_points)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(obj_points) < args.min_views:
        print(f"error: only {len(obj_points)} views, need {args.min_views}",
              file=sys.stderr)
        return 1

    rms, mtx, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    intr = CameraIntrinsics(
        fx=float(mtx[0, 0]), fy=float(mtx[1, 1]),
        cx=float(mtx[0, 2]), cy=float(mtx[1, 2]),
        width=gray.shape[1], height=gray.shape[0],
        dist_coeffs=tuple(float(v) for v in dist.ravel()[:5]),
        calibrated=True,
    )
    intr.save(args.output)

    hfov = 2.0 * np.degrees(np.arctan(intr.width / (2.0 * intr.fx)))
    print(f"\nreprojection RMS : {rms:.3f} px  (under ~0.5 is good)")
    print(f"fx, fy           : {intr.fx:.1f}, {intr.fy:.1f}")
    print(f"horizontal FOV   : {hfov:.1f} deg")
    print(f"written to       : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())