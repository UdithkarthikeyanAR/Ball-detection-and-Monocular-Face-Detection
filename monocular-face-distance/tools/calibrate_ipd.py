#!/usr/bin/env python3
"""Solve for the user's interpupillary distance from one known reference range.

After the camera focal length, the assumed IPD is the largest remaining error
term: the adult population spread is about 3.5 mm on a 63 mm mean, which is a
5-6% scale error that no amount of filtering will remove. The benchmark shows a
68 mm subject measured with the 63 mm default lands at 8% error; calibrated,
the same subject drops back under 1.5%.

The trick used here is that depth scales *linearly* in the assumed IPD:

    Z_estimated = Z_true * (IPD_assumed / IPD_true)

so measuring once at a known distance and inverting the ratio recovers the true
IPD directly -- no optimisation loop needed.

Usage
-----
    python tools/calibrate_ipd.py --true-distance-mm 600 \
        --camera-calibration calibration.json --output ipd.json

Measure from the camera lens to the bridge of your nose with a tape measure,
sit still, and hold your head level and facing the camera.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facedist.camera import resolve_intrinsics  # noqa: E402
from facedist.estimator import DistanceEstimator  # noqa: E402
from facedist.face_model import CanonicalFaceModel, DEFAULT_IPD_MM  # noqa: E402
from facedist.landmarks import LandmarkDetector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--true-distance-mm", type=float, required=True)
    parser.add_argument("--camera-calibration", default=None)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--output", default="ipd.json")
    args = parser.parse_args()

    intr = resolve_intrinsics(args.width, args.height, args.camera_calibration)
    if not intr.calibrated:
        print("warning: camera is not calibrated. Focal-length error will be "
              "absorbed into the IPD estimate, which makes the resulting value "
              "camera-specific. Run calibrate_camera.py first.\n",
              file=sys.stderr)

    estimator = DistanceEstimator(intr, CanonicalFaceModel(DEFAULT_IPD_MM))
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"error: cannot open camera {args.camera}", file=sys.stderr)
        return 1

    ratios: list[float] = []
    print(f"Hold still at {args.true_distance_mm:.0f} mm. Q to abort.")

    try:
        with LandmarkDetector(max_faces=1) as detector:
            while len(ratios) < args.samples:
                ok, frame = cap.read()
                if not ok:
                    break
                faces = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                status, colour = "no face", (0, 0, 255)

                if faces:
                    est = estimator.estimate(faces[0].landmarks_px)
                    if est.valid and abs(est.pose.yaw) < 12.0:
                        ratios.append(est.distance_mm / args.true_distance_mm)
                        status = f"captured {len(ratios)}/{args.samples}"
                        colour = (0, 255, 0)
                    elif est.valid:
                        status = "face the camera squarely"
                        colour = (0, 200, 255)

                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, colour, 2)
                cv2.imshow("ipd calibration", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(ratios) < args.samples // 2:
        print("error: not enough clean samples", file=sys.stderr)
        return 1

    # Median, not mean: a few frames with bad landmarks should not move it.
    ratio = statistics.median(ratios)
    ipd = DEFAULT_IPD_MM / ratio
    spread = statistics.stdev(ratios) if len(ratios) > 1 else 0.0

    if not 40.0 <= ipd <= 85.0:
        print(f"error: implausible IPD ({ipd:.1f} mm). Check that the measured "
              "distance is correct and the camera calibration matches this "
              "capture resolution.", file=sys.stderr)
        return 1

    Path(args.output).write_text(json.dumps({
        "ipd_mm": round(ipd, 2),
        "reference_distance_mm": args.true_distance_mm,
        "samples": len(ratios),
        "ratio_stdev": round(spread, 4),
        "camera_calibrated": intr.calibrated,
    }, indent=2))

    print(f"\nestimated IPD : {ipd:.1f} mm  (population mean {DEFAULT_IPD_MM})")
    print(f"sample spread : {spread * 100:.2f} %")
    print(f"written to    : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
