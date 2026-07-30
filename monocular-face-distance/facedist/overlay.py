"""Overlay drawn onto the video frame itself.

Kept separate from the web UI so the same drawing works in the plain OpenCV
window (``python -m facedist.cli``) and in the browser panel. Nothing here
computes anything -- it only renders what the pipeline already produced.

The overlay deliberately shows the *spec's own input variables* along the
bottom (x, w_px, f, c_x, W). During a demo that strip is what lets someone check
the arithmetic by hand against the problem statement, which is more convincing
than any claim in a slide.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# BGR, matched to the web palette.
PHOS = (142, 195, 240)      # apricot #F0C38E in BGR
SLATE = (174, 129, 137)     # lavender, baseline: deliberately dull
AMBER = (155, 170, 241)   # coral #F1AA9B in BGR
HOT = (122, 134, 228)     # deep coral
RULE = (107, 77, 84)
WHITE = (235, 235, 235)
DIM = (140, 140, 140)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _tint(quality: float, confirmed: bool):
    if not confirmed:
        return AMBER
    if quality >= 0.70:
        return PHOS
    if quality >= 0.45:
        return AMBER
    return HOT


def draw_face(frame, result, intrinsics, show_baseline_mm: float | None = None):
    """Draw one face: box, eye baseline, optical axis, deviation ray."""
    h, w = frame.shape[:2]
    q = result.quality.overall if result.quality else 0.0
    colour = _tint(q, result.confirmed)

    x1, y1, x2, y2 = (int(v) for v in result.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Corner ticks -- reads as a measurement reticle rather than a detector box.
    t = max(10, (x2 - x1) // 8)
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * t, cy), colour, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy * t), colour, 3)

    # Optical axis: the reference θ is measured from.
    pcx, pcy = int(intrinsics.cx), int(intrinsics.cy)
    cv2.line(frame, (pcx, 0), (pcx, h), (70, 60, 55), 1, cv2.LINE_AA)
    cv2.drawMarker(frame, (pcx, pcy), (110, 100, 90), cv2.MARKER_CROSS, 22, 1)

    # Ray to the tracked anatomical point, annotated with θ.
    fx_, fy_ = int(result.ref_x_px), int(result.ref_y_px)
    cv2.line(frame, (pcx, pcy), (fx_, fy_), colour, 1, cv2.LINE_AA)
    cv2.circle(frame, (fx_, fy_), 4, colour, -1, cv2.LINE_AA)
    if math.isfinite(result.theta_deg):
        side = "R" if result.theta_deg > 0 else "L"
        label = f"{result.theta_deg:+.1f}deg {side}"
        mx, my = (pcx + fx_) // 2, (pcy + fy_) // 2 - 8
        cv2.putText(frame, label, (mx - 34, my), _FONT, 0.5, colour, 2, cv2.LINE_AA)

    lines = [f"Z {result.distance_cm:.1f} cm  +/- {result.sigma_mm / 10.0:.1f}",
             f"theta {result.theta_deg:+.2f}  phi {result.phi_deg:+.2f}",
             f"yaw {result.yaw:+.0f} pitch {result.pitch:+.0f} roll {result.roll:+.0f}"]
    if show_baseline_mm is not None and math.isfinite(show_baseline_mm):
        lines.append(f"spec baseline {show_baseline_mm / 10.0:.1f} cm")

    y = max(y1 - 10 - 17 * (len(lines) - 1), 16)
    for i, line in enumerate(lines):
        c = SLATE if line.startswith("spec") else colour
        cv2.putText(frame, line, (x1, y), _FONT, 0.5, c, 2, cv2.LINE_AA)
        y += 17
    return frame


def draw_input_strip(frame, result, intrinsics, face_width_mm: float):
    """Bottom strip echoing the problem statement's own input variables."""
    h, w = frame.shape[:2]
    x1, _, x2, _ = result.bbox
    w_px = abs(x2 - x1)
    text = (f"spec inputs   x={result.ref_x_px:.0f}px   w_px={w_px:.0f}px   "
            f"f={intrinsics.fx:.1f}px   c_x={intrinsics.cx:.0f}px   "
            f"W={face_width_mm / 1000.0:.3f}m")
    bar_h = 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, h - bar_h - 8), (w - 8, h - 8), (18, 14, 11), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(frame, text, (18, h - 18), _FONT, 0.46, DIM, 1, cv2.LINE_AA)
    return frame


def draw_banner(frame, text: str, colour=AMBER):
    """Top banner, used for the uncalibrated warning."""
    w = frame.shape[1]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30), (14, 26, 44), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, text, (14, 20), _FONT, 0.5, colour, 1, cv2.LINE_AA)
    return frame