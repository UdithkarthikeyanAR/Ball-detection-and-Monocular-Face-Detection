"""Measurement quality: a defensible confidence number, not a decorative bar.

The UI shows one headline percentage. It is a weighted mean of four sub-scores,
each of which is an actual diagnostic the estimator already produces:

    quality = 0.30*detection + 0.30*cue_match + 0.25*stability + 0.15*pose

**detection** -- how well the 3D face model reprojects onto the observed
landmarks. Directly the PnP residual, normalised by the face's on-screen size so
it means the same thing near and far. A bad fit here means occlusion, motion
blur, or an extreme expression.

**cue_match** -- agreement between the two *independent* depth cues: the
12-point PnP solve and the 2-point interpupillary relation. They share a scale
assumption but nothing else, so when they disagree beyond noise, something the
noise model does not cover is wrong (unusual face shape, bad landmarks, wrong
focal length). This is the single most informative field on the panel and it
costs nothing -- both cues are computed anyway.

**stability** -- dispersion of the recent smoothed depth history. Catches
flicker that a single frame cannot show.

**pose** -- frontality, cos(yaw)*cos(pitch). Not an error measurement but a
*conditioning* one: a strongly turned head gives the solver less usable
geometry, so the estimate is more fragile even when this frame happens to land
well. It carries the smallest weight for that reason.

Each sub-score is clipped to [0, 1] against an explicit tolerance, so a reading
of 0 means "at or past the tolerance", never "unknown". Missing inputs return
``None`` and are excluded from the weighted mean with the weights renormalised,
rather than being silently scored as zero.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

__all__ = ["QualityReport", "QualityMonitor"]

# Thresholds below are calibrated against *measured achievable* performance,
# not against perfection. That distinction matters: a generic anthropometric
# face model can never reproject onto a specific individual's landmarks
# exactly, so scoring against a zero-residual ideal caps the detection term at
# ~59% no matter how well the system is working. A confidence figure that
# cannot exceed 59% when everything is correct is mis-scaled, and an operator
# rightly stops trusting it.
#
# Each pair below is (value that should score ~1.0, value that should score 0):
# the first is what the system actually achieves under good conditions on
# simulated subjects with realistic 3 mm shape variation and 1.5 px landmark
# noise; the second is where the measurement genuinely stops being usable.

#: Reprojection RMS as a fraction of face span. 1.5% is a good real fit.
DETECTION_GOOD, DETECTION_BAD = 0.015, 0.060
#: Disagreement between the two independent depth cues. ~4% is the floor
#: imposed by using a population-mean face model rather than the subject's own.
CUE_GOOD, CUE_BAD = 0.040, 0.150
#: Depth dispersion as a fraction of depth.
STABILITY_GOOD, STABILITY_BAD = 0.004, 0.030
#: Frontality (cos yaw * cos pitch). Benchmarked depth error is FLAT from 0 to
#: 50 degrees of yaw -- 27-36 mm MAE throughout -- because PnP solves for
#: rotation explicitly rather than assuming a frontal face. The old floor of
#: cos(60) punished rotation that costs no accuracy, which contradicted the
#: system's own measured behaviour. It now only bites past ~65 degrees, where
#: landmarks genuinely start self-occluding.
POSE_FLOOR = 0.260

WEIGHTS = {
    "detection": 0.30,
    "cue_match": 0.30,
    "stability": 0.25,
    "pose": 0.15,
}


def _score(value: float, good: float, bad: float) -> float:
    """Map an error onto [1, 0], where `good` scores 1.0 and `bad` scores 0.

    Two-point rather than linear-to-zero, so that "as good as this system
    gets" reads as full marks instead of as a fraction of an unreachable
    ideal.
    """
    if not math.isfinite(value) or bad <= good:
        return 0.0
    return float(np.clip((bad - value) / (bad - good), 0.0, 1.0))


@dataclass
class QualityReport:
    """Sub-scores in [0, 1], the headline percentage, and a plain-English verdict."""

    detection: float | None
    cue_match: float | None
    stability: float | None
    pose: float
    overall: float
    stability_cm: float
    """Observed depth dispersion in cm -- shown next to the stability score."""
    verdict: str
    """'stable' | 'settling' | 'low-confidence'."""
    verdict_detail: str
    hints: list[str] = field(default_factory=list)
    """Actionable suggestions, e.g. 'Turn forward'."""

    def as_dict(self) -> dict:
        def pct(v):
            return None if v is None else round(100.0 * v, 1)
        return {
            "overall": round(100.0 * self.overall, 1),
            "detection": pct(self.detection),
            "cue_match": pct(self.cue_match),
            "stability": pct(self.stability),
            "pose": pct(self.pose),
            "stability_cm": round(self.stability_cm, 2),
            "verdict": self.verdict,
            "verdict_detail": self.verdict_detail,
            "hints": self.hints,
        }


class QualityMonitor:
    """Tracks recent history per face and produces a QualityReport per frame."""

    def __init__(self, history: int = 45, min_history: int = 8):
        self.history = history
        self.min_history = min_history
        self._depth: dict[int, deque] = {}

    def reset(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._depth.clear()
        else:
            self._depth.pop(track_id, None)

    def update(
        self,
        track_id: int,
        distance_mm: float,
        face_span_px: float,
        reproj_rms_px: float,
        pnp_distance_mm: float | None,
        ipd_distance_mm: float | None,
        yaw_deg: float,
        pitch_deg: float,
        confirmed: bool,
    ) -> QualityReport:
        buf = self._depth.setdefault(track_id, deque(maxlen=self.history))
        if math.isfinite(distance_mm):
            buf.append(float(distance_mm))

        # -- detection ------------------------------------------------------
        detection = None
        if face_span_px > 1.0 and math.isfinite(reproj_rms_px):
            detection = _score(reproj_rms_px / face_span_px,
                               DETECTION_GOOD, DETECTION_BAD)

        # -- cue agreement --------------------------------------------------
        cue_match = None
        if (pnp_distance_mm and ipd_distance_mm
                and pnp_distance_mm > 1.0 and ipd_distance_mm > 1.0):
            rel = abs(pnp_distance_mm - ipd_distance_mm) / pnp_distance_mm
            cue_match = _score(rel, CUE_GOOD, CUE_BAD)

        # -- stability ------------------------------------------------------
        stability, stability_cm = None, 0.0
        if len(buf) >= self.min_history:
            arr = np.asarray(buf, dtype=float)
            spread = float(np.std(arr))
            stability_cm = spread / 10.0
            mean = float(np.mean(arr))
            if mean > 1.0:
                stability = _score(spread / mean,
                                   STABILITY_GOOD, STABILITY_BAD)

        # -- pose / conditioning --------------------------------------------
        frontality = (math.cos(math.radians(yaw_deg))
                      * math.cos(math.radians(pitch_deg)))
        pose = float(np.clip(
            (frontality - POSE_FLOOR) / (1.0 - POSE_FLOOR), 0.0, 1.0
        ))

        # -- weighted mean over available terms ------------------------------
        parts = {
            "detection": detection,
            "cue_match": cue_match,
            "stability": stability,
            "pose": pose,
        }
        total_w = sum(WEIGHTS[k] for k, v in parts.items() if v is not None)
        overall = (
            sum(WEIGHTS[k] * v for k, v in parts.items() if v is not None) / total_w
            if total_w > 0 else 0.0
        )

        verdict, detail = self._verdict(
            overall, confirmed, len(buf), stability, cue_match
        )
        return QualityReport(
            detection=detection, cue_match=cue_match, stability=stability,
            pose=pose, overall=overall, stability_cm=stability_cm,
            verdict=verdict, verdict_detail=detail,
            hints=self._hints(yaw_deg, pitch_deg, cue_match, detection),
        )

    # -- helpers ------------------------------------------------------------

    def _verdict(self, overall, confirmed, n, stability, cue_match):
        if not confirmed or n < self.min_history:
            return "settling", "Acquiring track - hold still for a moment"
        if cue_match is not None and cue_match < 0.25:
            return ("low-confidence",
                    "Depth cues disagree - check calibration or face profile")
        if stability is not None and stability < 0.25:
            return "low-confidence", "Reading is fluctuating - reduce motion"
        if overall >= 0.70:
            return "stable", "Measurement accepted - high confidence"
        if overall >= 0.45:
            return "settling", "Usable, but confidence is moderate"
        return "low-confidence", "Measurement unreliable in current conditions"

    @staticmethod
    def _hints(yaw_deg, pitch_deg, cue_match, detection) -> list[str]:
        hints: list[str] = []
        if abs(yaw_deg) > 45.0:
            hints.append("Turn forward")
        if pitch_deg > 18.0:
            hints.append("Lower your chin")
        elif pitch_deg < -18.0:
            hints.append("Raise your chin")
        if detection is not None and detection < 0.35:
            hints.append("Improve lighting or hold steady")
        if cue_match is not None and cue_match < 0.30:
            hints.append("Run per-subject calibration")
        return hints