"""Face detection + dense landmark extraction via MediaPipe.

Why FaceMesh/FaceLandmarker rather than a detector plus a separate 68-point
regressor:

* one model pass gives both the face region and 468 landmarks, so there is no
  crop-and-rerun latency and no error from a loose detector box;
* it runs comfortably at 30+ FPS on a laptop CPU;
* it ships as a wheel, so there is nothing to convert or licence -- which
  matters, because the Basel 3DMM route needs an application-gated academic
  licence and a per-frame nonlinear fit that cannot hit real time.

**Two MediaPipe APIs are supported.** MediaPipe removed the legacy
``mp.solutions.face_mesh`` module in the 0.10.3x series, keeping only the Tasks
API. Installs in the wild span both, so this module probes for the legacy API
first and falls back to Tasks, normalising either one to the same
:class:`FaceObservation`. The Tasks path needs a ``.task`` bundle, which is
fetched once and cached under ``~/.cache/facedist``.

Landmark indices are identical across both paths (the Tasks model is the same
FaceMesh topology, 478 points with iris refinement), so the geometry layer does
not care which one ran.

The import is deliberately lazy so that the geometry, tracking and evaluation
code stays importable -- and testable -- on machines without MediaPipe.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Official float16 FaceLandmarker bundle for the Tasks API.
TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def default_model_dir() -> Path:
    return Path(
        os.environ.get("FACEDIST_CACHE",
                       Path.home() / ".cache" / "facedist")
    )


def ensure_task_model(path: Path | None = None) -> Path:
    """Return a local path to face_landmarker.task, downloading it once."""
    target = Path(path) if path else default_model_dir() / "face_landmarker.task"
    if target.exists() and target.stat().st_size > 100_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[facedist] downloading FaceLandmarker model -> {target}",
          file=sys.stderr)
    try:
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(TASK_MODEL_URL, tmp)
        tmp.replace(target)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the MediaPipe model from {TASK_MODEL_URL}.\n"
            f"Download it manually and place it at {target}, or pass\n"
            f"  LandmarkDetector(model_path=...)\n"
            f"Underlying error: {exc}"
        ) from exc
    return target


@dataclass(frozen=True)
class FaceObservation:
    """One detected face in one frame."""

    landmarks_px: np.ndarray   # (468+, 2) float64, image pixel coordinates
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2
    centre: tuple[float, float]


def _observation_from_points(pts: np.ndarray) -> FaceObservation:
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return FaceObservation(
        landmarks_px=pts,
        bbox=(float(x1), float(y1), float(x2), float(y2)),
        centre=(float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)),
    )


class LandmarkDetector:
    """Wrapper over MediaPipe FaceMesh (legacy) or FaceLandmarker (Tasks)."""

    def __init__(
        self,
        max_faces: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
        model_path: str | Path | None = None,
        prefer_tasks: bool = False,
    ):
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "mediapipe is required for live landmark detection.\n"
                "Install it with:  pip install mediapipe\n"
                "(The geometry, tracking and evaluation modules work without it.)"
            ) from exc

        self._mp = mp
        self.backend: str = ""
        self._frame_index = 0

        legacy = getattr(mp, "solutions", None) if not prefer_tasks else None
        if legacy is not None and hasattr(legacy, "face_mesh"):
            self._mesh = legacy.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.backend = "solutions.face_mesh"
            return

        # --- Tasks API (MediaPipe >= 0.10.3x) ------------------------------
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model = ensure_task_model(Path(model_path) if model_path else None)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self.backend = "tasks.FaceLandmarker"

    # -- inference ----------------------------------------------------------

    def detect(self, frame_rgb: np.ndarray) -> list[FaceObservation]:
        """Run the model on an RGB frame and return per-face observations."""
        height, width = frame_rgb.shape[:2]

        if self.backend == "solutions.face_mesh":
            frame_rgb.flags.writeable = False
            result = self._mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True
            faces = result.multi_face_landmarks or []
            return [
                _observation_from_points(
                    np.array([(lm.x * width, lm.y * height) for lm in f.landmark],
                             dtype=np.float64)
                )
                for f in faces
            ]

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        # Tasks VIDEO mode requires a monotonically increasing timestamp.
        self._frame_index += 1
        result = self._landmarker.detect_for_video(mp_image, self._frame_index * 33)
        return [
            _observation_from_points(
                np.array([(lm.x * width, lm.y * height) for lm in face],
                         dtype=np.float64)
            )
            for face in (result.face_landmarks or [])
        ]

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        obj = getattr(self, "_mesh", None) or getattr(self, "_landmarker", None)
        if obj is not None:
            try:
                obj.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def __enter__(self) -> "LandmarkDetector":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
