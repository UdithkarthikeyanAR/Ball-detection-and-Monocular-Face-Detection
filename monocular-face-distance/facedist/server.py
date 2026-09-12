"""Web UI server: MJPEG video plus JSON telemetry.

Why a browser rather than Qt or a raw OpenCV window: all the computer vision
stays in Python where it belongs, and all the layout, typography and charting
happens in HTML where it is actually pleasant to build. It also means the panel
can be shown on a second screen or a phone during a demo without changing
anything.

Two endpoints carry the load:

``/video.mjpg``   multipart JPEG stream of the annotated frame
``/api/state``    the numbers, polled by the page at ~12 Hz

The capture and inference loop runs in one background thread and publishes the
latest frame and telemetry under a lock. The HTTP handlers never block on the
camera, so a slow or disconnected browser cannot stall inference -- readers get
whatever the newest frame is and drop the rest, which is the correct behaviour
for live measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from .baseline import BaselineWidthEstimator, DEFAULT_FACE_WIDTH_MM
from .face_model import DEFAULT_IPD_MM
from .overlay import draw_banner, draw_face, draw_input_strip
from .pipeline import FaceDistancePipeline

WEB_DIR = Path(__file__).parent / "web"
HISTORY = 240


class Engine:
    """Capture + inference loop, publishing the latest frame and telemetry."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.state: dict = {"faces": [], "fps": 0.0, "ok": False,
                            "message": "starting camera"}
        self.hist_ours: deque = deque(maxlen=HISTORY)
        self.hist_base: deque = deque(maxlen=HISTORY)
        self.hist_sigma: deque = deque(maxlen=HISTORY)
        self.show_smoothed = True
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self.running = False

    # -- controls exposed to the UI ----------------------------------------

    def reset(self):
        with self.lock:
            self.hist_ours.clear()
            self.hist_base.clear()
            self.hist_sigma.clear()
        if getattr(self, "pipeline", None):
            self.pipeline.tracker.reset()
            self.pipeline.quality.reset()

    def toggle_smoothed(self):
        self.show_smoothed = not self.show_smoothed
        return self.show_smoothed

    # -- main loop ----------------------------------------------------------

    def _loop(self):
        a = self.args
        source = a.source if a.source is not None else a.camera

        # Opening a network URL blocks until the connection succeeds or the
        # OS times out, which can be 30s. Say so, or the panel looks hung.
        if isinstance(source, str) and "://" in source:
            with self.lock:
                self.state = dict(self.state, ok=False, message=(
                    f"Connecting to {source} ... (this can take up to 30s if "
                    "the address is wrong)"))

        cap = cv2.VideoCapture(source)
        if a.source is None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        if not cap.isOpened():
            if isinstance(source, str) and "://" in source:
                msg = (f"Cannot reach {source}. Check that the URL is exactly "
                       "what the phone app shows (many need a /video or "
                       "/videofeed suffix), that both devices are on the same "
                       "Wi-Fi, and that the address opens in a browser first.")
            else:
                msg = (f"Cannot open video source {source!r}. "
                       "Close any other app using the camera (Zoom, Meet, OBS), "
                       "or try --camera 1.")
            print(f"\n  ERROR: {msg}\n")
            with self.lock:
                self.state = {"faces": [], "fps": 0.0, "ok": False,
                              "fatal": True, "message": msg}
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or a.width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or a.height

        ipd = a.ipd_mm or DEFAULT_IPD_MM
        if a.ipd_file and Path(a.ipd_file).exists():
            ipd = float(json.loads(Path(a.ipd_file).read_text())["ipd_mm"])

        self.pipeline = FaceDistancePipeline(
            width=width, height=height, calibration_path=a.calibration,
            ipd_mm=ipd, hfov_deg=a.hfov, max_faces=a.max_faces,
        )
        intr = self.pipeline.intrinsics
        baseline = BaselineWidthEstimator(intr, face_width_mm=a.face_width_mm)

        # A network stream, a file, and a local device fail in different ways
        # and must be recovered differently. Treating them alike is what makes
        # a dropped IP-camera frame spin the CPU.
        is_stream = isinstance(source, str) and "://" in source
        is_file = isinstance(source, str) and not is_stream
        if is_stream:
            # Keep only the newest frame; otherwise latency grows without bound
            # as decoding falls behind the network.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        failures = 0
        warned_size = False

        while self.running:
            ok, frame = cap.read()

            if not ok:
                failures += 1
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)          # never spin
                    continue
                if is_stream:
                    # Exponential-ish backoff, then rebuild the capture. A
                    # sleep here is mandatory: without it a dead stream burns
                    # a full core and starves inference.
                    delay = min(0.4 * failures, 5.0)
                    with self.lock:
                        self.state = dict(
                            self.state,
                            ok=False,
                            message=(f"Stream dropped ({failures}). "
                                     f"Reconnecting in {delay:.1f}s ..."),
                        )
                    time.sleep(delay)
                    cap.release()
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if failures > 40:
                        with self.lock:
                            self.state = dict(
                                self.state, ok=False, fatal=True,
                                message=("Stream unreachable after 40 attempts. "
                                         "Check the URL and that phone and "
                                         "laptop are on the same network."),
                            )
                        break
                    continue
                # Local device: a failed read means it was unplugged or taken.
                time.sleep(0.05)
                if failures > 20:
                    with self.lock:
                        self.state = dict(self.state, ok=False, fatal=True,
                                          message="Camera stopped returning frames.")
                    break
                continue

            failures = 0

            # An IP camera can hand back a different resolution than we sized
            # the intrinsics for -- and after a reconnect it can change again.
            # Silently accepting that would make every distance wrong by the
            # scale ratio, so normalise to the configured size.
            if frame.shape[1] != width or frame.shape[0] != height:
                if not warned_size:
                    print(f"  note: source is {frame.shape[1]}x{frame.shape[0]}, "
                          f"resizing to {width}x{height} to match intrinsics")
                    warned_size = True
                frame = cv2.resize(frame, (width, height))

            if a.mirror and a.source is None:
                frame = cv2.flip(frame, 1)

            results = self.pipeline.process(frame)
            faces = []
            for r in results:
                base_mm = baseline.estimate(r.bbox)
                shown_mm = r.distance_mm if self.show_smoothed else r.raw_distance_mm
                draw_face(frame, r, intr, show_baseline_mm=base_mm)
                draw_input_strip(frame, r, intr, a.face_width_mm)
                faces.append(self._face_payload(r, base_mm, shown_mm, intr))

            if not intr.calibrated:
                draw_banner(
                    frame,
                    f"UNCALIBRATED  assuming {a.hfov:.0f} deg FOV "
                    f"- scale error up to 20%. Run tools/calibrate_camera.py",
                )

            if faces:
                primary = faces[0]
                with self.lock:
                    self.hist_ours.append(primary["depth_cm"])
                    self.hist_base.append(
                        primary["baseline_cm"]
                        if math.isfinite(primary["baseline_cm"]) else None
                    )
                    self.hist_sigma.append(primary["sigma_cm"])

            ok_jpeg, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), a.jpeg_quality]
            )
            with self.lock:
                if ok_jpeg:
                    self.jpeg = buf.tobytes()
                self.state = {
                    "ok": True,
                    "message": "" if faces else "no face detected",
                    "fps": round(self.pipeline.fps, 1),
                    "calibrated": intr.calibrated,
                    "smoothed": self.show_smoothed,
                    "intrinsics": {
                        "fx": round(intr.fx, 1), "cx": round(intr.cx, 1),
                        "cy": round(intr.cy, 1),
                        "hfov": round(intr.hfov_deg, 1),
                        "width": width, "height": height,
                    },
                    "face_width_m": a.face_width_mm / 1000.0,
                    "ipd_mm": ipd,
                    "faces": faces,
                    "history": {
                        "ours": list(self.hist_ours),
                        "baseline": list(self.hist_base),
                        "sigma": list(self.hist_sigma),
                    },
                }
        cap.release()
        self.pipeline.close()

    @staticmethod
    def _face_payload(r, base_mm, shown_mm, intr) -> dict:
        x1, _, x2, _ = r.bbox
        q = r.quality.as_dict() if r.quality else {}
        return {
            "id": r.track_id,
            "depth_cm": round(shown_mm / 10.0, 2),
            "depth_m": round(shown_mm / 1000.0, 4),
            "sigma_cm": round(r.sigma_mm / 10.0, 2),
            "theta_deg": round(r.theta_deg, 2),
            "theta_rad": round(math.radians(r.theta_deg), 5),
            "phi_deg": round(r.phi_deg, 2),
            "sigma_theta_deg": round(r.sigma_theta_deg, 3),
            "baseline_cm": round(base_mm / 10.0, 2)
            if math.isfinite(base_mm) else float("nan"),
            "delta_cm": round((shown_mm - base_mm) / 10.0, 2)
            if math.isfinite(base_mm) else 0.0,
            "yaw": round(r.yaw, 1), "pitch": round(r.pitch, 1),
            "roll": round(r.roll, 1),
            "velocity_cm_s": round(r.velocity_mm_s / 10.0, 1),
            "confirmed": r.confirmed,
            "bearing_source": r.bearing_source,
            "quality": q,
            "inputs": {
                "x_px": round(r.ref_x_px, 1),
                "w_px": round(abs(x2 - x1), 1),
                "f_px": round(intr.fx, 1),
                "cx_px": round(intr.cx, 1),
            },
        }


def create_app(engine: Engine) -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        page = WEB_DIR / "index.html"
        if not page.exists():
            # A bare 404 here is useless -- it looks identical to a broken
            # server. Say exactly which path is missing and how to fix it.
            return Response(
                "<html><body style='font:14px ui-monospace,monospace;"
                "background:#0B0E14;color:#E4E9F2;padding:40px;line-height:1.7'>"
                "<h2 style='color:#FF6B5A;margin:0 0 16px'>index.html not found</h2>"
                f"<p>The server expected it at:</p><p style='color:#7FD4C1'>{page}</p>"
                "<p>It is probably at the project root instead of inside the "
                "<code>facedist</code> package. Fix with:</p>"
                "<pre style='background:#131823;padding:14px;border-radius:6px'>"
                f"mkdir -p {WEB_DIR}\nmv web/index.html {WEB_DIR}/</pre>"
                "<p>Then restart the server.</p></body></html>",
                status=500, mimetype="text/html",
            )
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/video.mjpg")
    def video():
        def gen():
            boundary = b"--frame\r\n"
            while engine.running:
                with engine.lock:
                    data = engine.jpeg
                if data is None:
                    time.sleep(0.03)
                    continue
                yield (boundary + b"Content-Type: image/jpeg\r\n"
                       + f"Content-Length: {len(data)}\r\n\r\n".encode()
                       + data + b"\r\n")
                time.sleep(1 / 60)
        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/state")
    def state():
        with engine.lock:
            return jsonify(engine.state)

    @app.post("/api/action")
    def action():
        name = (request.json or {}).get("action", "")
        if name == "reset":
            engine.reset()
        elif name == "smoothed":
            engine.toggle_smoothed()
        else:
            return jsonify({"ok": False, "error": f"unknown action {name!r}"}), 400
        return jsonify({"ok": True, "smoothed": engine.show_smoothed})

    return app


def _check_camera(args) -> int:
    """Probe the camera and report, without starting a web server."""
    src = args.source if args.source is not None else args.camera
    print(f"\n  Opening video source {src!r} ...")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        if isinstance(src, str) and "://" in src:
            # "Connection refused" means the host answered but the port is
            # closed -- almost always the phone app not actually serving,
            # rather than a network problem. Worth saying, because the
            # generic webcam advice sends people down the wrong path.
            print("  FAILED to open the stream.\n"
                  "  - Is the server actually STARTED in the phone app?\n"
                  "    Opening the app is not enough; tap Start server.\n"
                  "  - Keep the app in the foreground; it stops when\n"
                  "    backgrounded or when the screen locks.\n"
                  "  - Confirm the IP belongs to the PHONE, not this laptop:\n"
                  "        ip addr show | grep 'inet '\n"
                  "  - Check the port and path the app displays. Try the\n"
                  "    single-frame endpoint in a browser first:\n"
                  "        <base-url>/shot.jpg\n"
                  "  - Use http, not https.\n")
        else:
            print("  FAILED. Nothing could open that source.\n"
                  "  - Close Zoom / Meet / Teams / OBS and try again\n"
                  "  - Try a different index:  --camera 1  (then 2)\n"
                  "  - On macOS, grant camera access to your terminal in\n"
                  "    System Settings > Privacy & Security > Camera\n")
        return 1
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("  Opened, but returned no frames. The device is likely held "
              "by another app.\n")
        return 1
    h, w = frame.shape[:2]
    print(f"  OK - {w}x{h} frame captured.\n"
          f"  Now run:  python -m facedist.server\n")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="facedist web instrument panel")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--source", default=None, help="video file; overrides --camera")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--calibration", default=None)
    p.add_argument("--ipd-file", default=None)
    p.add_argument("--ipd-mm", type=float, default=None)
    p.add_argument("--hfov", type=float, default=60.0)
    p.add_argument("--max-faces", type=int, default=2)
    p.add_argument("--face-width-mm", type=float, default=DEFAULT_FACE_WIDTH_MM)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--open-browser", action="store_true", default=True,
                   help="open the panel automatically on start (default)")
    p.add_argument("--no-open-browser", dest="open_browser",
                   action="store_false")
    p.add_argument("--check", action="store_true",
                   help="test the camera and exit, without starting the server")
    p.add_argument("--mirror", action="store_true", default=True)
    p.add_argument("--no-mirror", dest="mirror", action="store_false")
    args = p.parse_args(argv)

    if args.check:
        return _check_camera(args)

    # Fail loudly *before* touching the camera if the port is taken -- otherwise
    # the camera light comes on, Flask dies, and it looks like the app hung.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((args.host, args.port))
    except OSError:
        print(f"\n  ERROR: port {args.port} is already in use.")
        print(f"  Another copy is probably still running. Either open")
        print(f"  http://{args.host}:{args.port} in your browser, or start this")
        print(f"  one on a free port:  python -m facedist.server --port 8010\n")
        return 1
    finally:
        probe.close()

    # Check the page exists before the camera light comes on, so a missing
    # asset is not mistaken for a camera or browser problem.
    if not (WEB_DIR / "index.html").exists():
        print(f"\n  ERROR: missing {WEB_DIR / 'index.html'}")
        print(f"  Create that folder and file, or move an existing one:")
        print(f"      mkdir -p {WEB_DIR}")
        print(f"      mv web/index.html {WEB_DIR}/\n")
        return 1

    engine = Engine(args)
    engine.start()
    app = create_app(engine)

    url = f"http://{args.host}:{args.port}"
    bar = "=" * 58
    print(f"\n{bar}\n  facedist panel is running\n\n"
          f"      OPEN THIS IN YOUR BROWSER:   {url}\n\n"
          f"  Press Ctrl+C here to stop.\n{bar}\n")

    if args.open_browser:
        # Delay past Flask's bind so the first request does not race the server.
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    try:
        app.run(host=args.host, port=args.port, threaded=True,
                debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())