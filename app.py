"""BallVision web panel — MJPEG stream + JSON telemetry.

Drop-in replacement for the OpenCV window in detect.py.
Run:  python app.py
Then open http://127.0.0.1:8000 in your browser.
"""

import math
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

# ── config ────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/best.pt"
CAMERA      = "http://192.168.1.4:8080/video"   # or 0 for laptop webcam
CONF        = 0.35
HISTORY     = 120
# ─────────────────────────────────────────────────────────────────────────

app   = Flask(__name__)
model = YOLO(MODEL_PATH)

# shared state written by the capture thread, read by Flask
_lock      = threading.Lock()
_jpeg      = None
_state     = {"ok": False, "fps": 0.0, "count": 0, "confidence": 0.0,
              "message": "starting"}
_hist_cnt  = deque(maxlen=HISTORY)
_hist_conf = deque(maxlen=HISTORY)


def _capture_loop():
    global _jpeg, _state

    src = CAMERA
    if isinstance(src, str) and "://" in src:
        print(f"  connecting to {src} …")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        with _lock:
            _state = {"ok": False, "fps": 0.0, "count": 0,
                      "confidence": 0.0,
                      "message": f"Cannot open source: {src!r}"}
        return

    prev = time.perf_counter()
    failures = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if isinstance(src, str) and "://" in src:
                delay = min(0.4 * failures, 5.0)
                with _lock:
                    _state = dict(_state, ok=False,
                                  message=f"Stream dropped. Reconnecting "
                                          f"({failures}) in {delay:.1f}s …")
                time.sleep(delay)
                cap.release()
                cap = cv2.VideoCapture(src)
                if failures > 40:
                    with _lock:
                        _state = dict(_state, ok=False,
                                      message="Stream unreachable after 40 tries.")
                    break
            else:
                time.sleep(0.05)
                if failures > 20:
                    break
            continue

        failures = 0
        frame = cv2.flip(frame, 1)

        results  = model(frame, verbose=False, conf=CONF)
        annotated = results[0].plot()

        now  = time.perf_counter()
        fps  = 1.0 / max(now - prev, 1e-3)
        prev = now

        boxes      = results[0].boxes
        count      = len(boxes)
        confidence = 0.0
        labels     = []
        if count > 0:
            confidence = float(max(b.conf for b in boxes)) * 100
            names = model.names or {}
            labels = list({names.get(int(b.cls), "ball") for b in boxes})

        with _lock:
            _hist_cnt.append(count)
            _hist_conf.append(round(confidence, 1))
            _state = {
                "ok":         True,
                "message":    "",
                "fps":        round(fps, 1),
                "count":      count,
                "confidence": round(confidence, 1),
                "labels":     labels,
                "history": {
                    "count": list(_hist_cnt),
                    "conf":  list(_hist_conf),
                },
            }

        _, buf = cv2.imencode(".jpg", annotated,
                              [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        with _lock:
            _jpeg = buf.tobytes()


threading.Thread(target=_capture_loop, daemon=True).start()


# ── routes ────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video.mjpg")
def video():
    def gen():
        while True:
            with _lock:
                data = _jpeg
            if data is None:
                time.sleep(0.03)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   + f"Content-Length: {len(data)}\r\n\r\n".encode()
                   + data + b"\r\n")
            time.sleep(1 / 60)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/state")
def state():
    with _lock:
        return jsonify(_state)


if __name__ == "__main__":
    import webbrowser
    print("\n  BallVision panel -> http://127.0.0.1:8000\n")
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    app.run(host="127.0.0.1", port=8000, threaded=True,
            debug=False, use_reloader=False)
