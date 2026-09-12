"""BallVision web panel — Optimized"""

import threading
import time
from collections import deque

import cv2
import torch
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

MODEL_PATH = "models/best.pt"

# Use phone camera
CAMERA = "http://192.168.1.2:8080/video"

# Use laptop webcam instead:
# CAMERA = 0

CONF = 0.35
HISTORY = 120

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

IMG_SIZE = 640

JPEG_QUALITY = 60

SKIP_FRAMES = 2

# ------------------------------------------------------------------

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nRunning on {device.upper()}")

model = YOLO(MODEL_PATH)
model.to(device)

_lock = threading.Lock()

_jpeg = None

_state = {
    "ok": False,
    "fps": 0,
    "count": 0,
    "confidence": 0,
    "message": "Starting..."
}

_hist_cnt = deque(maxlen=HISTORY)
_hist_conf = deque(maxlen=HISTORY)


def capture_loop():

    global _jpeg
    global _state

    src = CAMERA

    print(f"Connecting to {src}")

    cap = cv2.VideoCapture(src)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():

        with _lock:
            _state = {
                "ok": False,
                "fps": 0,
                "count": 0,
                "confidence": 0,
                "message": f"Cannot open source: {src}"
            }

        return

    prev = time.perf_counter()

    frame_id = 0

    while True:

        ok, frame = cap.read()

        if not ok:

            time.sleep(0.1)

            continue

        frame_id += 1

        if frame_id % SKIP_FRAMES != 0:
            continue

        frame = cv2.flip(frame, 1)

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        results = model(
            frame,
            conf=CONF,
            imgsz=IMG_SIZE,
            device=device,
            half=(device == "cuda"),
            verbose=False,
        )

        annotated = results[0].plot()

        now = time.perf_counter()

        fps = 1.0 / max(now - prev, 1e-6)

        prev = now

        boxes = results[0].boxes

        count = len(boxes)

        conf = 0.0

        labels = []

        if count:

            conf = float(max(b.conf for b in boxes)) * 100

            names = model.names

            labels = list({
                names.get(int(b.cls), "ball")
                for b in boxes
            })

        with _lock:

            _hist_cnt.append(count)
            _hist_conf.append(round(conf, 1))

            _state = {
                "ok": True,
                "message": "",
                "fps": round(fps, 1),
                "count": count,
                "confidence": round(conf, 1),
                "labels": labels,
                "history": {
                    "count": list(_hist_cnt),
                    "conf": list(_hist_conf)
                }
            }

        _, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                JPEG_QUALITY
            ]
        )

        with _lock:
            _jpeg = buffer.tobytes()


threading.Thread(
    target=capture_loop,
    daemon=True
).start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video.mjpg")
def video():

    def generate():

        while True:

            with _lock:
                frame = _jpeg

            if frame is None:

                time.sleep(0.03)

                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )

            time.sleep(1 / 30)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/state")
def state():

    with _lock:

        return jsonify(_state)


if __name__ == "__main__":

    import webbrowser

    print("\nBallVision running at http://127.0.0.1:8000\n")

    threading.Timer(
        1,
        lambda: webbrowser.open("http://127.0.0.1:8000")
    ).start()

    app.run(
        host="127.0.0.1",
        port=8000,
        threaded=True,
        debug=False,
        use_reloader=False,
    )