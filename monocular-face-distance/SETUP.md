# Setup

## One command

```bash
cd facedist-proj
bash setup.sh
```

Creates the virtualenv, installs everything, downloads the face model, checks
all 16 package files are present, runs the tests, and probes the camera. It
stops at the first real failure and tells you what to do.

Then:

```bash
source .venv/bin/activate
python -m facedist.server
```

The browser opens on its own at <http://127.0.0.1:8000>.

---

## If you prefer to do it by hand

```bash
cd facedist-proj
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "from facedist.landmarks import ensure_task_model; print(ensure_task_model())"
python -m pytest tests -q          # expect: 32 passed
python -m facedist.server
```

**Using zsh?** Do not paste trailing `# comments` — zsh does not strip them
interactively and they are passed to the command as arguments.

---

## Every new terminal

```bash
cd ~/Desktop/hacktronix/facedist-proj
source .venv/bin/activate
```

Skipping this is the most common cause of `ModuleNotFoundError` after a working
install. In VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose
the one under `.venv`, and the built-in terminal will activate it for you.

---

## Expected layout

You must run commands from the folder that **contains** `facedist/`, never from
inside it.

```
facedist-proj/            <- open THIS in VS Code, run commands HERE
├── setup.sh
├── SETUP.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── facedist/
│   ├── __init__.py
│   ├── baseline.py       spec equations, verbatim
│   ├── bearing.py        theta and phi
│   ├── camera.py         intrinsics, undistortion
│   ├── cli.py            OpenCV window + single-image mode
│   ├── estimator.py      solvePnP -> depth + bearing
│   ├── evaluate.py       the benchmark
│   ├── face_model.py     metric 3D face model
│   ├── landmarks.py      MediaPipe wrapper
│   ├── overlay.py        video overlay
│   ├── pipeline.py       orchestration
│   ├── quality.py        confidence scoring
│   ├── server.py         web panel
│   ├── simulation.py     synthetic ground truth
│   ├── tracking.py       Kalman filter
│   └── web/index.html    the UI
├── tests/test_facedist.py
└── tools/
    ├── calibrate_camera.py
    └── calibrate_ipd.py
```

`facedist/` must contain **15 .py files plus web/**. Missing any of them means
the extraction was incomplete -- re-extract rather than hand-copying.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named 'facedist'` | Wrong folder, or venv not active | `cd` to the folder holding `facedist/`, then `source .venv/bin/activate` |
| `No module named 'facedist.server'` | Incomplete extraction | Re-extract the zip; do not hand-copy files |
| `externally-managed-environment` | No virtualenv | `python3 -m venv .venv && source .venv/bin/activate` |
| `attempted relative import` | Ran the file directly | Use `python -m facedist.server`, not `python facedist/server.py` |
| `ERROR: file or directory not found: #` | zsh passed your comment as an argument | Drop the `#` comment from the command |
| Page 404s at `/` | `index.html` in the wrong place | Must be `facedist/web/index.html` |
| Port already in use | An older copy is still running | `python -m facedist.server --port 8010` |
| Camera light on, nothing visible | Browser never opened | Go to <http://127.0.0.1:8000> |
| Numbers all `—` | No face detected | Sit 50-100 cm away, decent light |

Diagnose the camera on its own, with no server involved:

```bash
python -m facedist.server --check
```

---

## Version notes

Built and tested against OpenCV 4.13 and MediaPipe 0.10.33. Newer majors
(OpenCV 5.x, MediaPipe 1.x) generally work -- `python -m pytest tests -q`
exercises every OpenCV call the estimator makes, so if it reports 32 passed you
are fine. If it does not:

```bash
pip install "opencv-python>=4.8,<5" "mediapipe>=0.10,<1"
```

OpenCV must be at least 4.4 -- `SOLVEPNP_SQPNP` does not exist before that.
