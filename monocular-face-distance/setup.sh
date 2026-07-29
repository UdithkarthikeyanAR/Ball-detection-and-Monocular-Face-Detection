#!/usr/bin/env bash
# One-command setup. Run:  bash setup.sh
#
# Does the whole thing: virtualenv, dependencies, face model download, and a
# full test run. Every step is checked, and it stops at the first real failure
# with a message that says what to do -- rather than continuing and failing
# later somewhere confusing.
set -u

BOLD=$'\033[1m'; GREEN=$'\033[32m'; RED=$'\033[31m'; AMBER=$'\033[33m'; OFF=$'\033[0m'
step() { printf "\n${BOLD}==> %s${OFF}\n" "$1"; }
ok()   { printf "  ${GREEN}OK${OFF}  %s\n" "$1"; }
bad()  { printf "  ${RED}FAILED${OFF}  %s\n" "$1"; }
warn() { printf "  ${AMBER}!${OFF}  %s\n" "$1"; }

cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
printf "${BOLD}facedist setup${OFF}\n%s\n" "$ROOT"

# -- 0. sanity: are we in the right folder? ---------------------------------
if [ ! -f "facedist/__init__.py" ]; then
  bad "No facedist/ package here."
  echo "  You are in the wrong folder. cd to the one containing facedist/"
  exit 1
fi

# -- 1. python --------------------------------------------------------------
step "Checking Python"
PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null) || continue
    maj=${v%%.*}; min=${v##*.}
    if [ "$maj" = "3" ] && [ "$min" -ge 9 ] && [ "$min" -le 13 ]; then PY="$c"; break; fi
  fi
done
if [ -z "$PY" ]; then
  bad "Need Python 3.9-3.13 (mediapipe does not publish wheels outside that)."
  exit 1
fi
ok "$PY $($PY -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')"

# -- 2. virtualenv ----------------------------------------------------------
step "Creating virtual environment (.venv)"
if [ -d ".venv" ]; then
  warn "Reusing the existing .venv (delete it to start clean)"
else
  if ! "$PY" -m venv .venv 2>/dev/null; then
    bad "venv creation failed."
    echo "  Install it:  sudo apt install python3-venv python3-pip"
    exit 1
  fi
  ok "created"
fi
# shellcheck disable=SC1091
. .venv/bin/activate || { bad "could not activate .venv"; exit 1; }
ok "activated"

# -- 3. dependencies --------------------------------------------------------
step "Installing dependencies (a few hundred MB, be patient)"
python -m pip install --upgrade pip --quiet
if ! python -m pip install -r requirements.txt --quiet; then
  bad "pip install failed."
  echo "  Low memory?   python -m pip install --no-cache-dir -r requirements.txt"
  echo "  Build errors? sudo apt install build-essential python3-dev"
  exit 1
fi
ok "installed"

step "Verifying imports"
python - <<'PY'
import importlib, sys
need = [("numpy", "__version__"), ("cv2", "__version__"), ("scipy", "__version__"),
        ("flask", None), ("mediapipe", "__version__"), ("pytest", "__version__")]
missing = []
for name, attr in need:
    try:
        m = importlib.import_module(name)
        print(f"  {name:11} {getattr(m, attr, 'ok') if attr else 'ok'}")
    except Exception as e:
        print(f"  {name:11} MISSING ({e.__class__.__name__})"); missing.append(name)
sys.exit(1 if missing else 0)
PY
[ $? -ne 0 ] && { bad "some packages did not install"; exit 1; }
ok "all present"

# -- 4. package layout ------------------------------------------------------
step "Checking package layout"
MISS=0
for f in __init__.py baseline.py bearing.py camera.py cli.py estimator.py \
         evaluate.py face_model.py landmarks.py overlay.py pipeline.py \
         quality.py server.py simulation.py tracking.py web/index.html; do
  [ -f "facedist/$f" ] || { bad "missing facedist/$f"; MISS=1; }
done
[ "$MISS" -eq 1 ] && { echo "  Re-extract the zip -- files are missing."; exit 1; }
ok "all 16 files present"

python -c "from facedist.server import create_app" 2>/dev/null \
  && ok "facedist.server imports" || { bad "facedist.server will not import"; exit 1; }

# -- 5. face model ----------------------------------------------------------
step "Downloading the MediaPipe face model (cached, ~3 MB)"
if python -c "from facedist.landmarks import ensure_task_model; print('  ->', ensure_task_model())" 2>/dev/null; then
  ok "model ready"
else
  warn "download failed -- you are offline, or MediaPipe still has the legacy API."
  warn "Not fatal now, but do this before demoing on untrusted Wi-Fi."
fi

# -- 6. tests ---------------------------------------------------------------
step "Running the test suite (no camera needed)"
if python -m pytest tests -q; then
  ok "tests pass"
else
  bad "tests failed -- paste the output above for diagnosis"
  exit 1
fi

# -- 7. camera --------------------------------------------------------------
step "Checking the camera"
python -m facedist.server --check || warn "No camera. Everything else is fine."

printf "\n${GREEN}${BOLD}Setup complete.${OFF}\n\n"
printf "  Start the panel:\n"
printf "      ${BOLD}source .venv/bin/activate${OFF}\n"
printf "      ${BOLD}python -m facedist.server${OFF}\n\n"
printf "  Then open http://127.0.0.1:8000 (it opens by itself).\n\n"
printf "  ${AMBER}Before you demo:${OFF} run tools/calibrate_camera.py with a printed\n"
printf "  chessboard. Uncalibrated, focal length is a guess and 10-20%% of that\n"
printf "  error lands directly in every distance you show.\n\n"
