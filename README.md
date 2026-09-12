# Ball Detection and Monocular Face Detection

A computer vision project that combines ball detection with monocular face detection, distance estimation, and face-position analysis.

## Features

- Ball detection using computer vision
- Monocular face detection
- Approximate face-distance estimation
- Face deviation-angle estimation
- Face bearing estimation
- Face tracking
- Camera calibration tools
- Testing and evaluation tools
- BallVision web interface

## Project Modules

### Ball Detection

The ball detection module detects and tracks balls using computer vision and YOLO-based models.

Main files include:

- `detect.py`
- `train.py`
- `train_multiball.py`
- `app.py`
- `phone_test.py`

### Monocular Face Detection

The monocular face detection module estimates face position and approximate distance using a single camera.

It includes:

- Face landmark detection
- Face-distance estimation
- Bearing and deviation-angle calculation
- Camera calibration
- Face tracking
- Simulation and evaluation tools

Detailed documentation is available here:

[Monocular Face Distance Documentation](./monocular-face-distance/README.md)

Setup instructions:

[Setup Guide](./monocular-face-distance/SETUP.md)

## Project Structure

```text
Ball-detection-and-Monocular-Face-Detection/
├── dataset/
├── images/
├── models/
├── monocular-face-distance/
├── monocular-face-distance-backup/
├── runs/
├── static/
├── templates/
├── app.py
├── detect.py
├── train.py
├── train_multiball.py
├── requirements.txt
└── README.md
