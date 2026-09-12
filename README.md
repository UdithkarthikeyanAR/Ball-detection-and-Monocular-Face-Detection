# Ball Detection and Monocular Face Detection

A computer vision project that combines **ball detection** with **monocular face detection**, distance estimation, and face-position analysis.

## Overview

This project contains two main computer vision modules:

1. **Ball Detection** – Detects and tracks balls using computer vision and YOLO-based models.
2. **Monocular Face Detection** – Detects faces using a single camera and estimates face distance, bearing, and deviation angle.

The project is developed for educational, research, and practical computer vision applications.

## Features

- Ball detection using computer vision
- YOLO-based object detection
- Monocular face detection
- Approximate face-distance estimation
- Face deviation-angle estimation
- Face bearing estimation
- Face tracking
- Face landmark detection
- Camera calibration tools
- Testing and evaluation tools
- Simulation support
- BallVision web interface

## Project Modules

## 1. Ball Detection

The ball detection module detects and tracks balls using computer vision and YOLO-based models.

### Main Files

- `detect.py` – Ball detection
- `train.py` – Model training
- `train_multiball.py` – Multi-ball training
- `app.py` – BallVision web application
- `phone_test.py` – Phone-camera testing
- `models/` – Model files
- `dataset/` – Dataset files
- `runs/` – Training and detection results

## 2. Monocular Face Detection

The monocular face detection module estimates face position and approximate distance using a single camera.

### Features

- Face landmark detection
- Face-distance estimation
- Bearing calculation
- Deviation-angle calculation
- Face tracking
- Camera calibration
- Simulation
- Evaluation and testing tools

### Main Files

- `estimator.py` – Face-distance estimation
- `face_model.py` – Face model handling
- `landmarks.py` – Face landmark processing
- `camera.py` – Camera-related functions
- `bearing.py` – Bearing calculation
- `tracking.py` – Face tracking
- `pipeline.py` – Detection pipeline
- `server.py` – Server functionality
- `evaluate.py` – Evaluation tools
- `simulation.py` – Simulation tools
- `calibrateipd.py` – IPD calibration
- `tools/` – Camera and IPD calibration tools
- `tests/` – Testing files

## Documentation

Detailed documentation for the monocular face detection module is available here:

- [Monocular Face Distance README](./monocular-face-distance/README.md)
- [Monocular Face Distance Setup Guide](./monocular-face-distance/SETUP.md)

## Project Structure

```text
Ball-detection-and-Monocular-Face-Detection/
├── dataset/
├── images/
├── models/
├── monocular-face-distance/
│   ├── facedist/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── bearing.py
│   │   ├── calibrateipd.py
│   │   ├── camera.py
│   │   ├── cli.py
│   │   ├── estimator.py
│   │   ├── evaluate.py
│   │   ├── face_model.py
│   │   ├── landmarks.py
│   │   ├── pipeline.py
│   │   ├── server.py
│   │   ├── simulation.py
│   │   └── tracking.py
│   ├── tests/
│   ├── tools/
│   ├── README.md
│   ├── SETUP.md
│   ├── pyproject.toml
│   └── requirements.txt
├── monocular-face-distance-backup/
├── runs/
├── static/
├── templates/
├── app.py
├── detect.py
├── phone_test.py
├── train.py
├── train_multiball.py
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- OpenCV
- YOLO
- Computer Vision
- Face Detection
- Face Landmark Detection
- Monocular Distance Estimation
- HTML
- Web Applications

## Installation

Install the main project dependencies:

```bash
pip install -r requirements.txt
```

For the monocular face detection module, navigate to its directory:

```bash
cd monocular-face-distance
```

Then install its dependencies:

```bash
pip install -r requirements.txt
```

For additional setup and calibration instructions, refer to:

```text
monocular-face-distance/SETUP.md
```

## Usage

### Ball Detection

Use the main project files to run the ball detection system.

```bash
python detect.py
```

### BallVision Web Interface

Run the web application using:

```bash
python app.py
```

### Monocular Face Detection

Navigate to the monocular face detection module:

```bash
cd monocular-face-distance
```

Follow the instructions in the module documentation:

```text
README.md
SETUP.md
```

## Documentation Links

- [Monocular Face Detection Documentation](./monocular-face-distance/README.md)
- [Monocular Face Detection Setup](./monocular-face-distance/SETUP.md)

## Contributors

- **UdithkarthikeyanAR**
- **Praveen172213**

## Purpose

This project is developed for educational, research, and practical applications in computer vision, object detection, face detection, and distance estimation.

## License

This project is intended for educational and development purposes.
