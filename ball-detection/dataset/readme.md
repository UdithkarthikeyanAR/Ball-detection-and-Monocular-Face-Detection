# Ball Detection

A real-time ball detection system built using **YOLO11**.

## Features

- 🎾 Real-time ball detection
- 📱 Phone camera support (IP Webcam)
- ⚡ CUDA GPU acceleration
- 📊 Live FPS and confidence display

## Dataset

The training dataset is **not included** in this repository.

Place it in:

```
dataset/
├── images/
├── labels/
└── data.yaml
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python detect.py
```

## Train

```bash
python train.py
```