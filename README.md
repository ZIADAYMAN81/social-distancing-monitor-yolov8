# Social Distancing Monitor (YOLOv8)

[![tests](https://github.com/ZIADAYMAN81/social-distancing-monitor-yolov8/actions/workflows/tests.yml/badge.svg)](https://github.com/ZIADAYMAN81/social-distancing-monitor-yolov8/actions/workflows/tests.yml)

> Visual Computing coursework, The British University in Egypt, 2025. Author: Ziad Ayman.

A real-time computer vision tool that finds people in a webcam or video feed with **YOLOv8** and flags anyone standing closer than a safe distance. It runs on an ordinary laptop CPU at about **12 frames per second**.

## What it does
1. **Detects people** in every frame with a pretrained YOLOv8 model (COCO "person" class).
2. **Estimates where each person stands** (the bottom-centre of their box) and measures the distance between every pair.
3. **Flags pairs that are too close**, drawing red boxes and a connecting line, with green boxes for everyone else and a live statistics panel.
4. **Records the run**: an annotated video, a per-frame CSV log and a JSON summary.

## Features
- **Perspective-aware distance limit** (`--mode relative`, the default). People further from the camera look smaller, so a fixed pixel limit is unfair across the frame. The limit is scaled by the two people's average body height in pixels: `1.2 × height` is roughly 2 metres for an adult. A fixed pixel limit is still available with `--mode pixels`.
- **Webcam or video file** input, with an optional preview window or a headless mode for servers.
- **Frame skipping** for slower machines, and a frame cap for quick tests.
- **Exports**: annotated `.mp4`, per-frame CSV (people, people too close, closest pair) and a run summary.
- **Clean, tested code**: the distance logic is pure Python with no model dependency, covered by unit tests.

## Results
Measured on a laptop CPU (no GPU), YOLOv8n, 1920 × 1080 street footage, first 300 frames:

| Metric | Value |
|---|---|
| Processing speed | **12.1 frames per second** |
| People detected | 2,794 (about 9 per frame) |
| Most people too close in a single frame | 11 |

## Project structure
```
social_distancing/
├── geometry.py   # people, ground points, pairwise distances, perspective-aware limits (no dependencies)
├── detector.py   # YOLOv8 person detector
├── render.py     # boxes, links and statistics panel
├── monitor.py    # video loop, CSV log, video writer, run summary
└── __main__.py   # command-line interface
tests/
└── test_geometry.py
```

## Getting started
```bash
pip install -r requirements.txt

# Webcam (press q to quit)
python -m social_distancing --source 0

# Video file, headless, with every output saved
python -m social_distancing --source street.mp4 --no-display \
    --save-video outputs/annotated.mp4 --log-csv outputs/log.csv --summary-json outputs/summary.json

# Fixed pixel limit instead of the perspective-aware one
python -m social_distancing --source street.mp4 --mode pixels --min-distance 120

python -m pytest -q    # run the tests
```
The YOLOv8 weights (`yolov8n.pt`) download automatically on first run. See `python -m social_distancing --help` for every option.

## Limitations and next steps
- Distances are measured in the image, not on the ground. A one-off **homography calibration** (marking four points on the floor) would turn them into true metres.
- Each frame is analysed independently. Adding a **tracker** (for example ByteTrack) would allow counting how long each pair stays too close.
- Sample footage is not included in the repository; use your own webcam or video.

## Tech stack
Python · YOLOv8 (Ultralytics) · OpenCV · NumPy · pytest

## License
MIT, see [LICENSE](LICENSE).
