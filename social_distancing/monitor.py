"""Run the monitor over a webcam or video file, with optional video, CSV and summary output."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2

from .detector import PersonDetector
from .geometry import analyse
from .render import draw


@dataclass
class Settings:
    source: Union[int, str] = 0
    weights: str = "yolov8n.pt"
    confidence: float = 0.5
    mode: str = "relative"
    min_distance_px: float = 100.0
    height_ratio: float = 1.2
    frame_skip: int = 1
    max_frames: Optional[int] = None
    output_video: Optional[str] = None
    log_csv: Optional[str] = None
    summary_json: Optional[str] = None
    display: bool = True


@dataclass
class Summary:
    frames_read: int = 0
    frames_analysed: int = 0
    total_detections: int = 0
    frames_with_violation: int = 0
    peak_people_too_close: int = 0
    seconds: float = 0.0
    average_fps: float = 0.0
    settings: dict = field(default_factory=dict)


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def run(settings: Settings) -> Summary:
    cap = cv2.VideoCapture(settings.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {settings.source!r}")

    detector = PersonDetector(settings.weights, settings.confidence)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    writer = None
    if settings.output_video:
        _ensure_parent(settings.output_video)
        writer = cv2.VideoWriter(settings.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, size)

    log_file = log = None
    if settings.log_csv:
        _ensure_parent(settings.log_csv)
        log_file = open(settings.log_csv, "w", newline="", encoding="utf-8")
        log = csv.writer(log_file)
        log.writerow(["frame", "timestamp_s", "people", "people_too_close", "closest_pair_px"])

    summary = Summary(settings=asdict(settings))
    start = time.perf_counter()
    last_annotated = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            summary.frames_read += 1
            if settings.max_frames and summary.frames_read > settings.max_frames:
                summary.frames_read -= 1
                break

            skip = (summary.frames_read - 1) % max(settings.frame_skip, 1) != 0
            if skip and last_annotated is not None:
                annotated = last_annotated  # reuse the most recent analysis on skipped frames
            else:
                people = detector(frame)
                pairs, violators = analyse(people, settings.mode, settings.min_distance_px, settings.height_ratio)
                summary.frames_analysed += 1
                summary.total_detections += len(people)
                summary.frames_with_violation += bool(violators)
                summary.peak_people_too_close = max(summary.peak_people_too_close, len(violators))
                live_fps = summary.frames_analysed / max(time.perf_counter() - start, 1e-6)
                annotated = last_annotated = draw(frame, people, pairs, violators, fps=live_fps)
                if log:
                    closest = min((p.pixels for p in pairs), default=None)
                    log.writerow([summary.frames_read, round(summary.frames_read / fps_in, 3), len(people),
                                  len(violators), "" if closest is None else round(closest, 1)])

            if writer:
                writer.write(annotated)
            if settings.display:
                cv2.imshow("Social Distancing Monitor (press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        if settings.display:
            cv2.destroyAllWindows()

    summary.seconds = round(time.perf_counter() - start, 2)
    summary.average_fps = round(summary.frames_analysed / max(summary.seconds, 1e-6), 2)
    if settings.summary_json:
        _ensure_parent(settings.summary_json)
        Path(settings.summary_json).write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary
