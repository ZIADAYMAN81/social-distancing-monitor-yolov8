"""Command-line entry point: ``python -m social_distancing --help``."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .monitor import Settings, run


def parse_args(argv=None) -> Settings:
    p = argparse.ArgumentParser(prog="social_distancing",
                                description="Flag people standing too close together in a webcam or video feed.")
    p.add_argument("--source", default="0", help="webcam index (for example 0) or path to a video file")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights (downloaded automatically)")
    p.add_argument("--conf", type=float, default=0.5, help="minimum detection confidence")
    p.add_argument("--mode", choices=["relative", "pixels"], default="relative",
                   help="'relative' scales the limit by body height (perspective-aware); 'pixels' uses a fixed limit")
    p.add_argument("--min-distance", type=float, default=100.0, help="pixel limit used in 'pixels' mode")
    p.add_argument("--height-ratio", type=float, default=1.2, help="limit in body heights used in 'relative' mode")
    p.add_argument("--frame-skip", type=int, default=1, help="analyse every Nth frame for extra speed")
    p.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    p.add_argument("--save-video", default=None, help="write the annotated video here (.mp4)")
    p.add_argument("--log-csv", default=None, help="write per-frame statistics here")
    p.add_argument("--summary-json", default=None, help="write a run summary here")
    p.add_argument("--no-display", action="store_true", help="run headless, without a preview window")
    a = p.parse_args(argv)
    source = int(a.source) if a.source.isdigit() else a.source
    return Settings(source=source, weights=a.weights, confidence=a.conf, mode=a.mode,
                    min_distance_px=a.min_distance, height_ratio=a.height_ratio, frame_skip=a.frame_skip,
                    max_frames=a.max_frames, output_video=a.save_video, log_csv=a.log_csv,
                    summary_json=a.summary_json, display=not a.no_display)


def main(argv=None) -> None:
    summary = run(parse_args(argv))
    print(json.dumps({k: v for k, v in asdict(summary).items() if k != "settings"}, indent=2))


if __name__ == "__main__":
    main()
