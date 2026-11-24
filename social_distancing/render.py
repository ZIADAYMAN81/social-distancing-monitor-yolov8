"""Drawing helpers for the annotated output."""
from __future__ import annotations

from typing import Optional, Sequence, Set

import cv2
import numpy as np

from .geometry import PairDistance, Person

RED = (0, 0, 255)
GREEN = (0, 200, 0)
WHITE = (255, 255, 255)


def draw(frame: np.ndarray, people: Sequence[Person], pairs: Sequence[PairDistance],
         violators: Set[int], fps: Optional[float] = None, show_all_pairs: bool = False) -> np.ndarray:
    """Return a copy of ``frame`` with boxes, too-close links and a statistics panel."""
    out = frame.copy()
    for idx, p in enumerate(people):
        colour = RED if idx in violators else GREEN
        cv2.rectangle(out, (p.x1, p.y1), (p.x2, p.y2), colour, 2)
        cv2.putText(out, f"{p.confidence:.2f}", (p.x1, max(p.y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    for pair in pairs:
        if not (pair.too_close or show_all_pairs):
            continue
        a, b = people[pair.i].ground_point, people[pair.j].ground_point
        colour = RED if pair.too_close else GREEN
        cv2.line(out, a, b, colour, 1, cv2.LINE_AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        cv2.putText(out, f"{pair.pixels:.0f}px", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

    n_close = len(violators)
    lines = [f"People: {len(people)}", f"Too close: {n_close}", f"Safe: {len(people) - n_close}"]
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    cv2.rectangle(out, (5, 5), (175, 18 + 22 * len(lines)), (0, 0, 0), -1)
    for k, text in enumerate(lines):
        colour = RED if text.startswith("Too close") and n_close else WHITE
        cv2.putText(out, text, (12, 26 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    return out
