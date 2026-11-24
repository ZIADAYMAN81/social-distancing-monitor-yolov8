"""Pure distance logic, independent of any model or video library (fully unit-tested)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import hypot
from typing import Iterable, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class Person:
    """A detected person as an axis-aligned box in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0

    @property
    def height(self) -> int:
        return max(self.y2 - self.y1, 1)

    @property
    def ground_point(self) -> Tuple[int, int]:
        """Bottom-centre of the box: an approximation of where the person is standing."""
        return ((self.x1 + self.x2) // 2, self.y2)


@dataclass(frozen=True)
class PairDistance:
    i: int
    j: int
    pixels: float
    threshold: float

    @property
    def too_close(self) -> bool:
        return self.pixels < self.threshold


def pair_threshold(a: Person, b: Person, mode: str, min_distance_px: float, height_ratio: float) -> float:
    """Minimum allowed distance between two people.

    * ``pixels``: one fixed pixel distance for the whole frame.
    * ``relative``: scaled by the two people's average box height, so people further from the
      camera (who appear smaller) get a proportionally smaller pixel limit.
      ``height_ratio=1.2`` means "about 1.2 body heights", roughly 2 metres for an adult.
    """
    if mode == "pixels":
        return float(min_distance_px)
    if mode == "relative":
        return height_ratio * (a.height + b.height) / 2.0
    raise ValueError(f"unknown distance mode: {mode!r} (use 'pixels' or 'relative')")


def analyse(
    people: Sequence[Person],
    mode: str = "relative",
    min_distance_px: float = 100.0,
    height_ratio: float = 1.2,
) -> Tuple[List[PairDistance], Set[int]]:
    """Return every pairwise distance and the set of people involved in at least one violation."""
    pairs: List[PairDistance] = []
    violators: Set[int] = set()
    for i, j in combinations(range(len(people)), 2):
        (xa, ya), (xb, yb) = people[i].ground_point, people[j].ground_point
        limit = pair_threshold(people[i], people[j], mode, min_distance_px, height_ratio)
        pair = PairDistance(i, j, hypot(xa - xb, ya - yb), limit)
        pairs.append(pair)
        if pair.too_close:
            violators.update((i, j))
    return pairs, violators


def safe_people(n_people: int, violators: Iterable[int]) -> Set[int]:
    """Everyone who is not part of any violation (each person is in exactly one group)."""
    return set(range(n_people)) - set(violators)
