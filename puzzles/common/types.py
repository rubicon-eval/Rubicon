"""Shared types and helpers for all puzzle engines."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class Pos(NamedTuple):
    row: int
    col: int


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @property
    def label(self) -> str:
        return self.name

    @property
    def delta(self) -> tuple[int, int]:
        return self.value


def add_pos(p: Pos, d: Direction) -> Pos:
    dr, dc = d.value
    return Pos(p.row + dr, p.col + dc)


def manhattan_distance(a: Pos, b: Pos) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def in_bounds(pos: Pos, height: int, width: int) -> bool:
    return 0 <= pos.row < height and 0 <= pos.col < width
