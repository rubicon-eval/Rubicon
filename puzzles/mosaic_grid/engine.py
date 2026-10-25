"""Mosaic Grid game engine — state, transitions, legality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from puzzles.common.types import Pos, Direction, add_pos, in_bounds


class TileType(Enum):
    FLOOR = "floor"
    WALL = "wall"
    EXIT = "exit"


@dataclass(frozen=True)
class PlateInfo:
    id: str
    pos: Pos
    linked_door_ids: tuple[str, ...]


@dataclass(frozen=True)
class DoorInfo:
    id: str
    pos: Pos
    required_plate_ids: tuple[str, ...]


@dataclass(frozen=True)
class MosaicGrid:
    """Static puzzle layout (does not change during gameplay)."""

    width: int
    height: int
    walls: frozenset[Pos]
    plates: dict[str, PlateInfo]   # plate_id -> PlateInfo
    doors: dict[str, DoorInfo]     # door_id -> DoorInfo
    start: Pos
    exit_pos: Pos

    def __post_init__(self) -> None:
        # --- Position lookup dicts (O(1) plate_at / door_at) ---
        plate_by_pos: dict[Pos, PlateInfo] = {p.pos: p for p in self.plates.values()}
        door_by_pos: dict[Pos, DoorInfo] = {d.pos: d for d in self.doors.values()}
        object.__setattr__(self, "_plate_by_pos", plate_by_pos)
        object.__setattr__(self, "_door_by_pos", door_by_pos)

        # --- Bitmask infrastructure ---
        # Plate ID → bit index (sorted order for determinism)
        sorted_pids = sorted(self.plates.keys())
        pid_to_bit: dict[str, int] = {pid: i for i, pid in enumerate(sorted_pids)}
        bit_to_pid: dict[int, str] = {i: pid for pid, i in pid_to_bit.items()}
        object.__setattr__(self, "_pid_to_bit", pid_to_bit)
        object.__setattr__(self, "_bit_to_pid", bit_to_pid)

        # Per-door bitmask of required plates
        door_required_mask: dict[str, int] = {}
        for did, door in self.doors.items():
            mask = 0
            for pid in door.required_plate_ids:
                mask |= 1 << pid_to_bit[pid]
            door_required_mask[did] = mask
        object.__setattr__(self, "_door_required_mask", door_required_mask)

        # Position → plate bit index (for apply_action)
        plate_bit_by_pos: dict[Pos, int] = {}
        for pid, plate in self.plates.items():
            plate_bit_by_pos[plate.pos] = pid_to_bit[pid]
        object.__setattr__(self, "_plate_bit_by_pos", plate_bit_by_pos)

    def plate_at(self, pos: Pos) -> Optional[PlateInfo]:
        return self._plate_by_pos.get(pos)  # type: ignore[attr-defined]

    def door_at(self, pos: Pos) -> Optional[DoorInfo]:
        return self._door_by_pos.get(pos)  # type: ignore[attr-defined]

    @property
    def all_walkable_positions(self) -> list[Pos]:
        """All positions that could potentially be walked on."""
        positions = []
        for r in range(self.height):
            for c in range(self.width):
                p = Pos(r, c)
                if p not in self.walls:
                    positions.append(p)
        return positions


@dataclass(frozen=True)
class MosaicState:
    """Dynamic game state (changes every move).

    painted: int bitmask — bit (row * width + col) set for each painted position.
    activated_plates: int bitmask — bit index from grid._pid_to_bit.
    """

    agent_pos: Pos
    painted: int
    activated_plates: int

    def to_hashable(self) -> tuple:
        return (self.agent_pos, self.painted, self.activated_plates)

    def is_door_open(self, grid: MosaicGrid, door_id: str) -> bool:
        mask = grid._door_required_mask[door_id]  # type: ignore[attr-defined]
        return (self.activated_plates & mask) == mask


def get_tile_at(grid: MosaicGrid, state: MosaicState, pos: Pos) -> str:
    """Return the effective tile type at a position given the current state."""
    if pos in grid.walls:
        return "wall"
    if (state.painted >> (pos.row * grid.width + pos.col)) & 1:
        return "painted"
    door = grid.door_at(pos)
    if door is not None:
        if state.is_door_open(grid, door.id):
            return "door_open"
        return "door_closed"
    if grid.plate_at(pos) is not None:
        return "plate"
    if pos == grid.exit_pos:
        return "exit"
    return "floor"


def is_passable(grid: MosaicGrid, state: MosaicState, pos: Pos) -> bool:
    """Check if a position can be moved onto."""
    if not in_bounds(pos, grid.height, grid.width):
        return False
    tile = get_tile_at(grid, state, pos)
    return tile in ("floor", "plate", "door_open", "exit")


def get_legal_actions(grid: MosaicGrid, state: MosaicState) -> list[Direction]:
    """Return list of legal movement directions."""
    actions = []
    for d in Direction:
        new_pos = add_pos(state.agent_pos, d)
        if is_passable(grid, state, new_pos):
            actions.append(d)
    return actions


def apply_action(grid: MosaicGrid, state: MosaicState, action: Direction) -> MosaicState:
    """Apply a movement action, returning the new state.

    Paint on DEPARTURE: the tile the agent leaves becomes painted
    (unless it's the EXIT tile).
    """
    new_pos = add_pos(state.agent_pos, action)

    # Paint the tile being left (departure painting)
    new_painted = state.painted
    if state.agent_pos != grid.exit_pos:
        bit = state.agent_pos.row * grid.width + state.agent_pos.col
        new_painted = state.painted | (1 << bit)

    # Check plate activation at new position
    new_activated = state.activated_plates
    plate_bit = grid._plate_bit_by_pos.get(new_pos)  # type: ignore[attr-defined]
    if plate_bit is not None and not ((state.activated_plates >> plate_bit) & 1):
        new_activated = state.activated_plates | (1 << plate_bit)

    return MosaicState(
        agent_pos=new_pos,
        painted=new_painted,
        activated_plates=new_activated,
    )


def is_solved(grid: MosaicGrid, state: MosaicState) -> bool:
    """Check if the puzzle is solved (agent on exit)."""
    return state.agent_pos == grid.exit_pos


# ---------------------------------------------------------------------------
# Bitmask encode / decode helpers
# ---------------------------------------------------------------------------

def encode_painted(positions, width: int) -> int:
    """Encode an iterable of Pos into a painted bitmask."""
    mask = 0
    for p in positions:
        mask |= 1 << (p.row * width + p.col)
    return mask


def decode_painted(mask: int, width: int) -> list[Pos]:
    """Decode a painted bitmask into a sorted list of Pos."""
    positions: list[Pos] = []
    m = mask
    while m:
        bit = (m & -m).bit_length() - 1
        m &= m - 1
        positions.append(Pos(bit // width, bit % width))
    positions.sort()
    return positions


def encode_activated(plate_ids, pid_to_bit: dict[str, int]) -> int:
    """Encode an iterable of plate IDs into an activated_plates bitmask."""
    mask = 0
    for pid in plate_ids:
        mask |= 1 << pid_to_bit[pid]
    return mask


def decode_activated(mask: int, bit_to_pid: dict[int, str]) -> list[str]:
    """Decode an activated_plates bitmask into a sorted list of plate IDs."""
    pids: list[str] = []
    m = mask
    while m:
        bit = (m & -m).bit_length() - 1
        m &= m - 1
        pids.append(bit_to_pid[bit])
    pids.sort()
    return pids
