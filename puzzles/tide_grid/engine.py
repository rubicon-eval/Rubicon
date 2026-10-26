"""Tide Grid game engine — state, transitions, legality checks.

Core idea: grid tiles cycle between passable (dry) and impassable (flooded).
The agent must time movement through tidal zones whose accessibility windows
overlap non-trivially.  Every action (including WAIT) costs 1 oxygen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from puzzles.common.types import Pos, Direction, add_pos, in_bounds


# ---------------------------------------------------------------------------
# Tile types
# ---------------------------------------------------------------------------

class Tile(Enum):
    FLOOR = "floor"
    WALL = "wall"
    TIDAL = "tidal"
    EXIT = "exit"


# ---------------------------------------------------------------------------
# Action enum (5 values: 4 movement + WAIT)
# ---------------------------------------------------------------------------

class TideGridAction(Enum):
    UP = Direction.UP
    DOWN = Direction.DOWN
    LEFT = Direction.LEFT
    RIGHT = Direction.RIGHT
    WAIT = None

    def __init__(self, direction: Optional[Direction]):
        self._direction = direction

    @property
    def is_move(self) -> bool:
        return self._direction is not None

    @property
    def is_wait(self) -> bool:
        return self._direction is None

    @property
    def direction(self) -> Optional[Direction]:
        return self._direction

    @property
    def label(self) -> str:
        return self.name


MOVE_ACTIONS = [a for a in TideGridAction if a.is_move]


# ---------------------------------------------------------------------------
# Static data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TidalZone:
    """A group of tidal tiles sharing the same phase cycle."""

    zone_id: str                # "A", "B", etc.
    tiles: frozenset[Pos]
    dry_duration: int           # turns the zone is passable
    flood_duration: int         # turns the zone is impassable
    phase_offset: int           # offset into cycle at t=0

    @property
    def cycle_length(self) -> int:
        return self.dry_duration + self.flood_duration

    def is_dry_at(self, t: int) -> bool:
        """Return True if the zone is passable (dry) at time t."""
        phase = (t + self.phase_offset) % self.cycle_length
        return phase < self.dry_duration


@dataclass(frozen=True)
class Beacon:
    """A collectible item the agent must pick up."""

    beacon_id: str
    pos: Pos


@dataclass(frozen=True)
class TideGridGrid:
    """Static puzzle layout (does not change during gameplay)."""

    width: int
    height: int
    walls: frozenset[Pos]
    tidal_zones: dict[str, TidalZone]       # zone_id -> TidalZone
    beacons: dict[str, Beacon]              # beacon_id -> Beacon
    start: Pos
    exit_pos: Pos
    initial_oxygen: int
    # Derived: map tidal tile -> zone for fast lookup
    _tidal_tile_zone: dict[Pos, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        mapping: dict[Pos, str] = {}
        for zone in self.tidal_zones.values():
            for pos in zone.tiles:
                mapping[pos] = zone.zone_id
        # Bypass frozen with object.__setattr__
        object.__setattr__(self, "_tidal_tile_zone", mapping)

    def tile_type(self, pos: Pos) -> Tile:
        """Return the static tile type at a position."""
        if pos in self.walls:
            return Tile.WALL
        if pos in self._tidal_tile_zone:
            return Tile.TIDAL
        if pos == self.exit_pos:
            return Tile.EXIT
        return Tile.FLOOR

    def zone_at(self, pos: Pos) -> Optional[TidalZone]:
        """Return the tidal zone for a position, or None."""
        zid = self._tidal_tile_zone.get(pos)
        if zid is not None:
            return self.tidal_zones[zid]
        return None

    def beacon_at(self, pos: Pos) -> Optional[Beacon]:
        """Return beacon at position, or None."""
        for b in self.beacons.values():
            if b.pos == pos:
                return b
        return None


# ---------------------------------------------------------------------------
# Dynamic state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TideGridState:
    """Dynamic game state (changes every action)."""

    agent_pos: Pos
    time: int                               # global timestep (starts at 0)
    collected_beacons: frozenset[str]       # beacon IDs collected so far

    def to_hashable(self) -> tuple:
        return (self.agent_pos, self.time, self.collected_beacons)

    @property
    def oxygen_remaining(self, initial_oxygen: int = 0) -> int:
        # NOTE: use grid.initial_oxygen - self.time in practice
        return initial_oxygen - self.time


# ---------------------------------------------------------------------------
# Tile passability queries (time-dependent)
# ---------------------------------------------------------------------------

def is_passable_at(grid: TideGridGrid, pos: Pos, t: int) -> bool:
    """Check if a position is passable at time t."""
    if not in_bounds(pos, grid.height, grid.width):
        return False
    tile = grid.tile_type(pos)
    if tile == Tile.WALL:
        return False
    if tile == Tile.TIDAL:
        zone = grid.zone_at(pos)
        return zone.is_dry_at(t)
    # FLOOR and EXIT are always passable
    return True


# ---------------------------------------------------------------------------
# Legal actions and transitions
# ---------------------------------------------------------------------------

def get_legal_actions(grid: TideGridGrid, state: TideGridState) -> list[TideGridAction]:
    """Return list of legal actions from the current state.

    Legality at new_time = state.time + 1:
    - Movement: target tile must be passable at new_time
    - WAIT: current tile must be passable at new_time
    - Both: oxygen must be > 0 (i.e., state.time < grid.initial_oxygen)
    """
    if state.time >= grid.initial_oxygen:
        return []  # no oxygen left

    new_time = state.time + 1
    actions: list[TideGridAction] = []

    for a in MOVE_ACTIONS:
        target = add_pos(state.agent_pos, a.direction)
        if is_passable_at(grid, target, new_time):
            actions.append(a)

    # WAIT: current tile must remain passable at new_time
    if is_passable_at(grid, state.agent_pos, new_time):
        actions.append(TideGridAction.WAIT)

    return actions


def apply_action(
    grid: TideGridGrid,
    state: TideGridState,
    action: TideGridAction,
) -> TideGridState:
    """Apply an action, returning the new state.

    Movement: changes position + auto-collects beacon if present.
    WAIT: stays in place (beacon already collected if on one).
    Both: advance time by 1.
    """
    new_time = state.time + 1

    if action.is_move:
        new_pos = add_pos(state.agent_pos, action.direction)
    else:
        new_pos = state.agent_pos

    # Auto-collect beacon
    new_collected = state.collected_beacons
    beacon = grid.beacon_at(new_pos)
    if beacon is not None and beacon.beacon_id not in state.collected_beacons:
        new_collected = state.collected_beacons | frozenset({beacon.beacon_id})

    return TideGridState(
        agent_pos=new_pos,
        time=new_time,
        collected_beacons=new_collected,
    )


def is_solved(grid: TideGridGrid, state: TideGridState) -> bool:
    """Check if the puzzle is solved (all beacons collected AND on exit)."""
    if state.agent_pos != grid.exit_pos:
        return False
    return state.collected_beacons == frozenset(grid.beacons.keys())


def make_initial_state(grid: TideGridGrid) -> TideGridState:
    """Create the initial state for a grid."""
    return TideGridState(
        agent_pos=grid.start,
        time=0,
        collected_beacons=frozenset(),
    )
