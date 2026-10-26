"""Chain Vault game engine — state, transitions, legality checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from puzzles.common.types import Pos, Direction, add_pos, in_bounds


# ---------------------------------------------------------------------------
# Action enum (4 directional moves)
# ---------------------------------------------------------------------------

class ChainVaultAction(Enum):
    UP = Direction.UP
    DOWN = Direction.DOWN
    LEFT = Direction.LEFT
    RIGHT = Direction.RIGHT

    def __init__(self, direction: Direction):
        self._direction = direction

    @property
    def is_move(self) -> bool:
        return True

    @property
    def direction(self) -> Direction:
        return self._direction

    @property
    def label(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Static data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyInfo:
    id: str
    pos: Pos
    color: str


@dataclass(frozen=True)
class DoorInfo:
    id: str
    pos: Pos
    required_keys: tuple[str, ...]  # e.g. ("red",) or ("red", "blue")


@dataclass(frozen=True)
class ChainVaultGrid:
    """Static puzzle layout (does not change during gameplay)."""

    width: int
    height: int
    walls: frozenset[Pos]
    keys: dict[str, KeyInfo]    # key_id -> KeyInfo
    doors: dict[str, DoorInfo]  # door_id -> DoorInfo
    start: Pos
    exit_pos: Pos

    def key_at(self, pos: Pos) -> Optional[KeyInfo]:
        for k in self.keys.values():
            if k.pos == pos:
                return k
        return None

    def door_at(self, pos: Pos) -> Optional[DoorInfo]:
        for d in self.doors.values():
            if d.pos == pos:
                return d
        return None


# ---------------------------------------------------------------------------
# Inventory helpers — sorted tuple of (color, count) for hashability
# ---------------------------------------------------------------------------

def _inventory_to_dict(inventory: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return dict(inventory)


def _dict_to_inventory(d: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((c, n) for c, n in d.items() if n > 0))


def _add_key(inventory: tuple[tuple[str, int], ...], color: str) -> tuple[tuple[str, int], ...]:
    d = _inventory_to_dict(inventory)
    d[color] = d.get(color, 0) + 1
    return _dict_to_inventory(d)


def _consume_keys(
    inventory: tuple[tuple[str, int], ...],
    colors: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    d = _inventory_to_dict(inventory)
    for c in colors:
        d[c] = d.get(c, 0) - 1
    return _dict_to_inventory(d)


def _has_keys(inventory: tuple[tuple[str, int], ...], colors: tuple[str, ...]) -> bool:
    d = _inventory_to_dict(inventory)
    need: dict[str, int] = {}
    for c in colors:
        need[c] = need.get(c, 0) + 1
    return all(d.get(c, 0) >= n for c, n in need.items())


# ---------------------------------------------------------------------------
# Dynamic state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainVaultState:
    """Dynamic game state (changes every action)."""

    agent_pos: Pos
    inventory: tuple[tuple[str, int], ...]   # sorted (color, count) pairs
    opened_doors: frozenset[str]             # set of door IDs
    remaining_keys: frozenset[str]           # set of key IDs still on ground

    def to_hashable(self) -> tuple:
        return (self.agent_pos, self.inventory, self.opened_doors, self.remaining_keys)

    def is_door_open(self, door_id: str) -> bool:
        return door_id in self.opened_doors

    def has_keys_for_door(self, grid: ChainVaultGrid, door_id: str) -> bool:
        door = grid.doors[door_id]
        return _has_keys(self.inventory, door.required_keys)


# ---------------------------------------------------------------------------
# Tile queries
# ---------------------------------------------------------------------------

def get_tile_at(grid: ChainVaultGrid, state: ChainVaultState, pos: Pos) -> str:
    """Return the effective tile type at a position given the current state."""
    if pos in grid.walls:
        return "wall"
    door = grid.door_at(pos)
    if door is not None:
        if state.is_door_open(door.id):
            return "door_open"
        return "door_closed"
    key = grid.key_at(pos)
    if key is not None and key.id in state.remaining_keys:
        return "key"
    if pos == grid.exit_pos:
        return "exit"
    return "floor"


def is_passable(grid: ChainVaultGrid, state: ChainVaultState, pos: Pos) -> bool:
    """Check if a position can be moved onto."""
    if not in_bounds(pos, grid.height, grid.width):
        return False
    tile = get_tile_at(grid, state, pos)
    return tile in ("floor", "key", "door_open", "exit")


def get_action_target(state: ChainVaultState, action: ChainVaultAction) -> Pos:
    """Return the adjacent cell targeted by an action."""
    return add_pos(state.agent_pos, action.direction)


def get_door_opened_by_action(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    action: ChainVaultAction,
) -> Optional[DoorInfo]:
    """Return the locked door this action would open, if any."""
    target = get_action_target(state, action)
    if not in_bounds(target, grid.height, grid.width):
        return None

    door = grid.door_at(target)
    if door is None or state.is_door_open(door.id):
        return None
    if not _has_keys(state.inventory, door.required_keys):
        return None
    return door


def action_opens_door(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    action: ChainVaultAction,
) -> bool:
    """Check whether an action opens an adjacent locked door."""
    return get_door_opened_by_action(grid, state, action) is not None


def is_legal_action(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    action: ChainVaultAction,
) -> bool:
    """Check if an action is legal in the current state."""
    target = get_action_target(state, action)
    return is_passable(grid, state, target) or action_opens_door(grid, state, action)


# ---------------------------------------------------------------------------
# Legal actions and transitions
# ---------------------------------------------------------------------------

def get_legal_actions(grid: ChainVaultGrid, state: ChainVaultState) -> list[ChainVaultAction]:
    """Return list of legal actions from the current state."""
    return [action for action in ChainVaultAction if is_legal_action(grid, state, action)]


def apply_action(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    action: ChainVaultAction,
) -> ChainVaultState:
    """Apply an action, returning the new state.

    MOVE into floor/key/open door/exit: changes position + auto-pickup key.
    MOVE into a locked door with sufficient keys: consumes keys, opens the
    door permanently, and the agent stays in place.
    """
    door = get_door_opened_by_action(grid, state, action)
    if door is not None:
        return ChainVaultState(
            agent_pos=state.agent_pos,
            inventory=_consume_keys(state.inventory, door.required_keys),
            opened_doors=state.opened_doors | frozenset({door.id}),
            remaining_keys=state.remaining_keys,
        )

    new_pos = get_action_target(state, action)
    if not is_passable(grid, state, new_pos):
        raise ValueError(f"Illegal action {action.name} from {state.agent_pos}")

    new_inventory = state.inventory
    new_remaining = state.remaining_keys

    # Auto-pickup key
    key = grid.key_at(new_pos)
    if key is not None and key.id in state.remaining_keys:
        new_inventory = _add_key(state.inventory, key.color)
        new_remaining = state.remaining_keys - frozenset({key.id})

    return ChainVaultState(
        agent_pos=new_pos,
        inventory=new_inventory,
        opened_doors=state.opened_doors,
        remaining_keys=new_remaining,
    )


def is_solved(grid: ChainVaultGrid, state: ChainVaultState) -> bool:
    """Check if the puzzle is solved (agent on exit)."""
    return state.agent_pos == grid.exit_pos
