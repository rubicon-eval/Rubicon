"""Mosaic Grid oracle — A* search, solvability, deadlock detection."""

from __future__ import annotations

from collections import deque
from typing import Optional

from puzzles.common.types import Pos, Direction, manhattan_distance, add_pos
from puzzles.common.astar import astar_search
from .engine import (
    MosaicGrid,
    MosaicState,
    get_legal_actions,
    apply_action,
    is_solved,
    is_passable,
)


# ---------------------------------------------------------------------------
# Direction deltas (avoid repeated attribute lookup in inner loops)
# ---------------------------------------------------------------------------
_DELTAS = [d.value for d in Direction]
_DIRS = list(Direction)


# ---------------------------------------------------------------------------
# Step 1: BFS floor-distance cache
# ---------------------------------------------------------------------------

def _get_floor_distances(grid: MosaicGrid) -> dict[Pos, int]:
    """BFS from exit_pos on static floor (walls removed, doors passable).

    Returns dict mapping every reachable floor tile to its shortest distance
    to the exit.  Cached on the grid object to amortize across calls.
    """
    cached = getattr(grid, '_floor_distances', None)
    if cached is not None:
        return cached

    exit_pos = grid.exit_pos
    walls = grid.walls
    width, height = grid.width, grid.height

    dist: dict[Pos, int] = {exit_pos: 0}
    queue: deque[Pos] = deque([exit_pos])

    while queue:
        pos = queue.popleft()
        d = dist[pos]
        for dr, dc in _DELTAS:
            nb = Pos(pos.row + dr, pos.col + dc)
            if (0 <= nb.row < height and 0 <= nb.col < width
                    and nb not in walls and nb not in dist):
                dist[nb] = d + 1
                queue.append(nb)

    object.__setattr__(grid, '_floor_distances', dist)
    return dist


# ---------------------------------------------------------------------------
# Step 2: Improved heuristic using floor distances
# ---------------------------------------------------------------------------

def _make_heuristic(grid: MosaicGrid):
    """Factory: BFS floor distance + unactivated required plates."""
    required_mask = 0
    for door in grid.doors.values():
        for pid in door.required_plate_ids:
            required_mask |= 1 << grid._pid_to_bit[pid]  # type: ignore[attr-defined]

    floor_dist = _get_floor_distances(grid)
    exit_pos = grid.exit_pos

    def _h(state: MosaicState) -> int:
        h = floor_dist.get(state.agent_pos,
                           manhattan_distance(state.agent_pos, exit_pos))
        remaining = required_mask & ~state.activated_plates
        h += bin(remaining).count("1")
        return h

    return _h


def _required_plates(grid: MosaicGrid) -> frozenset[str]:
    """Return the set of plate IDs that are required to open doors
    blocking any path to the exit."""
    required = set()
    for door in grid.doors.values():
        for pid in door.required_plate_ids:
            required.add(pid)
    return frozenset(required)


def heuristic(grid: MosaicGrid, state: MosaicState) -> int:
    """Admissible heuristic: floor distance to exit + unactivated required plates."""
    return _make_heuristic(grid)(state)


# ---------------------------------------------------------------------------
# Step 3: Quick unreachability pre-filter
# ---------------------------------------------------------------------------

def _quick_unreachable(grid: MosaicGrid, state: MosaicState) -> bool:
    """O(V) flood fill: is exit definitely unreachable?

    BFS from agent_pos on (floor - painted tiles), treating all doors as
    passable regardless of plate state.  If exit_pos is not reached, the
    state is definitely unsolvable (painted tiles permanently block).
    """
    exit_pos = grid.exit_pos
    agent_pos = state.agent_pos
    if agent_pos == exit_pos:
        return False

    walls = grid.walls
    painted = state.painted
    width, height = grid.width, grid.height

    visited = {agent_pos}
    queue: deque[Pos] = deque([agent_pos])

    while queue:
        pos = queue.popleft()
        for dr, dc in _DELTAS:
            nb = Pos(pos.row + dr, pos.col + dc)
            if nb == exit_pos:
                return False  # reachable
            if (0 <= nb.row < height and 0 <= nb.col < width
                    and nb not in walls and nb not in visited
                    and not ((painted >> (nb.row * width + nb.col)) & 1)):
                visited.add(nb)
                queue.append(nb)

    return True  # exit not reached -> definitely unsolvable


# ---------------------------------------------------------------------------
# Step 5: Inlined fast successor generation
# ---------------------------------------------------------------------------

def _make_fast_successors(grid: MosaicGrid):
    """Closure that inlines get_legal_actions + apply_action for A*."""
    width = grid.width
    height = grid.height
    walls = grid.walls
    door_by_pos: dict[Pos, object] = grid._door_by_pos  # type: ignore[attr-defined]
    door_required_mask: dict[str, int] = grid._door_required_mask  # type: ignore[attr-defined]
    plate_bit_by_pos: dict[Pos, int] = grid._plate_bit_by_pos  # type: ignore[attr-defined]
    exit_pos = grid.exit_pos

    def _successors(state: MosaicState) -> list[tuple[Direction, MosaicState]]:
        agent_pos = state.agent_pos
        painted = state.painted
        activated = state.activated_plates

        # Departure painting (once for all successors)
        if agent_pos != exit_pos:
            bit = agent_pos.row * width + agent_pos.col
            new_painted = painted | (1 << bit)
        else:
            new_painted = painted

        result: list[tuple[Direction, MosaicState]] = []
        row, col = agent_pos.row, agent_pos.col

        for i in range(4):
            dr, dc = _DELTAS[i]
            nr, nc = row + dr, col + dc

            # Bounds check
            if not (0 <= nr < height and 0 <= nc < width):
                continue

            nb = Pos(nr, nc)

            # Wall check
            if nb in walls:
                continue

            # Painted check
            if (new_painted >> (nr * width + nc)) & 1:
                continue

            # Door check (closed door blocks)
            door = door_by_pos.get(nb)
            if door is not None:
                mask = door_required_mask[door.id]  # type: ignore[union-attr]
                if (activated & mask) != mask:
                    continue

            # Position is passable — compute new state
            new_activated = activated
            pb = plate_bit_by_pos.get(nb)
            if pb is not None and not ((activated >> pb) & 1):
                new_activated = activated | (1 << pb)

            result.append((_DIRS[i], MosaicState(nb, new_painted, new_activated)))

        return result

    return _successors


# ---------------------------------------------------------------------------
# Step 6: Single-int state hash
# ---------------------------------------------------------------------------

def _make_hash_fn(grid: MosaicGrid):
    """Pack (agent_pos, activated_plates, painted) into one integer."""
    width = grid.width
    num_positions = width * grid.height
    pos_bits = num_positions.bit_length()
    plate_shift = pos_bits
    painted_shift = pos_bits + max(1, len(grid.plates))

    def _hash(state: MosaicState) -> int:
        return ((state.agent_pos.row * width + state.agent_pos.col)
                | (state.activated_plates << plate_shift)
                | (state.painted << painted_shift))

    return _hash


# ---------------------------------------------------------------------------
# Original successor helper (kept for external use)
# ---------------------------------------------------------------------------

def _get_successors(grid: MosaicGrid, state: MosaicState) -> list[tuple[Direction, MosaicState]]:
    """Get all successor (action, state) pairs."""
    result = []
    for action in get_legal_actions(grid, state):
        next_state = apply_action(grid, state, action)
        result.append((action, next_state))
    return result


# ---------------------------------------------------------------------------
# Steps 4 & 7: Wired-up is_solvable and oracle_solve
# ---------------------------------------------------------------------------

def is_solvable(
    grid: MosaicGrid,
    state: MosaicState,
    max_expansions: int = 500_000,
) -> bool:
    """Check if the puzzle is solvable from the given state using A*."""
    if _quick_unreachable(grid, state):
        return False
    return astar_search(
        initial_state=state,
        get_successors=_make_fast_successors(grid),
        is_goal=lambda s: s.agent_pos == grid.exit_pos,
        heuristic=_make_heuristic(grid),
        hash_state=_make_hash_fn(grid),
        max_expansions=max_expansions,
        return_path=False,
    )


def oracle_solve(
    grid: MosaicGrid,
    state: MosaicState,
    max_expansions: int = 500_000,
) -> Optional[list[Direction]]:
    """Find the optimal solution path using A*."""
    if _quick_unreachable(grid, state):
        return None
    return astar_search(
        initial_state=state,
        get_successors=_make_fast_successors(grid),
        is_goal=lambda s: s.agent_pos == grid.exit_pos,
        heuristic=_make_heuristic(grid),
        hash_state=_make_hash_fn(grid),
        max_expansions=max_expansions,
        return_path=True,
    )


# ---------------------------------------------------------------------------
# Deadlock / walking-dead utilities (unchanged API)
# ---------------------------------------------------------------------------

def is_walking_dead(
    grid: MosaicGrid,
    state: MosaicState,
    max_expansions: int = 500_000,
) -> bool:
    """Check if the state is Walking Dead: has legal moves but is unsolvable."""
    has_legal = len(get_legal_actions(grid, state)) > 0
    if not has_legal:
        return False
    return not is_solvable(grid, state, max_expansions)


def apply_actions(
    grid: MosaicGrid,
    state: MosaicState,
    actions: list[Direction],
) -> MosaicState:
    """Sequentially apply a list of actions."""
    for action in actions:
        state = apply_action(grid, state, action)
    return state


def find_fatal_step(
    grid: MosaicGrid,
    initial_state: MosaicState,
    actions: list[Direction],
    max_expansions: int = 500_000,
) -> int:
    """Find the first action index where the puzzle becomes unsolvable.

    Returns the index of the action that caused the deadlock, or -1 if
    no deadlock occurs in the trace.
    """
    state = initial_state
    for i, action in enumerate(actions):
        state = apply_action(grid, state, action)
        if not is_solvable(grid, state, max_expansions):
            return i
    return -1
