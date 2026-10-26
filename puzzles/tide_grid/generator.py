"""Tide Grid procedural generation and deadlock discovery.

Key design principle: ANTI-GREEDY BY CONSTRUCTION.
Tidal phase alignment makes the "obvious" beacon ordering waste oxygen on
WAITs, while the optimal ordering times zone crossings for minimal delay.
Ten greedy strategies (6 basic + 4 smart) are validated as hard filters.
"""

from __future__ import annotations

import random
import time as _time
from collections import deque
from math import gcd
from dataclasses import dataclass, replace
from typing import Callable, Optional

from puzzles.common.types import Pos, Direction, add_pos, in_bounds, manhattan_distance
from .engine import (
    Tile,
    TideGridGrid,
    TideGridState,
    TideGridAction,
    TidalZone,
    Beacon,
    MOVE_ACTIONS,
    get_legal_actions,
    apply_action,
    is_solved,
    is_passable_at,
    make_initial_state,
)
from .oracle import (
    is_solvable,
    oracle_solve,
    is_walking_dead,
    apply_actions,
    find_fatal_step,
    count_reachable_dead_states,
    heuristic,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DeadlockEntry:
    """A discovered deadlock: a path from S0 to a Walking Dead state."""

    path: list[TideGridAction]
    fatal_step: int              # index of the action that caused solvability flip
    deadlock_state: TideGridState
    pre_deadlock_state: TideGridState
    k_horizon: int = 0           # reachable states from deadlock (Walking Dead depth)
    strategy_name: str = ""      # greedy strategy that produced this deadlock


@dataclass
class GeneratedInstance:
    """A fully validated puzzle instance with solution and deadlock catalog."""

    grid: TideGridGrid
    initial_state: TideGridState
    solution: list[TideGridAction]
    deadlock_catalog: list[DeadlockEntry]
    seed: int
    template_name: str = "strait"
    grid_width: int = 0
    grid_height: int = 0
    num_beacons: int = 0
    num_zones: int = 0
    solution_length: int = 0
    tier1_deadlocks: int = 0
    tier2_deadlocks: int = 0
    max_k_horizon: int = 0
    generation_attempts: int = 0
    fingerprint: str = ""


@dataclass
class GenerationResult:
    """Result of generation with rejection statistics."""

    instance: Optional[GeneratedInstance]
    stats: dict[str, int]
    total_attempts: int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Grid generation helpers
# ---------------------------------------------------------------------------

def _flood_fill(pos: Pos, passable: set[Pos]) -> set[Pos]:
    """Return the connected component containing `pos`."""
    visited: set[Pos] = set()
    queue = deque([pos])
    while queue:
        p = queue.popleft()
        if p in visited or p not in passable:
            continue
        visited.add(p)
        for d in Direction:
            np_ = add_pos(p, d)
            if np_ in passable and np_ not in visited:
                queue.append(np_)
    return visited


def _is_connected(positions: set[Pos]) -> bool:
    """Check if a set of positions forms a single connected component."""
    if not positions:
        return True
    start = next(iter(positions))
    component = _flood_fill(start, positions)
    return len(component) == len(positions)


def _bfs_distance(start: Pos, passable: set[Pos]) -> dict[Pos, int]:
    """BFS from start, returns distance map over passable tiles."""
    dist: dict[Pos, int] = {start: 0}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for d in Direction:
            np_ = add_pos(p, d)
            if np_ in passable and np_ not in dist:
                dist[np_] = dist[p] + 1
                queue.append(np_)
    return dist


def _scale_cycle_params(dry: int, flood: int, rng: random.Random) -> tuple[int, int]:
    """Scale dry/flood by 1.5-2x, preserving ratio. E2 enhancement."""
    scale = rng.uniform(1.5, 2.0)
    new_dry = max(1, round(dry * scale))
    new_flood = max(1, round(flood * scale))
    if new_dry + new_flood <= dry + flood:
        new_dry, new_flood = dry + 1, flood + 1
    return new_dry, new_flood


def _ensure_coprime(flood: int, dry: int, other_cycles: list[int]) -> int:
    """Increment flood until dry+flood is coprime with all other_cycles."""
    cycle = dry + flood
    for oc in other_cycles:
        while gcd(cycle, oc) > 1:
            flood += 1
            cycle = dry + flood
    return flood


# ---------------------------------------------------------------------------
# D4 symmetry transforms (square grids only)
# ---------------------------------------------------------------------------

_DIR_ROT_TABLE = {
    0: {Direction.UP: Direction.UP, Direction.DOWN: Direction.DOWN,
        Direction.LEFT: Direction.LEFT, Direction.RIGHT: Direction.RIGHT},
    90: {Direction.UP: Direction.RIGHT, Direction.RIGHT: Direction.DOWN,
         Direction.DOWN: Direction.LEFT, Direction.LEFT: Direction.UP},
    180: {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
          Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT},
    270: {Direction.UP: Direction.LEFT, Direction.LEFT: Direction.DOWN,
          Direction.DOWN: Direction.RIGHT, Direction.RIGHT: Direction.UP},
}

_FLIP_H_DIR = {
    Direction.UP: Direction.UP, Direction.DOWN: Direction.DOWN,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT,
}

# Build action lookup by direction
_ACTION_BY_DIR: dict[Optional[Direction], TideGridAction] = {
    a.direction: a for a in TideGridAction
}


def _transform_pos(pos: Pos, size: int, rotation: int, flip_h: bool) -> Pos:
    """Apply rotation then optional horizontal flip to a grid position."""
    r, c = pos
    s = size - 1
    if rotation == 90:
        r, c = c, s - r
    elif rotation == 180:
        r, c = s - r, s - c
    elif rotation == 270:
        r, c = s - c, r
    if flip_h:
        c = s - c
    return Pos(r, c)


def _transform_direction(d: Direction, rotation: int, flip_h: bool) -> Direction:
    """Map a direction through rotation then optional horizontal flip."""
    d = _DIR_ROT_TABLE[rotation][d]
    if flip_h:
        d = _FLIP_H_DIR[d]
    return d


def _transform_action(
    action: TideGridAction, rotation: int, flip_h: bool,
) -> TideGridAction:
    """Build a new action with the transformed direction (WAIT unchanged)."""
    if action.is_wait:
        return action
    new_dir = _transform_direction(action.direction, rotation, flip_h)
    return _ACTION_BY_DIR[new_dir]


def _transform_grid(
    grid: TideGridGrid, rotation: int, flip_h: bool,
) -> TideGridGrid:
    """Transform all positions in a grid."""
    s = grid.width  # square grid
    tp = lambda p: _transform_pos(p, s, rotation, flip_h)
    new_zones = {}
    for zid, zone in grid.tidal_zones.items():
        new_zones[zid] = TidalZone(
            zone_id=zone.zone_id,
            tiles=frozenset(tp(t) for t in zone.tiles),
            dry_duration=zone.dry_duration,
            flood_duration=zone.flood_duration,
            phase_offset=zone.phase_offset,
        )
    return TideGridGrid(
        width=grid.width,
        height=grid.height,
        walls=frozenset(tp(w) for w in grid.walls),
        tidal_zones=new_zones,
        beacons={
            bid: Beacon(beacon_id=bi.beacon_id, pos=tp(bi.pos))
            for bid, bi in grid.beacons.items()
        },
        start=tp(grid.start),
        exit_pos=tp(grid.exit_pos),
        initial_oxygen=grid.initial_oxygen,
    )


def _transform_state(
    state: TideGridState, size: int, rotation: int, flip_h: bool,
) -> TideGridState:
    """Transform agent_pos; time and collected pass through unchanged."""
    return TideGridState(
        agent_pos=_transform_pos(state.agent_pos, size, rotation, flip_h),
        time=state.time,
        collected_beacons=state.collected_beacons,
    )


def _transform_instance(
    inst: GeneratedInstance, rng: random.Random,
) -> GeneratedInstance:
    """Apply a random D4 symmetry to a finished instance (square grids only)."""
    if inst.grid.width != inst.grid.height:
        return inst  # D4 only for square grids
    rotation = rng.choice([0, 90, 180, 270])
    flip_h = rng.choice([True, False])
    if rotation == 0 and not flip_h:
        return inst

    s = inst.grid.width
    new_grid = _transform_grid(inst.grid, rotation, flip_h)
    new_state = _transform_state(inst.initial_state, s, rotation, flip_h)
    new_solution = [
        _transform_action(a, rotation, flip_h) for a in inst.solution
    ]
    new_catalog = []
    for entry in inst.deadlock_catalog:
        new_catalog.append(DeadlockEntry(
            path=[_transform_action(a, rotation, flip_h) for a in entry.path],
            fatal_step=entry.fatal_step,
            deadlock_state=_transform_state(entry.deadlock_state, s, rotation, flip_h),
            pre_deadlock_state=_transform_state(entry.pre_deadlock_state, s, rotation, flip_h),
            k_horizon=entry.k_horizon,
            strategy_name=entry.strategy_name,
        ))
    return GeneratedInstance(
        grid=new_grid,
        initial_state=new_state,
        solution=new_solution,
        deadlock_catalog=new_catalog,
        seed=inst.seed,
        template_name=inst.template_name,
        grid_width=inst.grid_width,
        grid_height=inst.grid_height,
        num_beacons=inst.num_beacons,
        num_zones=inst.num_zones,
        solution_length=inst.solution_length,
        tier1_deadlocks=inst.tier1_deadlocks,
        tier2_deadlocks=inst.tier2_deadlocks,
        max_k_horizon=inst.max_k_horizon,
        generation_attempts=inst.generation_attempts,
        fingerprint=inst.fingerprint,
    )


# ---------------------------------------------------------------------------
# Greedy strategies (10 strategies — all must fail for anti-greedy property)
# ---------------------------------------------------------------------------

def _move_toward_manhattan(
    state: TideGridState,
    target: Pos,
    legal: list[TideGridAction],
) -> Optional[TideGridAction]:
    """Pick the legal MOVE action that strictly reduces manhattan distance to target.

    Returns None if no legal move makes progress (i.e., all moves maintain or
    increase manhattan distance). This ensures the agent doesn't oscillate.
    """
    current_md = manhattan_distance(state.agent_pos, target)
    moves = [a for a in legal if a.is_move]
    improving = [
        a for a in moves
        if manhattan_distance(add_pos(state.agent_pos, a.direction), target) < current_md
    ]
    if not improving:
        return None
    return min(
        improving,
        key=lambda a: manhattan_distance(add_pos(state.agent_pos, a.direction), target),
    )


def _near_flooded_tidal_on_path(
    grid: TideGridGrid,
    state: TideGridState,
    target: Pos,
) -> bool:
    """Check if agent is near (within 2 steps of) a tidal tile that lies
    between agent and target and is currently flooded."""
    t = state.time + 1
    agent_md = manhattan_distance(state.agent_pos, target)
    for zone in grid.tidal_zones.values():
        if zone.is_dry_at(t):
            continue
        for tile in zone.tiles:
            via = (manhattan_distance(state.agent_pos, tile)
                   + manhattan_distance(tile, target))
            if via > agent_md + 2:
                continue
            if manhattan_distance(state.agent_pos, tile) <= 2:
                return True
    return False


def _nearest_crossing_toward(
    grid: TideGridGrid,
    state: TideGridState,
    target: Pos,
) -> Optional[Pos]:
    """Find the nearest tidal tile that lies between agent and target.

    A tidal tile is 'between' if going through it could be on a path to
    the target: agent→tile + tile→target <= agent→target + 2 (small detour).
    """
    agent_md = manhattan_distance(state.agent_pos, target)
    best: Optional[Pos] = None
    best_dist = float("inf")
    for zone in grid.tidal_zones.values():
        for tile in zone.tiles:
            via = (manhattan_distance(state.agent_pos, tile)
                   + manhattan_distance(tile, target))
            # Allow small detour (at most 2 extra steps)
            if via > agent_md + 2:
                continue
            d = manhattan_distance(state.agent_pos, tile)
            if d < best_dist:
                best_dist = d
                best = tile
    return best


def _greedy_step(
    grid: TideGridGrid,
    state: TideGridState,
    target: Pos,
    legal: list[TideGridAction],
) -> Optional[TideGridAction]:
    """Greedy step toward target: move if possible, WAIT at flooded crossing.

    When blocked by a flooded tidal tile on the path, WAITs in place
    (simulating 'wait for the tide') instead of oscillating.  This is the
    natural greedy behavior and is exactly the bait trap.
    """
    action = _move_toward_manhattan(state, target, legal)
    if action is not None:
        return action

    # Can't reduce manhattan to target — find nearest crossing on path
    crossing = _nearest_crossing_toward(grid, state, target)
    if crossing is not None:
        t_next = state.time + 1
        zone = grid.zone_at(crossing)

        # If crossing is flooded, WAIT for it to open
        if zone is not None and not zone.is_dry_at(t_next):
            waits = [a for a in legal if a.is_wait]
            if waits:
                return waits[0]

        # If crossing is dry (or we need to reach it), move toward crossing
        action = _move_toward_manhattan(state, crossing, legal)
        if action is not None:
            return action

        # Adjacent to crossing but can't step on it yet — WAIT
        waits = [a for a in legal if a.is_wait]
        if waits:
            return waits[0]

    # Fallback: WAIT if possible
    waits = [a for a in legal if a.is_wait]
    if waits:
        return waits[0]

    return None


_DIRECTION_ORDERINGS: list[tuple[Direction, ...]] = [
    (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT),
    (Direction.DOWN, Direction.LEFT, Direction.UP, Direction.RIGHT),
    (Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN),
    (Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP),
    (Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT),
    (Direction.DOWN, Direction.RIGHT, Direction.UP, Direction.LEFT),
    (Direction.LEFT, Direction.DOWN, Direction.RIGHT, Direction.UP),
    (Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN),
]


def _bfs_next_step(
    grid: TideGridGrid,
    state: TideGridState,
    target: Pos,
    direction_order: Optional[tuple[Direction, ...]] = None,
) -> Optional[TideGridAction]:
    """BFS on the static grid (treating tidal as always passable) to find
    the first move toward target.

    If direction_order is provided, neighbors are explored in that order
    (controls tiebreaking among equal-length paths).
    """
    passable: set[Pos] = set()
    for r in range(grid.height):
        for c in range(grid.width):
            p = Pos(r, c)
            if grid.tile_type(p) != Tile.WALL:
                passable.add(p)

    ordered_actions = MOVE_ACTIONS
    if direction_order is not None:
        ordered_actions = [_ACTION_BY_DIR[d] for d in direction_order]

    # BFS from agent, track first action taken for each reachable tile
    dist: dict[Pos, int] = {state.agent_pos: 0}
    first_action: dict[Pos, TideGridAction] = {}
    q = deque([(state.agent_pos, None)])
    while q:
        p, fa = q.popleft()
        for a in ordered_actions:
            np_ = add_pos(p, a.direction)
            if np_ in passable and np_ not in dist:
                dist[np_] = dist[p] + 1
                act = fa if fa is not None else a
                first_action[np_] = act
                q.append((np_, act))

    return first_action.get(target)


def simulate_greedy_nearest_beacon(
    grid: TideGridGrid,
    state: TideGridState,
    max_steps: int = 60,
) -> tuple[list[TideGridAction], TideGridState]:
    """Strategy 1: go to closest uncollected beacon by manhattan, then exit.

    When blocked by a flooded tidal crossing, WAIT at the crossing until
    it opens — this is the bait trap (waiting drains oxygen).
    """
    trace: list[TideGridAction] = []
    prev_pos = None
    wait_count = 0

    for _ in range(max_steps):
        if is_solved(grid, state):
            break

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        # Stuck detection: WAITing too long at same position
        if state.agent_pos == prev_pos:
            wait_count += 1
            if wait_count > 10:
                break
        else:
            wait_count = 0
        prev_pos = state.agent_pos

        # Choose target: nearest uncollected beacon, or exit if all collected
        uncollected = [
            grid.beacons[b] for b in grid.beacons
            if b not in state.collected_beacons
        ]
        if uncollected:
            target = min(uncollected,
                         key=lambda b: manhattan_distance(state.agent_pos, b.pos)).pos
        else:
            target = grid.exit_pos

        action = _greedy_step(grid, state, target, legal)
        if action is None:
            break

        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


def _simulate_shortest_static_path_order(
    grid: TideGridGrid,
    state: TideGridState,
    direction_order: tuple[Direction, ...],
    max_steps: int = 60,
) -> tuple[list[TideGridAction], TideGridState]:
    """Inner loop for shortest_static_path with a fixed direction ordering."""
    trace: list[TideGridAction] = []
    prev_pos = None
    wait_count = 0

    for _ in range(max_steps):
        if is_solved(grid, state):
            break

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        if state.agent_pos == prev_pos:
            wait_count += 1
            if wait_count > 10:
                break
        else:
            wait_count = 0
        prev_pos = state.agent_pos

        uncollected = [
            grid.beacons[b] for b in grid.beacons
            if b not in state.collected_beacons
        ]
        target = grid.exit_pos
        if uncollected:
            # Use BFS distance for beacon selection (consistent with BFS pathfinding)
            passable: set[Pos] = set()
            for r in range(grid.height):
                for c in range(grid.width):
                    p = Pos(r, c)
                    if grid.tile_type(p) != Tile.WALL:
                        passable.add(p)
            bfs_dist = _bfs_distance(state.agent_pos, passable)
            best_beacon = None
            best_dist = float("inf")
            for b in uncollected:
                d = bfs_dist.get(b.pos, float("inf"))
                if d < best_dist:
                    best_dist = d
                    best_beacon = b
            if best_beacon is not None:
                target = best_beacon.pos
            else:
                target = min(uncollected,
                             key=lambda b: manhattan_distance(state.agent_pos, b.pos)).pos

        action = _bfs_next_step(grid, state, target, direction_order)
        if action is not None and action in legal:
            trace.append(action)
            state = apply_action(grid, state, action)
        else:
            waits = [a for a in legal if a.is_wait]
            if waits:
                trace.append(waits[0])
                state = apply_action(grid, state, waits[0])
            else:
                break

    return trace, state


def simulate_greedy_shortest_static_path(
    grid: TideGridGrid,
    state: TideGridState,
    max_steps: int = 60,
) -> tuple[list[TideGridAction], TideGridState]:
    """Strategy 2: BFS to each beacon ignoring tidal phases (treat as passable).

    Go to closest beacon by BFS distance, then exit. WAIT at flooded crossings.
    Tries all 8 direction orderings and returns best result (solved, or longest trace).
    """
    best_trace: list[TideGridAction] = []
    best_state = state
    for order in _DIRECTION_ORDERINGS:
        trace, final = _simulate_shortest_static_path_order(
            grid, state, order, max_steps,
        )
        if is_solved(grid, final):
            return trace, final
        if len(trace) > len(best_trace):
            best_trace, best_state = trace, final
    return best_trace, best_state


def simulate_greedy_soonest_opening(
    grid: TideGridGrid,
    state: TideGridState,
    max_steps: int = 60,
) -> tuple[list[TideGridAction], TideGridState]:
    """Strategy 3: head to nearest beacon; when blocked, WAIT for the zone
    that opens soonest instead of detouring."""
    trace: list[TideGridAction] = []
    prev_pos = None
    wait_count = 0

    for _ in range(max_steps):
        if is_solved(grid, state):
            break

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        if state.agent_pos == prev_pos:
            wait_count += 1
            if wait_count > 10:
                break
        else:
            wait_count = 0
        prev_pos = state.agent_pos

        uncollected = [
            grid.beacons[b] for b in grid.beacons
            if b not in state.collected_beacons
        ]
        if uncollected:
            target = min(uncollected,
                         key=lambda b: manhattan_distance(state.agent_pos, b.pos)).pos
        else:
            target = grid.exit_pos

        action = _move_toward_manhattan(state, target, legal)
        if action is not None:
            trace.append(action)
            state = apply_action(grid, state, action)
            continue

        # Blocked — WAIT for nearest zone to open (simulates patience at crossing)
        waits = [a for a in legal if a.is_wait]
        if waits:
            trace.append(waits[0])
            state = apply_action(grid, state, waits[0])
        else:
            break

    return trace, state


def simulate_greedy_min_wait(
    grid: TideGridGrid,
    state: TideGridState,
    max_steps: int = 60,
) -> tuple[list[TideGridAction], TideGridState]:
    """Strategy 4: choose beacon ordering that locally minimizes WAIT count.

    For each uncollected beacon, estimate WAITs needed to reach it (based on
    tidal phases), then pick the one with fewest WAITs. WAIT at crossings.
    """
    trace: list[TideGridAction] = []
    prev_pos = None
    wait_count = 0

    def _estimate_waits(pos: Pos, target: Pos, t: int) -> int:
        """Rough estimate of WAITs needed: manhattan + check tidal blocking."""
        md = manhattan_distance(pos, target)
        waits = 0
        for zone in grid.tidal_zones.values():
            for tile in zone.tiles:
                if (min(pos.row, target.row) <= tile.row <= max(pos.row, target.row) and
                        min(pos.col, target.col) <= tile.col <= max(pos.col, target.col)):
                    arrival_est = t + manhattan_distance(pos, tile)
                    if not zone.is_dry_at(arrival_est):
                        for dt in range(1, zone.cycle_length + 1):
                            if zone.is_dry_at(arrival_est + dt):
                                waits += dt
                                break
        return md + waits

    for _ in range(max_steps):
        if is_solved(grid, state):
            break

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        if state.agent_pos == prev_pos:
            wait_count += 1
            if wait_count > 10:
                break
        else:
            wait_count = 0
        prev_pos = state.agent_pos

        uncollected = [
            grid.beacons[b] for b in grid.beacons
            if b not in state.collected_beacons
        ]
        if uncollected:
            target = min(
                uncollected,
                key=lambda b: _estimate_waits(state.agent_pos, b.pos, state.time),
            ).pos
        else:
            target = grid.exit_pos

        action = _greedy_step(grid, state, target, legal)
        if action is None:
            break

        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


def simulate_greedy_avoid_tidal(
    grid: TideGridGrid,
    state: TideGridState,
    max_steps: int = 60,
) -> tuple[list[TideGridAction], TideGridState]:
    """Strategy 5: prefer moves that avoid tidal tiles.

    Head toward nearest beacon, preferring non-tidal tiles. When forced
    to cross tidal, WAIT if flooded.
    """
    trace: list[TideGridAction] = []
    prev_pos = None
    wait_count = 0

    for _ in range(max_steps):
        if is_solved(grid, state):
            break

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        if state.agent_pos == prev_pos:
            wait_count += 1
            if wait_count > 10:
                break
        else:
            wait_count = 0
        prev_pos = state.agent_pos

        uncollected = [
            grid.beacons[b] for b in grid.beacons
            if b not in state.collected_beacons
        ]
        if uncollected:
            target = min(uncollected,
                         key=lambda b: manhattan_distance(state.agent_pos, b.pos)).pos
        else:
            target = grid.exit_pos

        # Prefer non-tidal moves that strictly reduce manhattan distance
        current_md = manhattan_distance(state.agent_pos, target)
        moves = [a for a in legal if a.is_move]
        improving = [
            a for a in moves
            if manhattan_distance(add_pos(state.agent_pos, a.direction), target) < current_md
        ]
        non_tidal_improving = [
            a for a in improving
            if grid.tile_type(add_pos(state.agent_pos, a.direction)) != Tile.TIDAL
        ]

        if non_tidal_improving:
            action = min(
                non_tidal_improving,
                key=lambda a: manhattan_distance(
                    add_pos(state.agent_pos, a.direction), target),
            )
        elif improving:
            action = min(
                improving,
                key=lambda a: manhattan_distance(
                    add_pos(state.agent_pos, a.direction), target),
            )
        else:
            # No improving moves — WAIT at crossing
            waits = [a for a in legal if a.is_wait]
            action = waits[0] if waits else None

        if action is None:
            break

        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


def simulate_greedy_rush_exit(
    grid: TideGridGrid,
    state: TideGridState,
    max_steps: int = 60,
) -> tuple[list[TideGridAction], TideGridState]:
    """Strategy 6: after collecting any beacon, head directly to exit.

    Only collects other beacons if they happen to be on the path.
    Prioritizes reaching the exit quickly. WAIT at flooded crossings.
    """
    trace: list[TideGridAction] = []
    prev_pos = None
    wait_count = 0

    for _ in range(max_steps):
        if is_solved(grid, state):
            break

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        if state.agent_pos == prev_pos:
            wait_count += 1
            if wait_count > 10:
                break
        else:
            wait_count = 0
        prev_pos = state.agent_pos

        uncollected = [
            grid.beacons[b] for b in grid.beacons
            if b not in state.collected_beacons
        ]

        if len(state.collected_beacons) > 0 and uncollected:
            # Rush to exit, but if uncollected beacon is very close, grab it
            nearest_beacon = min(uncollected,
                                 key=lambda b: manhattan_distance(state.agent_pos, b.pos))
            beacon_dist = manhattan_distance(state.agent_pos, nearest_beacon.pos)
            if beacon_dist <= 2:
                target = nearest_beacon.pos
            else:
                target = grid.exit_pos
        elif uncollected:
            target = min(uncollected,
                         key=lambda b: manhattan_distance(state.agent_pos, b.pos)).pos
        else:
            target = grid.exit_pos

        action = _greedy_step(grid, state, target, legal)
        if action is None:
            break

        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


# ---------------------------------------------------------------------------
# Smart greedy strategies (timing/oxygen-aware)
# ---------------------------------------------------------------------------

def simulate_greedy_phase_aware_beacon(
    grid: TideGridGrid,
    state: TideGridState,
    max_steps: int = 60,
) -> tuple[list[TideGridAction], TideGridState]:
    """Strategy 7: tidal schedule reader — estimate full trip cost per beacon.

    For each uncollected beacon, estimate total cost = manhattan distance +
    WAIT penalty at each tidal zone crossing on the path.  Pick the beacon
    with lowest total cost (beacon + exit trip).  Models an LLM that reads
    tidal schedules and plans ahead.

    Differs from min_wait: min_wait only counts whether crossings are flooded
    at arrival.  phase_aware_beacon adds the actual wait duration and computes
    full trip cost (pos → beacon → exit).
    """
    trace: list[TideGridAction] = []
    prev_pos = None
    wait_count = 0

    def _trip_cost(pos: Pos, target: Pos, t: int) -> int:
        """Estimate cost of reaching target from pos at time t, including waits."""
        md = manhattan_distance(pos, target)
        total_wait = 0
        for zone in grid.tidal_zones.values():
            for tile in zone.tiles:
                # Check if tile is in the bounding box between pos and target
                if (min(pos.row, target.row) <= tile.row <= max(pos.row, target.row) and
                        min(pos.col, target.col) <= tile.col <= max(pos.col, target.col)):
                    arrival_est = t + manhattan_distance(pos, tile)
                    if not zone.is_dry_at(arrival_est):
                        # Compute actual wait duration until zone opens
                        for dt in range(1, zone.cycle_length + 1):
                            if zone.is_dry_at(arrival_est + dt):
                                total_wait += dt
                                break
                        else:
                            total_wait += zone.cycle_length
        return md + total_wait

    for _ in range(max_steps):
        if is_solved(grid, state):
            break

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        if state.agent_pos == prev_pos:
            wait_count += 1
            if wait_count > 10:
                break
        else:
            wait_count = 0
        prev_pos = state.agent_pos

        uncollected = [
            grid.beacons[b] for b in grid.beacons
            if b not in state.collected_beacons
        ]
        if uncollected:
            # Pick beacon with lowest full trip cost (pos→beacon→exit)
            def full_trip_cost(b: Beacon) -> int:
                leg1 = _trip_cost(state.agent_pos, b.pos, state.time)
                leg2_t = state.time + leg1
                leg2 = _trip_cost(b.pos, grid.exit_pos, leg2_t)
                return leg1 + leg2
            target = min(uncollected, key=full_trip_cost).pos
        else:
            target = grid.exit_pos

        action = _greedy_step(grid, state, target, legal)
        if action is None:
            break

        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


def simulate_greedy_oxygen_guard(
    grid: TideGridGrid,
    state: TideGridState,
    max_steps: int = 60,
) -> tuple[list[TideGridAction], TideGridState]:
    """Strategy 8: fuel gauge monitor — reject actions that breach oxygen budget.

    For each legal action, hypothetically apply it and check if remaining
    oxygen exceeds heuristic(grid, hypo_state) (NN-TSP lower bound).
    Reject actions that would make the heuristic estimate unsolvability.
    Among viable actions, prefer those that reduce manhattan to target.

    Models an LLM that monitors its fuel gauge.
    """
    trace: list[TideGridAction] = []
    prev_pos = None
    wait_count = 0

    for _ in range(max_steps):
        if is_solved(grid, state):
            break

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        if state.agent_pos == prev_pos:
            wait_count += 1
            if wait_count > 10:
                break
        else:
            wait_count = 0
        prev_pos = state.agent_pos

        # Pick target: nearest uncollected beacon or exit
        uncollected = [
            grid.beacons[b] for b in grid.beacons
            if b not in state.collected_beacons
        ]
        if uncollected:
            target = min(uncollected,
                         key=lambda b: manhattan_distance(state.agent_pos, b.pos)).pos
        else:
            target = grid.exit_pos

        # Filter actions by oxygen budget feasibility
        remaining_oxygen = grid.initial_oxygen - state.time
        viable: list[TideGridAction] = []
        for a in legal:
            hypo_state = apply_action(grid, state, a)
            hypo_remaining = grid.initial_oxygen - hypo_state.time
            h = heuristic(grid, hypo_state)
            if hypo_remaining >= h:
                viable.append(a)

        if not viable:
            # All actions bust the budget — take the least-bad one
            viable = legal

        # Among viable, prefer moves that reduce manhattan to target
        action = _move_toward_manhattan(state, target, viable)
        if action is not None:
            trace.append(action)
            state = apply_action(grid, state, action)
            continue

        # No improving move among viable — WAIT if viable
        waits = [a for a in viable if a.is_wait]
        if waits:
            trace.append(waits[0])
            state = apply_action(grid, state, waits[0])
        else:
            # Take any viable action
            trace.append(viable[0])
            state = apply_action(grid, state, viable[0])

    return trace, state


def simulate_greedy_farthest_beacon_first(
    grid: TideGridGrid,
    state: TideGridState,
    max_steps: int = 60,
) -> tuple[list[TideGridAction], TideGridState]:
    """Strategy 9: visit farthest uncollected beacon first.

    Simple max instead of min for beacon selection by manhattan distance.
    If both nearest-first and farthest-first fail, the puzzle requires
    genuine timing reasoning.
    """
    trace: list[TideGridAction] = []
    prev_pos = None
    wait_count = 0

    for _ in range(max_steps):
        if is_solved(grid, state):
            break

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        if state.agent_pos == prev_pos:
            wait_count += 1
            if wait_count > 10:
                break
        else:
            wait_count = 0
        prev_pos = state.agent_pos

        uncollected = [
            grid.beacons[b] for b in grid.beacons
            if b not in state.collected_beacons
        ]
        if uncollected:
            target = max(uncollected,
                         key=lambda b: manhattan_distance(state.agent_pos, b.pos)).pos
        else:
            target = grid.exit_pos

        action = _greedy_step(grid, state, target, legal)
        if action is None:
            break

        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


def simulate_greedy_bfs_tidal_aware(
    grid: TideGridGrid,
    state: TideGridState,
    max_steps: int = 60,
) -> tuple[list[TideGridAction], TideGridState]:
    """Strategy 10: dynamic BFS treating currently-flooded tidal tiles as walls.

    Tries all 8 direction orderings. When BFS finds no path (tidal blocking),
    WAITs. Most expensive strategy.
    """
    best_trace: list[TideGridAction] = []
    best_state = state

    for order in _DIRECTION_ORDERINGS:
        trace, final = _simulate_bfs_tidal_aware_order(
            grid, state, order, max_steps,
        )
        if is_solved(grid, final):
            return trace, final
        if len(trace) > len(best_trace):
            best_trace, best_state = trace, final

    return best_trace, best_state


def _simulate_bfs_tidal_aware_order(
    grid: TideGridGrid,
    state: TideGridState,
    direction_order: tuple[Direction, ...],
    max_steps: int = 60,
) -> tuple[list[TideGridAction], TideGridState]:
    """Inner loop for bfs_tidal_aware with a fixed direction ordering."""
    trace: list[TideGridAction] = []
    prev_pos = None
    wait_count = 0

    ordered_actions = [_ACTION_BY_DIR[d] for d in direction_order]

    for _ in range(max_steps):
        if is_solved(grid, state):
            break

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        if state.agent_pos == prev_pos:
            wait_count += 1
            if wait_count > 10:
                break
        else:
            wait_count = 0
        prev_pos = state.agent_pos

        uncollected = [
            grid.beacons[b] for b in grid.beacons
            if b not in state.collected_beacons
        ]
        if uncollected:
            target = min(uncollected,
                         key=lambda b: manhattan_distance(state.agent_pos, b.pos)).pos
        else:
            target = grid.exit_pos

        # Time-dependent BFS: treat currently-flooded tidal tiles as impassable
        t_next = state.time + 1
        passable: set[Pos] = set()
        for r in range(grid.height):
            for c in range(grid.width):
                p = Pos(r, c)
                if is_passable_at(grid, p, t_next):
                    passable.add(p)

        # BFS from agent with time-dependent passable set
        dist: dict[Pos, int] = {state.agent_pos: 0}
        first_action: dict[Pos, TideGridAction] = {}
        q = deque([(state.agent_pos, None)])
        while q:
            p, fa = q.popleft()
            for a in ordered_actions:
                np_ = add_pos(p, a.direction)
                if np_ in passable and np_ not in dist:
                    dist[np_] = dist[p] + 1
                    act = fa if fa is not None else a
                    first_action[np_] = act
                    q.append((np_, act))

        action = first_action.get(target)
        if action is not None and action in legal:
            trace.append(action)
            state = apply_action(grid, state, action)
        else:
            # BFS found no path (tidal blocking) — WAIT
            waits = [a for a in legal if a.is_wait]
            if waits:
                trace.append(waits[0])
                state = apply_action(grid, state, waits[0])
            else:
                break

    return trace, state


# Strategy registry
GREEDY_STRATEGIES: list[tuple[str, Callable]] = [
    # Existing basic strategies (cheap first)
    ("nearest_beacon", simulate_greedy_nearest_beacon),
    ("shortest_static_path", simulate_greedy_shortest_static_path),
    ("soonest_opening", simulate_greedy_soonest_opening),
    ("min_wait", simulate_greedy_min_wait),
    ("avoid_tidal", simulate_greedy_avoid_tidal),
    ("rush_exit", simulate_greedy_rush_exit),
    # Smart strategies (timing/oxygen-aware, expensive last)
    ("phase_aware_beacon", simulate_greedy_phase_aware_beacon),
    ("oxygen_guard", simulate_greedy_oxygen_guard),
    ("farthest_beacon_first", simulate_greedy_farthest_beacon_first),
    ("bfs_tidal_aware", simulate_greedy_bfs_tidal_aware),
]


def all_greedy_strategies_fail(
    grid: TideGridGrid,
    initial_state: TideGridState,
) -> bool:
    """Check that NO greedy strategy solves the puzzle.

    Returns True only if ALL strategies fail (don't reach exit with all beacons).
    """
    for name, strategy_fn in GREEDY_STRATEGIES:
        _trace, final_state = strategy_fn(grid, initial_state)
        if is_solved(grid, final_state):
            return False
    return True


# ---------------------------------------------------------------------------
# Deadlock discovery
# ---------------------------------------------------------------------------

def discover_deadlocks(
    grid: TideGridGrid,
    initial_state: TideGridState,
    max_depth: int = 15,
    max_states: int = 50_000,
) -> list[DeadlockEntry]:
    """Forward sweep from initial state to discover Walking Dead states.

    In Tide Grid, every action consumes oxygen and can potentially cause
    deadlock.  We use oxygen pruning for fast solvability checks.

    Monotonicity: once unsolvable, always unsolvable (oxygen only decreases,
    time only increases, beacons only accumulate).
    """
    catalog: list[DeadlockEntry] = []
    queue: deque[tuple[TideGridState, list[TideGridAction], bool]] = deque()
    queue.append((initial_state, [], True))
    visited: set = {initial_state.to_hashable()}
    states_explored = 0

    while queue and states_explored < max_states:
        state, path, parent_solvable = queue.popleft()

        if len(path) >= max_depth:
            continue

        for action in get_legal_actions(grid, state):
            next_state = apply_action(grid, state, action)
            h = next_state.to_hashable()
            if h in visited:
                continue
            visited.add(h)
            states_explored += 1
            new_path = path + [action]

            # Determine solvability
            if not parent_solvable:
                next_solvable = False
            else:
                next_solvable = is_solvable(grid, next_state)

            # Detect solvability flip
            if parent_solvable and not next_solvable:
                if is_walking_dead(grid, next_state):
                    catalog.append(DeadlockEntry(
                        path=new_path,
                        fatal_step=len(new_path) - 1,
                        deadlock_state=next_state,
                        pre_deadlock_state=state,
                    ))

            # Continue exploring from solvable or Walking Dead states
            if next_solvable or (not next_solvable and
                                  len(get_legal_actions(grid, next_state)) > 0):
                queue.append((next_state, new_path, next_solvable))

    return catalog


def discover_greedy_deadlocks(
    grid: TideGridGrid,
    initial_state: TideGridState,
) -> list[DeadlockEntry]:
    """Trace-based deadlock discovery: run each greedy strategy and find
    where solvability flips.

    Produces deadlocks where the fatal step = "chose to go toward bait
    crossing instead of correct one", which is meaningful for RQ3.
    """
    catalog: list[DeadlockEntry] = []

    for name, strategy_fn in GREEDY_STRATEGIES:
        trace, final = strategy_fn(grid, initial_state)

        # Skip strategies that solve the puzzle
        if is_solved(grid, final):
            continue

        # Skip empty traces
        if not trace:
            continue

        # Find where solvability flips
        fatal = find_fatal_step(grid, initial_state, trace)
        if fatal < 0:
            continue

        dead_state = apply_actions(grid, initial_state, trace[:fatal + 1])
        pre_state = apply_actions(grid, initial_state, trace[:fatal])

        if not is_walking_dead(grid, dead_state):
            continue

        kh = count_reachable_dead_states(grid, dead_state)
        catalog.append(DeadlockEntry(
            path=trace[:fatal + 1],
            fatal_step=fatal,
            deadlock_state=dead_state,
            pre_deadlock_state=pre_state,
            k_horizon=kh,
            strategy_name=name,
        ))

    return catalog


def discover_oracle_adjacent_deadlocks(
    grid: TideGridGrid,
    initial_state: TideGridState,
    solution: list[TideGridAction],
) -> list[DeadlockEntry]:
    """Find deadlocks one step off the oracle solution.

    Walk the oracle solution step-by-step. At each step *i*, try every
    legal action that differs from the oracle move. If any leads to a
    Walking Dead state, record it as an oracle-adjacent deadlock whose
    path is ``solution[:i] + [alt_action]``.

    Optimization: only test alternatives when oxygen margin is tight
    (remaining_oxygen - heuristic <= len(solution) - i + 3). Early in the
    solution when oxygen is plentiful, one wrong step rarely causes deadlock.

    Unlike chain_vault (only tests USE_KEY alternatives): Tide Grid must
    test ALL alternatives since any action consumes oxygen.

    Returns a list of DeadlockEntry with ``k_horizon`` already computed.
    """
    catalog: list[DeadlockEntry] = []
    state = initial_state

    for i, oracle_action in enumerate(solution):
        # Optimization: skip when oxygen margin is comfortable
        remaining_oxygen = grid.initial_oxygen - state.time
        h = heuristic(grid, state)
        steps_left = len(solution) - i
        if remaining_oxygen - h > steps_left + 3:
            state = apply_action(grid, state, oracle_action)
            continue

        legal = get_legal_actions(grid, state)
        for alt in legal:
            if alt == oracle_action:
                continue
            alt_state = apply_action(grid, state, alt)
            if is_walking_dead(grid, alt_state):
                k_horizon = count_reachable_dead_states(grid, alt_state)
                catalog.append(DeadlockEntry(
                    path=list(solution[:i]) + [alt],
                    fatal_step=i,
                    deadlock_state=alt_state,
                    pre_deadlock_state=state,
                    k_horizon=k_horizon,
                    strategy_name="oracle_adjacent",
                ))
        # Advance along the oracle path
        state = apply_action(grid, state, oracle_action)

    return catalog


def generate_hard_rq2_pair(
    grid: TideGridGrid,
    initial_state: TideGridState,
    solution: list[TideGridAction],
    catalog_entry: DeadlockEntry,
) -> Optional[tuple[TideGridState, list[TideGridAction], list[TideGridAction]]]:
    """Generate a hard matched deadlock/solvable pair for RQ2.

    Both paths share the first ``fatal_step`` actions (the oracle prefix)
    and differ only on the last move.

    Returns (initial_state, deadlock_actions, solvable_actions) or None.
    """
    i = catalog_entry.fatal_step
    bad_action = catalog_entry.path[i]  # the non-oracle action

    deadlock_actions = list(solution[:i]) + [bad_action]
    solvable_actions = list(solution[:i + 1])

    # Verify deadlock path → Walking Dead
    dead_state = apply_actions(grid, initial_state, deadlock_actions)
    if not is_walking_dead(grid, dead_state):
        return None

    # Verify solvable path → still solvable
    sol_state = apply_actions(grid, initial_state, solvable_actions)
    if not is_solvable(grid, sol_state):
        return None

    return initial_state, deadlock_actions, solvable_actions


def _walking_dead_continuation(
    grid: TideGridGrid,
    dead_state: TideGridState,
    max_steps: int,
    num_attempts: int = 10,
) -> list[TideGridAction]:
    """Generate a Walking Dead continuation that maximizes trace length.

    Tries ``num_attempts`` RNG seeds.  On each attempt, prefers moves to
    non-tidal floor tiles so the agent avoids getting immediately trapped
    inside a flooding zone.  Returns the longest continuation found.
    """
    best: list[TideGridAction] = []

    for attempt in range(num_attempts):
        rng = random.Random(attempt * 7 + 13)
        continuation: list[TideGridAction] = []
        state = dead_state

        for _ in range(max_steps):
            legal = get_legal_actions(grid, state)
            if not legal:
                break

            # Prefer moves to non-tidal tiles (floor/exit) to avoid
            # getting stranded inside a tidal zone when it floods.
            floor_actions = [
                a for a in legal
                if a.is_move and grid.tile_type(
                    add_pos(state.agent_pos, a.direction)
                ) in (Tile.FLOOR, Tile.EXIT)
            ]
            if floor_actions:
                action = rng.choice(floor_actions)
            else:
                action = rng.choice(legal)

            continuation.append(action)
            state = apply_action(grid, state, action)

        if len(continuation) > len(best):
            best = continuation

        # Early exit if we already have a full-length continuation
        if len(best) >= max_steps:
            break

    return best


def generate_hard_rq3_instance(
    grid: TideGridGrid,
    initial_state: TideGridState,
    solution: list[TideGridAction],
    catalog_entry: DeadlockEntry,
    target_length: int = 0,
) -> Optional[tuple[list[TideGridAction], int, int]]:
    """Generate a hard RQ3 trace from an oracle-adjacent deadlock.

    The trace follows the oracle for ``fatal_step`` actions, takes one
    wrong turn, then continues with random legal Walking Dead moves to
    reach ``target_length``.

    When ``target_length`` is 0 (default), it is computed dynamically as
    ``max(18, len(trace_to_deadlock) + 6)`` so that complex templates
    with long solutions (gauntlet, deadline, switchback) are not silently
    dropped by the continuation-length check.

    Returns (full_trace, fatal_step, cause_symptom_gap) or None.
    """
    i = catalog_entry.fatal_step
    bad_action = catalog_entry.path[i]

    trace_to_deadlock = list(solution[:i]) + [bad_action]
    if target_length == 0:
        target_length = max(18, len(trace_to_deadlock) + 6)
    continuation_length = target_length - len(trace_to_deadlock)
    if continuation_length < 3:
        return None

    continuation = _walking_dead_continuation(
        grid, catalog_entry.deadlock_state, continuation_length,
    )

    full_trace = trace_to_deadlock + continuation
    cause_symptom_gap = len(full_trace) - i - 1

    if cause_symptom_gap < 3:
        return None

    return full_trace, i, cause_symptom_gap


# ---------------------------------------------------------------------------
# Template 1: Strait Crossing
# ---------------------------------------------------------------------------

def generate_strait_grid(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[TideGridGrid, TideGridState]]:
    """Generate a Strait Crossing instance with bait-by-construction.

    Layout: wall barrier at middle row with two tidal chokepoints (Zone A, Zone B).

    Bait mechanism:
    - Agent starts top-left region (near Zone A crossing)
    - Beacon B1 placed below Zone A (bottom-left) — nearest beacon, across bait crossing
    - Beacon B2 placed top-right — accessible via floor without crossing
    - Exit placed bottom-right — must cross barrier to reach
    - Zone A offset calculated: floods at natural arrival_time from start → bait
    - Zone B offset calculated: dry at arrival_time via B2 → correct path
    - Oxygen set to generous placeholder; recalibrated by validation pipeline
    """
    if width < 6 or height < 6:
        return None

    barrier_row = height // 2

    # Pick two columns for tidal crossings (at least 2 apart)
    # Zone A (bait) on the left side, Zone B (correct) on the right
    left_cols = list(range(1, width // 2))
    right_cols = list(range(width // 2 + 1, width - 1))
    if not left_cols or not right_cols:
        return None

    cross_a = rng.choice(left_cols)
    cross_b = rng.choice(right_cols)

    # Build walls: entire barrier row except crossing columns
    walls: set[Pos] = set()
    for c in range(width):
        if c not in (cross_a, cross_b):
            walls.add(Pos(barrier_row, c))

    # --- Placement with controlled randomness ---
    # Top-left region for agent start
    top_left = [Pos(r, c) for r in range(barrier_row)
                for c in range(width // 2)
                if Pos(r, c) not in walls]
    # Bottom-left region for bait beacon B1
    bot_left = [Pos(r, c) for r in range(barrier_row + 1, height)
                for c in range(width // 2)
                if Pos(r, c) not in walls]
    # Top-right region for safe beacon B2
    top_right = [Pos(r, c) for r in range(barrier_row)
                 for c in range(width // 2 + 1, width)
                 if Pos(r, c) not in walls]
    # Bottom-right region for exit
    bot_right = [Pos(r, c) for r in range(barrier_row + 1, height)
                 for c in range(width // 2 + 1, width)
                 if Pos(r, c) not in walls]

    if not top_left or not bot_left or not top_right or not bot_right:
        return None

    start_pos = rng.choice(top_left)
    b1_pos = rng.choice(bot_left)       # bait beacon — below Zone A
    b2_pos = rng.choice(top_right)      # safe beacon — same side as agent
    exit_pos = rng.choice(bot_right)    # exit — below Zone B

    # --- Zone parameters: bait by phase alignment ---
    crossing_a_pos = Pos(barrier_row, cross_a)
    crossing_b_pos = Pos(barrier_row, cross_b)

    # Greedy arrival time at Zone A: agent goes straight toward b1 (nearest beacon)
    greedy_arrival_a = manhattan_distance(start_pos, crossing_a_pos)

    # Correct-path arrival time at Zone B: agent goes to b2 first, then to Zone B
    correct_arrival_b = (manhattan_distance(start_pos, b2_pos)
                         + manhattan_distance(b2_pos, crossing_b_pos))

    # --- E2/E3/E4 coin flips (consumed at fixed point for RNG consistency) ---
    enhanced_cycles  = width >= 7 and rng.random() < 0.5  # E2
    enhanced_coprime = width >= 7 and rng.random() < 0.5  # E3
    enhanced_similar = width >= 7 and rng.random() < 0.5  # E4

    # Zone A (bait): dry window is SHORT, and offset so it's FLOODED at greedy_arrival_a
    dry_a = rng.randint(2, 3)
    flood_a = rng.randint(3, 5)
    if enhanced_cycles:
        dry_a, flood_a = _scale_cycle_params(dry_a, flood_a, rng)
    cycle_a = dry_a + flood_a
    # We want zone A flooded at time greedy_arrival_a:
    # (greedy_arrival_a + offset_a) % cycle_a >= dry_a  (in flood phase)
    # Set offset so arrival lands in the middle of flood phase
    flood_mid = dry_a + flood_a // 2
    offset_a = (flood_mid - greedy_arrival_a % cycle_a) % cycle_a

    # Zone B (correct): offset so it's DRY at correct_arrival_b
    dry_b = rng.randint(2, 4)
    flood_b = rng.randint(2, 4)
    if enhanced_similar:
        dry_b, flood_b = dry_a, flood_a
    elif enhanced_cycles:
        dry_b, flood_b = _scale_cycle_params(dry_b, flood_b, rng)
    if enhanced_coprime:
        flood_b = _ensure_coprime(flood_b, dry_b, [cycle_a])
    cycle_b = dry_b + flood_b
    # We want zone B dry at time correct_arrival_b:
    # (correct_arrival_b + offset_b) % cycle_b < dry_b  (in dry phase)
    # Set offset so arrival lands in the middle of dry phase
    dry_mid = dry_b // 2
    offset_b = (dry_mid - correct_arrival_b % cycle_b) % cycle_b

    zone_a = TidalZone(
        zone_id="A",
        tiles=frozenset({crossing_a_pos}),
        dry_duration=dry_a,
        flood_duration=flood_a,
        phase_offset=offset_a,
    )
    zone_b = TidalZone(
        zone_id="B",
        tiles=frozenset({crossing_b_pos}),
        dry_duration=dry_b,
        flood_duration=flood_b,
        phase_offset=offset_b,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=b1_pos),
        "b2": Beacon(beacon_id="b2", pos=b2_pos),
    }

    # Generous oxygen placeholder — recalibrated by validation pipeline
    oxygen = width + height + rng.randint(10, 20)

    grid = TideGridGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a, "B": zone_b},
        beacons=beacons,
        start=start_pos,
        exit_pos=exit_pos,
        initial_oxygen=oxygen,
    )
    return grid, make_initial_state(grid)


# ---------------------------------------------------------------------------
# Template 2: Tidal Maze
# ---------------------------------------------------------------------------

def generate_tidal_maze_grid(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[TideGridGrid, TideGridState]]:
    """Generate a Tidal Maze instance with bait-by-construction.

    Layout: two perpendicular tidal corridors (horizontal + vertical) divide
    the grid into quadrants. Center cell is a wall.

    Bait mechanism:
    - Agent in TL quadrant
    - Bait beacon (b1) in BL quadrant — nearest beacon, requires Zone A crossing
      with bad phase alignment (flooded at natural arrival time)
    - Correct beacon (b2) in TR quadrant — requires Zone B crossing with good
      phase alignment (dry at natural arrival time)
    - Exit in BR quadrant — requires crossing both zones
    - Going to nearest beacon (b1) first misaligns timing for subsequent crossings
    """
    if width < 7 or height < 7:
        return None
    if width % 2 == 0 or height % 2 == 0:
        return None

    mid_row = height // 2
    mid_col = width // 2

    # Center wall
    walls = {Pos(mid_row, mid_col)}

    # Zone A: horizontal corridor at mid_row
    zone_a_tiles = frozenset(
        {Pos(mid_row, c) for c in range(width) if c != mid_col}
    )
    # Zone B: vertical corridor at mid_col
    zone_b_tiles = frozenset(
        {Pos(r, mid_col) for r in range(height) if r != mid_row}
    )

    tidal_tiles = zone_a_tiles | zone_b_tiles

    # Quadrants
    quadrants: dict[str, list[Pos]] = {
        "TL": [], "TR": [], "BL": [], "BR": [],
    }
    for r in range(height):
        for c in range(width):
            p = Pos(r, c)
            if p in walls or p in tidal_tiles:
                continue
            if r < mid_row and c < mid_col:
                quadrants["TL"].append(p)
            elif r < mid_row and c > mid_col:
                quadrants["TR"].append(p)
            elif r > mid_row and c < mid_col:
                quadrants["BL"].append(p)
            elif r > mid_row and c > mid_col:
                quadrants["BR"].append(p)

    for q in quadrants.values():
        if len(q) < 2:
            return None

    # Fixed quadrant assignment for bait structure
    start_pos = rng.choice(quadrants["TL"])
    b1_pos = rng.choice(quadrants["BL"])       # bait beacon — nearest, bad zone A timing
    b2_pos = rng.choice(quadrants["TR"])       # correct beacon — good zone B timing
    exit_pos = rng.choice(quadrants["BR"])

    # --- Zone parameters: bait by phase alignment ---
    # Zone A crossing point closest to start (for going to b1)
    # The agent crosses Zone A at mid_row; closest column is start's column clamped
    cross_a_col = min(max(start_pos.col, 0), mid_col - 1)
    cross_a_pos = Pos(mid_row, cross_a_col)

    # Zone B crossing point closest to start (for going to b2)
    cross_b_row = min(max(start_pos.row, 0), mid_row - 1)
    cross_b_pos = Pos(cross_b_row, mid_col)

    # Greedy arrival at Zone A: go straight to bait beacon b1 (nearest)
    greedy_arrival_a = manhattan_distance(start_pos, cross_a_pos)

    # Correct-path arrival at Zone B: go to b2 first via Zone B
    correct_arrival_b = manhattan_distance(start_pos, cross_b_pos)

    # --- E2/E3/E4 coin flips ---
    enhanced_cycles  = width >= 7 and rng.random() < 0.5  # E2
    enhanced_coprime = width >= 7 and rng.random() < 0.5  # E3
    enhanced_similar = width >= 7 and rng.random() < 0.5  # E4

    # Zone A (bait): FLOODED at greedy_arrival_a
    dry_a = rng.randint(3, 5)
    flood_a = rng.randint(2, 4)
    if enhanced_cycles:
        dry_a, flood_a = _scale_cycle_params(dry_a, flood_a, rng)
    cycle_a = dry_a + flood_a
    flood_mid_a = dry_a + flood_a // 2
    offset_a = (flood_mid_a - greedy_arrival_a % cycle_a) % cycle_a

    # Zone B (correct): DRY at correct_arrival_b
    dry_b = rng.randint(3, 5)
    flood_b = rng.randint(2, 4)
    if enhanced_similar:
        dry_b, flood_b = dry_a, flood_a
    elif enhanced_cycles:
        dry_b, flood_b = _scale_cycle_params(dry_b, flood_b, rng)
    if enhanced_coprime:
        flood_b = _ensure_coprime(flood_b, dry_b, [cycle_a])
    cycle_b = dry_b + flood_b
    dry_mid_b = dry_b // 2
    offset_b = (dry_mid_b - correct_arrival_b % cycle_b) % cycle_b

    # Ensure zones have different cycles or offsets
    if (cycle_a == cycle_b) and (offset_a == offset_b):
        offset_b = (offset_b + 1) % cycle_b

    zone_a = TidalZone(
        zone_id="A", tiles=zone_a_tiles,
        dry_duration=dry_a, flood_duration=flood_a, phase_offset=offset_a,
    )
    zone_b = TidalZone(
        zone_id="B", tiles=zone_b_tiles,
        dry_duration=dry_b, flood_duration=flood_b, phase_offset=offset_b,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=b1_pos),
        "b2": Beacon(beacon_id="b2", pos=b2_pos),
    }

    # Generous oxygen placeholder — recalibrated by validation pipeline
    oxygen = width + height + rng.randint(12, 22)

    grid = TideGridGrid(
        width=width, height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a, "B": zone_b},
        beacons=beacons,
        start=start_pos,
        exit_pos=exit_pos,
        initial_oxygen=oxygen,
    )
    return grid, make_initial_state(grid)


# ---------------------------------------------------------------------------
# Template 3: Gauntlet
# ---------------------------------------------------------------------------

def generate_gauntlet_grid(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[TideGridGrid, TideGridState]]:
    """Generate a Gauntlet instance with cascading sequential timing.

    Layout: 3 horizontal barrier walls at H//4, H//2, 3H//4 with single-tile
    tidal crossings that zigzag left-right-left.  The agent must pass through
    all 3 barriers in series to reach the exit.

    Bait mechanism (cascading timing):
    - Beacon b2 in section 0 (free, same section as start)
    - Beacon b1 in section 1 (bait, across Zone A)
    - Greedy collects b2 first (nearest) → arrives at Zone A during flood phase
    - Wait cascade: delay at A shifts arrival at B and C into flood phases too
    - Optimal: brief wait (0-2 turns) syncs Zone A dry window → clean cascade
    """
    if width < 6 or height < 7:
        return None

    H, W = height, width

    # --- Barrier rows ---
    b1_row = H // 4
    b2_row = H // 2
    b3_row = 3 * H // 4

    # Barrier rows must be distinct and leave room for sections
    if b1_row < 1 or b2_row - b1_row < 2 or b3_row - b2_row < 2 or H - 1 - b3_row < 1:
        return None

    # --- Crossing columns: zigzag left, right, left ---
    left_cols = list(range(1, W // 2))
    right_cols = list(range(W // 2 + 1, W - 1))
    if not left_cols or not right_cols:
        return None

    cross_a_col = rng.choice(left_cols)    # Barrier 1: left
    cross_b_col = rng.choice(right_cols)   # Barrier 2: right
    cross_c_col = rng.choice(left_cols)    # Barrier 3: left

    cross_a_pos = Pos(b1_row, cross_a_col)
    cross_b_pos = Pos(b2_row, cross_b_col)
    cross_c_pos = Pos(b3_row, cross_c_col)

    # --- Build walls: 3 full barrier rows, each with one tidal gap ---
    walls: set[Pos] = set()
    for c in range(W):
        if c != cross_a_col:
            walls.add(Pos(b1_row, c))
        if c != cross_b_col:
            walls.add(Pos(b2_row, c))
        if c != cross_c_col:
            walls.add(Pos(b3_row, c))

    # --- Section floor tiles ---
    section0 = [Pos(r, c) for r in range(0, b1_row)
                for c in range(W) if Pos(r, c) not in walls]
    section1 = [Pos(r, c) for r in range(b1_row + 1, b2_row)
                for c in range(W) if Pos(r, c) not in walls]
    section3 = [Pos(r, c) for r in range(b3_row + 1, H)
                for c in range(W) if Pos(r, c) not in walls]

    if len(section0) < 2 or len(section1) < 1 or len(section3) < 1:
        return None

    # --- Place start, beacons, exit ---
    start_pos = rng.choice(section0)
    b2_candidates = [p for p in section0 if p != start_pos]
    if not b2_candidates:
        return None
    b2_pos = rng.choice(b2_candidates)     # free beacon — same section as start
    b1_pos = rng.choice(section1)          # bait beacon — across Zone A
    exit_pos = rng.choice(section3)        # exit — past all 3 barriers

    # --- E2/E3/E4 coin flips ---
    enhanced_cycles  = width >= 7 and rng.random() < 0.5  # E2
    enhanced_coprime = width >= 7 and rng.random() < 0.5  # E3
    enhanced_similar = width >= 7 and rng.random() < 0.5  # E4

    # --- Zone A: FLOODED at greedy arrival ---
    greedy_arrival_a = (manhattan_distance(start_pos, b2_pos)
                        + manhattan_distance(b2_pos, cross_a_pos))

    dry_a = rng.randint(2, 3)
    flood_a = rng.randint(3, 5)
    if enhanced_cycles:
        dry_a, flood_a = _scale_cycle_params(dry_a, flood_a, rng)
    cycle_a = dry_a + flood_a
    flood_mid_a = dry_a + flood_a // 2
    offset_a = (flood_mid_a - greedy_arrival_a % cycle_a) % cycle_a

    # Build temporary Zone A to compute optimal arrival
    zone_a_tmp = TidalZone(
        zone_id="A", tiles=frozenset({cross_a_pos}),
        dry_duration=dry_a, flood_duration=flood_a, phase_offset=offset_a,
    )

    # Verify Zone A is actually flooded at greedy arrival
    if zone_a_tmp.is_dry_at(greedy_arrival_a):
        return None

    # --- Optimal arrival at A: first dry time >= min travel ---
    min_travel_a = manhattan_distance(start_pos, cross_a_pos)
    optimal_arrival_a = None
    for t in range(min_travel_a, min_travel_a + cycle_a):
        if zone_a_tmp.is_dry_at(t):
            optimal_arrival_a = t
            break
    if optimal_arrival_a is None:
        return None

    # --- Cascading arrivals (optimal path) ---
    # From Zone A crossing → collect b1 → Zone B crossing → Zone C crossing
    optimal_arrival_b = (optimal_arrival_a
                         + manhattan_distance(cross_a_pos, b1_pos)
                         + manhattan_distance(b1_pos, cross_b_pos))
    optimal_arrival_c = (optimal_arrival_b
                         + manhattan_distance(cross_b_pos, cross_c_pos))

    # --- Zone B: DRY at optimal arrival ---
    dry_b = rng.randint(2, 3)
    flood_b = rng.randint(3, 5)
    if enhanced_similar:
        dry_b, flood_b = dry_a, flood_a
    elif enhanced_cycles:
        dry_b, flood_b = _scale_cycle_params(dry_b, flood_b, rng)
    if enhanced_coprime:
        flood_b = _ensure_coprime(flood_b, dry_b, [cycle_a])
    cycle_b = dry_b + flood_b
    dry_mid_b = dry_b // 2
    offset_b = (dry_mid_b - optimal_arrival_b % cycle_b) % cycle_b

    # --- Zone C: DRY at optimal arrival ---
    dry_c = rng.randint(2, 3)
    flood_c = rng.randint(3, 5)
    if enhanced_similar:
        dry_c, flood_c = dry_a, flood_a
    elif enhanced_cycles:
        dry_c, flood_c = _scale_cycle_params(dry_c, flood_c, rng)
    if enhanced_coprime:
        flood_c = _ensure_coprime(flood_c, dry_c, [cycle_a, cycle_b])
    cycle_c = dry_c + flood_c
    dry_mid_c = dry_c // 2
    offset_c = (dry_mid_c - optimal_arrival_c % cycle_c) % cycle_c

    # --- Cascade verification: greedy must hit floods at B and C ---
    zone_b_tmp = TidalZone(
        zone_id="B", tiles=frozenset({cross_b_pos}),
        dry_duration=dry_b, flood_duration=flood_b, phase_offset=offset_b,
    )
    zone_c_tmp = TidalZone(
        zone_id="C", tiles=frozenset({cross_c_pos}),
        dry_duration=dry_c, flood_duration=flood_c, phase_offset=offset_c,
    )

    # Greedy waits at A until dry, then continues
    greedy_cross_a = greedy_arrival_a
    for _ in range(cycle_a):
        if zone_a_tmp.is_dry_at(greedy_cross_a):
            break
        greedy_cross_a += 1
    else:
        return None

    # Greedy arrival at B after crossing A and collecting b1
    greedy_arrival_b = (greedy_cross_a
                        + manhattan_distance(cross_a_pos, b1_pos)
                        + manhattan_distance(b1_pos, cross_b_pos))
    if zone_b_tmp.is_dry_at(greedy_arrival_b):
        return None  # cascade broken — greedy arrives dry at B

    # Greedy waits at B until dry, then continues to C
    greedy_cross_b = greedy_arrival_b
    for _ in range(cycle_b):
        if zone_b_tmp.is_dry_at(greedy_cross_b):
            break
        greedy_cross_b += 1
    else:
        return None

    greedy_arrival_c = (greedy_cross_b
                        + manhattan_distance(cross_b_pos, cross_c_pos))
    if zone_c_tmp.is_dry_at(greedy_arrival_c):
        return None  # cascade broken — greedy arrives dry at C

    # --- Ensure zones have distinct timing parameters ---
    params = [(cycle_a, offset_a), (cycle_b, offset_b), (cycle_c, offset_c)]
    if len(set(params)) < 2:
        offset_c = (offset_c + 1) % cycle_c

    # --- Build grid ---
    zone_a = TidalZone(
        zone_id="A", tiles=frozenset({cross_a_pos}),
        dry_duration=dry_a, flood_duration=flood_a, phase_offset=offset_a,
    )
    zone_b = TidalZone(
        zone_id="B", tiles=frozenset({cross_b_pos}),
        dry_duration=dry_b, flood_duration=flood_b, phase_offset=offset_b,
    )
    zone_c = TidalZone(
        zone_id="C", tiles=frozenset({cross_c_pos}),
        dry_duration=dry_c, flood_duration=flood_c, phase_offset=offset_c,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=b1_pos),
        "b2": Beacon(beacon_id="b2", pos=b2_pos),
    }

    # Generous oxygen placeholder — recalibrated by validation pipeline
    oxygen = width + height + rng.randint(15, 30)

    grid = TideGridGrid(
        width=width, height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a, "B": zone_b, "C": zone_c},
        beacons=beacons,
        start=start_pos,
        exit_pos=exit_pos,
        initial_oxygen=oxygen,
    )
    return grid, make_initial_state(grid)


def generate_roundtrip_grid(
    width: int, height: int, rng: random.Random,
) -> Optional[tuple[TideGridGrid, TideGridState]]:
    """Generate a Roundtrip instance testing bidirectional timing.

    Layout: single horizontal barrier at H//2 with one tidal crossing.
    Section 0 (above barrier): start, B2 (free beacon), exit.
    Section 1 (below barrier): B1 (must cross to collect).

    Bait mechanism (phase trapping):
    - B1 is closer to start → all greedy strategies cross first
    - Greedy enters at the first dry tick, but the round-trip distance D
      shifts the return phase into a flood window
    - Agent trapped below barrier, WAITs through flood → oxygen depleted
    - Optimal delays crossing (collects B2 first or WAITs) to enter at a
      phase where both outbound and return hit dry windows
    """
    if width < 6 or height < 6:
        return None

    H, W = height, width
    barrier_row = H // 2

    # --- Single crossing column (interior, not edge) ---
    interior_cols = list(range(1, W - 1))
    if not interior_cols:
        return None
    crossing_col = rng.choice(interior_cols)
    crossing_pos = Pos(barrier_row, crossing_col)

    # --- Build walls: full barrier row except crossing ---
    walls: set[Pos] = set()
    for c in range(W):
        if c != crossing_col:
            walls.add(Pos(barrier_row, c))

    # --- Section floor tiles ---
    section0 = [Pos(r, c) for r in range(0, barrier_row)
                for c in range(W) if Pos(r, c) not in walls]
    section1 = [Pos(r, c) for r in range(barrier_row + 1, H)
                for c in range(W) if Pos(r, c) not in walls]

    if len(section0) < 3 or len(section1) < 1:
        return None

    # --- Place start near crossing (bottom of section 0) ---
    # Prefer positions close to the barrier for start
    near_barrier = [p for p in section0 if p.row == barrier_row - 1]
    if not near_barrier:
        near_barrier = [p for p in section0 if p.row >= barrier_row - 2]
    if not near_barrier:
        near_barrier = section0
    start_pos = rng.choice(near_barrier)

    # --- Place B1 in section 1 ---
    b1_pos = rng.choice(section1)

    # --- Place B2 far in section 0 (manhattan(start,B2) > manhattan(start,B1)) ---
    dist_start_b1 = manhattan_distance(start_pos, b1_pos)
    b2_candidates = [p for p in section0
                     if p != start_pos
                     and manhattan_distance(start_pos, p) > dist_start_b1]
    if not b2_candidates:
        # Fallback: just pick something far from start
        b2_candidates = [p for p in section0
                         if p != start_pos
                         and manhattan_distance(start_pos, p) >= 2]
    if not b2_candidates:
        return None
    b2_pos = rng.choice(b2_candidates)

    # --- Place exit in section 0 (not start, not B2) ---
    exit_candidates = [p for p in section0
                       if p != start_pos and p != b2_pos]
    if not exit_candidates:
        return None
    exit_pos = rng.choice(exit_candidates)

    # --- Phase algebra ---
    D = 2 * manhattan_distance(crossing_pos, b1_pos)  # round-trip distance below
    if D < 2:
        return None

    # --- Greedy arrival times ---
    # nearest_beacon greedy: go straight to crossing (B1 is closer)
    t_g = manhattan_distance(start_pos, crossing_pos)
    # farthest_beacon_first greedy: collect B2 first, then crossing
    t_c = (manhattan_distance(start_pos, b2_pos)
           + manhattan_distance(b2_pos, crossing_pos))

    # --- E2/E3/E4 coin flips (E3/E4 no-op for single-zone) ---
    enhanced_cycles  = width >= 7 and rng.random() < 0.5  # E2
    enhanced_coprime = width >= 7 and rng.random() < 0.5  # E3 (no-op)
    enhanced_similar = width >= 7 and rng.random() < 0.5  # E4 (no-op)

    # --- Search over dry/flood combinations for valid phase algebra ---
    # Enumerate all (dry, flood) pairs and pick one that satisfies constraints
    candidates: list[tuple[int, int, int, int]] = []  # (dry, flood, offset, D_mod)
    if enhanced_cycles:
        dry_range = list(range(3, 7))    # E2: scaled from [2,5)
        flood_range = list(range(5, 11)) # E2: scaled from [3,7)
    else:
        dry_range = list(range(2, 5))    # [2, 3, 4]
        flood_range = list(range(3, 7))  # [3, 4, 5, 6]
    rng.shuffle(dry_range)
    rng.shuffle(flood_range)

    for dry in dry_range:
        for flood in flood_range:
            cycle = dry + flood
            D_mod = D % cycle

            # Require D_mod in [flood+1, cycle-1] — ensures phase 0 is bad
            # but good phases exist in the wrap-around range
            if D_mod < flood + 1 or D_mod > cycle - 1:
                continue

            # Verify good phases exist: some p < dry where (p + D_mod) % cycle < dry
            has_good = any((p + D_mod) % cycle < dry for p in range(dry))
            if not has_good:
                continue

            # Search offsets: both t_g and t_c must map to bad/flood phases
            for offset in range(cycle):
                # Check greedy arrival t_g
                phase_g = (t_g + offset) % cycle
                # Greedy waits for first dry tick
                entry_phase_g = phase_g if phase_g < dry else 0

                # Return phase for greedy
                return_phase_g = (entry_phase_g + D_mod) % cycle
                g_trapped = return_phase_g >= dry  # return lands in flood

                # Check cautious greedy arrival t_c
                phase_c = (t_c + offset) % cycle
                entry_phase_c = phase_c if phase_c < dry else 0

                return_phase_c = (entry_phase_c + D_mod) % cycle
                c_trapped = return_phase_c >= dry

                if g_trapped and c_trapped:
                    candidates.append((dry, flood, offset, D_mod))
                    break

    if not candidates:
        return None

    dry, flood, offset, D_mod = candidates[0]
    cycle = dry + flood

    # --- Verify the oracle can actually reach a good entry phase ---
    # The oracle needs to arrive at the crossing at a time t where
    # (t + offset) % cycle is a good phase (both entry and return dry)
    min_travel = manhattan_distance(start_pos, crossing_pos)
    min_travel_via_b2 = (manhattan_distance(start_pos, b2_pos)
                         + manhattan_distance(b2_pos, crossing_pos))
    oracle_can_enter = False
    for base_t in [min_travel, min_travel_via_b2]:
        for wait in range(cycle):
            t = base_t + wait
            phase = (t + offset) % cycle
            if phase < dry and (phase + D_mod) % cycle < dry:
                oracle_can_enter = True
                break
        if oracle_can_enter:
            break
    if not oracle_can_enter:
        return None

    # --- Build zone and grid ---
    zone_a = TidalZone(
        zone_id="A", tiles=frozenset({crossing_pos}),
        dry_duration=dry, flood_duration=flood, phase_offset=offset,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=b1_pos),
        "b2": Beacon(beacon_id="b2", pos=b2_pos),
    }

    oxygen = width + height + rng.randint(10, 20)

    grid = TideGridGrid(
        width=width, height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a},
        beacons=beacons,
        start=start_pos,
        exit_pos=exit_pos,
        initial_oxygen=oxygen,
    )
    return grid, make_initial_state(grid)


def generate_confluence_grid(
    width: int, height: int, rng: random.Random,
) -> Optional[tuple[TideGridGrid, TideGridState]]:
    """Generate a Confluence instance testing schedule intersection.

    Layout:
      Top section (rows 0..br-1):      Start, Exit (free movement)
      Barrier row (br):                 WALL..tA..WALL..DIV_WALL..WALL..tB..WALL
      Left pocket (rows br+1..H-2):    B1, walled at div_col
      Right pocket (rows br+1..H-2):   B2, walled at div_col
      Connecting passage (row H-1):     No divider — links pockets at bottom

    Agent path (optimal):
      Start (top) → cross Zone A down → left pocket → collect B1
      B1 → through connecting passage (row H-1) → right pocket → collect B2
      B2 → cross Zone B up → top section → exit

    Bait mechanism (schedule mismatch):
      - B1 closer to start → nearest-first greedy crosses Zone A first
      - Greedy enters Zone A at first available dry tick
      - After traversing below (D steps), arrives at Zone B during flood window
      - Farthest-first greedy crosses Zone B first → arrives at Zone A during flood
      - Both orderings trapped by the same schedule mismatch
      - Optimal delays entry to find a time where both crossings align
    """
    if width < 7 or height < 6:
        return None

    H, W = height, width

    # --- Barrier row and divider column ---
    barrier_row = max(2, min(H // 3, H - 4))
    div_col = W // 2

    # --- Crossing columns: one in each half ---
    left_cols = list(range(1, div_col))
    right_cols = list(range(div_col + 1, W - 1))
    if not left_cols or not right_cols:
        return None

    crossing_a_col = rng.choice(left_cols)
    crossing_b_col = rng.choice(right_cols)
    crossing_a_pos = Pos(barrier_row, crossing_a_col)
    crossing_b_pos = Pos(barrier_row, crossing_b_col)

    # --- Build walls ---
    walls: set[Pos] = set()
    # Barrier row: all cols except crossings
    for c in range(W):
        if c != crossing_a_col and c != crossing_b_col:
            walls.add(Pos(barrier_row, c))
    # Divider column: rows br+1 to H-2 (not H-1, that's the passage)
    for r in range(barrier_row + 1, H - 1):
        walls.add(Pos(r, div_col))

    # --- Floor tiles ---
    top_section = [Pos(r, c) for r in range(0, barrier_row)
                   for c in range(W) if Pos(r, c) not in walls]
    left_pocket = [Pos(r, c) for r in range(barrier_row + 1, H)
                   for c in range(0, div_col) if Pos(r, c) not in walls]
    right_pocket = [Pos(r, c) for r in range(barrier_row + 1, H)
                    for c in range(div_col + 1, W) if Pos(r, c) not in walls]
    # Passage row tiles (row H-1, full width)
    passage_tiles = [Pos(H - 1, c) for c in range(W) if Pos(H - 1, c) not in walls]

    if len(top_section) < 3 or len(left_pocket) < 1 or len(right_pocket) < 1:
        return None
    if not passage_tiles:
        return None

    # --- Place start in top section, left side (near Zone A) ---
    start_candidates = [p for p in top_section if p.col <= div_col]
    if not start_candidates:
        start_candidates = top_section
    start_pos = rng.choice(start_candidates)

    # --- Place B1 in left pocket, B2 in right pocket ---
    b1_pos = rng.choice(left_pocket)
    b2_pos = rng.choice(right_pocket)

    # Require manhattan(start, B1) < manhattan(start, B2) for greedy ordering
    if manhattan_distance(start_pos, b1_pos) >= manhattan_distance(start_pos, b2_pos):
        return None

    # --- Place exit in top section (not start) ---
    exit_candidates = [p for p in top_section
                       if p != start_pos and p != b1_pos and p != b2_pos]
    if not exit_candidates:
        return None
    exit_pos = rng.choice(exit_candidates)

    # --- Compute D: total steps below barrier (Zone A entry → B1 → passage → B2 → Zone B exit) ---
    a_below = Pos(barrier_row + 1, crossing_a_col)
    b_below = Pos(barrier_row + 1, crossing_b_col)

    d_to_b1 = manhattan_distance(a_below, b1_pos)
    # B1 → passage (bottom) → B2: go down to row H-1, across, back up
    d_b1_to_b2 = ((H - 1 - b1_pos.row)
                   + abs(b2_pos.col - b1_pos.col)
                   + (H - 1 - b2_pos.row))
    d_b2_to_b_below = manhattan_distance(b2_pos, b_below)

    D_below = d_to_b1 + d_b1_to_b2 + d_b2_to_b_below
    D = D_below + 2  # +1 for stepping onto Zone A tile, +1 for stepping onto Zone B tile

    if D < 4:
        return None

    # --- Greedy arrival times ---
    # A-first greedy: walks to just above Zone A, then crosses
    t_walk_a = manhattan_distance(start_pos, Pos(barrier_row - 1, crossing_a_col))
    # B-first greedy: walks to just above Zone B, then crosses
    t_walk_b = manhattan_distance(start_pos, Pos(barrier_row - 1, crossing_b_col))

    # --- E2/E3/E4 coin flips ---
    enhanced_cycles  = width >= 7 and rng.random() < 0.5  # E2
    enhanced_coprime = width >= 7 and rng.random() < 0.5  # E3
    enhanced_similar = width >= 7 and rng.random() < 0.5  # E4

    # --- Phase algebra: brute-force search ---
    if enhanced_cycles:
        dry_range = list(range(3, 7))     # E2: scaled from [2,5)
        flood_range = list(range(5, 11))  # E2: scaled from [3,7)
    else:
        dry_range = list(range(2, 5))     # [2, 3, 4]
        flood_range = list(range(3, 7))   # [3, 4, 5, 6]
    rng.shuffle(dry_range)
    rng.shuffle(flood_range)

    best_params = None

    for dry_a in dry_range:
        if best_params:
            break
        for flood_a in flood_range:
            if best_params:
                break
            cycle_a = dry_a + flood_a
            for dry_b in dry_range:
                if best_params:
                    break
                for flood_b in flood_range:
                    if best_params:
                        break
                    cycle_b = dry_b + flood_b

                    # E4: skip if params don't match zone A
                    if enhanced_similar and (dry_b != dry_a or flood_b != flood_a):
                        continue
                    # E3: skip if cycles not coprime
                    if enhanced_coprime and gcd(cycle_a, cycle_b) > 1:
                        continue

                    for offset_a in range(cycle_a):
                        if best_params:
                            break
                        for offset_b in range(cycle_b):
                            # --- A-first greedy crossing time ---
                            # Greedy walks to above Zone A, then enters at first dry tick
                            t_cross_a_greedy = None
                            for t in range(t_walk_a + 1, t_walk_a + 1 + cycle_a):
                                phase = (t + offset_a) % cycle_a
                                if phase < dry_a:
                                    t_cross_a_greedy = t
                                    break
                            if t_cross_a_greedy is None:
                                continue

                            # Check Zone B flooded at t_cross_a_greedy + D
                            t_arrive_b = t_cross_a_greedy + D
                            phase_b = (t_arrive_b + offset_b) % cycle_b
                            if phase_b < dry_b:
                                continue  # Zone B is dry — greedy not trapped

                            # --- B-first greedy crossing time ---
                            t_cross_b_greedy = None
                            for t in range(t_walk_b + 1, t_walk_b + 1 + cycle_b):
                                phase = (t + offset_b) % cycle_b
                                if phase < dry_b:
                                    t_cross_b_greedy = t
                                    break
                            if t_cross_b_greedy is None:
                                continue

                            # Check Zone A flooded at t_cross_b_greedy + D
                            t_arrive_a = t_cross_b_greedy + D
                            phase_a = (t_arrive_a + offset_a) % cycle_a
                            if phase_a < dry_a:
                                continue  # Zone A is dry — greedy not trapped

                            # --- Optimal exists: some arrival time where A-first works ---
                            # Check times from min possible to reasonable range
                            min_t = manhattan_distance(start_pos, crossing_a_pos)
                            lcm_cycles = cycle_a * cycle_b  # upper bound on period
                            optimal_exists = False
                            for t in range(min_t, min_t + lcm_cycles + cycle_a):
                                pa = (t + offset_a) % cycle_a
                                if pa >= dry_a:
                                    continue  # Zone A flooded at entry
                                pb = (t + D + offset_b) % cycle_b
                                if pb >= dry_b:
                                    continue  # Zone B flooded at arrival
                                optimal_exists = True
                                break

                            if not optimal_exists:
                                # Also check B-first optimal
                                min_t_b = manhattan_distance(start_pos, crossing_b_pos)
                                for t in range(min_t_b, min_t_b + lcm_cycles + cycle_b):
                                    pb = (t + offset_b) % cycle_b
                                    if pb >= dry_b:
                                        continue
                                    pa = (t + D + offset_a) % cycle_a
                                    if pa >= dry_a:
                                        continue
                                    optimal_exists = True
                                    break

                            if optimal_exists:
                                best_params = (dry_a, flood_a, offset_a,
                                               dry_b, flood_b, offset_b)
                                break

    if best_params is None:
        return None

    dry_a, flood_a, offset_a, dry_b, flood_b, offset_b = best_params

    # --- Build zones and grid ---
    zone_a = TidalZone(
        zone_id="A", tiles=frozenset({crossing_a_pos}),
        dry_duration=dry_a, flood_duration=flood_a, phase_offset=offset_a,
    )
    zone_b = TidalZone(
        zone_id="B", tiles=frozenset({crossing_b_pos}),
        dry_duration=dry_b, flood_duration=flood_b, phase_offset=offset_b,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=b1_pos),
        "b2": Beacon(beacon_id="b2", pos=b2_pos),
    }

    oxygen = width + height + rng.randint(10, 20)

    grid = TideGridGrid(
        width=width, height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a, "B": zone_b},
        beacons=beacons,
        start=start_pos,
        exit_pos=exit_pos,
        initial_oxygen=oxygen,
    )
    return grid, make_initial_state(grid)


def generate_patience_grid(
    width: int, height: int, rng: random.Random,
) -> Optional[tuple[TideGridGrid, TideGridState]]:
    """Generate a Patience instance testing WAIT-optimality.

    Layout: single horizontal barrier at H//2 with one tidal crossing.
    Section 0 (above barrier): start, B2 (free beacon), exit.
    Section 1 (below barrier): B1 (must cross to collect).

    Bait mechanism (WAIT-optimality):
    - Zone A has very short dry window (dry=1-2, flood=4-7)
    - Immediate movement arrives at crossing during flood
    - All greedy strategies move immediately → arrive flooded → WAIT at crossing
    - Optimal: WAIT at start, then walk → arrive exactly at dry tick
    - The optimal first action is WAIT (deeply counterintuitive)
    """
    if width < 6 or height < 6:
        return None

    H, W = height, width
    barrier_row = H // 2

    # --- Single crossing column (interior, not edge) ---
    interior_cols = list(range(1, W - 1))
    if not interior_cols:
        return None
    crossing_col = rng.choice(interior_cols)
    crossing_pos = Pos(barrier_row, crossing_col)

    # --- Build walls: full barrier row except crossing ---
    walls: set[Pos] = set()
    for c in range(W):
        if c != crossing_col:
            walls.add(Pos(barrier_row, c))

    # --- Section floor tiles ---
    section0 = [Pos(r, c) for r in range(0, barrier_row)
                for c in range(W) if Pos(r, c) not in walls]
    section1 = [Pos(r, c) for r in range(barrier_row + 1, H)
                for c in range(W) if Pos(r, c) not in walls]

    if len(section0) < 3 or len(section1) < 1:
        return None

    # --- Place start near barrier (bottom of section 0) ---
    near_barrier = [p for p in section0 if p.row == barrier_row - 1]
    if not near_barrier:
        near_barrier = [p for p in section0 if p.row >= barrier_row - 2]
    if not near_barrier:
        near_barrier = section0
    start_pos = rng.choice(near_barrier)

    # --- Place B1 in section 1 ---
    b1_pos = rng.choice(section1)

    # --- Place B2 in section 0, farther from start than B1 ---
    dist_start_b1 = manhattan_distance(start_pos, b1_pos)
    b2_candidates = [p for p in section0
                     if p != start_pos
                     and manhattan_distance(start_pos, p) > dist_start_b1]
    if not b2_candidates:
        b2_candidates = [p for p in section0
                         if p != start_pos
                         and manhattan_distance(start_pos, p) >= 2]
    if not b2_candidates:
        return None
    b2_pos = rng.choice(b2_candidates)

    # --- Place exit in section 0 (not start, not B2) ---
    exit_candidates = [p for p in section0
                       if p != start_pos and p != b2_pos]
    if not exit_candidates:
        return None
    exit_pos = rng.choice(exit_candidates)

    # --- Phase algebra for WAIT-optimality ---
    d = manhattan_distance(start_pos, crossing_pos)  # walk distance to crossing
    D = 2 * manhattan_distance(crossing_pos, b1_pos)  # round-trip distance below
    if D < 2:
        return None

    # Greedy arrival times
    t_g = d  # nearest-beacon greedy goes straight to crossing
    t_c = (manhattan_distance(start_pos, b2_pos)
           + manhattan_distance(b2_pos, crossing_pos))  # farthest-first via B2

    # --- E2/E3/E4 coin flips (E3/E4 no-op for single-zone) ---
    enhanced_cycles  = width >= 7 and rng.random() < 0.5  # E2
    enhanced_coprime = width >= 7 and rng.random() < 0.5  # E3 (no-op)
    enhanced_similar = width >= 7 and rng.random() < 0.5  # E4 (no-op)

    # Search over tight dry windows (1-2) with long floods (4-7)
    if enhanced_cycles:
        dry_range = [2, 3]                  # E2: scaled from [1,2]
        flood_range = list(range(6, 12))    # E2: scaled from [4,8)
    else:
        dry_range = [1, 2]
        flood_range = list(range(4, 8))     # [4, 5, 6, 7]
    rng.shuffle(dry_range)
    rng.shuffle(flood_range)

    best_params = None

    for dry in dry_range:
        if best_params:
            break
        for flood in flood_range:
            if best_params:
                break
            cycle = dry + flood
            D_mod = D % cycle

            # Roundtrip constraint: good phases must exist for both entry and return
            has_good = any(
                (p + D_mod) % cycle < dry for p in range(dry)
            )
            if not has_good:
                continue

            for offset in range(cycle):
                # 1. Greedy nearest (t_g) must land in flood
                phase_g = (t_g + offset) % cycle
                if phase_g < dry:
                    continue  # greedy arrives dry — not trapped

                # 2. Greedy farthest (t_c) must land in flood
                phase_c = (t_c + offset) % cycle
                if phase_c < dry:
                    continue

                # 3. Oracle can reach a good entry phase via waiting
                oracle_ok = False
                # Check direct path with wait
                for w in range(cycle):
                    entry_phase = (d + w + offset) % cycle
                    if entry_phase < dry and (entry_phase + D_mod) % cycle < dry:
                        oracle_ok = True
                        break
                # Also check via B2-first path
                if not oracle_ok:
                    for w in range(cycle):
                        entry_phase = (t_c + w + offset) % cycle
                        if entry_phase < dry and (entry_phase + D_mod) % cycle < dry:
                            oracle_ok = True
                            break

                if oracle_ok:
                    best_params = (dry, flood, offset)
                    break

    if best_params is None:
        return None

    dry, flood, offset = best_params

    # --- Build zone and grid ---
    zone_a = TidalZone(
        zone_id="A", tiles=frozenset({crossing_pos}),
        dry_duration=dry, flood_duration=flood, phase_offset=offset,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=b1_pos),
        "b2": Beacon(beacon_id="b2", pos=b2_pos),
    }

    oxygen = width + height + rng.randint(10, 20)

    grid = TideGridGrid(
        width=width, height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a},
        beacons=beacons,
        start=start_pos,
        exit_pos=exit_pos,
        initial_oxygen=oxygen,
    )
    return grid, make_initial_state(grid)


def generate_switchback_grid(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[TideGridGrid, TideGridState]]:
    """Generate a Switchback instance with multi-gate interference.

    Layout: 3 horizontal barrier walls at H//4, H//2, 3H//4 with single-tile
    tidal crossings that zigzag right-left-right (opposite of gauntlet's
    left-right-left).

    Bait mechanism (cascading timing with reversed zigzag):
    - Beacon b2 in section 0 (free, same section as start)
    - Beacon b1 in section 1 (bait, across Zone A)
    - Exit in section 3 (below all 3 barriers)
    - Greedy collects b2 first (nearest) → arrives at Zone A during flood phase
    - Wait cascade: delay at A shifts arrival at B and C into flood phases too
    - Optimal: brief wait (0-2 turns) syncs Zone A dry window → clean cascade
    - Zigzag right-left-right creates different inter-gate distances than
      gauntlet's left-right-left, producing distinct interference patterns
    """
    if width < 6 or height < 7:
        return None

    H, W = height, width

    # --- Barrier rows ---
    b1_row = H // 4
    b2_row = H // 2
    b3_row = 3 * H // 4

    # Barrier rows must be distinct and leave room for sections
    if b1_row < 1 or b2_row - b1_row < 2 or b3_row - b2_row < 2 or H - 1 - b3_row < 1:
        return None

    # --- Crossing columns: zigzag right, left, right (opposite of gauntlet) ---
    left_cols = list(range(1, W // 2))
    right_cols = list(range(W // 2 + 1, W - 1))
    if not left_cols or not right_cols:
        return None

    cross_a_col = rng.choice(right_cols)   # Barrier 1: right
    cross_b_col = rng.choice(left_cols)    # Barrier 2: left
    cross_c_col = rng.choice(right_cols)   # Barrier 3: right

    cross_a_pos = Pos(b1_row, cross_a_col)
    cross_b_pos = Pos(b2_row, cross_b_col)
    cross_c_pos = Pos(b3_row, cross_c_col)

    # --- Build walls: 3 full barrier rows, each with one tidal gap ---
    walls: set[Pos] = set()
    for c in range(W):
        if c != cross_a_col:
            walls.add(Pos(b1_row, c))
        if c != cross_b_col:
            walls.add(Pos(b2_row, c))
        if c != cross_c_col:
            walls.add(Pos(b3_row, c))

    # --- Section floor tiles ---
    section0 = [Pos(r, c) for r in range(0, b1_row)
                for c in range(W) if Pos(r, c) not in walls]
    section1 = [Pos(r, c) for r in range(b1_row + 1, b2_row)
                for c in range(W) if Pos(r, c) not in walls]
    section3 = [Pos(r, c) for r in range(b3_row + 1, H)
                for c in range(W) if Pos(r, c) not in walls]

    if len(section0) < 2 or len(section1) < 1 or len(section3) < 1:
        return None

    # --- Place start, beacons, exit ---
    start_pos = rng.choice(section0)
    b2_candidates = [p for p in section0 if p != start_pos]
    if not b2_candidates:
        return None
    b2_pos = rng.choice(b2_candidates)     # free beacon — same section as start
    b1_pos = rng.choice(section1)          # bait beacon — across Zone A
    exit_pos = rng.choice(section3)        # exit — past all 3 barriers

    # --- E2/E3/E4 coin flips ---
    enhanced_cycles  = width >= 7 and rng.random() < 0.5  # E2
    enhanced_coprime = width >= 7 and rng.random() < 0.5  # E3
    enhanced_similar = width >= 7 and rng.random() < 0.5  # E4

    # --- Zone A: FLOODED at greedy arrival ---
    greedy_arrival_a = (manhattan_distance(start_pos, b2_pos)
                        + manhattan_distance(b2_pos, cross_a_pos))

    dry_a = rng.randint(2, 3)
    flood_a = rng.randint(3, 5)
    if enhanced_cycles:
        dry_a, flood_a = _scale_cycle_params(dry_a, flood_a, rng)
    cycle_a = dry_a + flood_a
    flood_mid_a = dry_a + flood_a // 2
    offset_a = (flood_mid_a - greedy_arrival_a % cycle_a) % cycle_a

    # Build temporary Zone A to compute optimal arrival
    zone_a_tmp = TidalZone(
        zone_id="A", tiles=frozenset({cross_a_pos}),
        dry_duration=dry_a, flood_duration=flood_a, phase_offset=offset_a,
    )

    # Verify Zone A is actually flooded at greedy arrival
    if zone_a_tmp.is_dry_at(greedy_arrival_a):
        return None

    # --- Optimal arrival at A: first dry time >= min travel ---
    min_travel_a = manhattan_distance(start_pos, cross_a_pos)
    optimal_arrival_a = None
    for t in range(min_travel_a, min_travel_a + cycle_a):
        if zone_a_tmp.is_dry_at(t):
            optimal_arrival_a = t
            break
    if optimal_arrival_a is None:
        return None

    # --- Cascading arrivals (optimal path) ---
    # From Zone A crossing → collect b1 → Zone B crossing → Zone C crossing
    optimal_arrival_b = (optimal_arrival_a
                         + manhattan_distance(cross_a_pos, b1_pos)
                         + manhattan_distance(b1_pos, cross_b_pos))
    optimal_arrival_c = (optimal_arrival_b
                         + manhattan_distance(cross_b_pos, cross_c_pos))

    # --- Zone B: DRY at optimal arrival ---
    dry_b = rng.randint(2, 3)
    flood_b = rng.randint(3, 5)
    if enhanced_similar:
        dry_b, flood_b = dry_a, flood_a
    elif enhanced_cycles:
        dry_b, flood_b = _scale_cycle_params(dry_b, flood_b, rng)
    if enhanced_coprime:
        flood_b = _ensure_coprime(flood_b, dry_b, [cycle_a])
    cycle_b = dry_b + flood_b
    dry_mid_b = dry_b // 2
    offset_b = (dry_mid_b - optimal_arrival_b % cycle_b) % cycle_b

    # --- Zone C: DRY at optimal arrival ---
    dry_c = rng.randint(2, 3)
    flood_c = rng.randint(3, 5)
    if enhanced_similar:
        dry_c, flood_c = dry_a, flood_a
    elif enhanced_cycles:
        dry_c, flood_c = _scale_cycle_params(dry_c, flood_c, rng)
    if enhanced_coprime:
        flood_c = _ensure_coprime(flood_c, dry_c, [cycle_a, cycle_b])
    cycle_c = dry_c + flood_c
    dry_mid_c = dry_c // 2
    offset_c = (dry_mid_c - optimal_arrival_c % cycle_c) % cycle_c

    # --- Cascade verification: greedy must hit floods at B and C ---
    zone_b_tmp = TidalZone(
        zone_id="B", tiles=frozenset({cross_b_pos}),
        dry_duration=dry_b, flood_duration=flood_b, phase_offset=offset_b,
    )
    zone_c_tmp = TidalZone(
        zone_id="C", tiles=frozenset({cross_c_pos}),
        dry_duration=dry_c, flood_duration=flood_c, phase_offset=offset_c,
    )

    # Greedy waits at A until dry, then continues
    greedy_cross_a = greedy_arrival_a
    for _ in range(cycle_a):
        if zone_a_tmp.is_dry_at(greedy_cross_a):
            break
        greedy_cross_a += 1
    else:
        return None

    # Greedy arrival at B after crossing A and collecting b1
    greedy_arrival_b = (greedy_cross_a
                        + manhattan_distance(cross_a_pos, b1_pos)
                        + manhattan_distance(b1_pos, cross_b_pos))
    if zone_b_tmp.is_dry_at(greedy_arrival_b):
        return None  # cascade broken — greedy arrives dry at B

    # Greedy waits at B until dry, then continues to C
    greedy_cross_b = greedy_arrival_b
    for _ in range(cycle_b):
        if zone_b_tmp.is_dry_at(greedy_cross_b):
            break
        greedy_cross_b += 1
    else:
        return None

    greedy_arrival_c = (greedy_cross_b
                        + manhattan_distance(cross_b_pos, cross_c_pos))
    if zone_c_tmp.is_dry_at(greedy_arrival_c):
        return None  # cascade broken — greedy arrives dry at C

    # --- Ensure zones have distinct timing parameters ---
    params = [(cycle_a, offset_a), (cycle_b, offset_b), (cycle_c, offset_c)]
    if len(set(params)) < 2:
        offset_c = (offset_c + 1) % cycle_c

    # --- Build grid ---
    zone_a = TidalZone(
        zone_id="A", tiles=frozenset({cross_a_pos}),
        dry_duration=dry_a, flood_duration=flood_a, phase_offset=offset_a,
    )
    zone_b = TidalZone(
        zone_id="B", tiles=frozenset({cross_b_pos}),
        dry_duration=dry_b, flood_duration=flood_b, phase_offset=offset_b,
    )
    zone_c = TidalZone(
        zone_id="C", tiles=frozenset({cross_c_pos}),
        dry_duration=dry_c, flood_duration=flood_c, phase_offset=offset_c,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=b1_pos),
        "b2": Beacon(beacon_id="b2", pos=b2_pos),
    }

    # Generous oxygen placeholder — recalibrated by validation pipeline
    oxygen = width + height + rng.randint(15, 30)

    grid = TideGridGrid(
        width=width, height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a, "B": zone_b, "C": zone_c},
        beacons=beacons,
        start=start_pos,
        exit_pos=exit_pos,
        initial_oxygen=oxygen,
    )
    return grid, make_initial_state(grid)


def generate_tidal_bridge_grid(
    width: int, height: int, rng: random.Random,
) -> Optional[tuple[TideGridGrid, TideGridState]]:
    """Generate a Tidal Bridge instance testing duration-aware crossing.

    Layout: a 3-row-thick horizontal barrier centered in the grid. Each barrier
    row is solid wall except one column (the bridge column). The 3 bridge tiles
    form a vertical passage — the only way from section 0 to section 1. All 3
    bridge tiles share one zone (Zone A), flooding and drying together.

    Bait mechanism (duration-aware crossing):
    - Zone A: dry_duration=4, flood_duration in [3,5]. Crossing requires 3
      consecutive dry ticks (entering bridge tiles at phases p, p+1, p+2, all <4).
    - Safe entry: phases 0-1 (enough dry ticks to complete crossing).
    - Dangerous entry: phase 2 — agent enters tile 1 (phase 2, dry), moves to
      tile 2 (phase 3, dry), tries tile 3 (phase 4, FLOOD) — stuck on tile 2
      with no legal actions (walls left/right, tiles 1&3 flooded). Terminates.
    - Greedy collects B2 (free, section 0) first then heads to bridge, arriving
      at phase 2 due to the detour. Oracle WAITs for phase 0, crosses safely.
    """
    if width < 5 or height < 7:
        return None

    H, W = height, width

    # --- Barrier placement: 3-row thick barrier centered in the grid ---
    barrier_start = (H - 3) // 2
    # Ensure at least 2 rows per section
    if barrier_start < 2 or (barrier_start + 3) > H - 2:
        return None

    # --- Bridge column: interior column ---
    bridge_col = rng.randint(1, W - 2)

    # --- Build walls: 3 barrier rows, all wall except bridge column ---
    walls: set[Pos] = set()
    for dr in range(3):
        row = barrier_start + dr
        for c in range(W):
            if c != bridge_col:
                walls.add(Pos(row, c))

    # --- Bridge tiles: 3-tile vertical passage, single zone ---
    bridge_tiles = frozenset({
        Pos(barrier_start, bridge_col),
        Pos(barrier_start + 1, bridge_col),
        Pos(barrier_start + 2, bridge_col),
    })

    # --- Section floor tiles ---
    section0 = [Pos(r, c) for r in range(0, barrier_start)
                for c in range(W) if Pos(r, c) not in walls]
    section1 = [Pos(r, c) for r in range(barrier_start + 3, H)
                for c in range(W) if Pos(r, c) not in walls]

    if len(section0) < 2 or len(section1) < 2:
        return None

    # --- Entry/exit positions adjacent to bridge ---
    bridge_entry_top = Pos(barrier_start - 1, bridge_col)
    bridge_exit_bottom = Pos(barrier_start + 3, bridge_col)

    if bridge_entry_top not in set(section0) or bridge_exit_bottom not in set(section1):
        return None

    # --- Place start + B2 in section 0, B1 + exit in section 1 ---
    start_pos = rng.choice(section0)
    b2_candidates = [p for p in section0 if p != start_pos]
    if not b2_candidates:
        return None
    b2_pos = rng.choice(b2_candidates)

    b1_pos = rng.choice(section1)
    exit_candidates = [p for p in section1 if p != b1_pos]
    if not exit_candidates:
        return None
    exit_pos = rng.choice(exit_candidates)

    # --- Phase algebra ---
    # Greedy collects B2 first (free, same section), then heads to bridge entry
    greedy_arrival = (manhattan_distance(start_pos, b2_pos)
                      + manhattan_distance(b2_pos, bridge_entry_top))
    # Greedy entry time = stepping onto bridge tile 1
    greedy_entry_time = greedy_arrival + 1

    # --- E2/E3/E4 coin flips (E3/E4 no-op for single-zone) ---
    enhanced_cycles  = width >= 7 and rng.random() < 0.5  # E2
    enhanced_coprime = width >= 7 and rng.random() < 0.5  # E3 (no-op)
    enhanced_similar = width >= 7 and rng.random() < 0.5  # E4 (no-op)

    dry_a = 4
    flood_a = rng.randint(3, 5)
    if enhanced_cycles:
        dry_a, flood_a = _scale_cycle_params(4, flood_a, rng)
        dry_a = max(3, dry_a)  # ensure dry_a >= 3 for 3-tick crossing
    cycle_a = dry_a + flood_a

    # Target phase: enter at dry_a - 2 so 3rd tick hits flood
    target_phase = dry_a - 2
    offset_a = (target_phase - greedy_entry_time % cycle_a) % cycle_a

    # --- Build temp zone and verify ---
    zone_a_tmp = TidalZone(
        zone_id="A", tiles=bridge_tiles,
        dry_duration=dry_a, flood_duration=flood_a, phase_offset=offset_a,
    )

    # Verify: greedy enters dry (phase 2 < 4)
    if not zone_a_tmp.is_dry_at(greedy_entry_time):
        return None
    # Verify: phase 2 steps later is flood (phase 4 >= 4, stuck on tile 2)
    if zone_a_tmp.is_dry_at(greedy_entry_time + 2):
        return None

    # --- Verify optimal can cross: 3 consecutive dry ticks exist ---
    # Scan from earliest possible arrival for a window of 3 consecutive dry ticks
    min_travel = manhattan_distance(start_pos, bridge_entry_top) + 1
    optimal_entry = None
    for t in range(min_travel, min_travel + cycle_a + 1):
        if (zone_a_tmp.is_dry_at(t)
                and zone_a_tmp.is_dry_at(t + 1)
                and zone_a_tmp.is_dry_at(t + 2)):
            optimal_entry = t
            break
    if optimal_entry is None:
        return None

    # --- Build grid ---
    zone_a = TidalZone(
        zone_id="A", tiles=bridge_tiles,
        dry_duration=dry_a, flood_duration=flood_a, phase_offset=offset_a,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=b1_pos),
        "b2": Beacon(beacon_id="b2", pos=b2_pos),
    }

    # Generous oxygen placeholder — recalibrated by validation pipeline
    oxygen = width + height + rng.randint(15, 30)

    grid = TideGridGrid(
        width=width, height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a},
        beacons=beacons,
        start=start_pos,
        exit_pos=exit_pos,
        initial_oxygen=oxygen,
    )
    return grid, make_initial_state(grid)


def generate_deadline_grid(
    width: int, height: int, rng: random.Random,
) -> Optional[tuple[TideGridGrid, TideGridState]]:
    """Generate a Deadline instance testing urgency scheduling (EDF).

    Layout: single barrier row with 3 tidal crossings. Bottom section is walled
    everywhere EXCEPT the 3 crossing columns, creating 3 isolated vertical
    corridors (pockets). Each pocket contains exactly 1 beacon. The ONLY way to
    reach a beacon is to cross its zone from the top section.

    Bait mechanism (EDF scheduling):
    - Zone A (nearest beacon): very relaxed dry window (always accessible)
    - Zone B (middle beacon): narrow dry window, opens at EDF arrival time
    - Zone C (farthest beacon): narrow dry window, opens at EDF arrival time
    - Greedy visits nearest (A) first → burns time → arrives at B/C during flood
    - Optimal visits farthest (C) first (tightest deadline) → EDF ordering
    """
    if width < 7 or height < 7:
        return None

    H, W = height, width

    # --- Barrier row ---
    barrier_row = max(2, H // 3)
    top_rows = barrier_row       # rows 0..br-1
    bottom_rows = H - barrier_row - 1  # rows br+1..H-1
    if top_rows < 2 or bottom_rows < 2:
        return None

    # --- 3 crossing columns evenly spaced across width (interior) ---
    # Need at least 3 interior columns (1..W-2) with spacing >= 2
    interior = list(range(1, W - 1))
    if len(interior) < 5:
        return None  # can't fit 3 columns with spacing
    step = len(interior) / 3
    c_a = interior[int(step * 0.5)]
    c_b = interior[int(step * 1.5)]
    c_c = interior[int(step * 2.5)]
    crossing_cols = [c_a, c_b, c_c]

    # Ensure columns are distinct and at least 2 apart
    if c_a == c_b or c_b == c_c or c_a == c_c:
        return None
    if abs(c_a - c_b) < 2 or abs(c_b - c_c) < 2:
        return None

    # --- Build walls ---
    walls: set[Pos] = set()
    crossing_set = set(crossing_cols)

    # Barrier row: all wall except 3 crossing columns
    for c in range(W):
        if c not in crossing_set:
            walls.add(Pos(barrier_row, c))

    # Bottom section: all wall except 3 corridor columns
    for r in range(barrier_row + 1, H):
        for c in range(W):
            if c not in crossing_set:
                walls.add(Pos(r, c))

    # --- Crossing tiles (one per zone) ---
    cross_a_pos = Pos(barrier_row, c_a)
    cross_b_pos = Pos(barrier_row, c_b)
    cross_c_pos = Pos(barrier_row, c_c)

    # --- Enumerate pocket tiles and top section tiles ---
    top_tiles = [Pos(r, c) for r in range(barrier_row) for c in range(W)
                 if Pos(r, c) not in walls]
    pocket_a = [Pos(r, c_a) for r in range(barrier_row + 1, H)]
    pocket_b = [Pos(r, c_b) for r in range(barrier_row + 1, H)]
    pocket_c = [Pos(r, c_c) for r in range(barrier_row + 1, H)]

    if not pocket_a or not pocket_b or not pocket_c or len(top_tiles) < 3:
        return None

    # --- Place 1 beacon per pocket ---
    b1_pos = rng.choice(pocket_a)
    b2_pos = rng.choice(pocket_b)
    b3_pos = rng.choice(pocket_c)

    # --- Place start + exit in top section ---
    start_pos = rng.choice(top_tiles)
    exit_candidates = [p for p in top_tiles if p != start_pos
                       and manhattan_distance(p, start_pos) >= 2]
    if not exit_candidates:
        return None
    exit_pos = rng.choice(exit_candidates)

    # --- Sort beacons by manhattan distance from start → near/mid/far ---
    beacon_info = [
        (manhattan_distance(start_pos, b1_pos), "b1", b1_pos, c_a, cross_a_pos, pocket_a),
        (manhattan_distance(start_pos, b2_pos), "b2", b2_pos, c_b, cross_b_pos, pocket_b),
        (manhattan_distance(start_pos, b3_pos), "b3", b3_pos, c_c, cross_c_pos, pocket_c),
    ]
    beacon_info.sort(key=lambda x: x[0])

    _, near_id, near_pos, c_near, cross_near, pocket_near = beacon_info[0]
    _, mid_id, mid_pos, c_mid, cross_mid, pocket_mid = beacon_info[1]
    _, far_id, far_pos, c_far, cross_far, pocket_far = beacon_info[2]

    # --- Roundtrip time for each pocket ---
    def rt(cross_col: int, beacon_pos: Pos) -> int:
        """Roundtrip: cross down (1) + walk to beacon + walk back + cross up (1)."""
        d = manhattan_distance(Pos(barrier_row + 1, cross_col), beacon_pos)
        return 2 + 2 * d

    rt_near = rt(c_near, near_pos)
    rt_mid = rt(c_mid, mid_pos)
    rt_far = rt(c_far, far_pos)

    # --- EDF arrival times: far → mid → near ---
    # Walk from start to above the far crossing, then step onto crossing
    t_edf_far = manhattan_distance(start_pos, Pos(barrier_row - 1, c_far)) + 1
    t_edf_mid = (t_edf_far + rt_far
                 + manhattan_distance(Pos(barrier_row - 1, c_far),
                                      Pos(barrier_row - 1, c_mid)) + 1)
    t_edf_near = (t_edf_mid + rt_mid
                  + manhattan_distance(Pos(barrier_row - 1, c_mid),
                                       Pos(barrier_row - 1, c_near)) + 1)

    # --- Greedy arrival times: near → mid → far ---
    t_greedy_near = manhattan_distance(start_pos, Pos(barrier_row - 1, c_near)) + 1
    t_greedy_mid = (t_greedy_near + rt_near
                    + manhattan_distance(Pos(barrier_row - 1, c_near),
                                         Pos(barrier_row - 1, c_mid)) + 1)
    t_greedy_far = (t_greedy_mid + rt_mid
                    + manhattan_distance(Pos(barrier_row - 1, c_mid),
                                         Pos(barrier_row - 1, c_far)) + 1)

    # --- E2/E3/E4 coin flips ---
    enhanced_cycles  = width >= 7 and rng.random() < 0.5  # E2
    enhanced_coprime = width >= 7 and rng.random() < 0.5  # E3
    enhanced_similar = width >= 7 and rng.random() < 0.5  # E4

    # --- Zone C (far): narrow window opens at EDF arrival ---
    dry_c = rng.randint(3, 5)
    flood_c = rng.randint(20, 30)
    if enhanced_cycles:
        flood_c = max(flood_c, round(flood_c * rng.uniform(1.5, 2.0)))
    cycle_c = dry_c + flood_c
    offset_c = (cycle_c - t_edf_far % cycle_c) % cycle_c

    # --- Zone B (mid): narrow window opens at EDF arrival ---
    dry_b = rng.randint(3, 5)
    flood_b = rng.randint(20, 30)
    if enhanced_similar:
        dry_b, flood_b = dry_c, flood_c
    elif enhanced_cycles:
        flood_b = max(flood_b, round(flood_b * rng.uniform(1.5, 2.0)))
    if enhanced_coprime:
        flood_b = _ensure_coprime(flood_b, dry_b, [dry_c + flood_c])
    cycle_b = dry_b + flood_b
    offset_b = (cycle_b - t_edf_mid % cycle_b) % cycle_b

    # --- Zone A (near): always accessible (very long dry window) ---
    dry_a = max(30, t_edf_near + rt_near + 5)
    flood_a = rng.randint(3, 5)

    # Zone A offset: doesn't matter much, just start dry
    offset_a = 0

    # --- Build temporary zones for verification ---
    zone_a_tmp = TidalZone(
        zone_id="A", tiles=frozenset({cross_near}),
        dry_duration=dry_a, flood_duration=flood_a, phase_offset=offset_a,
    )
    zone_b_tmp = TidalZone(
        zone_id="B", tiles=frozenset({cross_mid}),
        dry_duration=dry_b, flood_duration=flood_b, phase_offset=offset_b,
    )
    zone_c_tmp = TidalZone(
        zone_id="C", tiles=frozenset({cross_far}),
        dry_duration=dry_c, flood_duration=flood_c, phase_offset=offset_c,
    )

    # --- Verify: EDF agent arrives in dry windows ---
    if not zone_c_tmp.is_dry_at(t_edf_far):
        return None
    if not zone_b_tmp.is_dry_at(t_edf_mid):
        return None
    if not zone_a_tmp.is_dry_at(t_edf_near):
        return None

    # --- Verify: greedy arrives at Zone B and Zone C in flood phase ---
    phase_greedy_b = (t_greedy_mid + offset_b) % cycle_b
    if phase_greedy_b < dry_b:
        return None  # greedy arrives dry at B — not a trap

    phase_greedy_c = (t_greedy_far + offset_c) % cycle_c
    if phase_greedy_c < dry_c:
        return None  # greedy arrives dry at C — not a trap

    # --- Build grid ---
    # Zone labels: A = near crossing, B = mid crossing, C = far crossing
    zone_a = TidalZone(
        zone_id="A", tiles=frozenset({cross_near}),
        dry_duration=dry_a, flood_duration=flood_a, phase_offset=offset_a,
    )
    zone_b = TidalZone(
        zone_id="B", tiles=frozenset({cross_mid}),
        dry_duration=dry_b, flood_duration=flood_b, phase_offset=offset_b,
    )
    zone_c = TidalZone(
        zone_id="C", tiles=frozenset({cross_far}),
        dry_duration=dry_c, flood_duration=flood_c, phase_offset=offset_c,
    )

    beacons = {
        near_id: Beacon(beacon_id=near_id, pos=near_pos),
        mid_id: Beacon(beacon_id=mid_id, pos=mid_pos),
        far_id: Beacon(beacon_id=far_id, pos=far_pos),
    }

    # Generous oxygen placeholder — recalibrated by validation pipeline
    oxygen = width + height + rng.randint(20, 40)

    grid = TideGridGrid(
        width=width, height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a, "B": zone_b, "C": zone_c},
        beacons=beacons,
        start=start_pos,
        exit_pos=exit_pos,
        initial_oxygen=oxygen,
    )
    return grid, make_initial_state(grid)


def generate_pendulum_grid(
    width: int, height: int, rng: random.Random,
) -> Optional[tuple[TideGridGrid, TideGridState]]:
    """Generate a Pendulum instance testing modular crossing arithmetic.

    Layout: a single full-width tidal zone row divides the grid into two open
    sections (no walls). Beacons are split: B1 above, B2+B3 below, exit below.
    The agent must cross the zone 3 times (down → up → down), and all 3
    crossing times must satisfy ``(t + offset) % cycle < dry``.

    Bait mechanism (modular arithmetic):
    - Greedy collects B1 first (above, nearest — no crossing needed), then
      crosses down → arrives during FLOOD → WAITs → oxygen drain
    - Optimal crosses down first at a pre-computed dry time, collects B2/B3,
      crosses up (dry), collects B1, crosses down again (dry), exits
    """
    if width < 6 or height < 6:
        return None

    H, W = height, width
    barrier_row = H // 2

    # Need at least 2 rows above and 2 rows below the zone row
    if barrier_row < 2 or (H - barrier_row - 1) < 2:
        return None

    # --- No walls at all — the zone row IS the barrier ---
    walls: set[Pos] = set()

    # --- Zone A tiles: full-width single row ---
    zone_tiles = frozenset(Pos(barrier_row, c) for c in range(W))

    # --- Section tiles ---
    section0 = [Pos(r, c) for r in range(barrier_row) for c in range(W)]
    section1 = [Pos(r, c) for r in range(barrier_row + 1, H) for c in range(W)]

    if len(section0) < 3 or len(section1) < 3:
        return None

    # --- Place start in section 0 (top half) ---
    start_pos = rng.choice(section0)

    # --- Place B1 in section 0 (closest to start by manhattan) ---
    b1_candidates = [p for p in section0
                     if p != start_pos
                     and manhattan_distance(start_pos, p) >= 2]
    if not b1_candidates:
        return None
    b1_pos = rng.choice(b1_candidates)

    # --- Place B2, B3, exit in section 1 with spacing >= 2 ---
    def _pick_spaced(pool: list[Pos], existing: list[Pos], min_dist: int = 2) -> Optional[Pos]:
        cands = [p for p in pool
                 if all(manhattan_distance(p, e) >= min_dist for e in existing)]
        return rng.choice(cands) if cands else None

    b2_pos = _pick_spaced(section1, [])
    if b2_pos is None:
        return None
    b3_pos = _pick_spaced(section1, [b2_pos])
    if b3_pos is None:
        return None
    exit_pos = _pick_spaced(section1, [b2_pos, b3_pos])
    if exit_pos is None:
        return None

    # --- Ensure B1 is nearest beacon to start (bait: greedy collects B1 first) ---
    d_b1 = manhattan_distance(start_pos, b1_pos)
    d_b2 = manhattan_distance(start_pos, b2_pos)
    d_b3 = manhattan_distance(start_pos, b3_pos)
    if d_b1 >= d_b2 or d_b1 >= d_b3:
        return None  # B1 must be strictly closest

    # --- Compute greedy nearest-first crossing time ---
    # Greedy collects B1 first (no crossing), then walks to zone row
    # Nearest column on the zone row from B1
    t_g = manhattan_distance(start_pos, b1_pos) + (barrier_row - b1_pos.row)

    # --- Compute optimal path traversal distances ---
    # Optimal path: start → zone (cross down) → B2 → B3 → zone (cross up) → B1 → zone (cross down) → exit
    # Crossing 1: start walks straight down to zone row
    min_travel = barrier_row - start_pos.row  # steps from start to zone row

    # Landing below after crossing 1: row barrier_row+1, same column as start
    landing1 = Pos(barrier_row + 1, start_pos.col)

    # Below traversal: landing1 → B2 → B3 → back to zone row (from below)
    D_below = (manhattan_distance(landing1, b2_pos)
               + manhattan_distance(b2_pos, b3_pos)
               + (b3_pos.row - barrier_row - 1) + abs(b3_pos.col - b3_pos.col))
    # More precisely: distance from B3 back to zone row from below
    # The agent returns to row barrier_row+1 at some column, then steps onto zone
    return2 = Pos(barrier_row + 1, b3_pos.col)
    D_below = (manhattan_distance(landing1, b2_pos)
               + manhattan_distance(b2_pos, b3_pos)
               + manhattan_distance(b3_pos, return2))

    # Landing above after crossing 2: row barrier_row-1, same column as B3
    landing2 = Pos(barrier_row - 1, b3_pos.col)

    # Above traversal: landing2 → B1 → back to zone row (from above)
    return3 = Pos(barrier_row - 1, b1_pos.col)
    D_above = (manhattan_distance(landing2, b1_pos)
               + manhattan_distance(b1_pos, return3))

    # --- E2/E3/E4 coin flips (E3/E4 no-op for single-zone) ---
    enhanced_cycles  = width >= 7 and rng.random() < 0.5  # E2
    enhanced_coprime = width >= 7 and rng.random() < 0.5  # E3 (no-op)
    enhanced_similar = width >= 7 and rng.random() < 0.5  # E4 (no-op)

    # --- Brute-force search over (dry, flood, offset, W) ---
    if enhanced_cycles:
        dry_range = list(range(2, 5))     # E2: scaled from [1,4)
        flood_range = list(range(6, 13))  # E2: scaled from [4,9)
    else:
        dry_range = list(range(1, 4))     # [1, 2, 3]
        flood_range = list(range(4, 9))   # [4, 5, 6, 7, 8]
    rng.shuffle(dry_range)
    rng.shuffle(flood_range)

    best = None  # (dry, flood, offset, wait)

    for dry in dry_range:
        if best is not None:
            break
        for flood in flood_range:
            if best is not None:
                break
            cycle = dry + flood

            for offset in range(cycle):
                # Greedy must be flooded at t_g
                phase_g = (t_g + offset) % cycle
                if phase_g < dry:
                    continue  # greedy arrives dry — not a trap

                # Search wait values for optimal
                for W_val in range(cycle):
                    t1 = min_travel + W_val          # crossing 1 (onto zone from above)
                    t2 = t1 + D_below + 2            # crossing 2 (onto zone from below)
                    t3 = t2 + D_above + 2            # crossing 3 (onto zone from above)

                    p1 = (t1 + offset) % cycle
                    p2 = (t2 + offset) % cycle
                    p3 = (t3 + offset) % cycle

                    if p1 < dry and p2 < dry and p3 < dry:
                        best = (dry, flood, offset, W_val)
                        break

    if best is None:
        return None

    dry, flood, offset, _ = best
    cycle = dry + flood

    # --- Also check farthest-first greedy is flooded (optional but nice) ---
    # Farthest-first goes to B3 (or B2) first → must cross zone early
    farthest_beacon = b3_pos if d_b3 >= d_b2 else b2_pos
    t_f = manhattan_distance(start_pos, Pos(barrier_row, farthest_beacon.col))
    phase_f = (t_f + offset) % cycle
    # Not a hard requirement — pipeline filters anyway — but skip if farthest is dry
    if phase_f < dry:
        return None

    # --- Build zone and grid ---
    zone_a = TidalZone(
        zone_id="A", tiles=zone_tiles,
        dry_duration=dry, flood_duration=flood, phase_offset=offset,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=b1_pos),
        "b2": Beacon(beacon_id="b2", pos=b2_pos),
        "b3": Beacon(beacon_id="b3", pos=b3_pos),
    }

    # Generous oxygen placeholder — recalibrated by validation pipeline
    oxygen = width + height + rng.randint(15, 30)

    grid = TideGridGrid(
        width=width, height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a},
        beacons=beacons,
        start=start_pos,
        exit_pos=exit_pos,
        initial_oxygen=oxygen,
    )
    return grid, make_initial_state(grid)


# ---------------------------------------------------------------------------
# Template 11: Tide Pool
# ---------------------------------------------------------------------------

def _foray_cost(
    zone_top: int,
    beacons: list[Pos],
    entry_col: int | None = None,
    col_range: range | None = None,
) -> int:
    """Min steps for an enter-collect-exit foray into a dead-end pool zone.

    The agent enters the zone from the row above zone_top (row zone_top - 1),
    steps down into zone_top, collects all *beacons* in some ordering, then
    returns to zone_top row to exit upward.  Cost = enter(1) + travel inside
    + exit-to-zone-top (manhattan from last beacon to zone_top row).

    If *entry_col* is given, only that column is tried.  Otherwise every column
    in *col_range* is tried and the minimum is returned.
    """
    from itertools import permutations

    if not beacons:
        return 0

    cols_to_try: list[int]
    if entry_col is not None:
        cols_to_try = [entry_col]
    elif col_range is not None:
        cols_to_try = list(col_range)
    else:
        raise ValueError("must provide entry_col or col_range")

    best = 10**9
    for ec in cols_to_try:
        entry_pos = Pos(zone_top, ec)  # first tile inside zone
        for perm in permutations(beacons):
            cost = manhattan_distance(entry_pos, perm[0])
            for i in range(1, len(perm)):
                cost += manhattan_distance(perm[i - 1], perm[i])
            # Return to zone_top row (exit upward)
            cost += perm[-1].row - zone_top
            # +1 for the step INTO the zone from above
            total = 1 + cost
            if total < best:
                best = total
    return best


def generate_tide_pool_grid(
    width: int, height: int, rng: random.Random,
) -> Optional[tuple[TideGridGrid, TideGridState]]:
    """Generate a Tide Pool instance testing window allocation.

    Layout: top section (open floor) with start + exit, then a 4-row-deep
    tidal zone block (Zone A) walled on left/right, sealed at the bottom by a
    full wall row.  All 3 beacons are INSIDE the tidal zone.

    Bait mechanism (window allocation):
    - Zone has brief dry windows — agent can't collect all 3 beacons in one
    - Greedy grabs the nearest pair (B1+B2) in foray 1 → leaves B3 (deep/far)
      for foray 2 → B3 round-trip exceeds dry window → stuck inside on flood
    - Optimal grabs the efficient pair (B2+B3, farthest) in foray 1 → B1
      (near) in foray 2 → easy
    """
    if width < 7 or height < 7:
        return None

    H, W = height, width

    # --- Zone geometry ---
    # zone_top: first row of tidal tiles.  Need >=2 rows above for start+exit.
    zone_top = H - 5 if H > 7 else 2
    if zone_top < 2:
        return None
    zone_rows = 4  # 4 rows of tidal tiles
    wall_row = zone_top + zone_rows  # seals bottom
    if wall_row >= H:
        return None

    # Interior columns for zone tiles (cols 1..W-2)
    zone_interior = range(1, W - 1)
    if len(zone_interior) < 3:
        return None  # need at least 3 interior cols for beacon spread

    # --- Build walls ---
    walls: set[Pos] = set()

    # Side walls for zone rows + wall row (col 0 and col W-1)
    for r in range(zone_top, wall_row + 1):
        walls.add(Pos(r, 0))
        walls.add(Pos(r, W - 1))

    # Full wall row at bottom of pool
    for c in range(W):
        walls.add(Pos(wall_row, c))

    # --- Zone A tiles: 4 rows x (W-2) interior cols ---
    zone_tiles: set[Pos] = set()
    for r in range(zone_top, zone_top + zone_rows):
        for c in zone_interior:
            zone_tiles.add(Pos(r, c))

    # --- Top section tiles (rows 0..zone_top-1, all columns) ---
    top_tiles = [Pos(r, c) for r in range(zone_top) for c in range(W)]

    if len(top_tiles) < 2:
        return None

    # --- Place start + exit in top section ---
    start_pos = rng.choice(top_tiles)
    exit_candidates = [p for p in top_tiles
                       if p != start_pos
                       and manhattan_distance(p, start_pos) >= 2]
    if not exit_candidates:
        return None
    exit_pos = rng.choice(exit_candidates)

    # --- Greedy entry column: column nearest to start ---
    greedy_entry_col = start_pos.col
    # Clamp to zone interior
    greedy_entry_col = max(zone_interior.start,
                           min(greedy_entry_col, zone_interior.stop - 1))

    # --- Place beacons inside zone ---
    # B1: near zone top, close to greedy entry col (cheap to collect)
    b1_row = zone_top  # top row of zone
    b1_col = greedy_entry_col
    # Nudge B1 slightly if needed to stay interior
    b1_col = max(zone_interior.start, min(b1_col, zone_interior.stop - 1))
    b1_pos = Pos(b1_row, b1_col)

    # B3: deep (bottom zone row), far from greedy entry col
    b3_row = zone_top + zone_rows - 1  # bottom zone row
    # Place B3 at the far end horizontally from greedy entry
    if greedy_entry_col <= (W - 1) // 2:
        b3_col = zone_interior.stop - 1  # rightmost interior col
    else:
        b3_col = zone_interior.start  # leftmost interior col
    b3_pos = Pos(b3_row, b3_col)

    # B2: moderate depth, close to B3 for efficient arc
    b2_row = zone_top + 1 + rng.randint(0, 1)  # row zone_top+1 or zone_top+2
    # B2 col: close to B3's column
    b2_col_offset = rng.choice([-1, 0, 1])
    b2_col = b3_col + b2_col_offset
    b2_col = max(zone_interior.start, min(b2_col, zone_interior.stop - 1))
    # Ensure B2 != B1 and B2 != B3
    if Pos(b2_row, b2_col) == b1_pos:
        b2_col = b2_col + (1 if b2_col < zone_interior.stop - 1 else -1)
    b2_pos = Pos(b2_row, b2_col)
    if b2_pos == b3_pos or b2_pos == b1_pos:
        return None  # degenerate placement, retry

    # Verify all beacons are on zone tiles
    if b1_pos not in zone_tiles or b2_pos not in zone_tiles or b3_pos not in zone_tiles:
        return None

    # --- Compute foray costs ---
    # Greedy pair: B1+B2 (nearest to greedy entry)
    gp = _foray_cost(zone_top, [b1_pos, b2_pos],
                     entry_col=greedy_entry_col)
    # Greedy remainder: B3 alone from greedy entry
    gr = _foray_cost(zone_top, [b3_pos],
                     entry_col=greedy_entry_col)
    # All three from any entry
    at = _foray_cost(zone_top, [b1_pos, b2_pos, b3_pos],
                     col_range=zone_interior)
    # Optimal pair: B2+B3 from best entry
    op = _foray_cost(zone_top, [b2_pos, b3_pos],
                     col_range=zone_interior)
    # Optimal remainder: B1 from best entry
    ore = _foray_cost(zone_top, [b1_pos],
                      col_range=zone_interior)

    # --- Constraint chain: max(GP+1, OP+1, OR+1) <= dry <= min(GR, AT) ---
    dry_lo = max(gp + 1, op + 1, ore + 1)
    dry_hi = min(gr, at)
    if dry_lo > dry_hi:
        return None  # no valid dry duration

    # Also ensure meaningful separation: greedy rem must exceed dry by margin
    if gr <= dry_lo:
        return None

    # --- E2/E3/E4 coin flips (E3/E4 no-op for single-zone) ---
    enhanced_cycles  = width >= 7 and rng.random() < 0.5  # E2
    enhanced_coprime = width >= 7 and rng.random() < 0.5  # E3 (no-op)
    enhanced_similar = width >= 7 and rng.random() < 0.5  # E4 (no-op)

    # --- Search over (dry, flood, offset) ---
    # Greedy arrival time at zone: manhattan from start to zone_top row entry
    greedy_travel = (zone_top - start_pos.row) + abs(start_pos.col - greedy_entry_col)

    best_params = None
    dry_candidates = list(range(dry_lo, dry_hi + 1))
    rng.shuffle(dry_candidates)

    for dry in dry_candidates:
        if best_params is not None:
            break
        if enhanced_cycles:
            flood_range = list(range(6, 16))  # E2: extended from [4,11)
        else:
            flood_range = list(range(4, 11))
        rng.shuffle(flood_range)
        for flood in flood_range:
            if best_params is not None:
                break
            cycle = dry + flood
            # Scan offsets: greedy must arrive during dry window
            for offset in range(cycle):
                # Greedy enters zone at t = greedy_travel + 1
                # (walks to zone_top-1, then steps down into zone_top)
                t_enter = greedy_travel + 1
                phase_enter = (t_enter + offset) % cycle
                if phase_enter >= dry:
                    continue  # greedy can't enter — not the right bait

                # Greedy foray 1 (B1+B2): fits in window
                # Remaining dry ticks when greedy enters = dry - phase_enter
                remaining_at_enter = dry - phase_enter
                if gp > remaining_at_enter:
                    continue  # greedy can't even finish foray 1

                # Greedy foray 1 exits at t_exit1 = t_enter + gp
                t_exit1 = t_enter + gp
                # Greedy waits for next dry window for foray 2
                phase_exit1 = (t_exit1 + offset) % cycle
                if phase_exit1 < dry:
                    # Still in same dry window — greedy re-enters immediately
                    t_enter2 = t_exit1 + 1
                else:
                    # Wait until next dry window
                    ticks_to_dry = (cycle - phase_exit1) % cycle
                    if ticks_to_dry == 0:
                        ticks_to_dry = cycle
                    t_enter2 = t_exit1 + ticks_to_dry

                # Verify: greedy enters foray 2 during dry
                phase_enter2 = (t_enter2 + offset) % cycle
                if phase_enter2 >= dry:
                    # Adjust: find next dry tick
                    found = False
                    for dt in range(1, cycle + 1):
                        if ((t_enter2 + dt) + offset) % cycle < dry:
                            t_enter2 = t_enter2 + dt
                            phase_enter2 = (t_enter2 + offset) % cycle
                            found = True
                            break
                    if not found:
                        continue

                remaining_at_enter2 = dry - phase_enter2
                # Greedy remainder (B3) must NOT fit: gr > remaining_at_enter2
                if gr <= remaining_at_enter2:
                    continue  # greedy can complete B3 — no trap

                # Optimal foray 1 (B2+B3): check can fit in a dry window
                # Optimal may enter at a different time/column
                # Just verify op < dry (optimal pair fits in full window)
                if op >= dry:
                    continue

                # Optimal remainder (B1): ore < dry
                if ore >= dry:
                    continue

                # All-three must NOT fit: at >= dry
                if at < dry:
                    continue

                best_params = (dry, flood, offset)

    if best_params is None:
        return None

    dry, flood, offset = best_params

    # --- Build zone and grid ---
    zone_a = TidalZone(
        zone_id="A",
        tiles=frozenset(zone_tiles),
        dry_duration=dry,
        flood_duration=flood,
        phase_offset=offset,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=b1_pos),
        "b2": Beacon(beacon_id="b2", pos=b2_pos),
        "b3": Beacon(beacon_id="b3", pos=b3_pos),
    }

    # Generous oxygen placeholder — recalibrated by validation pipeline
    oxygen = width + height + rng.randint(20, 40)

    grid = TideGridGrid(
        width=width, height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a},
        beacons=beacons,
        start=start_pos,
        exit_pos=exit_pos,
        initial_oxygen=oxygen,
    )
    return grid, make_initial_state(grid)


# ---------------------------------------------------------------------------
# Template 12: Airlock
# ---------------------------------------------------------------------------

def generate_airlock_grid(
    width: int, height: int, rng: random.Random,
) -> Optional[tuple[TideGridGrid, TideGridState]]:
    """Generate an Airlock instance testing phase transition precision.

    Layout: two adjacent full-width tidal zone rows with complementary phases.
    Zone A (row za) and Zone B (row za+1) are never both dry at the same time.
    Start, B1, and exit are in the top section; B2 is in the bottom section.
    The agent must cross down (A->B) to collect B2, then cross back up (B->A)
    to reach the exit -- a roundtrip through the airlock.

    Bait mechanism (phase transition precision):
    - B1 in top section (nearest to start) -- greedy collects it first
    - After collecting B1, greedy arrives at Zone A during A's FLOOD phase
    - Must WAIT d ticks for next A dry window -> oxygen wasted
    - Return crossing also hits a flood window -> another d-tick WAIT
    - Combined 2d-tick penalty exceeds the tight oxygen margin
    """
    if width < 6 or height < 6:
        return None

    H, W = height, width
    # Place zones to balance top/bottom sections (>= 2 rows each).
    # zone_a_row must satisfy: zone_a_row >= 2 AND H - zone_a_row - 2 >= 2
    # i.e. zone_a_row in [2, H-4].
    max_za = H - 4
    if max_za < 2:
        return None
    zone_a_row = min(H // 2, max_za)
    zone_b_row = zone_a_row + 1

    # --- No walls -- zone rows act as the barrier ---
    walls: set[Pos] = set()

    # --- Zone tiles: full-width rows ---
    zone_a_tiles = frozenset(Pos(zone_a_row, c) for c in range(W))
    zone_b_tiles = frozenset(Pos(zone_b_row, c) for c in range(W))

    # --- Section tiles ---
    top_tiles = [Pos(r, c) for r in range(zone_a_row) for c in range(W)]
    bot_tiles = [Pos(r, c) for r in range(zone_b_row + 1, H) for c in range(W)]

    if len(top_tiles) < 3 or len(bot_tiles) < 2:
        return None

    # --- Place start in top section ---
    start_pos = rng.choice(top_tiles)

    # --- Place B1 and exit in top section (spacing >= 2) ---
    def _pick_spaced(pool: list[Pos], existing: list[Pos],
                     min_dist: int = 2) -> Optional[Pos]:
        cands = [p for p in pool
                 if all(manhattan_distance(p, e) >= min_dist for e in existing)]
        return rng.choice(cands) if cands else None

    b1_pos = _pick_spaced(top_tiles, [start_pos])
    if b1_pos is None:
        return None
    exit_pos = _pick_spaced(top_tiles, [start_pos, b1_pos])
    if exit_pos is None:
        return None

    # --- Place B2 in bottom section ---
    b2_pos = rng.choice(bot_tiles)

    # --- Ensure B1 is strictly nearest beacon to start ---
    d_b1 = manhattan_distance(start_pos, b1_pos)
    d_b2 = manhattan_distance(start_pos, b2_pos)
    if d_b1 >= d_b2:
        return None

    # --- Greedy arrival time estimates at zone boundary ---
    # The engine checks passability at new_time = state.time + 1.
    # "Arrival at zone A" = agent at row za-1, steps DOWN, zone A checked at
    # state.time + 1.
    # Nearest-first: collects B1, walks to row za-1, then steps into A.
    t_above_nearest = (manhattan_distance(start_pos, b1_pos)
                       + (zone_a_row - 1 - b1_pos.row))
    t_enter_nearest = t_above_nearest + 1  # new_time for zone A entry

    # Farthest-first: goes straight down to row za-1 (skipping B1).
    t_above_farthest = zone_a_row - 1 - start_pos.row
    t_enter_farthest = t_above_farthest + 1

    # --- E2/E3/E4 coin flips (E3/E4 no-op for airlock) ---
    enhanced_cycles  = width >= 7 and rng.random() < 0.5  # E2
    enhanced_coprime = width >= 7 and rng.random() < 0.5  # E3 (no-op)
    enhanced_similar = width >= 7 and rng.random() < 0.5  # E4 (no-op)

    # --- Brute-force search over (d, offset_a) ---
    if enhanced_cycles:
        d_values = [3, 4, 5, 6, 7]  # E2: extended range
    else:
        d_values = [3, 4, 5]
    rng.shuffle(d_values)

    best = None  # (d, offset_a)

    for d in d_values:
        if best is not None:
            break
        cycle = 2 * d  # dry_a = flood_a = d, so cycle = 2*d

        for offset_a in range(cycle):
            # Greedy nearest-first must hit A's flood phase at entry
            phase_nearest = (t_enter_nearest + offset_a) % cycle
            if phase_nearest < d:
                continue  # nearest-first enters dry -- no trap

            # Greedy farthest-first must also hit A's flood phase at entry
            phase_farthest = (t_enter_farthest + offset_a) % cycle
            if phase_farthest < d:
                continue  # farthest-first enters dry -- no trap

            # Optimal must be able to enter A during a dry phase
            # (via initial WAIT of up to cycle-1 ticks above A)
            t_opt_base = zone_a_row - 1 - start_pos.row
            found_optimal = False
            for w in range(cycle):
                # Zone entry at new_time = t_opt_base + w + 1
                phase_opt = (t_opt_base + w + 1 + offset_a) % cycle
                if phase_opt < d:
                    found_optimal = True
                    break
            if not found_optimal:
                continue

            best = (d, offset_a)
            break

    if best is None:
        return None

    d, offset_a = best
    cycle = 2 * d
    offset_b = (offset_a + d) % cycle  # complementary phase

    # --- Build zones ---
    zone_a = TidalZone(
        zone_id="A", tiles=zone_a_tiles,
        dry_duration=d, flood_duration=d, phase_offset=offset_a,
    )
    zone_b = TidalZone(
        zone_id="B", tiles=zone_b_tiles,
        dry_duration=d, flood_duration=d, phase_offset=offset_b,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=b1_pos),
        "b2": Beacon(beacon_id="b2", pos=b2_pos),
    }

    # Generous oxygen placeholder -- recalibrated by validation pipeline
    oxygen = width + height + rng.randint(20, 40)

    grid = TideGridGrid(
        width=width, height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a, "B": zone_b},
        beacons=beacons,
        start=start_pos,
        exit_pos=exit_pos,
        initial_oxygen=oxygen,
    )
    return grid, make_initial_state(grid)


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TEMPLATE_REGISTRY: dict[str, Callable] = {
    "strait": generate_strait_grid,
    "tidal_maze": generate_tidal_maze_grid,
    "gauntlet": generate_gauntlet_grid,
    "roundtrip": generate_roundtrip_grid,
    "confluence": generate_confluence_grid,
    "patience": generate_patience_grid,
    "switchback": generate_switchback_grid,
    "tidal_bridge": generate_tidal_bridge_grid,
    "deadline": generate_deadline_grid,
    "pendulum": generate_pendulum_grid,
    "tide_pool": generate_tide_pool_grid,
    "airlock": generate_airlock_grid,
}

TEMPLATE_NAMES = list(TEMPLATE_REGISTRY.keys())

TEMPLATE_SOLUTION_RANGES: dict[str, tuple[int, int]] = {
    "strait": (10, 35),
    "tidal_maze": (12, 40),
    "gauntlet": (15, 50),
    "roundtrip": (10, 35),
    "confluence": (12, 40),
    "patience": (10, 35),
    "switchback": (15, 50),
    "tidal_bridge": (10, 35),
    "deadline": (15, 50),
    "pendulum": (12, 40),
    "tide_pool": (12, 45),
    "airlock": (10, 35),
}

TEMPLATE_MIN_DEADLOCKS: dict[str, int] = {
    "strait": 1,
    "tidal_maze": 1,
    "gauntlet": 1,
    "roundtrip": 1,
    "confluence": 1,
    "patience": 1,
    "switchback": 1,
    "tidal_bridge": 1,
    "deadline": 1,
    "pendulum": 1,
    "tide_pool": 1,
    "airlock": 1,
}

TEMPLATE_TIERS: dict[str, str] = {
    "strait": "simple", "roundtrip": "simple", "patience": "simple",
    "pendulum": "simple", "airlock": "simple",
    "tidal_maze": "medium", "confluence": "medium",
    "switchback": "medium", "tidal_bridge": "medium",
    "gauntlet": "complex", "deadline": "complex", "tide_pool": "complex",
}

TIER_GRID_SIZES: dict[str, list[tuple[int, int]]] = {
    "simple":  [(6, 6), (7, 7), (8, 8)],   # min 6x6
    "medium":  [(7, 7), (8, 8)],            # min 7x7
    "complex": [(7, 7), (8, 8)],            # min 7x7
}

TEMPLATE_SOLUTION_RANGES_BY_SIZE: dict[str, dict[int, tuple[int, int]]] = {
    # Simple tier (lo-2/hi-7 at 6, base at 7, lo+2/hi+5 at 8)
    "strait":       {6: (8, 28),  7: (10, 35), 8: (12, 40)},
    "roundtrip":    {6: (8, 28),  7: (10, 35), 8: (12, 40)},
    "patience":     {6: (8, 28),  7: (10, 35), 8: (12, 40)},
    "pendulum":     {6: (10, 32), 7: (12, 40), 8: (14, 45)},
    "airlock":      {6: (8, 28),  7: (10, 35), 8: (12, 40)},
    # Medium tier (7, 8 only)
    "tidal_maze":   {7: (12, 40), 8: (14, 45)},
    "confluence":   {7: (12, 40), 8: (14, 45)},
    "switchback":   {7: (15, 50), 8: (17, 55)},
    "tidal_bridge": {7: (10, 35), 8: (12, 40)},
    # Complex tier (7, 8 only)
    "gauntlet":     {7: (15, 50), 8: (17, 55)},
    "deadline":     {7: (15, 50), 8: (17, 55)},
    "tide_pool":    {7: (12, 45), 8: (14, 50)},
}

TEMPLATE_MIN_DEADLOCKS_BY_SIZE: dict[str, dict[int, int]] = {
    "strait":       {6: 1, 7: 1, 8: 2},
    "roundtrip":    {6: 1, 7: 1, 8: 2},
    "patience":     {6: 1, 7: 1, 8: 2},
    "pendulum":     {6: 1, 7: 1, 8: 2},
    "airlock":      {6: 1, 7: 1, 8: 2},
    "tidal_maze":   {7: 2, 8: 2},
    "confluence":   {7: 2, 8: 2},
    "switchback":   {7: 2, 8: 2},
    "tidal_bridge": {7: 2, 8: 2},
    "gauntlet":     {7: 2, 8: 2},
    "deadline":     {7: 2, 8: 2},
    "tide_pool":    {7: 2, 8: 2},
}


# ---------------------------------------------------------------------------
# RQ instance helpers
# ---------------------------------------------------------------------------

def generate_rq2_pair(
    grid: TideGridGrid,
    initial_state: TideGridState,
    catalog_entry: DeadlockEntry,
    k: int,
) -> Optional[tuple[TideGridState, list[TideGridAction], list[TideGridAction]]]:
    """Generate matched deadlock/solvable pair for RQ2.

    Returns (s_start, deadlock_actions, solvable_actions) or None.
    """
    T = catalog_entry.fatal_step
    if T < k - 1:
        return None

    window_start = T - k + 1
    path = catalog_entry.path

    # s_start = state after first window_start actions
    s_start = apply_actions(grid, initial_state, path[:window_start])

    # deadlock actions: the k actions leading to (and including) the fatal step
    deadlock_actions = path[window_start:T + 1]
    if len(deadlock_actions) != k:
        return None

    # solvable actions: oracle solution from s_start, first k actions
    sol = oracle_solve(grid, s_start)
    if sol is None or len(sol) < k:
        return None
    solvable_actions = sol[:k]

    # Verify the deadlock side actually becomes unsolvable
    dead_state = apply_actions(grid, s_start, deadlock_actions)
    if is_solvable(grid, dead_state):
        return None

    # Verify the solvable side remains solvable
    live_state = apply_actions(grid, s_start, solvable_actions)
    if not is_solvable(grid, live_state):
        return None

    return s_start, deadlock_actions, solvable_actions


def generate_rq3_instance(
    grid: TideGridGrid,
    initial_state: TideGridState,
    catalog_entry: DeadlockEntry,
    target_length: int = 0,
) -> Optional[tuple[list[TideGridAction], int, int]]:
    """Generate RQ3 trace with Walking Dead continuation.

    Returns (full_trace, fatal_step_index, cause_symptom_gap) or None.
    Requires gap >= 3.

    When ``target_length`` is 0 (default), it is computed dynamically as
    ``max(18, len(trace_to_deadlock) + 6)`` so that templates with long
    solutions are not silently dropped.
    """
    T = catalog_entry.fatal_step
    path = catalog_entry.path
    trace_to_deadlock = path[:T + 1]

    if target_length == 0:
        target_length = max(18, len(trace_to_deadlock) + 6)

    continuation = _walking_dead_continuation(
        grid, catalog_entry.deadlock_state,
        target_length - len(trace_to_deadlock),
    )

    full_trace = trace_to_deadlock + continuation
    gap = len(full_trace) - T - 1

    if gap < 3:
        return None

    return full_trace, T, gap


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------

def _generate_validated_inner(
    width: int = 7,
    height: int = 6,
    seed: int = 0,
    template: str = "strait",
    max_retries: int = 200,
    solution_range: Optional[tuple[int, int]] = None,
    min_deadlocks: Optional[int] = None,
    min_tier2: int = 1,
) -> tuple[Optional[GeneratedInstance], dict[str, int], int]:
    """Core generation loop with rejection tracking."""
    if template not in TEMPLATE_REGISTRY:
        raise ValueError(f"Unknown template {template!r}. Available: {TEMPLATE_NAMES}")

    grid_fn = TEMPLATE_REGISTRY[template]

    # Per-size lookups with flat-table fallback
    if solution_range is not None:
        sol_range = solution_range
    else:
        sol_range = TEMPLATE_SOLUTION_RANGES_BY_SIZE.get(template, {}).get(
            width, TEMPLATE_SOLUTION_RANGES.get(template, (8, 40)))

    if min_deadlocks is not None:
        min_dl = min_deadlocks
    else:
        min_dl = TEMPLATE_MIN_DEADLOCKS_BY_SIZE.get(template, {}).get(
            width, TEMPLATE_MIN_DEADLOCKS.get(template, 1))

    # Auto-scaling retries for larger grids
    effective_retries = max_retries
    if width >= 8:
        effective_retries = max(max_retries, 500)
    elif width >= 7:
        effective_retries = max(max_retries, 300)

    rejections: dict[str, int] = {
        "unsolvable": 0,
        "solution_range": 0,
        "greedy_passes": 0,
        "insufficient_deadlocks": 0,
        "insufficient_tier2": 0,
    }

    for attempt in range(effective_retries):
        rng = random.Random(seed + attempt)

        result = grid_fn(width, height, rng)
        if result is None:
            continue

        grid, initial_state = result

        # Filter 1: Must be solvable (with generous placeholder oxygen)
        solution = oracle_solve(grid, initial_state)
        if solution is None:
            rejections["unsolvable"] += 1
            continue

        # --- Oxygen recalibration: set from oracle solution length ---
        # E1: stochastic reduced oxygen slack at width >= 7
        enhanced_oxygen = width >= 7 and rng.random() < 0.5
        if enhanced_oxygen:
            if width <= 6:   tight_oxygen = len(solution) + rng.randint(1, 3)
            elif width >= 8: tight_oxygen = len(solution) + rng.randint(2, 5)
            else:            tight_oxygen = len(solution) + rng.randint(2, 4)
        else:
            if width <= 6:   tight_oxygen = len(solution) + rng.randint(2, 5)
            elif width >= 8: tight_oxygen = len(solution) + rng.randint(3, 7)
            else:            tight_oxygen = len(solution) + rng.randint(3, 6)
        grid = TideGridGrid(
            width=grid.width,
            height=grid.height,
            walls=grid.walls,
            tidal_zones=grid.tidal_zones,
            beacons=grid.beacons,
            start=grid.start,
            exit_pos=grid.exit_pos,
            initial_oxygen=tight_oxygen,
        )
        initial_state = make_initial_state(grid)

        # Re-solve with tight oxygen to verify
        solution = oracle_solve(grid, initial_state)
        if solution is None:
            rejections["unsolvable"] += 1
            continue

        # Filter 2: Solution length in range
        if not (sol_range[0] <= len(solution) <= sol_range[1]):
            rejections["solution_range"] += 1
            continue

        # Filter 3: ALL greedy strategies must fail
        if not all_greedy_strategies_fail(grid, initial_state):
            rejections["greedy_passes"] += 1
            continue

        # Filter 4: Must have enough deadlocks
        # Primary: greedy-trace deadlocks (meaningful structural baits)
        greedy_catalog = discover_greedy_deadlocks(grid, initial_state)
        # Supplement: BFS forward-sweep deadlocks
        bfs_catalog = discover_deadlocks(grid, initial_state,
                                         max_depth=12, max_states=10_000)
        # Merge: greedy first (higher quality), then BFS (deduplicated)
        seen_states: set[tuple] = set()
        catalog: list[DeadlockEntry] = []
        for entry in greedy_catalog:
            h = entry.deadlock_state.to_hashable()
            if h not in seen_states:
                seen_states.add(h)
                catalog.append(entry)
        for entry in bfs_catalog:
            h = entry.deadlock_state.to_hashable()
            if h not in seen_states:
                seen_states.add(h)
                catalog.append(entry)
        if len(catalog) < min_dl:
            rejections["insufficient_deadlocks"] += 1
            continue

        # Compute k_horizon lazily (stop when enough tiers are found)
        tier1 = tier2 = 0
        for entry in catalog:
            entry.k_horizon = count_reachable_dead_states(
                grid, entry.deadlock_state,
            )
            if entry.k_horizon >= 3:
                tier1 += 1
            if entry.k_horizon >= 5:
                tier2 += 1
            if tier1 >= min_dl and tier2 >= min_tier2:
                break

        if tier1 < min_dl:
            rejections["insufficient_deadlocks"] += 1
            continue
        if tier2 < min_tier2:
            rejections["insufficient_tier2"] += 1
            continue

        max_kh = max((e.k_horizon for e in catalog), default=0)

        inst = GeneratedInstance(
            grid=grid,
            initial_state=initial_state,
            solution=solution,
            deadlock_catalog=catalog,
            seed=seed + attempt,
            template_name=template,
            grid_width=grid.width,
            grid_height=grid.height,
            num_beacons=len(grid.beacons),
            num_zones=len(grid.tidal_zones),
            solution_length=len(solution),
            tier1_deadlocks=tier1,
            tier2_deadlocks=tier2,
            max_k_horizon=max_kh,
            generation_attempts=attempt + 1,
        )

        # Apply D4 transform for square grids
        inst = _transform_instance(inst, rng)
        return inst, rejections, attempt + 1

    return None, rejections, effective_retries


def generate_validated_instance(
    width: int = 7,
    height: int = 6,
    seed: int = 0,
    template: str = "strait",
    max_retries: int = 200,
    solution_range: Optional[tuple[int, int]] = None,
    min_deadlocks: Optional[int] = None,
    min_tier2: int = 1,
) -> Optional[GeneratedInstance]:
    """Public entry point: returns just the instance (or None)."""
    inst, _stats, _total = _generate_validated_inner(
        width=width, height=height, seed=seed, template=template,
        max_retries=max_retries, solution_range=solution_range,
        min_deadlocks=min_deadlocks, min_tier2=min_tier2,
    )
    return inst


def generate_validated_instance_with_stats(
    width: int = 7,
    height: int = 6,
    seed: int = 0,
    template: str = "strait",
    max_retries: int = 200,
    solution_range: Optional[tuple[int, int]] = None,
    min_deadlocks: Optional[int] = None,
    min_tier2: int = 1,
) -> GenerationResult:
    """Like generate_validated_instance but returns full stats."""
    t0 = _time.time()
    inst, stats, total = _generate_validated_inner(
        width=width, height=height, seed=seed, template=template,
        max_retries=max_retries, solution_range=solution_range,
        min_deadlocks=min_deadlocks, min_tier2=min_tier2,
    )
    return GenerationResult(
        instance=inst,
        stats=stats,
        total_attempts=total,
        elapsed_seconds=_time.time() - t0,
    )


def is_eval_ready(inst: GeneratedInstance, require_hard: bool = False) -> bool:
    """Return True if instance has viable RQ2, RQ3 variants."""
    grid = inst.grid
    s0 = inst.initial_state
    catalog = inst.deadlock_catalog

    # RQ2 standard: at least one eligible entry must produce a valid pair
    eligible_rq2 = [
        e for e in catalog if e.k_horizon >= 3 and e.fatal_step >= 2
    ][:3]
    has_rq2 = any(
        generate_rq2_pair(grid, s0, e, e.fatal_step + 1) is not None
        for e in eligible_rq2
    )
    if not has_rq2:
        return False

    # RQ3 standard: at least one tier-2 entry must produce a valid trace
    tier2 = [e for e in catalog if e.k_horizon >= 5][:2]
    has_rq3 = any(
        generate_rq3_instance(grid, s0, e) is not None for e in tier2
    )
    if not has_rq3:
        return False

    if not require_hard:
        return True

    # Hard variants require oracle-adjacent deadlocks
    oa_catalog = discover_oracle_adjacent_deadlocks(grid, s0, inst.solution)
    for entry in oa_catalog:
        entry.k_horizon = count_reachable_dead_states(grid, entry.deadlock_state)

    eligible_hard = [
        e for e in oa_catalog if e.k_horizon >= 3 and e.fatal_step >= 2
    ][:3]
    has_hard_rq2 = any(
        generate_hard_rq2_pair(grid, s0, inst.solution, e) is not None
        for e in eligible_hard
    )
    if not has_hard_rq2:
        return False

    tier2_hard = [e for e in oa_catalog if e.k_horizon >= 5][:5]
    has_hard_rq3 = any(
        generate_hard_rq3_instance(grid, s0, inst.solution, e) is not None
        for e in tier2_hard
    )
    return has_hard_rq3
