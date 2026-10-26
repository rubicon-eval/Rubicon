"""Chain Vault procedural generation and deadlock discovery.

Key design principle: ANTI-GREEDY BY CONSTRUCTION.
The trap door is always placed closer to the exit than the correct door,
ensuring that greedy strategies (move toward exit, use key on nearest door)
lead to deadlock. Multiple greedy strategies are validated as hard filters.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Optional

from puzzles.common.types import Pos, Direction, add_pos, in_bounds, manhattan_distance
from .engine import (
    ChainVaultGrid,
    ChainVaultState,
    ChainVaultAction,
    KeyInfo,
    DoorInfo,
    get_legal_actions,
    apply_action,
    is_solved,
    is_passable,
    action_opens_door,
    get_door_opened_by_action,
    _inventory_to_dict,
    _has_keys,
)
from .oracle import (
    is_solvable,
    oracle_solve,
    is_walking_dead,
    apply_actions,
    find_fatal_step,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DeadlockEntry:
    """A discovered deadlock: a path from S0 to a Walking Dead state."""

    path: list[ChainVaultAction]
    fatal_step: int          # index of the action that caused solvability flip
    deadlock_state: ChainVaultState
    pre_deadlock_state: ChainVaultState
    k_horizon: int = 0       # reachable states from deadlock (Walking Dead depth)


@dataclass
class GeneratedInstance:
    """A fully validated puzzle instance with solution and deadlock catalog."""

    grid: ChainVaultGrid
    initial_state: ChainVaultState
    solution: list[ChainVaultAction]
    deadlock_catalog: list[DeadlockEntry]
    seed: int
    template_name: str = "fork"
    grid_width: int = 0
    grid_height: int = 0
    num_keys: int = 0
    num_doors: int = 0
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


LOW_SIGNAL_HARD_TEMPLATES = frozenset({
    "fork",
    "key_behind_key",
    "key_collector",
    "order_dependent_lock",
    "sequential_dual_locks",
})


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


def _convergent_key_pos(
    trap_door: Pos,
    zone: set[Pos],
    rng: random.Random,
) -> Optional[Pos]:
    """Pick a key position near a trap door for convergent-forks bait.

    Usually the cell directly above the trap door; occasionally a diagonally
    adjacent cell (Chebyshev distance 1) for visual variety across seeds.
    Candidates: directly above, diagonal-left, diagonal-right.
    """
    ideal = Pos(trap_door.row - 1, trap_door.col)
    if ideal not in zone:
        return None
    nearby = [
        p for p in zone
        if p != ideal
        and abs(p.row - trap_door.row) <= 1
        and abs(p.col - trap_door.col) <= 1
    ]
    if not nearby or rng.random() < 0.5:
        return ideal
    return rng.choice(nearby)


def _pick_extra_trap_col(
    barrier_row: int,
    walls: set[Pos],
    used_cols: set[int],
    preferred_half: list[int],
    rng: random.Random,
) -> Optional[int]:
    """Pick an unused column on a horizontal barrier for an extra trap door.

    Finds a random barrier column from ``preferred_half`` that is currently a
    wall (i.e., in ``walls``) and not in ``used_cols``.  Returns the column
    number or ``None`` if no valid column exists.

    The caller is responsible for removing ``Pos(barrier_row, col)`` from
    ``walls`` and adding the new door to the door dict.
    """
    candidates = [
        c for c in preferred_half
        if Pos(barrier_row, c) in walls and c not in used_cols
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def _select_key_positions(
    candidates: list[Pos],
    n: int,
    rng: random.Random,
    min_spacing: int = 2,
    already_placed: frozenset[Pos] | None = None,
    reference: Pos | None = None,
) -> list[Pos] | None:
    """Select *n* key positions with spacing and distance constraints.

    Three-phase greedy selection mirroring mosaic_grid's
    ``_select_plate_positions``:

    Phase 1 (spacing + distinct distances from *reference*):
        Skip candidates that share a manhattan distance to *reference* with an
        already-selected key, or are < *min_spacing* from any selected /
        *already_placed* key.
    Phase 2 (spacing only):
        Drop the distinct-distance constraint.  Re-shuffle and retry.
    Phase 3 (relaxed, ``min_spacing=1``):
        Minimum viable fallback to prevent regression.

    Returns ``None`` only if all three phases fail (very unlikely).
    """
    placed = set(already_placed) if already_placed else set()

    def _too_close(pos: Pos, selected: list[Pos], spacing: int) -> bool:
        for p in selected:
            if manhattan_distance(pos, p) < spacing:
                return True
        for p in placed:
            if manhattan_distance(pos, p) < spacing:
                return True
        return False

    # --- Phase 1: spacing + distinct distances from reference ---
    if reference is not None:
        rng.shuffle(candidates)
        selected: list[Pos] = []
        used_distances: set[int] = set()
        for cand in candidates:
            d = manhattan_distance(cand, reference)
            if d in used_distances:
                continue
            if _too_close(cand, selected, min_spacing):
                continue
            selected.append(cand)
            used_distances.add(d)
            if len(selected) == n:
                return selected

    # --- Phase 2: spacing only ---
    rng.shuffle(candidates)
    selected = []
    for cand in candidates:
        if _too_close(cand, selected, min_spacing):
            continue
        selected.append(cand)
        if len(selected) == n:
            return selected

    # --- Phase 3: relaxed (min_spacing=1) ---
    if min_spacing > 1:
        rng.shuffle(candidates)
        selected = []
        for cand in candidates:
            if _too_close(cand, selected, 1):
                continue
            selected.append(cand)
            if len(selected) == n:
                return selected

    return None


def _trap_closer_to_exit(trap: Pos, correct: Pos, exit_pos: Pos) -> bool:
    """Return True if trap door is strictly closer to exit than correct door."""
    return manhattan_distance(trap, exit_pos) < manhattan_distance(correct, exit_pos)


# ---------------------------------------------------------------------------
# D4 orientation randomization (rotate + flip after validation)
# ---------------------------------------------------------------------------

# Lookup: direction -> ChainVaultAction
_ACTION_BY_DIR: dict[Direction, ChainVaultAction] = {
    action.direction: action for action in ChainVaultAction
}

# Direction rotation tables
_ROT90_DIR = {
    Direction.UP: Direction.RIGHT, Direction.RIGHT: Direction.DOWN,
    Direction.DOWN: Direction.LEFT, Direction.LEFT: Direction.UP,
}
_ROT180_DIR = {
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT,
}
_ROT270_DIR = {
    Direction.UP: Direction.LEFT, Direction.LEFT: Direction.DOWN,
    Direction.DOWN: Direction.RIGHT, Direction.RIGHT: Direction.UP,
}
_FLIP_H_DIR = {
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT,
    Direction.UP: Direction.UP, Direction.DOWN: Direction.DOWN,
}
_DIR_ROT_TABLE: dict[int, dict[Direction, Direction]] = {
    0: {d: d for d in Direction},
    90: _ROT90_DIR,
    180: _ROT180_DIR,
    270: _ROT270_DIR,
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
    action: ChainVaultAction, rotation: int, flip_h: bool,
) -> ChainVaultAction:
    """Build a new action with the transformed direction."""
    new_dir = _transform_direction(action.direction, rotation, flip_h)
    return _ACTION_BY_DIR[new_dir]


def _transform_grid(
    grid: ChainVaultGrid, rotation: int, flip_h: bool,
) -> ChainVaultGrid:
    """Transform all positions in a grid (walls, keys, doors, start, exit)."""
    s = grid.width  # square grid
    tp = lambda p: _transform_pos(p, s, rotation, flip_h)
    return ChainVaultGrid(
        width=grid.width,
        height=grid.height,
        walls=frozenset(tp(w) for w in grid.walls),
        keys={
            kid: replace(ki, pos=tp(ki.pos))
            for kid, ki in grid.keys.items()
        },
        doors={
            did: replace(di, pos=tp(di.pos))
            for did, di in grid.doors.items()
        },
        start=tp(grid.start),
        exit_pos=tp(grid.exit_pos),
    )


def _transform_state(
    state: ChainVaultState, size: int, rotation: int, flip_h: bool,
) -> ChainVaultState:
    """Transform agent_pos; inventory/doors/keys pass through unchanged."""
    return replace(
        state,
        agent_pos=_transform_pos(state.agent_pos, size, rotation, flip_h),
    )


def _permute_colors(
    inst: GeneratedInstance, rng: random.Random,
) -> GeneratedInstance:
    """Randomly permute color assignments across the instance (P1.12).

    Collects all colors used in the grid, generates a random permutation,
    and applies it to keys, doors, and all state inventories. Prevents
    color-role memorization (e.g., "in KBK, the first key is always red").
    """
    # Collect unique colors from keys
    colors = sorted({ki.color for ki in inst.grid.keys.values()})
    if len(colors) <= 1:
        return inst  # nothing to permute

    shuffled = list(colors)
    rng.shuffle(shuffled)
    if shuffled == colors:
        return inst  # identity permutation

    cmap = dict(zip(colors, shuffled))

    def _remap_inv(inv: tuple[tuple[str, int], ...]) -> tuple[tuple[str, int], ...]:
        return tuple(sorted((cmap.get(c, c), n) for c, n in inv))

    new_keys = {
        kid: replace(ki, color=cmap.get(ki.color, ki.color))
        for kid, ki in inst.grid.keys.items()
    }
    new_doors = {
        did: replace(di, required_keys=tuple(
            cmap.get(c, c) for c in di.required_keys))
        for did, di in inst.grid.doors.items()
    }
    new_grid = replace(inst.grid, keys=new_keys, doors=new_doors)
    new_state = replace(inst.initial_state, inventory=_remap_inv(inst.initial_state.inventory))
    new_catalog = []
    for entry in inst.deadlock_catalog:
        new_catalog.append(DeadlockEntry(
            path=entry.path,
            fatal_step=entry.fatal_step,
            deadlock_state=replace(entry.deadlock_state,
                                   inventory=_remap_inv(entry.deadlock_state.inventory)),
            pre_deadlock_state=replace(entry.pre_deadlock_state,
                                       inventory=_remap_inv(entry.pre_deadlock_state.inventory)),
            k_horizon=entry.k_horizon,
        ))
    return replace(inst, grid=new_grid, initial_state=new_state, deadlock_catalog=new_catalog)


def _transform_instance(
    inst: GeneratedInstance, rng: random.Random,
) -> GeneratedInstance:
    """Apply a random D4 symmetry (rotate then flip) and color permutation."""
    # D4 spatial transform
    rotation = rng.choice([0, 90, 180, 270])
    flip_h = rng.choice([True, False])
    if rotation != 0 or flip_h:
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
            ))
        inst = GeneratedInstance(
            grid=new_grid,
            initial_state=new_state,
            solution=new_solution,
            deadlock_catalog=new_catalog,
            seed=inst.seed,
            template_name=inst.template_name,
            grid_width=inst.grid_width,
            grid_height=inst.grid_height,
            num_keys=inst.num_keys,
            num_doors=inst.num_doors,
            solution_length=inst.solution_length,
            tier1_deadlocks=inst.tier1_deadlocks,
            tier2_deadlocks=inst.tier2_deadlocks,
            max_k_horizon=inst.max_k_horizon,
            generation_attempts=inst.generation_attempts,
            fingerprint=inst.fingerprint,
        )

    # Color permutation (P1.12)
    inst = _permute_colors(inst, rng)
    return inst


# ---------------------------------------------------------------------------
# k-horizon: reachable states from a Walking Dead state
# ---------------------------------------------------------------------------

def count_reachable_dead_states(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_states: int = 200,
) -> int:
    """BFS from a Walking Dead state to count reachable states.

    Measures the k-horizon: how many unique states the agent can still
    reach after entering a deadlock.  Higher values mean longer Walking Dead
    continuations are possible (needed for RQ3 cause-symptom gap).
    """
    visited = {state.to_hashable()}
    queue = deque([state])
    while queue and len(visited) < max_states:
        s = queue.popleft()
        for a in get_legal_actions(grid, s):
            ns = apply_action(grid, s, a)
            h = ns.to_hashable()
            if h not in visited:
                visited.add(h)
                queue.append(ns)
    return len(visited)


# ---------------------------------------------------------------------------
# BFS pathfinding helper (used by BFS-based greedy strategies)
# ---------------------------------------------------------------------------

def _bfs_to(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    target: Pos,
) -> tuple[Optional[Direction], Optional[int]]:
    """BFS shortest path from agent position to target.

    Treats locked doors as impassable walls. If target itself is impassable
    (e.g., a locked door tile the agent wants to stand adjacent to), navigates
    to the nearest passable cell adjacent to target instead.

    Returns (first_direction, path_length) or (None, None) if unreachable.
    """
    start = state.agent_pos

    locked_door_pos = {
        d.pos for d in grid.doors.values()
        if d.id not in state.opened_doors
    }
    passable: set[Pos] = {
        Pos(r, c)
        for r in range(grid.height)
        for c in range(grid.width)
        if Pos(r, c) not in grid.walls and Pos(r, c) not in locked_door_pos
    }

    # Goals: target itself if passable, else adjacent passable cells
    if target in passable:
        goals: set[Pos] = {target}
    else:
        goals = {
            add_pos(target, d) for d in Direction
            if in_bounds(add_pos(target, d), grid.width, grid.height)
            and add_pos(target, d) in passable
        }

    if not goals:
        return None, None

    queue: deque[tuple[Pos, Optional[Direction], int]] = deque([(start, None, 0)])
    visited: set[Pos] = {start}

    while queue:
        pos, first_dir, dist = queue.popleft()
        for d in Direction:
            np_ = add_pos(pos, d)
            if np_ not in passable or np_ in visited:
                continue
            fd = first_dir if first_dir is not None else d
            if np_ in goals:
                return fd, dist + 1
            visited.add(np_)
            queue.append((np_, fd, dist + 1))

    return None, None


# ---------------------------------------------------------------------------
# Anti-greedy fork grid generation
# ---------------------------------------------------------------------------


def _generate_fork_3way(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """3-way fork: 1 red key, 3 red doors (2 traps, 1 correct).

    MECHANIC: Agent must eliminate 2 trap doors to find the 1 correct door.
    "Closer to exit" heuristic fails — both traps are placed strategically.

    LAYOUT (vertical barrier, same as standard fork):
        Left side:  Start + K1 (near a trap door approach)
        Barrier:    3 red doors at rows r1, r2, r3
        Right side: 2 horizontal dividers → 3 zones
          Zone for correct door: contains exit
          Zone for each trap door: walkable dead end (>=2 cells)

    ANTI-GREEDY:
    - K1 near closest-to-exit trap door (convergent bait)
    - Correct door has max manhattan distance to exit
    """
    h, w = height, width
    if w < 5 or h < 6:
        return None

    # --- 1. Barrier column ---
    barrier_col = rng.randint(2, w - 3)

    # --- 2. Three door rows (sorted, pairwise gap >= 2) ---
    # Pick 3 rows from [0, h-1] with pairwise distance >= 2
    all_rows = list(range(h))
    rng.shuffle(all_rows)
    door_rows: list[int] = []
    for r in all_rows:
        if all(abs(r - dr) >= 2 for dr in door_rows):
            door_rows.append(r)
            if len(door_rows) == 3:
                break
    if len(door_rows) < 3:
        return None
    door_rows.sort()
    r1, r2, r3 = door_rows

    # --- 3. Divider rows (one between each consecutive door pair) ---
    d1_opts = list(range(r1 + 1, r2))
    d2_opts = list(range(r2 + 1, r3))
    if not d1_opts or not d2_opts:
        return None
    d1 = rng.choice(d1_opts)
    d2 = rng.choice(d2_opts)

    # --- 4. Assign correct door (farthest from exit side) ---
    # Exit will be on bottom or top edge; correct door = farthest from that edge
    exit_edge = rng.choice([0, h - 1])  # which row edge has exit
    door_dists = [(abs(r - exit_edge), r) for r in door_rows]
    door_dists.sort(reverse=True)  # farthest first
    correct_row = door_dists[0][1]
    trap_rows = [r for r in door_rows if r != correct_row]

    # --- 5. Build walls ---
    walls: set[Pos] = set()

    # Barrier column (except door rows)
    for r in range(h):
        if r not in door_rows:
            walls.add(Pos(r, barrier_col))

    # Divider 1: horizontal wall on right side at row d1
    for c in range(barrier_col + 1, w):
        walls.add(Pos(d1, c))

    # Divider 2: horizontal wall on right side at row d2
    for c in range(barrier_col + 1, w):
        walls.add(Pos(d2, c))

    # --- 6. Zone computation ---
    all_right = {
        Pos(r, c) for r in range(h) for c in range(barrier_col + 1, w)
        if Pos(r, c) not in walls
    }

    # Each door's entry point on right side
    entries = {r: Pos(r, barrier_col + 1) for r in door_rows}
    for r in door_rows:
        if entries[r] not in all_right:
            return None

    # Flood fill each zone from its door entry
    zones: dict[int, set[Pos]] = {}
    for r in door_rows:
        zones[r] = _flood_fill(entries[r], all_right)

    # Zones must be pairwise disjoint
    for i, ra in enumerate(door_rows):
        for rb in door_rows[i + 1:]:
            if zones[ra] & zones[rb]:
                return None

    # Each zone must have >= 2 cells
    for r in door_rows:
        if len(zones[r]) < 2:
            return None

    # --- 7. Exit placement (in correct zone, biased toward exit_edge) ---
    correct_zone = zones[correct_row]
    exit_candidates = [
        p for p in correct_zone if p != entries[correct_row]
    ]
    if not exit_candidates:
        return None
    # Bias toward the exit_edge row for stronger manhattan bait
    exit_candidates.sort(key=lambda p: abs(p.row - exit_edge))
    n_pick = max(1, len(exit_candidates) // 2)
    exit_pos = rng.choice(exit_candidates[:n_pick])

    # --- 8. Left side: key and start ---
    left_floor = {
        Pos(r, c) for r in range(h) for c in range(barrier_col)
        if Pos(r, c) not in walls
    }
    if not left_floor or not _is_connected(left_floor):
        return None

    # All door approaches must be reachable from left side
    approaches = {r: Pos(r, barrier_col - 1) for r in door_rows}
    for r in door_rows:
        if approaches[r] not in left_floor:
            return None

    # Key: convergent bait near trap door approach (left side of barrier).
    # Place key adjacent to the closest-to-exit trap's approach so greedy
    # agents pick it up and immediately USE_KEY on the trap door.
    trap_by_exit_dist = sorted(trap_rows, key=lambda r: abs(r - exit_edge))
    bait_trap_row = trap_by_exit_dist[0]
    bait_approach = approaches[bait_trap_row]  # Pos(bait_trap_row, barrier_col-1)
    # Candidates: cells in left_floor adjacent to the approach (Chebyshev ≤ 1)
    bait_candidates = [
        p for p in left_floor
        if p != bait_approach
        and abs(p.row - bait_approach.row) <= 1
        and abs(p.col - bait_approach.col) <= 1
    ]
    if not bait_candidates:
        return None
    key_pos = rng.choice(bait_candidates)

    # Start: different from key, on left side
    start_candidates = [p for p in left_floor if p != key_pos]
    if not start_candidates:
        return None
    start = rng.choice(start_candidates)

    # Verify connectivity
    left_comp = _flood_fill(start, left_floor)
    if key_pos not in left_comp:
        return None
    for r in door_rows:
        if approaches[r] not in left_comp:
            return None

    # --- 10. Extra walls (left side, 0-2, connectivity-preserving) ---
    protected = {start, key_pos} | set(approaches.values())
    extras = [p for p in left_floor if p not in protected]
    rng.shuffle(extras)
    n_extra = rng.randint(0, min(2, len(extras) // 4))
    for pos in extras:
        if n_extra <= 0:
            break
        rem = left_floor - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in protected):
            walls.add(pos)
            left_floor.discard(pos)
            n_extra -= 1

    # --- 11. Build grid ---
    keys = {
        "K1": KeyInfo(id="K1", pos=key_pos, color="red"),
    }
    doors = {
        "D_c": DoorInfo(
            id="D_c", pos=Pos(correct_row, barrier_col),
            required_keys=("red",)),
        "D_t1": DoorInfo(
            id="D_t1", pos=Pos(trap_rows[0], barrier_col),
            required_keys=("red",)),
        "D_t2": DoorInfo(
            id="D_t2", pos=Pos(trap_rows[1], barrier_col),
            required_keys=("red",)),
    }
    grid = ChainVaultGrid(
        width=w, height=h,
        walls=frozenset(walls), keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def generate_fork_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate an anti-greedy fork grid: barrier with 2 door gaps, 1 key.

    ANTI-GREEDY DESIGN (zone divider + convergent forks):
    1. Red key placed adjacent to trap door approach (convergent forks bait)
       => greedy picks up key next to trap -> USE_KEY fires
    2. Trap door leads to a real walkable zone (3+ cells, zone divider)
    3. Correct door leads to a connected region containing the exit
    4. Post-hoc greedy filter (all 6 strategies must fail)

    Strategy:
    - Vertical barrier with 2 door gaps (same color, red)
    - Trap door at grid edge (row 0 or height-1)
    - Correct door at opposite end (large manhattan gap)
    - Right side split by horizontal divider into Zone R_trap and Zone R_correct
    - Exit in Zone R_correct, only reachable through correct door
    """
    if rng is None:
        rng = random.Random()

    if height < 5 or width < 5:
        return None

    # --- Stochastic 3-way fork variant (P2.1) ---
    if width >= 6 and height >= 7 and rng.random() < 0.5:
        result = _generate_fork_3way(width, height, rng)
        if result is not None:
            return result

    # --- 1. Barrier column: need >= 2 cols on each side ---
    barrier_col = rng.randint(2, width - 3)

    # --- 2. Door placement: trap at edge, correct far away ---
    trap_row = rng.choice([0, height - 1])
    if trap_row == 0:
        # Correct door in bottom half, at least 3 rows from trap
        correct_min = max(3, height // 2)
        correct_row = rng.randint(correct_min, height - 1)
    else:
        # Correct door in top half, at least 3 rows from trap
        correct_max = min(height - 4, height // 2)
        correct_row = rng.randint(0, max(0, correct_max))

    if abs(trap_row - correct_row) < 3:
        return None

    trap_door_pos = Pos(trap_row, barrier_col)
    correct_door_pos = Pos(correct_row, barrier_col)

    # --- 3. Build walls ---
    walls: set[Pos] = set()

    # Barrier column (except door rows)
    for r in range(height):
        if r != trap_row and r != correct_row:
            walls.add(Pos(r, barrier_col))

    # Zone divider: horizontal wall on right side between trap and correct rows
    lo, hi = min(trap_row, correct_row), max(trap_row, correct_row)
    divider_options = list(range(lo + 1, hi))
    if not divider_options:
        return None
    # Bias toward correct_row to maximize Zone R_trap
    if correct_row > trap_row:
        divider_row = divider_options[-1]
    else:
        divider_row = divider_options[0]
    for c in range(barrier_col + 1, width):
        walls.add(Pos(divider_row, c))

    # --- Scaling: extra trap door on vertical barrier at w>=6 ---
    extra_trap_pos: Optional[Pos] = None
    if width >= 6:
        # Find barrier wall rows in the trap zone (between trap_row and divider)
        if trap_row < divider_row:
            trap_zone_rows = list(range(trap_row + 1, divider_row))
        else:
            trap_zone_rows = list(range(divider_row + 1, trap_row))
        barrier_wall_rows = [
            r for r in trap_zone_rows
            if Pos(r, barrier_col) in walls
        ]
        if barrier_wall_rows:
            extra_r = rng.choice(barrier_wall_rows)
            extra_trap_pos = Pos(extra_r, barrier_col)
            walls.discard(extra_trap_pos)

    trap_entry = Pos(trap_row, barrier_col + 1)
    correct_entry = Pos(correct_row, barrier_col + 1)

    # --- 4. Find valid exit position ---
    # Exit must be: on right side, in correct_reachable, closer to trap than correct
    all_right_floor = {
        Pos(r, c) for r in range(height) for c in range(barrier_col + 1, width)
        if Pos(r, c) not in walls
    }

    if trap_entry not in all_right_floor or correct_entry not in all_right_floor:
        return None

    trap_reachable = _flood_fill(trap_entry, all_right_floor)
    correct_reachable = _flood_fill(correct_entry, all_right_floor)

    # Trap must be isolated from correct region
    if trap_reachable & correct_reachable:
        return None

    # Trap must be a real walkable area (>=2 cells)
    if len(trap_reachable) < 2:
        return None

    # Exit in correct region (convergent forks + greedy filter handle anti-greedy)
    exit_candidates = [p for p in correct_reachable if p != correct_entry]
    if not exit_candidates:
        return None

    exit_pos = rng.choice(exit_candidates)

    # Correct region must have enough tiles for an interesting path
    if len(correct_reachable) < 3:
        return None

    # --- 5. Left side: start and key placement ---
    left_floor = {
        Pos(r, c) for r in range(height) for c in range(barrier_col)
        if Pos(r, c) not in walls
    }
    if not left_floor or not _is_connected(left_floor):
        return None

    # Both doors must be reachable from left side (agent needs to stand next to door)
    trap_door_left = Pos(trap_row, barrier_col - 1)
    correct_door_left = Pos(correct_row, barrier_col - 1)
    if trap_door_left not in left_floor or correct_door_left not in left_floor:
        return None

    # Key placement: convergent forks — key near trap door approach on left side
    key_ideal = Pos(trap_row, barrier_col - 1)
    if key_ideal not in left_floor:
        return None
    key_nearby = [
        p for p in left_floor
        if p != key_ideal
        and abs(p.row - key_ideal.row) <= 1
        and abs(p.col - key_ideal.col) <= 1
    ]
    if not key_nearby or rng.random() < 0.5:
        key_pos = key_ideal
    else:
        key_pos = rng.choice(key_nearby)

    # Start placement: different from key, on left side
    start_candidates = [p for p in left_floor if p != key_pos]
    if not start_candidates:
        return None
    start = rng.choice(start_candidates)

    # Verify connectivity on left side
    left_component = _flood_fill(start, left_floor)
    if key_pos not in left_component:
        return None
    if trap_door_left not in left_component or correct_door_left not in left_component:
        return None

    # --- 6. Optional random walls for variety (left side only) ---
    protected = {start, key_pos, trap_door_left, correct_door_left}
    extra_candidates = [p for p in left_floor if p not in protected]
    rng.shuffle(extra_candidates)
    n_extra = rng.randint(0, min(2, len(extra_candidates) // 3))

    for pos in extra_candidates:
        if n_extra <= 0:
            break
        remaining = left_floor - {pos}
        if (remaining and _is_connected(remaining) and
                all(p in remaining for p in protected)):
            walls.add(pos)
            left_floor.discard(pos)
            n_extra -= 1

    # --- P2.2: Content-informed fork — place decoy red key in trap zone ---
    # Both zones look populated; model must check which has the exit.
    content_informed = width >= 6 and rng.random() < 0.5
    decoy_key_pos: Optional[Pos] = None
    if content_informed:
        decoy_candidates = [p for p in trap_reachable if p != trap_entry]
        if decoy_candidates:
            decoy_key_pos = rng.choice(decoy_candidates)
        else:
            content_informed = False

    # --- 7. Build grid ---
    keys = {
        "K1": KeyInfo(id="K1", pos=key_pos, color="red"),
    }
    if content_informed and decoy_key_pos is not None:
        keys["K_decoy"] = KeyInfo(id="K_decoy", pos=decoy_key_pos, color="red")

    doors = {
        "D_c": DoorInfo(id="D_c", pos=correct_door_pos, required_keys=("red",)),
        "D_t": DoorInfo(id="D_t", pos=trap_door_pos, required_keys=("red",)),
    }
    if extra_trap_pos is not None:
        doors["D_t2"] = DoorInfo(
            id="D_t2", pos=extra_trap_pos, required_keys=("red",))

    grid = ChainVaultGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )

    all_key_ids = frozenset(keys.keys())
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=all_key_ids,
    )

    return grid, initial_state


# ---------------------------------------------------------------------------
# Anti-greedy key-behind-key grid generation
# ---------------------------------------------------------------------------

def _generate_kbk_depth3(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a depth-3 key-behind-key puzzle.

    CHAIN: K_r -> D_correct_r -> K_g -> D_correct_g -> K_b -> D_blue -> exit

    LAYOUT:
        Zone L (cols 0..b1-1)   : start, K_r
        Barrier 1 (col b1)      : D_trap_r (edge row) + D_correct_r (far row)
        Zone M (cols b1+1..b2-1): horizontal divider_m splits it into:
                                    M_trap (trap side, accessed via D_trap_r)
                                    M_work (correct side, K_g lives here)
        Barrier 2 (col b2)      : D_trap_g + D_correct_g (both in M_work rows)
        Zone R (cols b2+1..w-1) : horizontal divider_r2 splits it into:
                                    R_trap (trap side, dead-end, ≥2 cells)
                                    R_work (correct side, K_b + D_blue + exit)

    DEADLOCKS:
        1. K_r on D_trap_r  -> enters M_trap, K_g unreachable
        2. K_g on D_trap_g  -> enters R_trap, K_b unreachable

    Requirements: width >= 7 (Zone R needs ≥3 cols: b2 ≤ width-4).

    P2.4 CYCLIC VARIANT (stochastic at w>=7, h>=7):
        D_blue becomes (red+blue) dual-lock, K_r2 added in Zone R_work.
        Chain: R → G → (R₂+B) → exit. Agent must recognize that a second
        red key is needed despite having already used one red earlier.
    """
    if width < 7 or height < 5:
        return None

    # P2.4: stochastic cyclic variant
    cyclic = width >= 7 and height >= 7 and rng.random() < 0.5

    # --- 1. Barrier columns ---
    # b2 <= width-4 (Zone R = width-b2-1 >= 3 cols for exit pocket)
    # b2 >= b1+2 (Zone M = b2-b1-1 >= 1 col)
    # b1 >= 1 (Zone L >= 1 col)
    b1_max = width - 6   # need b2 = b1+2..width-4, so b1 <= width-6
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)

    b2_min = b1 + 2
    b2_max = width - 4
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Barrier 1 rows: trap at edge, correct far away ---
    trap_row = rng.choice([0, height - 1])
    if trap_row == 0:
        correct_min = max(3, height // 2)
        correct_row = rng.randint(correct_min, height - 1)
    else:
        correct_max = min(height - 4, height // 2)
        correct_row = rng.randint(0, max(0, correct_max))

    if abs(trap_row - correct_row) < 3:
        return None

    # --- 3. Zone M divider: horizontal wall between trap_row and correct_row ---
    # Spans only cols b1+1..b2-1 (not full width).
    lo, hi = min(trap_row, correct_row), max(trap_row, correct_row)
    divider_m_options = list(range(lo + 1, hi))
    if not divider_m_options:
        return None
    divider_m_row = rng.choice(divider_m_options)

    # M_trap rows: same side as trap_row; M_work rows: same side as correct_row
    if trap_row < correct_row:
        m_trap_rows = set(range(trap_row, divider_m_row))         # 0..divider_m-1
        m_work_rows = set(range(divider_m_row + 1, height))       # divider_m+1..h-1
    else:
        m_work_rows = set(range(0, divider_m_row))                # 0..divider_m-1
        m_trap_rows = set(range(divider_m_row + 1, height))       # divider_m+1..h-1

    if len(m_trap_rows) < 2 or len(m_work_rows) < 3:
        return None

    # correct_row must be in m_work_rows (agent enters M_work via D_correct_r)
    if correct_row not in m_work_rows:
        return None

    # --- 4. Barrier 2 door rows: both must be in M_work rows ---
    # b2_trap_row near correct_row (convergent forks bait)
    b2_trap_row = correct_row   # agent exits barrier 1 at correct_row → sees D_trap_g

    # b2_correct_row: far from b2_trap_row within M_work rows (need divider between them)
    m_work_far = sorted(r for r in m_work_rows if abs(r - b2_trap_row) >= 2)
    if not m_work_far:
        return None
    b2_correct_row = rng.choice(m_work_far)

    # --- 5. Zone R divider: between b2_trap_row and b2_correct_row ---
    # Spans only cols b2+1..width-1.
    lo2, hi2 = min(b2_trap_row, b2_correct_row), max(b2_trap_row, b2_correct_row)
    divider_r2_options = list(range(lo2 + 1, hi2))
    if not divider_r2_options:
        return None
    divider_r2_row = rng.choice(divider_r2_options)

    if b2_trap_row < b2_correct_row:
        r_trap_rows = set(range(b2_trap_row, divider_r2_row))
        r_work_rows = set(range(divider_r2_row + 1, height))
    else:
        r_work_rows = set(range(0, divider_r2_row))
        r_trap_rows = set(range(divider_r2_row + 1, height))

    # --- 6. Exit pocket in R_work far corner ---
    if b2_correct_row < b2_trap_row:
        # R_work is top rows → exit at top-right corner
        exit_row = min(r_work_rows)
        pocket_neighbor_row = exit_row + 1
    else:
        exit_row = max(r_work_rows)
        pocket_neighbor_row = exit_row - 1

    exit_pos = Pos(exit_row, width - 1)
    blue_door_pos = Pos(exit_row, width - 2)
    pocket_wall_pos = Pos(pocket_neighbor_row, width - 1)

    # --- 7. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1 (col b1): wall at every row except trap_row and correct_row
    for r in range(height):
        if r != trap_row and r != correct_row:
            walls.add(Pos(r, b1))

    # Zone M divider (row divider_m_row, cols b1+1..b2-1)
    for c in range(b1 + 1, b2):
        walls.add(Pos(divider_m_row, c))

    # Barrier 2 (col b2): wall at every row except b2_trap_row and b2_correct_row
    for r in range(height):
        if r != b2_trap_row and r != b2_correct_row:
            walls.add(Pos(r, b2))

    # Zone R divider (row divider_r2_row, cols b2+1..width-1)
    for c in range(b2 + 1, width):
        walls.add(Pos(divider_r2_row, c))

    # Exit pocket wall
    walls.add(pocket_wall_pos)

    # --- 8. Verify Zone M connectivity ---
    m_work_floor = {
        Pos(r, c)
        for r in m_work_rows
        for c in range(b1 + 1, b2)
        if Pos(r, c) not in walls
    }
    correct_entry_m = Pos(correct_row, b1 + 1)
    if correct_entry_m not in m_work_floor:
        return None
    if not _is_connected(m_work_floor):
        return None
    if len(m_work_floor) < 2:
        return None

    m_trap_floor = {
        Pos(r, c)
        for r in m_trap_rows
        for c in range(b1 + 1, b2)
        if Pos(r, c) not in walls
    }
    trap_entry_m = Pos(trap_row, b1 + 1)
    if trap_entry_m not in m_trap_floor:
        return None
    if len(m_trap_floor) < 2:
        return None
    if m_trap_floor & m_work_floor:
        return None

    # b2_trap_row and b2_correct_row must be reachable in M_work
    if (Pos(b2_trap_row, b2 - 1) not in m_work_floor
            or Pos(b2_correct_row, b2 - 1) not in m_work_floor):
        return None

    # --- 9. Verify Zone R connectivity ---
    r_work_floor = {
        Pos(r, c)
        for r in r_work_rows
        for c in range(b2 + 1, width)
        if Pos(r, c) not in walls
           and Pos(r, c) != blue_door_pos
           and Pos(r, c) != exit_pos
    }
    correct_entry_r = Pos(b2_correct_row, b2 + 1)
    if correct_entry_r not in r_work_floor:
        return None
    if not _is_connected(r_work_floor):
        return None
    if len(r_work_floor) < 2:
        return None

    r_trap_floor = {
        Pos(r, c)
        for r in r_trap_rows
        for c in range(b2 + 1, width)
        if Pos(r, c) not in walls
    }
    trap_entry_r = Pos(b2_trap_row, b2 + 1)
    if trap_entry_r not in r_trap_floor:
        return None
    if len(r_trap_floor) < 2:
        return None
    if r_trap_floor & r_work_floor:
        return None

    # Blue door reachable from R_work and adjacent to exit
    if not any(add_pos(blue_door_pos, d) in r_work_floor for d in Direction):
        return None
    if not any(add_pos(blue_door_pos, d) == exit_pos for d in Direction):
        return None

    # --- 10. Place K_b (and K_r2 if cyclic) in Zone R_work ---
    bk_candidates = [p for p in r_work_floor if p != correct_entry_r]
    if not bk_candidates:
        return None
    blue_key_pos = rng.choice(bk_candidates)

    # P2.4: place second red key for cyclic variant
    red2_key_pos: Optional[Pos] = None
    if cyclic:
        r2_candidates = [p for p in r_work_floor
                         if p != correct_entry_r and p != blue_key_pos]
        if r2_candidates:
            red2_key_pos = rng.choice(r2_candidates)
        else:
            cyclic = False  # fall back to standard

    # --- 11. Place K_g in Zone M_work (convergent forks bait near D_trap_g) ---
    # Ideal: cell just left of D_trap_g so agent sees trap door immediately on entry
    kg_ideal = Pos(b2_trap_row, b2 - 1)
    kg_nearby = [
        p for p in m_work_floor
        if abs(p.row - kg_ideal.row) <= 1 and abs(p.col - kg_ideal.col) <= 1
    ]
    if kg_ideal in m_work_floor and (not kg_nearby or rng.random() < 0.6):
        green_key_pos = kg_ideal
    elif kg_nearby:
        green_key_pos = rng.choice(kg_nearby)
    else:
        green_key_pos = rng.choice(list(m_work_floor))

    # --- 12. Zone L: start and K_r ---
    left_floor = {
        Pos(r, c)
        for r in range(height)
        for c in range(b1)
        if Pos(r, c) not in walls
    }
    if not left_floor or not _is_connected(left_floor):
        return None

    trap_door_left = Pos(trap_row, b1 - 1)
    correct_door_left = Pos(correct_row, b1 - 1)
    if trap_door_left not in left_floor or correct_door_left not in left_floor:
        return None

    # K_r: convergent forks — near trap door approach
    rk_ideal = Pos(trap_row, b1 - 1)
    if rk_ideal not in left_floor:
        return None
    rk_nearby = [
        p for p in left_floor
        if p != rk_ideal
        and abs(p.row - rk_ideal.row) <= 1
        and abs(p.col - rk_ideal.col) <= 1
    ]
    if not rk_nearby or rng.random() < 0.5:
        red_key_pos = rk_ideal
    else:
        red_key_pos = rng.choice(rk_nearby)

    start_candidates = [p for p in left_floor if p != red_key_pos]
    if not start_candidates:
        return None
    start = rng.choice(start_candidates)

    left_comp = _flood_fill(start, left_floor)
    if (red_key_pos not in left_comp or trap_door_left not in left_comp
            or correct_door_left not in left_comp):
        return None

    # --- 13. Optional extra walls (Zone L and Zone M_work) ---
    protected_l = {start, red_key_pos, trap_door_left, correct_door_left}
    extras_l = [p for p in left_floor if p not in protected_l]
    rng.shuffle(extras_l)
    n_extra = rng.randint(0, min(2, len(extras_l) // 3))
    for pos in extras_l:
        if n_extra <= 0:
            break
        rem = left_floor - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in protected_l):
            walls.add(pos)
            left_floor.discard(pos)
            n_extra -= 1

    protected_m = {
        correct_entry_m, green_key_pos,
        Pos(b2_trap_row, b2 - 1), Pos(b2_correct_row, b2 - 1),
    } & m_work_floor
    extras_m = [p for p in m_work_floor if p not in protected_m]
    rng.shuffle(extras_m)
    n_extra_m = rng.randint(0, min(1, len(extras_m) // 3))
    for pos in extras_m:
        if n_extra_m <= 0:
            break
        rem = m_work_floor - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in protected_m):
            walls.add(pos)
            m_work_floor.discard(pos)
            n_extra_m -= 1

    # --- 14. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=red_key_pos,   color="red"),
        "K_g": KeyInfo(id="K_g", pos=green_key_pos, color="green"),
        "K_b": KeyInfo(id="K_b", pos=blue_key_pos,  color="blue"),
    }
    if cyclic and red2_key_pos is not None:
        keys["K_r2"] = KeyInfo(id="K_r2", pos=red2_key_pos, color="red")

    # D_blue becomes red+blue dual-lock in cyclic variant
    exit_door_keys = ("red", "blue") if cyclic else ("blue",)
    doors = {
        "D_trap_r":   DoorInfo(id="D_trap_r",   pos=Pos(trap_row, b1),      required_keys=("red",)),
        "D_correct_r": DoorInfo(id="D_correct_r", pos=Pos(correct_row, b1),  required_keys=("red",)),
        "D_trap_g":   DoorInfo(id="D_trap_g",   pos=Pos(b2_trap_row, b2),   required_keys=("green",)),
        "D_correct_g": DoorInfo(id="D_correct_g", pos=Pos(b2_correct_row, b2), required_keys=("green",)),
        "D_blue":     DoorInfo(id="D_blue",     pos=blue_door_pos,           required_keys=exit_door_keys),
    }
    grid = ChainVaultGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def generate_key_behind_key_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a key-behind-key chain puzzle.

    Depth scales with grid size:
      - width < 7:  depth-2 (one barrier, red -> blue -> exit)
      - width == 7: depth-3 50 % of the time, depth-2 otherwise
      - width >= 8: depth-3 (two barriers, red -> green -> blue -> exit)

    DEPTH-2 CHAIN: K_r -> D_correct_r -> K_b -> D_blue -> exit
    DEPTH-3 CHAIN: K_r -> D_correct_r -> K_g -> D_correct_g -> K_b -> D_blue -> exit

    DEPTH-2 LAYOUT (zone divider + exit pocket):

        Zone L (cols 0..b-1)  : start, K_r (convergent forks bait)
        Barrier (col b)       : D_trap_r (edge row) + D_correct_r (far row)
        Zone R (cols b+1..w-1): horizontal divider splits into:
                                  R_trap  (dead-end, ≥2 cells)
                                  R_work  (K_b + D_blue + exit pocket)

    DEPTH-3 LAYOUT: see _generate_kbk_depth3.

    DEADLOCKS (depth-2): waste K_r on D_trap_r -> K_b permanently unreachable.
    DEADLOCKS (depth-3): same at barrier 1 + wasting K_g on D_trap_g.
    """
    if rng is None:
        rng = random.Random()
    if height < 5 or width < 5:
        return None

    # --- Depth selection ---
    use_depth3 = (width >= 8) or (width == 7 and rng.random() < 0.5)
    if use_depth3:
        result = _generate_kbk_depth3(width, height, rng)
        if result is not None:
            return result
        # Fall through to depth-2 if depth-3 fails (e.g. at constrained sizes)

    # ================================================================
    # DEPTH-2 implementation (original)
    # ================================================================

    # --- 1. Barrier column ---
    # Need ≥1 col on left, ≥3 cols on right (trap pocket + blue key zone +
    # exit column). So barrier_col ∈ [1, width-4].
    bc_max = width - 4
    if bc_max < 1:
        return None
    barrier_col = rng.randint(1, bc_max)

    # --- 2. Red door placement: trap at edge, correct far away ---
    trap_row = rng.choice([0, height - 1])
    if trap_row == 0:
        correct_min = max(3, height // 2)
        correct_row = rng.randint(correct_min, height - 1)
    else:
        correct_max = min(height - 4, height // 2)
        correct_row = rng.randint(0, max(0, correct_max))

    if abs(trap_row - correct_row) < 3:
        return None

    trap_door_pos = Pos(trap_row, barrier_col)
    correct_door_pos = Pos(correct_row, barrier_col)

    # Exit pocket at far corner of Zone R_correct (opposite trap_row)
    if trap_row == 0:
        exit_pos = Pos(height - 1, width - 1)
        blue_door_pos = Pos(height - 1, width - 2)
        pocket_wall_pos = Pos(height - 2, width - 1)
    else:
        exit_pos = Pos(0, width - 1)
        blue_door_pos = Pos(0, width - 2)
        pocket_wall_pos = Pos(1, width - 1)

    # --- 3. Build walls ---
    walls: set[Pos] = set()

    # Barrier column (except door rows)
    for r in range(height):
        if r != trap_row and r != correct_row:
            walls.add(Pos(r, barrier_col))

    # Zone divider: horizontal wall on right side between trap and correct rows
    lo, hi = min(trap_row, correct_row), max(trap_row, correct_row)
    divider_options = list(range(lo + 1, hi))
    if not divider_options:
        return None
    divider_row = rng.choice(divider_options)
    for c in range(barrier_col + 1, width):
        walls.add(Pos(divider_row, c))

    # Exit pocket wall: isolates exit from Zone R_correct
    walls.add(pocket_wall_pos)

    # --- Scaling: extra trap door on vertical barrier at w>=6 ---
    extra_trap_pos_kbk: Optional[Pos] = None
    if width >= 6:
        # Find barrier wall rows in the trap zone (between trap_row and divider)
        if trap_row < divider_row:
            kbk_trap_zone_rows = list(range(trap_row + 1, divider_row))
        else:
            kbk_trap_zone_rows = list(range(divider_row + 1, trap_row))
        kbk_wall_rows = [
            r for r in kbk_trap_zone_rows
            if Pos(r, barrier_col) in walls
        ]
        if kbk_wall_rows:
            er = rng.choice(kbk_wall_rows)
            extra_trap_pos_kbk = Pos(er, barrier_col)
            walls.discard(extra_trap_pos_kbk)

    trap_entry = Pos(trap_row, barrier_col + 1)
    correct_entry = Pos(correct_row, barrier_col + 1)

    # --- 4. Verify right-side regions ---
    # Exclude blue_door_pos and exit_pos from the floor set so the flood fill
    # doesn't leak through the blue door into the exit pocket.
    all_right = {
        Pos(r, c) for r in range(height) for c in range(barrier_col + 1, width)
        if Pos(r, c) not in walls
           and Pos(r, c) != blue_door_pos
           and Pos(r, c) != exit_pos
    }
    if trap_entry not in all_right or correct_entry not in all_right:
        return None

    trap_reachable = _flood_fill(trap_entry, all_right)
    correct_reachable = _flood_fill(correct_entry, all_right)

    # Trap must be isolated from correct region
    if trap_reachable & correct_reachable:
        return None

    # Trap must be a real walkable area (>=2 cells)
    if len(trap_reachable) < 2:
        return None

    # Correct region needs room for blue key
    if len(correct_reachable) < 3:
        return None

    # --- 5. Anti-greedy: convergent forks key placement + post-hoc greedy filter ---
    # (no hard manhattan check — greedy filter handles it)

    # --- 6. Verify blue door is adjacent to correct_reachable ---
    # Agent must be able to stand next to the blue door to USE_KEY.
    if not any(add_pos(blue_door_pos, d) in correct_reachable for d in Direction):
        return None

    # Verify blue door is adjacent to exit (agent walks through to exit)
    if not any(add_pos(blue_door_pos, d) == exit_pos for d in Direction):
        return None

    # --- 7. Blue key in correct_reachable ---
    bk_candidates = [p for p in correct_reachable if p != correct_entry]
    if not bk_candidates:
        return None
    blue_key_pos = rng.choice(bk_candidates)

    # --- 8. Left side: start and red key ---
    left_floor = {
        Pos(r, c) for r in range(height) for c in range(barrier_col)
        if Pos(r, c) not in walls
    }
    if not left_floor or not _is_connected(left_floor):
        return None

    trap_door_left = Pos(trap_row, barrier_col - 1)
    correct_door_left = Pos(correct_row, barrier_col - 1)
    if trap_door_left not in left_floor or correct_door_left not in left_floor:
        return None

    # Red key: convergent forks — key near trap door approach on left side
    rk_ideal = Pos(trap_row, barrier_col - 1)
    if rk_ideal not in left_floor:
        return None
    rk_nearby = [
        p for p in left_floor
        if p != rk_ideal
        and abs(p.row - rk_ideal.row) <= 1
        and abs(p.col - rk_ideal.col) <= 1
    ]
    if not rk_nearby or rng.random() < 0.5:
        red_key_pos = rk_ideal
    else:
        red_key_pos = rng.choice(rk_nearby)

    # Start: different from red key
    start_candidates = [p for p in left_floor if p != red_key_pos]
    if not start_candidates:
        return None
    start = rng.choice(start_candidates)

    left_comp = _flood_fill(start, left_floor)
    if (red_key_pos not in left_comp or trap_door_left not in left_comp
            or correct_door_left not in left_comp):
        return None

    # --- 9. Optional extra walls (left side) ---
    protected = {start, red_key_pos, trap_door_left, correct_door_left}
    extras = [p for p in left_floor if p not in protected]
    rng.shuffle(extras)
    n_extra = rng.randint(0, min(2, len(extras) // 3))
    for pos in extras:
        if n_extra <= 0:
            break
        rem = left_floor - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in protected):
            walls.add(pos)
            left_floor.discard(pos)
            n_extra -= 1

    # --- Scaling (w>=7): blue trap on divider wall (crosses into trap zone) ---
    extra_blue_trap_pos: Optional[Pos] = None
    if height >= 7:
        # Open a divider wall cell as a blue door (wastes blue key → trap zone)
        divider_wall_cols = [
            c for c in range(barrier_col + 1, width)
            if Pos(divider_row, c) in walls
        ]
        if divider_wall_cols:
            ebc = rng.choice(divider_wall_cols)
            extra_blue_trap_pos = Pos(divider_row, ebc)
            walls.discard(extra_blue_trap_pos)

    # --- 10. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=red_key_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=blue_key_pos, color="blue"),
    }
    doors = {
        "D_trap": DoorInfo(id="D_trap", pos=trap_door_pos, required_keys=("red",)),
        "D_correct": DoorInfo(id="D_correct", pos=correct_door_pos, required_keys=("red",)),
        "D_blue": DoorInfo(id="D_blue", pos=blue_door_pos, required_keys=("blue",)),
    }
    if extra_trap_pos_kbk is not None:
        doors["D_trap2"] = DoorInfo(
            id="D_trap2", pos=extra_trap_pos_kbk, required_keys=("red",))
    if extra_blue_trap_pos is not None:
        doors["D_trap_b"] = DoorInfo(
            id="D_trap_b", pos=extra_blue_trap_pos, required_keys=("blue",))
    grid = ChainVaultGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Budget Partitioning grid generation (replaces Conservation Trap T2)
# ---------------------------------------------------------------------------

def _generate_bp_cross_color(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a cross-color Budget Partitioning puzzle (P2.3).

    MECHANIC: 2 RED keys + 2 BLUE keys.  Five doors total:
      D_red  (RED, correct on barrier 1) → Zone 1_upper
      D_trap (RED, trap on barrier 1) → Zone 1T (dead end)
      D_blue (BLUE, mandatory on barrier_blue) → Zone 1_lower
      D_C    (RED+BLUE dual-lock on barrier 2) → exit zone
      D_trap_b (BLUE, trap on barrier_blue) → Zone 1T_lower (dead end)

    BUDGET: 1R(D_red) + 1B(D_blue) + 1R+1B(D_C) = 2R + 2B exactly.
    RED TRAP: D_trap wastes 1R → can't afford D_red + D_C.
    BLUE TRAP: D_trap_b wastes 1B → can't afford D_blue + D_C.

    LAYOUT (3 horizontal barriers, needs h>=7, w>=7):
      Zone 0        (rows 0..b1-1)       : start, K_r1, K_r2, K_b1, K_b2
      Barrier 1     (row b1)             : D_red + D_trap + walls
      Zone 1_upper  (rows b1+1..b_blue-1): via D_red → approach to D_blue
      Zone 1T       (rows b1+1..b_blue-1): via D_trap → dead end (vertical divider)
      Barrier_blue  (row b_blue)         : D_blue (correct side) + D_trap_b (trap side)
      Zone 1_lower  (rows b_blue+1..b2-1): via D_blue → approach to D_C
      Barrier 2     (row b2)             : D_C (red+blue dual-lock)
      Zone 2        (rows b2+1..h-1)     : exit
    """
    w, h = width, height
    if h < 7 or w < 7:
        return None

    # --- 1. Three barrier rows ---
    # Need b1 >= 1, b_blue >= b1+2, b2 >= b_blue+2, zone2 needs b2 <= h-2
    b1 = rng.randint(1, h - 6)               # leave room for 3 more rows + zones
    b_blue = rng.randint(b1 + 2, h - 4)      # at least 1 zone row between b1 and b_blue
    b2 = rng.randint(b_blue + 2, h - 2)      # at least 1 zone row between b_blue and b2

    # --- 2. Vertical divider ---
    v1 = rng.randint(2, w - 3)  # interior column, at least 2 cols each side

    trap_right = rng.choice([True, False])
    if trap_right:
        correct_cols = list(range(0, v1))
        trap_cols = list(range(v1 + 1, w))
    else:
        trap_cols = list(range(0, v1))
        correct_cols = list(range(v1 + 1, w))

    if len(correct_cols) < 2 or len(trap_cols) < 1:
        return None

    # --- 3. Door column selection ---
    c_red = rng.choice(correct_cols)    # D_red (correct, barrier 1)
    c_trap_r = rng.choice(trap_cols)    # D_trap (red trap, barrier 1)
    c_blue = rng.choice(correct_cols)   # D_blue (correct, barrier_blue)
    c_trap_b = rng.choice(trap_cols)    # D_trap_b (blue trap, barrier_blue)
    c_C = rng.choice(correct_cols)      # D_C (dual-lock, barrier 2)

    # --- 4. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1
    for c in range(w):
        if c != c_red and c != c_trap_r:
            walls.add(Pos(b1, c))

    # Vertical divider (rows b1+1..b2-1)
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, v1))

    # Barrier_blue
    for c in range(w):
        if c != c_blue and c != c_trap_b:
            walls.add(Pos(b_blue, c))

    # Barrier 2
    for c in range(w):
        if c != c_C:
            walls.add(Pos(b2, c))

    # --- 5. Zone connectivity checks ---

    # Zone 0
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    d_red_above = Pos(b1 - 1, c_red)
    d_trap_above = Pos(b1 - 1, c_trap_r)
    if len(zone0) < 5 or d_red_above not in zone0 or d_trap_above not in zone0:
        return None
    if not _is_connected(zone0):
        return None

    # Zone 1_upper_correct (rows b1+1..b_blue-1, correct_cols)
    z1uc = {
        Pos(r, c) for r in range(b1 + 1, b_blue)
        for c in correct_cols if Pos(r, c) not in walls
    }
    d_red_below = Pos(b1 + 1, c_red)
    d_blue_above = Pos(b_blue - 1, c_blue)
    if not z1uc or d_red_below not in z1uc or d_blue_above not in z1uc:
        return None
    z1uc_comp = _flood_fill(d_red_below, z1uc)
    if d_blue_above not in z1uc_comp:
        return None

    # Zone 1T_upper (rows b1+1..b_blue-1, trap_cols) — dead end
    z1t_upper = {
        Pos(r, c) for r in range(b1 + 1, b_blue)
        for c in trap_cols if Pos(r, c) not in walls
    }
    d_trap_below = Pos(b1 + 1, c_trap_r)
    if not z1t_upper or d_trap_below not in z1t_upper:
        return None
    # Must be isolated from correct zone
    if z1uc & z1t_upper:
        return None

    # Zone 1_lower_correct (rows b_blue+1..b2-1, correct_cols)
    z1lc = {
        Pos(r, c) for r in range(b_blue + 1, b2)
        for c in correct_cols if Pos(r, c) not in walls
    }
    d_blue_below = Pos(b_blue + 1, c_blue)
    d_C_above = Pos(b2 - 1, c_C)
    if not z1lc or d_blue_below not in z1lc or d_C_above not in z1lc:
        return None
    z1lc_comp = _flood_fill(d_blue_below, z1lc)
    if d_C_above not in z1lc_comp:
        return None

    # Zone 1T_lower (rows b_blue+1..b2-1, trap_cols) — dead end
    z1t_lower = {
        Pos(r, c) for r in range(b_blue + 1, b2)
        for c in trap_cols if Pos(r, c) not in walls
    }
    d_trap_b_below = Pos(b_blue + 1, c_trap_b)
    if not z1t_lower or d_trap_b_below not in z1t_lower:
        return None
    if z1lc & z1t_lower:
        return None

    # Zone 2 (exit zone)
    zone2 = {
        Pos(r, c) for r in range(b2 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    d_C_below = Pos(b2 + 1, c_C)
    if not zone2 or d_C_below not in zone2:
        return None
    zone2_comp = _flood_fill(d_C_below, zone2)

    # --- 6. Exit placement ---
    exit_candidates = [p for p in zone2_comp if p != d_C_below]
    if not exit_candidates:
        exit_candidates = list(zone2_comp)
    exit_pos = rng.choice(exit_candidates)

    # --- 7. Key + start placement in Zone 0 ---
    zone0_list = list(zone0)
    if len(zone0_list) < 5:
        return None
    rng.shuffle(zone0_list)
    start = zone0_list[0]
    key_candidates = [p for p in zone0_list if p != start]
    if len(key_candidates) < 4:
        return None
    kr1_pos, kr2_pos, kb1_pos, kb2_pos = key_candidates[:4]

    # Verify all are reachable from start
    zone0_comp = _flood_fill(start, zone0)
    needed = {kr1_pos, kr2_pos, kb1_pos, kb2_pos, d_red_above, d_trap_above}
    if not needed.issubset(zone0_comp):
        return None

    # --- 8. Anti-greedy: trap doors closer to exit side ---
    exit_row = exit_pos.row
    if not _trap_closer_to_exit(
        Pos(b1, c_trap_r), Pos(b1, c_red), exit_pos
    ):
        return None
    if not _trap_closer_to_exit(
        Pos(b_blue, c_trap_b), Pos(b_blue, c_blue), exit_pos
    ):
        return None

    # --- 9. Build grid ---
    keys = {
        "K_r1": KeyInfo(id="K_r1", pos=kr1_pos, color="red"),
        "K_r2": KeyInfo(id="K_r2", pos=kr2_pos, color="red"),
        "K_b1": KeyInfo(id="K_b1", pos=kb1_pos, color="blue"),
        "K_b2": KeyInfo(id="K_b2", pos=kb2_pos, color="blue"),
    }
    doors = {
        "D_red": DoorInfo(
            id="D_red", pos=Pos(b1, c_red), required_keys=("red",)),
        "D_trap_r": DoorInfo(
            id="D_trap_r", pos=Pos(b1, c_trap_r), required_keys=("red",)),
        "D_blue": DoorInfo(
            id="D_blue", pos=Pos(b_blue, c_blue), required_keys=("blue",)),
        "D_trap_b": DoorInfo(
            id="D_trap_b", pos=Pos(b_blue, c_trap_b), required_keys=("blue",)),
        "D_C": DoorInfo(
            id="D_C", pos=Pos(b2, c_C), required_keys=("red", "blue")),
    }
    grid = ChainVaultGrid(
        width=w, height=h,
        walls=frozenset(walls), keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def generate_budget_partitioning_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a Budget Partitioning puzzle (replaces Conservation Trap T2).

    MECHANIC: 2 RED keys + 1 BLUE key. Three doors require RED:
      D_A  (RED single-lock, correct) → Zone 1_correct (contains BLUE key)
      D_B  (RED single-lock, trap)    → Zone 1T (dead end, no keys)
      D_C  (RED+BLUE dual-lock)       → exit

    Exactly 2 RED keys required: one for D_A + one for D_C's RED component.
    Spending any RED on D_B leaves the agent one RED short for A+C → stuck.

    KEY DISTINCTION FROM CONSERVATION TRAP:
    - No bypass gap on Barrier 1. Agent must USE_KEY to enter Zone 1.
    - Two RED doors on Barrier 1 force explicit budget allocation.
    - BLUE key is hidden in Zone 1_correct (not freely available in Zone 0).
    - Tests whether the model counts: "I have 2 reds, I need 1 for A and 1 for C,
      so I cannot spend any on B even though B looks like the exit path."

    LAYOUT:
        Zone 0 (rows 0..b1-1)   : start + 2x RED keys
        Barrier 1 (row b1)       : D_A (far side) + D_B (exit side trap) + walls
        Zone 1 (rows b1+1..b2-1) : vertical divider (col v1) separates:
                                     Zone 1_correct (D_A side): BLUE key + D_C access
                                     Zone 1T         (D_B side): dead end, no keys
        Barrier 2 (row b2)       : D_C (RED+BLUE, in Zone 1_correct) + walls
        Zone 2 (rows b2+1..h-1)  : exit (biased toward Zone 1T side)

    CORRECT PATH:
        pick up 2 RED → D_A (1 RED) → pick up BLUE → D_C (1 RED + BLUE) → exit
    TRAP PATH:
        D_B (1 RED) → Zone 1T dead end → backtrack → D_A (1 RED) → BLUE
        → 0 RED + 1 BLUE → cannot open D_C → STUCK

    DEADLOCKS: ≥2 (waste RED on D_B with 0 BLUE; waste RED on D_B holding BLUE).
    """
    if rng is None:
        rng = random.Random()
    if height < 5 or width < 5:
        return None

    # --- Stochastic cross-color variant (P2.3) ---
    if width >= 7 and height >= 7 and rng.random() < 0.5:
        result = _generate_bp_cross_color(width, height, rng)
        if result is not None:
            return result

    # --- 1. Barrier rows (same structure as Conservation Trap) ---
    b1_max = height - 4
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min = b1 + 2
    b2_max_val = height - 2
    if b2_min > b2_max_val:
        return None
    b2 = rng.randint(b2_min, b2_max_val)

    # --- 2. Vertical divider (col v1) and Zone assignment ---
    # Zone 1T is the "exit side" (where the greedy agent heads).
    # Zone 1_correct is the "far side" (where D_A and D_C live).
    v1 = rng.randint(1, width - 2)   # divider col: always interior

    # Randomly assign which side is Zone 1T (left or right of v1).
    # trap_right=True → Zone 1T on right (cols v1+1..w-1), Zone 1_correct on left.
    trap_right = rng.choice([True, False])
    if trap_right:
        correct_cols = list(range(0, v1))           # Zone 1_correct
        trap_cols    = list(range(v1 + 1, width))   # Zone 1T
    else:
        trap_cols    = list(range(0, v1))
        correct_cols = list(range(v1 + 1, width))

    if not correct_cols or not trap_cols:
        return None

    # --- 3. Door column selection ---
    # c_A (Door A, correct): far from exit → first half of correct_cols (far from trap)
    n_far = max(1, len(correct_cols) // 2)
    c_A = rng.choice(correct_cols[:n_far])

    # c_B (Door B, trap): on the exit side → anywhere in trap_cols
    c_B = rng.choice(trap_cols)

    # c_C (Door C, dual-lock): in correct_cols (agent approaches from Zone 1_correct)
    c_C = rng.choice(correct_cols)

    # --- 4. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1 (row b1): wall at every col except c_A (D_A) and c_B (D_B)
    for c in range(width):
        if c != c_A and c != c_B:
            walls.add(Pos(b1, c))

    # Vertical divider (col v1, rows b1+1..b2-1): separates Zone 1_correct/Zone 1T
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, v1))

    # Barrier 2 (row b2): wall at every col except c_C (D_C)
    for c in range(width):
        if c != c_C:
            walls.add(Pos(b2, c))

    # --- Scaling: extra RED trap door on Barrier 1 at w>=6 ---
    extra_trap_cols: list[int] = []
    if width >= 6:
        used_b1: set[int] = {c_A, c_B}
        extra_candidates = [
            c for c in trap_cols
            if c not in used_b1 and Pos(b1, c) in walls
        ]
        if extra_candidates:
            c_T2 = rng.choice(extra_candidates)
            walls.discard(Pos(b1, c_T2))
            extra_trap_cols.append(c_T2)

    # --- 5. Verify zone connectivity ---

    # Zone 0 (rows 0..b1-1): must contain start, 2x RED, approach to both doors
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(width)
        if Pos(r, c) not in walls
    }
    da_above = Pos(b1 - 1, c_A)
    db_above = Pos(b1 - 1, c_B)
    if len(zone0) < 3 or da_above not in zone0 or db_above not in zone0:
        return None
    if not _is_connected(zone0):
        return None

    # Zone 1_correct (rows b1+1..b2-1, correct_cols): BLUE key + D_C approach
    z1c = {
        Pos(r, c)
        for r in range(b1 + 1, b2)
        for c in correct_cols
        if Pos(r, c) not in walls
    }
    da_below = Pos(b1 + 1, c_A)
    dc_above = Pos(b2 - 1, c_C)
    if not z1c or da_below not in z1c or dc_above not in z1c:
        return None
    z1c_comp = _flood_fill(da_below, z1c)
    if dc_above not in z1c_comp:
        return None

    # Zone 1T (rows b1+1..b2-1, trap_cols): dead end, no keys
    z1t = {
        Pos(r, c)
        for r in range(b1 + 1, b2)
        for c in trap_cols
        if Pos(r, c) not in walls
    }
    db_below = Pos(b1 + 1, c_B)
    if not z1t or db_below not in z1t:
        return None
    if z1t & z1c:   # zones must be isolated (no cross-contamination)
        return None

    # Zone 2 (rows b2+1..h-1): exit zone
    zone2 = {
        Pos(r, c) for r in range(b2 + 1, height) for c in range(width)
        if Pos(r, c) not in walls
    }
    dc_below = Pos(b2 + 1, c_C)
    if not zone2 or dc_below not in zone2:
        return None
    zone2_comp = _flood_fill(dc_below, zone2)

    # --- 6. Exit placement (biased toward Zone 1T / exit side) ---
    exit_candidates = [p for p in zone2_comp]
    exit_candidates.sort(key=lambda p: abs(p.col - c_B))
    n_pick = max(1, len(exit_candidates) // 2)
    exit_pos = rng.choice(exit_candidates[:n_pick])
    if exit_pos == dc_below and len(exit_candidates) > 1:
        # Avoid trivial placement directly below D_C
        exit_pos = rng.choice(exit_candidates[1:n_pick] or exit_candidates[1:])

    # --- 7. BLUE key in Zone 1_correct ---
    bk_candidates = [p for p in z1c_comp if p != da_below]
    if not bk_candidates:
        bk_candidates = list(z1c_comp)
    blue_key_pos = rng.choice(bk_candidates)

    # --- 8. Start + 2x RED keys in Zone 0 ---
    zone0_list = list(zone0)
    if len(zone0_list) < 3:
        return None
    rng.shuffle(zone0_list)
    start = zone0_list[0]

    key_candidates = [p for p in zone0_list if p != start]
    key_result = _select_key_positions(key_candidates, 2, rng, reference=exit_pos)
    if key_result is None:
        return None
    red1_pos, red2_pos = key_result

    zone0_comp = _flood_fill(start, zone0)
    if (red1_pos not in zone0_comp or red2_pos not in zone0_comp
            or da_above not in zone0_comp or db_above not in zone0_comp):
        return None

    # --- 9. Optional extra walls (Zone 0 only) ---
    protected = {start, red1_pos, red2_pos, da_above, db_above}
    extras = [p for p in zone0 if p not in protected]
    rng.shuffle(extras)
    n_extra = rng.randint(0, min(2, len(extras) // 4))
    for pos in extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    # --- 10. Build grid ---
    keys = {
        "K_r1": KeyInfo(id="K_r1", pos=red1_pos,    color="red"),
        "K_r2": KeyInfo(id="K_r2", pos=red2_pos,    color="red"),
        "K_b":  KeyInfo(id="K_b",  pos=blue_key_pos, color="blue"),
    }
    doors = {
        "D_A": DoorInfo(id="D_A", pos=Pos(b1, c_A), required_keys=("red",)),
        "D_B": DoorInfo(id="D_B", pos=Pos(b1, c_B), required_keys=("red",)),
        "D_C": DoorInfo(id="D_C", pos=Pos(b2, c_C), required_keys=("red", "blue")),
    }
    for i, tc in enumerate(extra_trap_cols):
        did = f"D_B{i + 2}"
        doors[did] = DoorInfo(id=did, pos=Pos(b1, tc), required_keys=("red",))

    grid = ChainVaultGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Anti-greedy conservation trap grid generation (legacy — not in registry)
# ---------------------------------------------------------------------------

def generate_conservation_trap_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a conservation trap grid with two horizontal barriers.

    MECHANIC: 2 keys (red + blue). Barrier 1 has a red single-lock trap door
    and an open bypass gap. Barrier 2 has a red+blue dual-lock as the only
    passage. Exit sits below barrier 2.

    Greedy agents pick up keys, head toward exit, encounter the trap door on
    barrier 1 (biased closer to exit than the bypass), waste the red key,
    walk through the open middle zone to the dual-lock, and discover they
    can't open it (red key already spent).

    Correct path: bypass around barrier 1, keep both keys, open dual-lock.

    LAYOUT:
        rows 0..b1-1    — upper zone: start, red key, blue key
        row b1          — barrier 1: walls + trap door (red) + bypass gap
        rows b1+1..b2-1 — middle zone: open space (no divider), reachable
                            from both the trap door and the bypass gap;
                            the dual-lock is visible ahead
        row b2          — barrier 2: walls + dual-lock (red+blue)
        rows b2+1..h-1  — exit zone

    KEY DESIGN DECISIONS:
    - Two barriers guarantee the dual-lock is the only path to exit.
    - The middle zone is a single connected open area — no divider wall.
      Both the trap door and bypass gap lead into this same space. The
      dual-lock (barrier 2) is biased toward the trap column so the
      trap door → dual-lock path looks like the direct route to exit.
    - The trap works because the path *looks* correct: open the trap door,
      walk to the dual-lock, try to open it — but the red key is already
      spent. No dead-end pocket needed; the trap is economic, not spatial.
    - The bypass is on a grid edge; the trap is on the opposite side. Exit is
      biased toward the trap side so greedy strategies head that way.
    - Anti-greediness is verified by the greedy strategy filters in
      generate_validated_instance (no manhattan-distance check here).

    DEADLOCKS: waste red on trap → can't open dual-lock (≥1).
    """
    if rng is None:
        rng = random.Random()
    if height < 5 or width < 4:
        return None

    # --- 1. Barrier rows ---
    # Need: upper(≥1 row) + b1 + middle(≥1 row) + b2 + exit(≥1 row)
    b1_max = height - 4
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min, b2_max = b1 + 2, height - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Bypass column (always a grid edge) ---
    bypass_col = rng.choice([0, width - 1])

    # --- 3. Trap door on barrier 1 (opposite side from bypass) ---
    # Biased toward opposite edge so exit (also on that side) is closer to
    # the trap than to the bypass — greedy agents head toward the trap.
    if bypass_col == 0:
        trap_candidates = list(range(width // 2, width))
    else:
        trap_candidates = list(range(0, (width + 1) // 2))
    trap_candidates = [c for c in trap_candidates if c != bypass_col]
    if not trap_candidates:
        return None
    trap_col = rng.choice(trap_candidates)

    # --- 4. Dual-lock on barrier 2 (biased toward trap side) ---
    # Place the dual-lock anywhere except the bypass column, biased toward
    # the trap column so the trap door → dual-lock path looks like the
    # direct route to the exit.
    dl_candidates = [c for c in range(width) if c != bypass_col]
    dl_candidates.sort(key=lambda c: abs(c - trap_col))
    n_pick = max(1, len(dl_candidates) // 2)
    dual_lock_col = rng.choice(dl_candidates[:n_pick])

    # --- 5. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: all columns except trap_col and bypass_col
    for c in range(width):
        if c != trap_col and c != bypass_col:
            walls.add(Pos(b1, c))

    # Barrier 2: all columns except dual_lock_col
    for c in range(width):
        if c != dual_lock_col:
            walls.add(Pos(b2, c))

    # --- Scaling: extra trap(s) on barrier 1 at w>=6 / w>=7 ---
    extra_traps_b1: list[int] = []
    if width >= 6:
        # Exit-half = trap side (opposite bypass)
        if bypass_col == 0:
            exit_half_ct = list(range(width // 2, width))
        else:
            exit_half_ct = list(range(0, (width + 1) // 2))
        used_b1: set[int] = {trap_col, bypass_col}
        col = _pick_extra_trap_col(b1, walls, used_b1, exit_half_ct, rng)
        if col is not None:
            walls.discard(Pos(b1, col))
            extra_traps_b1.append(col)
            used_b1.add(col)
        if width >= 7:
            col2 = _pick_extra_trap_col(b1, walls, used_b1, exit_half_ct, rng)
            if col2 is not None:
                walls.discard(Pos(b1, col2))
                extra_traps_b1.append(col2)

    # --- 6. Exit placement (bias toward trap side for anti-greedy) ---
    exit_row = rng.randint(b2 + 1, height - 1)
    exit_col_candidates = [
        c for c in range(width)
        if Pos(exit_row, c) not in walls
    ]
    if not exit_col_candidates:
        return None
    # Sort by proximity to trap column (closer = more anti-greedy)
    exit_col_candidates.sort(key=lambda c: abs(c - trap_col))
    n_pick = max(1, len(exit_col_candidates) // 2)
    exit_col = rng.choice(exit_col_candidates[:n_pick])
    exit_pos = Pos(exit_row, exit_col)

    trap_door_pos = Pos(b1, trap_col)
    dual_lock_pos = Pos(b2, dual_lock_col)
    if exit_pos == dual_lock_pos:
        return None

    # --- 7. Verify zone connectivity ---

    # Upper zone (rows 0..b1-1)
    upper = {
        Pos(r, c) for r in range(b1) for c in range(width)
        if Pos(r, c) not in walls
    }
    bypass_above = Pos(b1 - 1, bypass_col)
    if len(upper) < 3 or bypass_above not in upper or not _is_connected(upper):
        return None

    # Middle zone: single connected space (no divider)
    middle = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(width)
        if Pos(r, c) not in walls
    }
    bypass_below = Pos(b1 + 1, bypass_col)
    dl_above = Pos(b2 - 1, dual_lock_col)
    trap_below = Pos(b1 + 1, trap_col)
    if not middle or bypass_below not in middle or dl_above not in middle:
        return None
    if trap_below not in middle:
        return None
    middle_comp = _flood_fill(bypass_below, middle)
    if dl_above not in middle_comp or trap_below not in middle_comp:
        return None

    # Exit zone (rows b2+1..height-1)
    exit_floor = {
        Pos(r, c) for r in range(b2 + 1, height) for c in range(width)
        if Pos(r, c) not in walls
    }
    dl_below = Pos(b2 + 1, dual_lock_col)
    if dl_below not in exit_floor or exit_pos not in exit_floor:
        return None
    exit_comp = _flood_fill(dl_below, exit_floor)
    if exit_pos not in exit_comp:
        return None

    # --- 8. Place start + keys in upper zone ---
    upper_list = list(upper)
    if len(upper_list) < 3:
        return None
    rng.shuffle(upper_list)
    start = upper_list[0]

    key_candidates = [p for p in upper_list if p != start]
    key_result = _select_key_positions(
        key_candidates, 2, rng, reference=exit_pos,
    )
    if key_result is None:
        return None
    red_key_pos, blue_key_pos = key_result

    upper_comp = _flood_fill(start, upper)
    if (red_key_pos not in upper_comp or blue_key_pos not in upper_comp
            or bypass_above not in upper_comp):
        return None

    # --- 9. Optional extra walls (upper zone) for variety ---
    trap_above = Pos(b1 - 1, trap_col)
    protected = {start, red_key_pos, blue_key_pos, bypass_above}
    if trap_above in upper:
        protected.add(trap_above)
    extras = [p for p in upper if p not in protected]
    rng.shuffle(extras)
    n_extra = rng.randint(0, min(2, len(extras) // 4))
    for pos in extras:
        if n_extra <= 0:
            break
        rem = upper - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in protected):
            walls.add(pos)
            upper.discard(pos)
            n_extra -= 1

    # --- 10. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=red_key_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=blue_key_pos, color="blue"),
    }
    doors = {
        "D_trap": DoorInfo(id="D_trap", pos=trap_door_pos, required_keys=("red",)),
        "D_dual": DoorInfo(id="D_dual", pos=dual_lock_pos, required_keys=("red", "blue")),
    }
    for i, tc in enumerate(extra_traps_b1):
        did = f"D_trap{i + 2}"
        doors[did] = DoorInfo(
            id=did, pos=Pos(b1, tc), required_keys=("blue",))
    grid = ChainVaultGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Anti-greedy convergent forks grid generation
# ---------------------------------------------------------------------------

def _generate_cf_3way(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a 3-way convergent forks puzzle (P2.5).

    MECHANIC: 3 keys (red + blue + green). Barrier 1 has 3 trap doors
    (one per color) + bypass gap. Barrier 2 has a triple-lock (R+B+G).
    Exit sits below barrier 2.

    LAYOUT:
        rows 0..b1-1    — upper zone: start, K_r, K_b, K_g
        row b1          — barrier 1: walls + red trap + blue trap + green trap + bypass
        rows b1+1..b2-1 — middle zone
        row b2          — barrier 2: walls + triple-lock (R+B+G)
        rows b2+1..h-1  — exit zone

    Requires w>=7 (4 gaps on barrier 1: 3 traps + bypass), h>=7.
    """
    w, h = width, height
    if w < 7 or h < 7:
        return None

    # --- 1. Barrier rows ---
    b1_max = h - 4
    if b1_max < 2:
        return None
    b1 = rng.randint(2, b1_max)
    b2_min, b2_max = b1 + 2, h - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Bypass column (grid edge) ---
    bypass_col = rng.choice([0, w - 1])

    # --- 3. Three trap columns (biased away from bypass) ---
    if bypass_col == 0:
        pool = list(range(w // 2, w))
    else:
        pool = list(range(0, (w + 1) // 2))
    pool = [c for c in pool if c != bypass_col]
    if len(pool) < 3:
        # Try full range minus bypass
        pool = [c for c in range(w) if c != bypass_col]
    if len(pool) < 3:
        return None
    rng.shuffle(pool)
    trap_col_r, trap_col_b, trap_col_g = pool[0], pool[1], pool[2]

    # --- 4. Triple-lock column on barrier 2 ---
    dl_candidates = [c for c in range(w) if c != bypass_col]
    if not dl_candidates:
        return None
    trap_mid = (trap_col_r + trap_col_b + trap_col_g) / 3.0
    dl_candidates.sort(key=lambda c: abs(c - trap_mid))
    n_pick = max(1, len(dl_candidates) // 2)
    triple_lock_col = rng.choice(dl_candidates[:n_pick])

    # --- 5. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: all columns except bypass + 3 traps
    gaps = {bypass_col, trap_col_r, trap_col_b, trap_col_g}
    for c in range(w):
        if c not in gaps:
            walls.add(Pos(b1, c))

    # Barrier 2: all columns except triple-lock
    for c in range(w):
        if c != triple_lock_col:
            walls.add(Pos(b2, c))

    # --- 6. Key placement (convergent forks bait, one per trap) ---
    upper = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    red_key_pos = _convergent_key_pos(Pos(b1, trap_col_r), upper, rng)
    if red_key_pos is None:
        return None
    blue_key_pos = _convergent_key_pos(
        Pos(b1, trap_col_b), upper - {red_key_pos}, rng)
    if blue_key_pos is None:
        return None
    green_key_pos = _convergent_key_pos(
        Pos(b1, trap_col_g), upper - {red_key_pos, blue_key_pos}, rng)
    if green_key_pos is None:
        return None

    # --- 7. Exit placement ---
    exit_row = rng.randint(b2 + 1, h - 1)
    exit_col_candidates = [
        c for c in range(w) if Pos(exit_row, c) not in walls
    ]
    if not exit_col_candidates:
        return None
    exit_col_candidates.sort(key=lambda c: abs(c - bypass_col), reverse=True)
    n_pick = max(1, len(exit_col_candidates) // 2)
    exit_col = rng.choice(exit_col_candidates[:n_pick])
    exit_pos = Pos(exit_row, exit_col)
    if exit_pos == Pos(b2, triple_lock_col):
        return None

    # --- 8. Start placement ---
    bypass_above = Pos(b1 - 1, bypass_col)
    if len(upper) < 4 or bypass_above not in upper or not _is_connected(upper):
        return None

    start_candidates = [
        p for p in upper
        if p not in {red_key_pos, blue_key_pos, green_key_pos}
    ]
    if not start_candidates:
        return None
    start = rng.choice(start_candidates)

    upper_comp = _flood_fill(start, upper)
    if not {red_key_pos, blue_key_pos, green_key_pos, bypass_above}.issubset(upper_comp):
        return None

    # --- 9. Middle zone connectivity ---
    middle = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }
    bypass_below = Pos(b1 + 1, bypass_col)
    tl_above = Pos(b2 - 1, triple_lock_col)
    if not middle or bypass_below not in middle or tl_above not in middle:
        return None
    # Check trap entries are in middle
    trap_entries = [Pos(b1 + 1, c) for c in [trap_col_r, trap_col_b, trap_col_g]]
    for te in trap_entries:
        if te not in middle:
            return None
    middle_comp = _flood_fill(bypass_below, middle)
    if tl_above not in middle_comp:
        return None
    for te in trap_entries:
        if te not in middle_comp:
            return None

    # --- 10. Exit zone connectivity ---
    exit_floor = {
        Pos(r, c) for r in range(b2 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    tl_below = Pos(b2 + 1, triple_lock_col)
    if tl_below not in exit_floor or exit_pos not in exit_floor:
        return None
    exit_comp = _flood_fill(tl_below, exit_floor)
    if exit_pos not in exit_comp:
        return None

    # --- 11. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=red_key_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=blue_key_pos, color="blue"),
        "K_g": KeyInfo(id="K_g", pos=green_key_pos, color="green"),
    }
    doors = {
        "D_trap_r": DoorInfo(
            id="D_trap_r", pos=Pos(b1, trap_col_r), required_keys=("red",)),
        "D_trap_b": DoorInfo(
            id="D_trap_b", pos=Pos(b1, trap_col_b), required_keys=("blue",)),
        "D_trap_g": DoorInfo(
            id="D_trap_g", pos=Pos(b1, trap_col_g), required_keys=("green",)),
        "D_triple": DoorInfo(
            id="D_triple", pos=Pos(b2, triple_lock_col),
            required_keys=("red", "blue", "green")),
    }
    grid = ChainVaultGrid(
        width=w, height=h,
        walls=frozenset(walls), keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def _generate_cf_sequential_fork(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Sequential fork variant of Convergent Forks (requires width >= 6, height >= 6).

    MECHANIC: 3 colors (R, G, B). Barrier 1 has D_trap_r (red) + D_trap_g
    (green) + bypass gap. Barrier 2 has D_correct_g (green, leads to Area A
    with K_b) + D_dual (R+B dual-lock). Below barrier 2, a vertical divider
    separates Area A (K_b) from the exit zone.

    Sequential dependency: G → K_b → R+B exit.
      - Agent must save green to open D_correct_g (barrier 2) to obtain K_b.
      - Agent must save red + obtained blue for D_dual.

    Convergent bait: K_r above D_trap_r, K_g above D_trap_g on barrier 1.

    LAYOUT:
        rows 0..b1-1    — Zone 0: start, K_r (near D_trap_r), K_g (near D_trap_g)
        row b1          — barrier 1: D_trap_r + D_trap_g + bypass + walls
        rows b1+1..b2-1 — middle zone: open space
        row b2          — barrier 2: D_correct_g + D_dual (R+B) + walls
        rows b2+1..h-1  — below barrier 2:
            Area A (behind D_correct_g): K_b
            Exit zone (behind D_dual): exit
            Separated by vertical divider wall

    DEADLOCKS:
        - Waste R on D_trap_r → no R for dual-lock (≥1)
        - Waste G on D_trap_g → no G for D_correct_g → no K_b → no B for dual-lock (≥1)
    """
    w, h = width, height
    if w < 6 or h < 6:
        return None

    # --- 1. Barrier rows ---
    b1_max = h - 5  # need ≥1 row upper, ≥1 middle, barrier2, ≥1 below
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min, b2_max = b1 + 2, h - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Bypass column (grid edge) ---
    bypass_col = rng.choice([0, w - 1])

    # --- 3. Two trap columns on barrier 1 (biased away from bypass) ---
    if bypass_col == 0:
        pool = list(range(w // 2, w))
    else:
        pool = list(range(0, (w + 1) // 2))
    pool = [c for c in pool if c != bypass_col]
    if len(pool) < 2:
        return None
    rng.shuffle(pool)
    trap_col_r, trap_col_g = pool[0], pool[1]

    # --- 4. Barrier 2: D_correct_g and D_dual columns ---
    # Require separation ≥ 2 so there's at least 1 divider column between them.
    b2_cols = list(range(w))
    rng.shuffle(b2_cols)
    correct_g_col: Optional[int] = None
    dual_col: Optional[int] = None
    for i, c1 in enumerate(b2_cols):
        for c2 in b2_cols[i + 1:]:
            if abs(c1 - c2) >= 2:
                correct_g_col, dual_col = c1, c2
                break
        if correct_g_col is not None:
            break
    if correct_g_col is None or dual_col is None:
        return None

    # Divider column: between correct_g_col and dual_col
    lo, hi = min(correct_g_col, dual_col), max(correct_g_col, dual_col)
    divider_options = list(range(lo + 1, hi))
    if not divider_options:
        return None
    divider_col = rng.choice(divider_options)

    # --- 5. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: all columns except bypass, trap_r, trap_g
    for c in range(w):
        if c not in (bypass_col, trap_col_r, trap_col_g):
            walls.add(Pos(b1, c))

    # Barrier 2: all columns except correct_g_col and dual_col
    for c in range(w):
        if c not in (correct_g_col, dual_col):
            walls.add(Pos(b2, c))

    # Divider below barrier 2 (rows b2+1..h-1)
    for r in range(b2 + 1, h):
        walls.add(Pos(r, divider_col))

    # --- Scaling: extra trap(s) on barrier 1 at w>=7 / w>=8 ---
    extra_traps: list[int] = []
    if w >= 7:
        if bypass_col == 0:
            exit_half = list(range(w // 2, w))
        else:
            exit_half = list(range(0, (w + 1) // 2))
        used_b1: set[int] = {bypass_col, trap_col_r, trap_col_g}
        col = _pick_extra_trap_col(b1, walls, used_b1, exit_half, rng)
        if col is not None:
            walls.discard(Pos(b1, col))
            extra_traps.append(col)
            used_b1.add(col)
        if w >= 8:
            col2 = _pick_extra_trap_col(b1, walls, used_b1, exit_half, rng)
            if col2 is not None:
                walls.discard(Pos(b1, col2))
                extra_traps.append(col2)

    # --- 6. Key placement (convergent forks bait) ---
    upper = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    red_key_pos = _convergent_key_pos(Pos(b1, trap_col_r), upper, rng)
    if red_key_pos is None:
        return None
    green_key_pos = _convergent_key_pos(
        Pos(b1, trap_col_g), upper - {red_key_pos}, rng
    )
    if green_key_pos is None:
        return None
    if red_key_pos in walls or green_key_pos in walls:
        return None

    # --- 7. K_b placement in Area A (below barrier 2, on correct_g side) ---
    if correct_g_col < divider_col:
        area_a_cols = range(0, divider_col)
    else:
        area_a_cols = range(divider_col + 1, w)
    area_a = {
        Pos(r, c) for r in range(b2 + 1, h) for c in area_a_cols
        if Pos(r, c) not in walls
    }
    correct_g_below = Pos(b2 + 1, correct_g_col)
    if correct_g_below not in area_a:
        return None
    area_a_comp = _flood_fill(correct_g_below, area_a)
    if not area_a_comp:
        return None
    blue_key_candidates = list(area_a_comp)
    rng.shuffle(blue_key_candidates)
    blue_key_pos = blue_key_candidates[0]

    # --- 8. Exit placement in exit zone (below barrier 2, on dual side) ---
    if dual_col < divider_col:
        exit_zone_cols = range(0, divider_col)
    else:
        exit_zone_cols = range(divider_col + 1, w)
    exit_zone = {
        Pos(r, c) for r in range(b2 + 1, h) for c in exit_zone_cols
        if Pos(r, c) not in walls
    }
    dual_below = Pos(b2 + 1, dual_col)
    if dual_below not in exit_zone:
        return None
    exit_zone_comp = _flood_fill(dual_below, exit_zone)
    if not exit_zone_comp:
        return None
    # Bias exit away from bypass
    exit_candidates = sorted(
        exit_zone_comp, key=lambda p: abs(p.col - bypass_col), reverse=True
    )
    n_pick = max(1, len(exit_candidates) // 2)
    exit_pos = rng.choice(exit_candidates[:n_pick])
    if exit_pos == Pos(b2, dual_col):
        return None

    # --- 9. Start placement (upper zone) ---
    bypass_above = Pos(b1 - 1, bypass_col)
    if len(upper) < 3 or bypass_above not in upper or not _is_connected(upper):
        return None
    start_candidates = [
        p for p in upper if p != red_key_pos and p != green_key_pos
    ]
    if not start_candidates:
        return None
    rng.shuffle(start_candidates)
    start = start_candidates[0]

    upper_comp = _flood_fill(start, upper)
    if (red_key_pos not in upper_comp or green_key_pos not in upper_comp
            or bypass_above not in upper_comp):
        return None

    # --- 10. Middle zone connectivity ---
    middle = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }
    bypass_below = Pos(b1 + 1, bypass_col)
    correct_g_above = Pos(b2 - 1, correct_g_col)
    dual_above = Pos(b2 - 1, dual_col)
    trap_r_below = Pos(b1 + 1, trap_col_r)
    trap_g_below = Pos(b1 + 1, trap_col_g)
    required_middle = {bypass_below, correct_g_above, dual_above,
                       trap_r_below, trap_g_below}
    if not all(p in middle for p in required_middle):
        return None
    middle_comp = _flood_fill(bypass_below, middle)
    if not all(p in middle_comp for p in required_middle):
        return None

    # --- 11. Optional extra walls (upper zone, 0-2) ---
    trap_r_above = Pos(b1 - 1, trap_col_r)
    trap_g_above = Pos(b1 - 1, trap_col_g)
    protected = {start, red_key_pos, green_key_pos, bypass_above}
    if trap_r_above in upper:
        protected.add(trap_r_above)
    if trap_g_above in upper:
        protected.add(trap_g_above)
    extras = [p for p in upper if p not in protected]
    rng.shuffle(extras)
    n_extra = rng.randint(0, min(2, len(extras) // 4))
    for pos in extras:
        if n_extra <= 0:
            break
        rem = upper - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in protected):
            walls.add(pos)
            upper.discard(pos)
            n_extra -= 1

    # --- 12. Build grid ---
    trap_door_r_pos = Pos(b1, trap_col_r)
    trap_door_g_pos = Pos(b1, trap_col_g)
    correct_g_pos = Pos(b2, correct_g_col)
    dual_pos = Pos(b2, dual_col)

    keys = {
        "K_r": KeyInfo(id="K_r", pos=red_key_pos, color="red"),
        "K_g": KeyInfo(id="K_g", pos=green_key_pos, color="green"),
        "K_b": KeyInfo(id="K_b", pos=blue_key_pos, color="blue"),
    }
    doors = {
        "D_trap_r": DoorInfo(
            id="D_trap_r", pos=trap_door_r_pos, required_keys=("red",)),
        "D_trap_g": DoorInfo(
            id="D_trap_g", pos=trap_door_g_pos, required_keys=("green",)),
        "D_correct_g": DoorInfo(
            id="D_correct_g", pos=correct_g_pos, required_keys=("green",)),
        "D_dual": DoorInfo(
            id="D_dual", pos=dual_pos, required_keys=("red", "blue")),
    }
    trap_colors = ["red", "green"]
    for i, tc in enumerate(extra_traps):
        did = f"D_trap_extra{i + 1}"
        color = trap_colors[i % 2]
        doors[did] = DoorInfo(
            id=did, pos=Pos(b1, tc), required_keys=(color,))

    grid = ChainVaultGrid(
        width=w,
        height=h,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def generate_convergent_forks_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a convergent forks grid with two independent trap-avoidance decisions.

    MECHANIC: 2 keys (red + blue). Barrier 1 has BOTH a red trap door and a
    blue trap door, plus an open bypass gap. Barrier 2 has a dual-lock (R+B)
    as the only passage. Exit sits below barrier 2.

    Each key is placed directly above its matching trap door on barrier 1.
    When a greedy agent picks up either key, it is immediately adjacent to
    the matching trap — all 6 greedy strategies prioritize USE_KEY when
    adjacent, so the key is wasted.

    Correct path: collect both keys → bypass gap → middle zone → dual-lock → exit.

    LAYOUT:
        rows 0..b1-1    — upper zone: start, red key, blue key
        row b1          — barrier 1: walls + red trap + blue trap + bypass gap
        rows b1+1..b2-1 — middle zone: open space
        row b2          — barrier 2: walls + dual-lock (red+blue)
        rows b2+1..h-1  — exit zone

    DEADLOCKS: waste red on trap → can't open dual-lock; waste blue → same. (≥2)
    """
    if rng is None:
        rng = random.Random()
    if height < 5 or width < 5:
        return None

    # --- P2.5: 3-way convergent forks (triple-lock exit) at w>=7, h>=7 ---
    if width >= 7 and height >= 7 and rng.random() < 0.5:
        result = _generate_cf_3way(width, height, rng)
        if result is not None:
            return result

    # --- Sequential fork variant (50% at width >= 6 and height >= 6) ---
    if width >= 6 and height >= 6 and rng.random() < 0.5:
        result = _generate_cf_sequential_fork(width, height, rng)
        if result is not None:
            return result
        # Fall through to standard variant if sequential fork fails

    # --- 1. Barrier rows ---
    b1_max = height - 4
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min, b2_max = b1 + 2, height - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Bypass column (always a grid edge) ---
    bypass_col = rng.choice([0, width - 1])

    # --- 3. Two trap columns on barrier 1 (biased away from bypass) ---
    if bypass_col == 0:
        pool = list(range(width // 2, width))
    else:
        pool = list(range(0, (width + 1) // 2))
    pool = [c for c in pool if c != bypass_col]
    if len(pool) < 2:
        return None
    rng.shuffle(pool)
    trap_col_r, trap_col_b = pool[0], pool[1]

    # --- 4. Dual-lock column on barrier 2 (biased toward trap midpoint) ---
    dl_candidates = [c for c in range(width) if c != bypass_col]
    if not dl_candidates:
        return None
    trap_mid = (trap_col_r + trap_col_b) / 2.0
    dl_candidates.sort(key=lambda c: abs(c - trap_mid))
    n_pick = max(1, len(dl_candidates) // 2)
    dual_lock_col = rng.choice(dl_candidates[:n_pick])

    # --- 5. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: all columns except bypass, trap_r, trap_b
    for c in range(width):
        if c not in (bypass_col, trap_col_r, trap_col_b):
            walls.add(Pos(b1, c))

    # Barrier 2: all columns except dual_lock_col
    for c in range(width):
        if c != dual_lock_col:
            walls.add(Pos(b2, c))

    # --- Scaling: extra trap(s) on barrier 1 at w>=6 / w>=7 ---
    extra_traps_cf: list[int] = []
    if width >= 6:
        # Exit-half = trap side (opposite bypass)
        if bypass_col == 0:
            exit_half_cf = list(range(width // 2, width))
        else:
            exit_half_cf = list(range(0, (width + 1) // 2))
        used_b1_cf: set[int] = {bypass_col, trap_col_r, trap_col_b}
        col = _pick_extra_trap_col(b1, walls, used_b1_cf, exit_half_cf, rng)
        if col is not None:
            walls.discard(Pos(b1, col))
            extra_traps_cf.append(col)
            used_b1_cf.add(col)
        if width >= 7:
            col2 = _pick_extra_trap_col(b1, walls, used_b1_cf, exit_half_cf, rng)
            if col2 is not None:
                walls.discard(Pos(b1, col2))
                extra_traps_cf.append(col2)

    # --- 6. Key placement (convergent forks bait) ---
    upper = {
        Pos(r, c) for r in range(b1) for c in range(width)
        if Pos(r, c) not in walls
    }
    red_key_pos = _convergent_key_pos(Pos(b1, trap_col_r), upper, rng)
    if red_key_pos is None:
        return None
    blue_key_pos = _convergent_key_pos(Pos(b1, trap_col_b), upper - {red_key_pos}, rng)
    if blue_key_pos is None:
        return None
    if red_key_pos in walls or blue_key_pos in walls:
        return None

    trap_door_r_pos = Pos(b1, trap_col_r)
    trap_door_b_pos = Pos(b1, trap_col_b)
    dual_lock_pos = Pos(b2, dual_lock_col)

    # --- 7. Exit placement (biased away from bypass) ---
    exit_row = rng.randint(b2 + 1, height - 1)
    exit_col_candidates = [
        c for c in range(width)
        if Pos(exit_row, c) not in walls
    ]
    if not exit_col_candidates:
        return None
    # Sort by distance from bypass (farther = better)
    exit_col_candidates.sort(key=lambda c: abs(c - bypass_col), reverse=True)
    n_pick = max(1, len(exit_col_candidates) // 2)
    exit_col = rng.choice(exit_col_candidates[:n_pick])
    exit_pos = Pos(exit_row, exit_col)
    if exit_pos == dual_lock_pos:
        return None

    # --- 8. Start placement (upper zone, not on keys) ---
    bypass_above = Pos(b1 - 1, bypass_col)
    if len(upper) < 3 or bypass_above not in upper or not _is_connected(upper):
        return None

    start_candidates = [
        p for p in upper if p != red_key_pos and p != blue_key_pos
    ]
    if not start_candidates:
        return None
    rng.shuffle(start_candidates)
    start = start_candidates[0]

    # Verify upper connectivity: start → both keys → bypass_above
    upper_comp = _flood_fill(start, upper)
    if (red_key_pos not in upper_comp or blue_key_pos not in upper_comp
            or bypass_above not in upper_comp):
        return None

    # --- 9. Connectivity checks ---

    # Middle zone (rows b1+1..b2-1)
    middle = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(width)
        if Pos(r, c) not in walls
    }
    bypass_below = Pos(b1 + 1, bypass_col)
    dl_above = Pos(b2 - 1, dual_lock_col)
    trap_r_below = Pos(b1 + 1, trap_col_r)
    trap_b_below = Pos(b1 + 1, trap_col_b)
    if not middle or bypass_below not in middle or dl_above not in middle:
        return None
    if trap_r_below not in middle or trap_b_below not in middle:
        return None
    middle_comp = _flood_fill(bypass_below, middle)
    if dl_above not in middle_comp:
        return None
    if trap_r_below not in middle_comp or trap_b_below not in middle_comp:
        return None

    # Exit zone (rows b2+1..height-1)
    exit_floor = {
        Pos(r, c) for r in range(b2 + 1, height) for c in range(width)
        if Pos(r, c) not in walls
    }
    dl_below = Pos(b2 + 1, dual_lock_col)
    if dl_below not in exit_floor or exit_pos not in exit_floor:
        return None
    exit_comp = _flood_fill(dl_below, exit_floor)
    if exit_pos not in exit_comp:
        return None

    # --- 10. Optional extra walls (upper zone, 0-2) ---
    trap_r_above = Pos(b1 - 1, trap_col_r)
    trap_b_above = Pos(b1 - 1, trap_col_b)
    protected = {start, red_key_pos, blue_key_pos, bypass_above}
    if trap_r_above in upper:
        protected.add(trap_r_above)
    if trap_b_above in upper:
        protected.add(trap_b_above)
    extras = [p for p in upper if p not in protected]
    rng.shuffle(extras)
    n_extra = rng.randint(0, min(2, len(extras) // 4))
    for pos in extras:
        if n_extra <= 0:
            break
        rem = upper - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in protected):
            walls.add(pos)
            upper.discard(pos)
            n_extra -= 1

    # --- 11. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=red_key_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=blue_key_pos, color="blue"),
    }
    doors = {
        "D_trap_r": DoorInfo(id="D_trap_r", pos=trap_door_r_pos, required_keys=("red",)),
        "D_trap_b": DoorInfo(id="D_trap_b", pos=trap_door_b_pos, required_keys=("blue",)),
        "D_dual": DoorInfo(id="D_dual", pos=dual_lock_pos, required_keys=("red", "blue")),
    }
    cf_trap_colors = ["red", "blue"]
    for i, tc in enumerate(extra_traps_cf):
        did = f"D_trap_extra{i + 1}"
        color = cf_trap_colors[i % 2]
        doors[did] = DoorInfo(
            id=did, pos=Pos(b1, tc), required_keys=(color,))
    grid = ChainVaultGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Anti-greedy parallel wings grid generation
# ---------------------------------------------------------------------------

def generate_parallel_wings_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a parallel wings grid with gated right-wing key room.

    MECHANIC: 3 keys (K_b blue + K_r red + K_b2 blue).  A vertical divider
    splits the grid into left and right wings.  The right wing's key room is
    gated behind a blue single-lock door (D_gate) using a horizontal gate wall
    at row ``gate_row``.  The agent must spend the left-wing blue key (K_b) as
    an investment to unlock D_gate and access both K_r and K_b2.  The dual-
    lock exit then consumes K_r + K_b2.

    WHY THE TRAP WORKS:
    The blue trap door on the divider (D_trap_b) offers a tempting shortcut to
    the right wing — but spending K_b there means D_gate can never be opened.
    K_r and K_b2 are permanently locked away, and the dual-lock exit requires
    both.  Walking Dead state is reached with apparent freedom of movement.

    CORRECT PATH:
        Collect K_b (left wing) → bypass gap (row H-2) → navigate lower right
        → USE K_b on D_gate → collect K_r + K_b2 (upper right room)
        → return via bypass → USE K_r+K_b2 on D_dual → exit.

    LAYOUT:
        cols 0..D-1           — left wing  (K_b near blue trap; start at col 0)
        col  D                — divider    (D_trap_b + D_trap_r + bypass at H-2)
        cols D+1..W-1         — right wing
          rows 0..G-1         — upper right: gated key room (K_r + K_b2)
          row  G, col G_c     — D_gate (blue single-lock)
          row  G, other cols  — gate wall
          rows G+1..H-2       — lower right: open, accessible from bypass
        (H-1, 0)              — exit behind D_dual at (H-1, 1)

    DEADLOCKS:
      - Waste K_b on D_trap_b → D_gate unopenable → K_r/K_b2 permanently stuck
      - Waste K_r on D_trap_r (after collecting K_r) → D_dual needs red → stuck
    """
    if rng is None:
        rng = random.Random()
    if width < 5 or height < 5:
        return None

    # --- 1. Divider column (need ≥2 cols per wing) ---
    div_col = rng.randint(2, width - 3)

    # --- 2. Trap rows (must be above the bypass at row h-2) ---
    bypass_row = height - 2
    trap_pool = list(range(1, bypass_row))
    if len(trap_pool) < 2:
        return None
    rng.shuffle(trap_pool)
    trap_row_b, trap_row_r = sorted(trap_pool[:2])

    # --- 3. Gate row: horizontal wall in right wing dividing it into
    #         upper (gated key room) and lower (open, bypass-accessible) areas.
    gate_row_pool = list(range(1, bypass_row))
    if not gate_row_pool:
        return None
    gate_row = rng.choice(gate_row_pool)

    # Upper right must hold ≥2 cells for K_r and K_b2.
    upper_right_cols = width - div_col - 1
    if gate_row * upper_right_cols < 2:
        return None

    # --- 4. Gate column: the single gap in the gate wall ---
    gate_col = rng.randint(div_col + 1, width - 1)

    # --- 5. Build divider walls ---
    walls: set[Pos] = set()
    for r in range(height):
        if r == bypass_row:
            continue  # bypass gap
        if r in (trap_row_b, trap_row_r):
            continue  # trap door slots
        walls.add(Pos(r, div_col))

    # --- 6. Build gate wall in right wing ---
    for c in range(div_col + 1, width):
        if c == gate_col:
            continue  # D_gate slot
        walls.add(Pos(gate_row, c))

    # --- Scaling: extra trap(s) on divider at w>=6 / w>=7 ---
    extra_trap_rows_pw: list[int] = []
    enhanced_pw = width >= 6 and rng.random() < 0.5
    if width >= 6:
        used_divider_rows = {trap_row_b, trap_row_r, bypass_row, 0, height - 1}
        pw_candidates = [
            r for r in range(1, bypass_row)
            if Pos(r, div_col) in walls and r not in used_divider_rows
        ]
        if pw_candidates:
            er = rng.choice(pw_candidates)
            walls.discard(Pos(er, div_col))
            extra_trap_rows_pw.append(er)
            used_divider_rows.add(er)
        if width >= 7 or enhanced_pw:
            pw_candidates2 = [
                r for r in range(1, bypass_row)
                if Pos(r, div_col) in walls and r not in used_divider_rows
            ]
            if pw_candidates2:
                er2 = rng.choice(pw_candidates2)
                walls.discard(Pos(er2, div_col))
                extra_trap_rows_pw.append(er2)

    # --- 7. Exit pocket (bottom-left corner) ---
    walls.add(Pos(bypass_row, 0))  # pocket wall at (h-2, 0)
    exit_pos = Pos(height - 1, 0)
    dl_pos = Pos(height - 1, 1)

    # --- P2.6: Corridor walls in left wing (forces path past trap door) ---
    # At w>=6, add a horizontal wall strip between blue trap row and bypass,
    # leaving a gap only at div_col-1 (trap approach column). This forces
    # the agent to walk past the trap before reaching bypass.
    corridor_added = False
    if width >= 6 and rng.random() < 0.5:
        # Pick a row between blue trap and bypass for the corridor wall
        corridor_row_options = list(range(trap_row_b + 1, bypass_row))
        if corridor_row_options:
            corridor_row = rng.choice(corridor_row_options)
            # Wall spans cols 0..div_col-2, gap at div_col-1
            for c in range(div_col - 1):
                walls.add(Pos(corridor_row, c))
            corridor_added = True

    # --- 8. Key placement ---
    # K_b: left wing, divider-adjacent, above blue trap (convergent-forks bait)
    bk_candidates = list(range(0, trap_row_b))
    if not bk_candidates:
        return None
    bk_row = rng.choice(bk_candidates)
    blue_key_pos = Pos(bk_row, div_col - 1)  # K_b

    if blue_key_pos in walls:
        return None

    # K_r + K_b2: inside upper right area (rows 0..gate_row-1, right-wing cols)
    upper_right_cells = [
        Pos(r, c)
        for r in range(gate_row)
        for c in range(div_col + 1, width)
        if Pos(r, c) not in walls
    ]
    if len(upper_right_cells) < 2:
        return None
    key_positions = _select_key_positions(upper_right_cells, 2, rng)
    if key_positions is None:
        return None
    red_key_pos, blue2_key_pos = key_positions

    # --- 9. Start placement (left wing, col 0) ---
    start_rows = [
        r for r in range(height)
        if Pos(r, 0) not in walls
        and Pos(r, 0) != blue_key_pos
        and Pos(r, 0) != exit_pos
        and Pos(r, 0) != dl_pos
    ]
    if not start_rows:
        return None
    start = Pos(rng.choice(start_rows), 0)

    # --- 10. Connectivity checks ---

    # Left floor: cols 0..div_col-1, minus walls/exit/DL
    left_floor = {
        Pos(r, c) for r in range(height) for c in range(div_col)
        if Pos(r, c) not in walls
        and Pos(r, c) != exit_pos
        and Pos(r, c) != dl_pos
    }
    if start not in left_floor or blue_key_pos not in left_floor:
        return None
    if not _is_connected(left_floor):
        return None
    left_comp = _flood_fill(start, left_floor)
    if blue_key_pos not in left_comp:
        return None

    bypass_left = Pos(bypass_row, div_col - 1)
    if bypass_left not in left_comp:
        return None

    # DL approach: (h-2, 1) must be reachable (to USE_KEY on D_dual at (h-1,1))
    dl_approach = Pos(bypass_row, 1)
    if dl_approach not in left_comp:
        return None

    # Lower right floor: rows gate_row+1..height-1, right-wing cols
    lower_right_floor = {
        Pos(r, c)
        for r in range(gate_row + 1, height)
        for c in range(div_col + 1, width)
        if Pos(r, c) not in walls
    }
    bypass_right = Pos(bypass_row, div_col + 1)
    if bypass_right not in lower_right_floor:
        return None
    if not _is_connected(lower_right_floor):
        return None
    lower_right_comp = _flood_fill(bypass_right, lower_right_floor)

    # Agent must reach the cell below D_gate to USE_KEY on it
    gate_approach = Pos(gate_row + 1, gate_col)
    if gate_approach not in lower_right_comp:
        return None

    # Upper right floor: rows 0..gate_row-1, right-wing cols
    upper_right_floor = {
        Pos(r, c)
        for r in range(gate_row)
        for c in range(div_col + 1, width)
        if Pos(r, c) not in walls
    }
    if red_key_pos not in upper_right_floor or blue2_key_pos not in upper_right_floor:
        return None
    if not _is_connected(upper_right_floor):
        return None
    upper_right_comp = _flood_fill(red_key_pos, upper_right_floor)
    if blue2_key_pos not in upper_right_comp:
        return None
    # Cell directly above D_gate must be in upper right (agent exits room through it)
    gate_entry = Pos(gate_row - 1, gate_col)
    if gate_entry not in upper_right_comp:
        return None

    # Exit isolation
    if Pos(bypass_row, 0) not in walls:
        return None

    # --- 11. Optional extra walls (0-2 per section) ---

    # Left wing
    left_protected = {start, blue_key_pos, bypass_left, dl_approach}
    left_extras = [p for p in left_floor if p not in left_protected]
    rng.shuffle(left_extras)
    n_extra = rng.randint(0, min(2, len(left_extras) // 4))
    for pos in left_extras:
        if n_extra <= 0:
            break
        rem = left_floor - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in left_protected):
            walls.add(pos)
            left_floor.discard(pos)
            n_extra -= 1

    # Lower right wing
    lower_right_protected = {bypass_right, gate_approach}
    lower_right_extras = [p for p in lower_right_floor if p not in lower_right_protected]
    rng.shuffle(lower_right_extras)
    n_extra = rng.randint(0, min(2, len(lower_right_extras) // 4))
    for pos in lower_right_extras:
        if n_extra <= 0:
            break
        rem = lower_right_floor - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in lower_right_protected):
            walls.add(pos)
            lower_right_floor.discard(pos)
            n_extra -= 1

    # Upper right wing (gated room)
    upper_right_protected = {red_key_pos, blue2_key_pos, gate_entry}
    upper_right_extras = [p for p in upper_right_floor if p not in upper_right_protected]
    rng.shuffle(upper_right_extras)
    n_extra = rng.randint(0, min(2, len(upper_right_extras) // 4))
    for pos in upper_right_extras:
        if n_extra <= 0:
            break
        rem = upper_right_floor - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in upper_right_protected):
            walls.add(pos)
            upper_right_floor.discard(pos)
            n_extra -= 1

    # --- 12. Build grid ---
    keys = {
        "K_b":  KeyInfo(id="K_b",  pos=blue_key_pos,  color="blue"),
        "K_r":  KeyInfo(id="K_r",  pos=red_key_pos,   color="red"),
        "K_b2": KeyInfo(id="K_b2", pos=blue2_key_pos, color="blue"),
    }
    doors = {
        "D_trap_b": DoorInfo(
            id="D_trap_b", pos=Pos(trap_row_b, div_col), required_keys=("blue",)
        ),
        "D_trap_r": DoorInfo(
            id="D_trap_r", pos=Pos(trap_row_r, div_col), required_keys=("red",)
        ),
        "D_gate": DoorInfo(
            id="D_gate", pos=Pos(gate_row, gate_col), required_keys=("blue",)
        ),
        "D_dual": DoorInfo(
            id="D_dual", pos=dl_pos, required_keys=("red", "blue")
        ),
    }
    pw_trap_colors = ["blue", "red"]
    for i, tr in enumerate(extra_trap_rows_pw):
        did = f"D_trap_extra{i + 1}"
        color = pw_trap_colors[i % 2]
        doors[did] = DoorInfo(
            id=did, pos=Pos(tr, div_col), required_keys=(color,))
    grid = ChainVaultGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Anti-greedy relay grid generation (3 colors)
# ---------------------------------------------------------------------------

def generate_relay_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a relay grid with sequential gating across 3 colors.

    MECHANIC: Red key -> red gate (correct) -> zone 1 with blue+green keys ->
    bypass gap -> zone 2 -> blue+green dual-lock exit pocket -> exit.
    Red trap door leads to a real walkable zone (Zone 1T) split from the
    correct side (Zone 1C) by a vertical divider.  Blue and green traps on
    barrier 2 lead to zone 2 but waste keys needed for the dual-lock.

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + Red key
        Barrier 1 (row b1):         Red gate door + Red trap door
        Zone 1 (rows b1+1..b2-1):   Split by vertical divider:
          Zone 1T (trap side):      Real walkable area (2+ cells, no keys)
          Zone 1C (correct side):   Blue key + Green key (above matching traps)
        Barrier 2 (row b2):         Blue trap + Green trap + Bypass gap (far-side)
        Zone 2 (rows b2+1..h-1):    Open area + Dual-lock exit pocket at corner

    EXIT POCKET (parallel wings pattern):
    - exit_side=0: exit (h-1,0), DL (h-1,1), pocket wall (h-2,0)
    - exit_side=w-1: exit (h-1,w-1), DL (h-1,w-2), pocket wall (h-2,w-1)

    ANTI-GREEDY:
    - Barrier 1: Red trap on exit-side half.  Correct door on far side.
      Start + red key biased toward exit side → greedy hits trap first.
    - Barrier 2: Keys directly above matching traps (convergent forks pattern).
      Greedy agent picks up key → adjacent to trap → USE_KEY fires → wasted.

    CORRECT PATH: Start → red key → red gate → zone 1 → blue+green keys →
    bypass → zone 2 → dual-lock → exit.

    DEADLOCKS (>=3):
    1. Red on D_trap_r -> Zone 1T (real area but no keys, no barrier 2 access)
    2. Blue on D_trap_b -> enters zone 2 but can't open dual-lock (needs B+G)
    3. Green on D_trap_g -> enters zone 2 but can't open dual-lock (needs B+G)
    """
    if rng is None:
        rng = random.Random()
    if width < 5 or height < 6:
        return None

    h, w = height, width

    # --- Color palette permutation (breaks color-role memorization) ---
    palette = ["red", "blue", "green"]
    rng.shuffle(palette)
    c_gate, c_a, c_b = palette  # gate color, zone1 key A color, zone1 key B color

    # --- 1. Barrier rows ---
    # Need: zone0(≥1) + b1 + zone1(≥2) + b2 + zone2(≥1)
    b1_max = h - 5
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min, b2_max = b1 + 3, h - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Exit-side direction + exit pocket ---
    exit_side = rng.choice([0, w - 1])
    if exit_side == 0:
        exit_col = 0
        dl_col = 1
    else:
        exit_col = w - 1
        dl_col = w - 2

    exit_pos = Pos(h - 1, exit_col)
    dl_pos = Pos(h - 1, dl_col)
    pocket_wall_pos = Pos(h - 2, exit_col)

    # --- 3. Bypass column (far-side edge, opposite exit) ---
    bypass_col = w - 1 if exit_side == 0 else 0

    # --- 4. Column halves (exit-side vs far-side) for barrier 1 ---
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))

    if not exit_half or not far_half:
        return None

    # --- 5. Barrier 1 columns ---
    # Trap on exit-side half, correct on far-side half
    trap_r_col = rng.choice(exit_half)
    far_half_b1 = [c for c in far_half if c != trap_r_col]
    if not far_half_b1:
        return None
    correct_r_col = rng.choice(far_half_b1)

    # --- 6. Build barrier 1 walls ---
    walls: set[Pos] = set()
    for c in range(w):
        if c != trap_r_col and c != correct_r_col:
            walls.add(Pos(b1, c))

    # Zone 1 vertical divider between trap and correct sides
    lo_z1, hi_z1 = min(trap_r_col, correct_r_col), max(trap_r_col, correct_r_col)
    divider_options_z1 = list(range(lo_z1 + 1, hi_z1))
    if not divider_options_z1:
        return None
    # Bias toward trap_r_col to maximize Zone 1C (needs room for barrier 2 traps)
    if trap_r_col < correct_r_col:
        divider_col_z1 = divider_options_z1[0]
    else:
        divider_col_z1 = divider_options_z1[-1]
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, divider_col_z1))

    # --- 7. Barrier 2 columns ---
    # Trap pool: constrained to Zone 1C side of divider
    if trap_r_col < correct_r_col:
        zone1c_cols = set(range(divider_col_z1 + 1, w))
    else:
        zone1c_cols = set(range(0, divider_col_z1))
    trap_pool = [c for c in zone1c_cols
                 if c not in (exit_col, dl_col, bypass_col, divider_col_z1)]
    if len(trap_pool) < 2:
        return None
    rng.shuffle(trap_pool)
    trap_b_col, trap_g_col = trap_pool[0], trap_pool[1]

    # Build barrier 2 walls: openings = {trap_b, trap_g, bypass}
    for c in range(w):
        if c not in (trap_b_col, trap_g_col, bypass_col):
            walls.add(Pos(b2, c))

    # NO pocket walls below barrier 2 — traps lead to open zone 2

    # --- 8. Exit pocket walls ---
    walls.add(pocket_wall_pos)  # (h-2, exit_col) isolates exit from above

    # --- Scaling: extra trap(s) at w>=7 / w>=8 ---
    extra_trap_relay_b1: Optional[int] = None
    extra_trap_relay_b2: Optional[int] = None
    enhanced_relay = w >= 7 and rng.random() < 0.5
    if w >= 7:
        extra_trap_relay_b1 = _pick_extra_trap_col(
            b1, walls, {trap_r_col, correct_r_col}, exit_half, rng)
        if extra_trap_relay_b1 is not None:
            walls.discard(Pos(b1, extra_trap_relay_b1))
    if w >= 8 or enhanced_relay:
        extra_trap_relay_b2 = _pick_extra_trap_col(
            b2, walls, {trap_b_col, trap_g_col, bypass_col}, exit_half, rng)
        if extra_trap_relay_b2 is not None:
            walls.discard(Pos(b2, extra_trap_relay_b2))

    # --- 9. Key placement ---
    # Blue and green keys near matching traps (convergent forks)
    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }
    blue_key_pos = _convergent_key_pos(Pos(b2, trap_b_col), zone1, rng)
    if blue_key_pos is None:
        return None
    green_key_pos = _convergent_key_pos(Pos(b2, trap_g_col), zone1 - {blue_key_pos}, rng)
    if green_key_pos is None:
        return None
    if blue_key_pos in walls or green_key_pos in walls:
        return None

    # Red key in zone 0, biased closer to trap than correct (manhattan)
    trap_r_pos = Pos(b1, trap_r_col)
    correct_r_pos = Pos(b1, correct_r_col)

    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 2:
        return None

    rk_candidates = [
        p for p in zone0
        if manhattan_distance(p, trap_r_pos) < manhattan_distance(p, correct_r_pos)
    ]
    if not rk_candidates:
        rk_candidates = list(zone0)
    red_key_pos = rng.choice(rk_candidates)

    # --- 10. Start: zone 0, biased toward exit side, not on red key ---
    start_candidates = [p for p in zone0 if p != red_key_pos]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - exit_side))
    n_pick = max(1, len(start_candidates) // 2)
    start = rng.choice(start_candidates[:n_pick])

    # --- 11. Connectivity checks ---

    # Zone 0: start, red_key, approach cells for both red doors
    trap_r_approach = Pos(b1 - 1, trap_r_col)
    correct_r_approach = Pos(b1 - 1, correct_r_col)
    if trap_r_approach not in zone0 or correct_r_approach not in zone0:
        return None
    if not _is_connected(zone0):
        return None
    zone0_comp = _flood_fill(start, zone0)
    if red_key_pos not in zone0_comp:
        return None
    if trap_r_approach not in zone0_comp or correct_r_approach not in zone0_comp:
        return None

    # Zone 1C: correct entry -> keys -> bypass (correct side of divider)
    correct_r_entry = Pos(b1 + 1, correct_r_col)
    bypass_above = Pos(b2 - 1, bypass_col)
    if correct_r_entry not in zone1:
        return None
    if blue_key_pos not in zone1 or green_key_pos not in zone1:
        return None
    if bypass_above not in zone1:
        return None
    zone1c_comp = _flood_fill(correct_r_entry, zone1)
    if blue_key_pos not in zone1c_comp or green_key_pos not in zone1c_comp:
        return None
    if bypass_above not in zone1c_comp:
        return None

    # Zone 1T: trap entry reachable, real area (>=2 cells), disconnected from 1C
    trap_r_entry = Pos(b1 + 1, trap_r_col)
    if trap_r_entry not in zone1:
        return None
    zone1t_comp = _flood_fill(trap_r_entry, zone1)
    if len(zone1t_comp) < 2:
        return None
    if zone1t_comp & zone1c_comp:
        return None

    # Zone 2: flood fill excluding DL + exit; verify all barrier 2 entries
    # connected + at least one DL-adjacent cell reachable
    zone2 = {
        Pos(r, c) for r in range(b2 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
        and Pos(r, c) != exit_pos and Pos(r, c) != dl_pos
    }
    bypass_below = Pos(b2 + 1, bypass_col)
    trap_b_below = Pos(b2 + 1, trap_b_col)
    trap_g_below = Pos(b2 + 1, trap_g_col)
    if bypass_below not in zone2:
        return None
    zone2_comp = _flood_fill(bypass_below, zone2)
    # Trap entries must also be reachable (they lead to real zone 2 space)
    if trap_b_below not in zone2_comp or trap_g_below not in zone2_comp:
        return None
    # At least one cell adjacent to the DL must be reachable from zone 2
    dl_neighbors = [
        Pos(dl_pos.row + dr, dl_pos.col + dc)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if 0 <= dl_pos.row + dr < h and 0 <= dl_pos.col + dc < w
    ]
    if not any(n in zone2_comp for n in dl_neighbors):
        return None

    # Exit isolation: exit neighbors must all be wall/DL/OOB
    # (h-2, exit_col) = pocket wall, (h-1, dl_col) = DL, OOB edges
    if pocket_wall_pos not in walls:
        return None

    # --- 12. Optional extra walls (0-2 per zone, connectivity-preserving) ---

    # Zone 0 extras
    zone0_protected = {start, red_key_pos, trap_r_approach, correct_r_approach}
    zone0_extras = [p for p in zone0 if p not in zone0_protected]
    rng.shuffle(zone0_extras)
    n_extra = rng.randint(0, min(2, len(zone0_extras) // 4))
    for pos in zone0_extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone0_protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    # Zone 1C extras (only correct side — Zone 1T is the trap area)
    zone1c_cells = zone1 & zone1c_comp
    zone1_protected = {correct_r_entry, blue_key_pos, green_key_pos, bypass_above}
    zone1_extras = [p for p in zone1c_cells if p not in zone1_protected]
    rng.shuffle(zone1_extras)
    n_extra = rng.randint(0, min(2, len(zone1_extras) // 4))
    for pos in zone1_extras:
        if n_extra <= 0:
            break
        rem = zone1c_cells - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone1_protected):
            walls.add(pos)
            zone1c_cells.discard(pos)
            zone1.discard(pos)
            n_extra -= 1

    # --- 13. Build grid (using permuted color palette) ---
    keys = {
        f"K_{c_gate[0]}": KeyInfo(id=f"K_{c_gate[0]}", pos=red_key_pos, color=c_gate),
        f"K_{c_a[0]}": KeyInfo(id=f"K_{c_a[0]}", pos=blue_key_pos, color=c_a),
        f"K_{c_b[0]}": KeyInfo(id=f"K_{c_b[0]}", pos=green_key_pos, color=c_b),
    }
    doors = {
        "D_gate": DoorInfo(
            id="D_gate", pos=correct_r_pos, required_keys=(c_gate,)),
        "D_trap_gate": DoorInfo(
            id="D_trap_gate", pos=trap_r_pos, required_keys=(c_gate,)),
        f"D_trap_{c_a[0]}": DoorInfo(
            id=f"D_trap_{c_a[0]}", pos=Pos(b2, trap_b_col), required_keys=(c_a,)),
        f"D_trap_{c_b[0]}": DoorInfo(
            id=f"D_trap_{c_b[0]}", pos=Pos(b2, trap_g_col), required_keys=(c_b,)),
        "D_dual": DoorInfo(
            id="D_dual", pos=dl_pos, required_keys=(c_a, c_b)),
    }
    if extra_trap_relay_b1 is not None:
        doors["D_trap_gate2"] = DoorInfo(
            id="D_trap_gate2", pos=Pos(b1, extra_trap_relay_b1), required_keys=(c_gate,))
    if extra_trap_relay_b2 is not None:
        doors["D_trap_extra_b2"] = DoorInfo(
            id="D_trap_extra_b2", pos=Pos(b2, extra_trap_relay_b2), required_keys=(c_a,))
    grid = ChainVaultGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Investment door grid generation (spend 1 key to gain 2)
# ---------------------------------------------------------------------------

def _generate_id_1key(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """1-key investment variant: invest R → yields only K_b (ratio 1:1).

    K_g is on the left side with convergent bait near D_trap_g on the vertical
    barrier. The model must reason about what the investment room actually
    yields rather than assuming "invest always gives 2 keys."

    LAYOUT (same structure as standard, with modified barrier + key placement):
        Left side (cols 0..b-1):     Start + K_r + K_g
        Vertical barrier (col b):    D_invest(R) + D_trap_r(R) + D_trap_g(G)
                                     + Bypass gap
        Right side (cols b+1..w-1):
          Investment room (rows 0..r_split-1): K_b only
          Split wall (row r_split):            D_trap_b only
          Exit area (rows r_split+1..h-1):     Open + DL exit pocket

    ROW ORDERING: r_invest < r_trap_g < r_split < r_trap_r < r_bypass

    DEADLOCKS (≥3):
    1. R on D_trap_r → exit area but no B or G → can't open DL
    2. G on D_trap_g → investment room, get K_b, but no G → can't open DL
    3. B on D_trap_b → exit area with only G → can't open DL
    """
    h, w = height, width
    if w < 6 or h < 7:
        return None

    # --- 1. Barrier column ---
    b_max = w - 3
    if b_max < 2:
        return None
    b = rng.randint(2, b_max)

    # --- 2. Row layout: r_invest < r_trap_g < r_split < r_trap_r < r_bypass ---
    r_bypass = rng.randint(max(5, h - 3), h - 1)

    r_trap_r_max = r_bypass - 1
    if r_trap_r_max < 4:
        return None
    r_trap_r = rng.randint(4, r_trap_r_max)

    r_split_max = r_trap_r - 1
    if r_split_max < 3:
        return None
    r_split = rng.randint(3, r_split_max)

    r_trap_g_max = r_split - 1
    if r_trap_g_max < 1:
        return None
    r_trap_g = rng.randint(1, r_trap_g_max)

    r_invest_max = r_trap_g - 1
    if r_invest_max < 0:
        return None
    r_invest = rng.randint(0, r_invest_max)

    # --- 3. Exit pocket ---
    exit_col = rng.choice([b + 1, w - 1])
    if exit_col == w - 1:
        dl_col = w - 2
    else:
        dl_col = b + 2
    if dl_col < b + 1 or dl_col >= w:
        return None

    exit_pos = Pos(h - 1, exit_col)
    dl_pos = Pos(h - 1, dl_col)
    pocket_wall_pos = Pos(h - 2, exit_col)

    # --- 4. Split wall: single trap column (blue only) ---
    right_cols = list(range(b + 1, w))
    if not right_cols:
        return None
    rng.shuffle(right_cols)
    c_trap_b = right_cols[0]

    # --- 5. Build walls ---
    walls: set[Pos] = set()

    # Vertical barrier
    for r in range(h):
        if r not in (r_invest, r_trap_r, r_trap_g, r_bypass):
            walls.add(Pos(r, b))

    # Split wall (right side only, one opening for D_trap_b)
    for c in range(b + 1, w):
        if c != c_trap_b:
            walls.add(Pos(r_split, c))

    # Exit pocket wall
    walls.add(pocket_wall_pos)

    # --- Scaling: extra traps at w>=7 / w>=8 ---
    extra_trap_barrier: Optional[Pos] = None
    extra_trap_split: Optional[int] = None
    if w >= 7:
        inv_used_rows = {r_invest, r_trap_r, r_trap_g, r_bypass, 0, h - 1}
        inv_wall_rows = [
            r for r in range(h)
            if Pos(r, b) in walls and r not in inv_used_rows
        ]
        if inv_wall_rows:
            er = rng.choice(inv_wall_rows)
            extra_trap_barrier = Pos(er, b)
            walls.discard(extra_trap_barrier)
    if w >= 8:
        split_wall_cols = [
            c for c in range(b + 1, w)
            if Pos(r_split, c) in walls and c != c_trap_b
        ]
        if split_wall_cols:
            ec = rng.choice(split_wall_cols)
            extra_trap_split = ec
            walls.discard(Pos(r_split, ec))

    # --- 6. Key placement ---
    invest_door_pos = Pos(r_invest, b)
    trap_r_pos = Pos(r_trap_r, b)
    trap_g_pos = Pos(r_trap_g, b)

    # K_b in investment room (convergent bait: above D_trap_b on split wall)
    invest_room = {
        Pos(r, c) for r in range(r_split) for c in range(b + 1, w)
        if Pos(r, c) not in walls
    }
    blue_key_pos = _convergent_key_pos(Pos(r_split, c_trap_b), invest_room, rng)
    if blue_key_pos is None:
        return None
    if blue_key_pos in walls:
        return None

    # K_r on left side (biased closer to D_trap_r than D_invest)
    left_side = {
        Pos(r, c) for r in range(h) for c in range(b)
        if Pos(r, c) not in walls
    }
    if len(left_side) < 3:
        return None

    rk_candidates = [
        p for p in left_side
        if manhattan_distance(p, trap_r_pos) < manhattan_distance(p, invest_door_pos)
    ]
    if not rk_candidates:
        rk_candidates = list(left_side)
    red_key_pos = rng.choice(rk_candidates)

    # K_g on left side: convergent bait adjacent to D_trap_g
    # Ideal: directly left of D_trap_g at (r_trap_g, b-1)
    green_key_ideal = Pos(r_trap_g, b - 1)
    if green_key_ideal in left_side and green_key_ideal != red_key_pos:
        green_key_pos = green_key_ideal
    else:
        # Fallback: any left-side cell near D_trap_g
        gk_candidates = [
            p for p in left_side
            if p != red_key_pos
            and abs(p.row - r_trap_g) <= 1 and abs(p.col - b) <= 1
        ]
        if not gk_candidates:
            return None
        green_key_pos = rng.choice(gk_candidates)

    # --- 7. Start: left side, biased toward bottom ---
    start_candidates = [
        p for p in left_side if p != red_key_pos and p != green_key_pos
    ]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: -p.row)
    n_pick = max(1, len(start_candidates) // 2)
    start = rng.choice(start_candidates[:n_pick])

    # --- 8. Connectivity checks ---

    # Left side
    invest_approach = Pos(r_invest, b - 1)
    trap_r_approach = Pos(r_trap_r, b - 1)
    trap_g_approach = Pos(r_trap_g, b - 1)
    bypass_approach = Pos(r_bypass, b - 1)
    left_critical = {start, red_key_pos, green_key_pos, invest_approach,
                     trap_r_approach, trap_g_approach, bypass_approach}
    for p in left_critical:
        if p not in left_side:
            return None
    left_comp = _flood_fill(start, left_side)
    for p in left_critical:
        if p not in left_comp:
            return None

    # Investment room
    invest_entry = Pos(r_invest, b + 1)
    if invest_entry not in invest_room or blue_key_pos not in invest_room:
        return None
    invest_comp = _flood_fill(invest_entry, invest_room)
    if blue_key_pos not in invest_comp:
        return None
    # D_trap_g entry must also reach investment room
    trap_g_entry = Pos(r_trap_g, b + 1)
    if trap_g_entry not in invest_room:
        return None
    if trap_g_entry not in invest_comp:
        return None

    # Exit area
    exit_area = {
        Pos(r, c) for r in range(r_split + 1, h) for c in range(b + 1, w)
        if Pos(r, c) not in walls
        and Pos(r, c) != exit_pos and Pos(r, c) != dl_pos
    }
    trap_r_entry = Pos(r_trap_r, b + 1)
    bypass_entry = Pos(r_bypass, b + 1)
    if trap_r_entry not in exit_area or bypass_entry not in exit_area:
        return None
    exit_area_comp = _flood_fill(bypass_entry, exit_area)
    if trap_r_entry not in exit_area_comp:
        return None
    trap_b_below = Pos(r_split + 1, c_trap_b)
    if trap_b_below not in exit_area_comp:
        return None
    dl_neighbors = [
        Pos(dl_pos.row + dr, dl_pos.col + dc)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if 0 <= dl_pos.row + dr < h and 0 <= dl_pos.col + dc < w
    ]
    if not any(n in exit_area_comp for n in dl_neighbors):
        return None

    # Exit isolation
    if pocket_wall_pos not in walls:
        return None

    # --- 9. Extra walls (0-2 per region) ---
    left_protected = {start, red_key_pos, green_key_pos, invest_approach,
                      trap_r_approach, trap_g_approach, bypass_approach}
    left_extras = [p for p in left_side if p not in left_protected]
    rng.shuffle(left_extras)
    n_extra = rng.randint(0, min(2, len(left_extras) // 4))
    for pos in left_extras:
        if n_extra <= 0:
            break
        rem = left_side - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in left_protected):
            walls.add(pos)
            left_side.discard(pos)
            n_extra -= 1

    invest_protected = {invest_entry, blue_key_pos, trap_g_entry}
    invest_extras = [p for p in invest_room if p not in invest_protected]
    rng.shuffle(invest_extras)
    n_extra = rng.randint(0, min(2, len(invest_extras) // 4))
    for pos in invest_extras:
        if n_extra <= 0:
            break
        rem = invest_room - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in invest_protected):
            walls.add(pos)
            invest_room.discard(pos)
            n_extra -= 1

    # --- 10. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=red_key_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=blue_key_pos, color="blue"),
        "K_g": KeyInfo(id="K_g", pos=green_key_pos, color="green"),
    }
    doors = {
        "D_invest": DoorInfo(
            id="D_invest", pos=invest_door_pos, required_keys=("red",)),
        "D_trap_r": DoorInfo(
            id="D_trap_r", pos=trap_r_pos, required_keys=("red",)),
        "D_trap_g": DoorInfo(
            id="D_trap_g", pos=trap_g_pos, required_keys=("green",)),
        "D_trap_b": DoorInfo(
            id="D_trap_b", pos=Pos(r_split, c_trap_b), required_keys=("blue",)),
        "D_dual": DoorInfo(
            id="D_dual", pos=dl_pos, required_keys=("blue", "green")),
    }
    if extra_trap_barrier is not None:
        doors["D_trap_r2"] = DoorInfo(
            id="D_trap_r2", pos=extra_trap_barrier, required_keys=("red",))
    if extra_trap_split is not None:
        doors["D_trap_split"] = DoorInfo(
            id="D_trap_split", pos=Pos(r_split, extra_trap_split),
            required_keys=("blue",))

    grid = ChainVaultGrid(
        width=w,
        height=h,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def generate_investment_door_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate an investment door grid: spend 1 key to gain 2.

    MECHANIC: Red key → investment door (far from exit) → investment room with
    blue+green keys → return through opened door → left side → bypass →
    exit area → dual-lock → exit.

    LAYOUT:
        Left side (cols 0..b-1):     Start + Red key
        Vertical barrier (col b):    Investment door + Red trap + Bypass gap
        Right side (cols b+1..w-1):
          Investment room (rows 0..r_split-1): Blue key + Green key
          Split wall (row r_split):            Blue trap + Green trap
          Exit area (rows r_split+1..h-1):     Open + DL exit pocket

    ROW ORDERING: r_invest < r_split < r_trap < r_bypass

    ANTI-GREEDY:
    - Red trap closer to exit than investment door (manhattan) — guaranteed by
      row ordering since exit is at bottom.
    - Keys directly above matching split traps (convergent forks pattern).
    - Bypass gap provides free passage but requires backtracking.

    CORRECT PATH: Start → red key → investment door → investment room →
    blue+green keys → return through opened door → left side →
    down to bypass → exit area → DL → exit.

    DEADLOCKS (≥3):
    1. Red on red trap → exit area but no blue/green → can't open DL
    2. Blue on blue split trap → exit area with only green → can't open DL
    3. Green on green split trap → exit area with only blue → can't open DL
    """
    if rng is None:
        rng = random.Random()
    if width < 6 or height < 6:
        return None

    # --- 1-key investment variant (50% at height >= 7) ---
    if height >= 7 and rng.random() < 0.5:
        result = _generate_id_1key(width, height, rng)
        if result is not None:
            return result
        # Fall through to standard 2-key variant

    # P2.9: bad investment trap at w>=7, h>=7
    bad_invest = width >= 7 and height >= 7 and rng.random() < 0.5

    h, w = height, width

    # --- 1. Barrier column: ≥2 cols each side ---
    b_max = w - 3
    if b_max < 2:
        return None
    b = rng.randint(2, b_max)

    # --- 2. Row layout: r_invest < r_split < r_trap < r_bypass ---
    # Minimum: r_invest=0, r_split=2, r_trap=3, r_bypass=4 → needs h≥5
    r_bypass = rng.randint(max(4, h - 3), h - 1)

    r_trap_max = r_bypass - 1
    if r_trap_max < 3:
        return None
    r_trap = rng.randint(3, r_trap_max)

    r_split_max = r_trap - 1
    if r_split_max < 2:
        return None
    r_split = rng.randint(2, r_split_max)

    r_invest_max = r_split - 2
    if r_invest_max < 0:
        return None
    r_invest = rng.randint(0, r_invest_max)

    # --- 3. Exit pocket (random side within right area) ---
    exit_col = rng.choice([b + 1, w - 1])
    if exit_col == w - 1:
        dl_col = w - 2
    else:  # exit_col == b + 1
        dl_col = b + 2
    if dl_col < b + 1 or dl_col >= w:
        return None

    exit_pos = Pos(h - 1, exit_col)
    dl_pos = Pos(h - 1, dl_col)
    pocket_wall_pos = Pos(h - 2, exit_col)

    # --- 4. Split wall trap columns (2 distinct right-side cols) ---
    right_cols = list(range(b + 1, w))
    if len(right_cols) < 2:
        return None
    rng.shuffle(right_cols)
    c_trap_b, c_trap_g = right_cols[0], right_cols[1]

    # --- 5. Build vertical barrier ---
    walls: set[Pos] = set()
    for r in range(h):
        if r not in (r_invest, r_trap, r_bypass):
            walls.add(Pos(r, b))

    # --- 6. Build split wall (right side only) ---
    for c in range(b + 1, w):
        if c not in (c_trap_b, c_trap_g):
            walls.add(Pos(r_split, c))

    # --- 7. Exit pocket wall ---
    walls.add(pocket_wall_pos)

    # --- Scaling: extra trap(s) at w>=7 / w>=8 ---
    extra_trap_inv_barrier: Optional[Pos] = None
    extra_trap_inv_split: Optional[int] = None
    enhanced_inv = w >= 7 and rng.random() < 0.5
    if w >= 7:
        # Extra red trap on vertical barrier (new row)
        inv_used_rows = {r_invest, r_trap, r_bypass, 0, h - 1}
        inv_wall_rows = [
            r for r in range(h)
            if Pos(r, b) in walls and r not in inv_used_rows
        ]
        if inv_wall_rows:
            er = rng.choice(inv_wall_rows)
            extra_trap_inv_barrier = Pos(er, b)
            walls.discard(extra_trap_inv_barrier)
    if w >= 8 or enhanced_inv:
        # Extra trap on split wall
        split_wall_cols = [
            c for c in range(b + 1, w)
            if Pos(r_split, c) in walls and c not in {c_trap_b, c_trap_g}
        ]
        if split_wall_cols:
            ec = rng.choice(split_wall_cols)
            extra_trap_inv_split = ec
            walls.discard(Pos(r_split, ec))

    # --- P2.9: Bad investment door — red door on barrier entering exit area ---
    # Wastes a red key on a door to already-accessible exit area (bypass is free).
    # Pick any barrier wall row in the exit area that isn't already a door or
    # adjacent to r_invest (to avoid looking identical to real investment).
    bad_invest_row: Optional[int] = None
    if bad_invest:
        excluded = {r_invest, r_trap, r_bypass, r_invest - 1, r_invest + 1}
        bad_candidates = [
            r for r in range(r_split + 1, h)
            if Pos(r, b) in walls and r not in excluded
        ]
        if bad_candidates:
            bad_invest_row = rng.choice(bad_candidates)
            walls.discard(Pos(bad_invest_row, b))  # open barrier for door
        else:
            bad_invest = False

    # --- 8. Key placement (convergent forks pattern) ---
    invest_room = {
        Pos(r, c) for r in range(r_split) for c in range(b + 1, w)
        if Pos(r, c) not in walls
    }
    blue_key_pos = _convergent_key_pos(Pos(r_split, c_trap_b), invest_room, rng)
    if blue_key_pos is None:
        return None
    green_key_pos = _convergent_key_pos(Pos(r_split, c_trap_g), invest_room - {blue_key_pos}, rng)
    if green_key_pos is None:
        return None
    if blue_key_pos in walls or green_key_pos in walls:
        return None

    # --- 9. Red key: left side, biased closer to red trap ---
    invest_door_pos = Pos(r_invest, b)
    trap_r_pos = Pos(r_trap, b)

    left_side = {
        Pos(r, c) for r in range(h) for c in range(b)
        if Pos(r, c) not in walls
    }
    if len(left_side) < 2:
        return None

    rk_candidates = [
        p for p in left_side
        if manhattan_distance(p, trap_r_pos) < manhattan_distance(p, invest_door_pos)
    ]
    if not rk_candidates:
        rk_candidates = list(left_side)
    red_key_pos = rng.choice(rk_candidates)

    # --- 10. Start: left side, biased toward bottom (closer to exit) ---
    start_candidates = [p for p in left_side if p != red_key_pos]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: -p.row)
    n_pick = max(1, len(start_candidates) // 2)
    start = rng.choice(start_candidates[:n_pick])

    # --- 11. Connectivity checks ---

    # 11a. Left side: start, red_key, approach cells for barrier openings
    invest_approach = Pos(r_invest, b - 1)
    trap_approach = Pos(r_trap, b - 1)
    bypass_approach = Pos(r_bypass, b - 1)
    left_critical = {start, red_key_pos, invest_approach, trap_approach,
                     bypass_approach}
    if bad_invest and bad_invest_row is not None:
        left_critical.add(Pos(bad_invest_row, b - 1))
    for p in left_critical:
        if p not in left_side:
            return None
    left_comp = _flood_fill(start, left_side)
    for p in left_critical:
        if p not in left_comp:
            return None

    # 11b. Investment room (rows 0..r_split-1, cols b+1..w-1)
    invest_entry = Pos(r_invest, b + 1)
    if invest_entry not in invest_room:
        return None
    if blue_key_pos not in invest_room or green_key_pos not in invest_room:
        return None
    invest_comp = _flood_fill(invest_entry, invest_room)
    if blue_key_pos not in invest_comp or green_key_pos not in invest_comp:
        return None

    # 11c. Exit area (rows r_split+1..h-1, cols b+1..w-1, excl DL + exit)
    exit_area = {
        Pos(r, c) for r in range(r_split + 1, h) for c in range(b + 1, w)
        if Pos(r, c) not in walls
        and Pos(r, c) != exit_pos and Pos(r, c) != dl_pos
    }
    trap_entry = Pos(r_trap, b + 1)
    bypass_entry = Pos(r_bypass, b + 1)
    if trap_entry not in exit_area or bypass_entry not in exit_area:
        return None
    exit_area_comp = _flood_fill(bypass_entry, exit_area)
    if trap_entry not in exit_area_comp:
        return None
    if bad_invest and bad_invest_row is not None:
        bad_invest_entry = Pos(bad_invest_row, b + 1)
        if bad_invest_entry not in exit_area_comp:
            return None

    # Split trap below cells must be reachable in exit area
    trap_b_below = Pos(r_split + 1, c_trap_b)
    trap_g_below = Pos(r_split + 1, c_trap_g)
    if trap_b_below not in exit_area_comp or trap_g_below not in exit_area_comp:
        return None

    # DL-adjacent cell reachable
    dl_neighbors = [
        Pos(dl_pos.row + dr, dl_pos.col + dc)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if 0 <= dl_pos.row + dr < h and 0 <= dl_pos.col + dc < w
    ]
    if not any(n in exit_area_comp for n in dl_neighbors):
        return None

    # 11d. Exit isolation
    if pocket_wall_pos not in walls:
        return None

    # --- 12. Extra walls (0-2 per region, connectivity-preserving) ---

    # Left side extras
    left_protected = {start, red_key_pos, invest_approach, trap_approach,
                      bypass_approach}
    left_extras = [p for p in left_side if p not in left_protected]
    rng.shuffle(left_extras)
    n_extra = rng.randint(0, min(2, len(left_extras) // 4))
    for pos in left_extras:
        if n_extra <= 0:
            break
        rem = left_side - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in left_protected):
            walls.add(pos)
            left_side.discard(pos)
            n_extra -= 1

    # Investment room extras
    invest_protected = {invest_entry, blue_key_pos, green_key_pos}
    invest_extras = [p for p in invest_room if p not in invest_protected]
    rng.shuffle(invest_extras)
    n_extra = rng.randint(0, min(2, len(invest_extras) // 4))
    for pos in invest_extras:
        if n_extra <= 0:
            break
        rem = invest_room - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in invest_protected):
            walls.add(pos)
            invest_room.discard(pos)
            n_extra -= 1

    # --- 13. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=red_key_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=blue_key_pos, color="blue"),
        "K_g": KeyInfo(id="K_g", pos=green_key_pos, color="green"),
    }
    doors = {
        "D_invest": DoorInfo(
            id="D_invest", pos=invest_door_pos, required_keys=("red",)),
        "D_trap_r": DoorInfo(
            id="D_trap_r", pos=trap_r_pos, required_keys=("red",)),
        "D_trap_b": DoorInfo(
            id="D_trap_b", pos=Pos(r_split, c_trap_b), required_keys=("blue",)),
        "D_trap_g": DoorInfo(
            id="D_trap_g", pos=Pos(r_split, c_trap_g), required_keys=("green",)),
        "D_dual": DoorInfo(
            id="D_dual", pos=dl_pos, required_keys=("blue", "green")),
    }
    if bad_invest and bad_invest_row is not None:
        doors["D_bad_invest"] = DoorInfo(
            id="D_bad_invest", pos=Pos(bad_invest_row, b), required_keys=("red",))
    if extra_trap_inv_barrier is not None:
        doors["D_trap_r2"] = DoorInfo(
            id="D_trap_r2", pos=extra_trap_inv_barrier, required_keys=("red",))
    if extra_trap_inv_split is not None:
        doors["D_trap_split"] = DoorInfo(
            id="D_trap_split", pos=Pos(r_split, extra_trap_inv_split),
            required_keys=("blue",))
    grid = ChainVaultGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Switchback grid generation (same color used twice correctly)
# ---------------------------------------------------------------------------


def _generate_sb_3key(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """3-key switchback: 3 red keys, 4 same-color doors (w>=7, h>=7).

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + K1 + K2 + K3 (all red)
        Barrier 1 (row b1):         D_trap (exit-side) + D_A (far-side)
        Zone 1 (rows b1+1..b2-1):  Split: Zone 1T (trap) | Zone 1A (correct)
        Barrier 2 (row b2):         D_B
        Zone 2 (rows b2+1..b3-1):  Open passage
        Barrier 3 (row b3):         D_C
        Zone 3 (rows b3+1..h-1):   Exit

    CORRECT PATH: K1+K2+K3 → D_A → Zone 1A → D_B → Zone 2 → D_C → Exit.
    DEADLOCK: Waste any red on D_trap → only 2 left for 3 doors (A+B+C).
    """
    if width < 7 or height < 7:
        return None

    h, w = height, width

    # --- 1. Barrier rows: need zone0(≥1)+b1+zone1(≥1)+b2+zone2(≥1)+b3+zone3(≥1) = 7 rows min
    b1_max = h - 6
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)

    b2_min, b2_max = b1 + 2, h - 4
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    b3_min, b3_max = b2 + 2, h - 2
    if b3_min > b3_max:
        return None
    b3 = rng.randint(b3_min, b3_max)

    # --- 2. Exit side and column halves ---
    exit_side = rng.choice([0, w - 1])
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))
    if not exit_half or not far_half:
        return None

    # --- 3. Barrier 1: trap (exit-side) + correct (far-side) ---
    trap_col = rng.choice(exit_half)
    far_half_b1 = [c for c in far_half if c != trap_col]
    if not far_half_b1:
        return None
    correct_col = rng.choice(far_half_b1)

    # --- 4. Divider column ---
    lo, hi = min(trap_col, correct_col), max(trap_col, correct_col)
    divider_options = list(range(lo + 1, hi))
    if not divider_options:
        return None
    if correct_col > trap_col:
        divider_col = divider_options[-1]
    else:
        divider_col = divider_options[0]

    # --- 5. Door B column (correct side of divider) ---
    if correct_col > divider_col:
        door_b_candidates = list(range(divider_col + 1, w))
    else:
        door_b_candidates = list(range(0, divider_col))
    if not door_b_candidates:
        return None
    door_b_col = rng.choice(door_b_candidates)

    # --- 6. Door C column (any column) ---
    door_c_col = rng.randint(0, w - 1)

    # --- 7. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1
    for c in range(w):
        if c != trap_col and c != correct_col:
            walls.add(Pos(b1, c))

    # Divider in Zone 1
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, divider_col))

    # Barrier 2
    for c in range(w):
        if c != door_b_col:
            walls.add(Pos(b2, c))

    # Barrier 3
    for c in range(w):
        if c != door_c_col:
            walls.add(Pos(b3, c))

    # --- 8. Exit placement ---
    zone3 = {
        Pos(r, c) for r in range(b3 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    if not zone3:
        return None
    exit_candidates = sorted(zone3, key=lambda p: abs(p.col - exit_side))
    n_pick = max(1, len(exit_candidates) // 2)
    exit_pos = rng.choice(exit_candidates[:n_pick])

    # --- 9. Anti-greedy check ---
    trap_pos = Pos(b1, trap_col)
    correct_pos = Pos(b1, correct_col)
    if not _trap_closer_to_exit(trap_pos, correct_pos, exit_pos):
        return None

    # --- 10. Key placement (3 red keys in Zone 0) ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 4:  # need start + 3 keys
        return None

    # Key 1: convergent bait near trap
    key1_pos = _convergent_key_pos(Pos(b1, trap_col), zone0, rng)
    if key1_pos is None:
        return None

    # Key 2 + Key 3: elsewhere in Zone 0
    k23_candidates = [p for p in zone0 if p != key1_pos]
    k23_result = _select_key_positions(
        k23_candidates, 2, rng,
        already_placed=frozenset({key1_pos}), reference=exit_pos,
    )
    if k23_result is None:
        return None
    key2_pos, key3_pos = k23_result

    # --- 11. Start ---
    start_candidates = [
        p for p in zone0 if p not in {key1_pos, key2_pos, key3_pos}
    ]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - exit_side))
    n_pick = max(1, len(start_candidates) // 2)
    start = rng.choice(start_candidates[:n_pick])

    # --- 12. Connectivity checks ---
    trap_approach = Pos(b1 - 1, trap_col)
    correct_approach = Pos(b1 - 1, correct_col)
    if trap_approach not in zone0 or correct_approach not in zone0:
        return None
    zone0_comp = _flood_fill(start, zone0)
    for p in [key1_pos, key2_pos, key3_pos, trap_approach, correct_approach]:
        if p not in zone0_comp:
            return None

    # Zone 1 split
    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }
    correct_entry = Pos(b1 + 1, correct_col)
    door_b_approach = Pos(b2 - 1, door_b_col)
    if correct_entry not in zone1 or door_b_approach not in zone1:
        return None
    zone1a_comp = _flood_fill(correct_entry, zone1)
    if door_b_approach not in zone1a_comp:
        return None
    trap_entry = Pos(b1 + 1, trap_col)
    if trap_entry not in zone1:
        return None
    zone1t_comp = _flood_fill(trap_entry, zone1)
    if correct_entry in zone1t_comp:
        return None  # divider failed

    # Zone 2: door_b_entry → door_c_approach
    zone2 = {
        Pos(r, c) for r in range(b2 + 1, b3) for c in range(w)
        if Pos(r, c) not in walls
    }
    door_b_entry = Pos(b2 + 1, door_b_col)
    door_c_approach = Pos(b3 - 1, door_c_col)
    if door_b_entry not in zone2 or door_c_approach not in zone2:
        return None
    zone2_comp = _flood_fill(door_b_entry, zone2)
    if door_c_approach not in zone2_comp:
        return None

    # Zone 3: door_c_entry → exit
    door_c_entry = Pos(b3 + 1, door_c_col)
    if door_c_entry not in zone3:
        return None
    zone3_comp = _flood_fill(door_c_entry, zone3)
    if exit_pos not in zone3_comp:
        return None

    # --- 13. Build grid ---
    keys = {
        "K1": KeyInfo(id="K1", pos=key1_pos, color="red"),
        "K2": KeyInfo(id="K2", pos=key2_pos, color="red"),
        "K3": KeyInfo(id="K3", pos=key3_pos, color="red"),
    }
    doors = {
        "D_A": DoorInfo(id="D_A", pos=correct_pos, required_keys=("red",)),
        "D_B": DoorInfo(id="D_B", pos=Pos(b2, door_b_col), required_keys=("red",)),
        "D_C": DoorInfo(id="D_C", pos=Pos(b3, door_c_col), required_keys=("red",)),
        "D_trap": DoorInfo(id="D_trap", pos=trap_pos, required_keys=("red",)),
    }
    grid = ChainVaultGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def generate_switchback_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a switchback grid: same color used twice correctly.

    MECHANIC: 2 red keys → Door A (barrier 1, far-side) → Zone 1A →
    Door B (barrier 2) → Exit.  The trap on barrier 1 (exit-side) leads to
    Zone 1T — a real multi-cell area separated from Zone 1A by an internal
    divider wall.  Wasting a key on the trap leaves insufficient keys for
    both A and B.

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + Key 1 (red) + Key 2 (red)
        Barrier 1 (row b1):         Door A (far-side) + Trap (exit-side) + walls
        Zone 1 (rows b1+1..b2-1):  Split by divider column:
          Zone 1A (correct side):   Connects Door A entry → Door B approach
          Zone 1T (trap side):      Real area, disconnected from Door B
        Barrier 2 (row b2):         Door B (on correct side of divider) + walls
        Zone 2 (rows b2+1..h-1):   Exit

    ANTI-GREEDY (triple bait):
    - Key 1 directly above trap door (convergent forks pattern) — agent picks
      up key and is immediately adjacent to trap → USE_KEY temptation.
    - Zone 1T (trap side) is larger than Zone 1A (correct side) — divider
      biased toward correct_col so trap area looks like the "main" path.
    - Trap on exit-side half (closer to exit column), exit below trap side.
    - manhattan(trap, exit) < manhattan(correct, exit) — hard filter.

    CORRECT PATH: Start → Key 1 + Key 2 → Door A (far-side) → Zone 1A →
    Door B (barrier 2) → Exit.

    DEADLOCKS (≥2):
    1. Agent picks up K1 above trap → opens trap → Zone 1T → back to Zone 0 →
       needs A+B but only 1 key remains → stuck
    2. Multiple inventory states when opening trap → distinct deadlocks
    """
    if rng is None:
        rng = random.Random()
    if width < 5 or height < 5:
        return None

    # P2.10: 3-key, 4-door variant at 7×7+
    if width >= 7 and height >= 7 and rng.random() < 0.5:
        result = _generate_sb_3key(width, height, rng)
        if result is not None:
            return result

    h, w = height, width

    # --- 1. Barrier rows ---
    # Need: zone0(≥1) + b1 + zone1(≥1) + b2 + zone2(≥1)
    b1_max = h - 4
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min, b2_max = b1 + 2, h - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Exit-side direction ---
    exit_side = rng.choice([0, w - 1])

    # --- 3. Column halves (exit-side vs far-side) for barrier 1 ---
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))

    if not exit_half or not far_half:
        return None

    # --- 4. Barrier 1 columns ---
    # Trap on exit-side half, correct Door A on far-side half
    trap_col = rng.choice(exit_half)
    far_half_b1 = [c for c in far_half if c != trap_col]
    if not far_half_b1:
        return None
    correct_col = rng.choice(far_half_b1)

    # --- 5. Divider column between trap and correct ---
    # Splits Zone 1 into Zone 1T (trap side) and Zone 1A (correct side).
    # Biased toward correct_col to maximize Zone 1T — large trap area looks
    # like the "main" path, small Zone 1A looks like a narrow corridor.
    lo, hi = min(trap_col, correct_col), max(trap_col, correct_col)
    divider_options = list(range(lo + 1, hi))
    if not divider_options:
        return None  # trap and correct too close for divider
    # Pick divider closest to correct_col (maximize Zone 1T size)
    if correct_col > trap_col:
        divider_col = divider_options[-1]   # highest, nearest to correct
    else:
        divider_col = divider_options[0]    # lowest, nearest to correct

    # --- 6. Door B column: must be on correct_col's side of divider ---
    if correct_col > divider_col:
        door_b_candidates = list(range(divider_col + 1, w))
    else:
        door_b_candidates = list(range(0, divider_col))
    if not door_b_candidates:
        return None
    door_b_col = rng.choice(door_b_candidates)

    # --- 7. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: walls except trap and correct columns
    for c in range(w):
        if c != trap_col and c != correct_col:
            walls.add(Pos(b1, c))

    # Divider walls through all Zone 1 rows
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, divider_col))

    # Barrier 2: walls except door_b column
    for c in range(w):
        if c != door_b_col:
            walls.add(Pos(b2, c))

    # --- Scaling: extra red trap on barrier 1 at w>=6 ---
    extra_trap_b1: Optional[int] = None
    if w >= 6:
        extra_trap_b1 = _pick_extra_trap_col(
            b1, walls, {trap_col, correct_col}, exit_half, rng)
        if extra_trap_b1 is not None:
            walls.discard(Pos(b1, extra_trap_b1))

    # --- 8. Exit placement: Zone 2, biased toward exit_side ---
    zone2 = {
        Pos(r, c) for r in range(b2 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    if not zone2:
        return None
    exit_candidates = sorted(zone2, key=lambda p: abs(p.col - exit_side))
    n_pick = max(1, len(exit_candidates) // 2)
    exit_pos = rng.choice(exit_candidates[:n_pick])

    # --- 9. Key placement (both red, Zone 0) ---
    trap_pos = Pos(b1, trap_col)
    correct_pos = Pos(b1, correct_col)

    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 3:  # need start + 2 keys
        return None

    # Anti-greedy: trap must be strictly closer to exit than correct door
    if not _trap_closer_to_exit(trap_pos, correct_pos, exit_pos):
        return None

    # Key 1: near trap door (convergent forks bait pattern).
    # Agent picks up key → immediately adjacent to trap → USE_KEY temptation.
    key1_pos = _convergent_key_pos(Pos(b1, trap_col), zone0, rng)
    if key1_pos is None:
        return None

    # Key 2: elsewhere in Zone 0, spaced from Key 1
    k2_candidates = [p for p in zone0 if p != key1_pos]
    if not k2_candidates:
        return None
    k2_result = _select_key_positions(
        k2_candidates, 1, rng,
        already_placed=frozenset({key1_pos}), reference=exit_pos,
    )
    if k2_result is None:
        return None
    key2_pos = k2_result[0]

    # --- 10. Start: Zone 0, biased toward exit_side, not on keys ---
    start_candidates = [p for p in zone0 if p != key1_pos and p != key2_pos]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - exit_side))
    n_pick = max(1, len(start_candidates) // 2)
    start = rng.choice(start_candidates[:n_pick])

    # --- 11. Connectivity checks ---

    # Zone 0: start, key1, key2, approach cells for both doors
    trap_approach = Pos(b1 - 1, trap_col)
    correct_approach = Pos(b1 - 1, correct_col)
    if trap_approach not in zone0 or correct_approach not in zone0:
        return None
    zone0_comp = _flood_fill(start, zone0)
    if key1_pos not in zone0_comp or key2_pos not in zone0_comp:
        return None
    if trap_approach not in zone0_comp or correct_approach not in zone0_comp:
        return None

    # Zone 1 (full set, then check split)
    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }

    # Zone 1A (correct side): correct_entry → door_b_approach
    correct_entry = Pos(b1 + 1, correct_col)
    door_b_approach = Pos(b2 - 1, door_b_col)
    if correct_entry not in zone1 or door_b_approach not in zone1:
        return None
    zone1a_comp = _flood_fill(correct_entry, zone1)
    if door_b_approach not in zone1a_comp:
        return None

    # Zone 1T (trap side): must be accessible, disconnected from Zone 1A,
    # and at least as large — big trap area looks like the "main" path
    trap_entry = Pos(b1 + 1, trap_col)
    if trap_entry not in zone1:
        return None
    zone1t_comp = _flood_fill(trap_entry, zone1)
    if correct_entry in zone1t_comp:
        return None  # divider failed — zones are connected
    if len(zone1t_comp) < len(zone1a_comp):
        return None  # trap area should look like the main path

    # Zone 2: door_b_entry → exit connected
    door_b_entry = Pos(b2 + 1, door_b_col)
    if door_b_entry not in zone2:
        return None
    zone2_comp = _flood_fill(door_b_entry, zone2)
    if exit_pos not in zone2_comp:
        return None

    # --- 12. Extra walls (0-2 per zone, connectivity-preserving) ---

    # Zone 0 extras
    zone0_protected = {start, key1_pos, key2_pos, trap_approach, correct_approach}
    zone0_extras = [p for p in zone0 if p not in zone0_protected]
    rng.shuffle(zone0_extras)
    n_extra = rng.randint(0, min(2, len(zone0_extras) // 4))
    for pos in zone0_extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone0_protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    # Zone 1A extras (leave Zone 1T untouched — natural trap area)
    zone1a = set(zone1a_comp)
    zone1a_protected = {correct_entry, door_b_approach}
    zone1a_extras = [p for p in zone1a if p not in zone1a_protected]
    rng.shuffle(zone1a_extras)
    n_extra = rng.randint(0, min(2, len(zone1a_extras) // 4))
    for pos in zone1a_extras:
        if n_extra <= 0:
            break
        rem = zone1a - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone1a_protected):
            walls.add(pos)
            zone1a.discard(pos)
            n_extra -= 1

    # --- P2.11: Mixed-color pair — blue door on barrier 2 + blue trap ---
    mixed_color = w >= 6 and rng.random() < 0.5
    blue_key_pos: Optional[Pos] = None
    blue_trap_col: Optional[int] = None
    if mixed_color:
        # Blue key in Zone 0 (near trap for convergent bait)
        bk_candidates = [
            p for p in zone0
            if p not in {start, key1_pos, key2_pos}
        ]
        if bk_candidates:
            # Place near a barrier 1 wall col for trap bait
            blue_trap_candidates = [
                c for c in range(w)
                if Pos(b1, c) in walls and c != trap_col and c != correct_col
            ]
            if blue_trap_candidates:
                blue_trap_col = rng.choice(blue_trap_candidates)
                walls.discard(Pos(b1, blue_trap_col))
                # Blue key near blue trap (convergent bait)
                bk_near = [
                    p for p in bk_candidates
                    if abs(p.col - blue_trap_col) <= 1
                    and p.row == b1 - 1
                ]
                if bk_near:
                    blue_key_pos = rng.choice(bk_near)
                else:
                    blue_key_pos = rng.choice(bk_candidates)
                # Verify zone0 connectivity with blue key
                if blue_key_pos not in _flood_fill(start, zone0):
                    blue_key_pos = None
            if blue_key_pos is None:
                # Revert blue trap
                if blue_trap_col is not None:
                    walls.add(Pos(b1, blue_trap_col))
                    blue_trap_col = None
                mixed_color = False
        else:
            mixed_color = False

    # --- 13. Build grid ---
    keys = {
        "K1": KeyInfo(id="K1", pos=key1_pos, color="red"),
        "K2": KeyInfo(id="K2", pos=key2_pos, color="red"),
    }
    if mixed_color and blue_key_pos is not None:
        keys["K_b"] = KeyInfo(id="K_b", pos=blue_key_pos, color="blue")
    db_color = "blue" if mixed_color else "red"
    doors = {
        "D_A": DoorInfo(
            id="D_A", pos=correct_pos, required_keys=("red",)),
        "D_B": DoorInfo(
            id="D_B", pos=Pos(b2, door_b_col), required_keys=(db_color,)),
        "D_trap": DoorInfo(
            id="D_trap", pos=trap_pos, required_keys=("red",)),
    }
    if extra_trap_b1 is not None:
        doors["D_trap2"] = DoorInfo(
            id="D_trap2", pos=Pos(b1, extra_trap_b1), required_keys=("red",))
    if mixed_color and blue_trap_col is not None:
        doors["D_trap_b"] = DoorInfo(
            id="D_trap_b", pos=Pos(b1, blue_trap_col), required_keys=("blue",))
    grid = ChainVaultGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Scarce resource grid generation (asymmetric scarcity trap)
# ---------------------------------------------------------------------------


def _generate_sr_ambiguous(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Ambiguous scarcity variant: two scarce colors, both critical (w>=6).

    MECHANIC: 3R + 1B + 1G keys. Red is abundant (1 spare). Blue and green
    are both scarce (1 key each). Exit requires B+G dual-lock. Both blue and
    green traps on barrier 1 with convergent bait. The model can't identify a
    single bottleneck color by counting — it must determine that BOTH scarce
    colors are critical for the exit.

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + K_r1 + K_r2 + K_r3 + K_b + K_g
                                     K_b above D_trap_b, K_g above D_trap_g
        Barrier 1 (row b1):         D_r1 (far-side) + D_trap_b + D_trap_g
        Zone 1 (rows b1+1..b2-1):  Open passage
        Red Gate (row b2 = h-2):   D_r2 (only opening)
        Exit Row (row h-1):        D_exit (B+G dual-lock) + Exit (pocket)

    DEADLOCKS (≥2):
    1. Waste B on D_trap_b → no B for dual-lock
    2. Waste G on D_trap_g → no G for dual-lock
    """
    h, w = height, width
    if w < 6 or h < 5:
        return None

    # --- 1. Barrier rows ---
    b2 = h - 2
    b1_max = b2 - 2
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)

    # --- 2. Exit pocket ---
    exit_side = rng.choice([0, w - 1])
    if exit_side == 0:
        exit_col, dual_col = 0, 1
    else:
        exit_col, dual_col = w - 1, w - 2

    exit_pos = Pos(h - 1, exit_col)
    dual_pos = Pos(h - 1, dual_col)

    # --- 3. Column halves ---
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))

    if not exit_half or not far_half:
        return None

    # --- 4. Barrier 1: D_r1 (far) + D_trap_b + D_trap_g (exit-half) ---
    trap_b_col_candidates = [c for c in exit_half]
    if len(trap_b_col_candidates) < 2:
        return None
    rng.shuffle(trap_b_col_candidates)
    trap_b_col = trap_b_col_candidates[0]
    trap_g_col_candidates = [c for c in exit_half if c != trap_b_col]
    if not trap_g_col_candidates:
        return None
    trap_g_col = rng.choice(trap_g_col_candidates)

    far_half_b1 = [c for c in far_half if c != trap_b_col and c != trap_g_col]
    if not far_half_b1:
        return None
    r1_col = rng.choice(far_half_b1)

    # --- 5. Red gate column ---
    r2_candidates = [c for c in range(w) if c != exit_col and c != dual_col]
    if not r2_candidates:
        return None
    r2_col = rng.choice(r2_candidates)

    # --- 6. Anti-greedy: both traps closer to exit than correct door ---
    trap_b_pos = Pos(b1, trap_b_col)
    trap_g_pos = Pos(b1, trap_g_col)
    r1_pos = Pos(b1, r1_col)
    if not _trap_closer_to_exit(trap_b_pos, r1_pos, exit_pos):
        return None
    if not _trap_closer_to_exit(trap_g_pos, r1_pos, exit_pos):
        return None

    # --- 7. Build walls ---
    walls: set[Pos] = set()

    for c in range(w):
        if c not in (r1_col, trap_b_col, trap_g_col):
            walls.add(Pos(b1, c))

    for c in range(w):
        if c != r2_col:
            walls.add(Pos(b2, c))

    # --- 8. Key placement ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }

    kb_pos = _convergent_key_pos(Pos(b1, trap_b_col), zone0, rng)
    if kb_pos is None:
        return None
    kg_pos = _convergent_key_pos(Pos(b1, trap_g_col), zone0 - {kb_pos}, rng)
    if kg_pos is None:
        return None

    kr_candidates = [p for p in zone0 if p != kb_pos and p != kg_pos]
    if len(kr_candidates) < 3:
        return None
    kr_result = _select_key_positions(
        kr_candidates, 3, rng,
        already_placed=frozenset({kb_pos, kg_pos}), reference=exit_pos,
    )
    if kr_result is None:
        return None
    kr1_pos, kr2_pos, kr3_pos = kr_result

    # --- 9. Start ---
    key_positions = {kb_pos, kg_pos, kr1_pos, kr2_pos, kr3_pos}
    start_candidates = [p for p in zone0 if p not in key_positions]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - exit_side))
    n_pick = max(1, len(start_candidates) // 2)
    start = rng.choice(start_candidates[:n_pick])

    # --- 10. Connectivity checks ---
    trap_b_approach = Pos(b1 - 1, trap_b_col)
    trap_g_approach = Pos(b1 - 1, trap_g_col)
    r1_approach = Pos(b1 - 1, r1_col)
    if any(p not in zone0 for p in (trap_b_approach, trap_g_approach, r1_approach)):
        return None
    zone0_comp = _flood_fill(start, zone0)
    zone0_critical = {kb_pos, kg_pos, kr1_pos, kr2_pos, kr3_pos,
                      trap_b_approach, trap_g_approach, r1_approach}
    for p in zone0_critical:
        if p not in zone0_comp:
            return None

    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }
    r1_entry = Pos(b1 + 1, r1_col)
    trap_b_entry = Pos(b1 + 1, trap_b_col)
    trap_g_entry = Pos(b1 + 1, trap_g_col)
    r2_approach = Pos(b2 - 1, r2_col)
    if any(p not in zone1 for p in (r1_entry, trap_b_entry, trap_g_entry, r2_approach)):
        return None
    zone1_comp = _flood_fill(r1_entry, zone1)
    for p in (trap_b_entry, trap_g_entry, r2_approach):
        if p not in zone1_comp:
            return None

    # Exit row pocket isolation
    if Pos(b2, exit_col) not in walls or Pos(b2, dual_col) not in walls:
        return None

    # --- 11. Extra walls ---
    zone0_protected = {start, kb_pos, kg_pos, kr1_pos, kr2_pos, kr3_pos,
                       trap_b_approach, trap_g_approach, r1_approach}
    zone0_extras = [p for p in zone0 if p not in zone0_protected]
    rng.shuffle(zone0_extras)
    n_extra = rng.randint(0, min(2, len(zone0_extras) // 4))
    for pos in zone0_extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone0_protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    zone1_protected = {r1_entry, trap_b_entry, trap_g_entry, r2_approach}
    zone1_extras = [p for p in zone1 if p not in zone1_protected]
    rng.shuffle(zone1_extras)
    n_extra = rng.randint(0, min(2, len(zone1_extras) // 4))
    for pos in zone1_extras:
        if n_extra <= 0:
            break
        rem = zone1 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone1_protected):
            walls.add(pos)
            zone1.discard(pos)
            n_extra -= 1

    # --- 12. Build grid ---
    keys = {
        "K_r1": KeyInfo(id="K_r1", pos=kr1_pos, color="red"),
        "K_r2": KeyInfo(id="K_r2", pos=kr2_pos, color="red"),
        "K_r3": KeyInfo(id="K_r3", pos=kr3_pos, color="red"),
        "K_b":  KeyInfo(id="K_b",  pos=kb_pos,  color="blue"),
        "K_g":  KeyInfo(id="K_g",  pos=kg_pos,  color="green"),
    }
    doors = {
        "D_r1": DoorInfo(id="D_r1", pos=r1_pos, required_keys=("red",)),
        "D_r2": DoorInfo(id="D_r2", pos=Pos(b2, r2_col), required_keys=("red",)),
        "D_trap_b": DoorInfo(id="D_trap_b", pos=trap_b_pos, required_keys=("blue",)),
        "D_trap_g": DoorInfo(id="D_trap_g", pos=trap_g_pos, required_keys=("green",)),
        "D_exit": DoorInfo(id="D_exit", pos=dual_pos, required_keys=("blue", "green")),
    }
    grid = ChainVaultGrid(
        width=w,
        height=h,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def _generate_sr_2color(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Two-color bottleneck: 1B+1G keys, each has trap+correct door (w>=7,h>=7).

    LAYOUT:
        Zone 0 (rows 0..b1-1):    Start + K_r1 + K_r2 + K_b + K_g
        Barrier 1 (row b1):        D_r1 + D_trap_b + D_trap_g
        Zone 1 (rows b1+1..b2-1): Open passage
        Barrier 2 (row b2):        D_correct_b + D_correct_g
        Zone 2 (rows b2+1..b3-1): Open passage
        Barrier 3 (row b3):        D_r2
        Zone 3 (rows b3+1..h-1):  Exit

    CORRECT: K_r1+K_r2+K_b+K_g → D_r1 → Zone 1 → D_correct_b + D_correct_g →
    Zone 2 → D_r2 → Exit.

    DEADLOCKS: Waste B on D_trap_b → can't open D_correct_b → stuck.
               Waste G on D_trap_g → can't open D_correct_g → stuck.
    """
    h, w = height, width
    if w < 7 or h < 7:
        return None

    # --- 1. Three barrier rows ---
    b1_max = h - 6
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min, b2_max = b1 + 2, h - 4
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)
    b3_min, b3_max = b2 + 2, h - 2
    if b3_min > b3_max:
        return None
    b3 = rng.randint(b3_min, b3_max)

    # --- 2. Exit side ---
    exit_side = rng.choice([0, w - 1])
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))
    if not exit_half or not far_half:
        return None

    # --- 3. Barrier 1: D_r1(far) + D_trap_b + D_trap_g (exit half) ---
    if len(exit_half) < 2:
        return None
    rng.shuffle(exit_half)
    trap_b_col, trap_g_col = exit_half[0], exit_half[1]
    far_candidates = [c for c in far_half if c != trap_b_col and c != trap_g_col]
    if not far_candidates:
        return None
    r1_col = rng.choice(far_candidates)

    # --- 4. Barrier 2: D_correct_b + D_correct_g ---
    b2_cols = list(range(w))
    rng.shuffle(b2_cols)
    correct_b_col = b2_cols[0]
    correct_g_col = b2_cols[1] if len(b2_cols) > 1 else None
    if correct_g_col is None:
        return None

    # --- 5. Barrier 3: D_r2 ---
    r2_col = rng.randint(0, w - 1)

    # --- 6. Exit ---
    exit_col = rng.choice([exit_side])
    exit_pos = Pos(h - 1, exit_col)

    # --- 7. Anti-greedy ---
    trap_b_pos = Pos(b1, trap_b_col)
    trap_g_pos = Pos(b1, trap_g_col)
    r1_pos = Pos(b1, r1_col)
    if not _trap_closer_to_exit(trap_b_pos, r1_pos, exit_pos):
        return None
    if not _trap_closer_to_exit(trap_g_pos, r1_pos, exit_pos):
        return None

    # --- 8. Build walls ---
    walls: set[Pos] = set()
    for c in range(w):
        if c not in (r1_col, trap_b_col, trap_g_col):
            walls.add(Pos(b1, c))
    for c in range(w):
        if c not in (correct_b_col, correct_g_col):
            walls.add(Pos(b2, c))
    for c in range(w):
        if c != r2_col:
            walls.add(Pos(b3, c))

    # --- 9. Key placement ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 5:
        return None

    kb_pos = _convergent_key_pos(Pos(b1, trap_b_col), zone0, rng)
    if kb_pos is None:
        return None
    kg_pos = _convergent_key_pos(Pos(b1, trap_g_col), zone0 - {kb_pos}, rng)
    if kg_pos is None:
        return None

    kr_cands = [p for p in zone0 if p not in {kb_pos, kg_pos}]
    if len(kr_cands) < 2:
        return None
    kr_result = _select_key_positions(
        kr_cands, 2, rng,
        already_placed=frozenset({kb_pos, kg_pos}), reference=exit_pos,
    )
    if kr_result is None:
        return None
    kr1_pos, kr2_pos = kr_result

    # --- 10. Start ---
    key_positions = {kb_pos, kg_pos, kr1_pos, kr2_pos}
    start_candidates = [p for p in zone0 if p not in key_positions]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - exit_side))
    n_pick = max(1, len(start_candidates) // 2)
    start = rng.choice(start_candidates[:n_pick])

    # --- 11. Connectivity ---
    trap_b_approach = Pos(b1 - 1, trap_b_col)
    trap_g_approach = Pos(b1 - 1, trap_g_col)
    r1_approach = Pos(b1 - 1, r1_col)
    z0_critical = {kb_pos, kg_pos, kr1_pos, kr2_pos,
                   trap_b_approach, trap_g_approach, r1_approach}
    for p in z0_critical:
        if p not in zone0:
            return None
    z0_comp = _flood_fill(start, zone0)
    for p in z0_critical:
        if p not in z0_comp:
            return None

    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }
    r1_entry = Pos(b1 + 1, r1_col)
    cb_approach = Pos(b2 - 1, correct_b_col)
    cg_approach = Pos(b2 - 1, correct_g_col)
    for p in (r1_entry, cb_approach, cg_approach):
        if p not in zone1:
            return None
    z1_comp = _flood_fill(r1_entry, zone1)
    for p in (cb_approach, cg_approach):
        if p not in z1_comp:
            return None

    zone2 = {
        Pos(r, c) for r in range(b2 + 1, b3) for c in range(w)
        if Pos(r, c) not in walls
    }
    # Either correct door entry reaches r2 approach
    cb_entry = Pos(b2 + 1, correct_b_col)
    cg_entry = Pos(b2 + 1, correct_g_col)
    r2_approach = Pos(b3 - 1, r2_col)
    for p in (cb_entry, cg_entry, r2_approach):
        if p not in zone2:
            return None
    z2_comp = _flood_fill(cb_entry, zone2)
    if cg_entry not in z2_comp or r2_approach not in z2_comp:
        return None

    zone3 = {
        Pos(r, c) for r in range(b3 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    r2_entry = Pos(b3 + 1, r2_col)
    if r2_entry not in zone3:
        return None
    z3_comp = _flood_fill(r2_entry, zone3)
    if exit_pos not in z3_comp:
        return None

    # --- 12. Build grid ---
    keys = {
        "K_r1": KeyInfo(id="K_r1", pos=kr1_pos, color="red"),
        "K_r2": KeyInfo(id="K_r2", pos=kr2_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=kb_pos, color="blue"),
        "K_g": KeyInfo(id="K_g", pos=kg_pos, color="green"),
    }
    doors = {
        "D_r1": DoorInfo(id="D_r1", pos=r1_pos, required_keys=("red",)),
        "D_r2": DoorInfo(id="D_r2", pos=Pos(b3, r2_col), required_keys=("red",)),
        "D_trap_b": DoorInfo(id="D_trap_b", pos=trap_b_pos, required_keys=("blue",)),
        "D_trap_g": DoorInfo(id="D_trap_g", pos=trap_g_pos, required_keys=("green",)),
        "D_correct_b": DoorInfo(id="D_correct_b", pos=Pos(b2, correct_b_col),
                                required_keys=("blue",)),
        "D_correct_g": DoorInfo(id="D_correct_g", pos=Pos(b2, correct_g_col),
                                required_keys=("green",)),
    }
    grid = ChainVaultGrid(
        width=w, height=h,
        walls=frozenset(walls),
        keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start, inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def generate_scarce_resource_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a scarce resource grid: asymmetric scarcity trap.

    MECHANIC: 3 red keys + 1 blue key, 2 red doors + 1 blue correct door +
    1 blue trap door.  Red is abundant (1 spare key), blue is the bottleneck
    (1 key, 2 blue doors — only 1 leads forward).  The model must identify
    blue as the limiting constraint rather than focusing on the more numerous
    red keys.

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + K_r1 + K_r2 + K_r3 + K_b
                                     K_b directly above D_trap_b
        Barrier 1 (row b1):         D_r1 (far-side) + D_trap_b (exit-side) + walls
                                     Both openings lead to same Zone 1
        Zone 1 (rows b1+1..b2-1):  Open passage (no divider)
        Red Gate (row b2 = h-2):   D_r2 (only opening) + walls
        Exit Row (row h-1):        D_correct_b (blue) + Exit (pocket)

    ANTI-GREEDY:
    - D_trap_b on exit-side half (closer to exit)
    - D_r1 on far-side half (farther from exit)
    - K_b directly above D_trap_b → convergent forks bait
    - manhattan(D_trap_b, exit) < manhattan(D_r1, exit) — hard filter
    - Both barrier 1 doors lead to SAME Zone 1 — trap provides real passage
    - Red abundance (3 keys for 2 doors) is a distraction

    CORRECT PATH: Start → collect all keys → D_r1 (1 red) → Zone 1 →
    D_r2 (1 red) → Exit Row → D_correct_b (1 blue) → Exit.
    Uses: 2 red + 1 blue = 3 of 4 keys. 1 red key spare.

    DEADLOCKS (≥1):
    Agent opens D_trap_b with blue key → Zone 1 → D_r2 (red) → Exit Row →
    D_correct_b needs blue → 0 blue keys → stuck.
    """
    if rng is None:
        rng = random.Random()
    if width < 5 or height < 5:
        return None

    # P2.12: Two-color bottleneck at 7×7+
    if width >= 7 and height >= 7 and rng.random() < 0.5:
        result = _generate_sr_2color(width, height, rng)
        if result is not None:
            return result

    # --- Ambiguous scarcity variant (50% at width >= 6) ---
    if width >= 6 and rng.random() < 0.5:
        result = _generate_sr_ambiguous(width, height, rng)
        if result is not None:
            return result
        # Fall through to standard variant

    h, w = height, width

    # --- 1. Barrier rows ---
    # b2 = h - 2 (red gate row, fixed for exit pocket)
    b2 = h - 2
    b1_max = b2 - 2
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)

    # --- 2. Exit pocket ---
    exit_side = rng.choice([0, w - 1])
    if exit_side == 0:
        exit_col, blue_gate_col = 0, 1
    else:
        exit_col, blue_gate_col = w - 1, w - 2

    exit_pos = Pos(h - 1, exit_col)
    blue_gate_pos = Pos(h - 1, blue_gate_col)  # D_correct_b position

    # --- 3. Column halves for barrier 1 ---
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))

    if not exit_half or not far_half:
        return None

    # --- 4. Barrier 1 door columns ---
    # Trap blue door on exit-side half, D_r1 on far-side half
    trap_b_col = rng.choice(exit_half)
    far_half_b1 = [c for c in far_half if c != trap_b_col]
    if not far_half_b1:
        return None
    r1_col = rng.choice(far_half_b1)

    # --- 5. Red gate column (r2_col ≠ exit_col, ≠ blue_gate_col) ---
    # Ensures pocket isolation: (b2, exit_col) and (b2, blue_gate_col) are walls
    r2_candidates = [c for c in range(w) if c != exit_col and c != blue_gate_col]
    if not r2_candidates:
        return None
    r2_col = rng.choice(r2_candidates)

    # --- 6. Anti-greedy manhattan check ---
    trap_b_pos = Pos(b1, trap_b_col)
    r1_pos = Pos(b1, r1_col)
    if not _trap_closer_to_exit(trap_b_pos, r1_pos, exit_pos):
        return None

    # --- 7. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: walls except r1_col and trap_b_col
    for c in range(w):
        if c != r1_col and c != trap_b_col:
            walls.add(Pos(b1, c))

    # Red gate (row b2 = h-2): walls except r2_col
    for c in range(w):
        if c != r2_col:
            walls.add(Pos(b2, c))
    # (b2, exit_col) is a wall since r2_col ≠ exit_col → pocket wall for free

    # --- Scaling: extra red trap on barrier 1 at w>=6 ---
    extra_trap_b1: Optional[int] = None
    if w >= 6:
        extra_trap_b1 = _pick_extra_trap_col(
            b1, walls, {r1_col, trap_b_col}, exit_half, rng)
        if extra_trap_b1 is not None:
            walls.discard(Pos(b1, extra_trap_b1))

    # --- 8. Key placement ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }

    # K_b: near blue trap (convergent forks bait)
    kb_pos = _convergent_key_pos(Pos(b1, trap_b_col), zone0, rng)
    if kb_pos is None:
        return None

    # K_r1, K_r2, K_r3: spaced positions in Zone 0, away from K_b
    kr_candidates = [p for p in zone0 if p != kb_pos]
    if len(kr_candidates) < 3:
        return None
    kr_result = _select_key_positions(
        kr_candidates, 3, rng,
        already_placed=frozenset({kb_pos}), reference=exit_pos,
    )
    if kr_result is None:
        return None
    kr1_pos, kr2_pos, kr3_pos = kr_result

    # --- 9. Start: Zone 0, biased toward exit_side, not on keys ---
    key_positions = {kb_pos, kr1_pos, kr2_pos, kr3_pos}
    start_candidates = [p for p in zone0 if p not in key_positions]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - exit_side))
    n_pick = max(1, len(start_candidates) // 2)
    start = rng.choice(start_candidates[:n_pick])

    # --- 10. Connectivity checks ---

    # Zone 0: start, all keys, approach cells for both barrier 1 doors
    trap_approach = Pos(b1 - 1, trap_b_col)
    r1_approach = Pos(b1 - 1, r1_col)
    if trap_approach not in zone0 or r1_approach not in zone0:
        return None
    zone0_comp = _flood_fill(start, zone0)
    zone0_critical = {kb_pos, kr1_pos, kr2_pos, kr3_pos,
                      trap_approach, r1_approach}
    for p in zone0_critical:
        if p not in zone0_comp:
            return None

    # Zone 1: r1_entry, trap_entry, r2_approach — all connected (no divider)
    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }
    r1_entry = Pos(b1 + 1, r1_col)
    trap_entry = Pos(b1 + 1, trap_b_col)
    r2_approach = Pos(b2 - 1, r2_col)
    if r1_entry not in zone1 or trap_entry not in zone1 or r2_approach not in zone1:
        return None
    zone1_comp = _flood_fill(r1_entry, zone1)
    if trap_entry not in zone1_comp or r2_approach not in zone1_comp:
        return None

    # Exit row pocket isolation: walls above exit and blue gate from red gate barrier
    if Pos(b2, exit_col) not in walls:
        return None
    if Pos(b2, blue_gate_col) not in walls:
        return None

    # --- 11. Extra walls (0-2 per zone, connectivity-preserving) ---

    # Zone 0 extras
    zone0_protected = {start, kb_pos, kr1_pos, kr2_pos, kr3_pos,
                       trap_approach, r1_approach}
    zone0_extras = [p for p in zone0 if p not in zone0_protected]
    rng.shuffle(zone0_extras)
    n_extra = rng.randint(0, min(2, len(zone0_extras) // 4))
    for pos in zone0_extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone0_protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    # Zone 1 extras
    zone1_protected = {r1_entry, trap_entry, r2_approach}
    zone1_extras = [p for p in zone1 if p not in zone1_protected]
    rng.shuffle(zone1_extras)
    n_extra = rng.randint(0, min(2, len(zone1_extras) // 4))
    for pos in zone1_extras:
        if n_extra <= 0:
            break
        rem = zone1 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone1_protected):
            walls.add(pos)
            zone1.discard(pos)
            n_extra -= 1

    # --- 12. Build grid ---
    keys = {
        "K_r1": KeyInfo(id="K_r1", pos=kr1_pos, color="red"),
        "K_r2": KeyInfo(id="K_r2", pos=kr2_pos, color="red"),
        "K_r3": KeyInfo(id="K_r3", pos=kr3_pos, color="red"),
        "K_b":  KeyInfo(id="K_b",  pos=kb_pos,  color="blue"),
    }
    doors = {
        "D_r1": DoorInfo(
            id="D_r1", pos=r1_pos, required_keys=("red",)),
        "D_r2": DoorInfo(
            id="D_r2", pos=Pos(b2, r2_col), required_keys=("red",)),
        "D_trap_b": DoorInfo(
            id="D_trap_b", pos=trap_b_pos, required_keys=("blue",)),
        "D_correct_b": DoorInfo(
            id="D_correct_b", pos=blue_gate_pos, required_keys=("blue",)),
    }
    if extra_trap_b1 is not None:
        doors["D_trap_r"] = DoorInfo(
            id="D_trap_r", pos=Pos(b1, extra_trap_b1), required_keys=("red",))
    grid = ChainVaultGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        keys=keys,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Cascade trap grid generation (template 10)
# ---------------------------------------------------------------------------

def _generate_ct_depth2(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Depth-2 cascade trap: R waste → no K_y → no D_y → no K_b → no D_b → stuck.

    MECHANIC: 1 red + 1 yellow + 1 blue key (via depth-2 gating).
    Waste red → can't enter Zone 1C → K_y unreachable → D_y locked →
    K_b unreachable → D_b locked → exit unreachable.

    Extends the cause-symptom gap for RQ3 causal attribution.

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + K_r (near D_trap_r)
        Barrier 1 (row b1):         D_correct_r (far) + D_trap_r (exit-side)
        Zone 1 (rows b1+1..b2-1):  Split by divider:
          Zone 1C (correct side):   K_y
          Zone 1T (trap side):      Dead end (≥ Zone 1C size)
        Barrier 2 (row b2):         D_y (yellow) + D_b (blue) [correct side]
        Below barrier 2:
          Zone 2A (behind D_y):     K_b
          Zone 2B (behind D_b):     Exit
          Separated by vertical divider

    DEADLOCKS (≥2):
    1. Waste R with {R only} → Zone 1T → no K_y → no K_b → stuck
    2. Waste R with {R, Y, B} → Zone 1T → has Y and B but can't reach D_y/D_b
    """
    h, w = height, width
    if w < 6 or h < 6:
        return None

    # --- 1. Barrier rows (Zone 1 ≥ 1 row, below-b2 ≥ 1 row) ---
    b1_max = h - 4
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min, b2_max = b1 + 2, h - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Exit-side direction ---
    exit_side = rng.choice([0, w - 1])

    # --- 3. Column halves ---
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))

    if not exit_half or not far_half:
        return None

    # --- 4. Barrier 1 columns ---
    trap_col = rng.choice(exit_half)
    far_half_b1 = [c for c in far_half if c != trap_col]
    if not far_half_b1:
        return None
    correct_col = rng.choice(far_half_b1)

    # --- 5. Zone 1 divider (biased toward correct_col to maximize Zone 1T) ---
    lo, hi = min(trap_col, correct_col), max(trap_col, correct_col)
    z1_div_options = list(range(lo + 1, hi))
    if not z1_div_options:
        return None
    z1_div_col = rng.choice(z1_div_options)

    # --- 6. Barrier 2: D_y + D_b on correct side, separated by ≥2 ---
    if correct_col > z1_div_col:
        correct_side_cols = list(range(z1_div_col + 1, w))
    else:
        correct_side_cols = list(range(0, z1_div_col))

    # Need 2 door columns + 1 divider between them → at least 3 cols
    if len(correct_side_cols) < 3:
        return None

    rng.shuffle(correct_side_cols)
    dy_col: Optional[int] = None
    db_col: Optional[int] = None
    for i, c1 in enumerate(correct_side_cols):
        for c2 in correct_side_cols[i + 1:]:
            if abs(c1 - c2) >= 2:
                dy_col, db_col = c1, c2
                break
        if dy_col is not None:
            break
    if dy_col is None or db_col is None:
        return None

    # Below-barrier-2 divider
    lo2, hi2 = min(dy_col, db_col), max(dy_col, db_col)
    b2_div_options = list(range(lo2 + 1, hi2))
    if not b2_div_options:
        return None
    b2_div_col = rng.choice(b2_div_options)

    # --- 7. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1
    for c in range(w):
        if c not in (trap_col, correct_col):
            walls.add(Pos(b1, c))

    # Zone 1 divider
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, z1_div_col))

    # Barrier 2 (openings only at dy_col and db_col)
    for c in range(w):
        if c not in (dy_col, db_col):
            walls.add(Pos(b2, c))

    # Below-barrier-2 divider
    for r in range(b2 + 1, h):
        walls.add(Pos(r, b2_div_col))

    # --- 8. Anti-greedy manhattan check ---
    trap_pos = Pos(b1, trap_col)
    correct_pos = Pos(b1, correct_col)
    # Exit will be in Zone 2B; use approximate exit position for check
    approx_exit = Pos(h - 1, trap_col)
    if not _trap_closer_to_exit(trap_pos, correct_pos, approx_exit):
        return None

    # --- 9. Key placement ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 2:
        return None

    kr_pos = _convergent_key_pos(Pos(b1, trap_col), zone0, rng)
    if kr_pos is None:
        return None

    # K_y in Zone 1C
    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }
    correct_entry = Pos(b1 + 1, correct_col)
    if correct_entry not in zone1:
        return None
    zone1c_comp = _flood_fill(correct_entry, zone1)
    ky_candidates = list(zone1c_comp)
    if not ky_candidates:
        return None
    rng.shuffle(ky_candidates)
    ky_pos = ky_candidates[0]

    # K_b in Zone 2A (behind D_y)
    if dy_col < b2_div_col:
        z2a_cols = range(0, b2_div_col)
    else:
        z2a_cols = range(b2_div_col + 1, w)
    zone2a = {
        Pos(r, c) for r in range(b2 + 1, h) for c in z2a_cols
        if Pos(r, c) not in walls
    }
    dy_below = Pos(b2 + 1, dy_col)
    if dy_below not in zone2a:
        return None
    z2a_comp = _flood_fill(dy_below, zone2a)
    if not z2a_comp:
        return None
    kb_candidates = list(z2a_comp)
    rng.shuffle(kb_candidates)
    kb_pos = kb_candidates[0]

    # Exit in Zone 2B (behind D_b)
    if db_col < b2_div_col:
        z2b_cols = range(0, b2_div_col)
    else:
        z2b_cols = range(b2_div_col + 1, w)
    zone2b = {
        Pos(r, c) for r in range(b2 + 1, h) for c in z2b_cols
        if Pos(r, c) not in walls
    }
    db_below = Pos(b2 + 1, db_col)
    if db_below not in zone2b:
        return None
    z2b_comp = _flood_fill(db_below, zone2b)
    if not z2b_comp:
        return None
    # Bias exit away from bypass/trap side
    exit_candidates = sorted(z2b_comp, key=lambda p: abs(p.col - trap_col))
    n_pick = max(1, len(exit_candidates) // 2)
    exit_pos = rng.choice(exit_candidates[:n_pick])

    # Recheck manhattan with real exit
    if not _trap_closer_to_exit(trap_pos, correct_pos, exit_pos):
        return None

    # --- 10. Start ---
    far_side = w - 1 - exit_side
    start_candidates = [p for p in zone0 if p != kr_pos]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - far_side))
    n_pick = max(1, len(start_candidates) // 2)
    start = rng.choice(start_candidates[:n_pick])

    # --- 11. Connectivity checks ---
    trap_approach = Pos(b1 - 1, trap_col)
    correct_approach = Pos(b1 - 1, correct_col)
    if trap_approach not in zone0 or correct_approach not in zone0:
        return None
    zone0_comp = _flood_fill(start, zone0)
    if kr_pos not in zone0_comp or trap_approach not in zone0_comp:
        return None
    if correct_approach not in zone0_comp:
        return None

    # Zone 1C → K_y → D_y approach → D_b approach
    dy_approach = Pos(b2 - 1, dy_col)
    db_approach = Pos(b2 - 1, db_col)
    if dy_approach not in zone1c_comp or db_approach not in zone1c_comp:
        return None
    if ky_pos not in zone1c_comp:
        return None

    # Zone 1T: disconnected from Zone 1C, non-empty
    trap_entry = Pos(b1 + 1, trap_col)
    if trap_entry not in zone1:
        return None
    zone1t_comp = _flood_fill(trap_entry, zone1)
    if correct_entry in zone1t_comp:
        return None
    if not zone1t_comp:
        return None

    # --- 12. Extra walls ---
    zone0_protected = {start, kr_pos, trap_approach, correct_approach}
    zone0_extras = [p for p in zone0 if p not in zone0_protected]
    rng.shuffle(zone0_extras)
    n_extra = rng.randint(0, min(2, len(zone0_extras) // 4))
    for pos in zone0_extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone0_protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    zone1c = set(zone1c_comp)
    z1c_protected = {correct_entry, ky_pos, dy_approach, db_approach}
    z1c_extras = [p for p in zone1c if p not in z1c_protected]
    rng.shuffle(z1c_extras)
    n_extra = rng.randint(0, min(2, len(z1c_extras) // 4))
    for pos in z1c_extras:
        if n_extra <= 0:
            break
        rem = zone1c - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in z1c_protected):
            walls.add(pos)
            zone1c.discard(pos)
            n_extra -= 1

    # --- 13. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=kr_pos, color="red"),
        "K_y": KeyInfo(id="K_y", pos=ky_pos, color="yellow"),
        "K_b": KeyInfo(id="K_b", pos=kb_pos, color="blue"),
    }
    doors = {
        "D_correct_r": DoorInfo(
            id="D_correct_r", pos=correct_pos, required_keys=("red",)),
        "D_trap_r": DoorInfo(
            id="D_trap_r", pos=trap_pos, required_keys=("red",)),
        "D_y": DoorInfo(
            id="D_y", pos=Pos(b2, dy_col), required_keys=("yellow",)),
        "D_b": DoorInfo(
            id="D_b", pos=Pos(b2, db_col), required_keys=("blue",)),
    }
    grid = ChainVaultGrid(
        width=w, height=h,
        walls=frozenset(walls), keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def _generate_ct_2way(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Two-way cascade: two independent spatial traps (w>=7, h>=7).

    LAYOUT:
        Zone 0 (rows 0..b1-1):    Start + K_r + K_b + K_g
        Barrier 1 (row b1):        D_correct_r + D_trap_r + D_trap_g
        Zone 1 (rows b1+1..b2-1): Split by 2 dividers into 3 sections:
          Zone 1C: correct_r entry -> D_b approach
          Zone 1Tr: trap_r side (isolated)
          Zone 1Tg: trap_g side (isolated)
        Barrier 2 (row b2):        D_b (blue)
        Zone 2 (rows b2+1..b3-1): Open passage
        Barrier 3 (row b3):        D_g (green)
        Zone 3 (rows b3+1..h-1):  Exit

    Both D_b and D_g are mandatory (sequential barriers).
    DEADLOCKS:
    - Waste R on D_trap_r -> can't open D_correct_r -> stuck
    - Waste G on D_trap_g -> can't open D_g -> stuck
    """
    h, w = height, width
    if w < 7 or h < 7:
        return None

    # --- 1. Three barrier rows ---
    b1_max = h - 6
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min, b2_max = b1 + 2, h - 4
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)
    b3_min, b3_max = b2 + 2, h - 2
    if b3_min > b3_max:
        return None
    b3 = rng.randint(b3_min, b3_max)

    # --- 2. Exit side ---
    exit_side = rng.choice([0, w - 1])
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))
    if len(exit_half) < 2 or not far_half:
        return None

    # --- 3. Barrier 1: D_correct_r(far) + D_trap_r(exit) + D_trap_g(exit) ---
    rng.shuffle(exit_half)
    trap_r_col, trap_g_col = exit_half[0], exit_half[1]
    far_candidates = [c for c in far_half if c not in {trap_r_col, trap_g_col}]
    if not far_candidates:
        return None
    correct_r_col = rng.choice(far_candidates)

    # Need 2 dividers: one between trap_r and correct_r, one between trap_g and correct_r
    # Place correct_r in middle, traps on either side
    # Actually, simpler: correct_r on far side. Divider between traps and correct.
    # But we need TWO trap zones. Use a single divider between traps and correct,
    # then a wall between the two trap areas.

    # Zone 1 needs to split into 3 parts. Use divider between correct and closest trap,
    # and a second divider between the two traps.
    cols_sorted = sorted([trap_r_col, trap_g_col, correct_r_col])
    # Need dividers between adjacent pairs
    if cols_sorted[1] - cols_sorted[0] < 2 or cols_sorted[2] - cols_sorted[1] < 2:
        return None
    div1 = rng.randint(cols_sorted[0] + 1, cols_sorted[1] - 1)
    div2 = rng.randint(cols_sorted[1] + 1, cols_sorted[2] - 1)

    # Determine which zone is which
    # correct_r_col section is Zone 1C
    # trap_r_col section is Zone 1Tr
    # trap_g_col section is Zone 1Tg

    # --- 4. Barrier 2 & 3 columns (sequential mandatory doors) ---
    if correct_r_col < div1:
        correct_zone_cols = list(range(0, div1))
    elif correct_r_col > div2:
        correct_zone_cols = list(range(div2 + 1, w))
    else:
        correct_zone_cols = list(range(div1 + 1, div2))
    if not correct_zone_cols:
        return None
    db_col = rng.choice(correct_zone_cols)
    dg_col = rng.randint(0, w - 1)

    # --- 5. Build walls ---
    walls: set[Pos] = set()
    # Barrier 1
    for c in range(w):
        if c not in (correct_r_col, trap_r_col, trap_g_col):
            walls.add(Pos(b1, c))
    # Dividers
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, div1))
        walls.add(Pos(r, div2))
    # Barrier 2 (D_b only)
    for c in range(w):
        if c != db_col:
            walls.add(Pos(b2, c))
    # Barrier 3 (D_g only)
    for c in range(w):
        if c != dg_col:
            walls.add(Pos(b3, c))

    # --- 6. Exit ---
    zone3 = {
        Pos(r, c) for r in range(b3 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    if not zone3:
        return None
    exit_candidates = sorted(zone3, key=lambda p: abs(p.col - exit_side))
    n_pick = max(1, len(exit_candidates) // 2)
    exit_pos = rng.choice(exit_candidates[:n_pick])

    # --- 7. Anti-greedy ---
    trap_r_pos = Pos(b1, trap_r_col)
    trap_g_pos = Pos(b1, trap_g_col)
    correct_r_pos = Pos(b1, correct_r_col)
    if not _trap_closer_to_exit(trap_r_pos, correct_r_pos, exit_pos):
        return None

    # --- 8. Keys ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 4:
        return None

    kr_pos = _convergent_key_pos(Pos(b1, trap_r_col), zone0, rng)
    if kr_pos is None:
        return None
    kg_pos = _convergent_key_pos(Pos(b1, trap_g_col), zone0 - {kr_pos}, rng)
    if kg_pos is None:
        return None
    kb_cands = [p for p in zone0 if p not in {kr_pos, kg_pos}]
    if not kb_cands:
        return None
    kb_result = _select_key_positions(
        kb_cands, 1, rng,
        already_placed=frozenset({kr_pos, kg_pos}), reference=exit_pos,
    )
    if kb_result is None:
        return None
    kb_pos = kb_result[0]

    # --- 9. Start ---
    key_positions = {kr_pos, kg_pos, kb_pos}
    start_candidates = [p for p in zone0 if p not in key_positions]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - exit_side))
    n_pick = max(1, len(start_candidates) // 2)
    start = rng.choice(start_candidates[:n_pick])

    # --- 10. Connectivity ---
    trap_r_approach = Pos(b1 - 1, trap_r_col)
    trap_g_approach = Pos(b1 - 1, trap_g_col)
    correct_r_approach = Pos(b1 - 1, correct_r_col)
    z0_critical = {kr_pos, kg_pos, kb_pos,
                   trap_r_approach, trap_g_approach, correct_r_approach}
    for p in z0_critical:
        if p not in zone0:
            return None
    z0_comp = _flood_fill(start, zone0)
    for p in z0_critical:
        if p not in z0_comp:
            return None

    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }
    correct_r_entry = Pos(b1 + 1, correct_r_col)
    db_approach = Pos(b2 - 1, db_col)
    for p in (correct_r_entry, db_approach):
        if p not in zone1:
            return None
    z1c_comp = _flood_fill(correct_r_entry, zone1)
    if db_approach not in z1c_comp:
        return None

    # Trap zones must be disconnected from correct zone
    trap_r_entry = Pos(b1 + 1, trap_r_col)
    trap_g_entry = Pos(b1 + 1, trap_g_col)
    if trap_r_entry not in zone1 or trap_g_entry not in zone1:
        return None
    z1tr = _flood_fill(trap_r_entry, zone1)
    z1tg = _flood_fill(trap_g_entry, zone1)
    if correct_r_entry in z1tr or correct_r_entry in z1tg:
        return None  # trap zones must be isolated

    # Zone 2: db_entry -> dg_approach
    zone2 = {
        Pos(r, c) for r in range(b2 + 1, b3) for c in range(w)
        if Pos(r, c) not in walls
    }
    db_entry = Pos(b2 + 1, db_col)
    dg_approach = Pos(b3 - 1, dg_col)
    if db_entry not in zone2 or dg_approach not in zone2:
        return None
    z2_comp = _flood_fill(db_entry, zone2)
    if dg_approach not in z2_comp:
        return None

    # Zone 3: dg_entry -> exit
    dg_entry = Pos(b3 + 1, dg_col)
    if dg_entry not in zone3:
        return None
    z3_comp = _flood_fill(dg_entry, zone3)
    if exit_pos not in z3_comp:
        return None

    # --- 11. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=kr_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=kb_pos, color="blue"),
        "K_g": KeyInfo(id="K_g", pos=kg_pos, color="green"),
    }
    doors = {
        "D_correct_r": DoorInfo(id="D_correct_r", pos=correct_r_pos,
                                required_keys=("red",)),
        "D_trap_r": DoorInfo(id="D_trap_r", pos=trap_r_pos,
                             required_keys=("red",)),
        "D_trap_g": DoorInfo(id="D_trap_g", pos=trap_g_pos,
                             required_keys=("green",)),
        "D_b": DoorInfo(id="D_b", pos=Pos(b2, db_col),
                        required_keys=("blue",)),
        "D_g": DoorInfo(id="D_g", pos=Pos(b3, dg_col),
                        required_keys=("green",)),
    }
    grid = ChainVaultGrid(
        width=w, height=h,
        walls=frozenset(walls),
        keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start, inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def generate_cascade_trap_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a cascade trap grid: cross-color spatial cascade.

    MECHANIC: 1 red key + 1 blue key, 2 red doors (1 correct, 1 trap) +
    1 blue door.  The agent wastes the red key on the trap door, enters a
    real zone (Zone 1T), but that zone is disconnected from the blue door.
    Even though the agent still HAS the blue key, they can't physically
    REACH the blue door.

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + K_r + K_b
                                     K_r directly above D_trap_r
        Barrier 1 (row b1):         D_correct_r (far-side) + D_trap_r (exit-side) + walls
        Zone 1 (rows b1+1..b2-1):  Split by divider column:
          Zone 1C (correct side):   Connects D_correct_r entry -> D_b approach
          Zone 1T (trap side):      Real area, disconnected from D_b
        Barrier 2 (row b2):         D_b (blue, on correct side of divider) + walls
        Zone 2 (rows b2+1..h-1):   Exit

    ANTI-GREEDY (cross-color cascade):
    - D_trap_r on exit-side half (closer to exit)
    - D_correct_r on far-side half (farther from exit)
    - K_r directly above D_trap_r -> convergent forks bait
    - manhattan(D_trap_r, exit) < manhattan(D_correct_r, exit) -- hard filter
    - Zone 1T >= Zone 1C in size (trap area looks like main path)

    CORRECT PATH: Start -> K_r + K_b -> D_correct_r (red) -> Zone 1C ->
    D_b (blue) -> Zone 2 -> Exit.
    Uses: 1 red + 1 blue = 2 keys.

    DEADLOCKS (>=2):
    1. Agent opens D_trap_r with {1 red, 0 blue} -> Zone 1T -> back to
       Zone 0 -> no red left -> can't open D_correct_r -> can't reach D_b
    2. Agent opens D_trap_r with {1 red, 1 blue} -> same spatial isolation
       -> has blue key but can't reach blue door -> stuck
    Cross-color cascade: red waste -> blue spatially unreachable.
    """
    if rng is None:
        rng = random.Random()
    if width < 5 or height < 5:
        return None

    # P2.13: Two-way cascade at 7×7+
    if width >= 7 and height >= 7 and rng.random() < 0.5:
        result = _generate_ct_2way(width, height, rng)
        if result is not None:
            return result

    # --- Stochastic depth-2 variant (P1.9) ---
    if width >= 6 and height >= 6 and rng.random() < 0.5:
        result = _generate_ct_depth2(width, height, rng)
        if result is not None:
            return result

    h, w = height, width

    # --- 1. Barrier rows ---
    b1_max = h - 4
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min, b2_max = b1 + 2, h - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Exit-side direction ---
    exit_side = rng.choice([0, w - 1])

    # --- 3. Column halves ---
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))

    if not exit_half or not far_half:
        return None

    # --- 4. Barrier 1 columns: trap on exit-side, correct on far-side ---
    trap_col = rng.choice(exit_half)
    far_half_b1 = [c for c in far_half if c != trap_col]
    if not far_half_b1:
        return None
    correct_col = rng.choice(far_half_b1)

    # --- 5. Divider column (biased toward correct_col to maximize Zone 1T) ---
    lo, hi = min(trap_col, correct_col), max(trap_col, correct_col)
    divider_options = list(range(lo + 1, hi))
    if not divider_options:
        return None
    if correct_col > trap_col:
        divider_col = divider_options[-1]
    else:
        divider_col = divider_options[0]

    # --- 6. Blue door column (on correct side of divider) ---
    if correct_col > divider_col:
        door_b_candidates = list(range(divider_col + 1, w))
    else:
        door_b_candidates = list(range(0, divider_col))
    if not door_b_candidates:
        return None
    door_b_col = rng.choice(door_b_candidates)

    # --- 7. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: walls except trap_col and correct_col
    for c in range(w):
        if c != trap_col and c != correct_col:
            walls.add(Pos(b1, c))

    # Divider walls through all Zone 1 rows
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, divider_col))

    # Barrier 2: walls except door_b_col
    for c in range(w):
        if c != door_b_col:
            walls.add(Pos(b2, c))

    # --- Scaling: extra trap(s) at w>=6 / w>=7 ---
    extra_trap_ct_b1: Optional[int] = None
    extra_trap_ct_b2: Optional[int] = None
    if w >= 6:
        extra_trap_ct_b1 = _pick_extra_trap_col(
            b1, walls, {trap_col, correct_col}, exit_half, rng)
        if extra_trap_ct_b1 is not None:
            walls.discard(Pos(b1, extra_trap_ct_b1))
    if w >= 7:
        extra_trap_ct_b2 = _pick_extra_trap_col(
            b2, walls, {door_b_col}, exit_half, rng)
        if extra_trap_ct_b2 is not None:
            walls.discard(Pos(b2, extra_trap_ct_b2))

    # --- 8. Exit placement (biased toward trap_col to strengthen manhattan bait) ---
    zone2 = {
        Pos(r, c) for r in range(b2 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    if not zone2:
        return None
    exit_candidates = sorted(zone2, key=lambda p: abs(p.col - trap_col))
    n_pick = max(1, len(exit_candidates) // 2)
    exit_pos = rng.choice(exit_candidates[:n_pick])

    # --- 9. Anti-greedy manhattan check ---
    trap_pos = Pos(b1, trap_col)
    correct_pos = Pos(b1, correct_col)
    if not _trap_closer_to_exit(trap_pos, correct_pos, exit_pos):
        return None

    # --- 10. Key placement ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 3:
        return None

    # K_r: near trap door (convergent forks bait)
    kr_pos = _convergent_key_pos(Pos(b1, trap_col), zone0, rng)
    if kr_pos is None:
        return None

    # K_b: biased toward trap_col (double convergent forks —
    # both keys cluster near trap, so collecting them converges on trap door)
    kb_candidates = [p for p in zone0 if p != kr_pos]
    if not kb_candidates:
        return None
    kb_candidates.sort(key=lambda p: abs(p.col - trap_col))
    biased_pool = kb_candidates[:max(1, len(kb_candidates) // 3)]
    kb_result = _select_key_positions(
        biased_pool, 1, rng,
        already_placed=frozenset({kr_pos}), reference=exit_pos,
    )
    if kb_result is None:
        kb_result = _select_key_positions(
            kb_candidates, 1, rng,
            already_placed=frozenset({kr_pos}), reference=exit_pos,
        )
    if kb_result is None:
        return None
    kb_pos = kb_result[0]

    # --- 11. Start: biased toward far side (opposite exit) ---
    # Agent must traverse Zone 0 toward exit, collecting both keys en route,
    # arriving at the trap door — the whole journey converges on the trap.
    far_side = w - 1 - exit_side
    start_candidates = [p for p in zone0 if p != kr_pos and p != kb_pos]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - far_side))
    n_pick = max(1, len(start_candidates) // 2)
    start = rng.choice(start_candidates[:n_pick])

    # --- 12. Connectivity checks ---
    trap_approach = Pos(b1 - 1, trap_col)
    correct_approach = Pos(b1 - 1, correct_col)
    if trap_approach not in zone0 or correct_approach not in zone0:
        return None
    zone0_comp = _flood_fill(start, zone0)
    if kr_pos not in zone0_comp or kb_pos not in zone0_comp:
        return None
    if trap_approach not in zone0_comp or correct_approach not in zone0_comp:
        return None

    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }

    # Zone 1C: correct_entry -> door_b_approach connected
    correct_entry = Pos(b1 + 1, correct_col)
    door_b_approach = Pos(b2 - 1, door_b_col)
    if correct_entry not in zone1 or door_b_approach not in zone1:
        return None
    zone1c_comp = _flood_fill(correct_entry, zone1)
    if door_b_approach not in zone1c_comp:
        return None

    # Zone 1T: disconnected from Zone 1C, >= Zone 1C size
    trap_entry = Pos(b1 + 1, trap_col)
    if trap_entry not in zone1:
        return None
    zone1t_comp = _flood_fill(trap_entry, zone1)
    if correct_entry in zone1t_comp:
        return None  # divider failed
    if len(zone1t_comp) < len(zone1c_comp):
        return None  # trap area should look like main path

    # Zone 2: door_b_entry -> exit connected
    door_b_entry = Pos(b2 + 1, door_b_col)
    if door_b_entry not in zone2:
        return None
    zone2_comp = _flood_fill(door_b_entry, zone2)
    if exit_pos not in zone2_comp:
        return None

    # --- 13. Extra walls (0-2 per zone, connectivity-preserving) ---
    # Zone 0 extras
    zone0_protected = {start, kr_pos, kb_pos, trap_approach, correct_approach}
    zone0_extras = [p for p in zone0 if p not in zone0_protected]
    rng.shuffle(zone0_extras)
    n_extra = rng.randint(0, min(2, len(zone0_extras) // 4))
    for pos in zone0_extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone0_protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    # Zone 1C extras (Zone 1T left untouched)
    zone1c = set(zone1c_comp)
    zone1c_protected = {correct_entry, door_b_approach}
    zone1c_extras = [p for p in zone1c if p not in zone1c_protected]
    rng.shuffle(zone1c_extras)
    n_extra = rng.randint(0, min(2, len(zone1c_extras) // 4))
    for pos in zone1c_extras:
        if n_extra <= 0:
            break
        rem = zone1c - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone1c_protected):
            walls.add(pos)
            zone1c.discard(pos)
            n_extra -= 1

    # --- 14. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=kr_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=kb_pos, color="blue"),
    }
    doors = {
        "D_correct_r": DoorInfo(
            id="D_correct_r", pos=correct_pos, required_keys=("red",)),
        "D_trap_r": DoorInfo(
            id="D_trap_r", pos=trap_pos, required_keys=("red",)),
        "D_b": DoorInfo(
            id="D_b", pos=Pos(b2, door_b_col), required_keys=("blue",)),
    }
    if extra_trap_ct_b1 is not None:
        doors["D_trap_r2"] = DoorInfo(
            id="D_trap_r2", pos=Pos(b1, extra_trap_ct_b1), required_keys=("red",))
    if extra_trap_ct_b2 is not None:
        doors["D_trap_b2"] = DoorInfo(
            id="D_trap_b2", pos=Pos(b2, extra_trap_ct_b2), required_keys=("blue",))
    grid = ChainVaultGrid(
        width=width, height=height,
        walls=frozenset(walls), keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Template 11: Sequential Dual-Locks — Same-Color Fork helper (7×7+)
# ---------------------------------------------------------------------------

def _generate_sdl_same_color_fork(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Same-color fork variant of SDL (requires width >= 7, height >= 7).

    MECHANIC: 4 colors (R, B, Y, G). After correctly passing D_dual1 (R+B),
    the agent enters Zone 1 carrying K_y. Two same-color yellow doors
    (D_trap_y and D_correct_y) create a fork — the agent cannot distinguish
    them by color and must reason about what lies behind each.

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + K_r + K_b (near D_trap_b) + K_y
        Barrier 1 (row b1):         D_dual1 (R+B, far-side) + D_trap_b (B, exit-side)
        Zone 1 (rows b1+1..b2-1):  Split by outer divider:
          Zone 1T_b (exit-side):    Dead end (behind D_trap_b)
          Zone 1_main (far-side):   Behind D_dual1, contains inner Y-fork:
            Approach area (row b1+1)
            Fork barrier (row r_fork): D_trap_y (Y) + D_correct_y (Y)
            Below fork, split by inner divider:
              Zone 1T_y (exit-sub):  Dead end (behind D_trap_y)
              Zone 1C (far-sub):     K_y2 + K_g (behind D_correct_y)
        Barrier 2 (row b2):         D_dual2 (G+Y, on Zone 1C side)
        Zone 2 (rows b2+1..h-1):   Exit

    DEADLOCKS (>=2):
    1. Waste B on D_trap_b → can't open D_dual1 → stuck
    2. Open D_dual1 → waste Y on D_trap_y → no Y₂/G for D_dual2 → stuck

    CORRECT PATH: Start → K_r + K_b + K_y → D_dual1 (R+B) → Zone 1_main →
    D_correct_y (Y) → Zone 1C → K_y2 + K_g → D_dual2 (G+Y₂) → exit.
    """
    if width < 7 or height < 7:
        return None

    h, w = height, width

    # --- 1. Barrier rows: need b2 >= b1 + 4 for 3 Zone 1 rows ---
    b1_max = h - 6  # b1 + 4 <= h - 2 => b1 <= h - 6
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min = b1 + 4
    b2_max = h - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Exit-side direction ---
    exit_side = rng.choice([0, w - 1])

    # --- 3. Column halves ---
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))
    if not exit_half or not far_half:
        return None

    # --- 4. Barrier 1 columns ---
    trap_col = rng.choice(exit_half)
    far_half_b1 = [c for c in far_half if c != trap_col]
    if not far_half_b1:
        return None
    dual1_col = rng.choice(far_half_b1)

    # --- 5. Outer divider: biased TOWARD trap_col to maximize Zone 1_main ---
    lo, hi = min(trap_col, dual1_col), max(trap_col, dual1_col)
    divider_options = list(range(lo + 1, hi))
    if not divider_options:
        return None
    # Pick divider close to trap_col (opposite of standard SDL which biases
    # toward dual1_col to maximize Zone 1T — here we need Zone 1_main large)
    if trap_col < dual1_col:
        d_outer = divider_options[0]  # leftmost = closest to trap
    else:
        d_outer = divider_options[-1]  # rightmost = closest to trap

    # --- 6. Validate Zone 1_main width (need >= 4 cols) ---
    if dual1_col > d_outer:
        z1m_cols = list(range(d_outer + 1, w))
    else:
        z1m_cols = list(range(0, d_outer))
    if len(z1m_cols) < 4:
        return None

    # --- 7. Fork barrier row ---
    r_fork = b1 + 2  # row b1+1 = approach, b1+2 = fork barrier
    if r_fork >= b2:
        return None
    # Rows below fork: r_fork+1..b2-1 (need >= 1)
    if r_fork + 1 >= b2:
        return None

    # --- 8. Inner fork columns (within Zone 1_main) ---
    z1m_cols_sorted = sorted(z1m_cols)
    z1m_exit_sub_col = z1m_cols_sorted[0]   # closest to exit side
    z1m_far_sub_col = z1m_cols_sorted[-1]    # closest to far side

    # D_trap_y on exit-sub side, D_correct_y on far-sub side
    # Pick specific cols: trap closer to exit, correct closer to far
    trap_y_col = z1m_exit_sub_col
    correct_y_col = z1m_far_sub_col
    if abs(trap_y_col - correct_y_col) < 2:
        return None  # too close, can't fit inner divider between them

    # Inner divider between the two Y doors
    inner_lo, inner_hi = min(trap_y_col, correct_y_col), max(trap_y_col, correct_y_col)
    inner_div_options = list(range(inner_lo + 1, inner_hi))
    if not inner_div_options:
        return None
    # Bias toward TRAP side to maximize Zone 1C (needs >= 2 cells for K_y2 + K_g)
    if correct_y_col > trap_y_col:
        d_inner = inner_div_options[0]  # close to trap = large Zone 1C
    else:
        d_inner = inner_div_options[-1]  # close to trap = large Zone 1C
    if correct_y_col > d_inner:
        z1c_cols = [c for c in z1m_cols if c > d_inner]
    else:
        z1c_cols = [c for c in z1m_cols if c < d_inner]
    below_fork_rows = list(range(r_fork + 1, b2))
    z1c_cells_count = len(z1c_cols) * len(below_fork_rows)
    if z1c_cells_count < 2:
        return None

    # Zone 1T_y cols
    if trap_y_col < d_inner:
        z1ty_cols = [c for c in z1m_cols if c < d_inner]
    else:
        z1ty_cols = [c for c in z1m_cols if c > d_inner]
    z1ty_cells_count = len(z1ty_cols) * len(below_fork_rows)
    if z1ty_cells_count < 1:
        return None

    # --- 9. D_dual2 column (on Zone 1C side of barrier 2) ---
    if not z1c_cols:
        return None
    dual2_col = rng.choice(z1c_cols)

    # --- 10. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: walls except trap_col and dual1_col
    for c in range(w):
        if c != trap_col and c != dual1_col:
            walls.add(Pos(b1, c))

    # Outer divider: all Zone 1 rows
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, d_outer))

    # Fork barrier: walls across Zone 1_main at r_fork, except Y doors
    for c in z1m_cols:
        if c != trap_y_col and c != correct_y_col:
            walls.add(Pos(r_fork, c))

    # Inner divider: below fork rows, within Zone 1_main
    for r in below_fork_rows:
        walls.add(Pos(r, d_inner))

    # Barrier 2: walls except dual2_col
    for c in range(w):
        if c != dual2_col:
            walls.add(Pos(b2, c))

    # --- 11. Exit placement ---
    zone2 = {
        Pos(r, c) for r in range(b2 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    if not zone2:
        return None
    # Bias toward trap_col (exit-side) for manhattan bait
    exit_candidates = sorted(zone2, key=lambda p: abs(p.col - trap_col))
    n_pick = max(1, len(exit_candidates) // 3)
    exit_pos = rng.choice(exit_candidates[:n_pick])

    # --- 12. Anti-greedy manhattan check ---
    trap_pos_b1 = Pos(b1, trap_col)
    dual1_pos = Pos(b1, dual1_col)
    if not _trap_closer_to_exit(trap_pos_b1, dual1_pos, exit_pos):
        return None

    # Also: D_trap_y should be closer to exit than D_correct_y
    trap_y_pos = Pos(r_fork, trap_y_col)
    correct_y_pos = Pos(r_fork, correct_y_col)
    if not _trap_closer_to_exit(trap_y_pos, correct_y_pos, exit_pos):
        return None

    # --- 13. Zone 0 key placement ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 4:
        return None

    # K_b: near D_trap_b (convergent forks bait)
    kb_pos = _convergent_key_pos(Pos(b1, trap_col), zone0, rng)
    if kb_pos is None:
        return None

    # K_r, K_y: biased toward trap_col and barrier row
    kry_candidates = [p for p in zone0 if p != kb_pos]
    if len(kry_candidates) < 2:
        return None
    kry_candidates.sort(key=lambda p: abs(p.col - trap_col) + (b1 - 1 - p.row))
    biased_pool = kry_candidates[:max(2, len(kry_candidates) // 3)]
    kry_result = _select_key_positions(
        biased_pool, 2, rng,
        already_placed=frozenset({kb_pos}), reference=exit_pos,
    )
    if kry_result is None:
        kry_result = _select_key_positions(
            kry_candidates, 2, rng,
            already_placed=frozenset({kb_pos}), reference=exit_pos,
        )
    if kry_result is None:
        return None
    kr_pos, ky_pos = kry_result

    # --- 14. Start: biased toward far corner ---
    far_side = w - 1 - exit_side
    start_candidates = [
        p for p in zone0 if p != kb_pos and p != kr_pos and p != ky_pos
    ]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - far_side) - p.row)
    n_pick = max(1, len(start_candidates) // 3)
    start = rng.choice(start_candidates[:n_pick])

    # --- 15. Zone 1C key placement (K_y2, K_g) ---
    z1c_set = {
        Pos(r, c) for r in below_fork_rows for c in z1c_cols
        if Pos(r, c) not in walls
    }
    correct_y_entry = Pos(r_fork + 1, correct_y_col)
    dual2_approach = Pos(b2 - 1, dual2_col)
    # If entry or approach aren't in z1c_set, adjust
    if correct_y_entry not in z1c_set or dual2_approach not in z1c_set:
        return None
    z1c_comp = _flood_fill(correct_y_entry, z1c_set)
    if dual2_approach not in z1c_comp:
        return None
    key_spots = [p for p in z1c_comp if p != correct_y_entry and p != dual2_approach]
    if len(key_spots) < 2:
        # If only 1 spot, place both keys on same cell? No, we need 2 distinct.
        # Place one on entry or approach if needed.
        key_spots = [p for p in z1c_comp if p != correct_y_entry]
        if len(key_spots) < 2:
            key_spots = list(z1c_comp)
            if len(key_spots) < 2:
                return None
    rng.shuffle(key_spots)
    ky2_pos = key_spots[0]
    kg_pos = key_spots[1]

    # --- 16. Connectivity checks ---
    # Zone 0: start -> K_b -> K_r -> K_y -> trap approach -> dual1 approach
    trap_approach = Pos(b1 - 1, trap_col)
    correct_approach = Pos(b1 - 1, dual1_col)
    if trap_approach not in zone0 or correct_approach not in zone0:
        return None
    zone0_comp = _flood_fill(start, zone0)
    for p in [kb_pos, kr_pos, ky_pos, trap_approach, correct_approach]:
        if p not in zone0_comp:
            return None

    # Zone 1 flood fill: exclude Y door positions (locked doors block movement)
    zone1_all = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
        and Pos(r, c) != trap_y_pos
        and Pos(r, c) != correct_y_pos
    }

    # Zone 1T_b: behind D_trap_b, dead end, disconnected from Zone 1_main
    trap_b_entry = Pos(b1 + 1, trap_col)
    dual1_entry = Pos(b1 + 1, dual1_col)
    if trap_b_entry not in zone1_all or dual1_entry not in zone1_all:
        return None
    zone1tb_comp = _flood_fill(trap_b_entry, zone1_all)
    if dual1_entry in zone1tb_comp:
        return None  # outer divider failed

    # Zone 1_main (approach area): dual1_entry -> fork door approaches reachable
    zone1m_comp = _flood_fill(dual1_entry, zone1_all)
    trap_y_approach = Pos(r_fork - 1, trap_y_col)
    correct_y_approach = Pos(r_fork - 1, correct_y_col)
    if trap_y_approach not in zone1m_comp or correct_y_approach not in zone1m_comp:
        return None

    # Zone 1T_y: behind D_trap_y, dead end, disconnected from Zone 1C
    trap_y_entry = Pos(r_fork + 1, trap_y_col)
    if trap_y_entry not in zone1_all:
        return None
    zone1ty_comp = _flood_fill(trap_y_entry, zone1_all)
    if correct_y_entry in zone1ty_comp:
        return None  # inner divider failed

    # Zone 1C: correct_y_entry -> K_y2, K_g, dual2_approach all reachable
    if ky2_pos not in z1c_comp or kg_pos not in z1c_comp:
        return None

    # Zone 2: dual2_entry -> exit
    dual2_entry = Pos(b2 + 1, dual2_col)
    if dual2_entry not in zone2:
        return None
    zone2_comp = _flood_fill(dual2_entry, zone2)
    if exit_pos not in zone2_comp:
        return None

    # --- 17. Extra walls (0-2 per zone, connectivity-preserving) ---
    # Zone 0 extras
    zone0_protected = {start, kr_pos, kb_pos, ky_pos, trap_approach, correct_approach}
    zone0_extras = [p for p in zone0 if p not in zone0_protected]
    rng.shuffle(zone0_extras)
    n_extra = rng.randint(0, min(2, len(zone0_extras) // 4))
    for pos in zone0_extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone0_protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    # --- 18. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=kr_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=kb_pos, color="blue"),
        "K_y": KeyInfo(id="K_y", pos=ky_pos, color="yellow"),
        "K_y2": KeyInfo(id="K_y2", pos=ky2_pos, color="yellow"),
        "K_g": KeyInfo(id="K_g", pos=kg_pos, color="green"),
    }
    doors = {
        "D_dual1": DoorInfo(
            id="D_dual1", pos=dual1_pos, required_keys=("red", "blue")),
        "D_trap_b": DoorInfo(
            id="D_trap_b", pos=trap_pos_b1, required_keys=("blue",)),
        "D_trap_y": DoorInfo(
            id="D_trap_y", pos=trap_y_pos, required_keys=("yellow",)),
        "D_correct_y": DoorInfo(
            id="D_correct_y", pos=correct_y_pos, required_keys=("yellow",)),
        "D_dual2": DoorInfo(
            id="D_dual2", pos=Pos(b2, dual2_col),
            required_keys=("green", "yellow")),
    }
    grid = ChainVaultGrid(
        width=width, height=height,
        walls=frozenset(walls), keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def generate_sequential_dual_locks_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a sequential dual-locks grid: key replenishment with dual-lock chains.

    MECHANIC: 3 colors (red, blue, green), two dual-lock doors in sequence.
    The first dual-lock (red+blue) gates access to a second blue key (K_b2),
    which is then needed with green for the second dual-lock (blue+green).
    A cheap single-lock blue trap lures greedy agents into wasting their only
    initial blue key.

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + K_r + K_b1 + K_g
                                     K_b1 directly above D_trap_b (convergent forks bait)
                                     All keys biased toward trap_col
        Barrier 1 (row b1):         D_dual1 (red+blue, far-side) + D_trap_b (blue, exit-side) + walls
        Zone 1 (rows b1+1..b2-1):  Split by divider column:
          Zone 1C (correct side):   Behind D_dual1. Contains K_b2. Connects -> D_dual2
          Zone 1T (trap side):      Behind D_trap_b. Disconnected from K_b2 and D_dual2
        Barrier 2 (row b2):         D_dual2 (blue+green, on correct side) + walls
        Zone 2 (rows b2+1..h-1):   Exit

    ANTI-GREEDY (single-lock vs dual-lock cost):
    - D_trap_b is single-lock (1 key cost) vs D_dual1 is dual-lock (2 key cost)
    - D_trap_b on exit-side half (closer to exit)
    - K_b1 directly above D_trap_b -> convergent forks bait
    - All 3 Zone 0 keys biased toward trap_col (triple convergent forks)
    - Start on far side — journey converges on trap
    - Zone 1T >= Zone 1C — trap looks like main path

    CORRECT PATH: Start -> K_r + K_b1 + K_g -> D_dual1 (costs K_r + K_b1) ->
    Zone 1C -> K_b2 -> D_dual2 (costs K_b2 + K_g) -> Zone 2 -> Exit.
    Uses: all 4 keys.

    DEADLOCKS (>=2):
    1. Agent opens D_trap_b with K_b1 (any inventory state) -> Zone 1T ->
       no blue left -> can't open D_dual1 -> can't reach K_b2 -> stuck
    2. Different inventory states when blue is wasted -> distinct deadlock entries

    UNIQUE REASONING: Temporal resource planning with replenishment —
    the solution itself provides the resource (K_b2) for the second use.
    """
    if rng is None:
        rng = random.Random()
    if width < 6 or height < 6:
        return None

    # --- Same-color fork variant (50% at width >= 7 and height >= 7) ---
    if width >= 7 and height >= 7 and rng.random() < 0.5:
        result = _generate_sdl_same_color_fork(width, height, rng)
        if result is not None:
            return result
        # Fall through to standard variant if same-color fork fails

    h, w = height, width

    # --- 1. Barrier rows ---
    b1_max = h - 4
    if b1_max < 2:
        return None
    b1 = rng.randint(2, b1_max)
    b2_min, b2_max = b1 + 2, h - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Exit-side direction ---
    exit_side = rng.choice([0, w - 1])

    # --- 3. Column halves ---
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))

    if not exit_half or not far_half:
        return None

    # --- 4. Barrier 1 columns: trap on exit-side, dual1 on far-side ---
    trap_col = rng.choice(exit_half)
    far_half_b1 = [c for c in far_half if c != trap_col]
    if not far_half_b1:
        return None
    dual1_col = rng.choice(far_half_b1)

    # --- 5. Divider column (biased toward dual1_col to maximize Zone 1T) ---
    lo, hi = min(trap_col, dual1_col), max(trap_col, dual1_col)
    divider_options = list(range(lo + 1, hi))
    if not divider_options:
        return None
    if dual1_col > trap_col:
        divider_col = divider_options[-1]
    else:
        divider_col = divider_options[0]

    # --- 6. Dual2 door column (on correct side of divider) ---
    if dual1_col > divider_col:
        dual2_candidates = list(range(divider_col + 1, w))
    else:
        dual2_candidates = list(range(0, divider_col))
    if not dual2_candidates:
        return None
    dual2_col = rng.choice(dual2_candidates)

    # --- 7. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: walls except trap_col and dual1_col
    for c in range(w):
        if c != trap_col and c != dual1_col:
            walls.add(Pos(b1, c))

    # Divider walls through all Zone 1 rows
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, divider_col))

    # Barrier 2: walls except dual2_col
    for c in range(w):
        if c != dual2_col:
            walls.add(Pos(b2, c))

    # --- Scaling: extra trap(s) at w>=7 / w>=8 ---
    extra_trap_sdl_b2: Optional[int] = None
    extra_trap_sdl_b1: Optional[int] = None
    enhanced_sdl = w >= 7 and rng.random() < 0.5
    if w >= 7:
        extra_trap_sdl_b2 = _pick_extra_trap_col(
            b2, walls, {dual2_col}, exit_half, rng)
        if extra_trap_sdl_b2 is not None:
            walls.discard(Pos(b2, extra_trap_sdl_b2))
    if w >= 8 or enhanced_sdl:
        extra_trap_sdl_b1 = _pick_extra_trap_col(
            b1, walls, {trap_col, dual1_col}, exit_half, rng)
        if extra_trap_sdl_b1 is not None:
            walls.discard(Pos(b1, extra_trap_sdl_b1))

    # --- 8. Exit placement (strongly biased toward trap_col for manhattan bait) ---
    zone2 = {
        Pos(r, c) for r in range(b2 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    if not zone2:
        return None
    exit_candidates = sorted(zone2, key=lambda p: abs(p.col - trap_col))
    n_pick = max(1, len(exit_candidates) // 3)
    exit_pos = rng.choice(exit_candidates[:n_pick])

    # --- 9. Anti-greedy manhattan check ---
    trap_pos = Pos(b1, trap_col)
    dual1_pos = Pos(b1, dual1_col)
    if not _trap_closer_to_exit(trap_pos, dual1_pos, exit_pos):
        return None

    # --- 10. Zone 0 key placement ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 4:
        return None

    # K_b1: near trap door (convergent forks bait)
    kb1_pos = _convergent_key_pos(Pos(b1, trap_col), zone0, rng)
    if kb1_pos is None:
        return None

    # K_r, K_g: biased toward trap_col AND barrier row (on the path to trap)
    krg_candidates = [p for p in zone0 if p != kb1_pos]
    if len(krg_candidates) < 2:
        return None
    krg_candidates.sort(key=lambda p: abs(p.col - trap_col) + (b1 - 1 - p.row))
    biased_pool = krg_candidates[:max(2, len(krg_candidates) // 3)]
    krg_result = _select_key_positions(
        biased_pool, 2, rng,
        already_placed=frozenset({kb1_pos}), reference=exit_pos,
    )
    if krg_result is None:
        krg_result = _select_key_positions(
            krg_candidates, 2, rng,
            already_placed=frozenset({kb1_pos}), reference=exit_pos,
        )
    if krg_result is None:
        return None
    kr_pos, kg_pos = krg_result

    # --- 11. Start: biased toward top-far corner (maximizes diagonal to trap) ---
    far_side = w - 1 - exit_side
    start_candidates = [
        p for p in zone0 if p != kb1_pos and p != kr_pos and p != kg_pos
    ]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - far_side) - p.row)
    n_pick = max(1, len(start_candidates) // 3)
    start = rng.choice(start_candidates[:n_pick])

    # --- 12. K_b2 placement in Zone 1C (replenishment key) ---
    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }

    correct_entry = Pos(b1 + 1, dual1_col)
    dual2_approach = Pos(b2 - 1, dual2_col)
    if correct_entry not in zone1 or dual2_approach not in zone1:
        return None

    zone1c_comp = _flood_fill(correct_entry, zone1)
    if dual2_approach not in zone1c_comp:
        return None

    # Place K_b2 with convergent-forks bait: prefer position adjacent to
    # outer divider so we can add a blue trap door on the divider.
    # The trap opens into Zone 1T_b (already a dead-end zone).
    kb2_candidates = [
        p for p in zone1c_comp if p != correct_entry and p != dual2_approach
    ]
    if not kb2_candidates:
        return None

    # --- 12a. Zone 1 convergent forks bait (best-effort) ---
    z1_trap_door_pos: Optional[Pos] = None
    if dual1_col > divider_col:
        adj_col = divider_col + 1   # Zone 1C side of divider
        check_col = divider_col - 1  # Zone 1T side of divider
    else:
        adj_col = divider_col - 1
        check_col = divider_col + 1

    bait_candidates = [
        p for p in kb2_candidates if p.col == adj_col
    ]
    rng.shuffle(bait_candidates)
    for bait_pos in bait_candidates:
        div_wall = Pos(bait_pos.row, divider_col)
        behind = Pos(bait_pos.row, check_col)
        if div_wall in walls and behind in zone1:
            # Valid bait: K_b2 adjacent to divider, trap opens into Zone 1T
            z1_trap_door_pos = div_wall
            kb2_pos = bait_pos
            walls.discard(z1_trap_door_pos)
            break
    else:
        # No valid bait position — fall back to random K_b2 placement
        kb2_pos = rng.choice(kb2_candidates)

    # --- 13. Connectivity checks ---
    trap_approach = Pos(b1 - 1, trap_col)
    correct_approach = Pos(b1 - 1, dual1_col)
    if trap_approach not in zone0 or correct_approach not in zone0:
        return None
    zone0_comp = _flood_fill(start, zone0)
    if kb1_pos not in zone0_comp or kr_pos not in zone0_comp:
        return None
    if kg_pos not in zone0_comp:
        return None
    if trap_approach not in zone0_comp or correct_approach not in zone0_comp:
        return None

    # Recompute zone1 excluding the Z1 trap door (locked door blocks movement)
    z1_excluded = {z1_trap_door_pos} if z1_trap_door_pos is not None else set()
    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls and Pos(r, c) not in z1_excluded
    }

    # Zone 1C: entry -> K_b2 -> approach connected
    zone1c_comp = _flood_fill(correct_entry, zone1)
    if kb2_pos not in zone1c_comp or dual2_approach not in zone1c_comp:
        return None

    # Zone 1T: disconnected from Zone 1C, >= Zone 1C size
    trap_entry = Pos(b1 + 1, trap_col)
    if trap_entry not in zone1:
        return None
    zone1t_comp = _flood_fill(trap_entry, zone1)
    if correct_entry in zone1t_comp:
        return None  # divider failed
    if len(zone1t_comp) < len(zone1c_comp):
        return None  # trap area should look like main path

    # Zone 2: dual2_entry -> exit connected
    dual2_entry = Pos(b2 + 1, dual2_col)
    if dual2_entry not in zone2:
        return None
    zone2_comp = _flood_fill(dual2_entry, zone2)
    if exit_pos not in zone2_comp:
        return None

    # --- 14. Extra walls (0-2 per zone, connectivity-preserving) ---

    # P2.14: Zone 0 corridor walls at w>=7 — forces non-trivial key collection path
    zone0_protected = {
        start, kr_pos, kb1_pos, kg_pos, trap_approach, correct_approach,
    }
    if w >= 7 and b1 >= 3 and rng.random() < 0.5:
        corridor_row = rng.randint(1, b1 - 2)
        corridor_gap = rng.choice(list(range(w)))
        added_walls: list[Pos] = []
        for c in range(w):
            p = Pos(corridor_row, c)
            if c != corridor_gap and p not in zone0_protected and p in zone0:
                walls.add(p)
                zone0.discard(p)
                added_walls.append(p)
        # Verify connectivity
        ok = bool(zone0) and _is_connected(zone0)
        if ok:
            comp = _flood_fill(start, zone0)
            ok = all(p in comp for p in zone0_protected if p in zone0)
        if not ok:
            for p in added_walls:
                walls.discard(p)
                zone0.add(p)

    # Zone 0 extras (standard 0-2 random walls)
    zone0_extras = [p for p in zone0 if p not in zone0_protected]
    rng.shuffle(zone0_extras)
    n_extra = rng.randint(0, min(2, len(zone0_extras) // 4))
    for pos in zone0_extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone0_protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    # Zone 1C extras
    zone1c = set(zone1c_comp)
    zone1c_protected = {correct_entry, dual2_approach, kb2_pos}
    zone1c_extras = [p for p in zone1c if p not in zone1c_protected]
    rng.shuffle(zone1c_extras)
    n_extra = rng.randint(0, min(2, len(zone1c_extras) // 4))
    for pos in zone1c_extras:
        if n_extra <= 0:
            break
        rem = zone1c - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone1c_protected):
            walls.add(pos)
            zone1c.discard(pos)
            n_extra -= 1

    # --- 15. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=kr_pos, color="red"),
        "K_b1": KeyInfo(id="K_b1", pos=kb1_pos, color="blue"),
        "K_g": KeyInfo(id="K_g", pos=kg_pos, color="green"),
        "K_b2": KeyInfo(id="K_b2", pos=kb2_pos, color="blue"),
    }
    doors = {
        "D_dual1": DoorInfo(
            id="D_dual1", pos=dual1_pos, required_keys=("red", "blue")),
        "D_trap_b": DoorInfo(
            id="D_trap_b", pos=trap_pos, required_keys=("blue",)),
        "D_dual2": DoorInfo(
            id="D_dual2", pos=Pos(b2, dual2_col),
            required_keys=("blue", "green")),
    }
    if extra_trap_sdl_b2 is not None:
        doors["D_trap_b2"] = DoorInfo(
            id="D_trap_b2", pos=Pos(b2, extra_trap_sdl_b2), required_keys=("blue",))
    if extra_trap_sdl_b1 is not None:
        doors["D_trap_g"] = DoorInfo(
            id="D_trap_g", pos=Pos(b1, extra_trap_sdl_b1), required_keys=("green",))
    if z1_trap_door_pos is not None:
        doors["D_trap_b_z1"] = DoorInfo(
            id="D_trap_b_z1", pos=z1_trap_door_pos, required_keys=("blue",))
    grid = ChainVaultGrid(
        width=width, height=height,
        walls=frozenset(walls), keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Template 12: Key Collector — Path Commitment helper (7×7+)
# ---------------------------------------------------------------------------

def _generate_kc_path_commitment(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Path Commitment variant of Key Collector (requires width ≥ 7, height ≥ 6).

    MECHANIC: 3 keys (R, B, G) all in Zone 0.  Two single-lock trap gates on
    Barrier 1 — gate_r (RED) and gate_b (BLUE) — plus a free bypass.  The only
    exit is triple-lock D_exit (R+B+G) on Barrier 2.  Using gate_r wastes RED
    (agent arrives at D_exit with only B+G → can't open), and using gate_b
    wastes BLUE (arrives with only R+G → can't open).  Both keys must be
    preserved for the triple-lock.

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + K_r (near gate_r, bait)
                                     + K_b (near gate_b, bait) + K_g (free)
        Barrier 1 (row b1):         gate_r (RED) | gate_b (BLUE) | bypass (free)
        Zone 1 (rows b1+1..b2-1):  Open exploration zone (all passages connect here)
        Barrier 2 (row b2):         D_exit (R+B+G triple-lock, sole passage)
        Zone 2 (rows b2+1..h-1):   Exit

    ANTI-GREEDY (dual convergent forks):
        K_r placed directly above gate_r — strategies 1-6 pick up K_r then
        immediately USE_KEY gate_r (wastes RED → stuck).
        K_b placed directly above gate_b — same for BLUE.
        Two independent bait traps each lead to a walking-dead state.

    DEADLOCKS: ≥3
        (1) use gate_r alone  → only B+G, can't open D_exit
        (2) use gate_b alone  → only R+G, can't open D_exit
        (3) use gate_r + gate_b → only G, can't open D_exit
    """
    if width < 7 or height < 6:
        return None

    h, w = height, width

    # --- 1. Barrier rows ---
    b1_max = h - 4
    if b1_max < 2:
        return None
    b1 = rng.randint(2, b1_max)
    b2_min, b2_max = b1 + 2, h - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Trap side ---
    trap_side = rng.choice([0, w - 1])
    if trap_side == 0:
        trap_half = list(range(0, w // 2))
        far_half = list(range(w // 2, w))
    else:
        trap_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))

    # Need at least 2 trap cols (gate_r + gate_b) and 1 far col (bypass)
    if len(trap_half) < 2 or not far_half:
        return None

    # --- 3. Barrier 1: gate_r + gate_b on trap-half, bypass on far-half ---
    gate_r_col = rng.choice(trap_half)
    remaining_trap = [c for c in trap_half if c != gate_r_col]
    gate_b_col = rng.choice(remaining_trap)
    bypass_candidates = [c for c in far_half if c != gate_r_col and c != gate_b_col]
    if not bypass_candidates:
        return None
    bypass_col = rng.choice(bypass_candidates)

    # --- 4. Barrier 2: D_exit at any column ---
    exit_door_col = rng.randint(0, w - 1)

    # --- 5. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: walls except gate_r_col, gate_b_col, bypass_col
    for c in range(w):
        if c not in (gate_r_col, gate_b_col, bypass_col):
            walls.add(Pos(b1, c))

    # Barrier 2: walls except exit_door_col
    for c in range(w):
        if c != exit_door_col:
            walls.add(Pos(b2, c))

    # --- 6. Zone 2: exit placement ---
    zone2: set[Pos] = {
        Pos(r, c) for r in range(b2 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone2) < 2:
        return None
    exit_entry = Pos(b2 + 1, exit_door_col)
    if exit_entry not in zone2:
        return None
    zone2_comp = _flood_fill(exit_entry, zone2)
    if not zone2_comp:
        return None
    # Bias exit toward far side
    far_col = w - 1 - trap_side
    zone2_sorted = sorted(zone2_comp, key=lambda p: abs(p.col - far_col))
    n_pick = max(1, len(zone2_sorted) // 2)
    exit_pos = rng.choice(zone2_sorted[:n_pick])

    # --- 7. Zone 1 connectivity ---
    zone1: set[Pos] = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }
    if not zone1:
        return None
    bypass_entry = Pos(b1 + 1, bypass_col)
    gate_r_entry = Pos(b1 + 1, gate_r_col)
    gate_b_entry = Pos(b1 + 1, gate_b_col)
    exit_door_approach = Pos(b2 - 1, exit_door_col)
    zone1_critical = [bypass_entry, gate_r_entry, gate_b_entry, exit_door_approach]
    if not all(p in zone1 for p in zone1_critical):
        return None
    zone1_comp = _flood_fill(bypass_entry, zone1)
    if not all(p in zone1_comp for p in zone1_critical):
        return None

    # --- 8. Zone 0 key placement ---
    zone0: set[Pos] = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 4:
        return None

    # K_r: convergent forks bait near gate_r
    kr_pos = _convergent_key_pos(Pos(b1, gate_r_col), zone0, rng)
    if kr_pos is None:
        return None

    # K_b: convergent forks bait near gate_b (exclude K_r pos)
    zone0_for_kb = zone0 - {kr_pos}
    kb_pos = _convergent_key_pos(Pos(b1, gate_b_col), zone0_for_kb, rng)
    if kb_pos is None:
        return None

    # K_g: free placement, spaced from K_r and K_b
    kg_candidates = [p for p in zone0 if p not in (kr_pos, kb_pos)]
    if not kg_candidates:
        return None
    kg_result = _select_key_positions(
        kg_candidates, 1, rng,
        already_placed=frozenset({kr_pos, kb_pos}),
        reference=exit_pos,
    )
    if kg_result is None:
        return None
    kg_pos = kg_result[0]

    # --- 9. Start placement (far from trap_side, top of Zone 0) ---
    start_candidates = [p for p in zone0 if p not in (kr_pos, kb_pos, kg_pos)]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - far_col) - p.row)
    n_pick = max(1, len(start_candidates) // 3)
    start = rng.choice(start_candidates[:n_pick])

    # --- 10. Zone 0 connectivity ---
    bypass_approach = Pos(b1 - 1, bypass_col)
    gate_r_approach = Pos(b1 - 1, gate_r_col)
    gate_b_approach = Pos(b1 - 1, gate_b_col)
    zone0_critical = [kr_pos, kb_pos, kg_pos,
                      bypass_approach, gate_r_approach, gate_b_approach]
    if not all(p in zone0 for p in zone0_critical):
        return None
    zone0_comp = _flood_fill(start, zone0)
    if not all(p in zone0_comp for p in zone0_critical):
        return None

    # --- 11. Optional extra walls (decorative, connectivity-preserving) ---
    # Zone 0
    z0_protected = {start, kr_pos, kb_pos, kg_pos,
                    bypass_approach, gate_r_approach, gate_b_approach}
    z0_extras = [p for p in zone0 if p not in z0_protected]
    rng.shuffle(z0_extras)
    n_extra = rng.randint(0, min(2, len(z0_extras) // 4))
    for pos in z0_extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in z0_protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    # Zone 1
    z1_protected = set(zone1_critical)
    z1_extras = [p for p in zone1 if p not in z1_protected]
    rng.shuffle(z1_extras)
    n_extra = rng.randint(0, min(2, len(z1_extras) // 4))
    for pos in z1_extras:
        if n_extra <= 0:
            break
        rem = zone1 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in z1_protected):
            walls.add(pos)
            zone1.discard(pos)
            n_extra -= 1

    # --- 12. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=kr_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=kb_pos, color="blue"),
        "K_g": KeyInfo(id="K_g", pos=kg_pos, color="green"),
    }
    doors = {
        "gate_r": DoorInfo(id="gate_r", pos=Pos(b1, gate_r_col),
                           required_keys=("red",)),
        "gate_b": DoorInfo(id="gate_b", pos=Pos(b1, gate_b_col),
                           required_keys=("blue",)),
        "D_exit": DoorInfo(id="D_exit", pos=Pos(b2, exit_door_col),
                           required_keys=("red", "blue", "green")),
    }
    grid = ChainVaultGrid(
        width=width, height=height,
        walls=frozenset(walls), keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Template 12: Key Collector
# ---------------------------------------------------------------------------

def generate_key_collector_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a key collector grid: triple/quad-lock exit with sequential single-lock traps.

    MECHANIC: 3 keys (R, B, G), one triple-lock exit (R+B+G), and 2 single-lock
    trap doors (blue, green). The exit demands ALL three keys, so wasting any
    single colored key on a trap is immediately fatal — true scarcity.

    ENHANCED (50% at w>=6): 4th yellow key + quad-lock exit (R+B+G+Y).
    Yellow single-lock trap on barrier 1 alongside blue trap.

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + K_r + K_b (near trap_b via convergent forks)
        Barrier 1 (row b1):         D_trap_b (blue, trap-side) + bypass (free, far-side)
        Zone 1 (rows b1+1..b2-1):  K_g (near trap_g via convergent forks)
        Barrier 2 (row b2):         D_trap_g (green, trap-side) + D_exit (R+B+G, far-side)
        Zone 2 (rows b2+1..h-1):   Split by divider_col:
          Zone 2T (trap side):     Behind D_trap_g. Real walkable area, no exit.
          Zone 2C (far side):      Behind D_exit. Contains exit.

    ANTI-GREEDY (convergent forks + cost asymmetry):
    1. K_b placed above trap_b on barrier 1 — greedy agent collects blue, immediately
       wastes it on the cheap single-lock trap_b. Blue gone → can't open R+B+G → stuck.
    2. K_g placed above trap_g on barrier 2 — greedy agent collects green, immediately
       wastes it on the cheap single-lock trap_g into Zone 2T (real area, looks like
       progress). Green gone → can't open R+B+G → stuck.
    Both traps lead to real areas (Zone 1 / Zone 2T), not dead ends.

    CORRECT PATH: collect K_r + K_b → free bypass (barrier 1) → collect K_g →
    use D_exit (R+B+G) on barrier 2 → Zone 2C → exit.

    UNIQUE REASONING: Must collect and PRESERVE all 3 keys for a single triple-lock.
    Each barrier offers a cheaper single-lock temptation that's independently fatal.
    """
    if rng is None:
        rng = random.Random()
    if width < 6 or height < 6:
        return None

    # --- Path Commitment variant (50% at width ≥ 7) ---
    if width >= 7 and rng.random() < 0.5:
        result = _generate_kc_path_commitment(width, height, rng)
        if result is not None:
            return result
        # Fall through to standard variant if path commitment fails

    h, w = height, width

    # --- 1. Barrier rows ---
    b1_max = h - 4
    if b1_max < 2:
        return None
    b1 = rng.randint(2, b1_max)
    b2_min, b2_max = b1 + 2, h - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Trap side (column direction where traps are placed) ---
    trap_side = rng.choice([0, w - 1])

    # --- 3. Column halves ---
    if trap_side == 0:
        trap_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        trap_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))

    if not trap_half or not far_half:
        return None

    # --- 4. Barrier 1 columns: trap_b on trap-side, bypass on far-side ---
    trap_b_col = rng.choice(trap_half)
    far_half_b1 = [c for c in far_half if c != trap_b_col]
    if not far_half_b1:
        return None
    bypass_col = rng.choice(far_half_b1)

    # --- 5. Barrier 2 columns: trap_g on trap-side, exit_door on far-side ---
    trap_g_col = rng.choice(trap_half)
    far_half_b2 = [c for c in far_half if c != trap_g_col]
    if not far_half_b2:
        return None
    exit_door_col = rng.choice(far_half_b2)

    # --- 5b. Zone 2 divider (separates trap_g area from exit) ---
    lo = min(trap_g_col, exit_door_col)
    hi = max(trap_g_col, exit_door_col)
    divider_candidates = list(range(lo + 1, hi))
    if not divider_candidates:
        return None
    # Bias divider toward exit_door_col to maximize Zone 2T (trap looks bigger)
    if exit_door_col > trap_g_col:
        divider_col = divider_candidates[-1]
    else:
        divider_col = divider_candidates[0]

    # --- 6. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: walls except trap_b_col and bypass_col
    for c in range(w):
        if c != trap_b_col and c != bypass_col:
            walls.add(Pos(b1, c))

    # Barrier 2: walls except trap_g_col and exit_door_col
    for c in range(w):
        if c != trap_g_col and c != exit_door_col:
            walls.add(Pos(b2, c))

    # Zone 2 divider: wall column through all Zone 2 rows, with one green
    # "tease" door that provides false hope (agent in Zone 2T already used
    # green on trap_g, so this door is provably unopenable from the trap side)
    tease_row = b2 + 1  # top of divider — most visible from barrier 2
    for r in range(b2 + 1, h):
        if r != tease_row:
            walls.add(Pos(r, divider_col))

    # --- Scaling: extra trap(s) at w>=7 / w>=8 ---
    extra_trap_kc_b1: Optional[int] = None
    extra_trap_kc_b2: Optional[int] = None
    if w >= 7:
        extra_trap_kc_b1 = _pick_extra_trap_col(
            b1, walls, {trap_b_col, bypass_col}, trap_half, rng)
        if extra_trap_kc_b1 is not None:
            walls.discard(Pos(b1, extra_trap_kc_b1))
    if w >= 8:
        extra_trap_kc_b2 = _pick_extra_trap_col(
            b2, walls, {trap_g_col, exit_door_col}, trap_half, rng)
        if extra_trap_kc_b2 is not None:
            walls.discard(Pos(b2, extra_trap_kc_b2))

    # --- Enhanced Key Collector: 4th key (yellow) + quad-lock exit ---
    enhanced_kc = w >= 6 and rng.random() < 0.5
    trap_y_col: Optional[int] = None
    if enhanced_kc:
        used_b1_cols = {trap_b_col, bypass_col}
        if extra_trap_kc_b1 is not None:
            used_b1_cols.add(extra_trap_kc_b1)
        trap_y_col = _pick_extra_trap_col(
            b1, walls, used_b1_cols, trap_half, rng)
        if trap_y_col is None:
            enhanced_kc = False
        else:
            walls.discard(Pos(b1, trap_y_col))

    # --- 7. Exit placement (Zone 2C only — behind D_exit) ---
    tease_pos = Pos(tease_row, divider_col)
    zone2 = {
        Pos(r, c) for r in range(b2 + 1, h) for c in range(w)
        if Pos(r, c) not in walls and Pos(r, c) != tease_pos
    }
    if not zone2:
        return None

    exit_door_entry = Pos(b2 + 1, exit_door_col)
    if exit_door_entry not in zone2:
        return None
    zone2c = _flood_fill(exit_door_entry, zone2)
    if not zone2c:
        return None

    # Place exit in Zone 2C, biased toward exit_door_col
    zone2c_list = sorted(zone2c, key=lambda p: abs(p.col - exit_door_col))
    n_pick = max(1, len(zone2c_list) // 2)
    exit_pos = rng.choice(zone2c_list[:n_pick])

    # --- 7b. Verify Zone 2T has real space (trap leads to real area) ---
    trap_g_entry = Pos(b2 + 1, trap_g_col)
    if trap_g_entry not in zone2:
        return None
    zone2t = _flood_fill(trap_g_entry, zone2)
    if len(zone2t) < 2:
        return None  # trap area must be a real walkable space
    if exit_pos in zone2t:
        return None  # divider failed — trap_g can reach exit

    # --- 8. Anti-greedy: convergent forks + cost asymmetry ---
    # No manhattan checks needed — anti-greedy relies on:
    # (a) convergent forks: K_b near trap_b, K_g near trap_g
    # (b) cost asymmetry: single-lock traps are cheaper than triple-lock D_exit
    # Validated post-hoc by the greedy filter in generate_validated_instance.

    # --- 9. Zone 0 key placement ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 3:
        return None

    # K_b: near trap_b door (convergent forks bait)
    kb_pos = _convergent_key_pos(Pos(b1, trap_b_col), zone0, rng)
    if kb_pos is None:
        return None

    # K_y: near trap_y door (convergent forks bait) — enhanced only
    ky_pos: Optional[Pos] = None
    if enhanced_kc:
        ky_pos = _convergent_key_pos(Pos(b1, trap_y_col), zone0, rng)
        if ky_pos is None:
            # Revert: put barrier wall back and fall back to standard variant
            walls.add(Pos(b1, trap_y_col))
            enhanced_kc = False
            trap_y_col = None

    # K_r: spaced from K_b (and K_y if enhanced) in zone0
    kr_exclude = {kb_pos}
    if enhanced_kc and ky_pos is not None:
        kr_exclude.add(ky_pos)
    kr_candidates = [p for p in zone0 if p not in kr_exclude]
    if not kr_candidates:
        return None
    kr_result = _select_key_positions(
        kr_candidates, 1, rng,
        already_placed=frozenset(kr_exclude), reference=exit_pos,
    )
    if kr_result is None:
        return None
    kr_pos = kr_result[0]

    # --- P2.15: Hidden key behind red door in Zone 1 (pocket approach) ---
    # Isolate an edge/corner cell in Zone 1 behind a red door.
    # K_g goes in the pocket; K_r2 in Zone 0 provides the extra red key.
    hidden_key = w >= 7 and rng.random() < 0.5
    hidden_door_pos: Optional[Pos] = None
    hidden_pocket_pos: Optional[Pos] = None
    kr2_pos: Optional[Pos] = None
    if hidden_key:
        kr2_exclude = {kb_pos, kr_pos}
        if enhanced_kc and ky_pos is not None:
            kr2_exclude.add(ky_pos)
        kr2_cands = [p for p in zone0 if p not in kr2_exclude]
        if kr2_cands:
            kr2_pos = rng.choice(kr2_cands)
        else:
            hidden_key = False
    if hidden_key:
        # Critical Zone 1 positions that must remain connected
        z1_critical_hk = {
            Pos(b1 + 1, bypass_col), Pos(b1 + 1, trap_b_col),
            Pos(b2 - 1, trap_g_col), Pos(b2 - 1, exit_door_col),
        }
        if enhanced_kc:
            z1_critical_hk.add(Pos(b1 + 1, trap_y_col))
        z1_floor = {
            Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
            if Pos(r, c) not in walls
        }
        # Find edge/corner cells with 1-2 floor neighbors
        pocket_candidates: list[tuple[Pos, list[Pos]]] = []
        for p in z1_floor:
            if p in z1_critical_hk:
                continue
            floor_nbs = [
                Pos(p.row + dr, p.col + dc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if Pos(p.row + dr, p.col + dc) in z1_floor
            ]
            if 1 <= len(floor_nbs) <= 2:
                pocket_candidates.append((p, floor_nbs))
        rng.shuffle(pocket_candidates)
        pocket_ok = False
        hk_added_walls: list[Pos] = []
        for p_cell, floor_nbs in pocket_candidates:
            for d_pos in floor_nbs:
                to_wall = [nb for nb in floor_nbs if nb != d_pos]
                # Remaining Zone 1 after removing pocket, door, new walls
                test_z1 = z1_floor - {p_cell} - set(to_wall) - {d_pos}
                if not test_z1 or not _is_connected(test_z1):
                    continue
                if not all(cp in test_z1 for cp in z1_critical_hk):
                    continue
                # Door must be adjacent to at least one cell in test_z1
                door_adj = [
                    Pos(d_pos.row + dr, d_pos.col + dc)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if Pos(d_pos.row + dr, d_pos.col + dc) in test_z1
                ]
                if not door_adj:
                    continue
                # Apply pocket
                for nb in to_wall:
                    walls.add(nb)
                    hk_added_walls.append(nb)
                hidden_door_pos = d_pos
                hidden_pocket_pos = p_cell
                pocket_ok = True
                break
            if pocket_ok:
                break
        if not pocket_ok:
            hidden_key = False
            kr2_pos = None

    # --- 10. Zone 1 key placement ---
    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }
    if not zone1:
        return None

    # K_g: near trap_g door (convergent forks bait)
    if hidden_key and hidden_pocket_pos is not None:
        # K_g inside the pocket (behind the red door)
        kg_pos = hidden_pocket_pos
    else:
        kg_pos = _convergent_key_pos(Pos(b2, trap_g_col), zone1, rng)
    if kg_pos is None:
        return None

    # --- 11. Start placement (biased toward far side, away from traps) ---
    far_side = w - 1 - trap_side
    start_candidates = [p for p in zone0 if p != kb_pos and p != kr_pos]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - far_side) - p.row)
    n_pick = max(1, len(start_candidates) // 3)
    start = rng.choice(start_candidates[:n_pick])

    # --- 12. Connectivity checks ---
    # Zone 0: start, K_b, K_r, bypass approach, trap_b approach all connected
    bypass_approach = Pos(b1 - 1, bypass_col)
    trap_b_approach = Pos(b1 - 1, trap_b_col)
    if bypass_approach not in zone0 or trap_b_approach not in zone0:
        return None
    zone0_check = [kb_pos, kr_pos, bypass_approach, trap_b_approach]
    if hidden_key and kr2_pos is not None:
        zone0_check.append(kr2_pos)
    if enhanced_kc:
        trap_y_approach = Pos(b1 - 1, trap_y_col)
        if trap_y_approach not in zone0:
            return None
        zone0_check.extend([ky_pos, trap_y_approach])
    zone0_comp = _flood_fill(start, zone0)
    if not all(p in zone0_comp for p in zone0_check):
        return None

    # Zone 1: all critical points connected
    # When hidden_key is active, exclude door and pocket from Zone 1
    # (door is closed, pocket is only reachable after opening door)
    if hidden_key and hidden_door_pos is not None:
        zone1.discard(hidden_door_pos)
        zone1.discard(hidden_pocket_pos)
    bypass_entry = Pos(b1 + 1, bypass_col)
    trap_b_entry = Pos(b1 + 1, trap_b_col)
    trap_g_approach = Pos(b2 - 1, trap_g_col)
    exit_door_approach = Pos(b2 - 1, exit_door_col)
    zone1_critical = [
        bypass_entry, trap_b_entry,
        trap_g_approach, exit_door_approach,
    ]
    # kg_pos only needs connectivity when NOT behind a door
    if not hidden_key:
        zone1_critical.append(kg_pos)
    if enhanced_kc:
        trap_y_entry = Pos(b1 + 1, trap_y_col)
        if trap_y_entry not in zone1:
            return None
        zone1_critical.append(trap_y_entry)
    if not all(p in zone1 for p in zone1_critical):
        return None
    zone1_comp = _flood_fill(bypass_entry, zone1)
    if not all(p in zone1_comp for p in zone1_critical):
        return None

    # Zone 2C and Zone 2T already validated above (step 7/7b)

    # --- 13. Extra walls (0-2 per zone, connectivity-preserving) ---
    # Zone 0
    zone0_protected = {start, kb_pos, kr_pos, bypass_approach, trap_b_approach}
    if enhanced_kc:
        zone0_protected.add(ky_pos)
        zone0_protected.add(Pos(b1 - 1, trap_y_col))
    zone0_extras = [p for p in zone0 if p not in zone0_protected]
    rng.shuffle(zone0_extras)
    n_extra = rng.randint(0, min(2, len(zone0_extras) // 4))
    for pos in zone0_extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone0_protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    # Zone 1
    zone1_protected = set(zone1_critical)
    zone1_extras = [p for p in zone1 if p not in zone1_protected]
    rng.shuffle(zone1_extras)
    n_extra = rng.randint(0, min(2, len(zone1_extras) // 4))
    for pos in zone1_extras:
        if n_extra <= 0:
            break
        rem = zone1 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone1_protected):
            walls.add(pos)
            zone1.discard(pos)
            n_extra -= 1

    # --- 14. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=kr_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=kb_pos, color="blue"),
        "K_g": KeyInfo(id="K_g", pos=kg_pos, color="green"),
    }
    if enhanced_kc and ky_pos is not None:
        keys["K_y"] = KeyInfo(id="K_y", pos=ky_pos, color="yellow")
    if hidden_key and kr2_pos is not None:
        keys["K_r2"] = KeyInfo(id="K_r2", pos=kr2_pos, color="red")
    exit_keys = ("red", "blue", "green", "yellow") if enhanced_kc else ("red", "blue", "green")
    doors = {
        "D_trap_b": DoorInfo(
            id="D_trap_b", pos=Pos(b1, trap_b_col), required_keys=("blue",)),
        "D_trap_g": DoorInfo(
            id="D_trap_g", pos=Pos(b2, trap_g_col), required_keys=("green",)),
        "D_exit": DoorInfo(
            id="D_exit", pos=Pos(b2, exit_door_col),
            required_keys=exit_keys),
        "D_tease": DoorInfo(
            id="D_tease", pos=Pos(tease_row, divider_col),
            required_keys=("green",)),
    }
    if enhanced_kc and trap_y_col is not None:
        doors["D_trap_y"] = DoorInfo(
            id="D_trap_y", pos=Pos(b1, trap_y_col), required_keys=("yellow",))
    if extra_trap_kc_b1 is not None:
        doors["D_trap_r"] = DoorInfo(
            id="D_trap_r", pos=Pos(b1, extra_trap_kc_b1), required_keys=("red",))
    if extra_trap_kc_b2 is not None:
        doors["D_trap_r2"] = DoorInfo(
            id="D_trap_r2", pos=Pos(b2, extra_trap_kc_b2), required_keys=("red",))
    if hidden_key and hidden_door_pos is not None:
        doors["D_hidden"] = DoorInfo(
            id="D_hidden", pos=hidden_door_pos,
            required_keys=("red",))
    grid = ChainVaultGrid(
        width=width, height=height,
        walls=frozenset(walls), keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Template 13: Decoy Key (two-stage temptation)
# ---------------------------------------------------------------------------


def _generate_dk_3stage(
    width: int,
    height: int,
    rng: random.Random,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """P2.16 — 3-stage temptation: 3 barriers, yellow saved from Stage 1 to Stage 3.

    Keys: K_r (red), K_y (yellow), K_b (blue), K_g (green).

    Stage 1: K_y near D_trap_y on b1 -> resist -> use K_r on D_correct_r.
    Stage 2: K_b near D_trap_b on b2 -> resist -> use K_g on D_gate_g.
    Stage 3: D_trap_y3 on b3 (yellow) -> resist -> use K_b on D_gate_b.
    Final:   use K_y on D_exit_y -> exit.

    Deadlocks:
    1. USE_KEY(yellow) on D_trap_y -> Zone 1T (dead end).
    2. USE_KEY(blue) on D_trap_b -> Zone 2 -> D_gate_b needs blue -> stuck.
    3. USE_KEY(yellow) on D_trap_y3 -> Zone 3 -> D_exit_y needs yellow -> stuck.
    """
    h, w = height, width
    if h < 8 or w < 6:
        return None

    # --- Barrier rows ---
    # zone0(>=1) + b1 + zone1(>=1) + b2 + zone2(>=1) + b3 + zone3(>=2)
    b1_max = h - 7
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min, b2_max = b1 + 2, h - 5
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)
    b3_min, b3_max = b2 + 2, h - 3
    if b3_min > b3_max:
        return None
    b3 = rng.randint(b3_min, b3_max)

    # --- Exit pocket ---
    exit_side = rng.choice([0, w - 1])
    if exit_side == 0:
        exit_col, exit_door_col = 0, 1
    else:
        exit_col, exit_door_col = w - 1, w - 2
    exit_pos = Pos(h - 1, exit_col)
    exit_door_pos = Pos(h - 1, exit_door_col)
    pocket_wall_pos = Pos(h - 2, exit_col)

    # --- Column halves ---
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))
    if not exit_half or not far_half:
        return None

    # --- Barrier 1: D_trap_y (exit-side), D_correct_r (far-side) ---
    trap_y_col = rng.choice(exit_half)
    far_b1 = [c for c in far_half if c != trap_y_col]
    if not far_b1:
        return None
    correct_r_col = rng.choice(far_b1)

    # Divider between trap and correct on b1
    lo, hi = min(trap_y_col, correct_r_col), max(trap_y_col, correct_r_col)
    div_opts = list(range(lo + 1, hi))
    if not div_opts:
        return None
    if correct_r_col > trap_y_col:
        divider_col = div_opts[-1]
    else:
        divider_col = div_opts[0]

    # Zone 1C cols (correct side of divider)
    if correct_r_col > divider_col:
        zone1c_cols = list(range(divider_col + 1, w))
    else:
        zone1c_cols = list(range(0, divider_col))

    # --- Barrier 2: D_trap_b + D_gate_g in Zone 1C cols ---
    b2_cands = [c for c in zone1c_cols if c != exit_col and c != exit_door_col]
    if len(b2_cands) < 2:
        return None
    rng.shuffle(b2_cands)
    trap_b_col = b2_cands[0]
    gate_g_col = b2_cands[1]

    # --- Barrier 3: D_trap_y3 + D_gate_b (full width, avoid exit pocket) ---
    b3_cands = [c for c in range(w) if c != exit_col and c != exit_door_col]
    if len(b3_cands) < 2:
        return None
    rng.shuffle(b3_cands)
    trap_y3_col = b3_cands[0]
    gate_b_col = b3_cands[1]

    # --- Anti-greedy manhattan check ---
    if not _trap_closer_to_exit(Pos(b1, trap_y_col), Pos(b1, correct_r_col), exit_pos):
        return None

    # --- Build walls ---
    walls: set[Pos] = set()
    for c in range(w):
        if c != trap_y_col and c != correct_r_col:
            walls.add(Pos(b1, c))
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, divider_col))
    for c in range(w):
        if c != trap_b_col and c != gate_g_col:
            walls.add(Pos(b2, c))
    for c in range(w):
        if c != trap_y3_col and c != gate_b_col:
            walls.add(Pos(b3, c))
    walls.add(pocket_wall_pos)

    # --- Zones ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 3:
        return None

    zone1_all = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }
    correct_r_entry = Pos(b1 + 1, correct_r_col)
    trap_y_entry = Pos(b1 + 1, trap_y_col)
    if correct_r_entry not in zone1_all or trap_y_entry not in zone1_all:
        return None
    zone1c = _flood_fill(correct_r_entry, zone1_all)
    zone1t = _flood_fill(trap_y_entry, zone1_all)
    if len(zone1t) < 2 or (zone1t & zone1c):
        return None

    zone2 = {
        Pos(r, c) for r in range(b2 + 1, b3) for c in range(w)
        if Pos(r, c) not in walls
    }
    if not zone2:
        return None

    zone3 = {
        Pos(r, c) for r in range(b3 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    if not zone3:
        return None

    # --- Key placement ---
    ky_pos = _convergent_key_pos(Pos(b1, trap_y_col), zone0, rng)
    if ky_pos is None:
        return None
    kr_cands = [p for p in zone0 if p != ky_pos]
    if not kr_cands:
        return None
    kr_pos = rng.choice(kr_cands)
    kb_pos = _convergent_key_pos(Pos(b2, trap_b_col), zone1c, rng)
    if kb_pos is None:
        return None
    kg_cands = [p for p in zone1c if p != kb_pos]
    if not kg_cands:
        return None
    kg_pos = rng.choice(kg_cands)

    # --- Start ---
    start_cands = [p for p in zone0 if p != ky_pos and p != kr_pos]
    if not start_cands:
        return None
    start_cands.sort(key=lambda p: abs(p.col - exit_side))
    n_pick = max(1, len(start_cands) // 3)
    start = rng.choice(start_cands[:n_pick])

    # --- Connectivity checks ---
    trap_y_approach = Pos(b1 - 1, trap_y_col)
    correct_r_approach = Pos(b1 - 1, correct_r_col)
    if trap_y_approach not in zone0 or correct_r_approach not in zone0:
        return None
    z0_comp = _flood_fill(start, zone0)
    if not all(p in z0_comp for p in
               [ky_pos, kr_pos, trap_y_approach, correct_r_approach]):
        return None

    gate_g_approach = Pos(b2 - 1, gate_g_col)
    trap_b_approach = Pos(b2 - 1, trap_b_col)
    z1c_critical = [correct_r_entry, kb_pos, kg_pos,
                    gate_g_approach, trap_b_approach]
    if not all(p in zone1c for p in z1c_critical):
        return None
    z1c_comp = _flood_fill(correct_r_entry, zone1c)
    if not all(p in z1c_comp for p in z1c_critical):
        return None

    gate_g_entry = Pos(b2 + 1, gate_g_col)
    trap_b_entry = Pos(b2 + 1, trap_b_col)
    gate_b_approach = Pos(b3 - 1, gate_b_col)
    trap_y3_approach = Pos(b3 - 1, trap_y3_col)
    z2_critical = [gate_g_entry, trap_b_entry,
                   gate_b_approach, trap_y3_approach]
    if not all(p in zone2 for p in z2_critical):
        return None
    z2_comp = _flood_fill(gate_g_entry, zone2)
    if not all(p in z2_comp for p in z2_critical):
        return None

    gate_b_entry = Pos(b3 + 1, gate_b_col)
    trap_y3_entry = Pos(b3 + 1, trap_y3_col)
    exit_y_approach = Pos(h - 2, exit_door_col)
    z3_critical = [gate_b_entry, trap_y3_entry, exit_y_approach]
    if not all(p in zone3 for p in z3_critical):
        return None
    z3_comp = _flood_fill(gate_b_entry, zone3)
    if not all(p in z3_comp for p in z3_critical):
        return None

    if pocket_wall_pos not in walls:
        return None

    # --- Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=kr_pos, color="red"),
        "K_y": KeyInfo(id="K_y", pos=ky_pos, color="yellow"),
        "K_b": KeyInfo(id="K_b", pos=kb_pos, color="blue"),
        "K_g": KeyInfo(id="K_g", pos=kg_pos, color="green"),
    }
    doors = {
        "D_trap_y": DoorInfo(
            id="D_trap_y", pos=Pos(b1, trap_y_col),
            required_keys=("yellow",)),
        "D_correct_r": DoorInfo(
            id="D_correct_r", pos=Pos(b1, correct_r_col),
            required_keys=("red",)),
        "D_trap_b": DoorInfo(
            id="D_trap_b", pos=Pos(b2, trap_b_col),
            required_keys=("blue",)),
        "D_gate_g": DoorInfo(
            id="D_gate_g", pos=Pos(b2, gate_g_col),
            required_keys=("green",)),
        "D_trap_y3": DoorInfo(
            id="D_trap_y3", pos=Pos(b3, trap_y3_col),
            required_keys=("yellow",)),
        "D_gate_b": DoorInfo(
            id="D_gate_b", pos=Pos(b3, gate_b_col),
            required_keys=("blue",)),
        "D_exit_y": DoorInfo(
            id="D_exit_y", pos=exit_door_pos,
            required_keys=("yellow",)),
    }
    grid = ChainVaultGrid(
        width=width, height=height,
        walls=frozenset(walls), keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def generate_decoy_key_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate a decoy key grid: two-stage temptation with visible trap doors.

    MECHANIC: 3 keys (red, yellow, blue). Two barriers with trap doors.
    Stage 1: yellow key + visible yellow trap door on barrier 1 (exit-side).
    Stage 2: blue key + visible blue trap door on barrier 2.
    At each stage, the greedy action (use key on adjacent matching door) is fatal.

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + K_r (red) + K_y (yellow, near D_trap_y)
        Barrier 1 (row b1):         D_trap_y (yellow, exit-side) + D_correct_r (red, far-side)
        Zone 1 (rows b1+1..b2-1):  Split by vertical divider:
          Zone 1T (exit-side):      Real walkable area (>=2 cells), dead end
          Zone 1C (far-side):       K_b (blue, near D_trap_b)
        Barrier 2 (row b2):         D_trap_b (blue) + D_gate_y (yellow) [Zone 1C cols]
        Zone 2 (rows b2+1..h-1):   Open area + exit pocket (D_exit_b blue + exit)

    ANTI-GREEDY (two-stage temptation):
    - K_y placed near D_trap_y -> greedy opens D_trap_y -> Zone 1T (dead end)
    - K_b placed near D_trap_b -> greedy opens D_trap_b -> Zone 2 but blue wasted
    - manhattan(D_trap_y, exit) < manhattan(D_correct_r, exit) -- hard filter

    CORRECT PATH: Zone 0 -> K_r + K_y -> D_correct_r (red) -> Zone 1C ->
    K_b -> D_gate_y (yellow) -> Zone 2 -> D_exit_b (blue) -> exit.

    DEADLOCKS (>=2):
    1. USE_KEY(yellow) on D_trap_y -> Zone 1T -> stuck (no D_gate_y access)
    2. USE_KEY(blue) on D_trap_b -> Zone 2 -> D_exit_b needs blue -> stuck

    UNIQUE REASONING: "I have a key and there's a matching door right here --
    but I should NOT use it here." Forward-planning under uncertainty about
    future resource needs.
    """
    if rng is None:
        rng = random.Random()
    if width < 5 or height < 6:
        return None

    # --- P2.16: 3-stage temptation at 8×8+ ---
    if width >= 6 and height >= 8 and rng.random() < 0.5:
        result = _generate_dk_3stage(width, height, rng)
        if result is not None:
            return result

    # --- Stochastic mixed variant (P1.10) ---
    # At w>=6, 50% chance: Stage 1 flips to "should USE matching key"
    # (breaks "always save matching" meta-policy).
    mixed = width >= 6 and rng.random() < 0.5

    # --- P2.17: Quantity mismatch variant ---
    # 2 yellow keys (losing 1 is OK), 1 blue (losing it is fatal).
    # Only at w>=7 (ensures extra trap doors for sufficient deadlocks).
    # Exclusive with mixed variant.
    qty_mismatch = not mixed and width >= 7 and rng.random() < 0.5

    h, w = height, width

    # --- 1. Barrier rows ---
    # Need: zone0(>=1) + b1 + zone1(>=1) + b2 + zone2(>=2, for exit pocket)
    b1_max = h - 5
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min, b2_max = b1 + 2, h - 3
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Exit pocket (standard relay/scarce_resource pattern) ---
    exit_side = rng.choice([0, w - 1])
    if exit_side == 0:
        exit_col, exit_door_col = 0, 1
    else:
        exit_col, exit_door_col = w - 1, w - 2

    exit_pos = Pos(h - 1, exit_col)
    exit_door_pos = Pos(h - 1, exit_door_col)  # D_exit_b position
    pocket_wall_pos = Pos(h - 2, exit_col)

    # --- 3. Column halves (exit-side vs far-side) ---
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))

    if not exit_half or not far_half:
        return None

    # --- 4. Barrier 1 columns: trap_y on exit-side, correct_r on far-side ---
    trap_y_col = rng.choice(exit_half)
    far_half_b1 = [c for c in far_half if c != trap_y_col]
    if not far_half_b1:
        return None
    correct_r_col = rng.choice(far_half_b1)

    # --- 5. Vertical divider (Zone 1) ---
    lo = min(trap_y_col, correct_r_col)
    hi = max(trap_y_col, correct_r_col)
    divider_options = list(range(lo + 1, hi))
    if not divider_options:
        return None
    # Bias toward correct_r_col to maximize Zone 1T
    if correct_r_col > trap_y_col:
        divider_col = divider_options[-1]
    else:
        divider_col = divider_options[0]

    # --- 6. Barrier 2 columns (both in Zone 1C) ---
    # Zone 1C columns: far side of divider
    if correct_r_col > divider_col:
        zone1c_cols = list(range(divider_col + 1, w))
    else:
        zone1c_cols = list(range(0, divider_col))

    # Exclude exit pocket columns from candidates
    b2_candidates = [c for c in zone1c_cols
                     if c != exit_col and c != exit_door_col]
    if len(b2_candidates) < 2:
        return None
    rng.shuffle(b2_candidates)
    trap_b_col = b2_candidates[0]
    gate_y_col = b2_candidates[1]

    # --- 7. Anti-greedy manhattan check ---
    # D_trap_y must be closer to exit than D_correct_r
    if not _trap_closer_to_exit(Pos(b1, trap_y_col), Pos(b1, correct_r_col), exit_pos):
        return None

    # --- 8. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: all cols except trap_y_col and correct_r_col
    for c in range(w):
        if c != trap_y_col and c != correct_r_col:
            walls.add(Pos(b1, c))

    # Vertical divider through Zone 1
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, divider_col))

    # Barrier 2: all cols except trap_b_col and gate_y_col
    for c in range(w):
        if c != trap_b_col and c != gate_y_col:
            walls.add(Pos(b2, c))

    # Pocket wall
    walls.add(pocket_wall_pos)

    # --- Scaling: extra trap(s) at w>=7 / w>=8 ---
    extra_trap_dk_b1: Optional[int] = None
    extra_trap_dk_b2: Optional[int] = None
    if w >= 7:
        extra_trap_dk_b1 = _pick_extra_trap_col(
            b1, walls, {trap_y_col, correct_r_col}, exit_half, rng)
        if extra_trap_dk_b1 is not None:
            walls.discard(Pos(b1, extra_trap_dk_b1))
    if w >= 8:
        extra_trap_dk_b2 = _pick_extra_trap_col(
            b2, walls, {trap_b_col, gate_y_col}, exit_half, rng)
        if extra_trap_dk_b2 is not None:
            walls.discard(Pos(b2, extra_trap_dk_b2))

    # --- 9. Zone computation ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 3:
        return None

    zone1_all = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }

    # Zone 1C: flood fill from correct_r entry
    correct_r_entry = Pos(b1 + 1, correct_r_col)
    if correct_r_entry not in zone1_all:
        return None
    zone1c = _flood_fill(correct_r_entry, zone1_all)

    # Zone 1T: flood fill from trap_y entry
    trap_y_entry = Pos(b1 + 1, trap_y_col)
    if trap_y_entry not in zone1_all:
        return None
    zone1t = _flood_fill(trap_y_entry, zone1_all)

    # Zone 1T must be disconnected from Zone 1C and have >=2 cells
    if len(zone1t) < 2:
        return None
    if zone1t & zone1c:
        return None  # divider failed

    zone2 = {
        Pos(r, c) for r in range(b2 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    if not zone2:
        return None

    # --- 10. Key placement ---
    # K_y: near D_trap_y (convergent forks bait, stage 1)
    ky_pos = _convergent_key_pos(Pos(b1, trap_y_col), zone0, rng)
    if ky_pos is None:
        return None

    # K_r: random in zone0, not on K_y
    kr_candidates = [p for p in zone0 if p != ky_pos]
    if not kr_candidates:
        return None
    kr_pos = rng.choice(kr_candidates)

    # K_b: near D_trap_b (convergent forks bait, stage 2)
    kb_pos = _convergent_key_pos(Pos(b2, trap_b_col), zone1c, rng)
    if kb_pos is None:
        return None

    # P2.17: K_y2 (second yellow) in Zone 0
    ky2_pos: Optional[Pos] = None
    if qty_mismatch:
        ky2_cands = [p for p in zone0 if p != ky_pos and p != kr_pos]
        if ky2_cands:
            ky2_pos = rng.choice(ky2_cands)
        else:
            qty_mismatch = False

    # --- 11. Start placement (biased toward exit side — encounters trap first) ---
    start_exclude = {ky_pos, kr_pos}
    if qty_mismatch and ky2_pos is not None:
        start_exclude.add(ky2_pos)
    start_candidates = [p for p in zone0 if p not in start_exclude]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - exit_side))
    n_pick = max(1, len(start_candidates) // 3)
    start = rng.choice(start_candidates[:n_pick])

    # --- 12. Connectivity checks ---
    # Zone 0: start, K_y, K_r, trap_y approach, correct_r approach all connected
    trap_y_approach = Pos(b1 - 1, trap_y_col)
    correct_r_approach = Pos(b1 - 1, correct_r_col)
    if trap_y_approach not in zone0 or correct_r_approach not in zone0:
        return None
    z0_check = [ky_pos, kr_pos, trap_y_approach, correct_r_approach]
    if qty_mismatch and ky2_pos is not None:
        z0_check.append(ky2_pos)
    zone0_comp = _flood_fill(start, zone0)
    if not all(p in zone0_comp for p in z0_check):
        return None

    # Zone 1C: correct_r_entry -> K_b -> gate_y approach -> trap_b approach
    gate_y_approach = Pos(b2 - 1, gate_y_col)
    trap_b_approach = Pos(b2 - 1, trap_b_col)
    zone1c_critical = [correct_r_entry, kb_pos, gate_y_approach, trap_b_approach]
    if not all(p in zone1c for p in zone1c_critical):
        return None
    zone1c_comp = _flood_fill(correct_r_entry, zone1c)
    if not all(p in zone1c_comp for p in zone1c_critical):
        return None

    # Zone 2: gate_y entry + trap_b entry both reachable, D_exit_b approach reachable
    gate_y_entry = Pos(b2 + 1, gate_y_col)
    trap_b_entry = Pos(b2 + 1, trap_b_col)
    exit_b_approach = Pos(h - 2, exit_door_col)
    if gate_y_entry not in zone2 or trap_b_entry not in zone2:
        return None
    if exit_b_approach not in zone2:
        return None
    zone2_comp = _flood_fill(gate_y_entry, zone2)
    if not all(p in zone2_comp for p in [trap_b_entry, exit_b_approach]):
        return None

    # Pocket isolation: pocket_wall must be in walls
    if pocket_wall_pos not in walls:
        return None

    # --- 13. Extra walls (0-2 per zone, connectivity-preserving) ---
    # Zone 0
    zone0_protected = {start, ky_pos, kr_pos, trap_y_approach, correct_r_approach}
    if qty_mismatch and ky2_pos is not None:
        zone0_protected.add(ky2_pos)
    zone0_extras = [p for p in zone0 if p not in zone0_protected]
    rng.shuffle(zone0_extras)
    n_extra = rng.randint(0, min(2, len(zone0_extras) // 4))
    for pos in zone0_extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone0_protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    # Zone 1C
    zone1c_protected = set(zone1c_critical)
    zone1c_extras = [p for p in zone1c if p not in zone1c_protected]
    rng.shuffle(zone1c_extras)
    n_extra = rng.randint(0, min(2, len(zone1c_extras) // 4))
    for pos in zone1c_extras:
        if n_extra <= 0:
            break
        rem = zone1c - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone1c_protected):
            walls.add(pos)
            zone1c.discard(pos)
            n_extra -= 1

    # --- 14. Build grid ---
    if mixed:
        # Mixed: Stage 1 = USE matching (yellow on yellow), Stage 2 = SAVE matching
        # kr_pos (random) → K_y, ky_pos (bait S1) → K_b, kb_pos (bait S2) → K_y2
        keys = {
            "K_y": KeyInfo(id="K_y", pos=kr_pos, color="yellow"),
            "K_b": KeyInfo(id="K_b", pos=ky_pos, color="blue"),
            "K_y2": KeyInfo(id="K_y2", pos=kb_pos, color="yellow"),
        }
        doors = {
            "D_trap_b": DoorInfo(
                id="D_trap_b", pos=Pos(b1, trap_y_col), required_keys=("blue",)),
            "D_correct_y": DoorInfo(
                id="D_correct_y", pos=Pos(b1, correct_r_col), required_keys=("yellow",)),
            "D_trap_y": DoorInfo(
                id="D_trap_y", pos=Pos(b2, trap_b_col), required_keys=("yellow",)),
            "D_gate_b": DoorInfo(
                id="D_gate_b", pos=Pos(b2, gate_y_col), required_keys=("blue",)),
            "D_exit_y": DoorInfo(
                id="D_exit_y", pos=exit_door_pos, required_keys=("yellow",)),
        }
        if extra_trap_dk_b1 is not None:
            doors["D_trap_b2"] = DoorInfo(
                id="D_trap_b2", pos=Pos(b1, extra_trap_dk_b1), required_keys=("blue",))
        if extra_trap_dk_b2 is not None:
            doors["D_trap_y2"] = DoorInfo(
                id="D_trap_y2", pos=Pos(b2, extra_trap_dk_b2), required_keys=("yellow",))
    else:
        # Standard: SAVE matching at both stages
        keys = {
            "K_r": KeyInfo(id="K_r", pos=kr_pos, color="red"),
            "K_y": KeyInfo(id="K_y", pos=ky_pos, color="yellow"),
            "K_b": KeyInfo(id="K_b", pos=kb_pos, color="blue"),
        }
        # P2.17: second yellow key (losing 1 yellow is OK, losing blue is fatal)
        if qty_mismatch and ky2_pos is not None:
            keys["K_y2"] = KeyInfo(id="K_y2", pos=ky2_pos, color="yellow")
        doors = {
            "D_trap_y": DoorInfo(
                id="D_trap_y", pos=Pos(b1, trap_y_col), required_keys=("yellow",)),
            "D_correct_r": DoorInfo(
                id="D_correct_r", pos=Pos(b1, correct_r_col), required_keys=("red",)),
            "D_trap_b": DoorInfo(
                id="D_trap_b", pos=Pos(b2, trap_b_col), required_keys=("blue",)),
            "D_gate_y": DoorInfo(
                id="D_gate_y", pos=Pos(b2, gate_y_col), required_keys=("yellow",)),
            "D_exit_b": DoorInfo(
                id="D_exit_b", pos=exit_door_pos, required_keys=("blue",)),
        }
        if extra_trap_dk_b1 is not None:
            doors["D_trap_y2"] = DoorInfo(
                id="D_trap_y2", pos=Pos(b1, extra_trap_dk_b1), required_keys=("yellow",))
        if extra_trap_dk_b2 is not None:
            doors["D_trap_b2"] = DoorInfo(
                id="D_trap_b2", pos=Pos(b2, extra_trap_dk_b2), required_keys=("blue",))
    grid = ChainVaultGrid(
        width=width, height=height,
        walls=frozenset(walls), keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


def generate_order_dependent_lock_grid(
    width: int,
    height: int,
    rng: random.Random | None = None,
) -> Optional[tuple[ChainVaultGrid, ChainVaultState]]:
    """Generate an order-dependent lock grid: same-color fork with ordering.

    MECHANIC: 2 red keys (K_r in Zone 0, K_r2 hidden in Zone 1A) + 1 blue
    key (K_b in Zone 0).  2 red doors on barrier 1 (Door_A correct, Door_B
    trap) + 1 red+blue dual-lock exit on barrier 2.  The agent must open
    Door_A FIRST to reach K_r2, then use K_r2+K_b on the dual-lock exit.
    Opening Door_B first wastes the only red key accessible from Zone 0,
    making Door_A and the dual-lock permanently unreachable.

    LAYOUT:
        Zone 0 (rows 0..b1-1):     Start + K_r + K_b
                                     K_r directly above Door_B (bait)
        Barrier 1 (row b1):         Door_A (red, far-side) + Door_B (red, exit-side)
        Zone 1 (rows b1+1..b2-1):  Split by divider column:
          Zone 1A (far-side):       K_r2 (red key) + connects to D_exit approach
          Zone 1B (exit-side):      Real area (>=2 cells), disconnected from D_exit
        Barrier 2 (row b2):         D_exit (red+blue dual-lock, Zone 1A side)
        Zone 2 (rows b2+1..h-1):   Exit

    ANTI-GREEDY (triple bait):
    - K_r directly above Door_B (convergent forks) — USE_KEY temptation
    - Door_B on exit-side half (closer to exit column)
    - Zone 1B >= Zone 1A in size (trap area looks like "main" path)
    - manhattan(Door_B, exit) < manhattan(Door_A, exit) — hard filter

    CORRECT PATH: Start -> K_r + K_b -> Door_A (red) -> Zone 1A -> K_r2 ->
    D_exit (red+blue) -> Zone 2 -> Exit.
    Uses: 2 red + 1 blue = 3 keys.

    DEADLOCKS (>=2):
    1. Open Door_B with {1 red, 0 blue} -> Zone 1B -> back to Zone 0 ->
       0 red left -> can't open Door_A -> can't get K_r2 -> stuck
    2. Open Door_B with {1 red, 1 blue} -> Zone 1B -> back to Zone 0 ->
       0 red, 1 blue -> can't open Door_A or D_exit -> stuck
    """
    if rng is None:
        rng = random.Random()
    if width < 5 or height < 5:
        return None

    # --- Stochastic false compensation variant (P1.11) ---
    # At w>=6, 50% chance: add K_b2 (blue decoy) in trap zone.
    # Agent enters trap, finds extra blue key — but R is wasted,
    # so extra blue can't help (exit needs R+B).
    false_comp = width >= 6 and rng.random() < 0.5

    # P2.18 (content-informed same-color fork) skipped: green decoy key
    # conflicts with _all_keys_relevant filter — green is deliberately
    # irrelevant, but the pipeline rejects irrelevant key colors.
    content_informed = False

    h, w = height, width

    # --- 1. Barrier rows ---
    b1_max = h - 4
    if b1_max < 1:
        return None
    b1 = rng.randint(1, b1_max)
    b2_min, b2_max = b1 + 2, h - 2
    if b2_min > b2_max:
        return None
    b2 = rng.randint(b2_min, b2_max)

    # --- 2. Exit-side direction ---
    exit_side = rng.choice([0, w - 1])

    # --- 3. Column halves ---
    if exit_side == 0:
        exit_half = list(range(0, (w + 1) // 2))
        far_half = list(range(w // 2, w))
    else:
        exit_half = list(range(w // 2, w))
        far_half = list(range(0, (w + 1) // 2))

    if not exit_half or not far_half:
        return None

    # --- 4. Barrier 1 columns: trap on exit-side, correct on far-side ---
    trap_col = rng.choice(exit_half)
    far_half_b1 = [c for c in far_half if c != trap_col]
    if not far_half_b1:
        return None
    correct_col = rng.choice(far_half_b1)

    # --- 5. Divider column (biased toward correct_col to maximize Zone 1B) ---
    lo, hi = min(trap_col, correct_col), max(trap_col, correct_col)
    divider_options = list(range(lo + 1, hi))
    if not divider_options:
        return None
    if correct_col > trap_col:
        divider_col = divider_options[-1]
    else:
        divider_col = divider_options[0]

    # --- 6. D_exit column (on correct side of divider — Zone 1A side) ---
    if correct_col > divider_col:
        d_exit_candidates = list(range(divider_col + 1, w))
    else:
        d_exit_candidates = list(range(0, divider_col))
    if not d_exit_candidates:
        return None
    d_exit_col = rng.choice(d_exit_candidates)

    # --- 7. Build walls ---
    walls: set[Pos] = set()

    # Barrier 1: walls except trap_col and correct_col
    for c in range(w):
        if c != trap_col and c != correct_col:
            walls.add(Pos(b1, c))

    # Divider walls through all Zone 1 rows
    for r in range(b1 + 1, b2):
        walls.add(Pos(r, divider_col))

    # Barrier 2: walls except d_exit_col
    for c in range(w):
        if c != d_exit_col:
            walls.add(Pos(b2, c))

    # --- Scaling: extra trap(s) at w>=6 / w>=7 ---
    extra_trap_odl_b1: Optional[int] = None
    extra_trap_odl_b2: Optional[int] = None
    if w >= 6:
        extra_trap_odl_b1 = _pick_extra_trap_col(
            b1, walls, {trap_col, correct_col}, exit_half, rng)
        if extra_trap_odl_b1 is not None:
            walls.discard(Pos(b1, extra_trap_odl_b1))
    if w >= 7:
        extra_trap_odl_b2 = _pick_extra_trap_col(
            b2, walls, {d_exit_col}, exit_half, rng)
        if extra_trap_odl_b2 is not None:
            walls.discard(Pos(b2, extra_trap_odl_b2))

    # --- 8. Exit placement (biased toward trap_col to strengthen manhattan bait) ---
    zone2 = {
        Pos(r, c) for r in range(b2 + 1, h) for c in range(w)
        if Pos(r, c) not in walls
    }
    if not zone2:
        return None
    exit_candidates = sorted(zone2, key=lambda p: abs(p.col - trap_col))
    n_pick = max(1, len(exit_candidates) // 2)
    exit_pos = rng.choice(exit_candidates[:n_pick])

    # --- 9. Anti-greedy manhattan check ---
    trap_pos = Pos(b1, trap_col)
    correct_pos = Pos(b1, correct_col)
    if not _trap_closer_to_exit(trap_pos, correct_pos, exit_pos):
        return None

    # --- 10. Key placement ---
    zone0 = {
        Pos(r, c) for r in range(b1) for c in range(w)
        if Pos(r, c) not in walls
    }
    if len(zone0) < 3:
        return None

    # K_r: near trap door (convergent forks bait)
    kr_pos = _convergent_key_pos(Pos(b1, trap_col), zone0, rng)
    if kr_pos is None:
        return None

    # K_b: biased toward trap_col (double convergent forks —
    # both keys cluster near trap, so collecting them converges on trap door)
    kb_candidates = [p for p in zone0 if p != kr_pos]
    if not kb_candidates:
        return None
    kb_candidates.sort(key=lambda p: abs(p.col - trap_col))
    biased_pool = kb_candidates[:max(1, len(kb_candidates) // 3)]
    kb_result = _select_key_positions(
        biased_pool, 1, rng,
        already_placed=frozenset({kr_pos}), reference=exit_pos,
    )
    if kb_result is None:
        kb_result = _select_key_positions(
            kb_candidates, 1, rng,
            already_placed=frozenset({kr_pos}), reference=exit_pos,
        )
    if kb_result is None:
        return None
    kb_pos = kb_result[0]

    # K_r2: in Zone 1A (far from divider, ideally near D_exit approach)
    zone1 = {
        Pos(r, c) for r in range(b1 + 1, b2) for c in range(w)
        if Pos(r, c) not in walls
    }

    # Identify Zone 1A cells (correct side of divider)
    correct_entry = Pos(b1 + 1, correct_col)
    if correct_entry not in zone1:
        return None
    zone1a_comp = _flood_fill(correct_entry, zone1)

    d_exit_approach = Pos(b2 - 1, d_exit_col)
    if d_exit_approach not in zone1a_comp:
        return None

    # Place K_r2 in Zone 1A, biased toward d_exit_approach for route coherence
    kr2_candidates = [p for p in zone1a_comp
                      if p != correct_entry and p != d_exit_approach]
    if not kr2_candidates:
        # Fall back: place at correct_entry if that's the only option
        kr2_candidates = [p for p in zone1a_comp if p != d_exit_approach]
    if not kr2_candidates:
        return None
    kr2_candidates.sort(
        key=lambda p: manhattan_distance(p, d_exit_approach))
    n_pick = max(1, len(kr2_candidates) // 2)
    kr2_pos = rng.choice(kr2_candidates[:n_pick])

    # --- 11. Start: biased toward far side (opposite exit) ---
    far_side = w - 1 - exit_side
    start_candidates = [p for p in zone0 if p != kr_pos and p != kb_pos]
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda p: abs(p.col - far_side))
    n_pick = max(1, len(start_candidates) // 2)
    start = rng.choice(start_candidates[:n_pick])

    # --- 12. Connectivity checks ---
    trap_approach = Pos(b1 - 1, trap_col)
    correct_approach = Pos(b1 - 1, correct_col)
    if trap_approach not in zone0 or correct_approach not in zone0:
        return None
    zone0_comp = _flood_fill(start, zone0)
    if kr_pos not in zone0_comp or kb_pos not in zone0_comp:
        return None
    if trap_approach not in zone0_comp or correct_approach not in zone0_comp:
        return None

    # Zone 1A: correct_entry -> kr2 -> d_exit_approach connected
    if kr2_pos not in zone1a_comp:
        return None

    # Zone 1B: trap_entry accessible, disconnected from Zone 1A
    trap_entry = Pos(b1 + 1, trap_col)
    if trap_entry not in zone1:
        return None
    zone1b_comp = _flood_fill(trap_entry, zone1)
    if correct_entry in zone1b_comp:
        return None  # divider failed — zones not separated
    if len(zone1b_comp) < len(zone1a_comp):
        return None  # trap area should look like main path

    # Zone 1B must have >= 2 cells (real walkable area)
    if len(zone1b_comp) < 2:
        return None

    # False compensation: place K_b2 (decoy blue) in Zone 1B
    kb2_pos: Optional[Pos] = None
    if false_comp:
        zone1b_candidates = [p for p in zone1b_comp if p != trap_entry]
        if zone1b_candidates:
            rng.shuffle(zone1b_candidates)
            kb2_pos = zone1b_candidates[0]
        else:
            false_comp = False  # can't place, fall back to standard

    # P2.18: Content-informed — green key in Zone 1B + green door on barrier 2
    kg_pos: Optional[Pos] = None
    green_trap_col: Optional[int] = None
    if content_informed:
        # K_g in Zone 1B
        kg_cands = [p for p in zone1b_comp if p != trap_entry]
        if kg_cands:
            kg_pos = rng.choice(kg_cands)
        else:
            content_informed = False
    if content_informed:
        # Green door on barrier 2 accessible from Zone 1B side
        # Zone 1B columns
        if correct_col > divider_col:
            zone1b_cols = list(range(0, divider_col))
        else:
            zone1b_cols = list(range(divider_col + 1, w))
        gt_cands = [c for c in zone1b_cols
                    if Pos(b2, c) in walls and Pos(b2 - 1, c) in zone1b_comp]
        if gt_cands:
            green_trap_col = rng.choice(gt_cands)
            walls.discard(Pos(b2, green_trap_col))
        else:
            content_informed = False
            kg_pos = None

    # Zone 2: d_exit_entry -> exit connected
    d_exit_entry = Pos(b2 + 1, d_exit_col)
    if d_exit_entry not in zone2:
        return None
    zone2_comp = _flood_fill(d_exit_entry, zone2)
    if exit_pos not in zone2_comp:
        return None

    # P2.18: verify green trap entry reaches Zone 2 and exit
    if content_informed and green_trap_col is not None:
        green_entry = Pos(b2 + 1, green_trap_col)
        # Recompute zone2 since we opened a wall on barrier 2
        zone2 = {
            Pos(r, c) for r in range(b2 + 1, h) for c in range(w)
            if Pos(r, c) not in walls
        }
        if green_entry not in zone2:
            content_informed = False
            kg_pos = None
            green_trap_col = None
        else:
            zone2_comp = _flood_fill(d_exit_entry, zone2)
            if exit_pos not in zone2_comp or green_entry not in zone2_comp:
                content_informed = False
                kg_pos = None
                green_trap_col = None

    # --- 13. Extra walls (0-2 per zone, connectivity-preserving) ---
    # Zone 0 extras
    zone0_protected = {start, kr_pos, kb_pos, trap_approach, correct_approach}
    zone0_extras = [p for p in zone0 if p not in zone0_protected]
    rng.shuffle(zone0_extras)
    n_extra = rng.randint(0, min(2, len(zone0_extras) // 4))
    for pos in zone0_extras:
        if n_extra <= 0:
            break
        rem = zone0 - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone0_protected):
            walls.add(pos)
            zone0.discard(pos)
            n_extra -= 1

    # Zone 1A extras (Zone 1B left untouched — it's the trap area)
    zone1a = set(zone1a_comp)
    zone1a_protected = {correct_entry, d_exit_approach, kr2_pos}
    zone1a_extras = [p for p in zone1a if p not in zone1a_protected]
    rng.shuffle(zone1a_extras)
    n_extra = rng.randint(0, min(2, len(zone1a_extras) // 4))
    for pos in zone1a_extras:
        if n_extra <= 0:
            break
        rem = zone1a - {pos}
        if rem and _is_connected(rem) and all(p in rem for p in zone1a_protected):
            walls.add(pos)
            zone1a.discard(pos)
            n_extra -= 1

    # --- 14. Build grid ---
    keys = {
        "K_r": KeyInfo(id="K_r", pos=kr_pos, color="red"),
        "K_b": KeyInfo(id="K_b", pos=kb_pos, color="blue"),
        "K_r2": KeyInfo(id="K_r2", pos=kr2_pos, color="red"),
    }
    if false_comp and kb2_pos is not None:
        keys["K_b2"] = KeyInfo(id="K_b2", pos=kb2_pos, color="blue")
    if content_informed and kg_pos is not None:
        keys["K_g"] = KeyInfo(id="K_g", pos=kg_pos, color="green")
    doors = {
        "D_correct_r": DoorInfo(
            id="D_correct_r", pos=correct_pos, required_keys=("red",)),
        "D_trap_r": DoorInfo(
            id="D_trap_r", pos=trap_pos, required_keys=("red",)),
        "D_exit": DoorInfo(
            id="D_exit", pos=Pos(b2, d_exit_col),
            required_keys=("red", "blue")),
    }
    if extra_trap_odl_b1 is not None:
        doors["D_trap_r2"] = DoorInfo(
            id="D_trap_r2", pos=Pos(b1, extra_trap_odl_b1), required_keys=("red",))
    if extra_trap_odl_b2 is not None:
        doors["D_trap_b2"] = DoorInfo(
            id="D_trap_b2", pos=Pos(b2, extra_trap_odl_b2), required_keys=("red",))
    if content_informed and green_trap_col is not None:
        doors["D_trap_g"] = DoorInfo(
            id="D_trap_g", pos=Pos(b2, green_trap_col),
            required_keys=("green",))
    grid = ChainVaultGrid(
        width=width, height=height,
        walls=frozenset(walls), keys=keys, doors=doors,
        start=start, exit_pos=exit_pos,
    )
    initial_state = ChainVaultState(
        agent_pos=start,
        inventory=(),
        opened_doors=frozenset(),
        remaining_keys=frozenset(keys.keys()),
    )
    return grid, initial_state


# ---------------------------------------------------------------------------
# Deadlock discovery (forward sweep)
# ---------------------------------------------------------------------------

def discover_deadlocks(
    grid: ChainVaultGrid,
    initial_state: ChainVaultState,
    max_depth: int = 15,
    max_states: int = 50_000,
) -> list[DeadlockEntry]:
    """Forward sweep from initial state to discover Walking Dead states.

    Chain Vault optimization: only check is_solvable after actions that open
    a door and consume keys. Plain movement + key pickup cannot cause deadlock.

    Monotonicity: once unsolvable, always unsolvable (keys only get consumed,
    doors only get opened — neither is reversible).
    """
    catalog: list[DeadlockEntry] = []
    queue: deque[tuple[ChainVaultState, list[ChainVaultAction], bool]] = deque()
    queue.append((initial_state, [], True))
    visited: set = {initial_state.to_hashable()}
    states_explored = 0

    while queue and states_explored < max_states:
        state, path, parent_solvable = queue.popleft()

        if len(path) >= max_depth:
            continue

        for action in get_legal_actions(grid, state):
            opens_door = action_opens_door(grid, state, action)
            next_state = apply_action(grid, state, action)
            h = next_state.to_hashable()
            if h in visited:
                continue
            visited.add(h)
            states_explored += 1
            new_path = path + [action]

            # Determine solvability
            if not parent_solvable:
                # Monotonicity: once dead, always dead
                next_solvable = False
            elif opens_door:
                # Only key-consuming door opens can cause deadlock
                next_solvable = is_solvable(grid, next_state)
            else:
                # Movement/pickup can't cause deadlock
                next_solvable = True

            # Detect solvability flip
            if parent_solvable and not next_solvable:
                if is_walking_dead(grid, next_state):
                    catalog.append(DeadlockEntry(
                        path=new_path,
                        fatal_step=len(new_path) - 1,
                        deadlock_state=next_state,
                        pre_deadlock_state=state,
                    ))

            # Continue from solvable states or Walking Dead states
            if next_solvable or (not next_solvable and
                                  len(get_legal_actions(grid, next_state)) > 0):
                queue.append((next_state, new_path, next_solvable))

    return catalog


# ---------------------------------------------------------------------------
# Oracle-adjacent deadlock discovery (hard variants)
# ---------------------------------------------------------------------------

def discover_oracle_adjacent_deadlocks(
    grid: ChainVaultGrid,
    initial_state: ChainVaultState,
    solution: list[ChainVaultAction],
) -> list[DeadlockEntry]:
    """Find deadlocks one step off the oracle solution.

    Walk the oracle solution step-by-step.  At each step *i*, try every
    legal door-opening move that differs from the oracle move. If any leads
    to a Walking Dead state, record it as an oracle-adjacent deadlock whose
    path is ``solution[:i] + [alt_action]``.

    Chain Vault optimization: only key-consuming door-opening alternatives are
    tested because movement + key pickup can never flip solvability.

    Returns a list of DeadlockEntry with ``k_horizon`` already computed.
    """
    catalog: list[DeadlockEntry] = []
    state = initial_state

    for i, oracle_action in enumerate(solution):
        legal = get_legal_actions(grid, state)
        for alt in legal:
            if alt == oracle_action:
                continue
            if not action_opens_door(grid, state, alt):
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
                ))
        # Advance along the oracle path
        state = apply_action(grid, state, oracle_action)

    return catalog


# ---------------------------------------------------------------------------
# Multiple greedy strategies
# ---------------------------------------------------------------------------

def _move_toward(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    target: Pos,
    legal: list[ChainVaultAction],
) -> Optional[ChainVaultAction]:
    """Pick the legal MOVE action that minimizes manhattan distance to target."""
    moves = list(legal)
    if not moves:
        return None
    return min(
        moves,
        key=lambda a: manhattan_distance(add_pos(state.agent_pos, a.direction), target),
    )


def _door_opening_actions(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    legal: list[ChainVaultAction],
) -> list[ChainVaultAction]:
    """Return legal actions that open an adjacent locked door."""
    return [action for action in legal if action_opens_door(grid, state, action)]


def _count_door_opening_actions(
    grid: ChainVaultGrid,
    start_state: ChainVaultState,
    actions: list[ChainVaultAction],
) -> int:
    """Count how many actions in a trace open a door and consume keys."""
    count = 0
    state = start_state
    for action in actions:
        if action_opens_door(grid, state, action):
            count += 1
        state = apply_action(grid, state, action)
    return count


def _nearest_matching_door(
    grid: ChainVaultGrid,
    state: ChainVaultState,
) -> Optional[DoorInfo]:
    """Find the nearest locked door the agent has keys to open."""
    inv = state.inventory
    best_door = None
    best_dist = float("inf")
    for door in grid.doors.values():
        if not state.is_door_open(door.id) and _has_keys(inv, door.required_keys):
            d = manhattan_distance(state.agent_pos, door.pos)
            if d < best_dist:
                best_dist = d
                best_door = door
    return best_door


def simulate_greedy_toward_exit(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_steps: int = 40,
) -> tuple[list[ChainVaultAction], ChainVaultState]:
    """Strategy 1: minimize manhattan to exit, open any adjacent door first.

    Priority: 1) open adjacent door if legal  2) MOVE minimizing manhattan(agent, exit)
    """
    trace: list[ChainVaultAction] = []
    visited_states: set = set()

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break  # loop detected
        visited_states.add(h)

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        door_opens = _door_opening_actions(grid, state, legal)
        if door_opens:
            trace.append(door_opens[0])
            state = apply_action(grid, state, door_opens[0])
            continue

        action = _move_toward(grid, state, grid.exit_pos, legal)
        if action is None:
            break
        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


def simulate_greedy_nearest_key_first(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_steps: int = 40,
) -> tuple[list[ChainVaultAction], ChainVaultState]:
    """Strategy 2: go to nearest key first, then toward exit/doors.

    Priority: 1) open adjacent door if legal  2) MOVE toward nearest unpicked key
              3) MOVE toward exit
    """
    trace: list[ChainVaultAction] = []
    visited_states: set = set()

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break
        visited_states.add(h)

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        door_opens = _door_opening_actions(grid, state, legal)
        if door_opens:
            trace.append(door_opens[0])
            state = apply_action(grid, state, door_opens[0])
            continue

        # Go to nearest key if any remain on ground
        remaining = [grid.keys[kid].pos for kid in state.remaining_keys]
        if remaining:
            target = min(remaining, key=lambda p: manhattan_distance(state.agent_pos, p))
        else:
            target = grid.exit_pos

        action = _move_toward(grid, state, target, legal)
        if action is None:
            break
        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


def simulate_greedy_nearest_door(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_steps: int = 40,
) -> tuple[list[ChainVaultAction], ChainVaultState]:
    """Strategy 3: when holding a key, move toward nearest matching door.

    Priority: 1) open adjacent door if legal  2) MOVE toward nearest matching locked door
              3) MOVE toward nearest key  4) MOVE toward exit
    """
    trace: list[ChainVaultAction] = []
    visited_states: set = set()

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break
        visited_states.add(h)

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        door_opens = _door_opening_actions(grid, state, legal)
        if door_opens:
            trace.append(door_opens[0])
            state = apply_action(grid, state, door_opens[0])
            continue

        # Move toward nearest matching door
        nearest_door = _nearest_matching_door(grid, state)
        if nearest_door is not None:
            target = nearest_door.pos
        else:
            # No matching door — go pick up keys
            remaining = [grid.keys[kid].pos for kid in state.remaining_keys]
            if remaining:
                target = min(remaining, key=lambda p: manhattan_distance(state.agent_pos, p))
            else:
                target = grid.exit_pos

        action = _move_toward(grid, state, target, legal)
        if action is None:
            break
        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


def simulate_greedy_use_key_immediately(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_steps: int = 40,
) -> tuple[list[ChainVaultAction], ChainVaultState]:
    """Strategy 4: open a door whenever possible, else move toward nearest key/exit.

    This strategy aggressively spends keys the moment it can, never saving them.
    """
    trace: list[ChainVaultAction] = []
    visited_states: set = set()

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break
        visited_states.add(h)

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        # Aggressively open doors
        door_opens = _door_opening_actions(grid, state, legal)
        if door_opens:
            trace.append(door_opens[0])
            state = apply_action(grid, state, door_opens[0])
            continue

        # Move toward nearest key or exit
        remaining = [grid.keys[kid].pos for kid in state.remaining_keys]
        if remaining:
            target = min(remaining, key=lambda p: manhattan_distance(state.agent_pos, p))
        else:
            target = grid.exit_pos

        action = _move_toward(grid, state, target, legal)
        if action is None:
            break
        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


def simulate_bfs_toward_exit(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_steps: int = 60,
) -> tuple[list[ChainVaultAction], ChainVaultState]:
    """Strategy 5: BFS-navigate toward exit; open adjacent doors first.

    Like Strategy 1 but uses BFS path length instead of manhattan distance,
    so walls and locked doors are treated as true obstacles rather than
    transparent heuristic targets.

    Priority: 1) open adjacent door if legal  2) BFS move toward exit
              3) BFS move toward nearest remaining key (if exit unreachable)
    """
    trace: list[ChainVaultAction] = []
    visited_states: set = set()

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break
        visited_states.add(h)

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        door_opens = _door_opening_actions(grid, state, legal)
        if door_opens:
            trace.append(door_opens[0])
            state = apply_action(grid, state, door_opens[0])
            continue

        # BFS toward exit
        first_dir, _ = _bfs_to(grid, state, grid.exit_pos)

        if first_dir is None:
            # Exit unreachable — BFS toward nearest remaining key
            best_dir: Optional[Direction] = None
            best_dist: float = float("inf")
            for kid in state.remaining_keys:
                fd, dist = _bfs_to(grid, state, grid.keys[kid].pos)
                if fd is not None and dist < best_dist:
                    best_dist = dist
                    best_dir = fd
            first_dir = best_dir

        if first_dir is None:
            break

        move = next((a for a in legal if a.direction == first_dir), None)
        if move is None:
            break
        trace.append(move)
        state = apply_action(grid, state, move)

    return trace, state


def simulate_bfs_nearest_door(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_steps: int = 60,
) -> tuple[list[ChainVaultAction], ChainVaultState]:
    """Strategy 6: BFS-navigate toward nearest matching door.

    Like Strategy 3 but ranks doors by BFS path length instead of manhattan,
    correctly accounting for walls and locked-door obstacles.

    Priority: 1) open adjacent door if legal  2) BFS toward nearest matching locked door
              3) BFS toward nearest remaining key  4) BFS toward exit
    """
    trace: list[ChainVaultAction] = []
    visited_states: set = set()

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break
        visited_states.add(h)

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        door_opens = _door_opening_actions(grid, state, legal)
        if door_opens:
            trace.append(door_opens[0])
            state = apply_action(grid, state, door_opens[0])
            continue

        # BFS toward nearest matching door (ranked by path length)
        inv = state.inventory
        best_dir: Optional[Direction] = None
        best_dist: float = float("inf")
        for door in grid.doors.values():
            if state.is_door_open(door.id) or not _has_keys(inv, door.required_keys):
                continue
            fd, dist = _bfs_to(grid, state, door.pos)
            if fd is not None and dist < best_dist:
                best_dist = dist
                best_dir = fd

        if best_dir is None:
            # No matching door reachable — BFS toward nearest remaining key
            for kid in state.remaining_keys:
                fd, dist = _bfs_to(grid, state, grid.keys[kid].pos)
                if fd is not None and dist < best_dist:
                    best_dist = dist
                    best_dir = fd

        if best_dir is None:
            # No keys either — BFS toward exit
            fd, _ = _bfs_to(grid, state, grid.exit_pos)
            best_dir = fd

        if best_dir is None:
            break

        move = next((a for a in legal if a.direction == best_dir), None)
        if move is None:
            break
        trace.append(move)
        state = apply_action(grid, state, move)

    return trace, state


def simulate_save_for_multilock(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_steps: int = 60,
) -> tuple[list[ChainVaultAction], ChainVaultState]:
    """Strategy 7: Budget-aware key conservation.

    Heuristic: "If an unopened multi-lock door needs this color, don't waste
    it on a single-lock door."

    Door-opening decision:
      - Multi-lock door (2+ keys): always open.
      - Single-lock door (1 key): open only if NO unopened multi-lock door
        needs the same color.
    Movement: BFS toward exit, fallback to nearest key.
    """

    # Pre-compute colors needed by multi-lock doors (updated as doors open)
    def _multilock_colors(st: ChainVaultState) -> set[str]:
        colors: set[str] = set()
        for door in grid.doors.values():
            if not st.is_door_open(door.id) and len(door.required_keys) >= 2:
                colors.update(door.required_keys)
        return colors

    trace: list[ChainVaultAction] = []
    visited_states: set = set()

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break
        visited_states.add(h)

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        # Conditional door opening
        door_opens = _door_opening_actions(grid, state, legal)
        used = False
        if door_opens:
            ml_colors = _multilock_colors(state)
            for move in door_opens:
                door = get_door_opened_by_action(grid, state, move)
                if door is None:
                    continue
                if len(door.required_keys) >= 2:
                    # Multi-lock: always open
                    trace.append(move)
                    state = apply_action(grid, state, move)
                    used = True
                    break
                else:
                    # Single-lock: open only if no multi-lock door needs the color
                    door_color = door.required_keys[0]
                    if door_color not in ml_colors:
                        trace.append(move)
                        state = apply_action(grid, state, move)
                        used = True
                        break
        if used:
            continue

        # Movement: BFS toward exit, fallback to nearest key
        first_dir, _ = _bfs_to(grid, state, grid.exit_pos)

        if first_dir is None:
            best_dir: Optional[Direction] = None
            best_dist: float = float("inf")
            for kid in state.remaining_keys:
                fd, dist = _bfs_to(grid, state, grid.keys[kid].pos)
                if fd is not None and dist < best_dist:
                    best_dist = dist
                    best_dir = fd
            first_dir = best_dir

        if first_dir is None:
            break

        move = next((a for a in legal if a.direction == first_dir), None)
        if move is None:
            break
        trace.append(move)
        state = apply_action(grid, state, move)

    return trace, state


def simulate_exit_seekr(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_steps: int = 60,
) -> tuple[list[ChainVaultAction], ChainVaultState]:
    """Strategy 8: Exit-reachability check (one-step lookahead).

    Heuristic: "Only open a door if the exit becomes reachable after opening it."

    Door-opening decision:
      - Hypothetically open a door, then BFS to exit from the new state.
      - If exit is reachable: open the door.
      - If exit is NOT reachable: skip.
    Movement: BFS toward exit -> nearest key -> nearest matching door.
    """

    trace: list[ChainVaultAction] = []
    visited_states: set = set()

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break
        visited_states.add(h)

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        # Conditional door opening with lookahead
        door_opens = _door_opening_actions(grid, state, legal)
        used = False
        if door_opens:
            for move in door_opens:
                hypo_state = apply_action(grid, state, move)
                _, dist = _bfs_to(grid, hypo_state, grid.exit_pos)
                if dist is not None:
                    # Exit reachable after opening — do it
                    trace.append(move)
                    state = hypo_state
                    used = True
                    break
        if used:
            continue

        # Movement: BFS toward exit
        first_dir, _ = _bfs_to(grid, state, grid.exit_pos)

        if first_dir is None:
            # Exit unreachable — BFS toward nearest remaining key
            best_dir: Optional[Direction] = None
            best_dist: float = float("inf")
            for kid in state.remaining_keys:
                fd, dist = _bfs_to(grid, state, grid.keys[kid].pos)
                if fd is not None and dist < best_dist:
                    best_dist = dist
                    best_dir = fd
            first_dir = best_dir

        if first_dir is None:
            # No keys either — BFS toward nearest matching door
            inv = state.inventory
            best_dir2: Optional[Direction] = None
            best_dist2: float = float("inf")
            for door in grid.doors.values():
                if state.is_door_open(door.id) or not _has_keys(inv, door.required_keys):
                    continue
                fd, dist = _bfs_to(grid, state, door.pos)
                if fd is not None and dist < best_dist2:
                    best_dist2 = dist
                    best_dir2 = fd
            first_dir = best_dir2

        if first_dir is None:
            break

        move = next((a for a in legal if a.direction == first_dir), None)
        if move is None:
            break
        trace.append(move)
        state = apply_action(grid, state, move)

    return trace, state


def simulate_prefer_multilock(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_steps: int = 60,
) -> tuple[list[ChainVaultAction], ChainVaultState]:
    """Strategy 9: Cost-aware door selection — prefer multi-lock doors.

    Heuristic: "Single-lock doors are probably traps. Focus on multi-lock doors."

    Door-opening decision:
      - Multi-lock door (2+ keys): always open.
      - Single-lock door (1 key): open only if NO multi-lock door is still unopened.
    Movement: BFS toward nearest unopened multi-lock door -> exit -> nearest key.
    """

    trace: list[ChainVaultAction] = []
    visited_states: set = set()

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break
        visited_states.add(h)

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        # Check if any multi-lock door is still unopened
        any_multilock_remaining = any(
            len(d.required_keys) >= 2
            for d in grid.doors.values()
            if not state.is_door_open(d.id)
        )

        # Conditional door opening
        door_opens = _door_opening_actions(grid, state, legal)
        used = False
        if door_opens:
            for move in door_opens:
                door = get_door_opened_by_action(grid, state, move)
                if door is None:
                    continue
                if len(door.required_keys) >= 2:
                    # Multi-lock: always open
                    trace.append(move)
                    state = apply_action(grid, state, move)
                    used = True
                    break
                elif not any_multilock_remaining:
                    # No multi-lock doors left: open single-lock normally
                    trace.append(move)
                    state = apply_action(grid, state, move)
                    used = True
                    break
        if used:
            continue

        # Movement: BFS toward nearest unopened multi-lock door
        first_dir: Optional[Direction] = None
        best_dist: float = float("inf")
        if any_multilock_remaining:
            for door in grid.doors.values():
                if state.is_door_open(door.id) or len(door.required_keys) < 2:
                    continue
                fd, dist = _bfs_to(grid, state, door.pos)
                if fd is not None and dist < best_dist:
                    best_dist = dist
                    first_dir = fd

        if first_dir is None:
            # Fallback: BFS toward exit
            fd, _ = _bfs_to(grid, state, grid.exit_pos)
            first_dir = fd

        if first_dir is None:
            # Fallback: BFS toward nearest remaining key
            best_dist = float("inf")
            for kid in state.remaining_keys:
                fd, dist = _bfs_to(grid, state, grid.keys[kid].pos)
                if fd is not None and dist < best_dist:
                    best_dist = dist
                    first_dir = fd

        if first_dir is None:
            break

        move = next((a for a in legal if a.direction == first_dir), None)
        if move is None:
            break
        trace.append(move)
        state = apply_action(grid, state, move)

    return trace, state


# All greedy strategies that must fail for a puzzle to be non-greedy
GREEDY_STRATEGIES = [
    # Existing (cheap first)
    ("toward_exit", simulate_greedy_toward_exit),
    ("nearest_key_first", simulate_greedy_nearest_key_first),
    ("nearest_door", simulate_greedy_nearest_door),
    ("use_key_immediately", simulate_greedy_use_key_immediately),
    ("bfs_toward_exit", simulate_bfs_toward_exit),
    ("bfs_nearest_door", simulate_bfs_nearest_door),
    # Smart strategies (most expensive last)
    ("save_for_multilock", simulate_save_for_multilock),
    ("exit_seekr", simulate_exit_seekr),
    ("prefer_multilock", simulate_prefer_multilock),
]


def all_greedy_strategies_fail(
    grid: ChainVaultGrid,
    initial_state: ChainVaultState,
) -> bool:
    """Check that NO greedy strategy solves the puzzle.

    Returns True only if ALL strategies fail (don't reach exit).
    """
    for name, strategy_fn in GREEDY_STRATEGIES:
        _trace, final_state = strategy_fn(grid, initial_state)
        if is_solved(grid, final_state):
            return False
    return True


def _all_keys_relevant(
    grid: ChainVaultGrid,
    initial_state: ChainVaultState,
    solution: list[ChainVaultAction],
    catalog: list[DeadlockEntry],
) -> bool:
    """Every key's color must be either used in the oracle solution or wasted in a deadlock.

    A key color is 'relevant' if:
    - A key of that color is consumed by a door-opening move in the oracle
      solution, OR
    - A key of that color is consumed by the fatal door-opening move in at
      least one catalog entry

    Keys whose color satisfies neither condition are noise — the model can ignore them
    without affecting the reasoning challenge. Reject such puzzles.

    Note: this is a color-level check because chain vault's engine tracks inventory by
    color count. Two red keys are interchangeable; if red is needed anywhere, both are
    covered. This correctly handles templates like Switchback where 2 same-color keys
    are both required.
    """
    # Colors consumed by door-opening actions in the oracle solution
    solution_colors: set[str] = set()
    state = initial_state
    for action in solution:
        door = get_door_opened_by_action(grid, state, action)
        if door is not None:
            solution_colors.update(door.required_keys)
        state = apply_action(grid, state, action)

    # Colors consumed by the fatal door-opening action in each catalog entry
    deadlock_colors: set[str] = set()
    for entry in catalog:
        fatal = entry.path[entry.fatal_step]
        door = get_door_opened_by_action(grid, entry.pre_deadlock_state, fatal)
        if door is not None:
            deadlock_colors.update(door.required_keys)

    relevant = solution_colors | deadlock_colors
    return all(kinfo.color in relevant for kinfo in grid.keys.values())


# ---------------------------------------------------------------------------
# RQ instance construction helpers
# ---------------------------------------------------------------------------

def generate_rq2_pair(
    grid: ChainVaultGrid,
    initial_state: ChainVaultState,
    catalog_entry: DeadlockEntry,
    k: int,
) -> Optional[tuple[ChainVaultState, list[ChainVaultAction], list[ChainVaultAction]]]:
    """Generate a matched deadlock/solvable pair for RQ2.

    Returns (start_state, deadlock_actions, solvable_actions) or None.
    """
    path = catalog_entry.path
    T = catalog_entry.fatal_step

    window_start = T - k + 1
    if window_start < 0:
        return None

    # Compute start state
    s_start = apply_actions(grid, initial_state, path[:window_start])
    deadlock_actions = path[window_start: T + 1]

    # Verify
    if not is_solvable(grid, s_start):
        return None
    s_dead = apply_actions(grid, s_start, deadlock_actions)
    if not is_walking_dead(grid, s_dead):
        return None

    # Find matched solvable control
    control = oracle_solve(grid, s_start)
    if control and len(control) >= k:
        solvable_actions = control[:k]
    else:
        return None

    # Verify control stays solvable
    s_control = apply_actions(grid, s_start, solvable_actions)
    if not is_solvable(grid, s_control):
        return None

    return s_start, deadlock_actions, solvable_actions


def _opposite_direction(direction: Direction) -> Direction:
    return {
        Direction.UP: Direction.DOWN,
        Direction.DOWN: Direction.UP,
        Direction.LEFT: Direction.RIGHT,
        Direction.RIGHT: Direction.LEFT,
    }[direction]


def _pick_deadlock_rollout_action(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    legal: list[ChainVaultAction],
    visited: set[tuple],
    prev_action: Optional[ChainVaultAction],
    rng: random.Random,
) -> tuple[ChainVaultAction, ChainVaultState]:
    """Choose a plausible Walking Dead continuation action.

    The continuation should look purposeful rather than random. We bias toward:
    1. door-opening actions (they create better RQ3 distractors),
    2. unvisited states,
    3. successor states with additional door-opening opportunities,
    while discouraging immediate movement reversals.
    """
    scored: list[tuple[float, ChainVaultAction, ChainVaultState]] = []
    for action in legal:
        next_state = apply_action(grid, state, action)
        next_hash = next_state.to_hashable()
        next_legal = get_legal_actions(grid, next_state)
        future_opens = sum(
            action_opens_door(grid, next_state, next_action)
            for next_action in next_legal
        )

        score = 0.0
        if action_opens_door(grid, state, action):
            score += 8.0
        if next_hash not in visited:
            score += 3.0
        score += 2.0 * future_opens
        score += 0.1 * len(next_legal)

        if prev_action is not None and prev_action.is_move and action.is_move:
            if action.direction == _opposite_direction(prev_action.direction):
                score -= 2.5
            elif action.direction == prev_action.direction:
                score += 0.3

        # Keep the search deterministic but allow some exploration among ties.
        score += rng.random() * 0.5
        scored.append((score, action, next_state))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: min(3, len(scored))]
    floor = top[-1][0]
    weights = [item[0] - floor + 1.0 for item in top]
    chosen_idx = rng.choices(range(len(top)), weights=weights, k=1)[0]
    _score, chosen_action, chosen_state = top[chosen_idx]
    return chosen_action, chosen_state


def _search_walking_dead_continuation(
    grid: ChainVaultGrid,
    start_state: ChainVaultState,
    min_steps: int,
    max_steps: int,
    seed: int,
    trials_per_length: int = 12,
) -> list[ChainVaultAction]:
    """Search for a non-trivial continuation after entering a Walking Dead state."""
    best_actions: list[ChainVaultAction] = []
    best_quality: Optional[tuple[int, int, int]] = None

    for length_offset, target_steps in enumerate(range(max_steps, min_steps - 1, -1)):
        for trial in range(trials_per_length):
            rng = random.Random(seed + 7919 * length_offset + 104729 * trial)
            state = start_state
            visited = {start_state.to_hashable()}
            actions: list[ChainVaultAction] = []
            prev_action: Optional[ChainVaultAction] = None

            for _ in range(target_steps):
                legal = get_legal_actions(grid, state)
                if not legal:
                    break

                action, state = _pick_deadlock_rollout_action(
                    grid, state, legal, visited, prev_action, rng,
                )
                actions.append(action)
                visited.add(state.to_hashable())
                prev_action = action

            if len(actions) < min_steps:
                continue

            quality = (
                _count_door_opening_actions(grid, start_state, actions),
                len(visited),
                len(actions),
            )
            if best_quality is None or quality > best_quality:
                best_quality = quality
                best_actions = actions

    return best_actions


def _generate_walking_dead_trace(
    grid: ChainVaultGrid,
    prefix_actions: list[ChainVaultAction],
    dead_state: ChainVaultState,
    fatal_step: int,
    k_horizon: int,
    target_length: int,
    seed: int,
) -> Optional[tuple[list[ChainVaultAction], int, int]]:
    """Extend a fatal prefix into a higher-signal Walking Dead trace."""
    prefix_len = len(prefix_actions)
    min_total_length = max(prefix_len + 4, max(14, target_length - 2))

    bonus_steps = 4
    if k_horizon >= 120:
        bonus_steps = 6
    elif k_horizon >= 50:
        bonus_steps = 5

    max_total_length = min(24, max(min_total_length, target_length + bonus_steps))
    min_continuation = min_total_length - prefix_len
    max_continuation = max_total_length - prefix_len

    continuation = _search_walking_dead_continuation(
        grid,
        dead_state,
        min_steps=min_continuation,
        max_steps=max_continuation,
        seed=seed,
    )
    if not continuation:
        return None

    full_trace = prefix_actions + continuation
    cause_symptom_gap = len(full_trace) - fatal_step - 1
    if cause_symptom_gap < 3:
        return None

    return full_trace, fatal_step, cause_symptom_gap


def generate_rq3_instance(
    grid: ChainVaultGrid,
    initial_state: ChainVaultState,
    catalog_entry: DeadlockEntry,
    target_length: int = 18,
) -> Optional[tuple[list[ChainVaultAction], int, int]]:
    """Generate an RQ3 trace with Walking Dead continuation.

    Returns (full_trace, fatal_step, cause_symptom_gap) or None.
    """
    path = catalog_entry.path
    T = catalog_entry.fatal_step
    dead_state = catalog_entry.deadlock_state

    trace_to_deadlock = path[: T + 1]
    seed = 17_357 + 131 * T + 17 * len(path) + 7 * catalog_entry.k_horizon
    return _generate_walking_dead_trace(
        grid,
        trace_to_deadlock,
        dead_state,
        T,
        catalog_entry.k_horizon,
        target_length,
        seed,
    )


# ---------------------------------------------------------------------------
# Hard RQ instance construction (oracle-adjacent deadlocks)
# ---------------------------------------------------------------------------

def generate_hard_rq2_pair(
    grid: ChainVaultGrid,
    initial_state: ChainVaultState,
    solution: list[ChainVaultAction],
    catalog_entry: DeadlockEntry,
) -> Optional[tuple[ChainVaultState, list[ChainVaultAction], list[ChainVaultAction]]]:
    """Generate a hard matched deadlock/solvable pair for RQ2.

    Both paths share the first ``fatal_step`` actions (the oracle prefix)
    and differ only on the last move.

    Returns (initial_state, deadlock_actions, solvable_actions) or None.
    """
    i = catalog_entry.fatal_step
    bad_action = catalog_entry.path[i]  # the non-oracle action

    deadlock_actions = list(solution[:i]) + [bad_action]
    solvable_actions = list(solution[: i + 1])

    # Verify deadlock path → Walking Dead
    dead_state = apply_actions(grid, initial_state, deadlock_actions)
    if not is_walking_dead(grid, dead_state):
        return None

    # Verify solvable path → still solvable
    sol_state = apply_actions(grid, initial_state, solvable_actions)
    if not is_solvable(grid, sol_state):
        return None

    return initial_state, deadlock_actions, solvable_actions


def generate_hard_rq3_instance(
    grid: ChainVaultGrid,
    initial_state: ChainVaultState,
    solution: list[ChainVaultAction],
    catalog_entry: DeadlockEntry,
    target_length: int = 18,
) -> Optional[tuple[list[ChainVaultAction], int, int]]:
    """Generate a hard RQ3 trace from an oracle-adjacent deadlock.

    The trace follows the oracle for ``fatal_step`` actions, takes one
    wrong turn, then continues with random legal Walking Dead moves to
    reach ``target_length``.

    Returns (full_trace, fatal_step, cause_symptom_gap) or None.
    """
    i = catalog_entry.fatal_step
    bad_action = catalog_entry.path[i]

    trace_to_deadlock = list(solution[:i]) + [bad_action]
    dead_state = catalog_entry.deadlock_state
    seed = 29_911 + 257 * i + 19 * len(solution) + 11 * catalog_entry.k_horizon
    return _generate_walking_dead_trace(
        grid,
        trace_to_deadlock,
        dead_state,
        i,
        catalog_entry.k_horizon,
        target_length,
        seed,
    )


# ---------------------------------------------------------------------------
# Eval-readiness gate
# ---------------------------------------------------------------------------

def is_eval_ready(inst: GeneratedInstance, require_hard: bool = False) -> bool:
    """Return True if this instance has viable RQ2, RQ3 (and hard variants).

    Mirrors the exact dropping logic in build_eval so that instances can be
    filtered during generation before any I/O or rendering.

    Gates:
      - RQ2 standard: >= 1 entry with k_horizon>=3, fatal_step>=2 → valid pair
      - RQ3 standard: >= 1 entry with k_horizon>=5 → valid trace
      - RQ2 hard:     >= 1 oracle-adjacent entry with k_horizon>=3, fatal_step>=2
      - RQ3 hard:     >= 1 oracle-adjacent entry with k_horizon>=5
    """
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

    if inst.template_name in LOW_SIGNAL_HARD_TEMPLATES:
        return False

    # Hard variants require oracle-adjacent deadlocks
    oa_catalog = discover_oracle_adjacent_deadlocks(grid, s0, inst.solution)

    eligible_hard = [
        e for e in oa_catalog if e.k_horizon >= 3 and e.fatal_step >= 2
    ]
    has_hard_rq2 = False
    for e in eligible_hard:
        result = generate_hard_rq2_pair(grid, s0, inst.solution, e)
        if result is None:
            continue
        start_state, _deadlock_actions, solvable_actions = result
        shown_open = _count_door_opening_actions(grid, start_state, solvable_actions)
        current_state = apply_actions(grid, start_state, solvable_actions)
        legal_open = sum(
            action_opens_door(grid, current_state, action)
            for action in get_legal_actions(grid, current_state)
        )
        next_sol = oracle_solve(grid, current_state)
        best_next_opens = bool(
            next_sol and action_opens_door(grid, current_state, next_sol[0])
        )
        if shown_open >= 1 or legal_open >= 2 or best_next_opens:
            has_hard_rq2 = True
            break
    if not has_hard_rq2:
        return False

    tier2_hard = [e for e in oa_catalog if e.k_horizon >= 5]
    has_hard_rq3 = False
    for e in tier2_hard:
        result = generate_hard_rq3_instance(grid, s0, inst.solution, e)
        if result is None:
            continue
        full_trace, _fatal_step, _cause_gap = result
        if _count_door_opening_actions(grid, s0, full_trace) >= 2:
            has_hard_rq3 = True
            break
    return has_hard_rq3


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

# Maps template name → grid generator function (width, height, rng) → result
# Populated here with "fork"; new templates are added as they're implemented.
TEMPLATE_REGISTRY: dict[str, Callable] = {
    "fork": generate_fork_grid,
    "conservation_trap": generate_budget_partitioning_grid,
    "key_behind_key": generate_key_behind_key_grid,
    "convergent_forks": generate_convergent_forks_grid,
    "parallel_wings": generate_parallel_wings_grid,
    "relay": generate_relay_grid,
    "investment_door": generate_investment_door_grid,
    "switchback": generate_switchback_grid,
    "scarce_resource": generate_scarce_resource_grid,
    "cascade_trap": generate_cascade_trap_grid,
    "sequential_dual_locks": generate_sequential_dual_locks_grid,
    "key_collector": generate_key_collector_grid,
    "decoy_key": generate_decoy_key_grid,
    "order_dependent_lock": generate_order_dependent_lock_grid,
}

TEMPLATE_NAMES: list[str] = list(TEMPLATE_REGISTRY)

# Per-template solution length ranges (inclusive).
# Tighter than the old (6, 25) catch-all — keeps puzzles appropriately complex
# without padding them with trivial extra navigation.
TEMPLATE_SOLUTION_RANGES: dict[str, tuple[int, int]] = {
    "fork":                  (6,  20),
    "conservation_trap":     (10, 22),
    "key_behind_key":        (10, 22),
    "convergent_forks":      (10, 22),
    "parallel_wings":        (18, 34),
    "relay":                 (14, 28),
    "investment_door":       (12, 30),
    "switchback":            (10, 22),
    "scarce_resource":       (8,  20),
    "cascade_trap":          (10, 22),
    "sequential_dual_locks": (14, 28),
    "key_collector":         (14, 28),
    "decoy_key":             (12, 28),
    "order_dependent_lock":  (10, 22),
}

# Per-template minimum deadlock count.
# Fork only needs 1 (binary fork → exactly 1 wrong choice).
# Multi-key templates require ≥2 so both RQ2 and RQ3 variants are viable.
TEMPLATE_MIN_DEADLOCKS: dict[str, int] = {
    "fork":                  1,
    "conservation_trap":     2,
    "key_behind_key":        1,
    "convergent_forks":      2,
    "parallel_wings":        2,
    "relay":                 2,
    "investment_door":       2,
    "switchback":            2,
    "scarce_resource":       1,
    "cascade_trap":          2,
    "sequential_dual_locks": 2,
    "key_collector":         2,
    "decoy_key":             2,
    "order_dependent_lock":  2,
}

# --- Tier-based grid size system (P1: Grid Size Variation) ----------------

TEMPLATE_TIERS: dict[str, str] = {
    "fork": "simple",
    "switchback": "simple",
    "scarce_resource": "simple",
    "conservation_trap": "medium",
    "key_behind_key": "medium",
    "convergent_forks": "medium",
    "parallel_wings": "medium",
    "cascade_trap": "medium",
    "order_dependent_lock": "medium",
    "relay": "complex",
    "investment_door": "complex",
    "sequential_dual_locks": "complex",
    "key_collector": "complex",
    "decoy_key": "complex",
}

TIER_GRID_SIZES: dict[str, list[tuple[int, int]]] = {
    "simple":  [(5, 5), (6, 6)],
    "medium":  [(5, 5), (6, 6), (7, 7)],
    "complex": [(6, 6), (7, 7), (8, 8)],
}

# Per-template solution length ranges by grid width.
# Base sizes keep current TEMPLATE_SOLUTION_RANGES values.
# Larger grids: lo += 2, hi += 4 per size step.
TEMPLATE_SOLUTION_RANGES_BY_SIZE: dict[str, dict[int, tuple[int, int]]] = {
    "fork":                  {5: (6, 20),  6: (8, 24),  7: (8, 22),  8: (8, 24)},
    "switchback":            {5: (10, 22), 6: (12, 26), 7: (12, 28), 8: (14, 32)},
    "scarce_resource":       {5: (8, 20),  6: (10, 24), 7: (12, 30), 8: (14, 32)},
    "conservation_trap":     {5: (10, 22), 6: (12, 26), 7: (14, 30), 8: (12, 34)},
    "key_behind_key":        {5: (10, 22), 6: (12, 26), 7: (16, 36), 8: (18, 40)},
    "convergent_forks":      {5: (10, 22), 6: (12, 26), 7: (14, 30), 8: (14, 34)},
    "parallel_wings":        {5: (18, 34), 6: (20, 38), 7: (22, 42), 8: (24, 46)},
    "cascade_trap":          {5: (10, 22), 6: (12, 32), 7: (14, 34), 8: (16, 38)},
    "order_dependent_lock":  {5: (10, 22), 6: (12, 26), 7: (14, 30), 8: (16, 34)},
    "relay":                 {6: (14, 28), 7: (16, 36), 8: (18, 36)},
    "investment_door":       {6: (12, 30), 7: (14, 38), 8: (16, 38)},
    "sequential_dual_locks": {6: (14, 28), 7: (16, 36), 8: (18, 36)},
    "key_collector":         {6: (14, 32), 7: (16, 36), 8: (18, 40)},
    "decoy_key":             {6: (12, 28), 7: (14, 32), 8: (16, 36)},
}

# Per-template minimum deadlocks by grid width.
# +1 at the largest tier size (extra traps produce more deadlocks).
TEMPLATE_MIN_DEADLOCKS_BY_SIZE: dict[str, dict[int, int]] = {
    "fork":                  {5: 1, 6: 1, 7: 1, 8: 1},
    "switchback":            {5: 2, 6: 2, 7: 2, 8: 2},
    "scarce_resource":       {5: 1, 6: 1, 7: 1, 8: 1},
    "conservation_trap":     {5: 2, 6: 2, 7: 3, 8: 2},
    "key_behind_key":        {5: 1, 6: 1, 7: 2, 8: 2},
    "convergent_forks":      {5: 2, 6: 2, 7: 3, 8: 3},
    "parallel_wings":        {5: 2, 6: 2, 7: 3, 8: 2},
    "cascade_trap":          {5: 2, 6: 1, 7: 2, 8: 2},
    "order_dependent_lock":  {5: 2, 6: 2, 7: 3, 8: 2},
    "relay":                 {6: 2, 7: 2, 8: 3},
    "investment_door":       {6: 2, 7: 2, 8: 3},
    "sequential_dual_locks": {6: 2, 7: 2, 8: 3},
    "key_collector":         {6: 2, 7: 2, 8: 3},
    "decoy_key":             {6: 2, 7: 2, 8: 3},
}


# ---------------------------------------------------------------------------
# Full instance generation pipeline
# ---------------------------------------------------------------------------

def _generate_validated_inner(
    width: int = 5,
    height: int = 5,
    seed: int = 0,
    template: str = "fork",
    max_retries: int = 200,
    solution_range: Optional[tuple[int, int]] = None,
    min_deadlocks: Optional[int] = None,
) -> tuple[Optional[GeneratedInstance], dict[str, int], int]:
    """Core generation loop with rejection tracking.

    Returns (instance_or_None, rejection_counts, total_attempts).
    """
    if template not in TEMPLATE_REGISTRY:
        raise ValueError(f"Unknown template {template!r}. Available: {TEMPLATE_NAMES}")

    grid_fn = TEMPLATE_REGISTRY[template]

    # --- Per-size solution range (explicit override takes precedence) ---
    if solution_range is not None:
        sol_range = solution_range
    else:
        size_ranges = TEMPLATE_SOLUTION_RANGES_BY_SIZE.get(template, {})
        sol_range = size_ranges.get(width, TEMPLATE_SOLUTION_RANGES.get(template, (6, 30)))

    # --- Per-size min deadlocks (explicit override takes precedence) ---
    if min_deadlocks is not None:
        min_dl = min_deadlocks
    else:
        size_deadlocks = TEMPLATE_MIN_DEADLOCKS_BY_SIZE.get(template, {})
        min_dl = size_deadlocks.get(width, TEMPLATE_MIN_DEADLOCKS.get(template, 1))

    # Auto-scale retries for larger grids (raw yield drops with size)
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
        "irrelevant_key": 0,
    }

    for attempt in range(effective_retries):
        rng = random.Random(seed + attempt)

        result = grid_fn(width, height, rng)
        if result is None:
            continue

        grid, initial_state = result

        # Filter 1: Must be solvable
        solution = oracle_solve(grid, initial_state)
        if solution is None:
            rejections["unsolvable"] += 1
            continue

        # Filter 2: Solution length in range
        if not (sol_range[0] <= len(solution) <= sol_range[1]):
            rejections["solution_range"] += 1
            continue

        # Filter 3: ALL greedy strategies must fail (critical anti-greedy filter)
        if not all_greedy_strategies_fail(grid, initial_state):
            rejections["greedy_passes"] += 1
            continue

        # Filter 4: Must have enough deadlocks
        catalog = discover_deadlocks(grid, initial_state,
                                     max_depth=12, max_states=10_000)
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

        # Filter 5: All keys must be relevant (no noise keys)
        if not _all_keys_relevant(grid, initial_state, solution, catalog):
            rejections["irrelevant_key"] += 1
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
            num_keys=len(grid.keys),
            num_doors=len(grid.doors),
            solution_length=len(solution),
            tier1_deadlocks=tier1,
            tier2_deadlocks=tier2,
            max_k_horizon=max_kh,
            generation_attempts=attempt + 1,
        )
        inst = _transform_instance(inst, rng)
        return inst, rejections, attempt + 1

    return None, rejections, effective_retries


def generate_validated_instance(
    width: int = 5,
    height: int = 5,
    seed: int = 0,
    template: str = "fork",
    max_retries: int = 200,
    solution_range: Optional[tuple[int, int]] = None,
    min_deadlocks: Optional[int] = None,
) -> Optional[GeneratedInstance]:
    """Generate a validated anti-greedy Chain Vault instance.

    Args:
        width, height: Grid dimensions.
        seed: Base RNG seed (each retry increments by 1).
        template: Template name — must be a key in TEMPLATE_REGISTRY.
        max_retries: Maximum generation attempts before giving up.
        solution_range: Override the per-template solution length range.
        min_deadlocks: Override the per-template minimum deadlock count.

    Filters (ALL must pass):
    1. Solvable from start
    2. Solution length in range (per-template default unless overridden)
    3. ALL greedy strategies fail (HARD filter — the critical one)
    4. Has sufficient deadlocks (Walking Dead states)
    5. All keys relevant — every key's color appears in oracle solution or causes deadlock
    """
    inst, _stats, _total = _generate_validated_inner(
        width=width, height=height, seed=seed, template=template,
        max_retries=max_retries, solution_range=solution_range,
        min_deadlocks=min_deadlocks,
    )
    return inst


def generate_validated_instance_with_stats(
    width: int = 5,
    height: int = 5,
    seed: int = 0,
    template: str = "fork",
    max_retries: int = 200,
    solution_range: Optional[tuple[int, int]] = None,
    min_deadlocks: Optional[int] = None,
) -> GenerationResult:
    """Like generate_validated_instance but returns full stats."""
    import time as _time
    t0 = _time.monotonic()
    instance, stats, total = _generate_validated_inner(
        width=width, height=height, seed=seed, template=template,
        max_retries=max_retries, solution_range=solution_range,
        min_deadlocks=min_deadlocks,
    )
    elapsed = _time.monotonic() - t0
    return GenerationResult(
        instance=instance, stats=stats,
        total_attempts=total, elapsed_seconds=elapsed,
    )
