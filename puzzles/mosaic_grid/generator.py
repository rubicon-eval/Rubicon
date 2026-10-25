"""Mosaic Grid procedural generation and deadlock discovery."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

from puzzles.common.types import Pos, Direction, add_pos, in_bounds, manhattan_distance
from .engine import (
    MosaicGrid,
    MosaicState,
    PlateInfo,
    DoorInfo,
    get_legal_actions,
    apply_action,
    is_solved,
    is_passable,
    decode_activated,
)
from .oracle import (
    is_solvable,
    oracle_solve,
    is_walking_dead,
    apply_actions,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DeadlockEntry:
    """A discovered deadlock: a path from S0 to a Walking Dead state."""

    path: list[Direction]
    fatal_step: int          # index of the action that caused solvability flip
    deadlock_state: MosaicState
    pre_deadlock_state: MosaicState
    k_horizon: int = 0       # reachable states from deadlock (Walking Dead continuation depth)


@dataclass
class GeneratedInstance:
    """A fully validated puzzle instance with solution and deadlock catalog."""

    grid: MosaicGrid
    initial_state: MosaicState
    solution: list[Direction]
    deadlock_catalog: list[DeadlockEntry]
    seed: int
    template_name: str = "barrier"
    # Metadata fields (defaults preserve backward compatibility)
    grid_width: int = 0
    grid_height: int = 0
    num_plates: int = 0
    num_doors: int = 0
    solution_length: int = 0
    tier1_deadlocks: int = 0
    tier2_deadlocks: int = 0
    max_k_horizon: int = 0
    generation_attempts: int = 0
    fingerprint: str = ""


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def _flood_fill(pos: Pos, passable: set[Pos]) -> set[Pos]:
    """Return the connected component containing `pos`."""
    visited = set()
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


def _all_doors_required(grid) -> bool:
    """Check that every door is structurally required to reach the exit.

    For each door, BFS from start to exit treating that door as impassable.
    If the exit is still reachable, the door is not required → return False.
    """
    all_positions = {
        Pos(r, c) for r in range(grid.height) for c in range(grid.width)
    }
    floor = all_positions - set(grid.walls)
    for door in grid.doors.values():
        # Floor without this door
        passable = floor - {door.pos}
        if grid.start not in passable or grid.exit_pos not in passable:
            continue  # start or exit is the door itself — always required
        reachable = _flood_fill(grid.start, passable)
        if grid.exit_pos in reachable:
            return False  # can bypass this door
    return True


def _find_chokepoints(floor_tiles: set[Pos]) -> list[Pos]:
    """Find tiles whose removal disconnects the floor (articulation points)."""
    chokepoints = []
    for tile in floor_tiles:
        remaining = floor_tiles - {tile}
        if remaining and not _is_connected(remaining):
            chokepoints.append(tile)
    return chokepoints


def _bfs_path(start: Pos, goal: Pos, passable: set[Pos]) -> Optional[list[Pos]]:
    """BFS shortest path between two positions on a set of passable tiles.

    Returns the path as a list of positions (including start and goal),
    or None if no path exists.
    """
    if start not in passable or goal not in passable:
        return None
    if start == goal:
        return [start]
    parent: dict[Pos, Pos] = {}
    visited = {start}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for d in Direction:
            np_ = add_pos(p, d)
            if np_ in passable and np_ not in visited:
                visited.add(np_)
                parent[np_] = p
                if np_ == goal:
                    # Reconstruct path
                    path = [goal]
                    cur = goal
                    while cur != start:
                        cur = parent[cur]
                        path.append(cur)
                    path.reverse()
                    return path
                queue.append(np_)
    return None


def _find_separating_vertices(
    floor: set[Pos], start: Pos, exit_pos: Pos,
) -> list[Pos]:
    """Find floor tiles whose removal disconnects start from exit.

    O(V^2) — fine for 8x8 grids (~50 floor tiles = ~2500 BFS calls).
    """
    candidates = []
    for tile in floor:
        if tile == start or tile == exit_pos:
            continue
        remaining = floor - {tile}
        if exit_pos not in _flood_fill(start, remaining):
            candidates.append(tile)
    return candidates


def _select_door_positions(
    candidates: list[Pos],
    num_needed: int,
    floor: set[Pos],
    rng: random.Random,
    min_spacing: int = 2,
    existing_doors: list[Pos] | None = None,
) -> Optional[list[Pos]]:
    """Greedily select well-spaced door positions from separating vertices.

    Each candidate must have >= 2 free floor neighbors (accessible from both
    sides). Enforces minimum manhattan distance between selected doors AND
    against any already-placed doors (existing_doors).
    """
    already = list(existing_doors) if existing_doors else []

    # Filter: must have >= 2 floor neighbors (door accessible from both sides)
    viable = []
    for c in candidates:
        n_free = sum(
            1 for d in Direction
            if add_pos(c, d) in floor and add_pos(c, d) != c
        )
        if n_free >= 2:
            viable.append(c)

    if len(viable) < num_needed:
        return None

    rng.shuffle(viable)
    selected: list[Pos] = []
    for v in viable:
        if (all(manhattan_distance(v, s) >= min_spacing for s in selected) and
                all(manhattan_distance(v, a) >= min_spacing for a in already)):
            selected.append(v)
            if len(selected) == num_needed:
                return selected
    return None


def _create_chokepoints_by_narrowing(
    floor: set[Pos],
    walls: set[Pos],
    all_positions: set[Pos],
    start: Pos,
    exit_pos: Pos,
    protected: set[Pos],
    rng: random.Random,
) -> tuple[set[Pos], list[Pos]]:
    """Phase 2 fallback: strategically add walls to create chokepoints.

    1. BFS shortest path from start to exit
    2. Find floor tiles adjacent to path but not on path (corridor side-tiles)
    3. Try adding each as a wall (verify connectivity maintained)
    4. After each addition, re-check for new separating vertices
    5. Stop after finding enough or max 3 wall additions

    Returns (updated_walls, separating_vertices).
    """
    walls = set(walls)  # copy
    path = _bfs_path(start, exit_pos, floor)
    if path is None:
        return walls, []

    path_set = set(path)

    # Find floor tiles adjacent to path but not on path and not protected
    side_tiles = []
    for p in path:
        for d in Direction:
            np_ = add_pos(p, d)
            if (np_ in floor and np_ not in path_set
                    and np_ not in protected):
                side_tiles.append(np_)

    rng.shuffle(side_tiles)
    added = 0
    for tile in side_tiles:
        if added >= 3:
            break
        test_walls = walls | {tile}
        test_floor = all_positions - test_walls
        if test_floor and _is_connected(test_floor):
            walls.add(tile)
            added += 1
            # Check if we now have separating vertices
            new_floor = all_positions - walls
            sep = _find_separating_vertices(new_floor, start, exit_pos)
            # Filter out protected positions
            sep = [s for s in sep if s not in protected]
            if sep:
                return walls, sep

    # Final check after all additions
    final_floor = all_positions - walls
    sep = _find_separating_vertices(final_floor, start, exit_pos)
    sep = [s for s in sep if s not in protected]
    return walls, sep


def _add_extra_walls(
    walls: set[Pos],
    all_positions: set[Pos],
    protected: set[Pos],
    rng: random.Random,
    max_extra: int | None = None,
) -> set[Pos]:
    """Add random extra walls while maintaining connectivity and protecting key positions."""
    floor = all_positions - walls
    extra_candidates = list(floor - protected)
    rng.shuffle(extra_candidates)
    if max_extra is None:
        max_extra = rng.randint(1, max(1, len(extra_candidates) // 5))
    added = 0
    for pos in extra_candidates[: max_extra * 3]:
        if added >= max_extra:
            break
        test_walls = walls | {pos}
        test_floor = all_positions - test_walls
        if test_floor and _is_connected(test_floor):
            walls.add(pos)
            added += 1
    return walls


def _build_grid_and_state(
    width: int,
    height: int,
    walls: set[Pos],
    plates: dict[str, PlateInfo],
    doors: dict[str, DoorInfo],
    start: Pos,
    exit_pos: Pos,
) -> tuple[MosaicGrid, MosaicState]:
    """Common helper to build a MosaicGrid and initial MosaicState."""
    grid = MosaicGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        plates=plates,
        doors=doors,
        start=start,
        exit_pos=exit_pos,
    )
    initial_state = MosaicState(
        agent_pos=start,
        painted=0,
        activated_plates=0,
    )
    return grid, initial_state


def _make_plates_and_door(
    plate_positions: list[Pos],
    door_pos: Pos,
    door_id: str = "D1",
) -> tuple[dict[str, PlateInfo], dict[str, DoorInfo]]:
    """Create plates all linked to a single door."""
    plates = {}
    plate_ids = []
    for i, pos in enumerate(plate_positions):
        pid = f"P{i + 1}"
        plates[pid] = PlateInfo(id=pid, pos=pos, linked_door_ids=(door_id,))
        plate_ids.append(pid)
    doors = {
        door_id: DoorInfo(id=door_id, pos=door_pos, required_plate_ids=tuple(plate_ids)),
    }
    return plates, doors


def _make_plates_and_doors(
    groups: list[tuple[str, Pos, list[Pos]]],
) -> tuple[dict[str, PlateInfo], dict[str, DoorInfo]]:
    """Create plates and doors for multiple door groups.

    Args:
        groups: list of (door_id, door_pos, plate_positions) tuples.

    Plates are numbered globally (P1, P2, P3...) across all groups.
    Each plate is linked only to its group's door.
    """
    plates: dict[str, PlateInfo] = {}
    doors: dict[str, DoorInfo] = {}
    global_idx = 1
    for door_id, door_pos, plate_positions in groups:
        plate_ids = []
        for pos in plate_positions:
            pid = f"P{global_idx}"
            plates[pid] = PlateInfo(id=pid, pos=pos, linked_door_ids=(door_id,))
            plate_ids.append(pid)
            global_idx += 1
        doors[door_id] = DoorInfo(
            id=door_id, pos=door_pos, required_plate_ids=tuple(plate_ids),
        )
    return plates, doors


def _split_plates_by_proximity(
    plate_positions: list[Pos],
    anchor_a: Pos,
    anchor_b: Pos,
) -> tuple[list[Pos], list[Pos]]:
    """Split plates into two groups by proximity to two anchor positions.

    Each plate goes to whichever anchor is closer (manhattan distance).
    Ties go to group A.  Guarantees at least 1 plate per group by moving
    the closest plate from the larger group if one group is empty.
    """
    group_a: list[Pos] = []
    group_b: list[Pos] = []
    for p in plate_positions:
        da = manhattan_distance(p, anchor_a)
        db = manhattan_distance(p, anchor_b)
        if da <= db:
            group_a.append(p)
        else:
            group_b.append(p)
    # Guarantee at least 1 per group
    if not group_a and group_b:
        # Move the plate closest to anchor_a from group_b
        best = min(group_b, key=lambda p: manhattan_distance(p, anchor_a))
        group_b.remove(best)
        group_a.append(best)
    elif not group_b and group_a:
        best = min(group_a, key=lambda p: manhattan_distance(p, anchor_b))
        group_a.remove(best)
        group_b.append(best)
    return group_a, group_b


def _split_plates_n_way(
    plate_positions: list[Pos],
    anchors: list[Pos],
) -> list[list[Pos]]:
    """Split plates into N groups by proximity to N anchor positions.

    Each plate goes to whichever anchor is closest (manhattan distance).
    Ties go to the lowest-index anchor.  Guarantees at least 1 plate per
    group by stealing from the largest group if any group is empty.
    """
    n = len(anchors)
    groups: list[list[Pos]] = [[] for _ in range(n)]
    for p in plate_positions:
        best_i = min(range(n), key=lambda i: manhattan_distance(p, anchors[i]))
        groups[best_i].append(p)
    # Guarantee at least 1 per group
    for i in range(n):
        if not groups[i]:
            # Find the largest group and steal its closest plate to anchor[i]
            largest = max(range(n), key=lambda j: len(groups[j]))
            if len(groups[largest]) >= 2:
                best = min(groups[largest], key=lambda p: manhattan_distance(p, anchors[i]))
                groups[largest].remove(best)
                groups[i].append(best)
    return groups


def _order_doors_by_path(
    door_positions: list[Pos],
    floor: set[Pos],
    start: Pos,
    exit_pos: Pos,
) -> list[Pos]:
    """Order door positions by distance along the BFS shortest path from start to exit.

    Closest to start = D1.  If a door isn't on the path, it sorts last.
    """
    path = _bfs_path(start, exit_pos, floor)
    if path is None:
        return list(door_positions)
    path_index = {p: i for i, p in enumerate(path)}
    return sorted(door_positions, key=lambda d: path_index.get(d, len(path)))


def _compute_door_zones(
    ordered_doors: list[Pos],
    floor: set[Pos],
    start: Pos,
) -> list[set[Pos]]:
    """Compute spatial zones by progressively opening doors.

    - Zone 0: reachable from start with all doors closed
    - Zone 1: newly reachable after D1 opens (but D2, D3... still closed)
    - Zone 2: newly reachable after D2 opens, etc.
    """
    zones: list[set[Pos]] = []
    seen: set[Pos] = set()

    # Zone 0: all doors blocked
    passable_0 = floor - set(ordered_doors)
    zone_0 = _flood_fill(start, passable_0) if start in passable_0 else set()
    zones.append(zone_0)
    seen |= zone_0

    # Zone i (for each door we open in order)
    open_doors: set[Pos] = set()
    for i, door_pos in enumerate(ordered_doors):
        open_doors.add(door_pos)
        remaining_closed = set(ordered_doors) - open_doors
        passable_i = floor - remaining_closed
        reachable_i = _flood_fill(start, passable_i) if start in passable_i else set()
        zone_i = reachable_i - seen
        zones.append(zone_i)
        seen |= zone_i

    return zones


def _select_plates_in_zone(
    candidates: list[Pos],
    num_needed: int,
    start: Pos,
    rng: random.Random,
    min_spacing: int = 2,
    already_placed: set[Pos] | None = None,
) -> list[Pos] | None:
    """Select well-spaced plates within a single zone.

    Greedily picks candidates with minimum manhattan distance from each
    other AND from ``already_placed``.  Returns None if can't fill quota.
    """
    if num_needed <= 0:
        return []
    placed = set(already_placed) if already_placed else set()
    rng.shuffle(candidates)
    selected: list[Pos] = []
    for cand in candidates:
        if cand in placed:
            continue
        if any(manhattan_distance(cand, s) < min_spacing for s in selected):
            continue
        if any(manhattan_distance(cand, p) < min_spacing for p in placed):
            continue
        selected.append(cand)
        if len(selected) == num_needed:
            return selected
    # Relax spacing to 1 and retry
    if min_spacing > 1:
        rng.shuffle(candidates)
        selected = []
        for cand in candidates:
            if cand in placed:
                continue
            if any(manhattan_distance(cand, s) < 1 for s in selected):
                continue
            selected.append(cand)
            if len(selected) == num_needed:
                return selected
    return None


def _select_plates_for_zones(
    num_plates: int,
    ordered_doors: list[Pos],
    zones: list[set[Pos]],
    protected: set[Pos],
    rng: random.Random,
    start: Pos,
    min_plates_per_door: int = 2,
) -> list[tuple[str, Pos, list[Pos]]] | None:
    """Select fresh plates from zone floor tiles for staggered multi-door.

    Instead of redistributing pre-placed plates, selects NEW plates from
    zone floor tiles to guarantee staggered unlocking.

    Algorithm:
      1. Budget: each door gets max(min_plates_per_door, num_plates // n_doors),
         remainder to D1
      2. D1 (k=0): select quota from zone 0 candidates
      3. D_k (k>=1): select quota from zone k candidates (behind door k-1)
      4. Uses _select_plates_in_zone with cross-zone spacing enforcement
      5. Returns None if any zone can't fill its quota
    """
    n_doors = len(ordered_doors)
    if n_doors == 0 or len(zones) < n_doors + 1:
        return None

    # Budget allocation
    per_door = max(min_plates_per_door, num_plates // n_doors)
    budgets = [per_door] * n_doors
    # Give remainder to D1
    remainder = num_plates - sum(budgets)
    if remainder > 0:
        budgets[0] += remainder

    all_placed: set[Pos] = set()
    door_plates: list[list[Pos]] = [[] for _ in range(n_doors)]

    # D_k for k >= 1 first (behind-door zones are more constrained)
    for k in range(1, n_doors):
        zone = zones[k]  # zone behind door k-1
        candidates = [
            t for t in zone
            if t not in protected and t not in all_placed
               and t not in set(ordered_doors)
        ]
        selected = _select_plates_in_zone(
            candidates, budgets[k], start, rng,
            min_spacing=2, already_placed=all_placed,
        )
        if selected is None:
            return None
        door_plates[k] = selected
        all_placed.update(selected)

    # D1 (k=0): select from zone 0
    zone0 = zones[0]
    candidates_0 = [
        t for t in zone0
        if t not in protected and t not in all_placed
           and t not in set(ordered_doors)
    ]
    selected_0 = _select_plates_in_zone(
        candidates_0, budgets[0], start, rng,
        min_spacing=2, already_placed=all_placed,
    )
    if selected_0 is None:
        return None
    door_plates[0] = selected_0

    # Build result tuples
    result = []
    for k in range(n_doors):
        door_id = f"D{k + 1}"
        result.append((door_id, ordered_doors[k], door_plates[k]))
    return result


def _place_doors_at_chokepoints(
    walls: set[Pos],
    all_positions: set[Pos],
    protected: set[Pos],
    plate_positions: list[Pos],
    door_pos: Pos,
    num_doors: int,
    rng: random.Random,
    width: int,
    height: int,
    start: Pos,
    exit_pos: Pos,
) -> Optional[tuple[dict[str, PlateInfo], dict[str, DoorInfo], set[Pos]]]:
    """Place multiple doors at natural chokepoints (separating vertices).

    Phase 1: Find separating vertices on the current floor layout.
             If enough exist, select well-spaced positions as doors.

    Phase 2: If not enough natural chokepoints, strategically add 1-3 walls
             to narrow corridors, then re-check for separating vertices.

    Smart plate distribution: tries zone-based staggered plate selection first
    (D2's plates behind D1), falling back to proximity split when zones are
    too small (< 4 tiles).  This ensures staggering when the layout supports
    it while keeping viability high on templates with tight zones.

    Returns (plates, doors, updated_walls) or None if not enough chokepoints
    can be created. Templates fall back to single door on None.
    """
    if num_doors < 2:
        return None

    min_door_spacing = rng.randint(2, 3)
    all_protected = protected | {door_pos}
    # Remove old plate positions from protected so zone tiles are available
    # for fresh plate selection in the staggered path
    core_protected = (protected - set(plate_positions)) | {door_pos}

    def _try_place(cur_walls: set[Pos], sep: list[Pos]) -> Optional[tuple[dict, dict, set]]:
        """Try door selection, then smart plate assignment."""
        cur_floor = all_positions - cur_walls
        extra_doors_needed = num_doors - 1

        door_anchors = _select_door_positions(
            sep, extra_doors_needed, cur_floor, rng,
            min_spacing=min_door_spacing, existing_doors=[door_pos],
        )
        if door_anchors is None:
            return None

        all_anchors = [door_pos] + door_anchors
        ordered = _order_doors_by_path(all_anchors, cur_floor, start, exit_pos)
        zones = _compute_door_zones(ordered, cur_floor, start)

        # Check if behind-door zones are large enough for staggering
        min_zone_size = rng.randint(6, 8)
        zones_viable = all(len(z) >= min_zone_size for z in zones[1:])

        if zones_viable:
            # Try zone-based fresh plate selection (staggered)
            zone_protected = core_protected | set(ordered)
            zone_result = _select_plates_for_zones(
                len(plate_positions), ordered, zones,
                zone_protected, rng, start, min_plates_per_door=2,
            )
            if zone_result is not None:
                plates, doors = _make_plates_and_doors(zone_result)
                return plates, doors, cur_walls

        # Fallback: proximity split (non-staggered but always viable)
        groups = _split_plates_n_way(plate_positions, all_anchors)
        door_groups = [
            (f"D{i + 1}", all_anchors[i], groups[i])
            for i in range(len(all_anchors))
        ]
        plates, doors = _make_plates_and_doors(door_groups)
        return plates, doors, cur_walls

    floor = all_positions - walls

    # Phase 1: Natural separating vertices
    sep = _find_separating_vertices(floor, start, exit_pos)
    sep = [s for s in sep if s not in all_protected]

    result = _try_place(walls, sep)
    if result is not None:
        return result

    # Phase 2: Strategic wall narrowing
    new_walls, sep = _create_chokepoints_by_narrowing(
        floor, walls, all_positions, start, exit_pos, all_protected, rng,
    )
    if sep:
        result = _try_place(new_walls, sep)
        if result is not None:
            return result

    return None


def _select_plate_positions(
    candidates: list[Pos],
    num_plates: int,
    start: Pos,
    rng: random.Random,
    min_plate_spacing: int = 2,
) -> Optional[list[Pos]]:
    """Select plate positions with spacing and distance-from-start constraints.

    Enforces (hard):
    - min manhattan distance between any two plates (min_plate_spacing)
    Prefers (soft, falls back if not possible):
    - distinct manhattan distances from start (no two plates equidistant)
    Returns None if even spacing constraint can't be satisfied.
    """
    rng.shuffle(candidates)
    # Try with both constraints (distinct distances + spacing)
    selected: list[Pos] = []
    used_distances: set[int] = set()
    for cand in candidates:
        d = manhattan_distance(cand, start)
        if d in used_distances:
            continue
        if any(manhattan_distance(cand, p) < min_plate_spacing for p in selected):
            continue
        selected.append(cand)
        used_distances.add(d)
        if len(selected) == num_plates:
            return selected
    # Fall back: spacing only (no distance constraint)
    rng.shuffle(candidates)
    selected = []
    for cand in candidates:
        if any(manhattan_distance(cand, p) < min_plate_spacing for p in selected):
            continue
        selected.append(cand)
        if len(selected) == num_plates:
            return selected
    return None


def generate_barrier_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with a wall barrier that forces passage through a door.

    Strategy:
    1. Place a vertical wall barrier at a column, with one gap = the door
    2. Start on the left side, exit on the right side
    3. Place plates on the left side
    4. Add random extra walls for variety while maintaining connectivity

    When num_doors>=2, extra doors are placed at natural chokepoints via
    _place_doors_at_chokepoints.

    Returns (grid, initial_state) or None if generation fails.
    """
    if rng is None:
        rng = random.Random()

    # Choose barrier column (middle-ish area)
    barrier_col = rng.randint(width // 3, width * 2 // 3)

    # Single gap in barrier = D1
    door_row = rng.randint(0, height - 1)
    door_pos = Pos(door_row, barrier_col)

    # Build wall barrier
    walls: set[Pos] = set()
    for r in range(height):
        if r != door_row:
            walls.add(Pos(r, barrier_col))

    # Pick start (left side) and exit (right side)
    left_floor = [
        Pos(r, c) for r in range(height) for c in range(barrier_col)
        if Pos(r, c) not in walls
    ]
    right_floor = [
        Pos(r, c) for r in range(height) for c in range(barrier_col + 1, width)
        if Pos(r, c) not in walls
    ]

    if not left_floor or not right_floor:
        return None

    # Coin flip: randomise which side is start vs exit
    flip = rng.random() < 0.5
    if flip:
        start = rng.choice(right_floor)
        exit_pos = rng.choice(left_floor)
        start_side_floor = right_floor
    else:
        start = rng.choice(left_floor)
        exit_pos = rng.choice(right_floor)
        start_side_floor = left_floor

    # Add random extra walls on each side for variety
    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}
    extra_candidates = list(all_positions - walls - {start, exit_pos, door_pos})
    rng.shuffle(extra_candidates)

    n_extra = rng.randint(1, max(1, len(extra_candidates) // 5))
    for pos in extra_candidates[:n_extra * 3]:  # try more, accept fewer
        if len(walls) - height >= n_extra:
            break
        test_walls = walls | {pos}
        floor = all_positions - test_walls
        # Check left side connected
        left_check = {p for p in floor if p.col < barrier_col}
        right_check = {p for p in floor if p.col > barrier_col}
        if (left_check and _is_connected(left_check) and
                right_check and _is_connected(right_check) and
                start in left_check and exit_pos in right_check):
            walls.add(pos)

    # Place plates on the start side
    available_for_plates = [
        p for p in start_side_floor
        if p not in walls and p != start and p != door_pos
    ]
    plate_positions = _select_plate_positions(available_for_plates, num_plates, start, rng)
    if plate_positions is None:
        return None

    protected = {start, exit_pos, door_pos} | set(plate_positions)

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 2: Bottleneck
# ---------------------------------------------------------------------------

def generate_bottleneck_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with a partial vertical wall, free gap, and exit pocket.

    Differs from Barrier: wall covers <=60% of column, gap is a free floor
    tile allowing multi-crossing.  The exit sits in a pocket whose only
    entrance is the door, so the agent *must* activate plates before reaching
    the exit.  Plates on both sides force multi-crossing through the gap.
    """
    if rng is None:
        rng = random.Random()

    barrier_col = rng.randint(width // 3, width * 2 // 3)

    # Wall covers 40-60% of column (strictly less than barrier's 100%)
    max_wall = max(2, int(height * 0.6))
    min_wall = max(2, int(height * 0.4))
    wall_count = rng.randint(min_wall, max_wall)

    # Pick a floor gap row and build partial wall
    gap_row = rng.randint(0, height - 1)
    candidate_rows = [r for r in range(height) if r != gap_row]
    rng.shuffle(candidate_rows)
    wall_rows = candidate_rows[:wall_count]

    walls: set[Pos] = {Pos(r, barrier_col) for r in wall_rows}
    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}

    left_floor = [p for p in (all_positions - walls) if p.col < barrier_col]
    right_floor = [p for p in (all_positions - walls) if p.col > barrier_col]
    if not left_floor or not right_floor:
        return None

    # Coin flip: randomise which side is start vs exit
    flip = rng.random() < 0.5
    if flip:
        start = rng.choice(right_floor)
        exit_pos = rng.choice(left_floor)
    else:
        start = rng.choice(left_floor)
        exit_pos = rng.choice(right_floor)

    # --- Exit pocket: wall off all exit neighbours except one (the door) ---
    exit_neighbors = [
        add_pos(exit_pos, d) for d in Direction
        if in_bounds(add_pos(exit_pos, d), width, height)
    ]
    free_neighbors = [
        n for n in exit_neighbors
        if n not in walls and n != start
    ]
    if not free_neighbors:
        return None
    # Pick one neighbour as the door; wall off the rest
    rng.shuffle(free_neighbors)
    door_pos = free_neighbors[0]
    for n in free_neighbors[1:]:
        walls.add(n)

    # Re-derive floor after pocket walls and check connectivity.
    # door_pos is floor (becomes a door tile, not a wall) so include it.
    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    gap_pos = Pos(gap_row, barrier_col)

    # Plates on both sides of the barrier
    left_floor = [p for p in floor if p.col < barrier_col]
    right_floor = [p for p in floor if p.col > barrier_col]
    all_plate_cands = [
        p for p in left_floor + right_floor
        if p != start and p != exit_pos and p != door_pos
    ]
    plate_positions = _select_plate_positions(all_plate_cands, num_plates, start, rng)
    if plate_positions is None:
        return None

    protected = {start, exit_pos, door_pos, gap_pos} | set(plate_positions)
    grid_size = max(width, height)
    enhanced = grid_size >= 6 and rng.random() < 0.5
    if enhanced:
        floor_count = len(all_positions - walls - protected)
        extra_budget = max(2, floor_count // 6)
        _add_extra_walls(walls, all_positions, protected, rng, max_extra=extra_budget)
    else:
        _add_extra_walls(walls, all_positions, protected, rng)

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 3: Horizontal
# ---------------------------------------------------------------------------

def generate_horizontal_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with a horizontal wall and exit pocket.

    Full horizontal wall with a free gap for crossing. Exit pocket on the
    far side guarded by a door. All plates on the start side. The agent
    must collect all plates (painting tiles) while preserving a path to the
    gap, then cross the wall and enter the exit pocket through the door.
    The challenge is that collecting plates on the start side paints paths
    the agent may need to reach the gap.
    """
    if rng is None:
        rng = random.Random()

    if height < 5 or width < 5:
        return None

    # Horizontal wall row in middle area
    barrier_row = rng.randint(2, height - 3)

    # Free gap in the wall (not a door, just a passage)
    gap_col = rng.randint(0, width - 1)

    # Build horizontal wall with one gap
    walls: set[Pos] = set()
    for c in range(width):
        if c != gap_col:
            walls.add(Pos(barrier_row, c))

    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}

    # Start on one side, exit on the other (coin flip)
    top_floor = [p for p in (all_positions - walls) if p.row < barrier_row]
    bot_floor = [p for p in (all_positions - walls) if p.row > barrier_row]
    if not top_floor or not bot_floor:
        return None

    flip = rng.random() < 0.5
    if flip:
        start = rng.choice(bot_floor)
        exit_cands = top_floor
    else:
        start = rng.choice(top_floor)
        exit_cands = bot_floor

    # Exit must not be adjacent to the gap (otherwise exit pocket can't block)
    gap_pos = Pos(barrier_row, gap_col)
    gap_adj = {add_pos(gap_pos, d) for d in Direction}
    exit_cands = [p for p in exit_cands if p not in gap_adj and p != start]
    if not exit_cands:
        return None
    exit_pos = rng.choice(exit_cands)

    # --- Exit pocket: wall off all exit neighbours except one (the door) ---
    exit_neighbors = [
        add_pos(exit_pos, d) for d in Direction
        if in_bounds(add_pos(exit_pos, d), height, width)
    ]
    free_neighbors = [
        n for n in exit_neighbors
        if n not in walls and n != start
    ]
    if not free_neighbors:
        return None
    rng.shuffle(free_neighbors)
    door_pos = free_neighbors[0]
    for n in free_neighbors[1:]:
        walls.add(n)

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # All plates on the start side
    start_side = (
        [p for p in floor if p.row > barrier_row]
        if flip else
        [p for p in floor if p.row < barrier_row]
    )
    plate_cands = [
        p for p in start_side
        if p != start and p != exit_pos and p != door_pos
    ]
    plate_positions = _select_plate_positions(plate_cands, num_plates, start, rng)
    if plate_positions is None:
        return None

    protected = {start, exit_pos, door_pos, gap_pos} | set(plate_positions)
    _add_extra_walls(walls, all_positions, protected, rng)

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 4: Spiral
# ---------------------------------------------------------------------------

def generate_spiral_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with two partial horizontal walls creating a serpentine path.

    Two horizontal wall strips with opposing gaps. Plates in the middle band
    between the walls. Door at one gap. Forces serpentine traversal that paints
    tiles blocking the return path.
    """
    if rng is None:
        rng = random.Random()

    if height < 5 or width < 4:
        return None

    grid_size = max(width, height)
    enhanced = grid_size >= 7 and rng.random() < 0.5
    if enhanced and width < 6:
        enhanced = False  # not enough width for wider gaps
    if enhanced:
        min_sep = 3
        gap_width = 3
    else:
        min_sep = 2
        gap_width = 2

    # Two horizontal walls, at least min_sep rows apart
    row1 = rng.randint(1, height // 2 - 1)
    if row1 + min_sep > height - 2:
        return None
    row2 = rng.randint(row1 + min_sep, height - 2)

    # Opposing gaps: strip1 gap near right, strip2 gap near left
    gap1_start = rng.randint(max(0, width // 2), width - gap_width)
    gap1_cols = {gap1_start + i for i in range(gap_width)}
    gap2_start = rng.randint(0, min(width // 2 - 1, width - gap_width))
    gap2_cols = {gap2_start + i for i in range(gap_width)}

    walls: set[Pos] = set()
    for c in range(width):
        if c not in gap1_cols:
            walls.add(Pos(row1, c))
        if c not in gap2_cols:
            walls.add(Pos(row2, c))

    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}
    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Start top, exit bottom (with coin flip)
    top_floor = [p for p in floor if p.row < row1]
    bot_floor = [p for p in floor if p.row > row2]
    mid_floor = [p for p in floor if row1 < p.row < row2]
    if not top_floor or not bot_floor or not mid_floor:
        return None

    # Coin flip: randomise which end is start vs exit
    flip = rng.random() < 0.5
    if flip:
        start = rng.choice(bot_floor)
        exit_pos = rng.choice(top_floor)
        plate_cands = [
            p for p in bot_floor if p != start
        ] + [p for p in mid_floor]
    else:
        start = rng.choice(top_floor)
        exit_pos = rng.choice(bot_floor)
        plate_cands = [
            p for p in top_floor if p != start
        ] + [p for p in mid_floor]

    # --- Exit pocket: wall off all exit neighbours except one (the door) ---
    exit_neighbors = [
        add_pos(exit_pos, d) for d in Direction
        if in_bounds(add_pos(exit_pos, d), height, width)
    ]
    free_neighbors = [
        n for n in exit_neighbors
        if n not in walls and n != start
    ]
    if not free_neighbors:
        return None
    rng.shuffle(free_neighbors)
    door_pos = free_neighbors[0]
    for n in free_neighbors[1:]:
        walls.add(n)

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    plate_cands = [
        p for p in plate_cands
        if p != exit_pos and p != door_pos and p not in walls
    ]
    plate_positions = _select_plate_positions(plate_cands, num_plates, start, rng)
    if plate_positions is None:
        return None

    protected = {start, exit_pos, door_pos} | set(plate_positions)
    extra_budget = rng.randint(1, grid_size - 4) if enhanced else rng.randint(0, 2)
    _add_extra_walls(walls, all_positions, protected, rng, max_extra=extra_budget)

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 5: Cross
# ---------------------------------------------------------------------------

def generate_cross_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with two perpendicular wall segments forming a cross.

    A horizontal segment and a vertical segment cross near the grid center,
    with a 1-tile gap at the intersection so the center stays passable.
    This creates 4 semi-separated regions with narrow passages.  Exit pocket
    ensures the door is required.  Routing through passages paints tiles,
    limiting future paths.
    """
    if rng is None:
        rng = random.Random()

    if height < 5 or width < 5:
        return None

    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}

    # Cross center
    mid_row = rng.randint(2, height - 3)
    mid_col = rng.randint(2, width - 3)

    walls: set[Pos] = set()

    # Horizontal segment along mid_row (skip center tile and edges)
    for c in range(1, width - 1):
        if c != mid_col:
            walls.add(Pos(mid_row, c))

    # Vertical segment along mid_col (skip center tile and edges)
    for r in range(1, height - 1):
        if r != mid_row:
            walls.add(Pos(r, mid_col))

    # Open 1-2 extra gaps in each arm so regions aren't too isolated
    arm_tiles = {
        "up": [Pos(r, mid_col) for r in range(1, mid_row) if Pos(r, mid_col) in walls],
        "down": [Pos(r, mid_col) for r in range(mid_row + 1, height - 1) if Pos(r, mid_col) in walls],
        "left": [Pos(mid_row, c) for c in range(1, mid_col) if Pos(mid_row, c) in walls],
        "right": [Pos(mid_row, c) for c in range(mid_col + 1, width - 1) if Pos(mid_row, c) in walls],
    }
    for arm_name, tiles in arm_tiles.items():
        if tiles:
            rng.shuffle(tiles)
            # Open 1 gap per arm
            walls.discard(tiles[0])

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Start and exit in diagonal-opposite quadrants
    tl = [p for p in floor if p.row < mid_row and p.col < mid_col]
    tr = [p for p in floor if p.row < mid_row and p.col > mid_col]
    bl = [p for p in floor if p.row > mid_row and p.col < mid_col]
    br = [p for p in floor if p.row > mid_row and p.col > mid_col]

    pairs = [(tl, br), (tr, bl)]
    rng.shuffle(pairs)
    start = None
    exit_pos = None
    for qa, qb in pairs:
        if qa and qb:
            start = rng.choice(qa)
            exit_pos = rng.choice(qb)
            break
    if start is None:
        # Fallback: any two distant floor tiles
        floor_list = list(floor)
        rng.shuffle(floor_list)
        start = floor_list[0]
        exit_pos = floor_list[-1]

    # --- Exit pocket: wall off all exit neighbours except one (the door) ---
    exit_neighbors = [
        add_pos(exit_pos, d) for d in Direction
        if in_bounds(add_pos(exit_pos, d), height, width)
    ]
    free_neighbors = [
        n for n in exit_neighbors
        if n not in walls and n != start
    ]
    if not free_neighbors:
        return None
    rng.shuffle(free_neighbors)
    door_pos = free_neighbors[0]
    for n in free_neighbors[1:]:
        walls.add(n)

    # Re-derive floor after pocket walls and check connectivity
    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Plates spread across the grid
    plate_cands = [
        p for p in floor
        if p != start and p != exit_pos and p != door_pos
    ]
    plate_positions = _select_plate_positions(plate_cands, num_plates, start, rng)
    if plate_positions is None:
        return None

    protected = {start, exit_pos, door_pos} | set(plate_positions)
    _add_extra_walls(walls, all_positions, protected, rng, max_extra=rng.randint(1, 4))

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 6: Peninsula
# ---------------------------------------------------------------------------

def generate_peninsula_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with a dead-end corridor (peninsula) containing a plate.

    A 2-3 tile dead-end corridor extends from an edge, walled on the sides.
    A plate sits at the tip. Exit pocket ensures the door is required.
    Greedy enters the dead-end for the plate, painting the escape path.
    """
    if rng is None:
        rng = random.Random()

    if height < 4 or width < 4:
        return None

    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}
    walls: set[Pos] = set()

    # Build peninsula from a random edge
    # Choose: top, bottom, left, or right edge
    edge = rng.choice(["top", "bottom", "left", "right"])
    grid_size = max(width, height)
    enhanced = grid_size >= 6 and rng.random() < 0.5
    if enhanced:
        pen_len = rng.randint(3, min(5, grid_size - 2))
        pen_width = 2  # wide corridor allows backtracking
    else:
        pen_len = rng.randint(2, 3)
        pen_width = 1

    if edge in ("top", "bottom"):
        max_col = width - 2 - (pen_width - 1)
        if max_col < 1:
            return None
        pen_col = rng.randint(1, max_col)
        pen_cols = [pen_col + j for j in range(pen_width)]
        if edge == "top":
            for i in range(pen_len):
                if pen_cols[0] - 1 >= 0:
                    walls.add(Pos(i, pen_cols[0] - 1))
                if pen_cols[-1] + 1 < width:
                    walls.add(Pos(i, pen_cols[-1] + 1))
            tip = Pos(0, pen_cols[0])
            pen_tiles = [Pos(i, c) for i in range(pen_len) for c in pen_cols]
        else:  # bottom
            for i in range(pen_len):
                r = height - 1 - i
                if pen_cols[0] - 1 >= 0:
                    walls.add(Pos(r, pen_cols[0] - 1))
                if pen_cols[-1] + 1 < width:
                    walls.add(Pos(r, pen_cols[-1] + 1))
            tip = Pos(height - 1, pen_cols[0])
            pen_tiles = [Pos(height - 1 - i, c) for i in range(pen_len) for c in pen_cols]
    else:  # left or right
        max_row = height - 2 - (pen_width - 1)
        if max_row < 1:
            return None
        pen_row = rng.randint(1, max_row)
        pen_rows = [pen_row + j for j in range(pen_width)]
        if edge == "left":
            for i in range(pen_len):
                if pen_rows[0] - 1 >= 0:
                    walls.add(Pos(pen_rows[0] - 1, i))
                if pen_rows[-1] + 1 < height:
                    walls.add(Pos(pen_rows[-1] + 1, i))
            tip = Pos(pen_rows[0], 0)
            pen_tiles = [Pos(r, i) for i in range(pen_len) for r in pen_rows]
        else:  # right
            for i in range(pen_len):
                c = width - 1 - i
                if pen_rows[0] - 1 >= 0:
                    walls.add(Pos(pen_rows[0] - 1, c))
                if pen_rows[-1] + 1 < height:
                    walls.add(Pos(pen_rows[-1] + 1, c))
            tip = Pos(pen_rows[0], width - 1)
            pen_tiles = [Pos(r, width - 1 - i) for i in range(pen_len) for r in pen_rows]

    # Remove any wall that overlaps a peninsula tile
    walls -= set(pen_tiles)

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Main area = floor tiles not in peninsula
    main_floor = [p for p in floor if p not in pen_tiles]
    if len(main_floor) < 3:
        return None

    # Start in main area, far from peninsula
    rng.shuffle(main_floor)
    start = main_floor[0]

    # Exit in main area, away from start
    exit_candidates = [
        p for p in main_floor
        if p != start and manhattan_distance(p, start) >= max(width, height) // 2
    ]
    if not exit_candidates:
        exit_candidates = [p for p in main_floor if p != start]
    if not exit_candidates:
        return None
    exit_pos = rng.choice(exit_candidates)

    # --- Exit pocket: wall off all exit neighbours except one (the door) ---
    exit_neighbors = [
        add_pos(exit_pos, d) for d in Direction
        if in_bounds(add_pos(exit_pos, d), height, width)
    ]
    free_neighbors = [
        n for n in exit_neighbors
        if n not in walls and n != start
    ]
    if not free_neighbors:
        return None
    rng.shuffle(free_neighbors)
    door_pos = free_neighbors[0]
    for n in free_neighbors[1:]:
        walls.add(n)

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # When enhanced (wide corridor), plates can go at the peninsula tip since
    # the wider path allows backtracking.  Original narrow corridor excludes
    # peninsula tiles (entering paints your only escape route).
    pen_set = set(pen_tiles)
    if enhanced:
        main_plate_cands = [
            p for p in floor
            if p != start and p != exit_pos and p != door_pos
        ]
    else:
        main_plate_cands = [
            p for p in floor
            if p not in pen_set and p != start and p != exit_pos and p != door_pos
        ]
    plate_positions = _select_plate_positions(main_plate_cands, num_plates, start, rng)
    if plate_positions is None:
        return None

    protected = {start, exit_pos, door_pos} | set(plate_positions) | set(pen_tiles)
    _add_extra_walls(walls, all_positions, protected, rng)

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 7: Diagonal
# ---------------------------------------------------------------------------

def generate_diagonal_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with a staircase-shaped diagonal wall (>=3 steps).

    Wall steps from one corner area toward the opposite. Door at a chokepoint.
    Plates on the start side. Manhattan-greedy is useless against the jagged wall.
    """
    if rng is None:
        rng = random.Random()

    if height < 4 or width < 4:
        return None

    # Staircase from top-left area toward bottom-right (or flipped)
    flip = rng.choice([False, True])
    steps = rng.randint(3, min(height, width) - 1)

    walls: set[Pos] = set()
    for i in range(steps):
        if flip:
            r = height - 1 - i
            c = width - 1 - i
        else:
            r = i
            c = i
        if 0 <= r < height and 0 <= c < width:
            walls.add(Pos(r, c))
            # Extend each step horizontally or vertically for thickness
            if rng.random() < 0.5 and c + 1 < width:
                walls.add(Pos(r, c + 1))
            elif r + 1 < height:
                walls.add(Pos(r + 1, c))

    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}
    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Start on one side of diagonal, exit on other
    if flip:
        start_cands = [p for p in floor if p.row > height // 2 or p.col > width // 2]
        exit_cands = [p for p in floor if p.row < height // 2 or p.col < width // 2]
    else:
        start_cands = [p for p in floor if p.row < height // 2 or p.col < width // 2]
        exit_cands = [p for p in floor if p.row > height // 2 or p.col > width // 2]

    if not start_cands or not exit_cands:
        # Fall back to any floor tiles
        floor_list = list(floor)
        if len(floor_list) < 2:
            return None
        rng.shuffle(floor_list)
        start = floor_list[0]
        exit_pos = floor_list[1]
    else:
        start = rng.choice(start_cands)
        exit_pos = rng.choice(exit_cands)

    # --- Exit pocket: wall off all exit neighbours except one (the door) ---
    exit_neighbors = [
        add_pos(exit_pos, d) for d in Direction
        if in_bounds(add_pos(exit_pos, d), width, height)
    ]
    free_neighbors = [
        n for n in exit_neighbors
        if n not in walls and n != start
    ]
    if not free_neighbors:
        return None
    rng.shuffle(free_neighbors)
    door_pos = free_neighbors[0]
    for n in free_neighbors[1:]:
        walls.add(n)

    # Re-derive floor after pocket walls and check connectivity
    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Plates on start side
    plate_cands = [
        p for p in floor
        if p != start and p != exit_pos and p != door_pos
    ]
    if len(plate_cands) < num_plates:
        return None

    rng.shuffle(plate_cands)
    plate_positions = plate_cands[:num_plates]

    protected = {start, exit_pos, door_pos} | set(plate_positions)
    _add_extra_walls(walls, all_positions, protected, rng, max_extra=rng.randint(0, 2))

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 8: L-Shape
# ---------------------------------------------------------------------------

def generate_lshape_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with an L-shaped wall extending to two grid edges.

    The L-wall seals off a corner region. One gap in the wall is the door.
    Start + plates inside the L, exit outside. The agent must collect all
    plates (which are spread apart) and cross through the door to reach
    the exit. Plates have a minimum spacing of 2 tiles.

    When num_doors>=2, extra doors are placed at natural chokepoints via
    _place_doors_at_chokepoints.
    """
    if rng is None:
        rng = random.Random()

    if height < 5 or width < 5:
        return None

    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}

    # Arm directions: each arm extends toward a grid edge
    h_dir = rng.choice([-1, 1])
    v_dir = rng.choice([-1, 1])

    if h_dir == -1:
        corner_col = rng.randint(3, width - 2)
    else:
        corner_col = rng.randint(1, width - 4)
    if v_dir == -1:
        corner_row = rng.randint(3, height - 2)
    else:
        corner_row = rng.randint(1, height - 4)

    # Build horizontal arm (along corner_row, toward edge)
    wall_tiles: list[Pos] = []
    c = corner_col
    while 0 <= c < width:
        wall_tiles.append(Pos(corner_row, c))
        c += h_dir

    # Build vertical arm (along corner_col, away from corner row toward edge)
    r = corner_row + v_dir
    while 0 <= r < height:
        wall_tiles.append(Pos(r, corner_col))
        r += v_dir

    wall_set = set(wall_tiles)
    corner_pos = Pos(corner_row, corner_col)

    # Find gap candidates: wall tiles accessible from both sides
    gap_candidates: list[Pos] = []
    for p in wall_tiles:
        if p == corner_pos:
            continue
        if p.row == corner_row:
            above = Pos(p.row - 1, p.col)
            below = Pos(p.row + 1, p.col)
            if (in_bounds(above, height, width) and in_bounds(below, height, width)
                    and above not in wall_set and below not in wall_set):
                gap_candidates.append(p)
        else:
            left = Pos(p.row, p.col - 1)
            right = Pos(p.row, p.col + 1)
            if (in_bounds(left, height, width) and in_bounds(right, height, width)
                    and left not in wall_set and right not in wall_set):
                gap_candidates.append(p)

    if not gap_candidates:
        return None

    # Single gap = D1
    door_pos = rng.choice(gap_candidates)

    # Build walls (all L-wall tiles except the gap)
    walls: set[Pos] = set()
    for p in wall_tiles:
        if p != door_pos:
            walls.add(p)

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Determine inside region (corner quadrant enclosed by L + grid edges)
    if h_dir == -1 and v_dir == -1:
        inside_fn = lambda p: p.row < corner_row and p.col < corner_col
    elif h_dir == 1 and v_dir == 1:
        inside_fn = lambda p: p.row > corner_row and p.col > corner_col
    elif h_dir == -1 and v_dir == 1:
        inside_fn = lambda p: p.row > corner_row and p.col < corner_col
    else:
        inside_fn = lambda p: p.row < corner_row and p.col > corner_col

    inside = [p for p in floor if inside_fn(p) and p != door_pos]
    outside = [p for p in floor if not inside_fn(p) and p != door_pos
               and p not in wall_set]

    if len(inside) < num_plates + 1:
        return None
    if not outside:
        return None

    rng.shuffle(inside)
    rng.shuffle(outside)
    start = inside[0]
    exit_pos = outside[0]

    # Plates inside (start side)
    plate_cands = [p for p in inside if p != start]
    plate_positions = _select_plate_positions(plate_cands, num_plates, start, rng)
    if plate_positions is None:
        return None

    protected = {start, exit_pos, door_pos} | set(plate_positions)
    _add_extra_walls(walls, all_positions, protected, rng, max_extra=rng.randint(0, 3))

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 9: Island
# ---------------------------------------------------------------------------

def generate_island_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with a central wall block (island) and a ring around it.

    Central 2x2 or 2x3 wall block. Ring of floor tiles around it. Plates on
    opposite ring sides. Door on the ring. Ring paint blocks backtrack direction.
    """
    if rng is None:
        rng = random.Random()

    if height < 5 or width < 5:
        return None

    # Central block position
    bh = rng.choice([2, 3])
    bw = rng.choice([2, 3])
    br = (height - bh) // 2
    bc = (width - bw) // 2

    walls: set[Pos] = set()
    for r in range(br, br + bh):
        for c in range(bc, bc + bw):
            walls.add(Pos(r, c))

    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}
    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Ring = floor tiles adjacent to the block
    ring: list[Pos] = []
    for p in floor:
        for d in Direction:
            neighbor = add_pos(p, d)
            if neighbor in walls:
                ring.append(p)
                break

    if len(ring) < 4:
        return None

    # Start and exit on opposite sides of ring (randomised offset)
    offset = rng.randint(0, len(ring) - 1)
    start = ring[offset]
    exit_pos = ring[(offset + len(ring) // 2) % len(ring)]

    # --- Exit pocket: wall off all exit neighbours except one (the door) ---
    exit_neighbors = [
        add_pos(exit_pos, d) for d in Direction
        if in_bounds(add_pos(exit_pos, d), width, height)
    ]
    free_neighbors = [
        n for n in exit_neighbors
        if n not in walls and n != start
    ]
    if not free_neighbors:
        return None
    rng.shuffle(free_neighbors)
    door_pos = free_neighbors[0]
    for n in free_neighbors[1:]:
        walls.add(n)

    # Re-derive floor after pocket walls and check connectivity
    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Plates on ring and non-ring floor
    plate_cands = [
        p for p in floor
        if p != start and p != exit_pos and p != door_pos
    ]
    plate_positions = _select_plate_positions(plate_cands, num_plates, start, rng)
    if plate_positions is None:
        return None

    protected = {start, exit_pos, door_pos} | set(plate_positions) | set(ring)
    _add_extra_walls(walls, all_positions, protected, rng, max_extra=rng.randint(0, 2))

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 10: Corridor
# ---------------------------------------------------------------------------

def generate_corridor_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with multiple short wall segments creating branching paths.

    3-5 short wall segments (length 2-3, random orientation), each validated
    for connectivity. Creates junctions. Exit pocket ensures the door is
    structurally required. Wrong branch paints junction tile.
    """
    if rng is None:
        rng = random.Random()

    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}
    walls: set[Pos] = set()

    grid_size = max(width, height)
    enhanced = grid_size >= 6 and rng.random() < 0.5
    if enhanced:
        n_segments = rng.randint(3 + grid_size // 3, 5 + grid_size // 3)
        seg_len_max = min(4, grid_size // 2)
    else:
        n_segments = rng.randint(3, 5)
        seg_len_max = 3
    for _ in range(n_segments * 5):  # attempt multiple times
        if len(walls) // 2 >= n_segments:
            break

        seg_len = rng.randint(2, seg_len_max)
        orientation = rng.choice(["h", "v"])
        if orientation == "h":
            r = rng.randint(0, height - 1)
            c = rng.randint(0, width - seg_len)
            segment = {Pos(r, c + i) for i in range(seg_len)}
        else:
            r = rng.randint(0, height - seg_len)
            c = rng.randint(0, width - 1)
            segment = {Pos(r + i, c) for i in range(seg_len)}

        test_walls = walls | segment
        test_floor = all_positions - test_walls
        if test_floor and _is_connected(test_floor) and len(test_floor) >= num_plates + 3:
            walls |= segment

    floor = all_positions - walls
    if not _is_connected(floor) or len(floor) < num_plates + 3:
        return None

    floor_list = list(floor)
    rng.shuffle(floor_list)

    start = floor_list[0]
    exit_pos = floor_list[-1]

    # --- Exit pocket: wall off all exit neighbours except one (the door) ---
    exit_neighbors = [
        add_pos(exit_pos, d) for d in Direction
        if in_bounds(add_pos(exit_pos, d), height, width)
    ]
    free_neighbors = [
        n for n in exit_neighbors
        if n not in walls and n != start
    ]
    if not free_neighbors:
        return None
    rng.shuffle(free_neighbors)
    door_pos = free_neighbors[0]
    for n in free_neighbors[1:]:
        walls.add(n)

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    plate_cands = [
        p for p in floor
        if p != start and p != exit_pos and p != door_pos
    ]
    plate_positions = _select_plate_positions(plate_cands, num_plates, start, rng)
    if plate_positions is None:
        return None

    protected = {start, exit_pos, door_pos} | set(plate_positions)
    extra_budget = rng.randint(1, grid_size - 3) if enhanced else rng.randint(0, 2)
    _add_extra_walls(walls, all_positions, protected, rng, max_extra=extra_budget)

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 11: Moat
# ---------------------------------------------------------------------------

def generate_moat_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with U-shaped moat walls and an exit pocket.

    U-shaped wall structure creates a moat-like obstacle in part of the grid.
    Exit pocket ensures the agent must pass through the door. Plates outside
    the moat area. The moat walls constrain routing while the exit pocket
    guarantees door usage.
    """
    if rng is None:
        rng = random.Random()

    if height < 5 or width < 5:
        return None

    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}

    # Which edge the U opens toward
    open_side = rng.choice(["top", "bottom", "left", "right"])

    # U dimensions (clamped to grid size)
    u_depth = rng.randint(2, min(3, max(height, width) // 2))
    max_half_span = min(2, (min(width, height) - 3) // 2)
    if max_half_span < 1:
        return None
    u_half_span = rng.randint(1, max_half_span)

    walls: set[Pos] = set()

    if open_side == "top":
        base_row = height - 1
        center_col = rng.randint(u_half_span + 1, width - u_half_span - 2)
        # Left arm, right arm, base
        for i in range(u_depth):
            walls.add(Pos(base_row - i, center_col - u_half_span - 1))
            walls.add(Pos(base_row - i, center_col + u_half_span + 1))
        for c in range(center_col - u_half_span - 1, center_col + u_half_span + 2):
            if 0 <= c < width:
                walls.add(Pos(base_row, c))
    elif open_side == "bottom":
        base_row = 0
        center_col = rng.randint(u_half_span + 1, width - u_half_span - 2)
        for i in range(u_depth):
            walls.add(Pos(base_row + i, center_col - u_half_span - 1))
            walls.add(Pos(base_row + i, center_col + u_half_span + 1))
        for c in range(center_col - u_half_span - 1, center_col + u_half_span + 2):
            if 0 <= c < width:
                walls.add(Pos(base_row, c))
    elif open_side == "left":
        base_col = width - 1
        center_row = rng.randint(u_half_span + 1, height - u_half_span - 2)
        for i in range(u_depth):
            walls.add(Pos(center_row - u_half_span - 1, base_col - i))
            walls.add(Pos(center_row + u_half_span + 1, base_col - i))
        for r in range(center_row - u_half_span - 1, center_row + u_half_span + 2):
            if 0 <= r < height:
                walls.add(Pos(r, base_col))
    else:  # right
        base_col = 0
        center_row = rng.randint(u_half_span + 1, height - u_half_span - 2)
        for i in range(u_depth):
            walls.add(Pos(center_row - u_half_span - 1, base_col + i))
            walls.add(Pos(center_row + u_half_span + 1, base_col + i))
        for r in range(center_row - u_half_span - 1, center_row + u_half_span + 2):
            if 0 <= r < height:
                walls.add(Pos(r, base_col))

    # Clip walls to grid bounds
    walls = {p for p in walls if 0 <= p.row < height and 0 <= p.col < width}

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Start far from the moat, exit near the moat
    floor_list = list(floor)
    rng.shuffle(floor_list)

    # Place start away from the moat area
    if open_side in ("top", "bottom"):
        moat_row = 0 if open_side == "bottom" else height - 1
        start_cands = [p for p in floor_list if abs(p.row - moat_row) >= height // 2]
    else:
        moat_col = 0 if open_side == "right" else width - 1
        start_cands = [p for p in floor_list if abs(p.col - moat_col) >= width // 2]
    if not start_cands:
        start_cands = floor_list
    start = rng.choice(start_cands)

    # Exit: pick from floor tiles far from start
    exit_candidates = [
        p for p in floor_list
        if p != start and manhattan_distance(p, start) >= max(width, height) // 2
    ]
    if not exit_candidates:
        exit_candidates = [p for p in floor_list if p != start]
    exit_pos = rng.choice(exit_candidates)

    # --- Exit pocket: wall off all exit neighbours except one (the door) ---
    exit_neighbors = [
        add_pos(exit_pos, d) for d in Direction
        if in_bounds(add_pos(exit_pos, d), height, width)
    ]
    free_neighbors = [
        n for n in exit_neighbors
        if n not in walls and n != start
    ]
    if not free_neighbors:
        return None
    rng.shuffle(free_neighbors)
    door_pos = free_neighbors[0]
    for n in free_neighbors[1:]:
        walls.add(n)

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Plates in the main area (not exit, not door)
    plate_cands = [
        p for p in floor
        if p != start and p != exit_pos and p != door_pos
    ]
    plate_positions = _select_plate_positions(plate_cands, num_plates, start, rng)
    if plate_positions is None:
        return None

    protected = {start, exit_pos, door_pos} | set(plate_positions)
    _add_extra_walls(walls, all_positions, protected, rng, max_extra=rng.randint(0, 3))

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 12: Pinch
# ---------------------------------------------------------------------------

def generate_pinch_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with two wall clusters from opposite edges creating a narrow passage.

    Clusters extend from top-left and bottom-right areas, narrowing the
    traversable space. Exit pocket ensures the door is structurally required.
    Plates on both sides of the pinch.
    """
    if rng is None:
        rng = random.Random()

    if height < 4 or width < 4:
        return None

    walls: set[Pos] = set()
    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}

    # Cluster 1: from top-left area
    c1_size = rng.randint(2, 4)
    c1_candidates = [Pos(r, c) for r in range(height // 2) for c in range(width // 2)]
    rng.shuffle(c1_candidates)
    for pos in c1_candidates[:c1_size]:
        test_walls = walls | {pos}
        test_floor = all_positions - test_walls
        if test_floor and _is_connected(test_floor):
            walls.add(pos)

    # Cluster 2: from bottom-right area
    c2_size = rng.randint(2, 4)
    c2_candidates = [
        Pos(r, c) for r in range(height // 2, height) for c in range(width // 2, width)
    ]
    rng.shuffle(c2_candidates)
    for pos in c2_candidates[:c2_size]:
        test_walls = walls | {pos}
        test_floor = all_positions - test_walls
        if test_floor and _is_connected(test_floor):
            walls.add(pos)

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Start in bottom-left, exit in top-right (or fallback)
    tr_floor = [p for p in floor if p.row < height // 2 and p.col >= width // 2]
    bl_floor = [p for p in floor if p.row >= height // 2 and p.col < width // 2]
    if not tr_floor:
        tr_floor = [p for p in floor if p.row < height // 2]
    if not bl_floor:
        bl_floor = [p for p in floor if p.row >= height // 2]
    if not tr_floor or not bl_floor:
        return None

    start = rng.choice(bl_floor)
    exit_pos = rng.choice(tr_floor)

    # --- Exit pocket: wall off all exit neighbours except one (the door) ---
    exit_neighbors = [
        add_pos(exit_pos, d) for d in Direction
        if in_bounds(add_pos(exit_pos, d), height, width)
    ]
    free_neighbors = [
        n for n in exit_neighbors
        if n not in walls and n != start
    ]
    if not free_neighbors:
        return None
    rng.shuffle(free_neighbors)
    door_pos = free_neighbors[0]
    for n in free_neighbors[1:]:
        walls.add(n)

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Plates on both sides of pinch
    plate_cands = [p for p in floor if p != start and p != exit_pos and p != door_pos]
    plate_positions = _select_plate_positions(plate_cands, num_plates, start, rng)
    if plate_positions is None:
        return None

    protected = {start, exit_pos, door_pos} | set(plate_positions)
    _add_extra_walls(walls, all_positions, protected, rng, max_extra=rng.randint(0, 1))

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 13: Zigzag
# ---------------------------------------------------------------------------

def generate_zigzag_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 1,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with alternating wall stubs from left and right edges.

    Creates a comb-teeth / zigzag corridor pattern. Door near exit. Plates
    along the zigzag path. Forces alternating direction changes that paint
    critical transit tiles.
    """
    if rng is None:
        rng = random.Random()

    if height < 4 or width < 4:
        return None

    walls: set[Pos] = set()
    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}

    # Alternating stubs on interior rows only (skip top/bottom edges to
    # leave space for start/exit areas and avoid over-constraining 5x5).
    grid_size = max(width, height)
    enhanced = grid_size >= 6 and rng.random() < 0.5
    if enhanced:
        min_stub = 2
        stub_len = rng.randint(min_stub, max(min_stub, width * 2 // 3))
    else:
        min_stub = 1
        stub_len = rng.randint(1, max(1, width // 2))
    for r in range(1, height - 1):
        if r % 2 == 1:
            # Stub from left edge
            actual_len = rng.randint(min_stub, min(stub_len, width - 2))
            for c in range(actual_len):
                walls.add(Pos(r, c))
        else:
            # Stub from right edge
            actual_len = rng.randint(min_stub, min(stub_len, width - 2))
            for c in range(width - actual_len, width):
                walls.add(Pos(r, c))

    floor = all_positions - walls
    if not _is_connected(floor):
        return None
    if len(floor) < num_plates + 3:
        return None

    # Start at top, exit at bottom (with coin flip)
    top_floor = [p for p in floor if p.row == 0]
    bot_floor = [p for p in floor if p.row == height - 1]
    if not top_floor:
        top_floor = [p for p in floor if p.row <= 1]
    if not bot_floor:
        bot_floor = [p for p in floor if p.row >= height - 2]
    if not top_floor or not bot_floor:
        return None

    # Coin flip: randomise which end is start vs exit
    flip = rng.random() < 0.5
    if flip:
        start = rng.choice(bot_floor)
        exit_pos = rng.choice(top_floor)
    else:
        start = rng.choice(top_floor)
        exit_pos = rng.choice(bot_floor)

    # --- Exit pocket: wall off all exit neighbours except one (the door) ---
    exit_neighbors = [
        add_pos(exit_pos, d) for d in Direction
        if in_bounds(add_pos(exit_pos, d), height, width)
    ]
    free_neighbors = [
        n for n in exit_neighbors
        if n not in walls and n != start
    ]
    if not free_neighbors:
        return None
    rng.shuffle(free_neighbors)
    door_pos = free_neighbors[0]
    for n in free_neighbors[1:]:
        walls.add(n)

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Plates along the zigzag path
    plate_cands = [
        p for p in floor
        if p != start and p != exit_pos and p != door_pos
    ]
    plate_positions = _select_plate_positions(plate_cands, num_plates, start, rng)
    if plate_positions is None:
        return None

    protected = {start, exit_pos, door_pos} | set(plate_positions)
    extra_budget = rng.randint(1, max(1, grid_size - 4)) if enhanced else rng.randint(0, 1)
    _add_extra_walls(walls, all_positions, protected, rng, max_extra=extra_budget)

    if num_doors >= 2:
        multi = _place_doors_at_chokepoints(walls, all_positions, protected, plate_positions, door_pos, num_doors, rng, width, height, start, exit_pos)
        if multi is not None:
            plates, doors, walls = multi
            return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)

    plates, doors = _make_plates_and_door(plate_positions, door_pos)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Deadlock discovery (forward sweep)
# ---------------------------------------------------------------------------

def discover_deadlocks(
    grid: MosaicGrid,
    initial_state: MosaicState,
    max_depth: int = 15,
    max_states: int = 50_000,
    min_deadlocks: int = 0,
) -> list[DeadlockEntry]:
    """Forward sweep from initial state to discover Walking Dead states.

    Uses monotonicity optimization: children of unsolvable states are also
    unsolvable (painted tiles only grow), so we skip oracle calls for them.

    If min_deadlocks > 0, stops early once enough deadlocks are found
    (avoids exhaustive exploration on large grids).
    """
    catalog: list[DeadlockEntry] = []
    queue: deque[tuple[MosaicState, list[Direction], bool]] = deque()
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

            # Monotonicity: once unsolvable, always unsolvable
            if not parent_solvable:
                next_solvable = False
            else:
                next_solvable = is_solvable(grid, next_state)

            # Detect solvability flip
            if parent_solvable and not next_solvable:
                # Already know next_state is unsolvable (line above).
                # is_walking_dead only adds a has_legal check — do that directly.
                if len(get_legal_actions(grid, next_state)) > 0:
                    catalog.append(DeadlockEntry(
                        path=new_path,
                        fatal_step=len(new_path) - 1,
                        deadlock_state=next_state,
                        pre_deadlock_state=state,
                    ))
                    if min_deadlocks > 0 and len(catalog) >= min_deadlocks:
                        return catalog

            # Continue from solvable states or Walking Dead states
            if next_solvable or (not next_solvable and len(get_legal_actions(grid, next_state)) > 0):
                queue.append((next_state, new_path, next_solvable))

    return catalog


def count_reachable_dead_states(
    grid: MosaicGrid,
    state: MosaicState,
    max_states: int = 200,
) -> int:
    """BFS from a Walking Dead state to count reachable states.

    Measures the k-horizon: how many unique states the agent can still
    reach after entering a deadlock. Higher values mean longer Walking Dead
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


def filter_deadlock_catalog(
    catalog: list[DeadlockEntry],
    min_k_horizon: int = 3,
) -> list[DeadlockEntry]:
    """Filter catalog entries by k-horizon tier.

    Returns entries with k_horizon >= min_k_horizon.
    """
    return [e for e in catalog if e.k_horizon >= min_k_horizon]


# ---------------------------------------------------------------------------
# Greedy simulation
# ---------------------------------------------------------------------------

def _move_toward(
    grid: MosaicGrid,
    state: MosaicState,
    target: Pos,
    legal: list[Direction],
) -> Optional[Direction]:
    """Pick the legal move that minimizes manhattan distance to target."""
    if not legal:
        return None
    return min(legal, key=lambda a: manhattan_distance(add_pos(state.agent_pos, a), target))


def _bfs_next_step(
    grid: MosaicGrid,
    state: MosaicState,
    target: Pos,
    direction_order: tuple[Direction, ...] | None = None,
) -> Optional[Direction]:
    """BFS shortest path from agent to target; return the first step.

    Respects current painted tiles, walls, and door state.
    direction_order controls tiebreaking (default: canonical Direction order).
    """
    dirs = direction_order or tuple(Direction)
    start = state.agent_pos
    if start == target:
        return None
    queue: deque[tuple[Pos, Direction]] = deque()
    visited: set[Pos] = {start}
    for d in dirs:
        nbr = add_pos(start, d)
        if is_passable(grid, state, nbr):
            if nbr == target:
                return d
            queue.append((nbr, d))
            visited.add(nbr)
    while queue:
        pos, first_dir = queue.popleft()
        for d in dirs:
            nbr = add_pos(pos, d)
            if nbr in visited:
                continue
            if not is_passable(grid, state, nbr):
                continue
            if nbr == target:
                return first_dir
            visited.add(nbr)
            queue.append((nbr, first_dir))
    return None  # unreachable


def simulate_greedy_nearest_plate(
    grid: MosaicGrid,
    state: MosaicState,
    max_steps: int = 40,
) -> tuple[list[Direction], MosaicState]:
    """Strategy 1: always move toward the nearest unactivated plate,
    then toward the exit once all plates are activated.

    Returns (trace, final_state).
    """
    trace: list[Direction] = []
    visited_states: set = set()
    required_mask = 0
    for door in grid.doors.values():
        for pid in door.required_plate_ids:
            required_mask |= 1 << grid._pid_to_bit[pid]  # type: ignore[attr-defined]

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break  # loop detected
        visited_states.add(h)

        # Determine target: nearest unactivated plate, or exit
        remaining_mask = required_mask & ~state.activated_plates
        if remaining_mask:
            remaining = decode_activated(remaining_mask, grid._bit_to_pid)  # type: ignore[attr-defined]
            targets = [grid.plates[pid].pos for pid in remaining]
            target = min(targets, key=lambda p: manhattan_distance(state.agent_pos, p))
        else:
            target = grid.exit_pos

        legal = get_legal_actions(grid, state)
        action = _move_toward(grid, state, target, legal)
        if action is None:
            break
        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


def simulate_greedy_toward_exit(
    grid: MosaicGrid,
    state: MosaicState,
    max_steps: int = 40,
) -> tuple[list[Direction], MosaicState]:
    """Strategy 2: minimize manhattan to exit at every step.

    Only visits plates if they happen to be on the path. Tests whether
    beelining to exit works when plates are near the route.
    """
    trace: list[Direction] = []
    visited_states: set = set()

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break  # loop detected
        visited_states.add(h)

        legal = get_legal_actions(grid, state)
        action = _move_toward(grid, state, grid.exit_pos, legal)
        if action is None:
            break
        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


def simulate_greedy_visit_all_then_exit(
    grid: MosaicGrid,
    state: MosaicState,
    max_steps: int = 40,
) -> tuple[list[Direction], MosaicState]:
    """Strategy 3: visit all plates in nearest-first order (ignoring door state),
    then beeline to exit.

    Tests whether the 'do everything then leave' strategy paints the agent
    into a corner.
    """
    trace: list[Direction] = []
    visited_states: set = set()
    required_mask = 0
    for door in grid.doors.values():
        for pid in door.required_plate_ids:
            required_mask |= 1 << grid._pid_to_bit[pid]  # type: ignore[attr-defined]

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break  # loop detected
        visited_states.add(h)

        # Visit all plates first (nearest-first), then exit
        remaining_mask = required_mask & ~state.activated_plates
        if remaining_mask:
            remaining = decode_activated(remaining_mask, grid._bit_to_pid)  # type: ignore[attr-defined]
            targets = [grid.plates[pid].pos for pid in remaining]
            target = min(targets, key=lambda p: manhattan_distance(state.agent_pos, p))
        else:
            target = grid.exit_pos

        legal = get_legal_actions(grid, state)
        action = _move_toward(grid, state, target, legal)
        if action is None:
            break
        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


# Legacy wrapper for backward compatibility
def simulate_greedy(
    grid: MosaicGrid,
    state: MosaicState,
    max_steps: int = 40,
) -> tuple[list[Direction], MosaicState]:
    """Backward-compatible wrapper — calls simulate_greedy_nearest_plate."""
    return simulate_greedy_nearest_plate(grid, state, max_steps)


def _run_bfs_greedy(
    grid: MosaicGrid,
    initial_state: MosaicState,
    direction_order: tuple[Direction, ...],
    max_steps: int = 40,
) -> tuple[list[Direction], MosaicState]:
    """Run BFS-nearest-plate greedy with a specific direction tiebreaking order."""
    trace: list[Direction] = []
    state = initial_state
    visited_states: set = set()
    required_mask = 0
    for door in grid.doors.values():
        for pid in door.required_plate_ids:
            required_mask |= 1 << grid._pid_to_bit[pid]  # type: ignore[attr-defined]

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break
        visited_states.add(h)

        remaining_mask = required_mask & ~state.activated_plates
        if remaining_mask:
            remaining = decode_activated(remaining_mask, grid._bit_to_pid)  # type: ignore[attr-defined]
            targets = [grid.plates[pid].pos for pid in remaining]
            target = min(targets, key=lambda p: manhattan_distance(state.agent_pos, p))
        else:
            target = grid.exit_pos

        action = _bfs_next_step(grid, state, target, direction_order)
        if action is None:
            break
        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


# 8 direction orderings that cover diverse tiebreaking choices
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


def simulate_greedy_bfs_nearest_plate(
    grid: MosaicGrid,
    state: MosaicState,
    max_steps: int = 40,
) -> tuple[list[Direction], MosaicState]:
    """Strategy 4: BFS pathfind to nearest plate, then to exit.

    Tries 8 direction tiebreaking orders and returns the BEST result
    (solved, or longest trace). This catches puzzles that look trivially
    solvable to a human regardless of which equally-short BFS path they
    happen to pick.
    """
    best_trace: list[Direction] = []
    best_state = state
    for order in _DIRECTION_ORDERINGS:
        trace, final = _run_bfs_greedy(grid, state, order, max_steps)
        if is_solved(grid, final):
            return trace, final
        if len(trace) > len(best_trace):
            best_trace, best_state = trace, final
    return best_trace, best_state


# ---------------------------------------------------------------------------
# Smart greedy helpers (painting-aware)
# ---------------------------------------------------------------------------

def _bfs_reachable(grid: MosaicGrid, state: MosaicState, target: Pos) -> bool:
    """BFS from agent_pos to target, respecting painted tiles/walls/doors.

    Returns True if target is reachable, False otherwise.
    """
    start = state.agent_pos
    if start == target:
        return True
    queue: deque[Pos] = deque([start])
    visited: set[Pos] = {start}
    while queue:
        pos = queue.popleft()
        for d in Direction:
            nbr = add_pos(pos, d)
            if nbr in visited:
                continue
            if not is_passable(grid, state, nbr):
                continue
            if nbr == target:
                return True
            visited.add(nbr)
            queue.append(nbr)
    return False


def _bfs_distance(grid: MosaicGrid, state: MosaicState, target: Pos) -> Optional[int]:
    """BFS distance from agent_pos to target, respecting painted tiles/walls/doors.

    Returns the shortest-path distance, or None if unreachable.
    """
    start = state.agent_pos
    if start == target:
        return 0
    queue: deque[tuple[Pos, int]] = deque([(start, 0)])
    visited: set[Pos] = {start}
    while queue:
        pos, dist = queue.popleft()
        for d in Direction:
            nbr = add_pos(pos, d)
            if nbr in visited:
                continue
            if not is_passable(grid, state, nbr):
                continue
            if nbr == target:
                return dist + 1
            visited.add(nbr)
            queue.append((nbr, dist + 1))
    return None


# ---------------------------------------------------------------------------
# Smart greedy strategies (painting-aware)
# ---------------------------------------------------------------------------

def simulate_reachability_guard(
    grid: MosaicGrid,
    state: MosaicState,
    max_steps: int = 40,
) -> tuple[list[Direction], MosaicState]:
    """Strategy 5: exit-reachability guard.

    Never make a move that disconnects the agent from the exit.
    Among safe moves, pick the one nearest to the current target plate
    (or exit if all plates activated).

    Fails on puzzles requiring crossing chokepoints that temporarily
    make the exit unreachable.
    """
    trace: list[Direction] = []
    visited_states: set = set()
    required_mask = 0
    for door in grid.doors.values():
        for pid in door.required_plate_ids:
            required_mask |= 1 << grid._pid_to_bit[pid]  # type: ignore[attr-defined]

    for _ in range(max_steps):
        if is_solved(grid, state):
            break
        h = state.to_hashable()
        if h in visited_states:
            break
        visited_states.add(h)

        # Determine target
        remaining_mask = required_mask & ~state.activated_plates
        if remaining_mask:
            remaining = decode_activated(remaining_mask, grid._bit_to_pid)  # type: ignore[attr-defined]
            targets = [grid.plates[pid].pos for pid in remaining]
            target = min(targets, key=lambda p: manhattan_distance(state.agent_pos, p))
        else:
            target = grid.exit_pos

        legal = get_legal_actions(grid, state)
        if not legal:
            break

        # Filter to moves that keep exit reachable
        safe_moves: list[Direction] = []
        for action in legal:
            hypo_state = apply_action(grid, state, action)
            if _bfs_reachable(grid, hypo_state, grid.exit_pos):
                safe_moves.append(action)

        # Pick from safe moves; fall back to any legal move if none are safe
        pool = safe_moves if safe_moves else legal
        action = min(pool, key=lambda a: manhattan_distance(add_pos(state.agent_pos, a), target))
        trace.append(action)
        state = apply_action(grid, state, action)

    return trace, state


def simulate_farthest_plate_first(
    grid: MosaicGrid,
    state: MosaicState,
    max_steps: int = 40,
) -> tuple[list[Direction], MosaicState]:
    """Strategy 6: farthest-plate-first with BFS pathfinding.

    Target the farthest unactivated required plate by BFS distance,
    then BFS to exit once all plates are activated.

    Fails because farthest-first often paints tiles needed for later
    traversal, and some plates are behind doors requiring closer plates.
    """
    trace: list[Direction] = []
    visited_states: set = set()
    required_mask = 0
    for door in grid.doors.values():
        for pid in door.required_plate_ids:
            required_mask |= 1 << grid._pid_to_bit[pid]  # type: ignore[attr-defined]

    best_trace: list[Direction] = []
    best_state = state
    for order in _DIRECTION_ORDERINGS:
        trace = []
        sim_state = state
        visited_states = set()

        for _ in range(max_steps):
            if is_solved(grid, sim_state):
                break
            h = sim_state.to_hashable()
            if h in visited_states:
                break
            visited_states.add(h)

            remaining_mask = required_mask & ~sim_state.activated_plates
            if remaining_mask:
                remaining = decode_activated(remaining_mask, grid._bit_to_pid)  # type: ignore[attr-defined]
                # Pick farthest plate by BFS distance
                best_dist = -1
                target = None
                for pid in remaining:
                    d = _bfs_distance(grid, sim_state, grid.plates[pid].pos)
                    if d is not None and d > best_dist:
                        best_dist = d
                        target = grid.plates[pid].pos
                if target is None:
                    # All remaining plates unreachable; try exit
                    target = grid.exit_pos
            else:
                target = grid.exit_pos

            action = _bfs_next_step(grid, sim_state, target, order)
            if action is None:
                break
            trace.append(action)
            sim_state = apply_action(grid, sim_state, action)

        if is_solved(grid, sim_state):
            return trace, sim_state
        if len(trace) > len(best_trace):
            best_trace, best_state = trace, sim_state

    return best_trace, best_state


# All greedy strategies that must fail for a puzzle to be non-greedy
GREEDY_STRATEGIES = [
    # Existing (cheap)
    ("nearest_plate_first", simulate_greedy_nearest_plate),
    ("toward_exit", simulate_greedy_toward_exit),
    ("visit_all_then_exit", simulate_greedy_visit_all_then_exit),
    ("bfs_nearest_plate", simulate_greedy_bfs_nearest_plate),
    # Smart strategies (painting-aware, most expensive last)
    ("reachability_guard", simulate_reachability_guard),
    ("farthest_plate_first", simulate_farthest_plate_first),
]


def all_greedy_strategies_fail(
    grid: MosaicGrid,
    initial_state: MosaicState,
) -> bool:
    """Check that NO greedy strategy solves the puzzle.

    Returns True only if ALL strategies fail (don't reach exit).
    """
    for name, strategy_fn in GREEDY_STRATEGIES:
        _trace, final_state = strategy_fn(grid, initial_state)
        if is_solved(grid, final_state):
            return False
    return True


# ---------------------------------------------------------------------------
# RQ instance construction helpers
# ---------------------------------------------------------------------------

def discover_oracle_adjacent_deadlocks(
    grid: MosaicGrid,
    initial_state: MosaicState,
    solution: list[Direction],
) -> list[DeadlockEntry]:
    """Find deadlocks one step off the oracle solution.

    Walk the oracle solution step-by-step. At each step *i*, try every
    legal action that differs from the oracle move. If any leads to a
    Walking Dead state, record it as an oracle-adjacent deadlock whose
    path is ``solution[:i] + [alt_action]``.

    Returns a list of DeadlockEntry with ``k_horizon`` already computed.
    """
    catalog: list[DeadlockEntry] = []
    state = initial_state

    for i, oracle_action in enumerate(solution):
        legal = get_legal_actions(grid, state)
        for alt in legal:
            if alt == oracle_action:
                continue
            alt_state = apply_action(grid, state, alt)
            if is_walking_dead(grid, alt_state):
                k_horizon = count_reachable_dead_states(grid, alt_state)
                catalog.append(DeadlockEntry(
                    path=solution[:i] + [alt],
                    fatal_step=i,
                    deadlock_state=alt_state,
                    pre_deadlock_state=state,
                    k_horizon=k_horizon,
                ))
        # Advance along the oracle path
        state = apply_action(grid, state, oracle_action)

    return catalog


def generate_hard_rq2_pair(
    grid: MosaicGrid,
    initial_state: MosaicState,
    solution: list[Direction],
    catalog_entry: DeadlockEntry,
) -> Optional[tuple[MosaicState, list[Direction], list[Direction]]]:
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
    grid: MosaicGrid,
    initial_state: MosaicState,
    solution: list[Direction],
    catalog_entry: DeadlockEntry,
    target_length: int = 18,
) -> Optional[tuple[list[Direction], int, int]]:
    """Generate a hard RQ3 trace from an oracle-adjacent deadlock.

    The trace follows the oracle for ``fatal_step`` actions, takes one
    wrong turn, then continues with random legal Walking Dead moves to
    reach ``target_length``.

    Returns (full_trace, fatal_step, cause_symptom_gap) or None.
    """
    i = catalog_entry.fatal_step
    bad_action = catalog_entry.path[i]

    trace_to_deadlock = list(solution[:i]) + [bad_action]
    continuation_length = target_length - len(trace_to_deadlock)
    if continuation_length < 3:
        return None

    # Walking Dead continuation: random legal moves
    dead_state = catalog_entry.deadlock_state
    continuation: list[Direction] = []
    state = dead_state
    rng = random.Random(42)
    for _ in range(continuation_length):
        legal = get_legal_actions(grid, state)
        if not legal:
            break
        action = rng.choice(legal)
        continuation.append(action)
        state = apply_action(grid, state, action)

    full_trace = trace_to_deadlock + continuation
    cause_symptom_gap = len(full_trace) - i - 1

    if cause_symptom_gap < 3:
        return None

    return full_trace, i, cause_symptom_gap


def generate_rq2_pair(
    grid: MosaicGrid,
    initial_state: MosaicState,
    catalog_entry: DeadlockEntry,
    k: int,
) -> Optional[tuple[MosaicState, list[Direction], list[Direction]]]:
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


def generate_rq3_instance(
    grid: MosaicGrid,
    initial_state: MosaicState,
    catalog_entry: DeadlockEntry,
    target_length: int = 18,
) -> Optional[tuple[list[Direction], int, int]]:
    """Generate an RQ3 trace with Walking Dead continuation.

    Returns (full_trace, fatal_step, cause_symptom_gap) or None.
    """
    path = catalog_entry.path
    T = catalog_entry.fatal_step
    dead_state = catalog_entry.deadlock_state

    trace_to_deadlock = path[: T + 1]
    continuation_length = target_length - len(trace_to_deadlock)
    if continuation_length < 3:
        return None

    # Walking Dead continuation: random legal moves
    continuation: list[Direction] = []
    state = dead_state
    rng = random.Random(42)
    for _ in range(continuation_length):
        legal = get_legal_actions(grid, state)
        if not legal:
            break
        action = rng.choice(legal)
        continuation.append(action)
        state = apply_action(grid, state, action)

    full_trace = trace_to_deadlock + continuation
    cause_symptom_gap = len(full_trace) - T - 1

    if cause_symptom_gap < 3:
        return None

    return full_trace, T, cause_symptom_gap


# ---------------------------------------------------------------------------
# Eval-readiness pre-check
# ---------------------------------------------------------------------------

def is_eval_ready(inst: GeneratedInstance, require_hard: bool = False) -> bool:
    """Return True if this instance has viable RQ2, RQ3 (and hard variants).

    Mirrors the exact dropping logic in build_eval._build_worker so that
    instances can be filtered during generation before any I/O or rendering.
    """
    grid = inst.grid
    s0 = inst.initial_state
    catalog = inst.deadlock_catalog

    # RQ2: at least one eligible entry (k_horizon>=3, fatal_step>=2) must
    # produce a valid pair.
    eligible_rq2 = [e for e in catalog if e.k_horizon >= 3 and e.fatal_step >= 2][:3]
    has_rq2 = any(
        generate_rq2_pair(grid, s0, e, e.fatal_step + 1) is not None
        for e in eligible_rq2
    )
    if not has_rq2:
        return False

    # RQ3: at least one tier-2 entry (k_horizon>=5) must produce a valid trace.
    tier2 = [e for e in catalog if e.k_horizon >= 5][:2]
    has_rq3 = any(generate_rq3_instance(grid, s0, e) is not None for e in tier2)
    if not has_rq3:
        return False

    if not require_hard:
        return True

    # Hard variants require oracle-adjacent deadlocks.
    oa_catalog = discover_oracle_adjacent_deadlocks(grid, s0, inst.solution)

    eligible_hard = [e for e in oa_catalog if e.k_horizon >= 3 and e.fatal_step >= 2][:3]
    has_hard_rq2 = any(
        generate_hard_rq2_pair(grid, s0, inst.solution, e) is not None
        for e in eligible_hard
    )
    if not has_hard_rq2:
        return False

    tier2_hard = [e for e in oa_catalog if e.k_horizon >= 5][:2]
    has_hard_rq3 = any(
        generate_hard_rq3_instance(grid, s0, inst.solution, e) is not None
        for e in tier2_hard
    )
    return has_hard_rq3


# ---------------------------------------------------------------------------
# Template 14: Gate Cascade (multi-door, staggering guaranteed)
# ---------------------------------------------------------------------------

def generate_gate_cascade_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 2,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with parallel vertical barriers spanning full column height.

    Each barrier has one gap = door.  Sections between barriers contain plates
    for the NEXT door, guaranteeing staggered unlocking.

    Layout (2 doors, 8x8):
        Section 0    |D1|  Section 1   |D2|  Section 2
        (start,      |  |  (D2 plates  |  |  (exit)
         D1 plates)  |  |   here)      |  |

    Min grid size: width >= 2 * num_doors + 3.
    """
    if rng is None:
        rng = random.Random()

    if num_doors < 2:
        return None
    num_doors = max(2, num_doors)
    min_width = 2 * num_doors + 3
    if width < min_width or height < 4:
        return None

    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}

    # Evenly space barrier columns across the grid
    n_sections = num_doors + 1
    section_width = width // n_sections
    if section_width < 2:
        return None

    barrier_cols: list[int] = []
    for i in range(1, n_sections):
        col = i * section_width
        if col <= 0 or col >= width - 1:
            continue
        barrier_cols.append(col)

    if len(barrier_cols) < num_doors:
        return None
    barrier_cols = barrier_cols[:num_doors]

    # Build barriers: full column of walls with 1 random gap = door
    walls: set[Pos] = set()
    door_positions: list[Pos] = []
    for i, col in enumerate(barrier_cols):
        # Alternate gap rows: even barriers use top half, odd use bottom half
        if i % 2 == 0:
            gap_row = rng.randint(0, max(0, height // 2 - 1))
        else:
            gap_row = rng.randint(height // 2, height - 1)
        door_positions.append(Pos(gap_row, col))
        for r in range(height):
            if r != gap_row:
                walls.add(Pos(r, col))

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Compute section boundaries (column ranges)
    section_bounds: list[tuple[int, int]] = []  # (col_start, col_end) exclusive
    prev_col = 0
    for col in barrier_cols:
        section_bounds.append((prev_col, col))
        prev_col = col + 1
    section_bounds.append((prev_col, width))

    # Start in section 0, exit in last section
    def _section_floor(sec_idx: int) -> list[Pos]:
        lo, hi = section_bounds[sec_idx]
        return [p for p in floor if lo <= p.col < hi]

    sec0_floor = _section_floor(0)
    sec_last_floor = _section_floor(len(section_bounds) - 1)
    if not sec0_floor or not sec_last_floor:
        return None

    # Coin flip: start left exit right, or reverse
    flip = rng.random() < 0.5
    if flip:
        start = rng.choice(sec_last_floor)
        exit_pos = rng.choice(sec0_floor)
        # Reverse door ordering so D1 is closest to start
        door_positions = list(reversed(door_positions))
        section_bounds = list(reversed(section_bounds))
    else:
        start = rng.choice(sec0_floor)
        exit_pos = rng.choice(sec_last_floor)

    # Protected set
    protected = {start, exit_pos} | set(door_positions)

    # Plates for D_k: selected from section k (the section before barrier k+1)
    # For door ordering after potential flip:
    # D1 plates in the section on start's side of D1
    # D2 plates in the section between D1 and D2
    groups: list[tuple[str, Pos, list[Pos]]] = []
    per_door = max(2, num_plates // num_doors)
    remainder = num_plates - per_door * num_doors

    for k in range(num_doors):
        door_id = f"D{k + 1}"
        door_p = door_positions[k]

        # D1 plates in section 0, D2 plates in section 1, etc.
        sec_floor = _section_floor(k)
        already_used: set[Pos] = set()
        for g in groups:
            already_used.update(g[2])
        plate_cands = [
            p for p in sec_floor
            if p not in protected and p not in already_used
        ]

        budget = per_door + (1 if k == 0 and remainder > 0 else 0)
        selected = _select_plates_in_zone(
            plate_cands, budget, start, rng,
            min_spacing=2, already_placed=already_used,
        )
        if selected is None:
            # Relax: try with spacing=1
            selected = _select_plates_in_zone(
                plate_cands, budget, start, rng,
                min_spacing=1, already_placed=already_used,
            )
        if selected is None:
            return None
        groups.append((door_id, door_p, selected))

    # Add random extra walls within sections for variety
    all_plate_set = set()
    for g in groups:
        all_plate_set.update(g[2])
    full_protected = protected | all_plate_set
    _add_extra_walls(walls, all_positions, full_protected, rng, max_extra=rng.randint(1, 4))

    plates, doors = _make_plates_and_doors(groups)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 15: Nested Rooms (multi-door, staggering guaranteed)
# ---------------------------------------------------------------------------

def generate_nested_rooms_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 2,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with a rectangular room inside a room.

    One outer rectangular wall ring (D1 gap) plus an interior horizontal or
    vertical partition wall (D2 gap) divides the grid into three zones:
    outer floor, upper/left inner, and lower/right inner.

    Layout (2 doors, 8x8):
        . . . . . . . .   outer region (D1 plates)
        # # D1# # # # #   ring wall
        # . . . . . . #
        # . . . . . . #   upper inner (D2 plates)
        # . . D2. . . #   interior partition wall
        # . . . . . . #   lower inner (exit)
        # # # # # # # #   ring wall
        . . . . . . . .   outer region

    Min grid size: 7x7.
    """
    if rng is None:
        rng = random.Random()

    if num_doors < 2:
        return None
    num_doors = 2  # nested rooms only supports 2 doors

    if width < 7 or height < 7:
        return None

    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}
    walls: set[Pos] = set()

    # --- Outer ring at inset=1 ---
    ring_inset = 1
    ring_top = ring_inset
    ring_bot = height - 1 - ring_inset
    ring_left = ring_inset
    ring_right = width - 1 - ring_inset

    if ring_top >= ring_bot or ring_left >= ring_right:
        return None

    # Interior space: rows ring_top+1..ring_bot-1, cols ring_left+1..ring_right-1
    inner_top = ring_top + 1
    inner_bot = ring_bot - 1
    inner_left = ring_left + 1
    inner_right = ring_right - 1
    inner_rows = inner_bot - inner_top + 1
    inner_cols = inner_right - inner_left + 1

    if inner_rows < 3 or inner_cols < 3:
        return None

    # --- Choose partition orientation first, then place ring gap correctly ---
    use_horizontal = rng.random() < 0.5

    if use_horizontal:
        # Horizontal partition splits inner into plate_half (top) and exit_half (bottom)
        wall_row = rng.randint(inner_top + 1, inner_bot - 1)
        partition_tiles = [Pos(wall_row, c) for c in range(inner_left, inner_right + 1)]
        gap_col = rng.randint(inner_left, inner_right)
        d2_pos = Pos(wall_row, gap_col)

        plate_half = {
            Pos(r, c) for r in range(inner_top, wall_row)
            for c in range(inner_left, inner_right + 1)
        }
        exit_half = {
            Pos(r, c) for r in range(wall_row + 1, inner_bot + 1)
            for c in range(inner_left, inner_right + 1)
        }

        # Ring gap MUST open into plate_half (top side), not exit_half.
        # Valid edges: top, or left/right with row in plate_half range.
        ring_gap_options = ["top"]
        # Also allow side gaps if they open into the plate_half rows
        if inner_top <= ring_top + 1 < wall_row:
            ring_gap_options.extend(["left", "right"])
        edge_choice = rng.choice(ring_gap_options)
        if edge_choice == "top":
            gap_col_d1 = rng.randint(ring_left + 1, ring_right - 1)
            d1_pos = Pos(ring_top, gap_col_d1)
        elif edge_choice == "left":
            gap_row_d1 = rng.randint(ring_top + 1, min(wall_row - 1, ring_bot - 1))
            d1_pos = Pos(gap_row_d1, ring_left)
        else:
            gap_row_d1 = rng.randint(ring_top + 1, min(wall_row - 1, ring_bot - 1))
            d1_pos = Pos(gap_row_d1, ring_right)
    else:
        # Vertical partition splits inner into plate_half (left) and exit_half (right)
        wall_col = rng.randint(inner_left + 1, inner_right - 1)
        partition_tiles = [Pos(r, wall_col) for r in range(inner_top, inner_bot + 1)]
        gap_row = rng.randint(inner_top, inner_bot)
        d2_pos = Pos(gap_row, wall_col)

        plate_half = {
            Pos(r, c) for r in range(inner_top, inner_bot + 1)
            for c in range(inner_left, wall_col)
        }
        exit_half = {
            Pos(r, c) for r in range(inner_top, inner_bot + 1)
            for c in range(wall_col + 1, inner_right + 1)
        }

        # Ring gap MUST open into plate_half (left side).
        ring_gap_options = ["left"]
        if inner_left <= ring_left + 1 < wall_col:
            ring_gap_options.extend(["top", "bottom"])
        edge_choice = rng.choice(ring_gap_options)
        if edge_choice == "left":
            gap_row_d1 = rng.randint(ring_top + 1, ring_bot - 1)
            d1_pos = Pos(gap_row_d1, ring_left)
        elif edge_choice == "top":
            gap_col_d1 = rng.randint(ring_left + 1, min(wall_col - 1, ring_right - 1))
            d1_pos = Pos(ring_top, gap_col_d1)
        else:
            gap_col_d1 = rng.randint(ring_left + 1, min(wall_col - 1, ring_right - 1))
            d1_pos = Pos(ring_bot, gap_col_d1)

    # Build ring walls
    ring_walls: list[Pos] = []
    for c in range(ring_left, ring_right + 1):
        ring_walls.append(Pos(ring_top, c))
        ring_walls.append(Pos(ring_bot, c))
    for r in range(ring_top + 1, ring_bot):
        ring_walls.append(Pos(r, ring_left))
        ring_walls.append(Pos(r, ring_right))

    for p in ring_walls:
        if p != d1_pos:
            walls.add(p)

    # Build partition walls
    for p in partition_tiles:
        if p != d2_pos:
            walls.add(p)

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Outer region: floor tiles outside the ring
    outer_region = {
        p for p in floor
        if p.row < ring_top or p.row > ring_bot
        or p.col < ring_left or p.col > ring_right
    }
    # Clean inner regions
    plate_half = plate_half - walls
    exit_half = exit_half - walls

    if not outer_region or not plate_half or not exit_half:
        return None

    # Coin flip: start outside → exit in exit_half, or reverse
    flip = rng.random() < 0.5
    if flip:
        # Start in exit_half, exit outside
        # Path: exit_half → D1(partition) → plate_half → D2(ring) → outer
        # Reverse: D1 = partition, D2 = ring
        start_region = exit_half
        exit_region = outer_region
        door_positions = [d2_pos, d1_pos]
        regions = [exit_half, plate_half]
    else:
        # Start outside, exit in exit_half
        # Path: outer → D1(ring) → plate_half → D2(partition) → exit_half
        start_region = outer_region
        exit_region = exit_half
        door_positions = [d1_pos, d2_pos]
        regions = [outer_region, plate_half]

    door_set = set(door_positions)
    start_cands = [p for p in start_region if p not in door_set]
    exit_cands = [p for p in exit_region if p not in door_set]
    if not start_cands or not exit_cands:
        return None

    start = rng.choice(start_cands)
    exit_pos = rng.choice(exit_cands)

    protected = {start, exit_pos} | door_set

    # Place plates: D1 in regions[0], D2 in regions[1]
    groups: list[tuple[str, Pos, list[Pos]]] = []
    per_door = max(2, num_plates // num_doors)
    already_used: set[Pos] = set()

    for k in range(num_doors):
        door_id = f"D{k + 1}"
        door_p = door_positions[k]
        region = regions[k]
        plate_cands = [
            p for p in region
            if p not in protected and p not in already_used
        ]
        budget = per_door if k > 0 else per_door + max(0, num_plates - per_door * num_doors)
        selected = _select_plates_in_zone(
            plate_cands, budget, start, rng,
            min_spacing=2, already_placed=already_used,
        )
        if selected is None:
            selected = _select_plates_in_zone(
                plate_cands, budget, start, rng,
                min_spacing=1, already_placed=already_used,
            )
        if selected is None:
            return None
        groups.append((door_id, door_p, selected))
        already_used.update(selected)

    # Extra walls for variety
    all_plate_set = set()
    for g in groups:
        all_plate_set.update(g[2])
    full_protected = protected | all_plate_set
    _add_extra_walls(walls, all_positions, full_protected, rng, max_extra=rng.randint(0, 3))

    plates, doors = _make_plates_and_doors(groups)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template 16: Serpentine Gauntlet (multi-door, staggering guaranteed)
# ---------------------------------------------------------------------------

def generate_serpentine_grid(
    width: int,
    height: int,
    num_plates: int,
    rng: random.Random | None = None,
    num_doors: int = 2,
) -> Optional[tuple[MosaicGrid, MosaicState]]:
    """Generate a grid with horizontal wall strips creating a zigzag corridor.

    Doors at the turns between strips.  Plates in each corridor segment.

    Layout (2 doors, 8x8):
        →→→→→→→D1↓
        ---------↓
        ↑D2←←←←←←
        ↑---------
        →→→→→→EXIT

    Min grid size: height >= 2 * num_doors + 3.
    """
    if rng is None:
        rng = random.Random()

    if num_doors < 2:
        return None
    num_doors = max(2, num_doors)
    min_height = 2 * num_doors + 3
    if height < min_height or width < 5:
        return None

    all_positions = {Pos(r, c) for r in range(height) for c in range(width)}
    walls: set[Pos] = set()
    door_positions: list[Pos] = []

    # Place num_doors horizontal walls, each with a gap at alternating ends
    n_bands = num_doors + 1  # corridor bands
    band_height = height // n_bands
    if band_height < 2:
        return None

    wall_rows: list[int] = []
    for i in range(1, n_bands):
        row = i * band_height
        if row >= height - 1:
            row = height - 2
        wall_rows.append(row)

    if len(wall_rows) < num_doors:
        return None
    wall_rows = wall_rows[:num_doors]

    # Build walls with alternating gaps
    for i, row in enumerate(wall_rows):
        if i % 2 == 0:
            # Gap at right end
            gap_col = width - 1
        else:
            # Gap at left end
            gap_col = 0
        door_positions.append(Pos(row, gap_col))
        for c in range(width):
            if c != gap_col:
                walls.add(Pos(row, c))

    floor = all_positions - walls
    if not _is_connected(floor):
        return None

    # Compute band regions (between consecutive wall rows)
    band_regions: list[set[Pos]] = []
    prev_row = 0
    for row in wall_rows:
        band = {p for p in floor if prev_row <= p.row < row}
        band_regions.append(band)
        prev_row = row + 1
    # Last band: after last wall row
    band_regions.append({p for p in floor if prev_row <= p.row < height})

    # Coin flip: start top-left or bottom
    flip = rng.random() < 0.5
    if flip:
        start_region = band_regions[-1]
        exit_region = band_regions[0]
        door_positions = list(reversed(door_positions))
        band_regions = list(reversed(band_regions))
    else:
        start_region = band_regions[0]
        exit_region = band_regions[-1]

    start_cands = [p for p in start_region if p not in set(door_positions)]
    exit_cands = [p for p in exit_region if p not in set(door_positions)]
    if not start_cands or not exit_cands:
        return None

    start = rng.choice(start_cands)
    exit_pos = rng.choice(exit_cands)

    protected = {start, exit_pos} | set(door_positions)

    # Plates for D_k in band k (the band the agent traverses before reaching D_k)
    groups: list[tuple[str, Pos, list[Pos]]] = []
    per_door = max(2, num_plates // num_doors)
    already_used: set[Pos] = set()

    for k in range(num_doors):
        door_id = f"D{k + 1}"
        door_p = door_positions[k]
        band = band_regions[k]
        plate_cands = [
            p for p in band
            if p not in protected and p not in already_used
        ]
        budget = per_door if k > 0 else per_door + max(0, num_plates - per_door * num_doors)
        selected = _select_plates_in_zone(
            plate_cands, budget, start, rng,
            min_spacing=2, already_placed=already_used,
        )
        if selected is None:
            selected = _select_plates_in_zone(
                plate_cands, budget, start, rng,
                min_spacing=1, already_placed=already_used,
            )
        if selected is None:
            return None
        groups.append((door_id, door_p, selected))
        already_used.update(selected)

    # Extra walls within bands for variety
    all_plate_set = set()
    for g in groups:
        all_plate_set.update(g[2])
    full_protected = protected | all_plate_set
    _add_extra_walls(walls, all_positions, full_protected, rng, max_extra=rng.randint(1, 3))

    plates, doors = _make_plates_and_doors(groups)
    return _build_grid_and_state(width, height, walls, plates, doors, start, exit_pos)


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, callable] = {
    "barrier": generate_barrier_grid,
    "bottleneck": generate_bottleneck_grid,
    "horizontal": generate_horizontal_grid,
    "spiral": generate_spiral_grid,
    "cross": generate_cross_grid,
    "peninsula": generate_peninsula_grid,
    "lshape": generate_lshape_grid,
    "island": generate_island_grid,
    "corridor": generate_corridor_grid,
    "moat": generate_moat_grid,
    "pinch": generate_pinch_grid,
    "zigzag": generate_zigzag_grid,
    "gate_cascade": generate_gate_cascade_grid,
    "nested_rooms": generate_nested_rooms_grid,
    "serpentine": generate_serpentine_grid,
}

TEMPLATE_NAMES: list[str] = list(TEMPLATES.keys())

# Templates that only produce multi-door layouts (skip for num_doors < 2)
MULTI_DOOR_ONLY_TEMPLATES: set[str] = {"gate_cascade", "nested_rooms", "serpentine"}


# ---------------------------------------------------------------------------
# Full instance generation pipeline
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Result of generation with rejection statistics."""

    instance: Optional[GeneratedInstance]
    stats: dict[str, int]
    total_attempts: int
    elapsed_seconds: float


def _generate_validated_inner(
    width: int = 5,
    height: int = 5,
    num_plates: int | None = None,
    seed: int = 0,
    max_retries: int = 400,
    solution_range: tuple[int, int] | None = None,
    min_deadlocks: int = 2,
    min_tier2: int = 1,
    template: str = "auto",
    num_doors: int = 1,
) -> tuple[Optional[GeneratedInstance], dict[str, int], int]:
    """Core generation loop with rejection tracking.

    Returns (instance_or_None, rejection_counts, total_attempts).
    """
    grid_size = max(width, height)

    # Cap doors at 2 (multi-door templates designed for 2 doors max).
    max_doors = 2
    num_doors = min(num_doors, max_doors)

    # Auto-scale plate count by grid size.
    if num_plates is None:
        if grid_size <= 5:
            num_plates = 2
        elif grid_size <= 6:
            num_plates = 3
        elif grid_size <= 7:
            num_plates = 4
        else:
            num_plates = 6

    # Multi-door requires at least 2 plates per door for adequate coverage
    min_plates_for_doors = max(3, 2 * num_doors)
    if num_doors >= 2 and num_plates < min_plates_for_doors:
        num_plates = min_plates_for_doors

    if solution_range is None:
        if grid_size <= 5:
            solution_range = (6, 20)
        elif grid_size == 6:
            solution_range = (8, 30)
        else:
            solution_range = (10, 40)

    rejections: dict[str, int] = {
        "grid_gen_failed": 0,
        "door_count": 0,
        "unsolvable": 0,
        "doors_not_required": 0,
        "solution_range": 0,
        "greedy_passes": 0,
        "insufficient_deadlocks": 0,
        "insufficient_tier2": 0,
    }

    for attempt in range(max_retries):
        rng = random.Random(seed * max_retries + attempt)

        if template == "auto":
            tname = TEMPLATE_NAMES[attempt % len(TEMPLATE_NAMES)]
            # Skip multi-door-only templates for single-door requests
            if num_doors < 2 and tname in MULTI_DOOR_ONLY_TEMPLATES:
                continue
        else:
            tname = template

        result = TEMPLATES[tname](width, height, num_plates, rng, num_doors=num_doors)
        if result is None:
            rejections["grid_gen_failed"] += 1
            continue

        grid, initial_state = result

        # Filter 0: Must have the requested number of doors
        if len(grid.doors) < num_doors:
            rejections["door_count"] += 1
            continue

        # Filter 1: Must be solvable
        solution = oracle_solve(grid, initial_state)
        if solution is None:
            rejections["unsolvable"] += 1
            continue

        # Filter 1b: Every door must be structurally required
        if not _all_doors_required(grid):
            rejections["doors_not_required"] += 1
            continue

        # Filter 2: Solution length in range
        if not (solution_range[0] <= len(solution) <= solution_range[1]):
            rejections["solution_range"] += 1
            continue

        # Filter 3: ALL greedy strategies must fail
        if not all_greedy_strategies_fail(grid, initial_state):
            rejections["greedy_passes"] += 1
            continue

        # Filter 4: Must have enough deadlocks with sufficient k-horizon.
        # Use early stopping to avoid exhaustive exploration on large grids.
        needed = min_deadlocks + min_tier2 + 5
        catalog = discover_deadlocks(
            grid, initial_state, max_depth=12, max_states=10_000,
            min_deadlocks=needed,
        )

        # Compute k-horizon lazily, counting tier hits as we go
        tier1_count = 0
        tier2_count = 0
        for entry in catalog:
            entry.k_horizon = count_reachable_dead_states(grid, entry.deadlock_state)
            if entry.k_horizon >= 3:
                tier1_count += 1
            if entry.k_horizon >= 5:
                tier2_count += 1
            if tier1_count >= min_deadlocks and tier2_count >= min_tier2:
                break

        if tier1_count < min_deadlocks:
            rejections["insufficient_deadlocks"] += 1
            continue
        if tier2_count < min_tier2:
            rejections["insufficient_tier2"] += 1
            continue

        max_k = max((e.k_horizon for e in catalog), default=0)
        instance = GeneratedInstance(
            grid=grid,
            initial_state=initial_state,
            solution=solution,
            deadlock_catalog=catalog,
            seed=seed + attempt,
            template_name=tname,
            grid_width=width,
            grid_height=height,
            num_plates=num_plates,
            num_doors=len(grid.doors),
            solution_length=len(solution),
            tier1_deadlocks=tier1_count,
            tier2_deadlocks=tier2_count,
            max_k_horizon=max_k,
            generation_attempts=attempt + 1,
        )
        return instance, rejections, attempt + 1

    return None, rejections, max_retries


def generate_validated_instance(
    width: int = 5,
    height: int = 5,
    num_plates: int | None = None,
    seed: int = 0,
    max_retries: int = 400,
    solution_range: tuple[int, int] | None = None,
    min_deadlocks: int = 2,
    min_tier2: int = 1,
    template: str = "auto",
    num_doors: int = 1,
) -> Optional[GeneratedInstance]:
    """Generate a validated Mosaic Grid instance.

    Args:
        template: Template name, or "auto" to cycle through all templates.
        num_plates: Number of plates.  If None, auto-scaled by grid size:
            5x5 → 2, 6x6+ → 3.
        solution_range: (min, max) solution length.  If None, defaults are
            chosen based on grid size and plate count.
        min_deadlocks: Minimum tier-1 deadlocks (k-horizon >= 3).
        min_tier2: Minimum tier-2 deadlocks (k-horizon >= 5).
        num_doors: Number of doors (1 or 2). Templates fall back to 1 if 2
            cannot be placed.

    Filters:
    - Solvable from start
    - Solution length in range
    - All greedy strategies fail (non-trivial)
    - Has >= min_deadlocks deadlocks with k-horizon >= 3 (tier 1, for RQ2)
    - Has >= min_tier2 deadlocks with k-horizon >= 5 (tier 2, for RQ3)
    """
    instance, _, _ = _generate_validated_inner(
        width=width, height=height, num_plates=num_plates, seed=seed,
        max_retries=max_retries, solution_range=solution_range,
        min_deadlocks=min_deadlocks, min_tier2=min_tier2,
        template=template, num_doors=num_doors,
    )
    return instance


def generate_validated_instance_with_stats(
    width: int = 5,
    height: int = 5,
    num_plates: int | None = None,
    seed: int = 0,
    max_retries: int = 400,
    solution_range: tuple[int, int] | None = None,
    min_deadlocks: int = 2,
    min_tier2: int = 1,
    template: str = "auto",
    num_doors: int = 1,
) -> GenerationResult:
    """Like generate_validated_instance but returns full stats."""
    import time
    t0 = time.monotonic()
    instance, stats, total = _generate_validated_inner(
        width=width, height=height, num_plates=num_plates, seed=seed,
        max_retries=max_retries, solution_range=solution_range,
        min_deadlocks=min_deadlocks, min_tier2=min_tier2,
        template=template, num_doors=num_doors,
    )
    elapsed = time.monotonic() - t0
    return GenerationResult(
        instance=instance, stats=stats,
        total_attempts=total, elapsed_seconds=elapsed,
    )
