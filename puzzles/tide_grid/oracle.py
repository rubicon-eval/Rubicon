"""Tide Grid oracle — A* search, solvability, deadlock detection."""

from __future__ import annotations

from collections import deque
from typing import Optional

from puzzles.common.types import manhattan_distance
from puzzles.common.astar import astar_search
from .engine import (
    TideGridGrid,
    TideGridState,
    TideGridAction,
    get_legal_actions,
    apply_action,
    is_solved,
)


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def heuristic(grid: TideGridGrid, state: TideGridState) -> int:
    """Admissible heuristic: nearest-neighbor TSP through uncollected beacons
    then to exit, using manhattan distances.

    Ignores walls and tidal phases, so always a lower bound on true cost.
    Also serves as oxygen pruning: if oxygen_remaining < heuristic, skip.
    """
    uncollected = [
        grid.beacons[b].pos
        for b in grid.beacons
        if b not in state.collected_beacons
    ]
    if not uncollected:
        return manhattan_distance(state.agent_pos, grid.exit_pos)

    # Greedy nearest-neighbor tour (admissible lower bound)
    pos = state.agent_pos
    total = 0
    remaining = list(uncollected)
    while remaining:
        nearest = min(remaining, key=lambda p: manhattan_distance(pos, p))
        total += manhattan_distance(pos, nearest)
        pos = nearest
        remaining.remove(nearest)
    total += manhattan_distance(pos, grid.exit_pos)
    return total


# ---------------------------------------------------------------------------
# A* helpers
# ---------------------------------------------------------------------------

def _get_successors(
    grid: TideGridGrid,
    state: TideGridState,
) -> list[tuple[TideGridAction, TideGridState]]:
    """Get all successor (action, state) pairs."""
    result = []
    for action in get_legal_actions(grid, state):
        next_state = apply_action(grid, state, action)
        result.append((action, next_state))
    return result


def _oxygen_prune(grid: TideGridGrid, state: TideGridState) -> bool:
    """Fast-path: return True if definitely unsolvable (not enough oxygen)."""
    remaining = grid.initial_oxygen - state.time
    return remaining < heuristic(grid, state)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_solvable(
    grid: TideGridGrid,
    state: TideGridState,
    max_expansions: int = 500_000,
) -> bool:
    """Check if the puzzle is solvable from the given state using A*."""
    if _oxygen_prune(grid, state):
        return False
    return astar_search(
        initial_state=state,
        get_successors=lambda s: _get_successors(grid, s),
        is_goal=lambda s: is_solved(grid, s),
        heuristic=lambda s: heuristic(grid, s),
        hash_state=lambda s: s.to_hashable(),
        max_expansions=max_expansions,
        return_path=False,
    )


def oracle_solve(
    grid: TideGridGrid,
    state: TideGridState,
    max_expansions: int = 500_000,
) -> Optional[list[TideGridAction]]:
    """Find the optimal solution path using A*."""
    if _oxygen_prune(grid, state):
        return None
    return astar_search(
        initial_state=state,
        get_successors=lambda s: _get_successors(grid, s),
        is_goal=lambda s: is_solved(grid, s),
        heuristic=lambda s: heuristic(grid, s),
        hash_state=lambda s: s.to_hashable(),
        max_expansions=max_expansions,
        return_path=True,
    )


def is_walking_dead(
    grid: TideGridGrid,
    state: TideGridState,
    max_expansions: int = 500_000,
) -> bool:
    """Check if the state is Walking Dead: has legal moves but is unsolvable."""
    has_legal = len(get_legal_actions(grid, state)) > 0
    if not has_legal:
        return False
    return not is_solvable(grid, state, max_expansions)


def apply_actions(
    grid: TideGridGrid,
    state: TideGridState,
    actions: list[TideGridAction],
) -> TideGridState:
    """Sequentially apply a list of actions."""
    for action in actions:
        state = apply_action(grid, state, action)
    return state


def find_fatal_step(
    grid: TideGridGrid,
    initial_state: TideGridState,
    actions: list[TideGridAction],
    max_expansions: int = 500_000,
) -> int:
    """Find the first action index where the puzzle becomes unsolvable.

    In Tide Grid every action can potentially cause deadlock (oxygen is
    consumed by all actions including WAIT), so we check solvability after
    every step.  Oxygen pruning keeps this fast for most states.

    Returns the index of the fatal action, or -1 if no deadlock occurs.
    """
    state = initial_state
    for i, action in enumerate(actions):
        state = apply_action(grid, state, action)
        if not is_solvable(grid, state, max_expansions):
            return i
    return -1


def count_reachable_dead_states(
    grid: TideGridGrid,
    state: TideGridState,
    max_states: int = 200,
) -> int:
    """BFS from a Walking Dead state to count reachable states.

    Measures how many unique states the agent can reach while stuck in
    a Walking Dead situation.  Higher k_horizon = longer continuation.
    """
    visited: set[tuple] = {state.to_hashable()}
    queue = deque([state])

    while queue and len(visited) < max_states:
        current = queue.popleft()
        for action in get_legal_actions(grid, current):
            next_state = apply_action(grid, current, action)
            h = next_state.to_hashable()
            if h not in visited:
                visited.add(h)
                queue.append(next_state)

    return len(visited)
