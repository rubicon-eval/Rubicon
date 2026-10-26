"""Chain Vault oracle — A* search, solvability, deadlock detection."""

from __future__ import annotations

from typing import Optional

from puzzles.common.types import manhattan_distance
from puzzles.common.astar import astar_search
from .engine import (
    ChainVaultGrid,
    ChainVaultState,
    ChainVaultAction,
    get_legal_actions,
    apply_action,
    action_opens_door,
    is_solved,
)


def heuristic(grid: ChainVaultGrid, state: ChainVaultState) -> int:
    """Admissible heuristic: manhattan distance to exit.

    Each action reduces manhattan by at most 1. Door-opening moves do not move
    the agent at all, so manhattan distance is always a lower bound.
    """
    return manhattan_distance(state.agent_pos, grid.exit_pos)


def _get_successors(
    grid: ChainVaultGrid,
    state: ChainVaultState,
) -> list[tuple[ChainVaultAction, ChainVaultState]]:
    """Get all successor (action, state) pairs."""
    result = []
    for action in get_legal_actions(grid, state):
        next_state = apply_action(grid, state, action)
        result.append((action, next_state))
    return result


def is_solvable(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_expansions: int = 500_000,
) -> bool:
    """Check if the puzzle is solvable from the given state using A*."""
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
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_expansions: int = 500_000,
) -> Optional[list[ChainVaultAction]]:
    """Find the optimal solution path using A*."""
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
    grid: ChainVaultGrid,
    state: ChainVaultState,
    max_expansions: int = 500_000,
) -> bool:
    """Check if the state is Walking Dead: has legal moves but is unsolvable."""
    has_legal = len(get_legal_actions(grid, state)) > 0
    if not has_legal:
        return False
    return not is_solvable(grid, state, max_expansions)


def apply_actions(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    actions: list[ChainVaultAction],
) -> ChainVaultState:
    """Sequentially apply a list of actions."""
    for action in actions:
        state = apply_action(grid, state, action)
    return state


def find_fatal_step(
    grid: ChainVaultGrid,
    initial_state: ChainVaultState,
    actions: list[ChainVaultAction],
    max_expansions: int = 500_000,
) -> int:
    """Find the first action index where the puzzle becomes unsolvable.

    Chain Vault optimization: only check is_solvable after actions that open
    a door and consume keys, since plain movement + key pickup cannot deadlock.

    Returns the index of the fatal action, or -1 if no deadlock occurs.
    """
    state = initial_state
    for i, action in enumerate(actions):
        opens_door = action_opens_door(grid, state, action)
        state = apply_action(grid, state, action)
        if opens_door:
            if not is_solvable(grid, state, max_expansions):
                return i
    return -1
