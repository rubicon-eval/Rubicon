"""Generic A* search used by all puzzle oracles."""

from __future__ import annotations

import heapq
from typing import Any, Callable, Optional


def astar_search(
    initial_state: Any,
    get_successors: Callable[[Any], list[tuple[Any, Any]]],
    is_goal: Callable[[Any], bool],
    heuristic: Callable[[Any], float],
    hash_state: Callable[[Any], Any],
    max_expansions: int = 500_000,
    return_path: bool = False,
) -> bool | Optional[list[Any]]:
    """A* search parameterized by callables.

    Args:
        initial_state: The starting state.
        get_successors: (state) -> list of (action, next_state) pairs.
        is_goal: (state) -> True if this is a goal state.
        heuristic: (state) -> admissible cost-to-go estimate.
        hash_state: (state) -> hashable representation for visited set.
        max_expansions: Safety limit on node expansions.
        return_path: If False, returns bool (reachable). If True, returns
                     Optional[list[action]] — the action sequence, or None.

    Returns:
        If return_path=False: True if goal reachable, else False.
        If return_path=True: list of actions to reach goal, or None.
    """
    h0 = heuristic(initial_state)
    init_hash = hash_state(initial_state)

    # counter breaks ties in the priority queue
    counter = 0
    # (f, counter, g, state_hash)
    open_set: list[tuple[float, int, int, Any]] = [(h0, counter, 0, init_hash)]
    visited: set = {init_hash}

    # Map hash -> state (to recover states from hashes)
    state_of: dict[Any, Any] = {init_hash: initial_state}

    # Parent pointer map for path reconstruction: hash -> (parent_hash, action)
    came_from: dict[Any, tuple[Any, Any]] = {}

    expansions = 0

    while open_set and expansions < max_expansions:
        f, _, g, current_hash = heapq.heappop(open_set)
        current = state_of[current_hash]

        if is_goal(current):
            if not return_path:
                return True
            # Reconstruct path
            path = []
            h = current_hash
            while h in came_from:
                parent_h, action = came_from[h]
                path.append(action)
                h = parent_h
            path.reverse()
            return path

        expansions += 1

        for action, next_state in get_successors(current):
            nh = hash_state(next_state)
            if nh not in visited:
                visited.add(nh)
                state_of[nh] = next_state
                next_g = g + 1
                next_f = next_g + heuristic(next_state)
                counter += 1
                heapq.heappush(open_set, (next_f, counter, next_g, nh))
                if return_path:
                    came_from[nh] = (current_hash, action)

    return False if not return_path else None
