"""Chain Vault prompt builders — generates model evaluation prompts per RQ.

Each prompt contains only the game rules and the task question.
State information (grid image, positions, etc.) is provided separately
via rendered images or other modality-specific means.
"""

from __future__ import annotations

from puzzles.common.types import Pos, Direction
from .engine import (
    ChainVaultGrid,
    ChainVaultState,
    ChainVaultAction,
    get_tile_at,
    _inventory_to_dict,
)
from .renderer import generate_rules_text


# ---------------------------------------------------------------------------
# Text helpers (kept available but not included in prompts)
# ---------------------------------------------------------------------------

def _tile_symbol(grid: ChainVaultGrid, state: ChainVaultState, pos: Pos) -> str:
    """Return a 1-char symbol for a tile at the given position."""
    if pos == state.agent_pos:
        return "A"
    tile = get_tile_at(grid, state, pos)
    return {
        "wall": "#", "key": "K", "door_closed": "D",
        "door_open": "d", "exit": "E",
    }.get(tile, ".")


def render_text_grid(grid: ChainVaultGrid, state: ChainVaultState) -> str:
    """Render the grid as a labeled ASCII table (utility, not used in prompts)."""
    header = "     " + "".join(f" {c:2d} " for c in range(grid.width))
    lines = [header]
    for r in range(grid.height):
        row_str = f"  {r} "
        for c in range(grid.width):
            sym = _tile_symbol(grid, state, Pos(r, c))
            row_str += f" [{sym:^2s}]"
        lines.append(row_str)
    return "\n".join(lines)


def describe_state(grid: ChainVaultGrid, state: ChainVaultState) -> str:
    """Generate a full text description of a state (utility, not used in prompts)."""
    text_grid = render_text_grid(grid, state)
    inv = _inventory_to_dict(state.inventory)
    inv_str = ", ".join(f"{c}: {n}" for c, n in sorted(inv.items())) if inv else "empty"
    remaining = sorted(
        f"{grid.keys[kid].color} key at ({grid.keys[kid].pos.row}, {grid.keys[kid].pos.col})"
        for kid in state.remaining_keys
    )
    remaining_str = ", ".join(remaining) if remaining else "none"
    open_doors = [
        f"door at ({grid.doors[did].pos.row}, {grid.doors[did].pos.col})"
        for did in sorted(state.opened_doors)
    ]
    open_doors_str = ", ".join(open_doors) if open_doors else "none"
    return (
        f"Grid ({grid.width}x{grid.height}):\n{text_grid}\n\n"
        f"Agent position: row {state.agent_pos.row}, col {state.agent_pos.col}\n"
        f"Inventory: {{{inv_str}}}\n"
        f"Keys on ground: {remaining_str}\n"
        f"Open doors: {open_doors_str}"
    )


# ---------------------------------------------------------------------------
# Rules preamble
# ---------------------------------------------------------------------------

def build_rules_prompt(grid: ChainVaultGrid) -> str:
    """Build the game rules section of the prompt."""
    instance_rules = generate_rules_text(grid)
    return (
        "You are playing a grid-based puzzle called Chain Vault.\n"
        "\n"
        "## Rules\n"
        "\n"
        "1. You control an agent on a rectangular grid. Each cell is one of: "
        "floor, wall, key, locked door, open door, or exit.\n"
        "2. The agent can move one step at a time: UP, DOWN, LEFT, or RIGHT.\n"
        "3. The agent can move onto floor, keys, opened doors, and the exit. "
        "The agent CANNOT move onto walls or locked doors.\n"
        "4. **Key pickup:** Stepping onto a key tile automatically picks up "
        "the key. The tile becomes floor. Keys are colored (red, blue, green, yellow, etc.).\n"
        "5. **Using keys on doors:** When the agent moves toward a locked "
        "door (UP/DOWN/LEFT/RIGHT into an adjacent locked door), two things "
        "can happen:\n"
        "   - **Agent has required keys:** The key(s) are consumed, the door "
        "opens permanently, but **the agent does NOT move — it remains at "
        "the same (row, col) it was at before this action**. The agent can "
        "then walk through the now-open door on a later move.\n"
        "   - **Agent is missing required keys:** Nothing happens — the door "
        "stays locked and the agent remains at the same (row, col) it was at before this action.\n"
        "   In both cases, the agent's position does not change. "
        "Opening a door costs one action but does not move the agent.\n"
        "   **Example:** Agent is at (3, 2) with a red key. A red locked door "
        "is at (3, 3). Agent moves RIGHT:\n"
        "   → The red key is consumed, the door at (3, 3) opens, but the "
        "agent is still at **(3, 2)** — NOT (3, 3).\n"
        "   On the next move, the agent can move RIGHT again to walk through "
        "the now-open door to (3, 3).\n"
        "6. **Single-lock doors** require 1 key of matching color. "
        "**Dual-lock doors** require 1 key of each of two colors simultaneously. "
        "**Triple-lock doors** require 1 key of each of three colors simultaneously. "
        "**Quad-lock doors** require 1 key of each of four colors simultaneously.\n"
        "7. **Key consumption:** Keys are consumed when used for opening doors.\n"
        "8. **Goal:** The agent wins by stepping onto the exit tile.\n"
        "\n"
        f"{instance_rules}"
    )


def _format_action_list(actions: list[ChainVaultAction]) -> str:
    return "\n".join(f"{i + 1}. {a.name}" for i, a in enumerate(actions))


# ---------------------------------------------------------------------------
# RQ1: Simulation Fidelity
# ---------------------------------------------------------------------------

def build_rq1_prompt(
    grid: ChainVaultGrid,
    actions: list[ChainVaultAction],
) -> str:
    """Build the RQ1 main prompt: predict state after n actions."""
    n = len(actions)
    return (
        f"{build_rules_prompt(grid)}\n"
        "\n"
        "## Task\n"
        "\n"
        f"The following {n} actions are taken in order starting from the "
        "state shown in the image:\n"
        f"{_format_action_list(actions)}\n"
        "\n"
        f"After all {n} actions are executed, what is the agent's final "
        "position?\n"
        "Think step by step. Return the final position of the agent in the "
        "form (row, col): POSITION: (x, y)"
    )


def build_rq1_ablation_prompt(
    grid: ChainVaultGrid,
    direction: Direction,
) -> str:
    """Build the RQ1 ablation prompt: single-step boundary physics."""
    action_name = direction.name
    attempt_desc = f"move {direction.name}"

    return (
        f"{build_rules_prompt(grid)}\n"
        "\n"
        "## Task\n"
        "\n"
        f"Looking at the state shown in the image, the agent attempts to "
        f"{attempt_desc} (action: {action_name}).\n"
        "\n"
        "What happens? Choose one:\n"
        "A) The action succeeds — it is legal.\n"
        "B) The action fails — it is illegal.\n"
        "\n"
        "Respond in this format:\n"
        "Answer: [A or B]\n"
        "Explanation: [why the action is legal or illegal]"
    )


# ---------------------------------------------------------------------------
# RQ2: Foresight
# ---------------------------------------------------------------------------

def build_rq2_main_prompt(
    grid: ChainVaultGrid,
    actions: list[ChainVaultAction],
) -> str:
    """Build the RQ2 main prompt: next move or unsolvable."""
    k = len(actions)
    return (
        f"{build_rules_prompt(grid)}\n"
        "\n"
        "## Task\n"
        "\n"
        f"Starting from the state shown in the image, the following {k} "
        "actions were taken:\n"
        f"{_format_action_list(actions)}\n"
        "\n"
        "What is the best next move to win?"
    )


def build_rq2_ablation_prompt(
    grid: ChainVaultGrid,
    actions: list[ChainVaultAction],
) -> str:
    """Build the RQ2 ablation prompt: explicit solvability question."""
    k = len(actions)
    return (
        f"{build_rules_prompt(grid)}\n"
        "\n"
        "## Task\n"
        "\n"
        f"Starting from the state shown in the image, the following {k} "
        "actions were taken:\n"
        f"{_format_action_list(actions)}\n"
        "\n"
        "Is the puzzle still solvable? That is, does there exist a path from "
        "the agent's current position that can complete the puzzle?\n"
        "Think step by step. Return your final answer in the form: "
        "\"SOLVABLE: <YES/NO>\""
    )


# ---------------------------------------------------------------------------
# RQ3: Causal Attribution
# ---------------------------------------------------------------------------

def build_rq3_prompt(
    grid: ChainVaultGrid,
    actions: list[ChainVaultAction],
) -> str:
    """Build the RQ3 prompt: identify the fatal step in a trace."""
    trace_length = len(actions)
    return (
        f"{build_rules_prompt(grid)}\n"
        "\n"
        "## Task\n"
        "\n"
        f"Starting from the state shown in the image, the following "
        f"{trace_length} actions were taken:\n"
        f"{_format_action_list(actions)}\n"
        "\n"
        "After this full sequence, no remaining sequence of moves can "
        "complete the puzzle. The puzzle had become unsolvable at one of the "
        "steps listed above.\n"
        "\n"
        "Identify the **fatal step**: the action whose execution caused the "
        "puzzle to become unsolvable. Before this step, the puzzle was still "
        "solvable. After this step, it became impossible to win.\n"
        "\n"
        "Think step by step and finally respond the 1-indexed step number of "
        "the fatal move in the format:\n"
        "FATAL STEP: <1-INDEXED STEP NUMBER>"
    )


def build_rq3_ablation_prompt(
    grid: ChainVaultGrid,
    actions: list[ChainVaultAction],
    show_initial: bool = False,
) -> str:
    """Build the RQ3 ablation prompt: fatal step with final state given."""
    trace_length = len(actions)

    if show_initial:
        image_desc = (
            "The image shows two states side by side: the initial state "
            "(left) and the final state after all actions (right)."
        )
    else:
        image_desc = (
            "The image shows the final state after all actions."
        )

    return (
        f"{build_rules_prompt(grid)}\n"
        "\n"
        "## Task\n"
        "\n"
        f"{image_desc}\n"
        "\n"
        f"The following {trace_length} actions were taken in order:\n"
        f"{_format_action_list(actions)}\n"
        "\n"
        "After this full sequence, no remaining sequence of moves can "
        "complete the puzzle. The puzzle had become unsolvable at one of the "
        "steps listed above.\n"
        "\n"
        "Identify the **fatal step**: the action whose execution caused the "
        "puzzle to become unsolvable. Before this step, the puzzle was still "
        "solvable. After this step, it became impossible to win.\n"
        "\n"
        "Think step by step and finally respond the 1-indexed step number of "
        "the fatal move in the format:\n"
        "FATAL STEP: <1-INDEXED STEP NUMBER>"
    )
