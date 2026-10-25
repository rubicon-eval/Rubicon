"""Mosaic Grid prompt builders — generates model evaluation prompts per RQ.

Each prompt contains only the game rules and the task question.
State information (grid image, positions, etc.) is provided separately
via rendered images or other modality-specific means.
"""

from __future__ import annotations

from puzzles.common.types import Pos, Direction
from .engine import MosaicGrid, MosaicState, get_tile_at, apply_action, decode_painted, decode_activated
from .renderer import generate_rules_text


# ---------------------------------------------------------------------------
# Text helpers (kept available but not included in prompts)
# ---------------------------------------------------------------------------

def _tile_symbol(grid: MosaicGrid, state: MosaicState, pos: Pos) -> str:
    """Return a 1-char symbol for a tile at the given position."""
    if pos == state.agent_pos:
        return "A"
    tile = get_tile_at(grid, state, pos)
    return {
        "wall": "#", "painted": "~", "door_closed": "D",
        "door_open": "d", "plate": "P", "exit": "E",
    }.get(tile, ".")


def render_text_grid(grid: MosaicGrid, state: MosaicState) -> str:
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


def describe_state(grid: MosaicGrid, state: MosaicState) -> str:
    """Generate a full text description of a state (utility, not used in prompts)."""
    text_grid = render_text_grid(grid, state)
    painted = decode_painted(state.painted, grid.width)
    painted_str = ", ".join(f"({p.row}, {p.col})" for p in painted) if painted else "none"
    activated = decode_activated(state.activated_plates, grid._bit_to_pid)  # type: ignore[attr-defined]
    activated_str = ", ".join(
        f"plate at ({grid.plates[pid].pos.row}, {grid.plates[pid].pos.col})"
        for pid in activated
    ) if activated else "none"
    open_doors = [
        f"door at ({d.pos.row}, {d.pos.col})"
        for did, d in sorted(grid.doors.items())
        if state.is_door_open(grid, did)
    ]
    open_doors_str = ", ".join(open_doors) if open_doors else "none"
    return (
        f"Grid ({grid.width}x{grid.height}):\n{text_grid}\n\n"
        f"Agent position: row {state.agent_pos.row}, col {state.agent_pos.col}\n"
        f"Painted tiles: {painted_str}\n"
        f"Activated plates: {activated_str}\n"
        f"Open doors: {open_doors_str}"
    )


# ---------------------------------------------------------------------------
# Rules preamble
# ---------------------------------------------------------------------------

def build_rules_prompt(grid: MosaicGrid) -> str:
    """Build the game rules section of the prompt."""
    instance_rules = generate_rules_text(grid)
    return (
        "You are playing a grid-based puzzle called Mosaic Grid.\n"
        "\n"
        "## Rules\n"
        "\n"
        "1. You control an agent on a rectangular grid. Each cell is one of: "
        "floor, wall, pressure plate, door, or exit.\n"
        "2. The agent can move one step at a time: UP, DOWN, LEFT, or RIGHT.\n"
        "3. The agent can move onto floor, pressure plates, open doors, and the "
        "exit. The agent CANNOT move onto walls, painted tiles, or closed doors.\n"
        "4. **Paint on departure:** Every tile the agent leaves becomes permanently "
        "painted and impassable. The tile the agent currently stands on is NOT yet "
        "painted — painting happens when the agent moves away.\n"
        "5. **Pressure plates:** Stepping onto a pressure plate activates it. "
        "Plates are linked to doors — when all plates linked to a door are "
        "activated, that door opens permanently. Plates and doors that share "
        "the same color are linked.\n"
        "6. **Goal:** The agent wins by stepping onto the exit tile.\n"
        "\n"
        "## This Puzzle\n"
        "\n"
        f"{instance_rules}"
    )


def _format_action_list(actions: list[Direction]) -> str:
    return "\n".join(f"{i + 1}. {a.name}" for i, a in enumerate(actions))


# ---------------------------------------------------------------------------
# RQ1: Simulation Fidelity
# ---------------------------------------------------------------------------

def build_rq1_prompt(
    grid: MosaicGrid,
    actions: list[Direction],
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
    grid: MosaicGrid,
    actions: list[Direction],
    direction: Direction,
) -> str:
    """Build the RQ1 ablation prompt: after N actions, is a move legal?"""
    rules = build_rules_prompt(grid)
    n = len(actions)
    action_list = _format_action_list(actions)
    return (
        f"{rules}\n"
        "\n"
        "## Task\n"
        "\n"
        f"The following {n} actions are taken in order starting from the "
        "state shown in the image:\n"
        f"{action_list}\n"
        "\n"
        f"After all {n} actions, the agent attempts to move {direction.name}.\n"
        "\n"
        "What happens? Choose one:\n"
        "A) The agent moves successfully — the move is legal.\n"
        "B) The agent stays in place — the move is illegal.\n"
        "\n"
        "Respond in this format:\n"
        "Answer: [A or B]\n"
        "Explanation: [why the move is legal or illegal]"
    )


# ---------------------------------------------------------------------------
# RQ2: Foresight
# ---------------------------------------------------------------------------

def build_rq2_main_prompt(
    grid: MosaicGrid,
    actions: list[Direction],
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
    grid: MosaicGrid,
    actions: list[Direction],
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
    grid: MosaicGrid,
    actions: list[Direction],
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
    grid: MosaicGrid,
    actions: list[Direction],
    show_initial: bool = False,
) -> str:
    """Build the RQ3 ablation prompt: fatal step with final state given.

    The ablation removes the simulation burden — the model sees the final
    (and optionally initial) state as an image and only needs to reason
    about which step was causally fatal.

    Args:
        show_initial: If False, image shows only the final state
            (``ablation_final``).  If True, image shows initial + final
            side by side (``ablation_comparison``).
    """
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
