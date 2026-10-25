"""Tide Grid prompt builders — generates model evaluation prompts per RQ.

Each prompt contains only the game rules and the task question.
State information (grid image, positions, etc.) is provided separately
via rendered images or other modality-specific means.
"""

from __future__ import annotations

from puzzles.common.types import Direction
from .engine import (
    TideGridGrid,
    TideGridState,
    TideGridAction,
)
from .renderer import generate_rules_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_action_list(actions: list[TideGridAction]) -> str:
    return "\n".join(f"{i + 1}. {a.name}" for i, a in enumerate(actions))


# ---------------------------------------------------------------------------
# Rules preamble
# ---------------------------------------------------------------------------

def build_rules_prompt(grid: TideGridGrid) -> str:
    """Build the game rules section of the prompt."""
    instance_rules = generate_rules_text(grid)
    return (
        "You are playing a grid-based puzzle called Tide Grid.\n"
        "\n"
        "## Rules\n"
        "\n"
        "1. You control an agent on a rectangular grid. Each cell is one of: "
        "floor (always passable), wall (never passable), tidal (cycles between "
        "dry/passable and flooded/impassable), or exit.\n"
        "2. The agent can take one of 5 actions each turn: UP, DOWN, LEFT, "
        "RIGHT (move one cell), or WAIT (stay in place). Every action advances "
        "time by 1 and costs 1 oxygen.\n"
        "3. **Tidal zones:** Tidal cells belong to zones (A, B, etc.). Each zone "
        "cycles between dry (passable) and flooded (impassable) on a fixed "
        "repeating schedule. The exact schedule for each zone is listed below "
        "under \"Tidal Zones\".\n"
        "4. **Movement legality:** A move is legal if the target cell is passable "
        "at the NEW time (time + 1). WAIT is legal if the current cell is "
        "passable at the new time. Both require oxygen > 0.\n"
        "5. **Beacons:** Beacon items are placed on floor tiles. Stepping onto "
        "a beacon automatically collects it. All beacons must be collected "
        "before reaching the exit.\n"
        "6. **Oxygen:** The agent starts with a fixed oxygen supply. Each action "
        "(including WAIT) costs 1 oxygen. When oxygen reaches 0, no more "
        "actions can be taken.\n"
        "7. **Goal:** The agent wins by stepping onto the exit tile.\n"
        "\n"
        "## This Puzzle\n"
        "\n"
        f"{instance_rules}"
    )


# ---------------------------------------------------------------------------
# RQ1: Simulation Fidelity
# ---------------------------------------------------------------------------

def build_rq1_prompt(
    grid: TideGridGrid,
    actions: list[TideGridAction],
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
    grid: TideGridGrid,
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
        "Consider: Is the target cell passable at the new time? "
        "If the target is a tidal cell, check whether its zone is dry or "
        "flooded at time + 1. Also check if the agent has oxygen remaining.\n"
        "\n"
        "Respond in this format:\n"
        "Answer: [A or B]\n"
        "Explanation: [why the action is legal or illegal]"
    )


# ---------------------------------------------------------------------------
# RQ2: Foresight
# ---------------------------------------------------------------------------

def build_rq2_main_prompt(
    grid: TideGridGrid,
    actions: list[TideGridAction],
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
    grid: TideGridGrid,
    actions: list[TideGridAction],
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
    grid: TideGridGrid,
    actions: list[TideGridAction],
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
    grid: TideGridGrid,
    actions: list[TideGridAction],
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
