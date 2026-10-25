"""Mosaic Grid Matplotlib renderer — research-paper quality grid images."""

from __future__ import annotations

import math
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure

from puzzles.common.types import Pos, Direction
from puzzles.common.rendering import (
    COLORS,
    get_linkage_color,
    create_grid_figure,
    fill_cell,
    draw_circle,
    draw_star,
    draw_diamond,
    draw_text_in_cell,
    draw_highlight_border,
    save_figure,
)
from .engine import (
    MosaicGrid,
    MosaicState,
    get_tile_at,
    get_legal_actions,
    apply_action,
    is_solved,
)


# ---------------------------------------------------------------------------
# Linkage color mapping
# ---------------------------------------------------------------------------

def _build_linkage_map(grid: MosaicGrid) -> dict[str, dict[str, str]]:
    """Build a mapping from door_id -> linkage color dict.

    Each door gets a distinct color from the palette. All plates linked
    to that door share its color.
    """
    door_ids = sorted(grid.doors.keys())
    return {did: get_linkage_color(i) for i, did in enumerate(door_ids)}


def _plate_color(grid: MosaicGrid, plate_id: str, linkage_map: dict) -> dict[str, str]:
    """Get the linkage color for a plate (based on the door it's linked to)."""
    plate = grid.plates[plate_id]
    if plate.linked_door_ids:
        door_id = plate.linked_door_ids[0]
        if door_id in linkage_map:
            return linkage_map[door_id]
    return get_linkage_color(0)


# ---------------------------------------------------------------------------
# Text rules for model prompts
# ---------------------------------------------------------------------------

def generate_rules_text(grid: MosaicGrid) -> str:
    """Generate a text description of the puzzle rules for model prompts.

    Example output:
        Rules:
        - The blue plate at row 1, col 2 opens the blue door at row 3, col 2
        - Both green plates (at row 0, col 1 and row 2, col 0) must be
          activated to open the green door at row 1, col 4
    """
    linkage_map = _build_linkage_map(grid)
    lines = [
        "- Coordinates are (row, col) where row 0 is the top row and col 0 is the leftmost column",
        f"- The agent starts at row {grid.start.row}, col {grid.start.col}",
    ]

    for door_id in sorted(grid.doors.keys()):
        door = grid.doors[door_id]
        color_name = linkage_map[door_id]["name"]
        door_pos = door.pos

        plate_ids = list(door.required_plate_ids)
        plate_positions = [grid.plates[pid].pos for pid in plate_ids]

        if len(plate_ids) == 1:
            pp = plate_positions[0]
            lines.append(
                f"- The {color_name} plate at row {pp.row}, col {pp.col} "
                f"opens the {color_name} door at row {door_pos.row}, col {door_pos.col}"
            )
        else:
            pos_strs = [f"row {p.row}, col {p.col}" for p in plate_positions]
            joined = " and ".join(pos_strs)
            lines.append(
                f"- All {len(plate_ids)} {color_name} plates "
                f"(at {joined}) must be activated "
                f"to open the {color_name} door at row {door_pos.row}, col {door_pos.col}"
            )

    lines.append(
        f"- The exit is at row {grid.exit_pos.row}, col {grid.exit_pos.col}"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grid drawing
# ---------------------------------------------------------------------------

def _draw_grid(
    ax,
    grid: MosaicGrid,
    state: MosaicState,
    linkage_map: dict[str, dict[str, str]],
    highlight_cell: Optional[Pos] = None,
    step_label: Optional[str] = None,
) -> None:
    """Draw the full grid state on the given axes."""
    for r in range(grid.height):
        for c in range(grid.width):
            pos = Pos(r, c)
            tile = get_tile_at(grid, state, pos)

            if tile == "wall":
                fill_cell(ax, r, c, COLORS["wall"])

            elif tile == "painted":
                fill_cell(ax, r, c, COLORS["painted"])
                ax.plot(
                    [c + 0.3, c + 0.7], [r + 0.3, r + 0.7],
                    color="white", linewidth=1.5, alpha=0.5, zorder=4,
                )
                ax.plot(
                    [c + 0.3, c + 0.7], [r + 0.7, r + 0.3],
                    color="white", linewidth=1.5, alpha=0.5, zorder=4,
                )

            elif tile == "door_closed":
                door = grid.door_at(pos)
                lc = linkage_map.get(door.id, get_linkage_color(0)) if door else get_linkage_color(0)
                fill_cell(ax, r, c, lc["door_closed"])
                # Draw lock bars icon
                for offset in [-0.15, 0.0, 0.15]:
                    ax.plot(
                        [c + 0.25, c + 0.75], [r + 0.5 + offset, r + 0.5 + offset],
                        color="white", linewidth=2, alpha=0.7, zorder=6,
                    )

            elif tile == "door_open":
                door = grid.door_at(pos)
                lc = linkage_map.get(door.id, get_linkage_color(0)) if door else get_linkage_color(0)
                fill_cell(ax, r, c, lc["door_open"])
                # Draw open door frame
                rect = patches.Rectangle(
                    (c + 0.15, r + 0.15), 0.7, 0.7,
                    fill=False, edgecolor=lc["door_closed"],
                    linewidth=2, linestyle="--", zorder=6,
                )
                ax.add_patch(rect)

            elif tile == "plate":
                plate = grid.plate_at(pos)
                if plate:
                    lc = _plate_color(grid, plate.id, linkage_map)
                    activated = (state.activated_plates >> grid._pid_to_bit[plate.id]) & 1  # type: ignore[attr-defined]
                    if activated:
                        # Dimmed version — plate was activated
                        fill_cell(ax, r, c, lc["plate_fill"])
                        draw_diamond(ax, r, c, lc["plate_icon"], size=0.22)
                        # Small check mark
                        ax.plot(
                            [c + 0.35, c + 0.45, c + 0.65],
                            [r + 0.55, r + 0.65, r + 0.35],
                            color="white", linewidth=2, zorder=8,
                        )
                    else:
                        fill_cell(ax, r, c, lc["plate_fill"])
                        draw_diamond(ax, r, c, lc["plate_icon"], size=0.25)
                else:
                    fill_cell(ax, r, c, COLORS["floor"])

            elif tile == "exit":
                fill_cell(ax, r, c, "#FCF3CF")
                draw_star(ax, r, c, COLORS["exit"], size=0.3)

            else:  # floor
                fill_cell(ax, r, c, COLORS["floor"])

    # Draw agent
    if not is_solved(grid, state):
        draw_circle(
            ax, state.agent_pos.row, state.agent_pos.col,
            COLORS["agent"], radius=0.28, edge_color="white", linewidth=2.0,
        )
    else:
        draw_circle(
            ax, state.agent_pos.row, state.agent_pos.col,
            "#27AE60", radius=0.28, edge_color="white", linewidth=2.0,
        )

    # Highlight cell (for fatal step)
    if highlight_cell is not None:
        draw_highlight_border(ax, highlight_cell.row, highlight_cell.col,
                              color=COLORS["highlight"], linewidth=3.5)

    if step_label is not None:
        ax.set_title(step_label, fontsize=9, pad=3)


# ---------------------------------------------------------------------------
# Public render functions
# ---------------------------------------------------------------------------

def render_state(
    grid: MosaicGrid,
    state: MosaicState,
    title: Optional[str] = None,
    show_legend: bool = False,
    highlight_cell: Optional[Pos] = None,
) -> Figure:
    """Render a single game state as a Matplotlib figure."""
    linkage_map = _build_linkage_map(grid)
    fig, ax = create_grid_figure(grid.width, grid.height, cell_size=0.8)

    _draw_grid(ax, grid, state, linkage_map, highlight_cell=highlight_cell)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    fig.tight_layout()
    return fig


def render_trace(
    grid: MosaicGrid,
    initial_state: MosaicState,
    actions: list[Direction],
    cols: int = 4,
    highlight_fatal: Optional[int] = None,
    title: Optional[str] = None,
) -> Figure:
    """Render a sequence of states as a multi-panel trace montage."""
    linkage_map = _build_linkage_map(grid)
    n_steps = len(actions) + 1
    rows = math.ceil(n_steps / cols)

    cell_size = 0.55
    fig_w = cols * (grid.width * cell_size + 0.3) + 0.5
    fig_h = rows * (grid.height * cell_size + 0.6) + 0.5

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    state = initial_state
    step_idx = 0

    for row_idx in range(rows):
        for col_idx in range(cols):
            ax = axes[row_idx][col_idx]

            if step_idx >= n_steps:
                ax.set_visible(False)
                step_idx += 1
                continue

            ax.set_xlim(-0.05, grid.width + 0.05)
            ax.set_ylim(-0.05, grid.height + 0.05)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            highlight = None
            if highlight_fatal is not None and step_idx == highlight_fatal + 1:
                highlight = state.agent_pos

            if step_idx == 0:
                label = "Start"
            else:
                action_name = actions[step_idx - 1].name
                label = f"Step {step_idx}: {action_name}"
                if highlight_fatal is not None and step_idx == highlight_fatal + 1:
                    label += " [FATAL]"

            _draw_grid(ax, grid, state, linkage_map,
                       highlight_cell=highlight, step_label=label)

            if step_idx < len(actions):
                state = apply_action(grid, state, actions[step_idx])

            step_idx += 1

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0)

    fig.tight_layout()
    return fig


def render_side_by_side(
    grid: MosaicGrid,
    state_left: MosaicState,
    state_right: MosaicState,
    label_left: str = "A",
    label_right: str = "B",
    title: Optional[str] = None,
) -> Figure:
    """Render two states side by side for comparison (e.g., RQ2 pairs)."""
    linkage_map = _build_linkage_map(grid)
    cell_size = 0.7
    fig_w = 2 * (grid.width * cell_size + 0.3) + 1.0
    fig_h = grid.height * cell_size + 1.2

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    for ax, state, label in [(ax_l, state_left, label_left),
                              (ax_r, state_right, label_right)]:
        ax.set_xlim(-0.05, grid.width + 0.05)
        ax.set_ylim(-0.05, grid.height + 0.05)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        _draw_grid(ax, grid, state, linkage_map, step_label=label)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0)

    fig.tight_layout()
    return fig
