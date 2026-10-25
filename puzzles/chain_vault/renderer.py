"""Chain Vault Matplotlib renderer — research-paper quality grid images."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure

from puzzles.common.types import Pos, Direction
from puzzles.common.rendering import (
    COLORS,
    create_grid_figure,
    fill_cell,
    draw_circle,
    draw_star,
    draw_text_in_cell,
    draw_highlight_border,
    save_figure,
)
from .engine import (
    ChainVaultGrid,
    ChainVaultState,
    ChainVaultAction,
    get_tile_at,
    get_legal_actions,
    apply_action,
    is_solved,
    _inventory_to_dict,
)


# ---------------------------------------------------------------------------
# Color mapping for key/door colors
# ---------------------------------------------------------------------------

KEY_DOOR_COLORS = {
    "red": {
        "key_fill": "#FADBD8",
        "key_icon": COLORS["key_red"],
        "door_closed": "#F1948A",     # more saturated red
        "door_icon": "#C0392B",       # dark red bars
        "door_open": "#FADBD8",
        "label": "R",
    },
    "blue": {
        "key_fill": "#D6EAF8",
        "key_icon": COLORS["key_blue"],
        "door_closed": "#85C1E9",     # more saturated blue
        "door_icon": "#2471A3",       # dark blue bars
        "door_open": "#D6EAF8",
        "label": "B",
    },
    "green": {
        "key_fill": "#D5F5E3",
        "key_icon": COLORS["key_green"],
        "door_closed": "#82E0AA",     # more saturated green
        "door_icon": "#1E8449",       # dark green bars
        "door_open": "#D5F5E3",
        "label": "G",
    },
    "yellow": {
        "key_fill": "#FEF9E7",
        "key_icon": COLORS["key_yellow"],
        "door_closed": "#F7DC6F",     # more saturated yellow
        "door_icon": "#B7950B",       # dark yellow bars
        "door_open": "#FEF9E7",
        "label": "Y",
    },
}


def _door_lock_name(required_keys: tuple[str, ...]) -> str:
    """Return a human-readable lock name based on required key count."""
    nkeys = len(required_keys)
    if nkeys == 1:
        return "single-lock"
    if nkeys == 2:
        return "dual-lock"
    if nkeys == 3:
        return "triple-lock"
    if nkeys == 4:
        return "quad-lock"
    return f"{nkeys}-key"


def _get_door_colors(door_info) -> dict:
    """Get the primary color dict for a door based on its first required key."""
    if door_info.required_keys:
        color = door_info.required_keys[0]
        return KEY_DOOR_COLORS.get(color, KEY_DOOR_COLORS["red"])
    return KEY_DOOR_COLORS["red"]


def _door_label(door_info) -> str:
    """Get a short label for a door (e.g. 'R' or 'R+B')."""
    labels = []
    for c in door_info.required_keys:
        colors = KEY_DOOR_COLORS.get(c, KEY_DOOR_COLORS["red"])
        labels.append(colors["label"])
    return "+".join(labels)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_key_icon(ax, row: int, col: int, color: str, size: float = 0.25) -> None:
    """Draw a key-shaped icon in a cell."""
    cx, cy = col + 0.5, row + 0.5
    icon_color = KEY_DOOR_COLORS.get(color, KEY_DOOR_COLORS["red"])["key_icon"]

    # Key head (circle)
    head = patches.Circle(
        (cx - size * 0.3, cy - size * 0.4),
        size * 0.4,
        facecolor=icon_color,
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
    )
    ax.add_patch(head)

    # Key shaft
    ax.plot(
        [cx - size * 0.3, cx + size * 0.6],
        [cy - size * 0.4, cy + size * 0.3],
        color=icon_color, linewidth=2.5, solid_capstyle="round", zorder=6,
    )

    # Key teeth
    for t in [0.4, 0.6]:
        tx = cx - size * 0.3 + t * size * 1.2
        ty = cy - size * 0.4 + t * size * 0.9
        ax.plot(
            [tx, tx + size * 0.2], [ty, ty + size * 0.15],
            color=icon_color, linewidth=2, solid_capstyle="round", zorder=6,
        )


def _draw_door_bars(ax, row: int, col: int, door_info) -> None:
    """Draw thick colored vertical bars across a door cell (like a gate).

    Single-lock: all bars in the required key color.
    Dual-lock: bars alternate between the two required colors.
    """
    n_bars = 4
    bar_colors = []
    for c in door_info.required_keys:
        cdict = KEY_DOOR_COLORS.get(c, KEY_DOOR_COLORS["red"])
        bar_colors.append(cdict["door_icon"])

    margin_x = 0.15
    margin_y = 0.12
    bar_width = 0.07

    for i in range(n_bars):
        x = col + margin_x + i * (1.0 - 2 * margin_x) / (n_bars - 1)
        y_top = row + margin_y
        y_bot = row + 1.0 - margin_y
        bc = bar_colors[i % len(bar_colors)]
        ax.plot(
            [x, x], [y_top, y_bot],
            color=bc, linewidth=4.5, solid_capstyle="round", zorder=7,
        )

    # Horizontal cross-bar in the middle for "locked" feel
    y_mid = row + 0.5
    primary_color = bar_colors[0]
    ax.plot(
        [col + margin_x - 0.02, col + 1.0 - margin_x + 0.02],
        [y_mid, y_mid],
        color=primary_color, linewidth=3.0, solid_capstyle="round", zorder=7,
    )


def _draw_inventory_hud(
    ax,
    state: ChainVaultState,
    grid_width: int,
    y_start: float,
) -> None:
    """Draw inventory HUD below the grid showing current keys."""
    inv = _inventory_to_dict(state.inventory)
    if not inv:
        ax.text(
            grid_width / 2, y_start + 0.3,
            "Inventory: (empty)",
            ha="center", va="center", fontsize=8,
            color=COLORS["text_dark"], style="italic",
        )
        return

    ax.text(
        0.1, y_start + 0.15,
        "Inventory:",
        ha="left", va="center", fontsize=8,
        color=COLORS["text_dark"], fontweight="bold",
    )

    x_offset = 1.4
    for color, count in sorted(inv.items()):
        icon_color = KEY_DOOR_COLORS.get(color, KEY_DOOR_COLORS["red"])["key_icon"]
        label = KEY_DOOR_COLORS.get(color, KEY_DOOR_COLORS["red"])["label"]

        for i in range(count):
            # Small key icon
            head = patches.Circle(
                (x_offset + 0.1, y_start + 0.15),
                0.1,
                facecolor=icon_color,
                edgecolor="white",
                linewidth=0.8,
                zorder=6,
            )
            ax.add_patch(head)
            ax.plot(
                [x_offset + 0.1, x_offset + 0.35],
                [y_start + 0.15, y_start + 0.25],
                color=icon_color, linewidth=1.5, solid_capstyle="round", zorder=6,
            )
            ax.text(
                x_offset + 0.45, y_start + 0.15,
                f"{label}",
                ha="left", va="center", fontsize=7,
                color=icon_color, fontweight="bold",
            )
            x_offset += 0.7


# ---------------------------------------------------------------------------
# Text rules for model prompts
# ---------------------------------------------------------------------------

def generate_rules_text(grid: ChainVaultGrid) -> str:
    """Generate a text description of the puzzle rules for model prompts."""
    lines = [
        "Instance details:",
        "- Coordinates are (row, col) where row 0 is the top row and col 0 is the leftmost column",
    ]

    # Keys
    for kid in sorted(grid.keys.keys()):
        key = grid.keys[kid]
        lines.append(
            f"- {key.color.capitalize()} key at row {key.pos.row}, col {key.pos.col}"
        )

    # Doors
    for did in sorted(grid.doors.keys()):
        door = grid.doors[did]
        req = " + ".join(f"1 {c} key" for c in door.required_keys)
        dtype = _door_lock_name(door.required_keys)
        lines.append(
            f"- {dtype.capitalize()} door at row {door.pos.row}, col {door.pos.col} "
            f"(requires {req} to open)"
        )

    lines.append(
        f"- Exit at row {grid.exit_pos.row}, col {grid.exit_pos.col}"
    )
    lines.append(
        f"- Start at row {grid.start.row}, col {grid.start.col}"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grid drawing
# ---------------------------------------------------------------------------

def _draw_grid(
    ax,
    grid: ChainVaultGrid,
    state: ChainVaultState,
    highlight_cell: Optional[Pos] = None,
    step_label: Optional[str] = None,
    show_inventory: bool = False,
) -> None:
    """Draw the full grid state on the given axes."""
    for r in range(grid.height):
        for c in range(grid.width):
            pos = Pos(r, c)
            tile = get_tile_at(grid, state, pos)

            if tile == "wall":
                fill_cell(ax, r, c, COLORS["wall"])

            elif tile == "key":
                key = grid.key_at(pos)
                kc = KEY_DOOR_COLORS.get(key.color, KEY_DOOR_COLORS["red"])
                fill_cell(ax, r, c, kc["key_fill"])
                _draw_key_icon(ax, r, c, key.color)

            elif tile == "door_closed":
                door = grid.door_at(pos)
                dc = _get_door_colors(door)
                fill_cell(ax, r, c, dc["door_closed"])
                _draw_door_bars(ax, r, c, door)

            elif tile == "door_open":
                door = grid.door_at(pos)
                dc = _get_door_colors(door)
                fill_cell(ax, r, c, dc["door_open"])
                # Dashed border
                rect = patches.Rectangle(
                    (c + 0.15, r + 0.15), 0.7, 0.7,
                    fill=False, edgecolor=dc["door_icon"],
                    linewidth=2, linestyle="--", zorder=6,
                )
                ax.add_patch(rect)

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

    # Highlight cell
    if highlight_cell is not None:
        draw_highlight_border(ax, highlight_cell.row, highlight_cell.col,
                              color=COLORS["highlight"], linewidth=3.5)

    if step_label is not None:
        ax.set_title(step_label, fontsize=9, pad=3)

    # Inventory HUD
    if show_inventory:
        _draw_inventory_hud(ax, state, grid.width, grid.height + 0.1)


# ---------------------------------------------------------------------------
# Public render functions
# ---------------------------------------------------------------------------

def render_state(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    title: Optional[str] = None,
    show_inventory: bool = True,
    highlight_cell: Optional[Pos] = None,
) -> Figure:
    """Render a single game state as a Matplotlib figure."""
    hud_height = 0.6 if show_inventory else 0.0
    fig, ax = create_grid_figure(grid.width, grid.height, cell_size=0.8,
                                  legend_height=hud_height)

    _draw_grid(ax, grid, state, highlight_cell=highlight_cell,
               show_inventory=show_inventory)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    fig.tight_layout()
    return fig


def render_trace(
    grid: ChainVaultGrid,
    initial_state: ChainVaultState,
    actions: list[ChainVaultAction],
    cols: int = 4,
    highlight_fatal: Optional[int] = None,
    title: Optional[str] = None,
) -> Figure:
    """Render a sequence of states as a multi-panel trace montage."""
    n_steps = len(actions) + 1
    rows = math.ceil(n_steps / cols)

    cell_size = 0.55
    fig_w = cols * (grid.width * cell_size + 0.3) + 0.5
    fig_h = rows * (grid.height * cell_size + 0.8) + 0.5

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

            hud_height = 0.5
            ax.set_xlim(-0.05, grid.width + 0.05)
            ax.set_ylim(-0.05 - hud_height, grid.height + 0.05)
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

            _draw_grid(ax, grid, state, highlight_cell=highlight,
                       step_label=label, show_inventory=True)

            if step_idx < len(actions):
                state = apply_action(grid, state, actions[step_idx])

            step_idx += 1

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0)

    fig.tight_layout()
    return fig


def render_side_by_side(
    grid: ChainVaultGrid,
    state_left: ChainVaultState,
    state_right: ChainVaultState,
    label_left: str = "A",
    label_right: str = "B",
    title: Optional[str] = None,
) -> Figure:
    """Render two states side by side for comparison (e.g., RQ2 pairs)."""
    cell_size = 0.7
    hud_height = 0.6
    fig_w = 2 * (grid.width * cell_size + 0.3) + 1.0
    fig_h = grid.height * cell_size + 1.2 + hud_height

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    for ax, state, label in [(ax_l, state_left, label_left),
                              (ax_r, state_right, label_right)]:
        ax.set_xlim(-0.05, grid.width + 0.05)
        ax.set_ylim(-0.05 - hud_height, grid.height + 0.05)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        _draw_grid(ax, grid, state, step_label=label, show_inventory=True)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0)

    fig.tight_layout()
    return fig
