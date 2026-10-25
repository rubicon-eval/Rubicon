"""Tide Grid Matplotlib renderer — state, trace, and side-by-side views."""

from __future__ import annotations

import math
from typing import Optional

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from puzzles.common.types import Pos
from puzzles.common.rendering import (
    COLORS,
    create_grid_figure,
    fill_cell,
    draw_circle,
    draw_star,
    draw_text_in_cell,
    draw_highlight_border,
    draw_legend,
    save_figure,
    add_title,
)
from .engine import (
    Tile,
    TideGridGrid,
    TideGridState,
    TideGridAction,
    apply_action,
    is_solved,
    is_passable_at,
)


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

TIDE_COLORS = {
    "tidal_dry": "#D4A574",       # sandy tan when passable
    "tidal_flood": "#2471A3",     # deep blue when flooded
    "beacon": "#E67E22",          # orange beacon
    "beacon_collected": "#BDC3C7",  # gray when collected
    "oxygen_bar": "#27AE60",      # green oxygen bar
    "oxygen_low": "#E74C3C",      # red when oxygen low
    "zone_label": "#1A5276",      # dark blue zone label text
}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_beacon_icon(ax: Axes, row: int, col: int, color: str) -> None:
    """Draw a diamond beacon icon."""
    import matplotlib.patches as patches
    cx, cy = col + 0.5, row + 0.5
    size = 0.25
    diamond = patches.Polygon(
        [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)],
        facecolor=color,
        edgecolor="#A04000",
        linewidth=1.5,
        zorder=6,
    )
    ax.add_patch(diamond)


def _draw_oxygen_hud(
    ax: Axes,
    time: int,
    initial_oxygen: int,
    grid_width: int,
    y_start: float,
) -> None:
    """Draw oxygen remaining bar below the grid."""
    import matplotlib.patches as patches

    remaining = initial_oxygen - time
    fraction = remaining / initial_oxygen if initial_oxygen > 0 else 0
    bar_color = TIDE_COLORS["oxygen_bar"] if fraction > 0.3 else TIDE_COLORS["oxygen_low"]

    # Background bar
    bar_width = grid_width - 0.4
    bar_x = 0.2
    bg = patches.FancyBboxPatch(
        (bar_x, y_start), bar_width, 0.3,
        boxstyle="round,pad=0.02",
        facecolor="#E8E8E8", edgecolor="gray", linewidth=0.5,
    )
    ax.add_patch(bg)

    # Fill bar
    if fraction > 0:
        fill = patches.FancyBboxPatch(
            (bar_x, y_start), bar_width * fraction, 0.3,
            boxstyle="round,pad=0.02",
            facecolor=bar_color, edgecolor="none",
        )
        ax.add_patch(fill)

    # Label
    ax.text(
        grid_width / 2, y_start + 0.15,
        f"O\u2082: {remaining}/{initial_oxygen}  (t={time})",
        ha="center", va="center", fontsize=7,
        color=COLORS["text_dark"], weight="bold", zorder=10,
    )


# ---------------------------------------------------------------------------
# Single state rendering
# ---------------------------------------------------------------------------

def _render_state_on_ax(
    ax: Axes,
    grid: TideGridGrid,
    state: TideGridState,
    highlight_cell: Optional[Pos] = None,
    show_oxygen: bool = True,
) -> None:
    """Render a single state onto the given axes."""
    t = state.time

    # Draw tiles
    for r in range(grid.height):
        for c in range(grid.width):
            pos = Pos(r, c)
            tile = grid.tile_type(pos)

            if tile == Tile.WALL:
                fill_cell(ax, r, c, COLORS["wall"])
            elif tile == Tile.TIDAL:
                zone = grid.zone_at(pos)
                if zone.is_dry_at(t):
                    fill_cell(ax, r, c, TIDE_COLORS["tidal_dry"])
                    # Zone label
                    draw_text_in_cell(
                        ax, r, c, zone.zone_id,
                        fontsize=7, color=TIDE_COLORS["zone_label"],
                    )
                else:
                    fill_cell(ax, r, c, TIDE_COLORS["tidal_flood"])
                    draw_text_in_cell(
                        ax, r, c, zone.zone_id,
                        fontsize=7, color=COLORS["text_light"],
                    )
            elif tile == Tile.EXIT:
                fill_cell(ax, r, c, "#FCF3CF")
                draw_star(ax, r, c, COLORS["exit"], size=0.3)
            else:
                fill_cell(ax, r, c, COLORS["floor"])

    # Draw beacons
    for beacon in grid.beacons.values():
        if beacon.beacon_id in state.collected_beacons:
            # Show collected marker (faded)
            _draw_beacon_icon(ax, beacon.pos.row, beacon.pos.col, TIDE_COLORS["beacon_collected"])
        else:
            _draw_beacon_icon(ax, beacon.pos.row, beacon.pos.col, TIDE_COLORS["beacon"])

    # Draw agent
    agent_color = "#27AE60" if is_solved(grid, state) else COLORS["agent"]
    draw_circle(ax, state.agent_pos.row, state.agent_pos.col, agent_color, radius=0.28)

    # Highlight cell
    if highlight_cell is not None:
        draw_highlight_border(ax, highlight_cell.row, highlight_cell.col)

    # Oxygen HUD
    if show_oxygen:
        _draw_oxygen_hud(
            ax, state.time, grid.initial_oxygen,
            grid.width, grid.height + 0.1,
        )


def render_state(
    grid: TideGridGrid,
    state: TideGridState,
    title: Optional[str] = None,
    highlight_cell: Optional[Pos] = None,
) -> Figure:
    """Render a single game state as a Matplotlib figure."""
    fig, ax = create_grid_figure(
        grid.width, grid.height,
        title=title,
        legend_height=0.8,
    )
    _render_state_on_ax(ax, grid, state, highlight_cell=highlight_cell)
    return fig


# ---------------------------------------------------------------------------
# Trace rendering (multi-panel montage)
# ---------------------------------------------------------------------------

def render_trace(
    grid: TideGridGrid,
    initial_state: TideGridState,
    actions: list[TideGridAction],
    cols: int = 4,
    highlight_fatal: Optional[int] = None,
    title: Optional[str] = None,
) -> Figure:
    """Render sequence of states as a multi-panel trace montage."""
    import matplotlib.pyplot as plt

    n_panels = len(actions) + 1  # include start state
    rows = math.ceil(n_panels / cols)

    cell_w = grid.width * 0.55 + 0.3
    cell_h = grid.height * 0.55 + 1.0
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cell_w * cols, cell_h * rows),
    )
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    state = initial_state
    panel_idx = 0

    for ri in range(rows):
        for ci in range(cols):
            ax = axes[ri][ci]
            if panel_idx >= n_panels:
                ax.set_visible(False)
                panel_idx += 1
                continue

            ax.set_xlim(-0.05, grid.width + 0.05)
            ax.set_ylim(-0.05, grid.height + 0.95)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Determine highlight
            hl = None
            if panel_idx == 0:
                label = "Start"
            else:
                action = actions[panel_idx - 1]
                state = apply_action(grid, state, action)
                label = f"Step {panel_idx}: {action.label}"
                if highlight_fatal is not None and (panel_idx - 1) == highlight_fatal:
                    label += " [FATAL]"
                    hl = state.agent_pos

            _render_state_on_ax(ax, grid, state, highlight_cell=hl, show_oxygen=True)
            ax.set_title(label, fontsize=8, fontweight="bold", pad=2)

            panel_idx += 1

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Side-by-side rendering
# ---------------------------------------------------------------------------

def render_side_by_side(
    grid: TideGridGrid,
    state_left: TideGridState,
    state_right: TideGridState,
    label_left: str = "A",
    label_right: str = "B",
    title: Optional[str] = None,
) -> Figure:
    """Render two states side by side for comparison (e.g., RQ2 pairs)."""
    import matplotlib.pyplot as plt

    cell_w = grid.width * 0.8 + 0.4
    cell_h = grid.height * 0.8 + 1.2
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(cell_w * 2, cell_h))

    for ax, st, label in [(ax_l, state_left, label_left), (ax_r, state_right, label_right)]:
        ax.set_xlim(-0.05, grid.width + 0.05)
        ax.set_ylim(-0.05, grid.height + 0.95)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        _render_state_on_ax(ax, grid, st)
        ax.set_title(label, fontsize=10, fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Text grid for prompts
# ---------------------------------------------------------------------------

def render_text_grid(grid: TideGridGrid, state: TideGridState) -> str:
    """Render the grid as a labeled ASCII table for text prompts."""
    t = state.time
    lines = []
    # Header
    header = "     " + "  ".join(f"C{c}" for c in range(grid.width))
    lines.append(header)

    for r in range(grid.height):
        row_str = f"R{r}  "
        cells = []
        for c in range(grid.width):
            pos = Pos(r, c)
            tile = grid.tile_type(pos)

            if pos == state.agent_pos:
                cells.append(" @")
            elif tile == Tile.WALL:
                cells.append(" #")
            elif tile == Tile.EXIT:
                cells.append(" E")
            elif tile == Tile.TIDAL:
                zone = grid.zone_at(pos)
                if zone.is_dry_at(t):
                    cells.append(f"~{zone.zone_id}")
                else:
                    cells.append(f"#{zone.zone_id}")
            else:
                # Check for beacon
                beacon = grid.beacon_at(pos)
                if beacon and beacon.beacon_id not in state.collected_beacons:
                    cells.append(f" {beacon.beacon_id[-1]}")
                else:
                    cells.append(" .")
        row_str += "  ".join(cells)
        lines.append(row_str)

    return "\n".join(lines)


def _zone_schedule_str(zone: "TidalZone", max_time: int) -> str:
    """Build a human-readable schedule string for a tidal zone.

    Example: "dry at t=0, flooded t=1-3, dry t=4-6, flooded t=7-9, ..."
    """
    if max_time <= 0:
        return ""
    segments: list[str] = []
    t = 0
    while t <= max_time:
        is_dry = zone.is_dry_at(t)
        label = "dry" if is_dry else "flooded"
        start = t
        while t + 1 <= max_time and zone.is_dry_at(t + 1) == is_dry:
            t += 1
        if start == t:
            segments.append(f"{label} at t={start}")
        else:
            segments.append(f"{label} t={start}\u2013{t}")
        t += 1
    # Add ellipsis to indicate the pattern repeats
    return ", ".join(segments) + ", ..."


def generate_rules_text(grid: "TideGridGrid") -> str:
    """Generate text description of instance details for prompts."""
    lines = []

    lines.append("Coordinates are (row, col) where row 0 is the top row and col 0 is the leftmost column")
    lines.append("Start Time: t = 0")

    # Grid dimensions
    lines.append(f"Grid: {grid.width} columns x {grid.height} rows")
    lines.append(f"Oxygen: {grid.initial_oxygen}")
    lines.append(f"Start: row {grid.start.row}, col {grid.start.col}")
    lines.append(f"Exit: row {grid.exit_pos.row}, col {grid.exit_pos.col}")
    lines.append("")

    # Tidal zones — explicit schedule, no raw formula
    lines.append("Tidal Zones:")
    # Show schedule up to oxygen limit (that's the max relevant time)
    schedule_horizon = grid.initial_oxygen + 2
    for zone in sorted(grid.tidal_zones.values(), key=lambda z: z.zone_id):
        tiles_str = ", ".join(
            f"({p.row},{p.col})" for p in sorted(zone.tiles)
        )
        schedule = _zone_schedule_str(zone, schedule_horizon)
        lines.append(
            f"  Zone {zone.zone_id} (cells: {tiles_str}): "
            f"repeats every {zone.cycle_length} turns — "
            f"{zone.dry_duration} turns dry, then "
            f"{zone.flood_duration} turns flooded."
        )
        lines.append(f"    Schedule: {schedule}")
    lines.append("")

    # Beacons
    lines.append("Beacons:")
    for b in sorted(grid.beacons.values(), key=lambda b: b.beacon_id):
        lines.append(f"  {b.beacon_id}: row {b.pos.row}, col {b.pos.col}")

    return "\n".join(lines)
