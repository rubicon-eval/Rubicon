"""Shared Matplotlib grid drawing helpers for all puzzle renderers."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Colorblind-friendly palette
COLORS = {
    "floor": "#E8E8E8",
    "wall": "#2C3E50",
    "agent": "#2980B9",
    "exit": "#F1C40F",
    "painted": "#5DADE2",
    "plate_inactive": "#F9E79F",
    "plate_active": "#85C1E9",
    "door_closed": "#E74C3C",
    "door_open": "#82E0AA",
    "highlight": "#E74C3C",
    "text_dark": "#2C3E50",
    "text_light": "#FDFEFE",
    "bridge": "#D4A574",
    "void": "#1A1A2E",
    "fuse_unarmed": "#E67E22",
    "fuse_armed": "#E74C3C",
    "key_red": "#E74C3C",
    "key_blue": "#3498DB",
    "key_green": "#27AE60",
    "key_yellow": "#F1C40F",
}

# Distinct color pairs for plate-door linkage groups (colorblind-friendly).
# Each entry: (plate_fill, plate_icon, door_closed, door_open, color_name)
LINKAGE_PALETTE = [
    ("#AED6F1", "#2471A3", "#2471A3", "#A9CCE3", "blue"),
    ("#A9DFBF", "#1E8449", "#1E8449", "#A9DFBF", "green"),
    ("#F5CBA7", "#CA6F1E", "#CA6F1E", "#F5CBA7", "orange"),
    ("#D7BDE2", "#7D3C98", "#7D3C98", "#D7BDE2", "purple"),
]


def get_linkage_color(group_index: int) -> dict[str, str]:
    """Return color dict for a linkage group by index."""
    entry = LINKAGE_PALETTE[group_index % len(LINKAGE_PALETTE)]
    return {
        "plate_fill": entry[0],
        "plate_icon": entry[1],
        "door_closed": entry[2],
        "door_open": entry[3],
        "name": entry[4],
    }


def create_grid_figure(
    width: int,
    height: int,
    cell_size: float = 0.8,
    title: Optional[str] = None,
    legend_height: float = 0.0,
) -> tuple[Figure, Axes]:
    """Create a figure and axes for grid rendering."""
    fig_w = width * cell_size + 0.4
    fig_h = height * cell_size + 0.4 + legend_height
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_xlim(-0.05, width + 0.05)
    ax.set_ylim(-0.05 - legend_height, height + 0.05)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        add_title(fig, title)
    return fig, ax


def fill_cell(
    ax: Axes,
    row: int,
    col: int,
    color: str,
    edge_color: str = "white",
    linewidth: float = 1.5,
) -> None:
    """Draw a solid color rectangle for a cell."""
    rect = patches.FancyBboxPatch(
        (col + 0.02, row + 0.02),
        0.96,
        0.96,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=edge_color,
        linewidth=linewidth,
    )
    ax.add_patch(rect)


def draw_circle(
    ax: Axes,
    row: int,
    col: int,
    color: str,
    radius: float = 0.3,
    edge_color: str = "white",
    linewidth: float = 2.0,
) -> None:
    """Draw a circle icon (used for agent, plates)."""
    circle = patches.Circle(
        (col + 0.5, row + 0.5),
        radius,
        facecolor=color,
        edgecolor=edge_color,
        linewidth=linewidth,
        zorder=5,
    )
    ax.add_patch(circle)


def draw_star(
    ax: Axes,
    row: int,
    col: int,
    color: str,
    size: float = 0.3,
) -> None:
    """Draw a star icon (used for exit)."""
    import numpy as np

    cx, cy = col + 0.5, row + 0.5
    angles = np.linspace(0, 2 * np.pi, 11)[:-1]
    radii = [size if i % 2 == 0 else size * 0.4 for i in range(10)]
    xs = [cx + r * np.sin(a) for a, r in zip(angles, radii)]
    ys = [cy - r * np.cos(a) for a, r in zip(angles, radii)]
    star = patches.Polygon(
        list(zip(xs, ys)),
        facecolor=color,
        edgecolor="#B7950B",
        linewidth=1.5,
        zorder=5,
    )
    ax.add_patch(star)


def draw_diamond(
    ax: Axes,
    row: int,
    col: int,
    color: str,
    size: float = 0.3,
) -> None:
    """Draw a diamond icon (used for plates)."""
    cx, cy = col + 0.5, row + 0.5
    diamond = patches.Polygon(
        [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)],
        facecolor=color,
        edgecolor="#7D6608",
        linewidth=1.5,
        zorder=5,
    )
    ax.add_patch(diamond)


def draw_text_in_cell(
    ax: Axes,
    row: int,
    col: int,
    text: str,
    fontsize: int = 9,
    color: str = "#2C3E50",
    weight: str = "bold",
) -> None:
    """Draw centered text in a cell."""
    ax.text(
        col + 0.5,
        row + 0.5,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        weight=weight,
        zorder=10,
    )


def draw_highlight_border(
    ax: Axes,
    row: int,
    col: int,
    color: str = "#E74C3C",
    linewidth: float = 3.0,
) -> None:
    """Draw a colored border around a cell (for highlighting fatal steps)."""
    rect = patches.Rectangle(
        (col + 0.01, row + 0.01),
        0.98,
        0.98,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        zorder=15,
    )
    ax.add_patch(rect)


def draw_legend(
    ax: Axes,
    entries: list[tuple[str, str]],
    y_offset: float = -0.3,
) -> None:
    """Draw a legend below the grid. entries = [(color, label), ...]."""
    n = len(entries)
    total_width = n * 1.8
    start_x = (ax.get_xlim()[1] - total_width) / 2
    if start_x < 0:
        start_x = 0.1

    for i, (color, label) in enumerate(entries):
        x = start_x + i * 1.8
        rect = patches.Rectangle(
            (x, ax.get_ylim()[0] + y_offset + 0.05),
            0.3,
            0.3,
            facecolor=color,
            edgecolor="gray",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.4,
            ax.get_ylim()[0] + y_offset + 0.2,
            label,
            fontsize=7,
            va="center",
            color=COLORS["text_dark"],
        )


def add_title(fig: Figure, text: str) -> None:
    """Add a title to the figure."""
    fig.suptitle(text, fontsize=12, fontweight="bold", y=0.98)


def save_figure(fig: Figure, path: str, dpi: int = 200) -> None:
    """Save figure to file."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
