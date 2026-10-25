#!/usr/bin/env python3
"""Generate publication-quality analysis plots for a puzzle type.

Produces: per-RQ bar charts, combined grouped bars, heatmap, Pareto curves.
Uses colorblind-safe palette (Paul Tol's vibrant), distinct markers, and
adjustText for non-overlapping labels.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
from adjustText import adjust_text

ROOT_DIR = Path(__file__).parent.parent
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"

# ── Publication rcParams ────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.4,
    "lines.markersize": 8,
    "axes.grid": False,
    "figure.constrained_layout.use": True,
})

# ── Config ──────────────────────────────────────────────────────────────────

RQ_ORDER = [
    "rq1_main",
    "rq2_hard_ablation_deadlock",
    "rq2_hard_ablation_solvable",
    "rq2_hard_deadlock",
    "rq2_hard_solvable",
    "rq3_hard",
    "rq3_hard_ablation_final",
]

RQ_LABELS = {
    "rq1_main": "RQ1:\nSimulation",
    "rq2_hard_ablation_deadlock": "RQ2: Deadlock\n(ablation)",
    "rq2_hard_ablation_solvable": "RQ2: Solvable\n(ablation)",
    "rq2_hard_deadlock": "RQ2:\nDeadlock",
    "rq2_hard_solvable": "RQ2:\nSolvable",
    "rq3_hard": "RQ3:\nAttribution",
    "rq3_hard_ablation_final": "RQ3: Attribution\n(ablation)",
}

RQ_SHORT = {
    "rq1_main": "rq1-main",
    "rq2_hard_ablation_deadlock": "rq2-hard-ablation-deadlock",
    "rq2_hard_ablation_solvable": "rq2-hard-ablation-solvable",
    "rq2_hard_deadlock": "rq2-hard-deadlock",
    "rq2_hard_solvable": "rq2-hard-solvable",
    "rq3_hard": "rq3-hard",
    "rq3_hard_ablation_final": "rq3-hard-ablation-final",
}

MODEL_DISPLAY = {
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemma-3-12b-it": "Gemma 12B",
    "gemma-3-27b-it": "Gemma 27B",
    "glm-4.6v": "GLM-4.6V",
    "gpt-5.4": "GPT-5.4",
    "kimi-k2.5": "Kimi-K2.5",
    "ministral-3-14b-instruct-2512": "Ministral 14B",
    "ministral-3-8b-instruct-2512": "Ministral 8B",
    "qwen3-vl-8b-instruct": "Qwen3-VL 8B",
    "qwen3-vl-235b-a22b-instruct": "Qwen3-VL 235B",
    "qwen3.5-397b-a17b-fp8": "Qwen3.5",
}

MODEL_ORDER = [
    "gemini-3.1-pro-preview",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "glm-4.6v",
    "gpt-5.4",
    "kimi-k2.5",
    "ministral-3-14b-instruct-2512",
    "ministral-3-8b-instruct-2512",
    "qwen3-vl-8b-instruct",
    "qwen3-vl-235b-a22b-instruct",
    "qwen3.5-397b-a17b-fp8",
]

# Paul Tol's vibrant palette — colorblind-safe, 11 distinct colours
# https://personal.sron.nl/~pault/#sec:qualitative
_TOL_VIBRANT = [
    "#0077BB",  # blue
    "#33BBEE",  # cyan
    "#009988",  # teal
    "#332288",  # indigo
    "#EE7733",  # orange
    "#CC3311",  # red
    "#EE3377",  # magenta
    "#BBBBBB",  # grey
    "#AA3377",  # purple
    "#44AA99",  # mint
    "#88CCEE",  # light cyan
]

MODEL_COLORS = {m: c for m, c in zip(MODEL_ORDER, _TOL_VIBRANT)}

# Distinct markers so models are distinguishable without colour alone
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "p", "d"]
MODEL_MARKERS = {m: mk for m, mk in zip(MODEL_ORDER, _MARKERS)}

PUZZLE_TITLE = {
    "tide_grid": "Tide Grid",
    "mosaic_grid": "Mosaic Grid",
    "chain_vault": "Chain Vault",
}


# ── Data loading ────────────────────────────────────────────────────────────

def load_results(puzzle: str):
    base = RESULTS_DIR / puzzle
    data = {}
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        data[model] = {}
        for rq_file in model_dir.glob("*.jsonl"):
            with open(rq_file) as f:
                summary = json.loads(f.readline())
            data[model][rq_file.stem] = {
                "accuracy": summary["accuracy"],
                "avg_tokens": summary["avg_output_tokens"],
                "total": summary["total"],
                "correct": summary["correct"],
            }
    return data


def _title(puzzle: str) -> str:
    return PUZZLE_TITLE.get(puzzle, puzzle.replace("_", " ").title())


def _save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


# ── Individual bar charts per RQ ────────────────────────────────────────────

def make_bar_per_rq(data, puzzle, out_dir):
    for rq in RQ_ORDER:
        models = [m for m in MODEL_ORDER if m in data and rq in data[m]]
        if not models:
            continue
        accs = [data[m][rq]["accuracy"] for m in models]
        labels = [MODEL_DISPLAY.get(m, m) for m in models]
        colors = [MODEL_COLORS[m] for m in models]

        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        bars = ax.bar(range(len(models)), accs, color=colors,
                       edgecolor="white", linewidth=0.5, width=0.65, zorder=3)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 112)
        ax.set_title(f"{_title(puzzle)} \u2014 {RQ_LABELS[rq].replace(chr(10), ' ')}",
                     fontweight="bold")
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.grid(axis="y", alpha=0.25, linestyle="-", zorder=0)

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{acc:.0f}", ha="center", va="bottom", fontsize=9,
                    fontweight="medium")

        _save(fig, out_dir, f"bar_{RQ_SHORT[rq]}")
    print("  Per-RQ bar charts saved.")


# ── Combined grouped bar chart ──────────────────────────────────────────────

def make_bars_combined(data, puzzle, out_dir):
    models = [m for m in MODEL_ORDER if m in data]
    n_models = len(models)
    n_rqs = len(RQ_ORDER)
    width = 0.82 / n_models

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, m in enumerate(models):
        accs = [data[m].get(rq, {}).get("accuracy", 0) for rq in RQ_ORDER]
        x = np.arange(n_rqs) + i * width - (n_models - 1) * width / 2
        ax.bar(x, accs, width=width, label=MODEL_DISPLAY[m],
               color=MODEL_COLORS[m], edgecolor="white", linewidth=0.3, zorder=3)

    ax.set_xticks(np.arange(n_rqs))
    ax.set_xticklabels([RQ_LABELS[rq] for rq in RQ_ORDER], fontsize=9.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="y", alpha=0.2, linestyle="-", zorder=0)
    ax.set_title(f"{_title(puzzle)} \u2014 Model Performance by Research Question",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", ncol=4, framealpha=0.92, edgecolor="none",
              columnspacing=0.8, handlelength=1.2, fontsize=9.5)

    _save(fig, out_dir, "bars")
    print("  Combined bar chart saved.")


# ── Heatmap ─────────────────────────────────────────────────────────────────

def make_heatmap(data, puzzle, out_dir):
    models = [m for m in MODEL_ORDER if m in data]
    n_m, n_r = len(models), len(RQ_ORDER)

    matrix = np.full((n_m, n_r), np.nan)
    annotations = [[""]*n_r for _ in range(n_m)]
    for i, m in enumerate(models):
        for j, rq in enumerate(RQ_ORDER):
            if rq in data.get(m, {}):
                v = data[m][rq]["accuracy"]
                matrix[i, j] = v
                annotations[i][j] = f"{v:.0f}%"
            else:
                annotations[i][j] = "\u2014"

    # Colorblind-safe brick-emerald diverging: red (low) -> off-white (mid) -> green (high)
    cmap = mcolors.LinearSegmentedColormap.from_list("be", [
        "#B83028",   # brick red
        "#DC8070",   # muted salmon
        "#FAFAF5",   # off-white
        "#7CC09A",   # sage green
        "#1A7A45",   # emerald
    ])
    cmap.set_bad(color="#ECECEC")

    cell_w, cell_h = 1.25, 0.58
    fig, ax = plt.subplots(figsize=(n_r * cell_w + 2.2, n_m * cell_h + 1.2))

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Grid lines between cells
    for x in np.arange(-0.5, n_r, 1):
        ax.axvline(x, color="white", linewidth=2.0)
    for y in np.arange(-0.5, n_m, 1):
        ax.axhline(y, color="white", linewidth=2.0)

    ax.set_xticks(range(n_r))
    ax.set_xticklabels([RQ_LABELS[rq] for rq in RQ_ORDER],
                       fontsize=9.5, linespacing=0.9)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([MODEL_DISPLAY[m] for m in models], fontsize=11)

    # Cell annotations with luminance-adaptive contrast
    for i in range(n_m):
        for j in range(n_r):
            v = matrix[i, j]
            if np.isnan(v):
                color = "#888888"
            else:
                rgba = cmap(v / 100.0)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "white" if lum < 0.45 else "black"
            ax.text(j, i, annotations[i][j], ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.015, aspect=25)
    cbar.set_label("Accuracy (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_linewidth(0.4)

    ax.set_title(f"{_title(puzzle)} \u2014 Accuracy by Model and Research Question",
                 fontsize=13, fontweight="bold", pad=10)

    # Turn off default spines (heatmap doesn't need them)
    for spine in ax.spines.values():
        spine.set_visible(False)

    _save(fig, out_dir, "heatmap")
    print("  Heatmap saved.")


# ── Pareto efficiency plots ─────────────────────────────────────────────────

def _plot_pareto_scatter(ax, models, data, rqs_filter=None):
    """Plot scatter points with markers+colours and return texts list for adjustText."""
    texts = []
    xs, ys = [], []
    for m in models:
        rqs = rqs_filter or [rq for rq in RQ_ORDER if rq in data.get(m, {})]
        avail = [rq for rq in rqs if rq in data.get(m, {})]
        if not avail:
            continue
        avg_acc = np.mean([data[m][rq]["accuracy"] for rq in avail])
        avg_tok = np.mean([data[m][rq]["avg_tokens"] for rq in avail])
        xs.append(avg_tok)
        ys.append(avg_acc)
        ax.scatter(avg_tok, avg_acc, s=130, color=MODEL_COLORS[m],
                   marker=MODEL_MARKERS[m], label=MODEL_DISPLAY[m],
                   zorder=5, edgecolors="black", linewidth=0.4)
        texts.append(ax.text(avg_tok, avg_acc, MODEL_DISPLAY[m],
                             fontsize=9.5, ha="left", va="bottom"))
    return texts, xs, ys


def _draw_pareto_frontier(ax, xs, ys):
    """Draw Pareto frontier as dashed step line."""
    if len(xs) < 2:
        return
    pts = sorted(zip(xs, ys))
    frontier = []
    max_acc = -1
    for tok, acc in pts:
        if acc > max_acc:
            frontier.append((tok, acc))
            max_acc = acc
    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "--", color="#666666", alpha=0.5, linewidth=1.0,
                zorder=1, label="Pareto frontier")


def make_pareto(data, puzzle, out_dir):
    models = [m for m in MODEL_ORDER if m in data]

    for log_scale in (False, True):
        suffix = "log" if log_scale else "linear"
        fig, ax = plt.subplots(figsize=(6.0, 5.0))

        texts, xs, ys = _plot_pareto_scatter(ax, models, data)
        _draw_pareto_frontier(ax, xs, ys)

        if log_scale:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{x:,.0f}"))
        ax.set_xlabel("Average Output Tokens")
        ax.set_ylabel("Average Accuracy (%)")
        ax.set_title(f"{_title(puzzle)}: Pareto Efficiency "
                     f"({'Log' if log_scale else 'Linear'} Scale)",
                     fontweight="bold")
        ax.grid(True, alpha=0.15, linestyle="-")
        ax.set_axisbelow(True)

        # Legend with markers
        handles, labs = ax.get_legend_handles_labels()
        ax.legend(handles, labs, loc="best", framealpha=0.92, edgecolor="none",
                  handletextpad=0.4, borderpad=0.4, fontsize=9.5)

        # Non-overlapping labels
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#999999",
                    lw=0.5), expand=(1.3, 1.4), force_text=(0.8, 0.8),
                    ensure_inside_axes=True)

        _save(fig, out_dir, f"pareto_{suffix}")

    # ── Per-RQ Pareto subplots ──────────────────────────────────────────────
    for log_scale in (False, True):
        suffix = "log" if log_scale else "linear"
        rqs_with_data = [rq for rq in RQ_ORDER
                         if any(rq in data.get(m, {}) for m in models)]
        n_rqs = len(rqs_with_data)
        cols = min(4, n_rqs)
        rows = (n_rqs + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 3.2))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, rq in enumerate(rqs_with_data):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            texts = []
            for m in models:
                if rq not in data.get(m, {}):
                    continue
                acc = data[m][rq]["accuracy"]
                tok = data[m][rq]["avg_tokens"]
                ax.scatter(tok, acc, s=65, color=MODEL_COLORS[m],
                           marker=MODEL_MARKERS[m], edgecolors="black",
                           linewidth=0.3, zorder=5)
                texts.append(ax.text(tok, acc, MODEL_DISPLAY[m],
                                     fontsize=7, ha="left", va="bottom"))
            if log_scale:
                ax.set_xscale("log")
            ax.set_title(RQ_LABELS[rq].replace("\n", " "), fontsize=10,
                         fontweight="bold", pad=4)
            ax.set_ylim(-5, 110)
            ax.grid(True, alpha=0.12, linestyle="-")
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8)
            if r == rows - 1:
                ax.set_xlabel("Tokens", fontsize=9.5)
            if c == 0:
                ax.set_ylabel("Accuracy (%)", fontsize=9.5)

            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.4),
                        expand=(1.4, 1.5), force_text=(0.6, 0.6),
                        ensure_inside_axes=True)

        # Hide unused subplots
        for idx in range(n_rqs, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].set_visible(False)

        fig.suptitle(f"{_title(puzzle)}: Per-RQ Pareto "
                     f"({'Log' if log_scale else 'Linear'})",
                     fontsize=13, fontweight="bold")

        _save(fig, out_dir, f"pareto_per_rq_{suffix}")

    print("  Pareto plots saved.")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle", required=True)
    args = parser.parse_args()

    puzzle = args.puzzle
    out_dir = PLOTS_DIR / puzzle
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results for {puzzle}...")
    data = load_results(puzzle)
    print(f"Found {len(data)} models.")

    make_bar_per_rq(data, puzzle, out_dir)
    make_bars_combined(data, puzzle, out_dir)
    make_heatmap(data, puzzle, out_dir)
    make_pareto(data, puzzle, out_dir)
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
