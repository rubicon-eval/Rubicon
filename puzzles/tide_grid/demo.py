"""Tide Grid demo — generates example images for all templates at various grid sizes."""

from __future__ import annotations

import os

from puzzles.common.rendering import save_figure
from .engine import (
    TideGridGrid,
    TideGridState,
    TideGridAction,
    apply_action,
    is_solved,
)
from .oracle import (
    oracle_solve,
    apply_actions,
    find_fatal_step,
)
from .renderer import render_state, render_trace, render_side_by_side
from .generator import (
    DeadlockEntry,
    GeneratedInstance,
    GREEDY_STRATEGIES,
    TEMPLATE_NAMES,
    generate_validated_instance,
    generate_rq2_pair,
    generate_rq3_instance,
    discover_greedy_deadlocks,
    discover_oracle_adjacent_deadlocks,
    generate_hard_rq2_pair,
    generate_hard_rq3_instance,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demo_output")


# ---------------------------------------------------------------------------
# Per-template demo (up to 12 files per directory)
# ---------------------------------------------------------------------------

def _demo_template(
    grid: TideGridGrid,
    state: TideGridState,
    solution: list[TideGridAction],
    catalog: list[DeadlockEntry],
    out_dir: str,
) -> None:
    """Generate all demo images for one template instance into out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    # --- Initial state ---
    fig = render_state(grid, state, title="Initial State")
    save_figure(fig, os.path.join(out_dir, "initial.png"))

    # --- Oracle solution trace ---
    fig = render_trace(grid, state, solution, cols=4,
                       title=f"Oracle Solution ({len(solution)} steps)")
    save_figure(fig, os.path.join(out_dir, "solution_trace.png"))

    # --- RQ1: simulation fidelity ---
    n = min(5, len(solution))
    actions = solution[:n]
    gt_state = apply_actions(grid, state, actions)

    fig = render_side_by_side(
        grid, state, gt_state,
        label_left="Initial State",
        label_right=f"After {n} Actions (Ground Truth)",
        title=f"RQ1: Predict state after {n} actions",
    )
    save_figure(fig, os.path.join(out_dir, "rq1_side_by_side.png"))

    fig = render_trace(grid, state, actions, cols=3,
                       title=f"RQ1: {n}-Step Trace")
    save_figure(fig, os.path.join(out_dir, "rq1_trace.png"))

    # --- RQ2: foresight (deadlock/solvable pair) ---
    for entry in catalog:
        for k in [3, 2, 1, 5]:
            result = generate_rq2_pair(grid, state, entry, k)
            if result:
                s_start, dead_actions, solv_actions = result
                s_dead = apply_actions(grid, s_start, dead_actions)
                s_solv = apply_actions(grid, s_start, solv_actions)

                fig = render_side_by_side(
                    grid, s_dead, s_solv,
                    label_left=f"DEADLOCK (after {k} actions)",
                    label_right=f"SOLVABLE (after {k} actions)",
                    title=f"RQ2: Matched Pair (k={k})",
                )
                save_figure(fig, os.path.join(out_dir, "rq2_pair.png"))

                fig = render_state(grid, s_start,
                                   title=f"RQ2 Start State (before {k} actions)")
                save_figure(fig, os.path.join(out_dir, "rq2_start.png"))
                break
        else:
            continue
        break

    # --- RQ3: causal attribution (fatal step) ---
    for entry in catalog:
        result = generate_rq3_instance(grid, state, entry, target_length=10)
        if result:
            full_trace, fatal_step, gap = result

            verified_fatal = find_fatal_step(grid, state, full_trace)
            assert verified_fatal == fatal_step, \
                f"Fatal step mismatch: {verified_fatal} vs {fatal_step}"

            fig = render_trace(
                grid, state, full_trace, cols=4,
                highlight_fatal=fatal_step,
                title=f"RQ3: Fatal step at index {fatal_step} (gap={gap})",
            )
            save_figure(fig, os.path.join(out_dir, "rq3_trace.png"))
            break


# ---------------------------------------------------------------------------
# Demo all templates at a given grid size
# ---------------------------------------------------------------------------

def demo_all_templates(
    base_seed: int,
    width: int,
    height: int,
    subdir: str,
    max_retries: int = 200,
) -> None:
    """Generate demos for every template at a given grid size."""
    templates_dir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(templates_dir, exist_ok=True)

    for template in TEMPLATE_NAMES:
        instance = generate_validated_instance(
            width=width, height=height, template=template,
            seed=base_seed, max_retries=max_retries,
        )
        if instance is None:
            print(f"    {template}: generation failed — skipping")
            continue

        print(f"    {template}: solution={len(instance.solution)} steps, "
              f"deadlocks={len(instance.deadlock_catalog)}, "
              f"k_max={instance.max_k_horizon}, seed={instance.seed}, "
              f"oxygen={instance.grid.initial_oxygen}")

        tpl_dir = os.path.join(templates_dir, template)
        _demo_template(
            instance.grid,
            instance.initial_state,
            instance.solution,
            instance.deadlock_catalog,
            tpl_dir,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Tide Grid Demo")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Default / baseline: 7x7 (all 12 templates)
    print("\ntemplates/ (7x7, baseline):")
    demo_all_templates(base_seed=100, width=7, height=7, subdir="templates")

    # Smaller: 6x6 (strait + roundtrip + patience + airlock; tidal_maze needs 7x7 min odd, gauntlet/switchback need 7 height, confluence needs 7 width, tidal_bridge needs 7 height, deadline/tide_pool need 7x7, pendulum high rejection rate at 6x6)
    print("\ntemplates_6x6/:")
    demo_all_templates(base_seed=200, width=6, height=6, subdir="templates_6x6")

    # Larger: 8x8 (strait + gauntlet + roundtrip + confluence + patience + switchback + tidal_bridge + deadline + pendulum + tide_pool + airlock; tidal_maze needs odd dimensions)
    print("\ntemplates_8x8/:")
    demo_all_templates(base_seed=300, width=8, height=8, subdir="templates_8x8")

    print(f"\nAll demos complete! Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
