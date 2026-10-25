"""Chain Vault demo — generates puzzle images and RQ illustrations.

Produces all outputs to chain_vault/demo_output/templates/<template_name>/:
  initial.png          — puzzle initial state
  solution_trace.png   — oracle solution step-by-step
  rq1_side_by_side.png — initial vs. after N actions
  rq1_trace.png        — N-action sequence step-by-step
  rq2_start.png        — starting state for deadlock/solvable pair
  rq2_pair.png         — deadlock vs. solvable after k actions
  rq3_trace.png        — full trace with fatal step highlighted

Usage:
    uv run python -m puzzles.chain_vault.demo
"""

from __future__ import annotations

import os

# Headless rendering
import matplotlib
matplotlib.use("Agg")

from puzzles.common.rendering import save_figure
from puzzles.chain_vault.engine import (
    ChainVaultGrid,
    ChainVaultState,
    is_solved,
    _inventory_to_dict,
)
from puzzles.chain_vault.oracle import (
    oracle_solve,
    apply_actions,
    is_solvable,
    is_walking_dead,
    find_fatal_step,
)
from puzzles.chain_vault.renderer import (
    render_state,
    render_trace,
    render_side_by_side,
)
from puzzles.chain_vault.generator import (
    generate_validated_instance,
    generate_rq2_pair,
    generate_rq3_instance,
    TEMPLATE_NAMES,
)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demo_output")


def _demo_template(
    grid: ChainVaultGrid,
    state: ChainVaultState,
    catalog,
    tpl_dir: str,
) -> None:
    """Generate all demo images for one template instance into tpl_dir."""

    # --- Initial state ---
    fig = render_state(grid, state, title="Initial State")
    save_figure(fig, os.path.join(tpl_dir, "initial.png"))

    # --- Oracle solution ---
    solution = oracle_solve(grid, state)
    if solution is None:
        print("    ERROR: Puzzle is not solvable!")
        return

    final = apply_actions(grid, state, solution)
    assert is_solved(grid, final), "Oracle solution does not reach exit!"

    fig = render_trace(grid, state, solution, cols=4,
                       title=f"Oracle Solution ({len(solution)} steps)")
    save_figure(fig, os.path.join(tpl_dir, "solution_trace.png"))

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
    save_figure(fig, os.path.join(tpl_dir, "rq1_side_by_side.png"))

    fig = render_trace(grid, state, actions, cols=3,
                       title=f"RQ1: {n}-Step Trace")
    save_figure(fig, os.path.join(tpl_dir, "rq1_trace.png"))

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
                save_figure(fig, os.path.join(tpl_dir, "rq2_pair.png"))

                fig = render_state(grid, s_start,
                                   title=f"RQ2 Start State (before {k} actions)")
                save_figure(fig, os.path.join(tpl_dir, "rq2_start.png"))
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
            save_figure(fig, os.path.join(tpl_dir, "rq3_trace.png"))
            break


def demo_all_templates(
    base_seed: int = 100,
    width: int = 5,
    height: int = 5,
    subdir: str = "templates",
) -> None:
    """Generate demo images for every implemented template."""
    templates_dir = os.path.join(OUTPUT_DIR, subdir)

    for i, template in enumerate(TEMPLATE_NAMES):
        seed = base_seed + i * 1000

        instance = generate_validated_instance(
            width=width, height=height, template=template, seed=seed,
        )
        if instance is None:
            print(f"  WARNING: Could not generate {template}, skipping")
            continue

        tpl_dir = os.path.join(templates_dir, template)
        os.makedirs(tpl_dir, exist_ok=True)

        print(f"  {template}: solution={len(instance.solution)} steps, "
              f"deadlocks={len(instance.deadlock_catalog)}, seed={instance.seed}")

        _demo_template(
            instance.grid,
            instance.initial_state,
            instance.deadlock_catalog,
            tpl_dir,
        )


def main() -> None:
    print("Chain Vault Demo")
    print("=" * 60)

    demo_all_templates(base_seed=100, width=5, height=5, subdir="templates")
    demo_all_templates(base_seed=200, width=6, height=6, subdir="templates_6x6")
    demo_all_templates(base_seed=300, width=7, height=7, subdir="templates_7x7")
    demo_all_templates(base_seed=400, width=8, height=8, subdir="templates_8x8")

    print(f"\nAll demos complete! Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
