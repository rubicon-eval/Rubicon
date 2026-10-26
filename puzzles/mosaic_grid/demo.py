"""Mosaic Grid demo — generates example puzzles and RQ illustrations.

Produces all outputs to mosaic_grid/demo_output/:
1. Normal puzzle demo: initial state + oracle solution trace
2. RQ1 example: initial state + action sequence + ground-truth final state
3. RQ2 example: matched deadlock/solvable pair from same start
4. RQ3 example: full trace with fatal step highlighted

Usage:
    uv run python -m puzzles.mosaic_grid.demo
"""

from __future__ import annotations

import os
import sys

# Headless rendering
import matplotlib
matplotlib.use("Agg")

from puzzles.common.types import Direction
from puzzles.common.rendering import save_figure
from puzzles.mosaic_grid.engine import (
    MosaicGrid,
    MosaicState,
    apply_action,
    get_legal_actions,
    is_solved,
    decode_painted,
    decode_activated,
)
from puzzles.mosaic_grid.oracle import (
    oracle_solve,
    apply_actions,
    is_solvable,
    is_walking_dead,
    find_fatal_step,
)
from puzzles.mosaic_grid.renderer import (
    render_state,
    render_trace,
    render_side_by_side,
    generate_rules_text,
)
from puzzles.mosaic_grid.generator import (
    generate_validated_instance,
    generate_rq2_pair,
    generate_rq3_instance,
    simulate_greedy,
    TEMPLATE_NAMES,
)
from puzzles.mosaic_grid.prompts import (
    build_rq1_prompt,
    build_rq1_ablation_prompt,
    build_rq2_main_prompt,
    build_rq2_ablation_prompt,
    build_rq3_prompt,
)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demo_output")


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_normal_puzzle(grid: MosaicGrid, state: MosaicState, prefix: str = "normal") -> None:
    """Generate normal puzzle demo: initial state + oracle solution trace."""
    print(f"\n{'='*60}")
    print(f"  Normal Puzzle Demo ({prefix})")
    print(f"{'='*60}")

    # Render initial state
    fig = render_state(grid, state, title="Initial State")
    path = os.path.join(OUTPUT_DIR, f"{prefix}_initial.png")
    save_figure(fig, path)
    print(f"  Saved: {path}")

    # Solve
    solution = oracle_solve(grid, state)
    if solution is None:
        print("  ERROR: Puzzle is not solvable!")
        return

    print(f"  Oracle solution ({len(solution)} steps): {[a.name for a in solution]}")

    # Print rules text (what would be sent to the model)
    rules = generate_rules_text(grid)
    print(f"\n  Model prompt rules:\n  {rules.replace(chr(10), chr(10) + '  ')}")

    # Verify solution
    final = apply_actions(grid, state, solution)
    assert is_solved(grid, final), "Oracle solution does not reach exit!"

    # Render trace
    fig = render_trace(grid, state, solution, cols=4,
                       title=f"Oracle Solution ({len(solution)} steps)")
    path = os.path.join(OUTPUT_DIR, f"{prefix}_solution_trace.png")
    save_figure(fig, path)
    print(f"  Saved: {path}")


def demo_rq1(grid: MosaicGrid, state: MosaicState, prefix: str = "rq1") -> None:
    """RQ1 example: initial state + 5-step action sequence + ground truth."""
    print(f"\n{'='*60}")
    print(f"  RQ1 Demo: Simulation Fidelity ({prefix})")
    print(f"{'='*60}")

    solution = oracle_solve(grid, state)
    if solution is None:
        print("  ERROR: Not solvable!")
        return

    n = min(5, len(solution))
    actions = solution[:n]
    gt_state = apply_actions(grid, state, actions)

    print(f"  Actions ({n} steps): {[a.name for a in actions]}")
    print(f"  Ground truth agent_pos: {gt_state.agent_pos}")
    print(f"  Ground truth painted: {bin(gt_state.painted).count('1')} tiles")
    print(f"  Ground truth plates: {decode_activated(gt_state.activated_plates, grid._bit_to_pid)}")

    # Render initial + final side by side
    fig = render_side_by_side(
        grid, state, gt_state,
        label_left="Initial State",
        label_right=f"After {n} Actions (Ground Truth)",
        title=f"RQ1: Predict state after {n} actions",
    )
    path = os.path.join(OUTPUT_DIR, f"{prefix}_side_by_side.png")
    save_figure(fig, path)
    print(f"  Saved: {path}")

    # Render step-by-step trace
    fig = render_trace(grid, state, actions, cols=3,
                       title=f"RQ1: {n}-Step Trace")
    path = os.path.join(OUTPUT_DIR, f"{prefix}_trace.png")
    save_figure(fig, path)
    print(f"  Saved: {path}")


def demo_rq2(
    grid: MosaicGrid,
    state: MosaicState,
    catalog,
    prefix: str = "rq2",
) -> None:
    """RQ2 example: matched deadlock/solvable pair."""
    print(f"\n{'='*60}")
    print(f"  RQ2 Demo: Foresight ({prefix})")
    print(f"{'='*60}")

    # Try different k values and catalog entries
    for entry in catalog:
        for k in [3, 2, 1, 5]:
            result = generate_rq2_pair(grid, state, entry, k)
            if result:
                s_start, dead_actions, solv_actions = result

                # Apply actions to get final states
                s_dead = apply_actions(grid, s_start, dead_actions)
                s_solv = apply_actions(grid, s_start, solv_actions)

                print(f"  k={k}")
                print(f"  Start: {s_start.agent_pos}")
                print(f"  Deadlock actions: {[a.name for a in dead_actions]}")
                print(f"  Solvable actions: {[a.name for a in solv_actions]}")
                print(f"  Dead state walking_dead: {is_walking_dead(grid, s_dead)}")
                print(f"  Solv state solvable: {is_solvable(grid, s_solv)}")

                # Render the pair
                fig = render_side_by_side(
                    grid, s_dead, s_solv,
                    label_left=f"DEADLOCK (after {k} actions)",
                    label_right=f"SOLVABLE (after {k} actions)",
                    title=f"RQ2: Matched Pair (k={k})",
                )
                path = os.path.join(OUTPUT_DIR, f"{prefix}_pair_k{k}.png")
                save_figure(fig, path)
                print(f"  Saved: {path}")

                # Also render the start state
                fig = render_state(grid, s_start,
                                   title=f"RQ2 Start State (before {k} actions)")
                path = os.path.join(OUTPUT_DIR, f"{prefix}_start_k{k}.png")
                save_figure(fig, path)
                print(f"  Saved: {path}")
                return

    print("  WARNING: Could not generate RQ2 pair from catalog!")


def demo_rq3(
    grid: MosaicGrid,
    state: MosaicState,
    catalog,
    prefix: str = "rq3",
) -> None:
    """RQ3 example: full trace with fatal step highlighted."""
    print(f"\n{'='*60}")
    print(f"  RQ3 Demo: Causal Attribution ({prefix})")
    print(f"{'='*60}")

    # Try catalog entries
    for entry in catalog:
        result = generate_rq3_instance(grid, state, entry, target_length=10)
        if result:
            full_trace, fatal_step, gap = result

            print(f"  Trace length: {len(full_trace)}")
            print(f"  Oracle fatal step: {fatal_step}")
            print(f"  Cause-symptom gap: {gap}")
            print(f"  Actions: {[a.name for a in full_trace]}")

            # Verify with find_fatal_step
            verified_fatal = find_fatal_step(grid, state, full_trace)
            print(f"  Verified fatal step: {verified_fatal}")
            assert verified_fatal == fatal_step, \
                f"Fatal step mismatch: {verified_fatal} vs {fatal_step}"

            # Render trace with fatal step highlighted
            fig = render_trace(
                grid, state, full_trace, cols=4,
                highlight_fatal=fatal_step,
                title=f"RQ3: Fatal step at index {fatal_step} (gap={gap})",
            )
            path = os.path.join(OUTPUT_DIR, f"{prefix}_trace.png")
            save_figure(fig, path)
            print(f"  Saved: {path}")
            return

    print("  WARNING: Could not generate RQ3 instance from catalog!")


def demo_prompts(
    grid: MosaicGrid,
    state: MosaicState,
    catalog,
    prefix: str = "prompt",
) -> None:
    """Generate sample prompts for each RQ and write them to text files."""
    print(f"\n{'='*60}")
    print(f"  Prompt Demo ({prefix})")
    print(f"{'='*60}")

    solution = oracle_solve(grid, state)
    if solution is None:
        print("  ERROR: Not solvable!")
        return

    # --- RQ1 Main ---
    n = min(5, len(solution))
    rq1_actions = solution[:n]
    prompt = build_rq1_prompt(grid, rq1_actions)
    path = os.path.join(OUTPUT_DIR, f"{prefix}_rq1_main.txt")
    with open(path, "w") as f:
        f.write(prompt)
    print(f"  RQ1 Main ({n} steps): {path}")

    # --- RQ1 Ablation (pick a direction that hits a wall) ---
    for d in Direction:
        from puzzles.common.types import add_pos
        from puzzles.mosaic_grid.engine import is_passable
        target = add_pos(state.agent_pos, d)
        if not is_passable(grid, state, target):
            prompt = build_rq1_ablation_prompt(grid, d)
            path = os.path.join(OUTPUT_DIR, f"{prefix}_rq1_ablation.txt")
            with open(path, "w") as f:
                f.write(prompt)
            print(f"  RQ1 Ablation ({d.name} into obstacle): {path}")
            break

    # --- RQ2 Main (deadlock) ---
    for entry in catalog:
        for k in [3, 2, 1]:
            result = generate_rq2_pair(grid, state, entry, k)
            if result:
                s_start, dead_actions, solv_actions = result

                prompt = build_rq2_main_prompt(grid, dead_actions)
                path = os.path.join(OUTPUT_DIR, f"{prefix}_rq2_main_dead.txt")
                with open(path, "w") as f:
                    f.write(prompt)
                print(f"  RQ2 Main (deadlock, k={k}): {path}")

                # Solvable control
                prompt = build_rq2_main_prompt(grid, solv_actions)
                path = os.path.join(OUTPUT_DIR, f"{prefix}_rq2_main_solv.txt")
                with open(path, "w") as f:
                    f.write(prompt)
                print(f"  RQ2 Main (solvable, k={k}): {path}")

                # Ablation (deadlock)
                prompt = build_rq2_ablation_prompt(grid, dead_actions)
                path = os.path.join(OUTPUT_DIR, f"{prefix}_rq2_ablation_dead.txt")
                with open(path, "w") as f:
                    f.write(prompt)
                print(f"  RQ2 Ablation (deadlock): {path}")
                break
        else:
            continue
        break

    # --- RQ3 ---
    for entry in catalog:
        result = generate_rq3_instance(grid, state, entry, target_length=10)
        if result:
            full_trace, fatal_step, gap = result
            prompt = build_rq3_prompt(grid, full_trace)
            path = os.path.join(OUTPUT_DIR, f"{prefix}_rq3.txt")
            with open(path, "w") as f:
                f.write(prompt)
            print(f"  RQ3 (fatal step={fatal_step + 1}, gap={gap}): {path}")
            break


def demo_generated(seed: int = 100) -> None:
    """Demo with a procedurally generated instance."""
    print(f"\n{'='*60}")
    print(f"  Generated Instance Demo (seed={seed})")
    print(f"{'='*60}")

    instance = generate_validated_instance(width=5, height=5, num_plates=2, seed=seed)
    if instance is None:
        print(f"  ERROR: Could not generate instance (seed={seed}). Skipping.")
        return
    else:
        grid = instance.grid
        state = instance.initial_state
        catalog = instance.deadlock_catalog
        print(f"  Generated (seed={instance.seed}, template={instance.template_name})")
        print(f"  Solution: {len(instance.solution)} steps")
        print(f"  Deadlocks found: {len(catalog)}")

    demo_normal_puzzle(grid, state, prefix="gen_normal")
    demo_rq1(grid, state, prefix="gen_rq1")
    demo_rq2(grid, state, catalog, prefix="gen_rq2")
    demo_rq3(grid, state, catalog, prefix="gen_rq3")


def demo_all_templates(
    base_seed: int = 500,
    max_retries: int = 500,
    grid_size: int = 5,
    num_plates: int | None = None,
    subdir: str = "templates",
    num_doors: int = 1,
) -> None:
    """Generate one instance per template with full demo outputs (initial, solution, RQ1-3).

    Args:
        grid_size: Width and height of the grid.
        num_plates: Number of plates (None = auto-scale by grid size).
        subdir: Subdirectory name under OUTPUT_DIR for these demos.
        num_doors: Number of doors (1 or 2).
    """
    label = f"{grid_size}x{grid_size}"
    if num_plates is not None:
        label += f", {num_plates}p"
    if num_doors >= 2:
        label += f", {num_doors}d"
    print(f"\n{'='*60}")
    print(f"  Multi-Template Demo ({len(TEMPLATE_NAMES)} templates, {label})")
    print(f"{'='*60}")

    templates_dir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(templates_dir, exist_ok=True)

    success = 0
    for tname in TEMPLATE_NAMES:
        # Try multiple seeds to find an instance where all 3 RQs succeed.
        # Most templates work on the first try; a few (island, lshape) need
        # a different seed for RQ3 due to short solutions at certain seeds.
        instance = None
        for seed_offset in range(10):
            candidate = generate_validated_instance(
                width=grid_size, height=grid_size, num_plates=num_plates,
                seed=base_seed + seed_offset * 1000, max_retries=max_retries,
                template=tname, num_doors=num_doors,
            )
            if candidate is None:
                continue
            # Check if RQ3 is possible with this instance
            rq3_possible = any(
                generate_rq3_instance(candidate.grid, candidate.initial_state, entry, target_length=10)
                for entry in candidate.deadlock_catalog
            )
            if rq3_possible:
                instance = candidate
                break
            # Keep the first valid instance as fallback even without RQ3
            if instance is None:
                instance = candidate

        if instance is None:
            print(f"  [{tname:12s}]  FAILED (no valid instance in {max_retries} attempts)")
            continue

        grid = instance.grid
        state = instance.initial_state
        catalog = instance.deadlock_catalog
        solution = instance.solution
        k_max = max((e.k_horizon for e in catalog), default=0)

        # Create per-template output directory
        tpl_dir = os.path.join(templates_dir, tname)
        os.makedirs(tpl_dir, exist_ok=True)

        # --- Initial state ---
        fig = render_state(grid, state, title=f"{tname.capitalize()} — Initial State")
        save_figure(fig, os.path.join(tpl_dir, "initial.png"))

        # --- Oracle solution trace ---
        fig = render_trace(
            grid, state, solution, cols=4,
            title=f"{tname.capitalize()} — Oracle Solution ({len(solution)} steps)",
        )
        save_figure(fig, os.path.join(tpl_dir, "solution_trace.png"))

        # --- RQ1: Simulation Fidelity ---
        n = min(5, len(solution))
        rq1_actions = solution[:n]
        gt_state = apply_actions(grid, state, rq1_actions)

        fig = render_side_by_side(
            grid, state, gt_state,
            label_left="Initial State",
            label_right=f"After {n} Actions (Ground Truth)",
            title=f"{tname.capitalize()} — RQ1: Predict state after {n} actions",
        )
        save_figure(fig, os.path.join(tpl_dir, "rq1_side_by_side.png"))

        fig = render_trace(
            grid, state, rq1_actions, cols=3,
            title=f"{tname.capitalize()} — RQ1: {n}-Step Trace",
        )
        save_figure(fig, os.path.join(tpl_dir, "rq1_trace.png"))

        # --- RQ2: Foresight (deadlock vs solvable pair) ---
        rq2_ok = False
        for entry in catalog:
            for k in [3, 2, 1, 5]:
                result = generate_rq2_pair(grid, state, entry, k)
                if result:
                    s_start, dead_actions, solv_actions = result
                    s_dead = apply_actions(grid, s_start, dead_actions)
                    s_solv = apply_actions(grid, s_start, solv_actions)

                    fig = render_state(
                        grid, s_start,
                        title=f"{tname.capitalize()} — RQ2 Start (before {k} actions)",
                    )
                    save_figure(fig, os.path.join(tpl_dir, "rq2_start.png"))

                    fig = render_side_by_side(
                        grid, s_dead, s_solv,
                        label_left=f"DEADLOCK (after {k} actions)",
                        label_right=f"SOLVABLE (after {k} actions)",
                        title=f"{tname.capitalize()} — RQ2: Matched Pair (k={k})",
                    )
                    save_figure(fig, os.path.join(tpl_dir, "rq2_pair.png"))
                    rq2_ok = True
                    break
            if rq2_ok:
                break

        # --- RQ3: Causal Attribution (fatal step trace) ---
        rq3_ok = False
        for entry in catalog:
            result = generate_rq3_instance(grid, state, entry, target_length=10)
            if result:
                full_trace, fatal_step, gap = result
                fig = render_trace(
                    grid, state, full_trace, cols=4,
                    highlight_fatal=fatal_step,
                    title=f"{tname.capitalize()} — RQ3: Fatal step at {fatal_step} (gap={gap})",
                )
                save_figure(fig, os.path.join(tpl_dir, "rq3_trace.png"))
                rq3_ok = True
                break

        rq_status = f"rq1=ok rq2={'ok' if rq2_ok else 'SKIP'} rq3={'ok' if rq3_ok else 'SKIP'}"
        print(
            f"  [{tname:12s}]  seed={instance.seed:4d}  "
            f"sol={len(solution):2d}  "
            f"deadlocks={len(catalog):2d}  "
            f"k_max={k_max:3d}  "
            f"{rq_status}"
        )
        success += 1

    print(f"\n  Results: {success}/{len(TEMPLATE_NAMES)} templates produced valid instances")
    print(f"  Output in: {templates_dir}")


def main() -> None:
    ensure_output_dir()

    print("Mosaic Grid Demo")
    print("=" * 60)

    # --- Procedurally generated instance ---
    demo_generated(seed=100)

    # --- Multi-template demos (1 door) ---
    # 5x5 with 2 plates (baseline)
    demo_all_templates(base_seed=500, grid_size=5, num_plates=2, subdir="templates")
    # 6x6 with 3 plates
    demo_all_templates(base_seed=500, grid_size=6, num_plates=3, subdir="templates_6x6_3p")
    # 8x8 with 6 plates
    demo_all_templates(base_seed=500, grid_size=8, num_plates=6, subdir="templates_8x8_6p")

    # --- Multi-template demos (2 doors) ---
    # 6x6 with 2 doors (all templates)
    demo_all_templates(base_seed=600, grid_size=6, num_doors=2, subdir="templates_6x6_2d")
    # 7x7 with 2 doors (all templates)
    demo_all_templates(base_seed=600, grid_size=7, num_doors=2, max_retries=800, subdir="templates_7x7_2d")
    # 8x8 with 2 doors (all templates — higher retries for large grid + multi-door)
    demo_all_templates(base_seed=600, grid_size=8, num_doors=2, max_retries=800, subdir="templates_8x8_2d")
    # 8x8 with 3 doors (all templates — higher retries)
    demo_all_templates(base_seed=700, grid_size=8, num_doors=3, max_retries=800, subdir="templates_8x8_3d")

    print(f"\n{'='*60}")
    print(f"  All demos complete! Output in: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
