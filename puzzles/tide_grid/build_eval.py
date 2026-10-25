"""Build evaluation dataset: raw JSONL -> per-puzzle dirs + eval_index.jsonl."""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import shutil
import time
from collections import defaultdict
from puzzles.common.types import Direction
from .dataset import dict_to_instance, _pos_to_list
from .engine import (
    TideGridGrid,
    TideGridState,
    TideGridAction,
    get_legal_actions,
)
from .generator import (
    GeneratedInstance,
    DeadlockEntry,
    generate_rq2_pair,
    generate_rq3_instance,
    discover_oracle_adjacent_deadlocks,
    generate_hard_rq2_pair,
    generate_hard_rq3_instance,
)
from .oracle import apply_actions, oracle_solve
from .prompts import (
    build_rq1_prompt,
    build_rq1_ablation_prompt,
    build_rq2_main_prompt,
    build_rq2_ablation_prompt,
    build_rq3_prompt,
    build_rq3_ablation_prompt,
)
from .renderer import render_state, render_trace, render_side_by_side

from puzzles.common.rendering import save_figure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _puzzle_dir_name(inst: GeneratedInstance) -> str:
    """Compute the per-puzzle directory name."""
    return (
        f"{inst.template_name}_{inst.grid_width}x{inst.grid_height}"
        f"_{inst.num_beacons}b_{inst.num_zones}z_seed{inst.seed}"
    )


def _ground_truth_state(grid: TideGridGrid, state: TideGridState) -> dict:
    """Serialize a state into the ground-truth format for eval."""
    zone_states = {}
    for zone_id, zone in sorted(grid.tidal_zones.items()):
        zone_states[zone_id] = "dry" if zone.is_dry_at(state.time) else "flooded"
    return {
        "agent_pos": _pos_to_list(state.agent_pos),
        "time": state.time,
        "oxygen_remaining": grid.initial_oxygen - state.time,
        "collected_beacons": sorted(state.collected_beacons),
        "zone_states": zone_states,
    }


# ---------------------------------------------------------------------------
# Post-build sampling
# ---------------------------------------------------------------------------

def _allocate_and_sample(
    buckets: dict[str, list],
    target: int,
    rng: random.Random,
) -> list:
    """Allocate quota across buckets and sample within each."""
    templates = sorted(buckets.keys())
    num_templates = len(templates)
    base_quota = target // num_templates
    remainder = target % num_templates

    quotas: dict[str, int] = {}
    surplus = 0
    for i, tpl in enumerate(templates):
        q = base_quota + (1 if i < remainder else 0)
        available = len(buckets[tpl])
        if available <= q:
            quotas[tpl] = available
            surplus += q - available
        else:
            quotas[tpl] = q

    if surplus > 0:
        for tpl in templates:
            available = len(buckets[tpl])
            room = available - quotas[tpl]
            if room > 0:
                give = min(room, surplus)
                quotas[tpl] += give
                surplus -= give
                if surplus <= 0:
                    break

    result = []
    for tpl in templates:
        pool = buckets[tpl]
        q = quotas[tpl]
        if len(pool) <= q:
            result.extend(pool)
        else:
            result.extend(rng.sample(pool, q))

    return result


# Movement actions for ablation (excludes WAIT)
_MOVEMENT_ACTIONS = [
    TideGridAction.UP,
    TideGridAction.DOWN,
    TideGridAction.LEFT,
    TideGridAction.RIGHT,
]


# ---------------------------------------------------------------------------
# Per-RQ builders
# ---------------------------------------------------------------------------

def _build_rq1_entries(
    grid: TideGridGrid,
    initial_state: TideGridState,
    solution: list[TideGridAction],
    puzzle_dir: str,
    rq1_lengths: list[int],
    base_meta: dict,
    rq2_sequences: list[dict] | None = None,
    rq3_sequences: list[dict] | None = None,
) -> list[dict]:
    """Build RQ1 eval entries: trajectory simulation + baselines + ablation."""
    entries = []
    solution_length = len(solution)

    # --- Trajectory items (oracle solution prefixes) ---
    lengths = sorted(set(
        min(n, solution_length) for n in rq1_lengths
    ) | {solution_length})

    for n in lengths:
        actions = solution[:n]
        final_state = apply_actions(grid, initial_state, actions)
        prompt = build_rq1_prompt(grid, actions)
        gt = _ground_truth_state(grid, final_state)

        entries.append({
            "eval_id": f"{puzzle_dir}__rq1_main__n{n}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq1",
            "variant": "main",
            "image_path": f"{puzzle_dir}/images/initial.png",
            "prompt": prompt,
            "ground_truth": gt,
            "metadata": {
                **base_meta,
                "sub_type": "trajectory",
                "trajectory_length": n,
            },
        })

    # --- Ablation: after N actions, is moving [DIRECTION] legal? ---
    # Tide grid is time-dependent: check if the action is in get_legal_actions
    for n in lengths:
        actions = solution[:n]
        final_state = apply_actions(grid, initial_state, actions)
        legal_actions = get_legal_actions(grid, final_state)
        for action in _MOVEMENT_ACTIONS:
            direction = action.direction
            legal = action in legal_actions
            prompt = build_rq1_ablation_prompt(grid, direction)
            gt = {
                "answer": "A" if legal else "B",
                "is_legal": legal,
            }
            entries.append({
                "eval_id": f"{puzzle_dir}__rq1_ablation__n{n}_{direction.name.lower()}",
                "puzzle_dir": puzzle_dir,
                "rq": "rq1",
                "variant": "ablation",
                "image_path": f"{puzzle_dir}/images/initial.png",
                "prompt": prompt,
                "ground_truth": gt,
                "metadata": {
                    **base_meta,
                    "trajectory_length": n,
                    "direction": direction.name,
                },
            })

    # --- RQ2 baseline items (paired with each RQ2 pair) ---
    if rq2_sequences:
        for seq in rq2_sequences:
            pidx = seq["pair_index"]
            src = seq.get("source", "standard")
            src_tag = "hard_" if src == "hard" else ""
            src_id = "h" if src == "hard" else ""

            dead_actions = seq["deadlock_actions"]
            dead_state = apply_actions(grid, initial_state, dead_actions)
            entries.append({
                "eval_id": f"{puzzle_dir}__rq1_main__{src_tag}rq2_baseline_deadlock__p{pidx}{src_id}",
                "puzzle_dir": puzzle_dir,
                "rq": "rq1",
                "variant": "main",
                "image_path": f"{puzzle_dir}/images/initial.png",
                "prompt": build_rq1_prompt(grid, dead_actions),
                "ground_truth": _ground_truth_state(grid, dead_state),
                "metadata": {
                    **base_meta,
                    "sub_type": f"{src_tag}rq2_baseline_deadlock",
                    "rq2_pair_index": pidx,
                    "trajectory_length": len(dead_actions),
                },
            })

            sol_actions = seq["solvable_actions"]
            sol_state = apply_actions(grid, initial_state, sol_actions)
            entries.append({
                "eval_id": f"{puzzle_dir}__rq1_main__{src_tag}rq2_baseline_solvable__p{pidx}{src_id}",
                "puzzle_dir": puzzle_dir,
                "rq": "rq1",
                "variant": "main",
                "image_path": f"{puzzle_dir}/images/initial.png",
                "prompt": build_rq1_prompt(grid, sol_actions),
                "ground_truth": _ground_truth_state(grid, sol_state),
                "metadata": {
                    **base_meta,
                    "sub_type": f"{src_tag}rq2_baseline_solvable",
                    "rq2_pair_index": pidx,
                    "trajectory_length": len(sol_actions),
                },
            })

    # --- RQ3 baseline items (paired with each RQ3 trace) ---
    if rq3_sequences:
        for seq in rq3_sequences:
            tidx = seq["trace_index"]
            src = seq.get("source", "standard")
            src_tag = "hard_" if src == "hard" else ""
            src_id = "h" if src == "hard" else ""
            trace = seq["full_trace"]
            final_state = apply_actions(grid, initial_state, trace)
            entries.append({
                "eval_id": f"{puzzle_dir}__rq1_main__{src_tag}rq3_baseline__t{tidx}{src_id}",
                "puzzle_dir": puzzle_dir,
                "rq": "rq1",
                "variant": "main",
                "image_path": f"{puzzle_dir}/images/initial.png",
                "prompt": build_rq1_prompt(grid, trace),
                "ground_truth": _ground_truth_state(grid, final_state),
                "metadata": {
                    **base_meta,
                    "sub_type": f"{src_tag}rq3_baseline",
                    "rq3_trace_index": tidx,
                    "trajectory_length": len(trace),
                },
            })

    return entries


def _build_rq2_entries(
    grid: TideGridGrid,
    initial_state: TideGridState,
    deadlock_catalog: list[DeadlockEntry],
    puzzle_dir: str,
    base_meta: dict,
) -> tuple[list[dict], list[dict]]:
    """Build RQ2 eval entries from deadlock catalog.

    Returns:
        (entries, rq2_sequences) where rq2_sequences contains the action
        sequences for each successful pair.
    """
    entries = []
    rq2_sequences: list[dict] = []
    pair_idx = 0

    eligible = [
        e for e in deadlock_catalog
        if e.k_horizon >= 3 and e.fatal_step >= 2
    ][:3]

    for entry in eligible:
        k = entry.fatal_step + 1
        result = generate_rq2_pair(grid, initial_state, entry, k)
        if result is None:
            continue

        start_state, deadlock_actions, solvable_actions = result

        rq2_sequences.append({
            "pair_index": pair_idx,
            "deadlock_actions": deadlock_actions,
            "solvable_actions": solvable_actions,
        })

        # 1. Main (deadlock actions) -> UNSOLVABLE
        prompt_dead = build_rq2_main_prompt(grid, deadlock_actions)
        entries.append({
            "eval_id": f"{puzzle_dir}__rq2_main_deadlock__e{pair_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq2",
            "variant": "main_deadlock",
            "image_path": f"{puzzle_dir}/images/initial.png",
            "prompt": prompt_dead,
            "ground_truth": {"answer": "UNSOLVABLE"},
            "metadata": {
                **base_meta,
                "rq2_pair_index": pair_idx,
                "prefix_length": k,
                "k_horizon": entry.k_horizon,
                "actions": [a.name for a in deadlock_actions],
            },
        })

        # 2. Main (solvable actions) -> best next move
        sol_state = apply_actions(grid, start_state, solvable_actions)
        next_sol = oracle_solve(grid, sol_state)
        best_next = next_sol[0].name if next_sol else "UNKNOWN"

        prompt_sol = build_rq2_main_prompt(grid, solvable_actions)
        entries.append({
            "eval_id": f"{puzzle_dir}__rq2_main_solvable__e{pair_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq2",
            "variant": "main_solvable",
            "image_path": f"{puzzle_dir}/images/initial.png",
            "prompt": prompt_sol,
            "ground_truth": {"answer": best_next},
            "metadata": {
                **base_meta,
                "rq2_pair_index": pair_idx,
                "prefix_length": k,
                "k_horizon": entry.k_horizon,
                "actions": [a.name for a in solvable_actions],
            },
        })

        # 3. Ablation (deadlock) -> NO
        prompt_abl_dead = build_rq2_ablation_prompt(grid, deadlock_actions)
        entries.append({
            "eval_id": f"{puzzle_dir}__rq2_ablation_deadlock__e{pair_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq2",
            "variant": "ablation_deadlock",
            "image_path": f"{puzzle_dir}/images/initial.png",
            "prompt": prompt_abl_dead,
            "ground_truth": {"answer": "NO"},
            "metadata": {
                **base_meta,
                "rq2_pair_index": pair_idx,
                "prefix_length": k,
                "k_horizon": entry.k_horizon,
                "actions": [a.name for a in deadlock_actions],
            },
        })

        # 4. Ablation (solvable) -> YES
        prompt_abl_sol = build_rq2_ablation_prompt(grid, solvable_actions)
        entries.append({
            "eval_id": f"{puzzle_dir}__rq2_ablation_solvable__e{pair_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq2",
            "variant": "ablation_solvable",
            "image_path": f"{puzzle_dir}/images/initial.png",
            "prompt": prompt_abl_sol,
            "ground_truth": {"answer": "YES"},
            "metadata": {
                **base_meta,
                "rq2_pair_index": pair_idx,
                "prefix_length": k,
                "k_horizon": entry.k_horizon,
                "actions": [a.name for a in solvable_actions],
            },
        })

        pair_idx += 1

    return entries, rq2_sequences


def _build_rq3_entries(
    grid: TideGridGrid,
    initial_state: TideGridState,
    deadlock_catalog: list[DeadlockEntry],
    puzzle_dir: str,
    base_meta: dict,
) -> tuple[list[dict], list[dict]]:
    """Build RQ3 eval entries from deadlock catalog.

    Returns:
        (entries, rq3_sequences) where rq3_sequences contains the full
        trace for each successful instance.
    """
    entries = []
    rq3_sequences: list[dict] = []
    trace_idx = 0

    eligible = [e for e in deadlock_catalog if e.k_horizon >= 5][:2]

    for entry in eligible:
        result = generate_rq3_instance(grid, initial_state, entry)
        if result is None:
            continue

        full_trace, fatal_step, cause_symptom_gap = result

        rq3_sequences.append({
            "trace_index": trace_idx,
            "full_trace": full_trace,
            "fatal_step": fatal_step,
        })

        prompt = build_rq3_prompt(grid, full_trace)
        gt = {
            "fatal_step": fatal_step,
            "fatal_step_1indexed": fatal_step + 1,
            "fatal_action": full_trace[fatal_step].name,
            "cause_symptom_gap": cause_symptom_gap,
            "trace_length": len(full_trace),
        }

        entries.append({
            "eval_id": f"{puzzle_dir}__rq3_main__e{trace_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq3",
            "variant": "main",
            "image_path": f"{puzzle_dir}/images/initial.png",
            "prompt": prompt,
            "ground_truth": gt,
            "metadata": {
                **base_meta,
                "rq3_trace_index": trace_idx,
                "cause_symptom_gap": cause_symptom_gap,
                "trace_length": len(full_trace),
            },
        })

        # Ablation: final state only
        ablation_final_prompt = build_rq3_ablation_prompt(
            grid, full_trace, show_initial=False,
        )
        entries.append({
            "eval_id": f"{puzzle_dir}__rq3_ablation_final__e{trace_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq3",
            "variant": "ablation_final",
            "image_path": f"{puzzle_dir}/images/rq3_final_e{trace_idx}.png",
            "prompt": ablation_final_prompt,
            "ground_truth": gt,
            "metadata": {
                **base_meta,
                "rq3_trace_index": trace_idx,
                "cause_symptom_gap": cause_symptom_gap,
                "trace_length": len(full_trace),
            },
        })

        # Ablation: side-by-side initial + final
        ablation_comparison_prompt = build_rq3_ablation_prompt(
            grid, full_trace, show_initial=True,
        )
        entries.append({
            "eval_id": f"{puzzle_dir}__rq3_ablation_comparison__e{trace_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq3",
            "variant": "ablation_comparison",
            "image_path": f"{puzzle_dir}/images/rq3_comparison_e{trace_idx}.png",
            "prompt": ablation_comparison_prompt,
            "ground_truth": gt,
            "metadata": {
                **base_meta,
                "rq3_trace_index": trace_idx,
                "cause_symptom_gap": cause_symptom_gap,
                "trace_length": len(full_trace),
            },
        })

        trace_idx += 1

    return entries, rq3_sequences


# ---------------------------------------------------------------------------
# Hard (oracle-adjacent) RQ2 / RQ3 builders
# ---------------------------------------------------------------------------

def _build_hard_rq2_entries(
    grid: TideGridGrid,
    initial_state: TideGridState,
    solution: list[TideGridAction],
    oa_catalog: list[DeadlockEntry],
    puzzle_dir: str,
    base_meta: dict,
) -> tuple[list[dict], list[dict]]:
    """Build hard RQ2 entries from oracle-adjacent deadlocks."""
    entries: list[dict] = []
    hard_rq2_sequences: list[dict] = []
    pair_idx = 0

    eligible = [
        e for e in oa_catalog
        if e.k_horizon >= 3 and e.fatal_step >= 2
    ][:3]

    for entry in eligible:
        result = generate_hard_rq2_pair(grid, initial_state, solution, entry)
        if result is None:
            continue

        start_state, deadlock_actions, solvable_actions = result

        hard_rq2_sequences.append({
            "pair_index": pair_idx,
            "deadlock_actions": deadlock_actions,
            "solvable_actions": solvable_actions,
        })

        # 1. Hard deadlock -> UNSOLVABLE
        prompt_dead = build_rq2_main_prompt(grid, deadlock_actions)
        entries.append({
            "eval_id": f"{puzzle_dir}__rq2_hard_deadlock__e{pair_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq2",
            "variant": "hard_deadlock",
            "image_path": f"{puzzle_dir}/images/initial.png",
            "prompt": prompt_dead,
            "ground_truth": {"answer": "UNSOLVABLE"},
            "metadata": {
                **base_meta,
                "rq2_pair_index": pair_idx,
                "k_horizon": entry.k_horizon,
                "fatal_step": entry.fatal_step,
                "oracle_prefix_length": entry.fatal_step,
                "actions": [a.name for a in deadlock_actions],
            },
        })

        # 2. Hard solvable -> best next move
        sol_state = apply_actions(grid, start_state, solvable_actions)
        next_sol = oracle_solve(grid, sol_state)
        best_next = next_sol[0].name if next_sol else "UNKNOWN"

        prompt_sol = build_rq2_main_prompt(grid, solvable_actions)
        entries.append({
            "eval_id": f"{puzzle_dir}__rq2_hard_solvable__e{pair_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq2",
            "variant": "hard_solvable",
            "image_path": f"{puzzle_dir}/images/initial.png",
            "prompt": prompt_sol,
            "ground_truth": {"answer": best_next},
            "metadata": {
                **base_meta,
                "rq2_pair_index": pair_idx,
                "k_horizon": entry.k_horizon,
                "fatal_step": entry.fatal_step,
                "oracle_prefix_length": entry.fatal_step,
                "actions": [a.name for a in solvable_actions],
            },
        })

        # 3. Hard ablation deadlock -> NO
        prompt_abl_dead = build_rq2_ablation_prompt(grid, deadlock_actions)
        entries.append({
            "eval_id": f"{puzzle_dir}__rq2_hard_ablation_deadlock__e{pair_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq2",
            "variant": "hard_ablation_deadlock",
            "image_path": f"{puzzle_dir}/images/initial.png",
            "prompt": prompt_abl_dead,
            "ground_truth": {"answer": "NO"},
            "metadata": {
                **base_meta,
                "rq2_pair_index": pair_idx,
                "k_horizon": entry.k_horizon,
                "fatal_step": entry.fatal_step,
                "oracle_prefix_length": entry.fatal_step,
                "actions": [a.name for a in deadlock_actions],
            },
        })

        # 4. Hard ablation solvable -> YES
        prompt_abl_sol = build_rq2_ablation_prompt(grid, solvable_actions)
        entries.append({
            "eval_id": f"{puzzle_dir}__rq2_hard_ablation_solvable__e{pair_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq2",
            "variant": "hard_ablation_solvable",
            "image_path": f"{puzzle_dir}/images/initial.png",
            "prompt": prompt_abl_sol,
            "ground_truth": {"answer": "YES"},
            "metadata": {
                **base_meta,
                "rq2_pair_index": pair_idx,
                "k_horizon": entry.k_horizon,
                "fatal_step": entry.fatal_step,
                "oracle_prefix_length": entry.fatal_step,
                "actions": [a.name for a in solvable_actions],
            },
        })

        pair_idx += 1

    return entries, hard_rq2_sequences


def _build_hard_rq3_entries(
    grid: TideGridGrid,
    initial_state: TideGridState,
    solution: list[TideGridAction],
    oa_catalog: list[DeadlockEntry],
    puzzle_dir: str,
    base_meta: dict,
) -> tuple[list[dict], list[dict]]:
    """Build hard RQ3 entries from oracle-adjacent deadlocks."""
    entries: list[dict] = []
    hard_rq3_sequences: list[dict] = []
    trace_idx = 0

    eligible = [e for e in oa_catalog if e.k_horizon >= 5][:5]

    for entry in eligible:
        result = generate_hard_rq3_instance(
            grid, initial_state, solution, entry,
        )
        if result is None:
            continue

        full_trace, fatal_step, cause_symptom_gap = result

        hard_rq3_sequences.append({
            "trace_index": trace_idx,
            "full_trace": full_trace,
            "fatal_step": fatal_step,
        })

        prompt = build_rq3_prompt(grid, full_trace)
        gt = {
            "fatal_step": fatal_step,
            "fatal_step_1indexed": fatal_step + 1,
            "fatal_action": full_trace[fatal_step].name,
            "cause_symptom_gap": cause_symptom_gap,
            "trace_length": len(full_trace),
        }

        entries.append({
            "eval_id": f"{puzzle_dir}__rq3_hard__e{trace_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq3",
            "variant": "hard",
            "image_path": f"{puzzle_dir}/images/initial.png",
            "prompt": prompt,
            "ground_truth": gt,
            "metadata": {
                **base_meta,
                "rq3_trace_index": trace_idx,
                "cause_symptom_gap": cause_symptom_gap,
                "trace_length": len(full_trace),
                "oracle_prefix_length": fatal_step,
            },
        })

        # Ablation: final state only
        ablation_final_prompt = build_rq3_ablation_prompt(
            grid, full_trace, show_initial=False,
        )
        entries.append({
            "eval_id": f"{puzzle_dir}__rq3_hard_ablation_final__e{trace_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq3",
            "variant": "hard_ablation_final",
            "image_path": f"{puzzle_dir}/images/rq3_hard_final_e{trace_idx}.png",
            "prompt": ablation_final_prompt,
            "ground_truth": gt,
            "metadata": {
                **base_meta,
                "rq3_trace_index": trace_idx,
                "cause_symptom_gap": cause_symptom_gap,
                "trace_length": len(full_trace),
                "oracle_prefix_length": fatal_step,
            },
        })

        # Ablation: side-by-side initial + final
        ablation_comparison_prompt = build_rq3_ablation_prompt(
            grid, full_trace, show_initial=True,
        )
        entries.append({
            "eval_id": f"{puzzle_dir}__rq3_hard_ablation_comparison__e{trace_idx}",
            "puzzle_dir": puzzle_dir,
            "rq": "rq3",
            "variant": "hard_ablation_comparison",
            "image_path": f"{puzzle_dir}/images/rq3_hard_comparison_e{trace_idx}.png",
            "prompt": ablation_comparison_prompt,
            "ground_truth": gt,
            "metadata": {
                **base_meta,
                "rq3_trace_index": trace_idx,
                "cause_symptom_gap": cause_symptom_gap,
                "trace_length": len(full_trace),
                "oracle_prefix_length": fatal_step,
            },
        })

        trace_idx += 1

    return entries, hard_rq3_sequences


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _build_worker_init() -> None:
    """Per-process initialization: set matplotlib backend."""
    import matplotlib
    matplotlib.use("Agg")


def _build_worker(task: dict) -> list[dict]:
    """Per-instance worker. Returns list of eval index entries."""
    record = task["record"]
    output_dir = task["output_dir"]
    rq1_lengths = task["rq1_lengths"]
    overwrite = task["overwrite"]
    inst = dict_to_instance(record["instance"])
    puzzle_dir = _puzzle_dir_name(inst)
    puzzle_path = os.path.join(output_dir, puzzle_dir)

    # Skip if exists and not overwriting
    if os.path.exists(puzzle_path) and not overwrite:
        return [{"_status": "skipped", "puzzle_dir": puzzle_dir}]

    grid = inst.grid
    initial_state = inst.initial_state
    solution = inst.solution

    # Structural signature: zone timing params for within-family dedup
    zone_sig = tuple(
        (zid, z.dry_duration, z.flood_duration, z.phase_offset)
        for zid, z in sorted(grid.tidal_zones.items())
    )

    base_meta = {
        "template_name": inst.template_name,
        "grid_size": f"{inst.grid_width}x{inst.grid_height}",
        "num_beacons": inst.num_beacons,
        "num_zones": inst.num_zones,
        "solution_length": inst.solution_length,
        "seed": inst.seed,
        "fingerprint": inst.fingerprint,
        "zone_signature": str(zone_sig),
    }

    # --- Check RQ2/RQ3 viability BEFORE any I/O ---
    rq2_entries, rq2_sequences = _build_rq2_entries(
        grid, initial_state, inst.deadlock_catalog, puzzle_dir, base_meta,
    )
    rq3_entries, rq3_sequences = _build_rq3_entries(
        grid, initial_state, inst.deadlock_catalog, puzzle_dir, base_meta,
    )

    if not rq2_entries or not rq3_entries:
        return [{"_status": "dropped_missing_rq"}]

    # --- Hard (oracle-adjacent) variants ---
    oa_catalog = discover_oracle_adjacent_deadlocks(grid, initial_state, solution)

    hard_rq2_entries, hard_rq2_sequences = _build_hard_rq2_entries(
        grid, initial_state, solution, oa_catalog, puzzle_dir, base_meta,
    )
    hard_rq3_entries, hard_rq3_sequences = _build_hard_rq3_entries(
        grid, initial_state, solution, oa_catalog, puzzle_dir, base_meta,
    )

    has_hard_rq2 = bool(hard_rq2_entries)
    has_hard_rq3 = bool(hard_rq3_entries)
    base_meta["has_hard_rq2"] = has_hard_rq2
    base_meta["has_hard_rq3"] = has_hard_rq3
    base_meta["has_hard"] = has_hard_rq2 and has_hard_rq3

    # --- Puzzle passed — create directory and write all artifacts ---
    os.makedirs(puzzle_path, exist_ok=True)
    img_path = os.path.join(puzzle_path, "images")
    os.makedirs(img_path, exist_ok=True)

    # Write metadata.json
    tidal_zone_details = {}
    for zid, zone in sorted(grid.tidal_zones.items()):
        tidal_zone_details[zid] = {
            "dry_duration": zone.dry_duration,
            "flood_duration": zone.flood_duration,
            "cycle_length": zone.cycle_length,
            "phase_offset": zone.phase_offset,
            "num_tiles": len(zone.tiles),
            "initial_state": "dry" if zone.is_dry_at(0) else "flooded",
        }
    metadata = {
        **base_meta,
        "num_beacons": inst.num_beacons,
        "num_zones": inst.num_zones,
        "initial_oxygen": grid.initial_oxygen,
        "tidal_zones": tidal_zone_details,
        "tier1_deadlocks": inst.tier1_deadlocks,
        "tier2_deadlocks": inst.tier2_deadlocks,
        "max_k_horizon": inst.max_k_horizon,
        "generation_attempts": inst.generation_attempts,
    }
    with open(os.path.join(puzzle_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Render initial.png + solution_trace.png
    fig = render_state(grid, initial_state, title="Initial State")
    save_figure(fig, os.path.join(img_path, "initial.png"))

    fig = render_trace(grid, initial_state, solution, title="Oracle Solution")
    save_figure(fig, os.path.join(img_path, "solution_trace.png"))

    # Render RQ2 reference images
    for seq in rq2_sequences:
        pidx = seq["pair_index"]
        dead_state = apply_actions(grid, initial_state, seq["deadlock_actions"])
        fig = render_state(grid, dead_state, title=f"RQ2 Deadlock State (e{pidx})")
        save_figure(fig, os.path.join(img_path, f"rq2_state_e{pidx}.png"))

    # Render RQ3 reference images
    for seq in rq3_sequences:
        tidx = seq["trace_index"]
        fig = render_trace(
            grid, initial_state, seq["full_trace"],
            highlight_fatal=seq["fatal_step"],
            title=f"RQ3 Trace (e{tidx}) — Fatal step {seq['fatal_step'] + 1}",
        )
        save_figure(fig, os.path.join(img_path, f"rq3_trace_e{tidx}.png"))

        final_state = apply_actions(grid, initial_state, seq["full_trace"])
        fig = render_state(
            grid, final_state,
            title=f"RQ3 Final State (e{tidx})",
        )
        save_figure(fig, os.path.join(img_path, f"rq3_final_e{tidx}.png"))

        fig = render_side_by_side(
            grid, initial_state, final_state,
            label_left="Initial State",
            label_right="Final State",
            title=f"RQ3 Comparison (e{tidx})",
        )
        save_figure(fig, os.path.join(img_path, f"rq3_comparison_e{tidx}.png"))

    # Render hard RQ2 reference images
    for seq in hard_rq2_sequences:
        pidx = seq["pair_index"]
        dead_state = apply_actions(grid, initial_state, seq["deadlock_actions"])
        fig = render_state(grid, dead_state, title=f"RQ2 Hard Deadlock State (e{pidx})")
        save_figure(fig, os.path.join(img_path, f"rq2_hard_state_e{pidx}.png"))

    # Render hard RQ3 reference images
    for seq in hard_rq3_sequences:
        tidx = seq["trace_index"]
        fig = render_trace(
            grid, initial_state, seq["full_trace"],
            highlight_fatal=seq["fatal_step"],
            title=f"RQ3 Hard Trace (e{tidx}) — Fatal step {seq['fatal_step'] + 1}",
        )
        save_figure(fig, os.path.join(img_path, f"rq3_hard_trace_e{tidx}.png"))

        final_state = apply_actions(grid, initial_state, seq["full_trace"])
        fig = render_state(
            grid, final_state,
            title=f"RQ3 Hard Final State (e{tidx})",
        )
        save_figure(fig, os.path.join(img_path, f"rq3_hard_final_e{tidx}.png"))

        fig = render_side_by_side(
            grid, initial_state, final_state,
            label_left="Initial State",
            label_right="Final State",
            title=f"RQ3 Hard Comparison (e{tidx})",
        )
        save_figure(fig, os.path.join(img_path, f"rq3_hard_comparison_e{tidx}.png"))

    # Combine all sequences for RQ1 baselines
    for seq in rq2_sequences:
        seq.setdefault("source", "standard")
    for seq in hard_rq2_sequences:
        seq.setdefault("source", "hard")
    for seq in rq3_sequences:
        seq.setdefault("source", "standard")
    for seq in hard_rq3_sequences:
        seq.setdefault("source", "hard")
    all_rq2_sequences = rq2_sequences + hard_rq2_sequences
    all_rq3_sequences = rq3_sequences + hard_rq3_sequences

    # RQ1 last — depends on RQ2/RQ3 sequences for baseline items
    entries: list[dict] = []
    entries.extend(_build_rq1_entries(
        grid, initial_state, solution, puzzle_dir, rq1_lengths, base_meta,
        rq2_sequences=all_rq2_sequences,
        rq3_sequences=all_rq3_sequences,
    ))
    entries.extend(rq2_entries)
    entries.extend(rq3_entries)
    entries.extend(hard_rq2_entries)
    entries.extend(hard_rq3_entries)

    # Write per-puzzle eval items
    with open(os.path.join(puzzle_path, "eval_items.json"), "w") as f:
        json.dump(entries, f, indent=2)

    return entries


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_eval_dataset(
    input_path: str,
    output_dir: str,
    workers: int | None = None,
    rq1_trajectory_lengths: list[int] | None = None,
    overwrite: bool = False,
    puzzle_target: int = 0,
    sample_seed: int = 42,
    require_hard: bool = False,
) -> dict:
    """Build evaluation dataset from raw JSONL.

    Args:
        puzzle_target: Max puzzles in final eval index (0 = keep all).
        sample_seed: Random seed for reproducible sampling.
        require_hard: If True, filter out puzzles that lack hard
            (oracle-adjacent) RQ2 and RQ3 variants after building.
            All puzzles are still built and tagged; this flag only
            affects which puzzles appear in the final eval index.

    Returns aggregate stats dict.
    """
    from tqdm import tqdm

    if workers is None:
        workers = os.cpu_count() or 1
    if rq1_trajectory_lengths is None:
        rq1_trajectory_lengths = [5, 10, 15]

    os.makedirs(output_dir, exist_ok=True)

    # Read all records from input JSONL
    records: list[dict] = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("instance") is not None:
                records.append(record)

    print(f"Loaded {len(records)} puzzle instances from {input_path}")

    if not records:
        print("No instances to build.")
        return _empty_build_stats()

    # Load previous eval_index entries for carry-forward on skip
    prev_entries_by_puzzle: dict[str, list[dict]] = defaultdict(list)
    eval_path = os.path.join(output_dir, "eval_index.jsonl")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                prev_entries_by_puzzle[entry["puzzle_dir"]].append(entry)

    # Build task list (require_hard is handled post-build, not per-worker)
    tasks = [
        {
            "record": record,
            "output_dir": output_dir,
            "rq1_lengths": rq1_trajectory_lengths,
            "overwrite": overwrite,
        }
        for record in records
    ]

    # Process in parallel
    all_entries: list[dict] = []
    stats: dict[str, int] = {
        "total_puzzles": len(records),
        "puzzles_built": 0,
        "puzzles_reused": 0,
        "puzzles_skipped": 0,
        "puzzles_dropped_missing_rq": 0,
        "puzzles_without_hard": 0,
        "rq1_main": 0,
        "rq1_ablation": 0,
        "rq2_main_deadlock": 0,
        "rq2_main_solvable": 0,
        "rq2_ablation_deadlock": 0,
        "rq2_ablation_solvable": 0,
        "rq2_hard_deadlock": 0,
        "rq2_hard_solvable": 0,
        "rq2_hard_ablation_deadlock": 0,
        "rq2_hard_ablation_solvable": 0,
        "rq3_main": 0,
        "rq3_ablation_final": 0,
        "rq3_ablation_comparison": 0,
        "rq3_hard": 0,
        "rq3_hard_ablation_final": 0,
        "rq3_hard_ablation_comparison": 0,
        "total_eval_items": 0,
    }

    t0 = time.monotonic()

    with multiprocessing.Pool(workers, initializer=_build_worker_init) as pool:
        for entries in tqdm(
            pool.imap_unordered(_build_worker, tasks),
            total=len(tasks),
            desc="Building eval",
        ):
            if not entries:
                stats["puzzles_skipped"] += 1
                continue

            # Check for skipped-puzzle marker
            if len(entries) == 1 and entries[0].get("_status") == "skipped":
                pdir = entries[0]["puzzle_dir"]
                prev = prev_entries_by_puzzle.get(pdir, [])
                if prev:
                    stats["puzzles_reused"] += 1
                    for entry in prev:
                        variant = entry["variant"]
                        key = f"{entry['rq']}_{variant}"
                        if key in stats:
                            stats[key] += 1
                    all_entries.extend(prev)
                    puzzle_path = os.path.join(output_dir, pdir)
                    items_path = os.path.join(puzzle_path, "eval_items.json")
                    if not os.path.exists(items_path):
                        with open(items_path, "w") as f:
                            json.dump(prev, f, indent=2)
                else:
                    stats["puzzles_skipped"] += 1
                continue

            if len(entries) == 1 and isinstance(entries[0], dict) and entries[0].get("_status") == "dropped_missing_rq":
                stats["puzzles_dropped_missing_rq"] += 1
                continue

            stats["puzzles_built"] += 1
            # Track hard variant availability
            if entries and not entries[0].get("metadata", {}).get("has_hard", False):
                stats["puzzles_without_hard"] += 1
            for entry in entries:
                variant = entry["variant"]
                key = f"{entry['rq']}_{variant}"
                if key in stats:
                    stats[key] += 1
            all_entries.extend(entries)

    stats["puzzles_in_eval"] = stats["puzzles_built"] + stats["puzzles_reused"]
    stats["total_eval_items"] = len(all_entries)
    stats["elapsed_seconds"] = round(time.monotonic() - t0, 2)

    # --- Post-build require_hard filter ---
    if require_hard:
        before = len(set(e["puzzle_dir"] for e in all_entries))
        all_entries = [
            e for e in all_entries
            if e.get("metadata", {}).get("has_hard", False)
        ]
        after = len(set(e["puzzle_dir"] for e in all_entries))
        stats["puzzles_filtered_hard"] = before - after
        if before > after:
            print(f"require_hard filter: {before} -> {after} puzzles "
                  f"({before - after} dropped)")

    # --- Within-family structural dedup ---
    # For each (template, size) group, keep at most one puzzle per
    # zone_signature to prevent near-clone instances in the dataset.
    puzzle_entries_pre: dict[str, list[dict]] = defaultdict(list)
    for entry in all_entries:
        puzzle_entries_pre[entry["puzzle_dir"]].append(entry)

    family_seen_sigs: dict[str, set[str]] = defaultdict(set)
    dedup_dropped: set[str] = set()
    for pdir, items in puzzle_entries_pre.items():
        meta = items[0]["metadata"]
        family_key = f"{meta['template_name']}__{meta['grid_size']}"
        sig = meta.get("zone_signature", "")
        if sig and sig in family_seen_sigs[family_key]:
            dedup_dropped.add(pdir)
        else:
            family_seen_sigs[family_key].add(sig)

    if dedup_dropped:
        all_entries = [e for e in all_entries if e["puzzle_dir"] not in dedup_dropped]
        stats["puzzles_structural_dedup"] = len(dedup_dropped)
        print(f"Structural dedup: removed {len(dedup_dropped)} near-clone puzzles")

    # --- Post-build puzzle-level sampling ---
    if puzzle_target > 0:
        puzzle_entries: dict[str, list[dict]] = defaultdict(list)
        for entry in all_entries:
            puzzle_entries[entry["puzzle_dir"]].append(entry)

        if len(puzzle_entries) > puzzle_target:
            strat_buckets: dict[str, list[str]] = defaultdict(list)
            for pdir, items in puzzle_entries.items():
                meta = items[0]["metadata"]
                key = f"{meta['template_name']}__{meta['grid_size']}__{meta['num_beacons']}b_{meta['num_zones']}z"
                strat_buckets[key].append(pdir)

            rng = random.Random(sample_seed)
            selected = _allocate_and_sample(strat_buckets, puzzle_target, rng)

            kept = set(selected)
            all_entries = [e for e in all_entries if e["puzzle_dir"] in kept]

            stats["puzzles_sampled"] = f"{len(puzzle_entries)} -> {len(kept)}"
            print(f"Puzzle sampling: {len(puzzle_entries)} -> {len(kept)} puzzles "
                  f"(target {puzzle_target})")

        stats["total_eval_items"] = len(all_entries)

    # Write eval_index.jsonl
    eval_path = os.path.join(output_dir, "eval_index.jsonl")
    with open(eval_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    # Write build_stats.json
    stats_path = os.path.join(output_dir, "build_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # --- Cleanup: remove puzzle dirs not in the final eval index ---
    kept_dirs = set(entry["puzzle_dir"] for entry in all_entries)
    removed = 0
    for name in os.listdir(output_dir):
        dirpath = os.path.join(output_dir, name)
        if not os.path.isdir(dirpath):
            continue
        if not os.path.exists(os.path.join(dirpath, "metadata.json")):
            continue
        if name not in kept_dirs:
            shutil.rmtree(dirpath)
            removed += 1

    if removed:
        stats["puzzles_cleaned_up"] = removed

    reused = stats["puzzles_reused"]
    reused_msg = f", {reused} reused" if reused else ""
    without_hard = stats["puzzles_without_hard"]
    hard_msg = f", {without_hard} without hard variants" if without_hard else ""
    filtered_hard = stats.get("puzzles_filtered_hard", 0)
    filt_msg = f", {filtered_hard} filtered by require_hard" if filtered_hard else ""
    print(f"Done. {stats['puzzles_built']} puzzles built{reused_msg}, "
          f"{stats['puzzles_skipped']} skipped, "
          f"{stats['puzzles_dropped_missing_rq']} dropped (missing RQ2/RQ3)"
          f"{hard_msg}{filt_msg}, "
          f"{stats['total_eval_items']} eval items.")
    if removed:
        print(f"Cleaned up {removed} puzzle directories not in final eval index.")
    print(f"Eval index: {eval_path}")
    print(f"Build stats: {stats_path}")

    return stats


def _empty_build_stats() -> dict:
    return {
        "total_puzzles": 0,
        "puzzles_built": 0,
        "puzzles_reused": 0,
        "puzzles_in_eval": 0,
        "puzzles_skipped": 0,
        "puzzles_dropped_missing_rq": 0,
        "puzzles_without_hard": 0,
        "total_eval_items": 0,
        "elapsed_seconds": 0.0,
    }
