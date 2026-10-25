"""Dataset generation: serialization, fingerprinting, parallel orchestrator."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from typing import Any, Optional

from puzzles.common.types import Pos
from .engine import (
    ChainVaultGrid,
    ChainVaultState,
    ChainVaultAction,
    KeyInfo,
    DoorInfo,
)
from .generator import (
    GeneratedInstance,
    DeadlockEntry,
    generate_validated_instance_with_stats,
    is_eval_ready,
    TEMPLATE_NAMES,
    TEMPLATE_TIERS,
    TIER_GRID_SIZES,
)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _pos_to_list(p: Pos) -> list[int]:
    return [p.row, p.col]


def _list_to_pos(lst: list[int]) -> Pos:
    return Pos(lst[0], lst[1])


def _posset_to_sorted(ps: frozenset[Pos]) -> list[list[int]]:
    return sorted([_pos_to_list(p) for p in ps])


def _sorted_to_posset(lst: list[list[int]]) -> frozenset[Pos]:
    return frozenset(_list_to_pos(x) for x in lst)


def _key_to_dict(k: KeyInfo) -> dict:
    return {
        "id": k.id,
        "pos": _pos_to_list(k.pos),
        "color": k.color,
    }


def _dict_to_key(d: dict) -> KeyInfo:
    return KeyInfo(
        id=d["id"],
        pos=_list_to_pos(d["pos"]),
        color=d["color"],
    )


def _door_to_dict(d: DoorInfo) -> dict:
    return {
        "id": d.id,
        "pos": _pos_to_list(d.pos),
        "required_keys": list(d.required_keys),
    }


def _dict_to_door(d: dict) -> DoorInfo:
    return DoorInfo(
        id=d["id"],
        pos=_list_to_pos(d["pos"]),
        required_keys=tuple(d["required_keys"]),
    )


def _grid_to_dict(g: ChainVaultGrid) -> dict:
    return {
        "width": g.width,
        "height": g.height,
        "walls": _posset_to_sorted(g.walls),
        "keys": {k: _key_to_dict(v) for k, v in g.keys.items()},
        "doors": {k: _door_to_dict(v) for k, v in g.doors.items()},
        "start": _pos_to_list(g.start),
        "exit_pos": _pos_to_list(g.exit_pos),
    }


def _dict_to_grid(d: dict) -> ChainVaultGrid:
    return ChainVaultGrid(
        width=d["width"],
        height=d["height"],
        walls=_sorted_to_posset(d["walls"]),
        keys={k: _dict_to_key(v) for k, v in d["keys"].items()},
        doors={k: _dict_to_door(v) for k, v in d["doors"].items()},
        start=_list_to_pos(d["start"]),
        exit_pos=_list_to_pos(d["exit_pos"]),
    )


def _state_to_dict(s: ChainVaultState) -> dict:
    return {
        "agent_pos": _pos_to_list(s.agent_pos),
        "inventory": [[color, count] for color, count in s.inventory],
        "opened_doors": sorted(s.opened_doors),
        "remaining_keys": sorted(s.remaining_keys),
    }


def _dict_to_state(d: dict) -> ChainVaultState:
    return ChainVaultState(
        agent_pos=_list_to_pos(d["agent_pos"]),
        inventory=tuple(tuple(pair) for pair in d["inventory"]),
        opened_doors=frozenset(d["opened_doors"]),
        remaining_keys=frozenset(d["remaining_keys"]),
    )


_ACTION_NAME_ALIASES = {
    "MOVE_UP": "UP",
    "MOVE_DOWN": "DOWN",
    "MOVE_LEFT": "LEFT",
    "MOVE_RIGHT": "RIGHT",
    "USE_KEY_UP": "UP",
    "USE_KEY_DOWN": "DOWN",
    "USE_KEY_LEFT": "LEFT",
    "USE_KEY_RIGHT": "RIGHT",
}


def _action_from_name(name: str) -> ChainVaultAction:
    return ChainVaultAction[_ACTION_NAME_ALIASES.get(name, name)]


def _deadlock_to_dict(e: DeadlockEntry) -> dict:
    return {
        "path": [a.name for a in e.path],
        "fatal_step": e.fatal_step,
        "deadlock_state": _state_to_dict(e.deadlock_state),
        "pre_deadlock_state": _state_to_dict(e.pre_deadlock_state),
        "k_horizon": e.k_horizon,
    }


def _dict_to_deadlock(d: dict) -> DeadlockEntry:
    return DeadlockEntry(
        path=[_action_from_name(name) for name in d["path"]],
        fatal_step=d["fatal_step"],
        deadlock_state=_dict_to_state(d["deadlock_state"]),
        pre_deadlock_state=_dict_to_state(d["pre_deadlock_state"]),
        k_horizon=d["k_horizon"],
    )


def instance_to_dict(inst: GeneratedInstance) -> dict:
    """Serialize a GeneratedInstance to a JSON-compatible dict."""
    return {
        "grid": _grid_to_dict(inst.grid),
        "initial_state": _state_to_dict(inst.initial_state),
        "solution": [a.name for a in inst.solution],
        "deadlock_catalog": [_deadlock_to_dict(e) for e in inst.deadlock_catalog],
        "seed": inst.seed,
        "template_name": inst.template_name,
        "grid_width": inst.grid_width,
        "grid_height": inst.grid_height,
        "num_keys": inst.num_keys,
        "num_doors": inst.num_doors,
        "solution_length": inst.solution_length,
        "tier1_deadlocks": inst.tier1_deadlocks,
        "tier2_deadlocks": inst.tier2_deadlocks,
        "max_k_horizon": inst.max_k_horizon,
        "generation_attempts": inst.generation_attempts,
        "fingerprint": inst.fingerprint,
    }


def dict_to_instance(d: dict) -> GeneratedInstance:
    """Deserialize a dict back to a GeneratedInstance."""
    grid = _dict_to_grid(d["grid"])
    return GeneratedInstance(
        grid=grid,
        initial_state=_dict_to_state(d["initial_state"]),
        solution=[_action_from_name(name) for name in d["solution"]],
        deadlock_catalog=[_dict_to_deadlock(e) for e in d["deadlock_catalog"]],
        seed=d["seed"],
        template_name=d["template_name"],
        grid_width=d.get("grid_width", 0),
        grid_height=d.get("grid_height", 0),
        num_keys=d.get("num_keys", 0),
        num_doors=d.get("num_doors", 0),
        solution_length=d.get("solution_length", 0),
        tier1_deadlocks=d.get("tier1_deadlocks", 0),
        tier2_deadlocks=d.get("tier2_deadlocks", 0),
        max_k_horizon=d.get("max_k_horizon", 0),
        generation_attempts=d.get("generation_attempts", 0),
        fingerprint=d.get("fingerprint", ""),
    )


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def compute_fingerprint(inst: GeneratedInstance) -> str:
    """SHA-256 fingerprint (16 hex chars) of canonical grid layout."""
    g = inst.grid
    canonical = {
        "w": g.width,
        "h": g.height,
        "walls": sorted([p.row * g.width + p.col for p in g.walls]),
        "start": _pos_to_list(g.start),
        "exit": _pos_to_list(g.exit_pos),
        "keys": sorted([
            [_pos_to_list(k.pos), k.color] for k in g.keys.values()
        ]),
        "doors": sorted([
            [_pos_to_list(d.pos), list(d.required_keys)]
            for d in g.doors.values()
        ]),
    }
    blob = json.dumps(canonical, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _worker(task: dict) -> dict:
    """Worker function for multiprocessing pool.

    Takes a task dict, runs generation, returns a JSON-serializable result dict.
    If the task includes require_hard, also runs the eval-readiness check inline
    so only fully qualified instances reach the main process.
    """
    result = generate_validated_instance_with_stats(
        width=task["width"],
        height=task["height"],
        seed=task["seed"],
        max_retries=task["max_retries"],
        template=task["template"],
    )

    out: dict[str, Any] = {
        "task": task,
        "stats": result.stats,
        "total_attempts": result.total_attempts,
        "elapsed_seconds": result.elapsed_seconds,
        "success": result.instance is not None,
        "eval_ready": False,
    }

    if result.instance is not None:
        fp = compute_fingerprint(result.instance)
        result.instance.fingerprint = fp
        require_hard = task.get("require_hard", False)
        out["eval_ready"] = is_eval_ready(result.instance, require_hard)
        out["instance"] = instance_to_dict(result.instance)
        out["fingerprint"] = fp
    else:
        out["instance"] = None
        out["fingerprint"] = None

    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_dataset(
    templates: list[str],
    grid_sizes: list[int],
    seeds_per_config: int = 100,
    max_retries: int = 400,
    workers: int | None = None,
    output: str = "chain_vault_dataset.jsonl",
    resume: bool = False,
    seed_start: int = 0,
) -> dict:
    """Generate a dataset of puzzle instances in parallel.

    Returns aggregate stats dict.
    """
    from tqdm import tqdm

    if workers is None:
        workers = os.cpu_count() or 1

    # Build task list: (template, size) configs filtered by tier
    tasks: list[dict] = []
    for template in templates:
        tier = TEMPLATE_TIERS.get(template, "medium")
        allowed_sizes = TIER_GRID_SIZES.get(tier, [(5, 5), (6, 6)])
        for size in grid_sizes:
            if (size, size) not in allowed_sizes:
                continue
            for i in range(seeds_per_config):
                seed = seed_start + i
                tasks.append({
                    "template": template,
                    "width": size,
                    "height": size,
                    "seed": seed,
                    "max_retries": max_retries,
                })

    # Resume: load existing JSONL, skip done seeds, rebuild fingerprint set
    seen_fingerprints: set[str] = set()
    done_keys: set[str] = set()
    existing_count = 0

    if resume and os.path.exists(output):
        with open(output, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_info = record.get("task", {})
                key = _task_key(task_info)
                done_keys.add(key)
                fp = record.get("fingerprint")
                if fp:
                    seen_fingerprints.add(fp)
                existing_count += 1

    # Filter out already-done tasks
    if done_keys:
        tasks = [t for t in tasks if _task_key(t) not in done_keys]

    total_tasks = len(tasks) + existing_count
    print(f"Total configs: {total_tasks} | Already done: {existing_count} | Remaining: {len(tasks)}")

    if not tasks:
        print("Nothing to generate.")
        return _empty_agg_stats()

    # Failure output
    failure_path = output.rsplit(".", 1)[0] + "_failures.jsonl"

    # Aggregate stats
    agg: dict[str, Any] = {
        "total_tasks": total_tasks,
        "resumed": existing_count,
        "generated": 0,
        "failed": 0,
        "duplicates": 0,
        "rejection_totals": {},
        "elapsed_seconds": 0.0,
    }

    t0 = time.monotonic()

    with open(output, "a") as f_out, open(failure_path, "a") as f_fail:
        with multiprocessing.Pool(workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_worker, tasks),
                total=len(tasks),
                desc="Generating",
            ):
                # Accumulate rejection stats
                for reason, count in result["stats"].items():
                    agg["rejection_totals"][reason] = (
                        agg["rejection_totals"].get(reason, 0) + count
                    )

                if not result["success"]:
                    agg["failed"] += 1
                    f_fail.write(json.dumps(result) + "\n")
                    f_fail.flush()
                    continue

                # Dedup by fingerprint
                fp = result["fingerprint"]
                if fp in seen_fingerprints:
                    agg["duplicates"] += 1
                    continue

                seen_fingerprints.add(fp)
                agg["generated"] += 1
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()

    agg["elapsed_seconds"] = time.monotonic() - t0

    # Write aggregate stats
    stats_path = output.rsplit(".", 1)[0] + "_stats.json"
    with open(stats_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Done. {agg['generated']} instances, {agg['failed']} failures, {agg['duplicates']} duplicates.")
    print(f"Stats written to {stats_path}")

    return agg


def generate_to_target(
    templates: list[str],
    grid_sizes: list[int],
    puzzle_target: int,
    require_hard: bool = False,
    max_seeds: int = 2000,
    max_retries: int = 400,
    workers: int | None = None,
    output: str = "raw.jsonl",
    seed_start: int = 0,
    stall_seeds: int = 100,
    resume: bool = False,
) -> dict:
    """Generate until puzzle_target eval-ready unique instances are collected.

    Iterates seed-by-seed. Each seed's batch = all effective configs run in
    parallel. After each batch, results are deduplicated within the batch and
    against the global accepted set (by fingerprint). Only eval-ready,
    non-duplicate instances are written to output and counted toward the target.

    Args:
        puzzle_target: Stop when this many valid instances are accepted.
        require_hard: Passed to is_eval_ready — require hard RQ2/RQ3 variants.
        max_seeds: Hard ceiling on seeds tried before giving up.
        stall_seeds: Bail if this many consecutive seeds yield no new accepted
            instance.
        resume: If True and a checkpoint file exists alongside the output,
            continue from where the previous run stopped.

    Returns aggregate stats dict.
    """
    from tqdm import tqdm

    if workers is None:
        workers = os.cpu_count() or 1

    # Build effective config list filtered by tier
    configs: list[dict] = []
    for template in templates:
        tier = TEMPLATE_TIERS.get(template, "medium")
        allowed_sizes = TIER_GRID_SIZES.get(tier, [(5, 5), (6, 6)])
        for size in grid_sizes:
            if (size, size) not in allowed_sizes:
                continue
            configs.append({
                "template": template,
                "width": size,
                "height": size,
                "max_retries": max_retries,
                "require_hard": require_hard,
            })

    num_configs = len(configs)
    print(f"Configs per seed: {num_configs} | Target: {puzzle_target} | Max seeds: {max_seeds}")

    agg: dict[str, Any] = {
        "seeds_tried": 0,
        "total_tasks": 0,
        "raw_successes": 0,
        "accepted": 0,
        "failed": 0,
        "not_eval_ready": 0,
        "duplicates": 0,
        "elapsed_seconds": 0.0,
    }

    seen_fingerprints: set[str] = set()
    stall_count = 0
    t0 = time.monotonic()

    failure_path = output.rsplit(".", 1)[0] + "_failures.jsonl"

    # --- Resume: rebuild fingerprint set and find the last completed seed ---
    start_seed_offset = 0
    if resume and os.path.exists(output):
        max_seed_seen = -1

        with open(output) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _rec = json.loads(_line)
                    _fp = _rec.get("fingerprint")
                    if _fp:
                        seen_fingerprints.add(_fp)
                    _s = _rec.get("task", {}).get("seed")
                    if _s is not None:
                        max_seed_seen = max(max_seed_seen, _s)
                except json.JSONDecodeError:
                    continue

        if os.path.exists(failure_path):
            with open(failure_path) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _s = json.loads(_line).get("task", {}).get("seed")
                        if _s is not None:
                            max_seed_seen = max(max_seed_seen, _s)
                    except json.JSONDecodeError:
                        continue

        agg["accepted"] = len(seen_fingerprints)
        if max_seed_seen >= seed_start:
            start_seed_offset = max_seed_seen - seed_start + 1

        print(
            f"Resuming: {agg['accepted']} already accepted | "
            f"{len(seen_fingerprints)} fingerprints loaded | "
            f"continuing from seed offset {start_seed_offset}"
        )

    file_mode = "a" if (resume and os.path.exists(output)) else "w"
    failure_mode = "a" if (resume and os.path.exists(failure_path)) else "w"

    with open(output, file_mode) as f_out, open(failure_path, failure_mode) as f_fail:
        pbar = tqdm(total=puzzle_target, desc="Accepted", unit="puzzle", initial=agg["accepted"])
        with multiprocessing.Pool(workers) as pool:
            for seed_offset in range(start_seed_offset, max_seeds):
                if agg["accepted"] >= puzzle_target:
                    break

                seed = seed_start + seed_offset
                batch_tasks = [{**cfg, "seed": seed} for cfg in configs]

                batch_fps: set[str] = set()
                batch_accepted = 0
                batch_not_ready = 0
                batch_dups = 0
                batch_failed = 0
                tasks_done = 0

                pbar.set_postfix(
                    seed=seed,
                    progress=f"0/{num_configs}",
                    stall=stall_count,
                    refresh=True,
                )

                for result in pool.imap_unordered(_worker, batch_tasks):
                    agg["total_tasks"] += 1
                    tasks_done += 1

                    if not result["success"]:
                        agg["failed"] += 1
                        batch_failed += 1
                        f_fail.write(json.dumps(result) + "\n")
                        f_fail.flush()
                    elif not result["eval_ready"]:
                        agg["raw_successes"] += 1
                        agg["not_eval_ready"] += 1
                        batch_not_ready += 1
                    else:
                        agg["raw_successes"] += 1
                        fp = result["fingerprint"]

                        if fp in batch_fps or fp in seen_fingerprints:
                            agg["duplicates"] += 1
                            batch_dups += 1
                        else:
                            batch_fps.add(fp)
                            seen_fingerprints.add(fp)
                            agg["accepted"] += 1
                            batch_accepted += 1
                            f_out.write(json.dumps(result) + "\n")
                            f_out.flush()
                            pbar.update(1)

                    pbar.set_postfix(
                        seed=seed,
                        progress=f"{tasks_done}/{num_configs}",
                        stall=stall_count,
                        refresh=False,
                    )

                    if agg["accepted"] >= puzzle_target:
                        break

                agg["seeds_tried"] += 1

                pbar.write(
                    f"Seed {seed:4d} | +{batch_accepted} accepted  "
                    f"not_ready={batch_not_ready}  dup={batch_dups}  "
                    f"failed={batch_failed}  "
                    f"[total {agg['accepted']}/{puzzle_target}]"
                )

                if batch_accepted == 0:
                    stall_count += 1
                    if stall_count >= stall_seeds:
                        pbar.write(
                            f"Stalling: {stall_seeds} consecutive seeds with no new "
                            f"accepted puzzles. Accepted {agg['accepted']}/{puzzle_target}."
                        )
                        break
                else:
                    stall_count = 0

        pbar.close()

    agg["elapsed_seconds"] = round(time.monotonic() - t0, 2)
    print(f"Done. {agg['accepted']} accepted / {agg['raw_successes']} raw successes / "
          f"{agg['not_eval_ready']} not eval-ready / {agg['duplicates']} duplicates / "
          f"{agg['failed']} failed. Seeds tried: {agg['seeds_tried']}")
    return agg


def _task_key(task: dict) -> str:
    """Unique key for a generation task (for resume dedup)."""
    return f"{task.get('template')}_{task.get('width')}x{task.get('height')}_s{task.get('seed')}"


def _empty_agg_stats() -> dict:
    return {
        "total_tasks": 0,
        "resumed": 0,
        "generated": 0,
        "failed": 0,
        "duplicates": 0,
        "rejection_totals": {},
        "elapsed_seconds": 0.0,
    }
