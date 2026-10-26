#!/usr/bin/env python3
"""Check whether wrong model predictions in RQ2-solvable lead to deadlock.

For the solvable set of RQ2 puzzles, when the model gives a "wrong" step
(doesn't match ground truth), this script checks:
  1. Is the predicted action even legal in the current state?
  2. If legal, does taking that action lead to a deadlock (Walking Dead state)?
  3. If legal, is the puzzle still solvable after that action?

Usage:
    python scripts/check_wrong_steps_deadlock.py --all
    python scripts/check_wrong_steps_deadlock.py --puzzle chain_vault
    python scripts/check_wrong_steps_deadlock.py --puzzle chain_vault --model gemma-3-12b-it

Results saved to: results/wrong_step_deadlock/
Puzzle cache:     .puzzle_cache/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

# Ensure the code/ directory is on sys.path for puzzle imports
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE_DIR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PUZZLE_TYPES = ["chain_vault", "mosaic_grid", "tide_grid"]

# Directions that are actual game moves (per puzzle type)
DIRECTIONAL_MOVES = {"UP", "DOWN", "LEFT", "RIGHT"}
TIDE_GRID_MOVES = {"UP", "DOWN", "LEFT", "RIGHT", "WAIT"}

# Non-move predictions the judge can emit (these are *not* game actions)
JUDGE_ARTIFACTS = {"UNCLEAR", "UNSOLVABLE", None}

CACHE_DIR = os.path.join(_CODE_DIR, ".puzzle_cache")


# ---------------------------------------------------------------------------
# Grid-only fingerprint computation (mirrors each puzzle's dataset.py)
# ---------------------------------------------------------------------------

def _pos_to_list(p) -> list[int]:
    return [p.row, p.col]


def _fingerprint_chain_vault(grid) -> str:
    canonical = {
        "w": grid.width, "h": grid.height,
        "walls": sorted([p.row * grid.width + p.col for p in grid.walls]),
        "start": _pos_to_list(grid.start),
        "exit": _pos_to_list(grid.exit_pos),
        "keys": sorted([[_pos_to_list(k.pos), k.color] for k in grid.keys.values()]),
        "doors": sorted([[_pos_to_list(d.pos), list(d.required_keys)] for d in grid.doors.values()]),
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]


def _fingerprint_mosaic_grid(grid) -> str:
    canonical = {
        "w": grid.width, "h": grid.height,
        "walls": sorted([p.row * grid.width + p.col for p in grid.walls]),
        "start": _pos_to_list(grid.start),
        "exit": _pos_to_list(grid.exit_pos),
        "plates": sorted([_pos_to_list(p.pos) for p in grid.plates.values()]),
        "doors": sorted([_pos_to_list(d.pos) for d in grid.doors.values()]),
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]


def _fingerprint_tide_grid(grid) -> str:
    canonical = {
        "w": grid.width, "h": grid.height,
        "walls": sorted([p.row * grid.width + p.col for p in grid.walls]),
        "start": _pos_to_list(grid.start),
        "exit": _pos_to_list(grid.exit_pos),
        "zones": sorted([
            [z.zone_id,
             sorted([p.row * grid.width + p.col for p in z.tiles]),
             z.dry_duration, z.flood_duration, z.phase_offset]
            for z in grid.tidal_zones.values()
        ]),
        "beacons": sorted([
            [_pos_to_list(b.pos), b.beacon_id] for b in grid.beacons.values()
        ]),
        "initial_oxygen": grid.initial_oxygen,
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]


_FINGERPRINTERS = {
    "chain_vault": _fingerprint_chain_vault,
    "mosaic_grid": _fingerprint_mosaic_grid,
    "tide_grid": _fingerprint_tide_grid,
}


def compute_grid_fingerprint(puzzle_type: str, grid) -> str:
    return _FINGERPRINTERS[puzzle_type](grid)


# ---------------------------------------------------------------------------
# PuzzleContext: grid + initial_state wrapper
# ---------------------------------------------------------------------------

@dataclass
class PuzzleContext:
    grid: object
    initial_state: object
    puzzle_type: str


# ---------------------------------------------------------------------------
# puzzle_dir -> generation params parsers
# ---------------------------------------------------------------------------

def _parse_puzzle_dir_chain_vault(puzzle_dir: str) -> dict:
    m = re.match(r"^(.+?)_(\d+)x(\d+)_(\d+)k_(\d+)d_seed(\d+)$", puzzle_dir)
    if not m:
        raise ValueError(f"Cannot parse chain_vault puzzle_dir: {puzzle_dir}")
    return dict(template=m[1], width=int(m[2]), height=int(m[3]),
                num_keys=int(m[4]), num_doors=int(m[5]), seed=int(m[6]))


def _parse_puzzle_dir_mosaic_grid(puzzle_dir: str) -> dict:
    m = re.match(r"^(.+?)_(\d+)x(\d+)_(\d+)d_seed(\d+)$", puzzle_dir)
    if not m:
        raise ValueError(f"Cannot parse mosaic_grid puzzle_dir: {puzzle_dir}")
    return dict(template=m[1], width=int(m[2]), height=int(m[3]),
                num_doors=int(m[4]), seed=int(m[5]))


def _parse_puzzle_dir_tide_grid(puzzle_dir: str) -> dict:
    m = re.match(r"^(.+?)_(\d+)x(\d+)_(\d+)b_(\d+)z_seed(\d+)$", puzzle_dir)
    if not m:
        raise ValueError(f"Cannot parse tide_grid puzzle_dir: {puzzle_dir}")
    return dict(template=m[1], width=int(m[2]), height=int(m[3]),
                num_beacons=int(m[4]), num_zones=int(m[5]), seed=int(m[6]))


_PARSERS = {
    "chain_vault": _parse_puzzle_dir_chain_vault,
    "mosaic_grid": _parse_puzzle_dir_mosaic_grid,
    "tide_grid": _parse_puzzle_dir_tide_grid,
}


# ---------------------------------------------------------------------------
# Disk cache: serialise / deserialise via each puzzle's dataset.py helpers
# ---------------------------------------------------------------------------

def _cache_path(puzzle_type: str, puzzle_dir: str) -> str:
    return os.path.join(CACHE_DIR, puzzle_type, f"{puzzle_dir}.json")


def _save_to_cache(puzzle_type: str, puzzle_dir: str, ctx: PuzzleContext,
                    fingerprint: str):
    """Persist grid + initial_state as JSON using the puzzle's own serialisers."""
    if puzzle_type == "chain_vault":
        from puzzles.chain_vault.dataset import _grid_to_dict, _state_to_dict
        payload = {"grid": _grid_to_dict(ctx.grid),
                    "initial_state": _state_to_dict(ctx.initial_state)}
    elif puzzle_type == "mosaic_grid":
        from puzzles.mosaic_grid.dataset import _grid_to_dict, _state_to_dict
        payload = {"grid": _grid_to_dict(ctx.grid),
                    "initial_state": _state_to_dict(ctx.initial_state, ctx.grid)}
    elif puzzle_type == "tide_grid":
        from puzzles.tide_grid.dataset import _grid_to_dict, _state_to_dict
        payload = {"grid": _grid_to_dict(ctx.grid),
                    "initial_state": _state_to_dict(ctx.initial_state)}
    else:
        return

    payload["fingerprint"] = fingerprint
    fpath = _cache_path(puzzle_type, puzzle_dir)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(payload, f)


def _load_from_cache(
    puzzle_type: str, puzzle_dir: str,
) -> Optional[tuple[PuzzleContext, str]]:
    """Load grid + initial_state from disk cache.

    Returns (PuzzleContext, cached_fingerprint) or None.
    """
    fpath = _cache_path(puzzle_type, puzzle_dir)
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath) as f:
            payload = json.load(f)
        if puzzle_type == "chain_vault":
            from puzzles.chain_vault.dataset import _dict_to_grid, _dict_to_state
            grid = _dict_to_grid(payload["grid"])
            state = _dict_to_state(payload["initial_state"])
        elif puzzle_type == "mosaic_grid":
            from puzzles.mosaic_grid.dataset import _dict_to_grid, _dict_to_state
            grid = _dict_to_grid(payload["grid"])
            state = _dict_to_state(payload["initial_state"], grid)
        elif puzzle_type == "tide_grid":
            from puzzles.tide_grid.dataset import _dict_to_grid, _dict_to_state
            grid = _dict_to_grid(payload["grid"])
            state = _dict_to_state(payload["initial_state"])
        else:
            return None
        cached_fp = payload.get("fingerprint", "")
        return (PuzzleContext(grid=grid, initial_state=state,
                              puzzle_type=puzzle_type),
                cached_fp)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Puzzle regeneration (in-memory cache + disk cache)
# ---------------------------------------------------------------------------

_mem_cache: dict[str, PuzzleContext] = {}


def _regenerate(puzzle_type: str, params: dict) -> PuzzleContext:
    """Regenerate a puzzle from its generation parameters."""
    if puzzle_type == "chain_vault":
        from puzzles.chain_vault.generator import generate_validated_instance
        inst = generate_validated_instance(
            width=params["width"], height=params["height"],
            seed=params["seed"], template=params["template"], max_retries=400)
    elif puzzle_type == "mosaic_grid":
        from puzzles.mosaic_grid.generator import generate_validated_instance
        inst = generate_validated_instance(
            width=params["width"], height=params["height"],
            seed=params["seed"], template=params["template"],
            num_doors=params["num_doors"], max_retries=400)
    elif puzzle_type == "tide_grid":
        from puzzles.tide_grid.generator import generate_validated_instance
        inst = generate_validated_instance(
            width=params["width"], height=params["height"],
            seed=params["seed"], template=params["template"], max_retries=400)
    else:
        raise ValueError(f"Unknown puzzle type: {puzzle_type}")

    if inst is None:
        raise RuntimeError(f"Failed to regenerate {puzzle_type}: {params}")
    return PuzzleContext(grid=inst.grid, initial_state=inst.initial_state,
                         puzzle_type=puzzle_type)



def preload_puzzles(puzzle_type: str, puzzle_dirs_with_fp: dict[str, str]):
    """Preload all needed puzzles with fingerprint verification.

    Args:
        puzzle_dirs_with_fp: mapping puzzle_dir -> expected_fingerprint
    """
    to_load = [pd for pd in sorted(puzzle_dirs_with_fp)
               if f"{puzzle_type}::{pd}" not in _mem_cache]
    if not to_load:
        return 0

    from_disk = 0
    regenerated = 0
    failures = 0
    fp_mismatches = 0

    for i, pd in enumerate(to_load, 1):
        sys.stdout.write(f"\r    Loading puzzles: {i}/{len(to_load)}")
        sys.stdout.flush()
        expected_fp = puzzle_dirs_with_fp[pd]
        try:
            key = f"{puzzle_type}::{pd}"

            # Try disk cache first
            cached = _load_from_cache(puzzle_type, pd)
            if cached is not None:
                ctx, cached_fp = cached
                # Verify fingerprint: use cached fp if available,
                # otherwise recompute from grid
                check_fp = cached_fp or compute_grid_fingerprint(
                    puzzle_type, ctx.grid)
                if expected_fp and check_fp != expected_fp:
                    fp_mismatches += 1
                    continue
                _mem_cache[key] = ctx
                from_disk += 1
            else:
                # Regenerate
                params = _PARSERS[puzzle_type](pd)
                ctx = _regenerate(puzzle_type, params)
                # Verify regenerated fingerprint
                actual_fp = compute_grid_fingerprint(puzzle_type, ctx.grid)
                if expected_fp and actual_fp != expected_fp:
                    fp_mismatches += 1
                    continue
                _mem_cache[key] = ctx
                _save_to_cache(puzzle_type, pd, ctx, actual_fp)
                regenerated += 1
        except Exception:
            failures += 1

    status = (f"cached: {from_disk}, regenerated: {regenerated}, "
              f"failed: {failures}")
    if fp_mismatches:
        status += (f", \033[91mfp_mismatch: {fp_mismatches} — these puzzles "
                   f"will be SKIPPED (generator changed since dataset was "
                   f"built)\033[0m")
    print(f"\r    Loading puzzles: {len(to_load)}/{len(to_load)}  ({status})")
    return failures + fp_mismatches


# ---------------------------------------------------------------------------
# Action helpers (per puzzle type)
# ---------------------------------------------------------------------------

def _str_to_action(puzzle_type: str, s: str):
    if puzzle_type == "chain_vault":
        from puzzles.chain_vault.engine import ChainVaultAction
        return ChainVaultAction[s]
    elif puzzle_type == "mosaic_grid":
        from puzzles.common.types import Direction
        return Direction[s]
    elif puzzle_type == "tide_grid":
        from puzzles.tide_grid.engine import TideGridAction
        return TideGridAction[s]


def replay_actions(ctx: PuzzleContext, action_strs: list[str]):
    state = ctx.initial_state
    if ctx.puzzle_type == "chain_vault":
        from puzzles.chain_vault.engine import apply_action
    elif ctx.puzzle_type == "mosaic_grid":
        from puzzles.mosaic_grid.engine import apply_action
    elif ctx.puzzle_type == "tide_grid":
        from puzzles.tide_grid.engine import apply_action
    for a_str in action_strs:
        state = apply_action(ctx.grid, state, _str_to_action(ctx.puzzle_type, a_str))
    return state


def is_legal(ctx: PuzzleContext, state, action_str: str) -> bool:
    try:
        action = _str_to_action(ctx.puzzle_type, action_str)
    except (KeyError, ValueError):
        return False
    if ctx.puzzle_type == "chain_vault":
        from puzzles.chain_vault.engine import is_legal_action
        return is_legal_action(ctx.grid, state, action)
    elif ctx.puzzle_type == "mosaic_grid":
        from puzzles.mosaic_grid.engine import get_legal_actions
        return action in get_legal_actions(ctx.grid, state)
    elif ctx.puzzle_type == "tide_grid":
        from puzzles.tide_grid.engine import get_legal_actions
        return action in get_legal_actions(ctx.grid, state)
    return False


def apply_single(ctx: PuzzleContext, state, action_str: str):
    if ctx.puzzle_type == "chain_vault":
        from puzzles.chain_vault.engine import apply_action
    elif ctx.puzzle_type == "mosaic_grid":
        from puzzles.mosaic_grid.engine import apply_action
    elif ctx.puzzle_type == "tide_grid":
        from puzzles.tide_grid.engine import apply_action
    return apply_action(ctx.grid, state, _str_to_action(ctx.puzzle_type, action_str))


def check_solvable(ctx: PuzzleContext, state) -> bool:
    if ctx.puzzle_type == "chain_vault":
        from puzzles.chain_vault.oracle import is_solvable
    elif ctx.puzzle_type == "mosaic_grid":
        from puzzles.mosaic_grid.oracle import is_solvable
    elif ctx.puzzle_type == "tide_grid":
        from puzzles.tide_grid.oracle import is_solvable
    return is_solvable(ctx.grid, state)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WrongStepResult:
    eval_id: str
    puzzle_dir: str
    ground_truth: str
    predicted: str
    category: str           # "judge_artifact" | "illegal" | "deadlock" | "still_solvable" | "error"
    is_valid_move: bool
    is_legal: bool
    leads_to_deadlock: bool
    still_solvable: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _valid_moves_for(puzzle_type: str) -> set[str]:
    return TIDE_GRID_MOVES if puzzle_type == "tide_grid" else DIRECTIONAL_MOVES


def analyze_wrong_steps(
    puzzle_type: str,
    model_name: str,
    base_dir: str = ".",
) -> list[WrongStepResult]:
    dataset_path = os.path.join(
        base_dir, "dataset", f"dataset_{puzzle_type}", "rq2_hard_solvable.json")
    results_path = os.path.join(
        base_dir, "results", puzzle_type, model_name, "rq2_hard_solvable.jsonl")

    if not os.path.exists(dataset_path):
        print(f"    [SKIP] Dataset not found: {dataset_path}")
        return []
    if not os.path.exists(results_path):
        print(f"    [SKIP] Results not found: {results_path}")
        return []

    with open(dataset_path) as f:
        dataset = json.load(f)
    instances_by_id = {inst["eval_id"]: inst for inst in dataset}

    results = []
    with open(results_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("_type") == "summary":
                continue
            results.append(rec)

    wrong = [r for r in results if not r.get("correct", True)]
    if not wrong:
        print(f"    No wrong predictions — skipping")
        return []

    valid_moves = _valid_moves_for(puzzle_type)

    analysis: list[WrongStepResult] = []
    for i, r in enumerate(wrong, 1):
        sys.stdout.write(f"\r    Checking solvability: {i}/{len(wrong)}")
        sys.stdout.flush()

        eval_id = r["eval_id"]
        predicted = r.get("predicted")
        ground_truth = r.get("ground_truth", "")

        inst = instances_by_id.get(eval_id)
        if inst is None:
            analysis.append(WrongStepResult(
                eval_id=eval_id, puzzle_dir="", ground_truth=ground_truth,
                predicted=str(predicted), category="error",
                is_valid_move=False, is_legal=False,
                leads_to_deadlock=False, still_solvable=False,
                error="instance not found in dataset"))
            continue

        puzzle_dir = inst["puzzle_dir"]
        actions = inst["metadata"]["actions"]

        # --- Category 1: judge artifact (UNCLEAR, None, UNSOLVABLE) ---
        if predicted in JUDGE_ARTIFACTS or predicted not in valid_moves:
            analysis.append(WrongStepResult(
                eval_id=eval_id, puzzle_dir=puzzle_dir,
                ground_truth=ground_truth, predicted=str(predicted),
                category="judge_artifact",
                is_valid_move=False, is_legal=False,
                leads_to_deadlock=False, still_solvable=False,
                error=f"judge artifact: {predicted}"))
            continue

        key = f"{puzzle_type}::{puzzle_dir}"
        if key not in _mem_cache:
            analysis.append(WrongStepResult(
                eval_id=eval_id, puzzle_dir=puzzle_dir,
                ground_truth=ground_truth, predicted=predicted,
                category="error",
                is_valid_move=True, is_legal=False,
                leads_to_deadlock=False, still_solvable=False,
                error="puzzle not loaded (fingerprint mismatch or generation failure)"))
            continue

        ctx = _mem_cache[key]

        try:
            state = replay_actions(ctx, actions)
            legal = is_legal(ctx, state, predicted)

            if not legal:
                analysis.append(WrongStepResult(
                    eval_id=eval_id, puzzle_dir=puzzle_dir,
                    ground_truth=ground_truth, predicted=predicted,
                    category="illegal",
                    is_valid_move=True, is_legal=False,
                    leads_to_deadlock=False, still_solvable=False))
                continue

            new_state = apply_single(ctx, state, predicted)
            solvable = check_solvable(ctx, new_state)

            analysis.append(WrongStepResult(
                eval_id=eval_id, puzzle_dir=puzzle_dir,
                ground_truth=ground_truth, predicted=predicted,
                category="still_solvable" if solvable else "deadlock",
                is_valid_move=True, is_legal=True,
                leads_to_deadlock=not solvable,
                still_solvable=solvable))

        except Exception as e:
            analysis.append(WrongStepResult(
                eval_id=eval_id, puzzle_dir=puzzle_dir,
                ground_truth=ground_truth, predicted=predicted,
                category="error",
                is_valid_move=True, is_legal=False,
                leads_to_deadlock=False, still_solvable=False,
                error=str(e)))

    print()  # newline after progress
    return analysis


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(puzzle_type: str, model_name: str, results: list[WrongStepResult]):
    if not results:
        return

    total = len(results)
    by_cat = {}
    for r in results:
        by_cat[r.category] = by_cat.get(r.category, 0) + 1

    judge = by_cat.get("judge_artifact", 0)
    illegal = by_cat.get("illegal", 0)
    deadlock = by_cat.get("deadlock", 0)
    solvable = by_cat.get("still_solvable", 0)
    errors = by_cat.get("error", 0)

    parts = [f"Total wrong: {total}"]
    if judge:
        parts.append(f"Judge artifacts: {judge}")
    if illegal:
        parts.append(f"Illegal: {illegal}")
    parts.append(f"Deadlock: {deadlock}")
    parts.append(f"Still solvable: {solvable}")
    if errors:
        parts.append(f"Errors: {errors}")
    print(f"    {' | '.join(parts)}")

    legal_total = deadlock + solvable
    if legal_total > 0:
        pct = deadlock / legal_total * 100
        print(f"    --> Of legal wrong moves: {pct:.1f}% lead to deadlock, "
              f"{100-pct:.1f}% remain solvable")


def save_detailed_results(
    puzzle_type: str, model_name: str,
    results: list[WrongStepResult], output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir,
                         f"{puzzle_type}__{model_name}__wrong_step_deadlock.jsonl")

    by_cat = {}
    for r in results:
        by_cat[r.category] = by_cat.get(r.category, 0) + 1

    deadlock = by_cat.get("deadlock", 0)
    solvable = by_cat.get("still_solvable", 0)

    summary = {
        "_type": "summary",
        "_description": (
            "Analysis of wrong model predictions on RQ2-solvable instances. "
            "For each instance the model was shown a solvable puzzle state and "
            "asked for the best next move. The ground truth is the oracle-optimal "
            "next action. This file categorises every wrong prediction."
        ),
        "_category_definitions": {
            "judge_artifact": (
                "The LLM judge could not extract a game action from the model's "
                "response. Includes: UNCLEAR (judge unsure), None (parse failure), "
                "UNSOLVABLE (model incorrectly claimed the puzzle is unsolvable), "
                "WAIT on non-tide_grid puzzles (WAIT is not a valid action for "
                "chain_vault or mosaic_grid). These are NOT game moves at all."
            ),
            "illegal": (
                "The model predicted a real directional move (UP/DOWN/LEFT/RIGHT "
                "or WAIT for tide_grid) but that move is ILLEGAL in the current "
                "game state — e.g. moving into a wall or locked door. The game "
                "engine rejects this action."
            ),
            "deadlock": (
                "The model predicted a legal move (the game engine accepts it), "
                "but after taking that move the puzzle becomes UNSOLVABLE. "
                "The agent enters a Walking Dead state — it can still make legal "
                "moves but can never reach the exit. This is the worst kind of "
                "wrong answer: not just suboptimal, but fatal."
            ),
            "still_solvable": (
                "The model predicted a legal move that is different from the "
                "oracle-optimal move, but the puzzle REMAINS SOLVABLE after "
                "taking it. The move is suboptimal (not on the shortest path) "
                "but not fatal — the agent can still eventually reach the exit."
            ),
            "error": (
                "An unexpected error occurred during analysis (e.g. puzzle "
                "regeneration failed, instance not found). Should be rare."
            ),
        },
        "_field_definitions": {
            "eval_id": "Unique identifier for this evaluation instance",
            "puzzle_dir": "Puzzle directory name encoding template, grid size, and seed",
            "ground_truth": "The oracle-optimal next move for this state",
            "predicted": "What the model predicted (may be a move, UNCLEAR, None, etc.)",
            "category": "One of: judge_artifact, illegal, deadlock, still_solvable, error",
            "is_valid_move": "Is the prediction a real game action name? (false for UNCLEAR/None/UNSOLVABLE)",
            "is_legal": "If it is a valid move name, is it legal in the current game state?",
            "leads_to_deadlock": "If legal, does taking this move make the puzzle unsolvable?",
            "still_solvable": "If legal, does the puzzle remain solvable after this move?",
            "error": "Error message if something went wrong, null otherwise",
        },
        "puzzle": puzzle_type,
        "model": model_name,
        "total_wrong": len(results),
        "judge_artifacts": by_cat.get("judge_artifact", 0),
        "illegal": by_cat.get("illegal", 0),
        "deadlock": deadlock,
        "still_solvable": solvable,
        "errors": by_cat.get("error", 0),
        "deadlock_rate_of_legal_wrong": (
            round(deadlock / (deadlock + solvable) * 100, 2)
            if (deadlock + solvable) > 0 else None),
        "_deadlock_rate_explanation": (
            "deadlock / (deadlock + still_solvable) * 100. "
            "This is the percentage of legal-but-wrong moves that lead to a "
            "Walking Dead state. Excludes judge_artifacts and illegal moves."
        ),
    }

    with open(fpath, "w") as f:
        f.write(json.dumps(summary, indent=2) + "\n")
        for r in results:
            f.write(json.dumps({
                "eval_id": r.eval_id,
                "puzzle_dir": r.puzzle_dir,
                "ground_truth": r.ground_truth,
                "predicted": r.predicted,
                "category": r.category,
                "is_valid_move": r.is_valid_move,
                "is_legal": r.is_legal,
                "leads_to_deadlock": r.leads_to_deadlock,
                "still_solvable": r.still_solvable,
                "error": r.error,
            }) + "\n")


def print_aggregate_summary(
    all_results: dict[tuple[str, str], list[WrongStepResult]],
):
    print(f"\n{'='*100}")
    print(f"  AGGREGATE SUMMARY: Wrong Steps -> Deadlock Analysis")
    print(f"{'='*100}")
    print(f"  {'Puzzle':<16} {'Model':<35} {'Wrong':>5} {'Judge':>5} "
          f"{'Illegal':>7} {'Deadlk':>6} {'Solv':>5} {'DL%':>6}")
    print(f"  {'-'*16} {'-'*35} {'-'*5} {'-'*5} {'-'*7} {'-'*6} {'-'*5} {'-'*6}")

    for (puzzle_type, model_name), results in sorted(all_results.items()):
        total = len(results)
        by_cat = {}
        for r in results:
            by_cat[r.category] = by_cat.get(r.category, 0) + 1
        judge = by_cat.get("judge_artifact", 0)
        illegal = by_cat.get("illegal", 0)
        deadlock = by_cat.get("deadlock", 0)
        solvable = by_cat.get("still_solvable", 0)
        legal_total = deadlock + solvable
        dl_pct = f"{deadlock/legal_total*100:.1f}" if legal_total > 0 else "N/A"

        print(f"  {puzzle_type:<16} {model_name:<35} {total:>5} {judge:>5} "
              f"{illegal:>7} {deadlock:>6} {solvable:>5} {dl_pct:>6}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    os.chdir(_CODE_DIR)

    parser = argparse.ArgumentParser(
        description="Check if wrong RQ2-solvable predictions lead to deadlock.")
    parser.add_argument("--puzzle", type=str, default=None,
        help="Puzzle type (chain_vault, mosaic_grid, tide_grid). Default: all.")
    parser.add_argument("--model", type=str, default=None,
        help="Model name (subfolder in results/). Default: all models.")
    parser.add_argument("--all", action="store_true",
        help="Run for all puzzles and all models.")
    parser.add_argument("--output-dir", type=str,
        default="results/wrong_step_deadlock",
        help="Directory to save detailed JSONL results.")
    parser.add_argument("--no-save", action="store_true",
        help="Don't save detailed results, just print summary.")
    args = parser.parse_args()

    puzzle_types = [args.puzzle] if args.puzzle else PUZZLE_TYPES

    # Pre-scan: figure out total work and collect all puzzle_dirs per puzzle type
    # Maps puzzle_type -> {puzzle_dir -> expected_fingerprint}
    work_items: list[tuple[str, str]] = []
    all_puzzle_dirs: dict[str, dict[str, str]] = {pt: {} for pt in puzzle_types}

    for puzzle_type in puzzle_types:
        dataset_path = os.path.join(
            "dataset", f"dataset_{puzzle_type}", "rq2_hard_solvable.json")
        if not os.path.exists(dataset_path):
            continue
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Collect ALL puzzle_dirs with their fingerprints from the dataset
        for inst in dataset:
            pd = inst["puzzle_dir"]
            fp = inst["metadata"].get("fingerprint", "")
            all_puzzle_dirs[puzzle_type][pd] = fp

        if args.model:
            models = [args.model]
        else:
            results_dir = os.path.join("results", puzzle_type)
            if not os.path.isdir(results_dir):
                continue
            models = sorted(m for m in os.listdir(results_dir)
                            if os.path.isdir(os.path.join(results_dir, m)))
        for model_name in models:
            work_items.append((puzzle_type, model_name))

    total_combos = len(work_items)
    print(f"Processing {total_combos} puzzle/model combinations\n")

    global_start = time.time()
    combo_idx = 0
    all_results: dict[tuple[str, str], list[WrongStepResult]] = {}

    for puzzle_type in puzzle_types:
        if args.model:
            models = [args.model]
        else:
            results_dir = os.path.join("results", puzzle_type)
            if not os.path.isdir(results_dir):
                print(f"[SKIP] No results directory: {results_dir}")
                continue
            models = sorted(m for m in os.listdir(results_dir)
                            if os.path.isdir(os.path.join(results_dir, m)))

        print(f"{'─'*70}")
        print(f"  PUZZLE: {puzzle_type}  ({len(models)} models)")
        print(f"{'─'*70}")

        # Preload all puzzles for this puzzle type ONCE before iterating models
        dirs_with_fp = all_puzzle_dirs.get(puzzle_type, {})
        if dirs_with_fp:
            preload_puzzles(puzzle_type, dirs_with_fp)

        for model_name in models:
            combo_idx += 1
            elapsed = time.time() - global_start
            print(f"\n  [{combo_idx}/{total_combos}] {model_name}  "
                  f"(elapsed: {elapsed:.0f}s)")

            results = analyze_wrong_steps(puzzle_type, model_name, base_dir=".")
            if results:
                all_results[(puzzle_type, model_name)] = results
                print_summary(puzzle_type, model_name, results)
                if not args.no_save:
                    save_detailed_results(
                        puzzle_type, model_name, results, args.output_dir)

    if len(all_results) > 1:
        print_aggregate_summary(all_results)

    total_elapsed = time.time() - global_start
    print(f"\nDone in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
