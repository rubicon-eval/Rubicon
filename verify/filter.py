"""Filter validated puzzles into a clean output directory."""

from __future__ import annotations

import json
import os
import shutil


def filter_dataset(
    input_dir: str,
    output_dir: str,
    status: str = "validated",
    min_difficulty: int | None = None,
    max_difficulty: int | None = None,
) -> dict:
    """Copy validated puzzle dirs to output, rewrite eval_index.jsonl.

    Args:
        input_dir: Dataset root containing validation.json + puzzle dirs.
        output_dir: Clean output directory.
        status: Required validation status (default: "validated").
        min_difficulty: Minimum difficulty score (inclusive), or None.
        max_difficulty: Maximum difficulty score (inclusive), or None.

    Returns:
        Stats dict with counts.
    """
    val_path = os.path.join(input_dir, "validation.json")
    if not os.path.exists(val_path):
        print(f"No validation.json found in {input_dir}")
        return {"error": "no validation.json"}

    with open(val_path, "r") as f:
        validation = json.load(f)

    # Determine which puzzle dirs to keep
    kept: set[str] = set()
    for puzzle_dir, val in validation.items():
        if val.get("status") != status:
            continue
        diff = val.get("difficulty")
        if min_difficulty is not None and (diff is None or diff < min_difficulty):
            continue
        if max_difficulty is not None and (diff is None or diff > max_difficulty):
            continue
        kept.add(puzzle_dir)

    if not kept:
        print("No puzzles match the filter criteria.")
        return {"kept": 0, "items": 0}

    os.makedirs(output_dir, exist_ok=True)

    # Copy puzzle directories
    copied = 0
    for puzzle_dir in sorted(kept):
        src = os.path.join(input_dir, puzzle_dir)
        dst = os.path.join(output_dir, puzzle_dir)
        if not os.path.isdir(src):
            continue
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied += 1

    # Rewrite eval_index.jsonl — keep only entries for kept puzzles
    eval_in = os.path.join(input_dir, "eval_index.jsonl")
    eval_out = os.path.join(output_dir, "eval_index.jsonl")
    items_kept = 0
    items_total = 0

    if os.path.exists(eval_in):
        with open(eval_in, "r") as fin, open(eval_out, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                items_total += 1
                entry = json.loads(line)
                if entry.get("puzzle_dir") in kept:
                    fout.write(json.dumps(entry) + "\n")
                    items_kept += 1

    # Copy filtered validation.json
    filtered_val = {k: v for k, v in validation.items() if k in kept}
    with open(os.path.join(output_dir, "validation.json"), "w") as f:
        json.dump(filtered_val, f, indent=2)

    # Write filter stats
    stats = {
        "filter_status": status,
        "min_difficulty": min_difficulty,
        "max_difficulty": max_difficulty,
        "puzzles_kept": copied,
        "puzzles_total": len(validation),
        "items_kept": items_kept,
        "items_total": items_total,
    }
    with open(os.path.join(output_dir, "filter_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Filtered {copied}/{len(validation)} puzzles ({status})")
    print(f"Eval items: {items_kept}/{items_total}")
    print(f"Output: {output_dir}")

    return stats
