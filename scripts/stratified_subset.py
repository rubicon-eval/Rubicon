#!/usr/bin/env python3
"""Create a stratified random subset of puzzle datasets.

Selects N puzzles from each puzzle type using stratified random sampling
on (grid_size, trajectory_length) to preserve the distributional properties
of the full dataset. All 7 eval files are subsetted consistently (same
puzzles kept across rq1, rq2, rq3).

Usage:
    python -m scripts.stratified_subset --n 50 --seed 42
    python -m scripts.stratified_subset --n 50 --seed 42 --dry-run
    python -m scripts.stratified_subset --n 50 --seed 42 -o dataset/subsets
"""

import argparse
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
PUZZLE_TYPES = ["dataset_chain_vault", "dataset_mosaic_grid", "dataset_tide_grid"]
EVAL_FILES = [
    "rq1_main.json",
    "rq2_hard_deadlock.json",
    "rq2_hard_solvable.json",
    "rq2_hard_ablation_deadlock.json",
    "rq2_hard_ablation_solvable.json",
    "rq3_hard.json",
    "rq3_hard_ablation_final.json",
]


def load_items(dataset_path: Path, filename: str) -> list[dict]:
    with open(dataset_path / filename) as f:
        return json.load(f)


def get_puzzle_id(item: dict) -> str:
    """Extract a stable puzzle identifier from an eval item."""
    return item["metadata"]["fingerprint"]


def get_strata_key(item: dict) -> str:
    """Build stratification key from grid_size and trajectory_length."""
    m = item["metadata"]
    grid = m.get("grid_size", "?")
    traj = m.get("trajectory_length", 0)
    # Bucket trajectory length into {5, 10, 15} bands
    if traj <= 7:
        traj_bucket = 5
    elif traj <= 12:
        traj_bucket = 10
    else:
        traj_bucket = 15
    return f"{grid}_traj{traj_bucket}"


def stratified_sample(items: list[dict], n: int, rng: random.Random) -> set[str]:
    """Select n puzzle fingerprints via proportional stratified sampling.

    1. Group puzzles by (grid_size, trajectory_length_bucket).
    2. Allocate floor(n * stratum_size / total) to each stratum.
    3. Distribute remaining slots to largest-remainder strata.
    4. Randomly sample within each stratum.

    Returns set of selected fingerprints.
    """
    # Group unique puzzles by stratum
    strata: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for item in items:
        fp = get_puzzle_id(item)
        if fp in seen:
            continue
        seen.add(fp)
        key = get_strata_key(item)
        strata[key].append(fp)

    total = len(seen)
    if n > total:
        print(f"  Warning: requested {n} but only {total} puzzles available, taking all")
        return seen

    # Proportional allocation with largest-remainder method
    allocations = {}
    remainders = {}
    allocated = 0
    for key, fps in sorted(strata.items()):
        exact = n * len(fps) / total
        floor_val = math.floor(exact)
        allocations[key] = floor_val
        remainders[key] = exact - floor_val
        allocated += floor_val

    # Distribute remaining slots by largest remainder
    remaining = n - allocated
    for key in sorted(remainders, key=lambda k: remainders[k], reverse=True):
        if remaining <= 0:
            break
        # Don't exceed stratum size
        if allocations[key] < len(strata[key]):
            allocations[key] += 1
            remaining -= 1

    # Sample within each stratum
    selected = set()
    for key, fps in sorted(strata.items()):
        k = allocations[key]
        chosen = rng.sample(fps, k)
        selected.update(chosen)

    return selected


def subset_eval_file(items: list[dict], selected_fps: set[str]) -> list[dict]:
    """Filter eval items to only those whose puzzle fingerprint is selected."""
    return [item for item in items if get_puzzle_id(item) in selected_fps]


def print_distribution(items: list[dict], label: str):
    """Print the stratification distribution for a set of items."""
    strata = defaultdict(int)
    seen = set()
    for item in items:
        fp = get_puzzle_id(item)
        if fp in seen:
            continue
        seen.add(fp)
        strata[get_strata_key(item)] += 1
    print(f"  {label} (N={len(seen)}): {dict(sorted(strata.items()))}")


def process_puzzle_type(puzzle_type: str, n: int, rng: random.Random,
                        output_dir: Path | None, dry_run: bool):
    src = DATASET_DIR / puzzle_type
    if not src.is_dir():
        print(f"  Skipping {puzzle_type}: directory not found")
        return

    print(f"\n{'='*60}")
    print(f"  {puzzle_type}")
    print(f"{'='*60}")

    # Use rq1_main.json as the reference for stratification (1 item per puzzle)
    rq1_items = load_items(src, "rq1_main.json")

    # Show original distribution
    print_distribution(rq1_items, "Original")

    # Stratified sampling
    selected_fps = stratified_sample(rq1_items, n, rng)
    print(f"  Selected {len(selected_fps)} puzzles")

    # Show subset distribution
    subset_rq1 = subset_eval_file(rq1_items, selected_fps)
    print_distribution(subset_rq1, "Subset  ")

    if dry_run:
        print("  [Dry run] No files written.")
        return selected_fps

    # Write subsetted eval files
    out = output_dir / puzzle_type
    out.mkdir(parents=True, exist_ok=True)

    total_items = 0
    for eval_file in EVAL_FILES:
        items = load_items(src, eval_file)
        subset = subset_eval_file(items, selected_fps)
        total_items += len(subset)
        with open(out / eval_file, "w") as f:
            json.dump(subset, f, indent=2)
            f.write("\n")
        print(f"  {eval_file}: {len(items)} -> {len(subset)}")

    # Copy corresponding images
    src_images = src / "images"
    dst_images = out / "images"
    if src_images.is_dir():
        dst_images.mkdir(exist_ok=True)
        copied = 0
        for img_dir in sorted(src_images.iterdir()):
            if not img_dir.is_dir():
                continue
            # Check if any selected puzzle references this image dir
            # Image dirs are named by puzzle_dir, match via items
            # We copy all image dirs that are referenced by selected items
            should_copy = False
            for item in subset_rq1:
                img_path = item.get("image_path", "")
                if img_dir.name in img_path:
                    should_copy = True
                    break
            if should_copy:
                dst_img = dst_images / img_dir.name
                if dst_img.exists():
                    shutil.rmtree(dst_img)
                shutil.copytree(img_dir, dst_img)
                copied += 1
        print(f"  Images: copied {copied} puzzle image directories")

    # Copy replacements.json if it exists
    replacements = src / "replacements.json"
    if replacements.exists():
        shutil.copy2(replacements, out / "replacements.json")

    # Write build_stats
    stats = {"total_puzzles": len(selected_fps), "total_eval_items": total_items}
    for eval_file in EVAL_FILES:
        key = eval_file.replace(".json", "")
        with open(out / eval_file) as f:
            stats[key] = len(json.load(f))
    stats["subset_of"] = puzzle_type
    stats["subset_size"] = n
    stats["sampling_method"] = "stratified_random"
    stats["strata"] = "grid_size x trajectory_length_bucket"

    with open(out / "build_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    # Write the selected fingerprints for reproducibility / cross-referencing
    with open(out / "selected_puzzles.json", "w") as f:
        json.dump(sorted(selected_fps), f, indent=2)
        f.write("\n")

    print(f"  Output written to {out}/")
    return selected_fps


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified random subsets of puzzle datasets."
    )
    parser.add_argument("--n", type=int, default=50,
                        help="Number of puzzles to select per type (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory (default: dataset/subset_N)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show distributions without writing files")
    parser.add_argument("--types", nargs="+", default=PUZZLE_TYPES,
                        help="Puzzle types to process (default: all three)")

    args = parser.parse_args()
    rng = random.Random(args.seed)

    output_dir = args.output or DATASET_DIR / f"subset_{args.n}"

    print(f"Stratified subset selection: {args.n} puzzles per type, seed={args.seed}")
    if not args.dry_run:
        print(f"Output directory: {output_dir}")
        if output_dir.exists():
            print(f"Error: {output_dir} already exists. Remove it or choose a different path.")
            sys.exit(1)

    for puzzle_type in args.types:
        process_puzzle_type(puzzle_type, args.n, rng, output_dir, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
