#!/usr/bin/env python3
"""Filter a puzzle dataset by human-verified difficulty and status.

Usage examples:
    # Keep puzzles with difficulty >= 3
    python -m scripts.filter_dataset dataset_50 verification.json --min-difficulty 3

    # Keep puzzles with difficulty <= 2
    python -m scripts.filter_dataset dataset_50 verification.json --max-difficulty 2

    # Combine: difficulty between 2 and 4
    python -m scripts.filter_dataset dataset_50 verification.json --min-difficulty 2 --max-difficulty 4

    # Keep only verified puzzles with difficulty >= 2
    python -m scripts.filter_dataset dataset_50 verification.json --min-difficulty 2 --status approved

    # Filter by status only
    python -m scripts.filter_dataset dataset_50 verification.json --status approved

    # Custom output directory
    python -m scripts.filter_dataset dataset_50 verification.json --min-difficulty 3 -o dataset_hard

    # Preview what would be kept without writing files
    python -m scripts.filter_dataset dataset_50 verification.json --min-difficulty 3 --dry-run
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path


def load_verification(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def filter_puzzles(verification: dict, min_diff: int | None, max_diff: int | None,
                   status: str | None) -> set[str]:
    """Return set of puzzle names that pass the filters."""
    kept = set()
    for name, info in verification.items():
        d = info.get("difficulty", 0)
        s = info.get("status", "")
        if min_diff is not None and d < min_diff:
            continue
        if max_diff is not None and d > max_diff:
            continue
        if status is not None and s != status:
            continue
        kept.add(name)
    return kept


def build_filtered_dataset(src: Path, out: Path, kept: set[str], dry_run: bool):
    if out.exists() and not dry_run:
        print(f"Error: output directory {out} already exists. Remove it or choose a different name.")
        sys.exit(1)

    # Discover which puzzle dirs actually exist in source
    available = {p.name for p in src.iterdir() if p.is_dir()}
    selected = kept & available
    missing_from_src = kept - available
    skipped = available - kept

    print(f"Verification entries matched: {len(kept)}")
    print(f"Puzzles in source dataset:    {len(available)}")
    print(f"Puzzles selected (in both):   {len(selected)}")
    if missing_from_src:
        print(f"In verification but not in source ({len(missing_from_src)}): "
              f"{sorted(missing_from_src)[:5]}{'...' if len(missing_from_src) > 5 else ''}")
    print(f"Puzzles filtered out:         {len(skipped)}")

    if dry_run:
        print("\n[Dry run] No files written.")
        return

    out.mkdir(parents=True)
    images_dir = out / "images"
    images_dir.mkdir()

    # Collect eval items per rq_variant, and track referenced images
    rq_buckets: dict[str, list] = {}
    referenced_images: set[str] = set()  # relative paths like "puzzle_name/images/foo.png"

    # First pass: collect rq1_main items per puzzle for sampling
    RQ1_TARGETS = [5, 10, 15]
    rq1_by_puzzle: dict[str, list] = {}

    for name in sorted(selected):
        items_path = src / name / "eval_items.json"
        if not items_path.exists():
            continue
        with open(items_path) as f:
            items = json.load(f)
        for item in items:
            rq = item.get("rq", "")
            variant = item.get("variant", "")
            if rq == "rq1" and variant == "main":
                rq1_by_puzzle.setdefault(name, []).append(item)

    # Sample 1 rq1_main per puzzle, balancing across target trajectory length buckets.
    # For each puzzle, assign it to the most underrepresented bucket that the
    # puzzle can actually serve (has a candidate within reasonable range).
    BUCKET_RANGE = {5: (1, 7), 10: (8, 12), 15: (13, 999)}
    bucket_counts = {t: 0 for t in RQ1_TARGETS}
    rq1_selected: set[str] = set()  # eval_ids to keep

    def available_buckets(candidates):
        """Return which target buckets this puzzle has candidates for."""
        lengths = {c["metadata"].get("trajectory_length", 0) for c in candidates}
        avail = []
        for t, (lo, hi) in BUCKET_RANGE.items():
            if any(lo <= l <= hi for l in lengths):
                avail.append(t)
        return avail

    for name in sorted(rq1_by_puzzle):
        candidates = rq1_by_puzzle[name]
        avail = available_buckets(candidates)
        if not avail:
            # No bucket match — just pick closest to any target
            best = min(candidates, key=lambda c: min(abs(c["metadata"].get("trajectory_length", 0) - t) for t in RQ1_TARGETS))
            rq1_selected.add(best["eval_id"])
            continue

        # Pick the most underrepresented bucket this puzzle can fill
        target = min(avail, key=lambda t: bucket_counts[t])
        lo, hi = BUCKET_RANGE[target]
        # Among candidates in range, pick closest to target
        in_range = [c for c in candidates if lo <= c["metadata"].get("trajectory_length", 0) <= hi]
        best = min(in_range, key=lambda c: abs(c["metadata"].get("trajectory_length", 0) - target))
        rq1_selected.add(best["eval_id"])
        bucket_counts[target] += 1

    print(f"RQ1 sampling: {dict(bucket_counts)} across {len(rq1_selected)} puzzles")

    # For rq2/rq3 hard: pick 1 episode per puzzle, keep all aligned variants for that episode.
    # Episodes are identified by the __eN suffix in eval_id.
    rq2_episode: dict[str, str] = {}  # puzzle_name -> selected episode suffix (e.g. "e0")
    rq3_episode: dict[str, str] = {}

    for name in sorted(selected):
        items_path = src / name / "eval_items.json"
        if not items_path.exists():
            continue
        with open(items_path) as f:
            items = json.load(f)

        rq2_eps = set()
        rq3_eps = set()
        for item in items:
            eid = item["eval_id"]
            rq = item.get("rq", "")
            variant = item.get("variant", "")
            if rq == "rq2" and "hard" in variant and "ablation" not in variant:
                ep = eid.split("__")[-1]  # e.g. "e0"
                rq2_eps.add(ep)
            if rq == "rq3" and variant == "hard":
                ep = eid.split("__")[-1]
                rq3_eps.add(ep)

        # Randomly pick one episode each (independent for rq2/rq3)
        if rq2_eps:
            rq2_episode[name] = random.choice(sorted(rq2_eps))
        if rq3_eps:
            rq3_episode[name] = random.choice(sorted(rq3_eps))

    print(f"RQ2 hard: 1 episode per puzzle ({len(rq2_episode)} puzzles)")
    print(f"RQ3 hard: 1 episode per puzzle ({len(rq3_episode)} puzzles)")

    # Second pass: collect all items with filtering
    for name in sorted(selected):
        items_path = src / name / "eval_items.json"
        if not items_path.exists():
            continue
        with open(items_path) as f:
            items = json.load(f)
        for item in items:
            rq = item.get("rq", "")
            variant = item.get("variant", "")

            # For rq2 and rq3, keep only hard variants; drop rq3 comparison ablations
            if rq in ("rq2", "rq3") and "hard" not in variant:
                continue
            if rq == "rq3" and "comparison" in variant:
                continue
            if rq == "rq1" and "ablation" in variant:
                continue
            # For rq1_main, keep only the sampled item per puzzle
            if rq == "rq1" and variant == "main" and item["eval_id"] not in rq1_selected:
                continue
            # For rq2/rq3 hard, keep only the selected episode (and its ablations)
            if rq == "rq2" and "hard" in variant:
                ep = item["eval_id"].split("__")[-1]
                if name in rq2_episode and ep != rq2_episode[name]:
                    continue
            if rq == "rq3" and "hard" in variant:
                ep = item["eval_id"].split("__")[-1]
                if name in rq3_episode and ep != rq3_episode[name]:
                    continue

            key = f"{rq}_{variant}" if variant else rq

            # Rewrite image_path: "puzzle/images/file.png" -> "images/puzzle/file.png"
            img = item.get("image_path", "")
            if img:
                referenced_images.add(img)
                parts = Path(img).parts  # (puzzle_name, "images", filename)
                item["image_path"] = str(Path("images") / parts[0] / parts[-1])

            rq_buckets.setdefault(key, []).append(item)

    # Write one JSON file per rq_variant
    total_items = 0
    for key in sorted(rq_buckets):
        items = rq_buckets[key]
        total_items += len(items)
        with open(out / f"{key}.json", "w") as f:
            json.dump(items, f, indent=2)
            f.write("\n")

    # Copy only referenced images, organised as images/{puzzle_name}/
    copied_images = 0
    for img_rel in sorted(referenced_images):
        src_img = src / img_rel
        if not src_img.exists():
            print(f"  Warning: referenced image not found: {src_img}")
            continue
        # img_rel is like "puzzle_name/images/filename.png"
        parts = Path(img_rel).parts  # (puzzle_name, "images", filename)
        puzzle_name = parts[0]
        filename = parts[-1]
        dest_dir = images_dir / puzzle_name
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(src_img, dest_dir / filename)
        copied_images += 1

    # Write build_stats.json
    stats = {"total_puzzles": len(selected)}
    for key in sorted(rq_buckets):
        stats[key] = len(rq_buckets[key])
    stats["total_eval_items"] = total_items
    stats["total_images"] = copied_images

    with open(out / "build_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    print(f"\nDataset written to {out}/")
    print(f"  Puzzles: {len(selected)}")
    print(f"  Eval items: {total_items}")
    print(f"  RQ files: {len(rq_buckets)}")
    print(f"  Images copied: {copied_images}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter a puzzle dataset by difficulty threshold and verification status."
    )
    parser.add_argument("dataset", type=Path, help="Path to source dataset directory")
    parser.add_argument("verification", type=Path, help="Path to verification JSON file")
    parser.add_argument("--min-difficulty", type=int, default=None,
                        help="Keep puzzles with difficulty >= this value")
    parser.add_argument("--max-difficulty", type=int, default=None,
                        help="Keep puzzles with difficulty <= this value")
    parser.add_argument("--status", type=str, default=None,
                        help="Keep only puzzles with this status (e.g. 'approved', 'pending')")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory (default: <dataset>_filtered)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be kept without writing files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for episode sampling (default: 42)")

    args = parser.parse_args()
    random.seed(args.seed)

    if not args.dataset.is_dir():
        print(f"Error: {args.dataset} is not a directory")
        sys.exit(1)
    if not args.verification.is_file():
        print(f"Error: {args.verification} is not a file")
        sys.exit(1)
    if args.min_difficulty is None and args.max_difficulty is None and args.status is None:
        print("Error: specify at least one filter (--min-difficulty, --max-difficulty, or --status)")
        sys.exit(1)

    verification = load_verification(args.verification)
    kept = filter_puzzles(verification, args.min_difficulty, args.max_difficulty, args.status)

    if not kept:
        print("No puzzles match the given filters.")
        sys.exit(0)

    output = args.output or Path(f"{args.dataset}_filtered")
    build_filtered_dataset(args.dataset, output, kept, args.dry_run)


if __name__ == "__main__":
    main()
