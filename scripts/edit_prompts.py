#!/usr/bin/env python3
"""Interactively find-and-replace text in prompts across all RQ files in a filtered dataset.

The flow is always: Show prompt -> Define replacements -> Preview -> Apply.
Flags like --show and --replacements pre-fill steps but the full pipeline still runs.

Usage examples:
    # Full interactive flow: browse RQs, define replacements, preview, apply
    python -m scripts.edit_prompts filtered_dataset/

    # Start by showing a specific RQ, then enter replacements interactively
    python -m scripts.edit_prompts filtered_dataset/ --show rq1_main

    # Show a specific RQ, load replacements from file, preview, apply
    python -m scripts.edit_prompts filtered_dataset/ --show rq1_main --replacements replacements.json

    # Load replacements from file (skip show step), preview, apply
    python -m scripts.edit_prompts filtered_dataset/ --replacements replacements.json

    # Any combination with --dry-run to preview without writing
    python -m scripts.edit_prompts filtered_dataset/ --show rq2_main_deadlock --replacements replacements.json --dry-run

replacements.json format:
    [
        {"old": "Mosaic Grid", "new": "Tile Maze"},
        {"old": "pressure plate", "new": "switch"}
    ]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def find_rq_files(dataset: Path) -> dict[str, Path]:
    """Return {rq_name: path} for all RQ JSON files in the dataset."""
    rq_files = {}
    for f in sorted(dataset.glob("rq*.json")):
        rq_files[f.stem] = f
    return rq_files


def load_items(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def show_prompt(items: list[dict], index: int = 0):
    """Display a single prompt with line numbers for reference."""
    prompt = items[index]["prompt"]
    lines = prompt.split("\n")
    for i, line in enumerate(lines, 1):
        print(f"  {i:3d} | {line}")
    print(f"\n  [{len(items)} items in this RQ file, showing item {index}]")


def apply_replacements(prompt: str, replacements: list[dict]) -> str:
    for r in replacements:
        prompt = prompt.replace(r["old"], r["new"])
    return prompt


def preview_replacement(prompt: str, old_text: str, new_text: str):
    """Show lines affected by a replacement, with before/after."""
    lines = prompt.split("\n")
    for i, line in enumerate(lines, 1):
        if old_text in line:
            new_line = line.replace(old_text, new_text)
            print(f"  Line {i}:")
            print(f"    - {line}")
            print(f"    + {new_line}")


def apply_to_dataset(dataset: Path, rq_files: dict[str, Path], replacements: list[dict]):
    """Apply all replacements to all RQ files and write back."""
    total_modified = 0
    updated_files = []
    for name, path in rq_files.items():
        items = load_items(path)
        file_modified = False
        for item in items:
            original = item["prompt"]
            modified = apply_replacements(original, replacements)
            if modified != original:
                item["prompt"] = modified
                file_modified = True
                total_modified += 1
        if file_modified:
            with open(path, "w") as f:
                json.dump(items, f, indent=2)
                f.write("\n")
            updated_files.append(name)
            print(f"  Updated {name}.json")

    # Append to replacements.json for bookkeeping
    log_path = dataset / "replacements.json"
    history = []
    if log_path.exists():
        with open(log_path) as f:
            history = json.load(f)
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "replacements": replacements,
        "items_modified": total_modified,
        "files_updated": updated_files,
    })
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
        f.write("\n")

    print(f"\nDone. {total_modified} prompts modified across {len(rq_files)} RQ files.")
    print(f"Log saved to {log_path}")


# ── Step 1: Show ─────────────────────────────────────────────

def step_show(rq_files: dict[str, Path], show_rq: str | None):
    """Display prompts for inspection. Returns without blocking if nothing to show."""
    names = list(rq_files.keys())

    if show_rq:
        # --show was given, display that RQ
        if show_rq not in rq_files:
            print(f"Unknown RQ: {show_rq}")
            print(f"Available: {', '.join(names)}")
            sys.exit(1)
        items = load_items(rq_files[show_rq])
        print(f"--- Prompt from {show_rq} (item 0 of {len(items)}) ---\n")
        show_prompt(items)
        print()
        return

    # Interactive: let user pick
    print("Available RQ files:")
    for i, name in enumerate(names):
        items = load_items(rq_files[name])
        print(f"  [{i}] {name} ({len(items)} items)")

    print()
    choice = input("Enter RQ number or name to inspect (or Enter to skip): ").strip()
    if not choice:
        return

    if choice.isdigit() and int(choice) < len(names):
        chosen = names[int(choice)]
    elif choice in rq_files:
        chosen = choice
    else:
        print(f"Unknown selection: {choice}")
        sys.exit(1)

    items = load_items(rq_files[chosen])
    print(f"\n--- Prompt from {chosen} (item 0 of {len(items)}) ---\n")
    show_prompt(items)
    print()


# ── Step 2: Collect replacements ─────────────────────────────

def step_collect_replacements(replacements_path: Path | None) -> list[dict]:
    """Load from file and/or collect interactively. Returns combined list."""
    replacements = []

    # Load from file if provided
    if replacements_path:
        with open(replacements_path) as f:
            replacements = json.load(f)
        print(f"Loaded {len(replacements)} replacements from {replacements_path}:")
        for i, r in enumerate(replacements, 1):
            print(f"  {i}. \"{r['old']}\" -> \"{r['new']}\"")
        print()

    # Always offer to add more interactively
    print("Define additional replacements (empty 'old' text to finish):\n")
    while True:
        old = input("  Text to find: ")
        if not old:
            break
        new = input("  Replace with: ")
        replacements.append({"old": old, "new": new})
        print()

    return replacements


# ── Step 3: Preview ──────────────────────────────────────────

def step_preview(rq_files: dict[str, Path], replacements: list[dict]):
    """Show what will change across all RQ files. Returns total affected count."""
    print(f"\n{'='*60}")
    print(f"Replacements to apply ({len(replacements)}):")
    print(f"{'='*60}")
    for i, r in enumerate(replacements, 1):
        print(f"  {i}. \"{r['old']}\" -> \"{r['new']}\"")

    total_affected = 0
    print(f"\nMatches per RQ file:")
    for name, path in rq_files.items():
        items = load_items(path)
        affected = sum(
            1 for item in items
            if any(r["old"] in item["prompt"] for r in replacements)
        )
        total_affected += affected
        if affected:
            print(f"  {name}: {affected}/{len(items)} items affected")

    print(f"\nTotal items affected: {total_affected}")

    # Line-by-line diff from first matching file
    for name, path in rq_files.items():
        items = load_items(path)
        if any(r["old"] in items[0]["prompt"] for r in replacements):
            original = items[0]["prompt"]
            modified = apply_replacements(original, replacements)
            print(f"\n{'='*60}")
            print(f"Line diff (first item from {name}):")
            print(f"{'='*60}\n")
            for r in replacements:
                preview_replacement(original, r["old"], r["new"])
            print(f"\n--- Full modified prompt ---\n")
            for i, line in enumerate(modified.split("\n"), 1):
                print(f"  {i:3d} | {line}")
            break

    return total_affected


# ── Step 4: Apply ────────────────────────────────────────────

def step_apply(dataset: Path, rq_files: dict[str, Path], replacements: list[dict], dry_run: bool):
    """Confirm and write changes."""
    if dry_run:
        print("\n[Dry run] No files modified.")
        return

    print()
    confirm = input("Apply these replacements? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    apply_to_dataset(dataset, rq_files, replacements)


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find-and-replace text in prompts across all RQ files in a filtered dataset."
    )
    parser.add_argument("dataset", type=Path, help="Path to filtered dataset directory")
    parser.add_argument("--show", type=str, default=None,
                        help="Show prompt from this RQ before collecting replacements")
    parser.add_argument("--replacements", type=Path, default=None,
                        help="JSON file with replacements (can still add more interactively)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files")

    args = parser.parse_args()

    if not args.dataset.is_dir():
        print(f"Error: {args.dataset} is not a directory")
        sys.exit(1)

    rq_files = find_rq_files(args.dataset)
    if not rq_files:
        print(f"No rq*.json files found in {args.dataset}")
        sys.exit(1)

    # Step 1: Show
    step_show(rq_files, args.show)

    # Step 2: Collect replacements
    replacements = step_collect_replacements(args.replacements)
    if not replacements:
        print("No replacements defined. Exiting.")
        return

    # Step 3: Preview
    total = step_preview(rq_files, replacements)
    if total == 0:
        print("\nNo matches found. Nothing to do.")
        return

    # Step 4: Apply
    step_apply(args.dataset, rq_files, replacements, args.dry_run)


if __name__ == "__main__":
    main()
