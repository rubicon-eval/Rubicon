#!/usr/bin/env python3
"""
Run the judge for all response files in a puzzle's responses directory.

Usage:
    python scripts/judge_all.py --puzzle mosaic_grid --resp-dir responses/mosaic_grid

Scans resp-dir for all .jsonl files, extracts the RQ and model name from each,
and runs the appropriate judge.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# Import supported RQs from judge.py
sys.path.insert(0, str(Path(__file__).parent))
from judge import ALL_SUPPORTED


def get_model_short_name(jsonl_path: Path) -> str:
    """Read first line of JSONL and extract a short model name."""
    with open(jsonl_path) as f:
        first_line = f.readline().strip()
    if not first_line:
        return None
    data = json.loads(first_line)
    model_name = data.get("model_name", "")
    # e.g. "moonshotai/Kimi-K2.5" -> "kimi-k2.5"
    if "/" in model_name:
        short = model_name.split("/")[-1].lower()
    else:
        short = model_name.lower()
    return short


def parse_rq_from_filename(filename: str) -> str:
    """Extract RQ variant from filename like rq1_main__provider__model_name.jsonl."""
    # Everything before the first __ is the RQ variant
    parts = filename.split("__")
    if len(parts) >= 2:
        return parts[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run judge for all response files in a directory")
    parser.add_argument("--puzzle", required=True,
                        help="Puzzle type: mosaic_grid, tide_grid, chain_vault")
    parser.add_argument("--resp-dir", required=True,
                        help="Directory containing response JSONL files")
    parser.add_argument("--llm-fallback", action="store_true",
                        help="Use GPT-4.1 mini to extract answers from parse failures")
    args = parser.parse_args()

    resp_dir = Path(args.resp_dir)
    if not resp_dir.is_absolute():
        resp_dir = ROOT_DIR / resp_dir

    if not resp_dir.exists():
        sys.exit(f"ERROR: Response directory not found: {resp_dir}")

    # Find all JSONL files
    jsonl_files = sorted(resp_dir.glob("*.jsonl"))
    if not jsonl_files:
        sys.exit(f"ERROR: No .jsonl files found in {resp_dir}")

    print(f"Puzzle: {args.puzzle}")
    print(f"Response dir: {resp_dir}")
    print(f"Found {len(jsonl_files)} response file(s)\n")

    # Group by (rq, model)
    jobs = []
    for f in jsonl_files:
        rq = parse_rq_from_filename(f.name)
        if rq is None:
            print(f"  SKIP: can't parse RQ from {f.name}")
            continue
        if rq not in ALL_SUPPORTED:
            print(f"  SKIP: unsupported RQ '{rq}' in {f.name}")
            continue
        model = get_model_short_name(f)
        if not model:
            print(f"  SKIP: can't read model name from {f.name}")
            continue
        jobs.append((rq, model, f))

    if not jobs:
        sys.exit("ERROR: No valid response files to judge")

    print(f"Will judge {len(jobs)} file(s):\n")
    for rq, model, f in jobs:
        print(f"  {rq:40s} {model}")
    print(flush=True)

    # Run judge for each
    judge_script = Path(__file__).parent / "judge.py"
    python = sys.executable
    results_summary = []

    for rq, model, f in jobs:
        print(f"{'='*60}")
        print(f"Judging: {rq} / {model}")
        print(f"{'='*60}", flush=True)
        cmd = [
            python, str(judge_script),
            "--puzzle", args.puzzle,
            "--rq", rq,
            "--model", model,
            "--resp-dir", str(resp_dir),
        ]
        if args.llm_fallback:
            cmd.append("--llm-fallback")
        result = subprocess.run(cmd, capture_output=False)
        results_summary.append({
            "rq": rq,
            "model": model,
            "file": f.name,
            "returncode": result.returncode,
        })
        print()

    # Final summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for r in results_summary:
        status = "OK" if r["returncode"] == 0 else f"FAILED (exit {r['returncode']})"
        print(f"  {r['rq']:40s} {r['model']:20s} {status}")


if __name__ == "__main__":
    main()
