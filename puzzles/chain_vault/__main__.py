"""CLI entry point: python -m puzzles.chain_vault [demo|generate|build|full]."""

from __future__ import annotations

import argparse
import os
import sys


def _parse_comma_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def cmd_demo(args: argparse.Namespace) -> None:
    from puzzles.chain_vault.demo import main as demo_main
    demo_main()


def cmd_generate(args: argparse.Namespace) -> None:
    from puzzles.chain_vault.dataset import generate_dataset
    from puzzles.chain_vault.generator import TEMPLATE_NAMES

    templates = _parse_comma_list(args.templates) if args.templates else list(TEMPLATE_NAMES)
    grid_sizes = _parse_int_list(args.grid_sizes) if args.grid_sizes else [5, 6, 7, 8]

    generate_dataset(
        templates=templates,
        grid_sizes=grid_sizes,
        seeds_per_config=args.seeds_per_config,
        max_retries=args.max_retries,
        workers=args.workers,
        output=args.output,
        resume=args.resume,
        seed_start=args.seed_start,
    )


def cmd_build(args: argparse.Namespace) -> None:
    from puzzles.chain_vault.build_eval import build_eval_dataset

    rq1_lengths = _parse_int_list(args.rq1_lengths) if args.rq1_lengths else [5, 10, 15]

    build_eval_dataset(
        input_path=args.input,
        output_dir=args.output,
        workers=args.workers,
        rq1_trajectory_lengths=rq1_lengths,
        overwrite=args.overwrite,
        puzzle_target=args.puzzle_target,
        sample_seed=args.sample_seed,
        require_hard=args.require_hard,
    )


def cmd_verify(args: argparse.Namespace) -> None:
    from verify.app import run_verify
    run_verify(args.dataset, port=args.port, debug=not args.no_debug)


def cmd_filter(args: argparse.Namespace) -> None:
    from verify.filter import filter_dataset
    filter_dataset(
        input_dir=args.input,
        output_dir=args.output,
        status=args.status,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
    )


def cmd_full(args: argparse.Namespace) -> None:
    from puzzles.chain_vault.dataset import generate_dataset, generate_to_target
    from puzzles.chain_vault.build_eval import build_eval_dataset
    from puzzles.chain_vault.generator import TEMPLATE_NAMES

    templates = _parse_comma_list(args.templates) if args.templates else list(TEMPLATE_NAMES)
    grid_sizes = _parse_int_list(args.grid_sizes) if args.grid_sizes else [5, 6, 7, 8]
    rq1_lengths = _parse_int_list(args.rq1_lengths) if args.rq1_lengths else [5, 10, 15]

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    raw_jsonl = os.path.join(output_dir, "raw.jsonl")

    puzzle_target = args.puzzle_target

    # --- Single pass (no target) ---
    if puzzle_target <= 0:
        print("=" * 60)
        print("Step 1/2: Generating puzzle instances")
        print("=" * 60)
        generate_dataset(
            templates=templates,
            grid_sizes=grid_sizes,
            seeds_per_config=args.seeds_per_config,
            max_retries=args.max_retries,
            workers=args.workers,
            output=raw_jsonl,
            resume=args.resume,
            seed_start=args.seed_start,
        )

        print()
        print("=" * 60)
        print("Step 2/2: Building evaluation dataset")
        print("=" * 60)
        build_eval_dataset(
            input_path=raw_jsonl,
            output_dir=output_dir,
            workers=args.workers,
            rq1_trajectory_lengths=rq1_lengths,
            overwrite=args.overwrite,
            puzzle_target=0,
            sample_seed=args.sample_seed,
            require_hard=args.require_hard,
        )
        return

    # --- Target-driven generation ---
    print("=" * 60)
    print(f"Target-driven generation: {puzzle_target} puzzles")
    print("=" * 60)

    generate_to_target(
        templates=templates,
        grid_sizes=grid_sizes,
        puzzle_target=puzzle_target,
        require_hard=args.require_hard,
        max_seeds=args.max_seeds,
        max_retries=args.max_retries,
        workers=args.workers,
        output=raw_jsonl,
        seed_start=args.seed_start,
        stall_seeds=args.stall_seeds,
        resume=args.resume,
    )

    print()
    print("=" * 60)
    print("Building evaluation dataset (rendering, prompts, eval index)")
    print("=" * 60)
    build_eval_dataset(
        input_path=raw_jsonl,
        output_dir=output_dir,
        workers=args.workers,
        rq1_trajectory_lengths=rq1_lengths,
        overwrite=True,
        puzzle_target=puzzle_target,
        sample_seed=args.sample_seed,
        require_hard=args.require_hard,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m puzzles.chain_vault",
        description="Chain Vault puzzle generator",
    )
    sub = parser.add_subparsers(dest="command")

    # demo
    sub.add_parser("demo", help="Run the demo (renders example puzzles)")

    # generate
    gen = sub.add_parser("generate", help="Generate a dataset of puzzle instances")
    gen.add_argument("--templates", type=str, default=None,
                     help="Comma-separated template names (default: all)")
    gen.add_argument("--grid-sizes", type=str, default=None,
                     help="Comma-separated grid sizes (default: 5,6,7,8)")
    gen.add_argument("--seeds-per-config", type=int, default=100,
                     help="Seeds per (template, size) config (default: 100)")
    gen.add_argument("--max-retries", type=int, default=400,
                     help="Max retries per seed (default: 400)")
    gen.add_argument("--workers", type=int, default=None,
                     help="Parallel workers (default: cpu_count)")
    gen.add_argument("--output", type=str, default="chain_vault_dataset.jsonl",
                     help="Output JSONL path (default: chain_vault_dataset.jsonl)")
    gen.add_argument("--resume", action="store_true",
                     help="Resume from existing output file")
    gen.add_argument("--seed-start", type=int, default=0,
                     help="Starting seed value (default: 0)")

    # build
    bld = sub.add_parser("build", help="Convert raw JSONL to evaluation dataset")
    bld.add_argument("--input", type=str, default="chain_vault_dataset.jsonl",
                     help="Input JSONL from generate step (default: chain_vault_dataset.jsonl)")
    bld.add_argument("--output", type=str, default="dataset",
                     help="Output directory root (default: dataset)")
    bld.add_argument("--workers", type=int, default=None,
                     help="Parallel workers (default: cpu_count)")
    bld.add_argument("--rq1-lengths", type=str, default=None,
                     help="Comma-separated RQ1 trajectory lengths (default: 5,10,15)")
    bld.add_argument("--overwrite", action="store_true",
                     help="Overwrite existing puzzle directories")
    bld.add_argument("--puzzle-target", type=int, default=0,
                     help="Max puzzles in final eval index (0 = no limit)")
    bld.add_argument("--sample-seed", type=int, default=42,
                     help="Random seed for reproducible sampling (default: 42)")
    bld.add_argument("--require-hard", action="store_true",
                     help="Drop puzzles missing hard (oracle-adjacent) RQ2/RQ3 variants")

    # full
    ful = sub.add_parser("full", help="End-to-end pipeline: generate then build")
    ful.add_argument("--templates", type=str, default=None,
                     help="Comma-separated template names (default: all)")
    ful.add_argument("--grid-sizes", type=str, default=None,
                     help="Comma-separated grid sizes (default: 5,6,7,8)")
    ful.add_argument("--seeds-per-config", type=int, default=100,
                     help="Seeds per (template, size) config for single-pass mode (default: 100)")
    ful.add_argument("--max-retries", type=int, default=400,
                     help="Max retries per seed (default: 400)")
    ful.add_argument("--max-seeds", type=int, default=2000,
                     help="Hard ceiling on seeds tried in target-driven mode (default: 2000)")
    ful.add_argument("--stall-seeds", type=int, default=100,
                     help="Bail if this many consecutive seeds yield no new accepted puzzle (default: 100)")
    ful.add_argument("--rq1-lengths", type=str, default=None,
                     help="Comma-separated RQ1 trajectory lengths (default: 5,10,15)")
    ful.add_argument("--workers", type=int, default=None,
                     help="Parallel workers (default: cpu_count)")
    ful.add_argument("--output", type=str, default="dataset",
                     help="Output directory root (default: dataset)")
    ful.add_argument("--seed-start", type=int, default=0,
                     help="Starting seed value (default: 0)")
    ful.add_argument("--overwrite", action="store_true",
                     help="Overwrite existing puzzle directories in build step")
    ful.add_argument("--puzzle-target", type=int, default=0,
                     help="Target puzzle count (0 = single pass, no target)")
    ful.add_argument("--sample-seed", type=int, default=42,
                     help="Random seed for reproducible sampling (default: 42)")
    ful.add_argument("--require-hard", action="store_true",
                     help="Require hard (oracle-adjacent) RQ2/RQ3 variants")
    ful.add_argument("--resume", action="store_true",
                     help="Resume interrupted generation from last checkpoint")

    # verify
    ver = sub.add_parser("verify", help="Launch human verification web app")
    ver.add_argument("--dataset", type=str, required=True,
                     help="Dataset directory to verify")
    ver.add_argument("--port", type=int, default=5050,
                     help="Port for dev server (default: 5050)")
    ver.add_argument("--no-debug", action="store_true",
                     help="Disable Flask debug mode")

    # filter
    flt = sub.add_parser("filter", help="Copy validated puzzles to clean output")
    flt.add_argument("--input", type=str, required=True,
                     help="Input dataset directory (with validation.json)")
    flt.add_argument("--output", type=str, required=True,
                     help="Output directory for filtered dataset")
    flt.add_argument("--status", type=str, default="validated",
                     help="Required validation status (default: validated)")
    flt.add_argument("--min-difficulty", type=int, default=None,
                     help="Minimum difficulty score (inclusive)")
    flt.add_argument("--max-difficulty", type=int, default=None,
                     help="Maximum difficulty score (inclusive)")

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "full":
        cmd_full(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "filter":
        cmd_filter(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
