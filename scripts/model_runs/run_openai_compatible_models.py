#!/usr/bin/env python3
"""
Run one or more OpenAI-compatible models over a dataset file.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

from openai_compatible import (
    DEFAULT_CACHE_DIR,
    DEFAULT_RESPONSES_DIR,
    OpenAICompatibleStreamingClient,
    default_output_path,
    resolve_base_url,
)

try:  # pragma: no cover
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run run_openai_compatible.py for one or more models exposed by an OpenAI-compatible endpoint."
    )
    parser.add_argument("--dataset-file", help="Path to a JSON or JSONL dataset file.")
    parser.add_argument("--api-key", help="API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL.")
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help=(
            "Optional repeatable model name. "
            "If omitted, the script queries the endpoint's /models route."
        ),
    )
    parser.add_argument("--workers", type=int, default=8, help="Workers per model run.")
    parser.add_argument("--limit", type=int, help="Optional cap on pending examples.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional generation token cap.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="Presence penalty.")
    parser.add_argument("--timeout-seconds", type=float, default=300.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--output-dir",
        help="Optional output directory for JSONL responses. Defaults to responses/<dataset_parent>/.",
    )
    parser.add_argument(
        "--cache-root",
        help="Optional root directory for per-model cache directories.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for invoking run_openai_compatible.py.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-model outputs.")
    parser.add_argument("--fail-fast", action="store_true", help="Pass --fail-fast into each model run.")
    parser.add_argument(
        "--stop-on-model-error",
        action="store_true",
        help="Stop the sweep if any model process exits non-zero.",
    )
    parser.add_argument("--verbose", action="store_true", help="Pass --verbose into each model run.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them.")
    parser.add_argument("--list-models", action="store_true", help="Print available models from the endpoint and exit.")
    return parser.parse_args()


def resolve_models(
    requested_models: list[str] | None,
    *,
    api_key: str | None,
    base_url: str,
) -> list[str]:
    if requested_models:
        return requested_models
    client = OpenAICompatibleStreamingClient(
        model_name="placeholder",
        api_key=api_key,
        base_url=base_url,
    )
    return client.list_models()


def main() -> None:
    args = parse_args()
    base_url = resolve_base_url(args.base_url)

    if args.list_models:
        client = OpenAICompatibleStreamingClient(
            model_name="placeholder",
            api_key=args.api_key,
            base_url=base_url,
        )
        for model_name in client.list_models():
            print(model_name)
        return

    if not args.dataset_file:
        raise ValueError("--dataset-file is required unless you use --list-models.")

    dataset_path = Path(args.dataset_file).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else DEFAULT_RESPONSES_DIR / dataset_path.parent.name
    )
    cache_root = Path(args.cache_root).expanduser().resolve() if args.cache_root else DEFAULT_CACHE_DIR
    runner_path = Path(__file__).resolve().with_name("run_openai_compatible.py")
    models = resolve_models(args.models, api_key=args.api_key, base_url=base_url)

    print(f"dataset_file={dataset_path}")
    print(f"base_url={base_url}")
    print(f"runner={runner_path}")
    print(f"models={len(models)}")
    print(f"workers_per_model={args.workers}")
    print(f"max_tokens={args.max_tokens if args.max_tokens is not None else 'provider_max'}")
    print(f"temperature={args.temperature}")
    print(f"presence_penalty={args.presence_penalty}")
    print(f"timeout_seconds={args.timeout_seconds}")
    print(f"output_dir={output_dir}")
    print(f"cache_root={cache_root}")
    if args.limit is not None:
        print(f"limit={args.limit}")

    failures: list[tuple[str, int]] = []
    for index, model_name in enumerate(models, start=1):
        output_path = default_output_path(dataset_path, model_name, output_dir=output_dir)
        command = [
            args.python,
            str(runner_path),
            "--dataset-file",
            str(dataset_path),
            "--model",
            model_name,
            "--output",
            str(output_path),
            "--base-url",
            base_url,
            "--workers",
            str(args.workers),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--cache-dir",
            str(cache_root),
        ]

        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.api_key:
            command.extend(["--api-key", args.api_key])
        if args.max_tokens is not None:
            command.extend(["--max-tokens", str(args.max_tokens)])
        if args.temperature is not None:
            command.extend(["--temperature", str(args.temperature)])
        if args.presence_penalty is not None:
            command.extend(["--presence-penalty", str(args.presence_penalty)])
        if args.overwrite:
            command.append("--overwrite")
        if args.fail_fast:
            command.append("--fail-fast")
        if args.verbose:
            command.append("--verbose")
        if args.dry_run:
            command.append("--dry-run")

        print(f"[{index}/{len(models)}] model={model_name}")
        safe_command = list(command)
        if "--api-key" in safe_command:
            safe_command[safe_command.index("--api-key") + 1] = "<redacted>"
        print(f"command={shlex.join(safe_command)}")

        if args.dry_run:
            continue

        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failures.append((model_name, result.returncode))
            print(f"[{index}/{len(models)}] model_failed={model_name} exit_code={result.returncode}")
            if args.stop_on_model_error:
                break
        else:
            print(f"[{index}/{len(models)}] model_finished={model_name}")

    print(f"models_requested={len(models)}")
    print(f"models_failed={len(failures)}")
    if failures:
        for model_name, exit_code in failures:
            print(f"failure model={model_name} exit_code={exit_code}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
