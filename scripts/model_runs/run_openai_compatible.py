#!/usr/bin/env python3
"""
Run a dataset through an OpenAI-compatible multimodal endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

from openai_compatible import (
    DEFAULT_CACHE_DIR,
    OpenAICompatibleStreamingClient,
    append_jsonl,
    build_success_payload,
    default_output_path,
    load_completed_eval_ids,
    load_image_payload,
    load_records,
    resolve_base_url,
    resolve_image_path,
    sanitize_name,
    validate_record,
)

try:
    import diskcache
except ImportError:  # pragma: no cover
    diskcache = None

try:  # pragma: no cover
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a research dataset through an OpenAI-compatible endpoint using parallel requests."
    )
    parser.add_argument("--dataset-file", required=True, help="Path to a JSON or JSONL dataset file.")
    parser.add_argument("--model", required=True, help="Model name exposed by the endpoint.")
    parser.add_argument("--api-key", help="API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL.")
    parser.add_argument("--output", help="Optional output JSONL path.")
    parser.add_argument("--output-dir", help="Optional output directory for response JSONL files.")
    parser.add_argument("--limit", type=int, help="Optional cap on pending examples.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional generation token cap.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Defaults to 0.")
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty. Defaults to 0.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Maximum number of concurrent requests.")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Diskcache directory for persistent request caching.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="HTTP read timeout in seconds. Defaults to 300.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete any existing output before running.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately on the first failed example.")
    parser.add_argument("--verbose", action="store_true", help="Stream model output to stdout while requests run.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the planned run only.")
    return parser.parse_args()


def make_cache_key(
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    images_b64: list[str] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = 0.0,
    presence_penalty: float | None = 0.0,
) -> str:
    digest = hashlib.sha256()
    digest.update(base_url.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(model_name.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(prompt.encode("utf-8"))
    digest.update(json.dumps([max_tokens, temperature, presence_penalty]).encode("utf-8"))
    for image_b64 in images_b64 or []:
        digest.update(b"\x00")
        digest.update(image_b64.encode("utf-8"))
    return digest.hexdigest()


def prepare_item(record: dict[str, object], dataset_path: Path) -> dict[str, object]:
    prompt, raw_image_path = validate_record(record)
    image_path = resolve_image_path(raw_image_path, dataset_path)
    image_b64, mime_type = load_image_payload(image_path)
    return {
        "record": record,
        "prompt": prompt,
        "image_path": image_path,
        "image_b64": image_b64,
        "image_media_type": mime_type.split("/", 1)[1],
    }


def run_one(
    *,
    client: OpenAICompatibleStreamingClient,
    prepared_item: dict[str, object],
    dataset_path: Path,
    model_name: str,
    max_tokens: int | None,
    temperature: float | None,
    presence_penalty: float | None,
    verbose: bool,
) -> dict[str, object]:
    record = prepared_item["record"]
    prompt = prepared_item["prompt"]
    image_path = prepared_item["image_path"]
    image_b64 = prepared_item["image_b64"]
    image_media_type = prepared_item["image_media_type"]

    reasoning, content, input_tokens, output_tokens = client.generate_with_images(
        text=prompt,
        images=[image_b64],
        image_media_type=image_media_type,
        max_tokens=max_tokens,
        temperature=temperature,
        presence_penalty=presence_penalty,
        verbose=verbose,
    )

    return build_success_payload(
        record=record,
        model_name=model_name,
        dataset_path=dataset_path,
        image_path=image_path,
        reasoning=reasoning,
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_file).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    base_url = resolve_base_url(args.base_url)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(dataset_path, args.model, output_dir=output_dir)
    )

    if args.overwrite and output_path.exists() and not args.dry_run:
        output_path.unlink()

    records = load_records(dataset_path)
    completed_eval_ids = load_completed_eval_ids(
        output_path,
        method="openai_compatible",
        model_name=args.model,
    )
    if args.overwrite:
        completed_eval_ids = set()
    pending_records = [record for record in records if record.get("eval_id") not in completed_eval_ids]
    if args.limit is not None:
        pending_records = pending_records[: args.limit]

    prepared_items = [prepare_item(record, dataset_path) for record in pending_records]

    cache = None
    if diskcache is None:
        print("warning=diskcache_not_installed cache_mode=disabled")
    elif not args.dry_run:
        cache_dir = Path(args.cache_dir).expanduser().resolve() / sanitize_name(args.model)
        cache = diskcache.Cache(str(cache_dir))

    mem_cache: dict[str, tuple[str | None, str, int, int]] = {}
    cached_items: list[tuple[dict[str, object], tuple[str | None, str, int, int]]] = []
    items_to_run: list[tuple[str, dict[str, object]]] = []

    for prepared_item in prepared_items:
        key = make_cache_key(
            base_url=base_url,
            model_name=args.model,
            prompt=prepared_item["prompt"],
            images_b64=[prepared_item["image_b64"]],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            presence_penalty=args.presence_penalty,
        )
        if key in mem_cache:
            cached_items.append((prepared_item, mem_cache[key]))
            continue
        if cache is not None and key in cache:
            result = cache[key]
            mem_cache[key] = result
            cached_items.append((prepared_item, result))
            continue
        items_to_run.append((key, prepared_item))

    print(f"dataset_file={dataset_path}")
    print(f"base_url={base_url}")
    print(f"model={args.model}")
    print(f"output={output_path}")
    print(f"total_records={len(records)}")
    print(f"already_completed={len(completed_eval_ids)}")
    print(f"scheduled_now={len(pending_records)}")
    print(f"max_tokens={args.max_tokens if args.max_tokens is not None else 'provider_max'}")
    print(f"temperature={args.temperature}")
    print(f"presence_penalty={args.presence_penalty}")
    print(f"timeout_seconds={args.timeout_seconds}")
    print(f"cache_enabled={cache is not None}")
    print(f"cached_now={len(cached_items)}")
    print(f"to_run={len(items_to_run)}")
    if pending_records:
        print(f"workers={max(1, min(args.workers, len(items_to_run) or 1))}")

    if args.dry_run or not pending_records:
        if cache is not None:
            cache.close()
        return

    client = OpenAICompatibleStreamingClient(
        model_name=args.model,
        api_key=args.api_key,
        base_url=base_url,
        timeout_seconds=args.timeout_seconds,
    )
    max_workers = max(1, min(args.workers, len(items_to_run) or 1))

    success_count = 0
    error_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for prepared_item, cached_result in cached_items:
        result = build_success_payload(
            record=prepared_item["record"],
            model_name=args.model,
            dataset_path=dataset_path,
            image_path=prepared_item["image_path"],
            reasoning=cached_result[0],
            content=cached_result[1],
            input_tokens=cached_result[2],
            output_tokens=cached_result[3],
        )
        append_jsonl(output_path, result)
        success_count += 1
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        print(
            f"[cache] finished {result['eval_id']} "
            f"(input_tokens={result['input_tokens']}, output_tokens={result['output_tokens']})"
        )

    if not items_to_run:
        print(f"successful={success_count}")
        print(f"errors={error_count}")
        print(f"total_input_tokens={total_input_tokens}")
        print(f"total_output_tokens={total_output_tokens}")
        if cache is not None:
            cache.close()
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {}
        for start_index, (key, prepared_item) in enumerate(items_to_run, start=1):
            record = prepared_item["record"]
            eval_id = str(record.get("eval_id", f"record_{start_index}"))
            print(f"[start {start_index}/{len(items_to_run)}] running {eval_id}")
            future = executor.submit(
                run_one,
                client=client,
                prepared_item=prepared_item,
                dataset_path=dataset_path,
                model_name=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                presence_penalty=args.presence_penalty,
                verbose=args.verbose,
            )
            future_to_item[future] = (key, prepared_item)

        for completed_count, future in enumerate(as_completed(future_to_item), start=1):
            key, prepared_item = future_to_item[future]
            record = prepared_item["record"]
            eval_id = str(record.get("eval_id", f"record_{completed_count}"))
            try:
                result = future.result()
            except Exception as exc:
                error_count += 1
                print(f"[done {completed_count}/{len(items_to_run)}] failed {eval_id}: {exc}")
                if args.fail_fast:
                    for pending in future_to_item:
                        pending.cancel()
                    if cache is not None:
                        cache.close()
                    raise
                continue

            cache_value = (
                result.get("reasoning"),
                result["content"],
                result["input_tokens"],
                result["output_tokens"],
            )
            if cache is not None:
                cache[key] = cache_value
            mem_cache[key] = cache_value
            append_jsonl(output_path, result)
            success_count += 1
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            print(
                f"[done {completed_count}/{len(items_to_run)}] finished {eval_id} "
                f"(input_tokens={result['input_tokens']}, output_tokens={result['output_tokens']})"
            )

    print(f"successful={success_count}")
    print(f"errors={error_count}")
    print(f"total_input_tokens={total_input_tokens}")
    print(f"total_output_tokens={total_output_tokens}")
    if cache is not None:
        cache.close()
    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
