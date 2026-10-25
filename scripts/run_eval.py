#!/usr/bin/env python3
"""
Run model evaluation on datasets using Google GenAI.

Usage:
    python scripts/run_eval.py --puzzle mosaic_grid --model gemma-3-27b-it
    python scripts/run_eval.py --puzzle mosaic_grid --model gemma-3-27b-it --rq rq1_main
    python scripts/run_eval.py --puzzle mosaic_grid --model gemma-3-27b-it --rq rq1_main --batch-size 5
    python scripts/run_eval.py --puzzle mosaic_grid --model gemma-3-27b-it --resume

Outputs response JSONL files to responses/{puzzle}/.
Caches all results in Redis (localhost:6379) so progress is never lost.
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import redis
from dotenv import load_dotenv
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

DATASET_DIR = ROOT_DIR / "dataset"
RESPONSES_DIR = ROOT_DIR / "responses"

ALL_RQS = [
    "rq1_main",
    "rq2_hard_ablation_deadlock",
    "rq2_hard_ablation_solvable",
    "rq2_hard_deadlock",
    "rq2_hard_solvable",
    "rq3_hard",
    "rq3_hard_ablation_final",
]


def get_redis():
    """Connect to Redis. Returns None if unavailable."""
    try:
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        return r
    except redis.ConnectionError:
        print("WARNING: Redis not available, running without cache")
        return None


def cache_key(model: str, puzzle_dir: str, prompt: str, image_path: Path) -> str:
    """Cache key based on model + puzzle name + image content + prompt."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    content_hash = hashlib.sha256(f"{image_b64}{prompt}".encode()).hexdigest()[:16]
    return f"eval:{model}:{puzzle_dir}:{content_hash}"


def get_client():
    """Create Google GenAI client."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set GOOGLE_API_KEY in .env")
    from google import genai
    return genai.Client(api_key=api_key)


def run_single(client, model: str, prompt: str, image_path: Path,
               max_retries: int = 3, retry_base_delay: float = 30.0):
    """Run a single inference call with retry on rate-limit errors.

    Returns (content, input_tokens, output_tokens).
    """
    from google.genai import types

    for attempt in range(max_retries + 1):
        try:
            # Upload image
            image_file = client.files.upload(file=str(image_path))

            response = client.models.generate_content(
                model=model,
                contents=[image_file, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )

            content = response.text or ""
            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count or 0 if usage else 0
            output_tokens = usage.candidates_token_count if usage else None

            # Fallback: count output tokens via count_tokens if not reported
            if output_tokens is None and content:
                ct = client.models.count_tokens(model=model, contents=content)
                output_tokens = ct.total_tokens
            output_tokens = output_tokens or 0

            return content, input_tokens, output_tokens

        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            is_retryable = (
                status in (429, 503, 500)
                or "429" in str(e)
                or "Resource has been exhausted" in str(e)
                or "rate" in str(e).lower()
            )
            if is_retryable and attempt < max_retries:
                delay = retry_base_delay * (2 ** attempt)
                print(f"    Rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                continue
            raise


def load_existing_results(out_path: Path) -> set:
    """Load already-completed eval_ids from an existing output file."""
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    done.add(data["eval_id"])
    return done


def run_rq(client, model: str, puzzle: str, rq: str, batch_size: int,
           resume: bool, limit: int = None, rcache: redis.Redis = None,
           max_retries: int = 3):
    """Run evaluation for a single RQ."""
    dataset_path = DATASET_DIR / f"dataset_{puzzle}" / f"{rq}.json"
    if not dataset_path.exists():
        print(f"  SKIP: {dataset_path} not found")
        return
    image_base = DATASET_DIR / f"dataset_{puzzle}"

    with open(dataset_path) as f:
        items = json.load(f)

    # Output setup
    resp_dir = RESPONSES_DIR / puzzle
    resp_dir.mkdir(parents=True, exist_ok=True)
    out_path = resp_dir / f"{rq}__google__{model}.jsonl"

    # Resume: combine file-based and cache-based done IDs
    done_ids = set()
    if resume:
        done_ids = load_existing_results(out_path)

    pending = [item for item in items if item["eval_id"] not in done_ids]
    if limit:
        pending = pending[:limit]
    if not pending:
        print(f"  All {len(items)} items already completed")
        return

    # Check cache for pending items
    cache_hits = {}
    if rcache:
        for item in pending:
            img_path = image_base / item["image_path"]
            cached = rcache.get(cache_key(model, item["puzzle_dir"], item["prompt"], img_path))
            if cached:
                cache_hits[item["eval_id"]] = json.loads(cached)
        if cache_hits:
            print(f"  Cache: {len(cache_hits)} hits, {len(pending) - len(cache_hits)} to run")

    to_run = [item for item in pending if item["eval_id"] not in cache_hits]

    # Process with tqdm
    failed = 0
    open_mode = "a" if resume and done_ids else "w"

    pbar = tqdm(total=len(pending), desc=f"  {rq}", unit="item")

    # Write cache hits first (counted as instant progress)
    all_results = {}
    for eval_id, result in cache_hits.items():
        all_results[eval_id] = result
        pbar.update(1)

    with open(out_path, open_mode) as out_f:
        # Process remaining in batches
        for batch_start in range(0, len(to_run), batch_size):
            batch = to_run[batch_start:batch_start + batch_size]

            def _process(item):
                eval_id = item["eval_id"]
                prompt = item["prompt"]
                image_path = image_base / item["image_path"]

                try:
                    content, input_tokens, output_tokens = run_single(
                        client, model, prompt, image_path, max_retries=max_retries)
                    return {
                        "status": "ok",
                        "eval_id": eval_id,
                        "puzzle_dir": item["puzzle_dir"],
                        "rq": item["rq"],
                        "variant": item["variant"],
                        "method": "google",
                        "model_name": f"google/{model}",
                        "image_path": item["image_path"],
                        "content": content,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "reasoning": None,
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "eval_id": eval_id,
                        "puzzle_dir": item["puzzle_dir"],
                        "rq": item["rq"],
                        "variant": item["variant"],
                        "method": "google",
                        "model_name": f"google/{model}",
                        "image_path": item["image_path"],
                        "content": "",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning": None,
                        "error": str(e),
                    }

            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                futures = {pool.submit(_process, item): item["eval_id"]
                           for item in batch}
                for future in as_completed(futures):
                    result = future.result()
                    all_results[result["eval_id"]] = result
                    # Cache the result
                    if rcache and result["status"] == "ok":
                        item_for_key = next(it for it in batch if it["eval_id"] == result["eval_id"])
                        img_path = image_base / item_for_key["image_path"]
                        rcache.set(cache_key(model, item_for_key["puzzle_dir"], item_for_key["prompt"], img_path),
                                   json.dumps(result))
                    if result["status"] == "error":
                        failed += 1
                        pbar.set_postfix(errors=failed)
                    pbar.update(1)

        # Write all results in original order
        for item in pending:
            result = all_results[item["eval_id"]]
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    pbar.close()
    print(f"  Done: {len(pending)} completed ({len(cache_hits)} cached), {failed} errors -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--puzzle", required=True,
                        help="Puzzle type: mosaic_grid, tide_grid, chain_vault")
    parser.add_argument("--model", required=True,
                        help="Model name, e.g. gemma-3-27b-it, gemma-3-8b-it")
    parser.add_argument("--rq", default=None,
                        help="Specific RQ to run (default: all)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Concurrent requests per batch (default: 5)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output, skipping completed items")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max items to run per RQ (for testing)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip Redis cache (force fresh inference)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Max retries on rate-limit/server errors (default: 3)")
    args = parser.parse_args()

    client = get_client()
    rcache = None if args.no_cache else get_redis()

    rqs = [args.rq] if args.rq else ALL_RQS

    print(f"Puzzle: {args.puzzle}")
    print(f"Model:  {args.model}")
    print(f"Cache:  {'Redis connected' if rcache else 'disabled'}")
    print(f"RQs:    {len(rqs)} to run\n")

    for i, rq in enumerate(rqs, 1):
        print(f"[{i}/{len(rqs)}] {rq}")
        run_rq(client, args.model, args.puzzle, rq, args.batch_size,
               args.resume, args.limit, rcache, max_retries=args.retries)
        print()


if __name__ == "__main__":
    main()
