#!/usr/bin/env python3
"""
Run model evaluation using the Gemini Batch API (file-based).

This script uploads images, builds a JSONL batch request file, submits it
via the Gemini batch API, polls for completion, and downloads results into
the same JSONL format as run_eval.py.

Usage:
    # Run all RQs for a puzzle type using the subset_50 dataset
    python scripts/run_eval_batch.py --puzzle mosaic_grid --model gemini-3.5-pro-preview

    # Run a single RQ
    python scripts/run_eval_batch.py --puzzle mosaic_grid --model gemini-3.5-pro-preview --rq rq1_main

    # Use full dataset instead of subset
    python scripts/run_eval_batch.py --puzzle mosaic_grid --model gemini-3.5-pro-preview --dataset dataset_mosaic_grid

    # Resume: skip RQs that already have complete output files
    python scripts/run_eval_batch.py --puzzle mosaic_grid --model gemini-3.5-pro-preview --resume

    # Submit jobs and exit without waiting (collect results later)
    python scripts/run_eval_batch.py --puzzle mosaic_grid --model gemini-3.5-pro-preview --submit-only

    # Collect results from previously submitted jobs
    python scripts/run_eval_batch.py --puzzle mosaic_grid --model gemini-3.5-pro-preview --collect-only

Outputs response JSONL files to responses/{puzzle}/ in the same format as run_eval.py.
Batch job metadata is saved to batch_jobs/{puzzle}/{model}/ for tracking.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

DATASET_DIR = ROOT_DIR / "dataset"
RESPONSES_DIR = ROOT_DIR / "responses"
BATCH_JOBS_DIR = ROOT_DIR / "batch_jobs"

ALL_RQS = [
    "rq1_main",
    "rq2_hard_ablation_deadlock",
    "rq2_hard_ablation_solvable",
    "rq2_hard_deadlock",
    "rq2_hard_solvable",
    "rq3_hard",
    "rq3_hard_ablation_final",
]

COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def get_client():
    """Create Google GenAI client."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set GOOGLE_API_KEY in .env")
    from google import genai
    return genai.Client(api_key=api_key)


def upload_images(client, items: list[dict], image_base: Path,
                  job_dir: Path) -> dict[str, str]:
    """Upload unique images and return mapping of local path -> file URI.

    Caches upload results in job_dir/image_uploads.json to avoid re-uploading.
    """
    from google.genai import types

    cache_path = job_dir / "image_uploads.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"  Loaded {len(cached)} cached image uploads")
    else:
        cached = {}

    # Collect unique image paths
    unique_images = set()
    for item in items:
        img_rel = item["image_path"]
        unique_images.add(img_rel)

    to_upload = [img for img in unique_images if img not in cached]
    print(f"  Images: {len(unique_images)} unique, {len(to_upload)} to upload")

    for i, img_rel in enumerate(sorted(to_upload)):
        img_path = image_base / img_rel
        if not img_path.exists():
            print(f"    WARNING: image not found: {img_path}")
            continue

        for attempt in range(3):
            try:
                uploaded = client.files.upload(
                    file=str(img_path),
                    config=types.UploadFileConfig(
                        display_name=img_rel.replace("/", "_"),
                        mime_type="image/png",
                    ),
                )
                cached[img_rel] = uploaded.name
                if (i + 1) % 10 == 0 or i == len(to_upload) - 1:
                    print(f"    Uploaded {i + 1}/{len(to_upload)}")
                break
            except Exception as e:
                if attempt < 2 and ("429" in str(e) or "rate" in str(e).lower()):
                    time.sleep(10 * (attempt + 1))
                    continue
                print(f"    ERROR uploading {img_rel}: {e}")
                break

        # Save progress periodically
        if (i + 1) % 20 == 0:
            with open(cache_path, "w") as f:
                json.dump(cached, f, indent=2)

    # Final save
    with open(cache_path, "w") as f:
        json.dump(cached, f, indent=2)

    return cached


def build_batch_jsonl(items: list[dict], image_uploads: dict[str, str],
                      output_path: Path,
                      show_thinking: bool = False, thinking_level: str = "medium",
                      max_output_tokens: int = None):
    """Build a JSONL file for the batch API.

    Each line is: {"key": eval_id, "request": GenerateContentRequest}
    """
    with open(output_path, "w") as f:
        for item in items:
            img_rel = item["image_path"]
            file_name = image_uploads.get(img_rel)
            if not file_name:
                print(f"    WARNING: no upload for {img_rel}, skipping {item['eval_id']}")
                continue

            gen_config = {
                "temperature": 0.0,
            }
            if max_output_tokens:
                gen_config["maxOutputTokens"] = max_output_tokens
            if show_thinking:
                gen_config["thinkingConfig"] = {
                    "thinkingLevel": thinking_level.upper(),
                }

            request_line = {
                "key": item["eval_id"],
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "file_data": {
                                        "file_uri": f"https://generativelanguage.googleapis.com/v1beta/{file_name}",
                                        "mime_type": "image/png",
                                    }
                                },
                                {"text": item["prompt"]},
                            ],
                        }
                    ],
                    "generation_config": gen_config,
                },
            }
            f.write(json.dumps(request_line) + "\n")

    count = sum(1 for _ in open(output_path))
    print(f"  Built batch JSONL: {count} requests -> {output_path}")
    return count


def submit_batch(client, model: str, jsonl_path: Path, display_name: str) -> str:
    """Upload JSONL and submit batch job. Returns batch job name."""
    from google.genai import types

    print(f"  Uploading batch file: {jsonl_path}")
    uploaded_file = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(
            display_name=display_name,
            mime_type="application/jsonl",
        ),
    )
    print(f"  Uploaded as: {uploaded_file.name}")

    print(f"  Submitting batch job...")
    batch_job = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={"display_name": display_name},
    )
    print(f"  Batch job created: {batch_job.name}")
    print(f"  State: {batch_job.state.name}")
    return batch_job.name


def poll_batch(client, job_name: str, poll_interval: int = 30) -> object:
    """Poll a batch job until it completes. Returns the final job object."""
    print(f"  Polling {job_name} every {poll_interval}s...")
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        if state in COMPLETED_STATES:
            print(f"  Final state: {state}")
            return job
        print(f"    State: {state} ... waiting {poll_interval}s")
        time.sleep(poll_interval)


def download_results(client, job, items: list[dict], model: str) -> list[dict]:
    """Download and parse batch results into our response format."""
    results = []

    # Build lookup from eval_id to item for metadata
    item_lookup = {item["eval_id"]: item for item in items}

    if job.dest and job.dest.file_name:
        # File-based results
        print(f"  Downloading results from {job.dest.file_name}...")
        raw = client.files.download(file=job.dest.file_name)
        content = raw.decode("utf-8") if isinstance(raw, bytes) else raw

        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            eval_id = entry.get("key", "")
            item = item_lookup.get(eval_id, {})
            response = entry.get("response", {})

            # Extract text from response
            text = ""
            reasoning = None
            input_tokens = 0
            output_tokens = 0
            thinking_tokens = 0
            error = None

            if "candidates" in response:
                parts = response["candidates"][0].get("content", {}).get("parts", [])
                text_parts = []
                thought_parts = []
                for p in parts:
                    if p.get("thought"):
                        thought_parts.append(p.get("text", ""))
                    elif p.get("text"):
                        text_parts.append(p["text"])
                text = "".join(text_parts)
                reasoning = "\n".join(thought_parts) if thought_parts else None
                usage = response.get("usageMetadata", {})
                input_tokens = usage.get("promptTokenCount", 0)
                output_tokens = usage.get("candidatesTokenCount", 0)
                thinking_tokens = usage.get("thoughtsTokenCount", 0)
            elif "error" in response:
                error = response["error"].get("message", str(response["error"]))

            results.append({
                "status": "ok" if not error else "error",
                "eval_id": eval_id,
                "puzzle_dir": item.get("puzzle_dir", ""),
                "rq": item.get("rq", ""),
                "variant": item.get("variant", ""),
                "method": "google",
                "model_name": f"google/{model}",
                "image_path": item.get("image_path", ""),
                "content": text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "reasoning": reasoning,
                **({"error": error} if error else {}),
            })

    elif job.dest and job.dest.inlined_responses:
        # Inline results
        print(f"  Processing {len(job.dest.inlined_responses)} inline results...")
        # Inline responses are ordered same as input
        for i, inline_resp in enumerate(job.dest.inlined_responses):
            item = items[i] if i < len(items) else {}
            eval_id = item.get("eval_id", f"unknown_{i}")

            if inline_resp.response:
                # Extract thinking and answer parts
                text_parts = []
                thought_parts = []
                if inline_resp.response.candidates:
                    for part in inline_resp.response.candidates[0].content.parts:
                        if hasattr(part, "thought") and part.thought:
                            thought_parts.append(part.text or "")
                        elif part.text:
                            text_parts.append(part.text)
                text = "".join(text_parts) if text_parts else (inline_resp.response.text or "")
                reasoning = "\n".join(thought_parts) if thought_parts else None
                usage = inline_resp.response.usage_metadata
                input_tokens = usage.prompt_token_count or 0 if usage else 0
                output_tokens = usage.candidates_token_count or 0 if usage else 0
                thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0 if usage else 0
                results.append({
                    "status": "ok",
                    "eval_id": eval_id,
                    "puzzle_dir": item.get("puzzle_dir", ""),
                    "rq": item.get("rq", ""),
                    "variant": item.get("variant", ""),
                    "method": "google",
                    "model_name": f"google/{model}",
                    "image_path": item.get("image_path", ""),
                    "content": text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "thinking_tokens": thinking_tokens,
                    "reasoning": reasoning,
                })
            elif inline_resp.error:
                results.append({
                    "status": "error",
                    "eval_id": eval_id,
                    "puzzle_dir": item.get("puzzle_dir", ""),
                    "rq": item.get("rq", ""),
                    "variant": item.get("variant", ""),
                    "method": "google",
                    "model_name": f"google/{model}",
                    "image_path": item.get("image_path", ""),
                    "content": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thinking_tokens": 0,
                    "reasoning": None,
                    "error": str(inline_resp.error),
                })
    else:
        print("  WARNING: No results found in batch job response")

    return results


def write_results(results: list[dict], out_path: Path):
    """Write results in same JSONL format as run_eval.py."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    ok = sum(1 for r in results if r["status"] == "ok")
    err = sum(1 for r in results if r["status"] == "error")
    total_input = sum(r.get("input_tokens", 0) for r in results)
    total_output = sum(r.get("output_tokens", 0) for r in results)
    total_thinking = sum(r.get("thinking_tokens", 0) for r in results)
    print(f"  Written {len(results)} results ({ok} ok, {err} errors) -> {out_path}")
    print(f"  Tokens — input: {total_input:,}  output: {total_output:,}  thinking: {total_thinking:,}")


def run_rq_batch(client, model: str, puzzle: str, rq: str,
                 dataset_name: str, job_dir: Path,
                 resume: bool, submit_only: bool, collect_only: bool,
                 poll_interval: int,
                 show_thinking: bool = False, thinking_level: str = "medium",
                 max_output_tokens: int = None):
    """Run a single RQ via batch API."""
    dataset_path = DATASET_DIR / dataset_name / f"{rq}.json"
    if not dataset_path.exists():
        print(f"  SKIP: {dataset_path} not found")
        return

    image_base = DATASET_DIR / dataset_name

    with open(dataset_path) as f:
        items = json.load(f)

    # Output path (same convention as run_eval.py)
    resp_dir = RESPONSES_DIR / puzzle
    resp_dir.mkdir(parents=True, exist_ok=True)
    out_path = resp_dir / f"{rq}__google__{model}.jsonl"

    # Resume check
    if resume and out_path.exists():
        existing_count = sum(1 for line in open(out_path) if line.strip())
        if existing_count >= len(items):
            print(f"  Already complete ({existing_count}/{len(items)} items) -> skipping")
            return

    # Job tracking file
    rq_job_dir = job_dir / rq
    rq_job_dir.mkdir(parents=True, exist_ok=True)
    job_meta_path = rq_job_dir / "job_meta.json"

    if collect_only:
        # Just collect results from a previous submission
        if not job_meta_path.exists():
            print(f"  No previous job found for {rq}, skipping")
            return
        with open(job_meta_path) as f:
            job_meta = json.load(f)
        job_name = job_meta["job_name"]
        print(f"  Collecting results for {job_name}...")
        job = poll_batch(client, job_name, poll_interval)
        if job.state.name == "JOB_STATE_SUCCEEDED":
            results = download_results(client, job, items, model)
            write_results(results, out_path)
        else:
            print(f"  Job {job_name} ended with state: {job.state.name}")
        return

    # Step 1: Upload images
    print("  Step 1: Upload images")
    image_uploads = upload_images(client, items, image_base, job_dir)

    # Step 2: Build JSONL
    print("  Step 2: Build batch JSONL")
    jsonl_path = rq_job_dir / f"{rq}_batch.jsonl"
    count = build_batch_jsonl(items, image_uploads, jsonl_path,
                              show_thinking=show_thinking,
                              thinking_level=thinking_level,
                              max_output_tokens=max_output_tokens)
    if count == 0:
        print("  No requests to submit, skipping")
        return

    # Step 3: Submit batch job
    print("  Step 3: Submit batch job")
    display_name = f"{puzzle}__{rq}__{model}"
    job_name = submit_batch(client, model, jsonl_path, display_name)

    # Save job metadata
    job_meta = {
        "job_name": job_name,
        "model": model,
        "puzzle": puzzle,
        "rq": rq,
        "dataset": dataset_name,
        "item_count": count,
        "show_thinking": show_thinking,
        "thinking_level": thinking_level if show_thinking else None,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(job_meta_path, "w") as f:
        json.dump(job_meta, f, indent=2)

    if submit_only:
        print(f"  Job submitted: {job_name}")
        print(f"  Run with --collect-only to retrieve results later")
        return

    # Step 4: Poll for completion
    print("  Step 4: Waiting for completion")
    job = poll_batch(client, job_name, poll_interval)

    if job.state.name == "JOB_STATE_SUCCEEDED":
        # Step 5: Download results
        print("  Step 5: Download results")
        results = download_results(client, job, items, model)
        write_results(results, out_path)
    else:
        print(f"  Job ended with state: {job.state.name}")
        if hasattr(job, "error") and job.error:
            print(f"  Error: {job.error}")


def main():
    parser = argparse.ArgumentParser(
        description="Run model evaluation using Gemini Batch API"
    )
    parser.add_argument("--puzzle", required=True,
                        help="Puzzle type: mosaic_grid, tide_grid, chain_vault")
    parser.add_argument("--model", required=True,
                        help="Model name, e.g. gemini-3.5-pro-preview")
    parser.add_argument("--rq", default=None,
                        help="Specific RQ to run (default: all)")
    parser.add_argument("--dataset", default=None,
                        help="Dataset directory name (default: subset_50/dataset_{puzzle})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip RQs that already have complete output files")
    parser.add_argument("--submit-only", action="store_true",
                        help="Submit batch jobs and exit without waiting for results")
    parser.add_argument("--collect-only", action="store_true",
                        help="Collect results from previously submitted batch jobs")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between status polls (default: 30)")
    parser.add_argument("--show-thinking", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Enable thinking/reasoning and capture it in output (default: on, use --no-show-thinking to disable)")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Max output tokens (default: API default)")
    parser.add_argument("--thinking-level", default="medium",
                        choices=["minimal", "low", "medium", "high"],
                        help="Thinking depth level (default: low). Only used with --show-thinking")

    args = parser.parse_args()
    client = get_client()

    # Default to subset_50 dataset
    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = f"subset_50/dataset_{args.puzzle}"

    # Verify dataset exists
    dataset_path = DATASET_DIR / dataset_name
    if not dataset_path.is_dir():
        sys.exit(f"ERROR: Dataset not found: {dataset_path}")

    rqs = [args.rq] if args.rq else ALL_RQS

    # Job tracking directory
    job_dir = BATCH_JOBS_DIR / args.puzzle / args.model
    job_dir.mkdir(parents=True, exist_ok=True)

    print(f"Puzzle:   {args.puzzle}")
    print(f"Model:    {args.model}")
    print(f"Dataset:  {dataset_name}")
    print(f"Thinking: {args.thinking_level if args.show_thinking else 'off'}")
    print(f"RQs:      {len(rqs)} to run")
    print(f"Jobs:     {job_dir}")
    print()

    for i, rq in enumerate(rqs, 1):
        print(f"[{i}/{len(rqs)}] {rq}")
        run_rq_batch(
            client, args.model, args.puzzle, rq,
            dataset_name, job_dir,
            args.resume, args.submit_only, args.collect_only,
            args.poll_interval,
            show_thinking=args.show_thinking,
            thinking_level=args.thinking_level,
            max_output_tokens=args.max_output_tokens,
        )
        print()

    print("All done.")


if __name__ == "__main__":
    main()
