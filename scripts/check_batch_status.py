#!/usr/bin/env python3
"""Check status of all submitted Gemini batch jobs.

Usage:
    python scripts/check_batch_status.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")
BATCH_JOBS_DIR = ROOT_DIR / "batch_jobs"


def get_client():
    """Create Google GenAI client."""
    import os
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set GOOGLE_API_KEY in .env")
    from google import genai
    return genai.Client(api_key=api_key)


def main():
    client = get_client()

    job_metas = sorted(BATCH_JOBS_DIR.rglob("job_meta.json"))
    if not job_metas:
        print("No batch jobs found.")
        return

    print(f"Found {len(job_metas)} batch job(s)\n")

    for meta_path in job_metas:
        with open(meta_path) as f:
            meta = json.load(f)

        job_name = meta["job_name"]
        try:
            job = client.batches.get(name=job_name)
            state = job.state.name
        except Exception as e:
            state = f"ERROR: {e}"

        print(f"  {meta['puzzle']:15s} {meta['rq']:35s} {meta['model']}")
        print(f"    Job:       {job_name}")
        print(f"    State:     {state}")
        print(f"    Submitted: {meta.get('submitted_at', 'unknown')}")
        print(f"    Items:     {meta.get('item_count', '?')}")
        print()


if __name__ == "__main__":
    main()
