#!/usr/bin/env python3
"""
Judge for RQ1/RQ2/RQ3: evaluate model responses against ground truth.

Usage:
    python scripts/judge.py --puzzle mosaic_grid --rq rq1_main --model kimi-k2.5 --resp-dir responses/mosaic_grid
    python scripts/judge.py --puzzle mosaic_grid --rq rq2_hard_deadlock --model kimi-k2.5 --resp-dir responses/mosaic_grid

For rq2_hard_deadlock and rq2_hard_solvable, an LLM judge (gpt-4.1-mini) is used.
Requires `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_BASE_URL` env vars for those variants.

Files:
  - Dataset:   dataset/dataset_{puzzle}/{rq}.json
  - Responses: {resp-dir}/{rq}__*{model}*.jsonl   (fuzzy match on model short name)
  - Results:   results/{puzzle}/{model}/{rq}.jsonl
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")
RESULTS_DIR = ROOT_DIR / "results"


# ── Position extraction (RQ1) ────────────────────────────────────────────────

def extract_position(content: str):
    """Extract (row, col) from model response. Requires POSITION: (row, col) tag."""
    m = re.search(r"POSITION:\s*\((\d+)\s*,\s*(\d+)\)", content)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    return None


# ── Solvability extraction (RQ2 ablation) ────────────────────────────────────

def extract_solvable(content: str):
    """Extract YES/NO from model response. Requires SOLVABLE: <YES/NO> tag."""
    m = re.search(r"SOLVABLE:\s*(YES|NO)", content, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


# ── Fatal step extraction (RQ3) ──────────────────────────────────────────────

def extract_fatal_step(content: str):
    """Extract step number from model response. Requires FATAL STEP: <number> tag."""
    m = re.search(r"FATAL\s+STEP:\s*(\d+)", content, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


# ── Format issue checks ─────────────────────────────────────────────────────

def check_format_issues_rq1(content: str) -> list[str]:
    """Check for format issues in RQ1 response."""
    issues = []
    if not content or not content.strip():
        issues.append("empty_response")
        return issues
    if not re.search(r"POSITION:\s*\((\d+)\s*,\s*(\d+)\)", content):
        if re.findall(r"\((\d+)\s*,\s*(\d+)\)", content):
            issues.append("missing_POSITION_tag_but_has_coordinates")
        else:
            issues.append("no_coordinates_found")
    return issues


def check_format_issues_rq2_ablation(content: str) -> list[str]:
    """Check for format issues in RQ2 ablation response."""
    issues = []
    if not content or not content.strip():
        issues.append("empty_response")
        return issues
    if not re.search(r"SOLVABLE:\s*(YES|NO)", content, re.IGNORECASE):
        if re.search(r"\b(yes|no)\b", content, re.IGNORECASE):
            issues.append("missing_SOLVABLE_tag_but_has_yes_no")
        else:
            issues.append("no_solvable_answer_found")
    return issues


def check_format_issues_rq3(content: str) -> list[str]:
    """Check for format issues in RQ3 response."""
    issues = []
    if not content or not content.strip():
        issues.append("empty_response")
        return issues
    if not re.search(r"FATAL\s+STEP:\s*(\d+)", content, re.IGNORECASE):
        if re.search(r"\b(step|fatal)\b", content, re.IGNORECASE):
            issues.append("missing_FATAL_STEP_tag_but_mentions_step")
        else:
            issues.append("no_fatal_step_found")
    return issues


# ── Repetition loop detection ────────────────────────────────────────────────


# ── LLM extraction fallback (for parse failures) ────────────────────────────

LLM_EXTRACT_POSITION_PROMPT = """\
Extract the agent's final predicted position from this model response.
Return ONLY a coordinate in the format: (row, col)
If no position can be determined, return: NONE

Response:
---
{content}
---"""

LLM_EXTRACT_SOLVABLE_PROMPT = """\
Extract the model's final answer about whether the puzzle is solvable.
Return ONLY one of: YES, NO, or NONE (if no answer can be determined)

Response:
---
{content}
---"""

LLM_EXTRACT_FATAL_STEP_PROMPT = """\
Extract the step number the model identifies as the fatal/critical step.
Return ONLY a number, or NONE if no step number can be determined.

Response:
---
{content}
---"""


def llm_extract(client, prompt: str, content: str) -> str:
    """Call GPT-4.1 mini to extract an answer from a model response."""
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a precise extraction tool. Extract exactly what is asked. Do not reason or explain."},
            {"role": "user", "content": prompt.format(content=content)},
        ],
        max_completion_tokens=30,
        temperature=0.0,
        model=LLM_MODEL,
    )
    return response.choices[0].message.content.strip()


def llm_extract_position(client, content: str):
    """Use LLM to extract position from a response that failed regex parsing."""
    raw = llm_extract(client, LLM_EXTRACT_POSITION_PROMPT, content)
    m = re.search(r"\((\d+)\s*,\s*(\d+)\)", raw)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    return None


def llm_extract_solvable(client, content: str):
    """Use LLM to extract YES/NO from a response that failed regex parsing."""
    raw = llm_extract(client, LLM_EXTRACT_SOLVABLE_PROMPT, content).upper()
    if "YES" in raw:
        return "YES"
    if "NO" in raw:
        return "NO"
    return None


def llm_extract_fatal_step(client, content: str):
    """Use LLM to extract fatal step number from a response that failed regex parsing."""
    raw = llm_extract(client, LLM_EXTRACT_FATAL_STEP_PROMPT, content)
    m = re.search(r"(\d+)", raw)
    if m:
        return int(m.group(1))
    return None


# ── Judging logic per RQ ─────────────────────────────────────────────────────

def judge_rq1(gt, content, llm_client=None):
    """Judge RQ1: compare predicted position to ground_truth.agent_pos."""
    gt_pos = gt["agent_pos"]
    predicted = extract_position(content)
    format_issues = check_format_issues_rq1(content)
    used_llm_fallback = False
    if predicted is None and llm_client and content and content.strip():
        predicted = llm_extract_position(llm_client, content)
        if predicted is not None:
            used_llm_fallback = True
    if predicted is None:
        return {"predicted": None, "ground_truth": gt_pos,
                "correct": False, "parse_fail": True,
                "format_issues": format_issues,
                "llm_fallback": used_llm_fallback}
    return {
        "predicted": predicted,
        "ground_truth": gt_pos,
        "correct": predicted == list(gt_pos),
        "parse_fail": False,
        "format_issues": format_issues,
        "llm_fallback": used_llm_fallback,
    }


def judge_rq2_ablation(gt, content, llm_client=None):
    """Judge RQ2 ablation: compare predicted solvability to ground_truth.answer."""
    gt_answer = gt["answer"].upper()
    predicted = extract_solvable(content)
    format_issues = check_format_issues_rq2_ablation(content)
    used_llm_fallback = False
    if predicted is None and llm_client and content and content.strip():
        predicted = llm_extract_solvable(llm_client, content)
        if predicted is not None:
            used_llm_fallback = True
    if predicted is None:
        return {"predicted": None, "ground_truth": gt_answer,
                "correct": False, "parse_fail": True,
                "format_issues": format_issues,
                "llm_fallback": used_llm_fallback}
    return {
        "predicted": predicted,
        "ground_truth": gt_answer,
        "correct": predicted == gt_answer,
        "parse_fail": False,
        "format_issues": format_issues,
        "llm_fallback": used_llm_fallback,
    }


def judge_rq3(gt, content, llm_client=None):
    """Judge RQ3: compare predicted fatal step to ground_truth.fatal_step_1indexed."""
    gt_step = gt["fatal_step_1indexed"]
    predicted = extract_fatal_step(content)
    format_issues = check_format_issues_rq3(content)
    used_llm_fallback = False
    if predicted is None and llm_client and content and content.strip():
        predicted = llm_extract_fatal_step(llm_client, content)
        if predicted is not None:
            used_llm_fallback = True
    if predicted is None:
        return {"predicted": None, "ground_truth": gt_step,
                "correct": False, "parse_fail": True,
                "format_issues": format_issues,
                "llm_fallback": used_llm_fallback}
    return {
        "predicted": predicted,
        "ground_truth": gt_step,
        "correct": predicted == gt_step,
        "parse_fail": False,
        "format_issues": format_issues,
        "llm_fallback": used_llm_fallback,
    }


# ── LLM Judge for RQ2 hard (non-ablation) ───────────────────────────────────

LLM_MODEL = os.environ.get("RUBICON_JUDGE_MODEL", "gpt-4.1-mini")

_LLM_JUDGE_PROMPT_BASE = """\
You are a judge evaluating a model's response to a grid puzzle question.

The model was asked: "What is the best next move to win?" after a sequence of actions.

Here is the model's response:
---
{content}
---

IMPORTANT: Focus on the model's FINAL answer/conclusion, not intermediate reasoning. The model \
may discuss multiple possibilities, express uncertainty, or note that some moves are blocked \
along the way — what matters is ONLY the final action or conclusion it commits to.

Classify the model's final answer into one of these categories:
1. STEP:<direction> — the model recommends a specific directional move (UP, DOWN, LEFT, or RIGHT)
{wait_section}
{unsolvable_number}. UNSOLVABLE — the model's final conclusion is that the puzzle CANNOT be solved, is in a \
   permanent deadlock, or is impossible to win. The model must explicitly state that there is \
   NO valid move at all and NO path to victory — not just that some directions are blocked.

{distinctions_section}

Return ONLY one of: {valid_options}."""

# ── tide_grid: WAIT is a valid action ────────────────────────────────────────
_TIDE_GRID_WAIT_SECTION = """\
2. STEP:WAIT — the model recommends WAIT as its next action. This is a concrete action choice, \
   NOT the same as declaring the puzzle unsolvable. A model that says "the best move is WAIT" or \
   "the only option is to WAIT" is choosing an action, not giving up."""

_TIDE_GRID_DISTINCTIONS = """\
CRITICAL DISTINCTIONS:
- "All directions are blocked, so I must WAIT" → STEP:WAIT (choosing an action)
- "The only possible move is WAIT" → STEP:WAIT (choosing an action)
- "There is no valid move, the puzzle is unsolvable" → UNSOLVABLE (giving up)
- "The agent is permanently trapped with no way to win" → UNSOLVABLE (giving up)
- "It seems impossible but the best move is DOWN" → STEP:DOWN (choosing an action)"""

# ── chain_vault / mosaic_grid: WAIT is NOT a valid action ────────────────────
_NO_WAIT_SECTION = """\
   NOTE: WAIT is NOT a valid action in this puzzle. The only valid moves are UP, DOWN, LEFT, \
   and RIGHT. Do NOT classify any response as STEP:WAIT."""

_NO_WAIT_DISTINCTIONS = """\
CRITICAL DISTINCTIONS:
- "There is no valid move, the puzzle is unsolvable" → UNSOLVABLE (giving up)
- "The agent is permanently trapped with no way to win" → UNSOLVABLE (giving up)
- "It seems impossible but the best move is DOWN" → STEP:DOWN (choosing an action)

IMPORTANT — DOOR/KEY ACTIONS COUNT AS DIRECTIONAL MOVES:
In some puzzles, "opening a door" or "using a key on a door" is performed by moving in a direction \
(UP/DOWN/LEFT/RIGHT) toward the door. This IS a directional move, NOT a WAIT. If the model says \
"use the key on the door" or "open the door", determine which direction the door is relative to \
the agent and classify as that direction. If the model describes a two-step plan (e.g., "open the \
door, then move RIGHT"), the answer is the FIRST action — the directional move toward the door.

Examples:
- "Use the yellow key on the door at (2,3). After that, move RIGHT to (2,4)." → The door is to \
  the RIGHT of the agent, so the next move is RIGHT → STEP:RIGHT
- "The best next move is to open the locked door by moving LEFT." → STEP:LEFT
- "First open the door (move UP), then proceed through it." → STEP:UP
- "The immediate action is to use the keys on the door. After the door opens, move RIGHT." → The \
  action of using keys on the door IS a directional move toward the door → determine the direction \
  from context and return STEP:<direction>. If the direction toward the door cannot be determined \
  but the model also recommends a subsequent directional move, use that direction."""


def get_llm_judge_prompt(puzzle: str) -> str:
    """Return the appropriate LLM judge prompt based on puzzle type."""
    if puzzle == "tide_grid":
        return _LLM_JUDGE_PROMPT_BASE.format(
            content="{content}",
            wait_section=_TIDE_GRID_WAIT_SECTION,
            unsolvable_number="3",
            distinctions_section=_TIDE_GRID_DISTINCTIONS,
            valid_options="STEP:UP, STEP:DOWN, STEP:LEFT, STEP:RIGHT, STEP:WAIT, UNSOLVABLE, or UNCLEAR",
        )
    else:
        return _LLM_JUDGE_PROMPT_BASE.format(
            content="{content}",
            wait_section=_NO_WAIT_SECTION,
            unsolvable_number="2",
            distinctions_section=_NO_WAIT_DISTINCTIONS,
            valid_options="STEP:UP, STEP:DOWN, STEP:LEFT, STEP:RIGHT, UNSOLVABLE, or UNCLEAR",
        )


def get_llm_client():
    """Create OpenAI client for the external LLM judge."""
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    base_url = os.environ.get("AZURE_OPENAI_BASE_URL")
    if not api_key:
        sys.exit("ERROR: Set AZURE_OPENAI_API_KEY in .env "
                 "(required for rq2_hard_deadlock / rq2_hard_solvable)")
    if not base_url:
        sys.exit("ERROR: Set AZURE_OPENAI_BASE_URL in .env "
                 "(required for rq2_hard_deadlock / rq2_hard_solvable)")
    from openai import OpenAI
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def call_llm_judge(client, content: str, puzzle: str) -> str:
    """Call Azure OpenAI to classify the model response."""
    prompt = get_llm_judge_prompt(puzzle)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a precise classification judge."},
            {"role": "user", "content": prompt.format(content=content)},
        ],
        max_completion_tokens=50,
        temperature=0.0,
        model=LLM_MODEL,
    )
    return response.choices[0].message.content.strip()


def parse_llm_verdict(verdict: str, puzzle: str):
    """Parse the LLM judge's verdict into a standardized prediction."""
    verdict_upper = verdict.upper().strip()
    if verdict_upper == "UNSOLVABLE":
        return "UNSOLVABLE"
    m = re.match(r"STEP:\s*(UP|DOWN|LEFT|RIGHT|WAIT)", verdict_upper)
    if m:
        direction = m.group(1)
        # WAIT is only valid for tide_grid
        if direction == "WAIT" and puzzle != "tide_grid":
            return "UNCLEAR"
        return direction
    return "UNCLEAR"


def judge_rq2_llm(gt, content, llm_client, puzzle: str):
    """Judge RQ2 hard (non-ablation): use LLM to classify response."""
    gt_answer = gt["answer"].upper()

    try:
        raw_verdict = call_llm_judge(llm_client, content, puzzle)
    except Exception as e:
        print(f"    LLM judge error: {e}")
        raw_verdict = "UNCLEAR"
        time.sleep(1)

    predicted = parse_llm_verdict(raw_verdict, puzzle)

    is_correct = predicted == gt_answer if predicted not in ("UNCLEAR",) else False

    format_issues = []
    if predicted == "UNCLEAR":
        format_issues.append("llm_unclear")

    return {
        "predicted": predicted,
        "ground_truth": gt_answer,
        "correct": is_correct,
        "parse_fail": predicted == "UNCLEAR",
        "format_issues": format_issues,
        "llm_raw_verdict": raw_verdict,
    }


# ── RQ variant -> judge mapping ─────────────────────────────────────────────

# Variants that use simple regex-based judging
REGEX_JUDGES = {
    "rq1_main": judge_rq1,
    "rq2_hard_ablation_deadlock": judge_rq2_ablation,
    "rq2_hard_ablation_solvable": judge_rq2_ablation,
    "rq3_hard": judge_rq3,
    "rq3_hard_ablation_final": judge_rq3,
}

# Variants that use LLM-as-a-judge
LLM_JUDGES = {
    "rq2_hard_deadlock",
    "rq2_hard_solvable",
}

ALL_SUPPORTED = set(REGEX_JUDGES.keys()) | LLM_JUDGES


# ── File resolution ──────────────────────────────────────────────────────────

def find_response_file(resp_dir: Path, rq: str, model_short: str) -> Path:
    """Find the response JSONL matching the model short name (case-insensitive)."""
    pattern = str(resp_dir / f"{rq}__*.jsonl")
    candidates = glob.glob(pattern)
    model_lower = model_short.lower()
    matches = [c for c in candidates if model_lower in Path(c).name.lower()]
    if len(matches) == 1:
        return Path(matches[0])
    if len(matches) == 0:
        sys.exit(f"ERROR: No response file matching model '{model_short}' "
                 f"in {pattern}")
    sys.exit(f"ERROR: Multiple response files match '{model_short}':\n"
             + "\n".join(f"  {m}" for m in matches))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Judge model responses")
    parser.add_argument("--puzzle", required=True,
                        help="Puzzle type: mosaic_grid, tide_grid, chain_vault")
    parser.add_argument("--rq", required=True,
                        help="RQ with variant, e.g. rq1_main, rq2_hard_deadlock")
    parser.add_argument("--model", required=True,
                        help="Short model name for fuzzy file matching, e.g. kimi-k2.5")
    parser.add_argument("--resp-dir", required=True,
                        help="Directory containing response JSONL files")
    parser.add_argument("--response-file", help="Exact response JSONL file; bypasses fuzzy filename matching")
    parser.add_argument("--llm-fallback", action="store_true",
                        help="Use GPT-4.1 mini to extract answers from parse failures")
    args = parser.parse_args()

    if args.rq not in ALL_SUPPORTED:
        sys.exit(f"ERROR: Unknown RQ '{args.rq}'. Supported: {sorted(ALL_SUPPORTED)}")

    uses_llm = args.rq in LLM_JUDGES
    llm_client = None
    if uses_llm or args.llm_fallback:
        llm_client = get_llm_client()

    # Load dataset
    dataset_path = ROOT_DIR / "dataset" / f"dataset_{args.puzzle}" / f"{args.rq}.json"
    if not dataset_path.exists():
        sys.exit(f"ERROR: Dataset not found: {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)
    gt_map = {item["eval_id"]: item["ground_truth"] for item in dataset}

    # Load responses
    resp_dir = Path(args.resp_dir)
    if not resp_dir.is_absolute():
        resp_dir = ROOT_DIR / resp_dir
    resp_path = Path(args.response_file).resolve() if args.response_file else find_response_file(resp_dir, args.rq, args.model)
    print(f"Dataset:   {dataset_path}")
    print(f"Responses: {resp_path}")
    if uses_llm:
        print(f"LLM Judge: {LLM_MODEL}")

    responses_raw = []
    with open(resp_path) as f:
        for line in f:
            line = line.strip()
            if line:
                responses_raw.append(json.loads(line))

    # Deduplicate: keep first response per eval_id
    seen_eval_ids = set()
    responses = []
    duplicate_count = 0
    for resp in responses_raw:
        eid = resp["eval_id"]
        if eid in seen_eval_ids:
            duplicate_count += 1
            continue
        seen_eval_ids.add(eid)
        responses.append(resp)

    if duplicate_count:
        print(f"  WARNING: Found {duplicate_count} duplicate eval_id(s) — kept first occurrence of each")

    # Filter to responses with ground truth
    judge_items = []
    total_output_tokens = 0
    for resp in responses:
        eval_id = resp["eval_id"]
        content = resp.get("content", "")
        output_tokens = resp.get("output_tokens", 0)
        total_output_tokens += output_tokens
        gt = gt_map.get(eval_id)
        if gt is None:
            print(f"  WARNING: no ground truth for {eval_id}, skipping")
            continue
        judge_items.append((eval_id, gt, content, output_tokens))

    # Detect responses that hit the token limit (model never finished reasoning)
    # Only applies to qwen3-vl-8b-instruct which has hard token caps that vary
    # by puzzle and configuration (20k or 32768)
    QWEN3_VL_8B_TOKEN_LIMITS = {20000, 32768}
    token_limit_ids = set()
    is_qwen3_vl_8b = "qwen3-vl-8b" in args.model.lower()
    if is_qwen3_vl_8b:
        for eval_id, gt, content, output_tokens in judge_items:
            if output_tokens in QWEN3_VL_8B_TOKEN_LIMITS:
                token_limit_ids.add(eval_id)
        if token_limit_ids:
            print(f"  Token limit cutoffs ({args.model}): {len(token_limit_ids)} (will skip judging)")

    # Judge each response
    results = []
    correct = 0
    total = len(judge_items)
    parse_failures = 0

    # Helper to build a token-limit result entry
    def _token_limit_result(eval_id, gt, output_tokens):
        gt_answer = gt["answer"].upper() if isinstance(gt, dict) and "answer" in gt else gt
        return {
            "eval_id": eval_id,
            "ground_truth": gt_answer,
            "predicted": None,
            "correct": False,
            "output_tokens": output_tokens,
            "format_issues": ["token_limit_cutoff"],
            "llm_raw_verdict": None,
        }

    if uses_llm:
        # Concurrent LLM judging with thread pool
        LLM_WORKERS = 10

        def _llm_task(item):
            eval_id, gt, content, output_tokens = item
            verdict = judge_rq2_llm(gt, content, llm_client,
                                    puzzle=args.puzzle)
            return eval_id, verdict, output_tokens

        # Only send non-cutoff items to LLM judge
        items_to_judge = [i for i in judge_items if i[0] not in token_limit_ids]
        print(f"  Running LLM judge with {LLM_WORKERS} workers...")
        completed = 0
        verdict_map = {}
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
            futures = {pool.submit(_llm_task, item): item[0]
                       for item in items_to_judge}
            for future in as_completed(futures):
                eval_id, verdict, output_tokens = future.result()
                verdict_map[eval_id] = (verdict, output_tokens)
                completed += 1
                if completed % 10 == 0:
                    print(f"  Judged {completed}/{len(items_to_judge)}...")

        # Collect results in original order
        for eval_id, gt, content, output_tokens in judge_items:
            if eval_id in token_limit_ids:
                parse_failures += 1
                results.append(_token_limit_result(eval_id, gt, output_tokens))
                continue
            verdict, output_tokens = verdict_map[eval_id]
            format_issues = verdict.get("format_issues", [])
            if verdict.get("parse_fail"):
                parse_failures += 1
            if verdict["correct"]:
                correct += 1
            result = {
                "eval_id": eval_id,
                "ground_truth": verdict.get("ground_truth"),
                "predicted": verdict.get("predicted"),
                "correct": verdict["correct"],
                "output_tokens": output_tokens,
                "format_issues": format_issues,
                "llm_raw_verdict": verdict.get("llm_raw_verdict"),
            }
            results.append(result)
    else:
        # Regex judging, with optional LLM fallback for parse failures
        fallback_client = llm_client if args.llm_fallback else None
        llm_fallback_count = 0
        for eval_id, gt, content, output_tokens in judge_items:
            if eval_id in token_limit_ids:
                parse_failures += 1
                result = _token_limit_result(eval_id, gt, output_tokens)
                result["llm_fallback"] = False
                results.append(result)
                continue
            verdict = REGEX_JUDGES[args.rq](gt, content, llm_client=fallback_client)
            format_issues = verdict.get("format_issues", [])
            if verdict.get("llm_fallback"):
                llm_fallback_count += 1
            if verdict.get("parse_fail"):
                parse_failures += 1
            if verdict["correct"]:
                correct += 1
            results.append({
                "eval_id": eval_id,
                "ground_truth": verdict.get("ground_truth"),
                "predicted": verdict.get("predicted"),
                "correct": verdict["correct"],
                "output_tokens": output_tokens,
                "format_issues": format_issues,
                "llm_fallback": verdict.get("llm_fallback", False),
            })
        if llm_fallback_count:
            print(f"  LLM fallback extracted: {llm_fallback_count} items")

    # Collect format issue stats
    format_issue_counts = {}
    items_with_issues = []
    for r in results:
        for issue in r["format_issues"]:
            format_issue_counts[issue] = format_issue_counts.get(issue, 0) + 1
        if r["format_issues"]:
            items_with_issues.append({
                "eval_id": r["eval_id"],
                "issues": r["format_issues"],
            })

    # Summary
    acc = correct / total * 100 if total else 0
    judge_type = "LLM" if uses_llm else ("regex+llm_fallback" if args.llm_fallback else "regex")
    print(f"\n=== {args.rq.upper()} Judge Results ({args.model}) [{judge_type}] ===")
    print(f"Puzzle:             {args.puzzle}")
    print(f"Total:              {total}")
    print(f"Correct:            {correct}")
    print(f"Accuracy:           {acc:.2f}%")
    print(f"Parse failures:     {parse_failures}")
    if args.llm_fallback and not uses_llm:
        total_fb = sum(1 for r in results if r.get("llm_fallback"))
        print(f"LLM fallback used:  {total_fb}")
    print(f"Total output tokens: {total_output_tokens}")
    if total:
        print(f"Avg output tokens:   {total_output_tokens / total:.1f}")

    if format_issue_counts:
        print(f"\n--- Format Issues ---")
        for issue, count in sorted(format_issue_counts.items(),
                                   key=lambda x: -x[1]):
            print(f"  {issue}: {count}")
        print(f"  Total items affected: {len(items_with_issues)}")
    else:
        print(f"\nNo format issues detected.")

    # Write results
    results_dir = RESULTS_DIR / args.puzzle / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.rq}.jsonl"

    summary = {
        "_type": "summary",
        "judge_type": judge_type.lower(),
        "puzzle": args.puzzle,
        "rq": args.rq,
        "model": args.model,
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 2),
        "parse_failures": parse_failures,
        "total_output_tokens": total_output_tokens,
        "avg_output_tokens": round(total_output_tokens / total, 1) if total else 0,
        "format_issues": format_issue_counts,
        "items_with_format_issues": items_with_issues,
    }
    if uses_llm or args.llm_fallback:
        summary["llm_judge_model"] = LLM_MODEL
    if args.llm_fallback and not uses_llm:
        summary["llm_fallback_count"] = sum(1 for r in results if r.get("llm_fallback"))

    with open(out_path, "w") as f:
        f.write(json.dumps(summary) + "\n")
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
