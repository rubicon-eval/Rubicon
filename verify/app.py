"""Flask app for human verification of puzzle datasets."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)


# ---------------------------------------------------------------------------
# Validation persistence
# ---------------------------------------------------------------------------

def load_validation(dataset_dir: str) -> dict:
    """Load validation.json from dataset root."""
    path = os.path.join(dataset_dir, "validation.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_validation(dataset_dir: str, data: dict) -> None:
    """Atomic write of validation.json via tmp + os.replace."""
    path = os.path.join(dataset_dir, "validation.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_puzzles(dataset_dir: str) -> list[dict]:
    """Scan dataset subdirs for metadata.json, merge with validation."""
    validation = load_validation(dataset_dir)
    puzzles = []

    for name in sorted(os.listdir(dataset_dir)):
        puzzle_path = os.path.join(dataset_dir, name)
        meta_path = os.path.join(puzzle_path, "metadata.json")
        if not os.path.isdir(puzzle_path) or not os.path.exists(meta_path):
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Count eval items
        items_path = os.path.join(puzzle_path, "eval_items.json")
        num_items = 0
        if os.path.exists(items_path):
            with open(items_path, "r") as f:
                num_items = len(json.load(f))

        val = validation.get(name, {})
        puzzles.append({
            "dir": name,
            "metadata": meta,
            "num_items": num_items,
            "status": val.get("status", "pending"),
            "difficulty": val.get("difficulty"),
            "notes": val.get("notes", ""),
        })

    return puzzles


def organize_eval_items(items: list[dict]) -> dict:
    """Data-driven grouping of eval items by rq → variant → sub-group.

    Returns nested dict:
        {rq: {variant: {group_key: [items]}}}

    For RQ1, baseline items (sub_type like rq2_baseline_*) are separated
    from pure trajectory items by incorporating sub_type into the key.
    For RQ2/RQ3, group_key comes from pair_index or trace_index.
    """
    by_rq: dict[str, dict[str, dict[str, list[dict]]]] = {}

    for item in items:
        rq = item.get("rq", "unknown")
        variant = item.get("variant", "unknown")
        meta = item.get("metadata", {})
        sub_type = meta.get("sub_type")

        # Find index key (pair_index / trace_index)
        idx_part = ""
        for k, v in meta.items():
            if k.endswith("_pair_index") or k.endswith("_trace_index"):
                idx_part = f"{k}={v}"
                break

        # Build group key
        if rq == "rq1" and sub_type and sub_type != "trajectory":
            # Baseline items: prefix with sub_type so they sort apart
            group_key = f"{sub_type}:{idx_part}" if idx_part else sub_type
        elif idx_part:
            group_key = idx_part
        else:
            group_key = "all"

        by_rq.setdefault(rq, {})
        by_rq[rq].setdefault(variant, {})
        by_rq[rq][variant].setdefault(group_key, [])
        by_rq[rq][variant][group_key].append(item)

    return by_rq


def describe_item(item: dict) -> dict:
    """Generate human-readable label and compact answer for an eval item."""
    rq = item.get("rq", "")
    variant = item.get("variant", "")
    meta = item.get("metadata", {})
    gt = item.get("ground_truth", {})

    # --- Build a short human label ---
    label = ""
    if rq == "rq1" and variant == "main":
        sub = meta.get("sub_type", "trajectory")
        n = meta.get("trajectory_length", "?")
        if sub == "trajectory":
            label = f"Simulate {n} steps from initial state"
        elif "rq2_baseline" in sub:
            kind = "deadlock" if "deadlock" in sub else "solvable"
            tier = "hard " if sub.startswith("hard_") else ""
            label = f"Simulate {n} steps ({kind} path, {tier}RQ2 baseline)"
        elif "rq3_baseline" in sub:
            tier = "hard " if sub.startswith("hard_") else ""
            label = f"Simulate {n} steps (fatal trace, {tier}RQ3 baseline)"
        else:
            label = f"Simulate {n} steps"
    elif rq == "rq1" and variant == "ablation":
        direction = meta.get("direction", "?")
        n = meta.get("trajectory_length", "?")
        label = f"After {n} steps, is moving {direction} legal?"
    elif rq == "rq2":
        hard_prefix = "HARD: " if "hard" in variant else ""
        if "deadlock" in variant:
            label = f"{hard_prefix}Is the puzzle still solvable after these moves? (deadlock path)"
        else:
            label = f"{hard_prefix}Is the puzzle still solvable after these moves? (solvable path)"
        if "ablation" in variant:
            label = label.replace("Is the puzzle still solvable", "Can the exit be reached")
    elif rq == "rq3":
        gap = gt.get("cause_symptom_gap", "?")
        hard_prefix = "HARD: " if "hard" in variant else ""
        if variant in ("main", "hard"):
            label = f"{hard_prefix}Which step caused the deadlock? (gap={gap})"
        elif variant in ("ablation_final", "hard_ablation_final"):
            label = f"{hard_prefix}Identify fatal step from final state only (gap={gap})"
        elif variant in ("ablation_comparison", "hard_ablation_comparison"):
            label = f"{hard_prefix}Identify fatal step from initial vs final (gap={gap})"
        else:
            label = f"{hard_prefix}Identify the fatal step"
    else:
        label = f"{rq} / {variant}"

    # --- Build compact answer string ---
    answer = ""
    if "answer" in gt:
        answer = str(gt["answer"])
    elif "fatal_step_1indexed" in gt:
        answer = f"Step {gt['fatal_step_1indexed']} ({gt.get('fatal_action', '?')})"
    elif "agent_pos" in gt:
        answer = f"Agent at {gt['agent_pos']}"
        extras = []
        if gt.get("open_doors"):
            extras.append(f"{len(gt['open_doors'])} doors open")
        if gt.get("opened_doors"):
            extras.append(f"{len(gt['opened_doors'])} doors open")
        if gt.get("activated_plates"):
            extras.append(f"{len(gt['activated_plates'])} plates active")
        n_painted = len(gt.get("painted", []))
        if n_painted:
            extras.append(f"{n_painted} painted")
        if gt.get("inventory"):
            inv = gt["inventory"]
            total_keys = sum(inv.values()) if isinstance(inv, dict) else 0
            if total_keys:
                extras.append(f"{total_keys} keys held")
        if gt.get("remaining_keys"):
            extras.append(f"{len(gt['remaining_keys'])} keys left")
        # Tide grid specifics
        if gt.get("oxygen_remaining") is not None:
            extras.append(f"O\u2082={gt['oxygen_remaining']}")
        if gt.get("collected_beacons"):
            extras.append(f"{len(gt['collected_beacons'])} beacons")
        if gt.get("zone_states"):
            zs = gt["zone_states"]
            flooded = [k for k, v in zs.items() if v == "flooded"]
            if flooded:
                extras.append(f"flooded: {','.join(flooded)}")
            else:
                extras.append("all dry")
        if extras:
            answer += ", " + ", ".join(extras)

    return {"label": label, "answer": answer}


def get_puzzle_images(dataset_dir: str, puzzle_dir: str) -> dict:
    """List all PNGs, categorize as model-facing vs reference.

    Returns:
        {"model": set of image paths referenced by eval items,
         "reference": set of other PNGs,
         "all": sorted list of all PNG filenames}
    """
    puzzle_path = os.path.join(dataset_dir, puzzle_dir)
    images_path = os.path.join(puzzle_path, "images")
    all_pngs = sorted(
        f for f in os.listdir(images_path) if f.endswith(".png")
    ) if os.path.isdir(images_path) else []

    # Collect image paths referenced in eval items
    items_path = os.path.join(puzzle_path, "eval_items.json")
    model_paths: set[str] = set()
    if os.path.exists(items_path):
        with open(items_path, "r") as f:
            for item in json.load(f):
                img = item.get("image_path", "")
                # image_path is like "puzzle_dir/images/initial.png"
                basename = os.path.basename(img)
                if basename:
                    model_paths.add(basename)

    reference = set(all_pngs) - model_paths
    return {
        "model": sorted(model_paths & set(all_pngs)),
        "reference": sorted(reference),
        "all": all_pngs,
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(dataset_dir: str) -> Flask:
    """Create Flask app with dataset_dir in config."""
    dataset_dir = os.path.abspath(dataset_dir)

    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.config["DATASET_DIR"] = dataset_dir
    app.jinja_env.filters["describe"] = describe_item
    # Explicit variant display order: main before ablation,
    # and within each group a defined sequence.
    _variant_order = {
        # Standard (0–14)
        "main": 0,
        "main_deadlock": 1,
        "main_solvable": 2,
        "ablation": 10,
        "ablation_deadlock": 11,
        "ablation_solvable": 12,
        "ablation_final": 13,
        "ablation_comparison": 14,
        # Hard / oracle-adjacent (20+)
        "hard_deadlock": 20,
        "hard_solvable": 21,
        "hard": 22,
        "hard_ablation_deadlock": 30,
        "hard_ablation_solvable": 31,
        "hard_ablation_final": 32,
        "hard_ablation_comparison": 33,
    }
    app.jinja_env.filters["sort_variants"] = lambda vs: sorted(
        vs, key=lambda v: (_variant_order.get(v, 5 if "ablation" not in v else 15), v)
    )
    app.jinja_env.tests["hard_variant"] = lambda v: v.startswith("hard")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        return redirect(url_for("dashboard"))

    @app.route("/dashboard")
    def dashboard():
        ds = app.config["DATASET_DIR"]
        puzzles = load_puzzles(ds)

        total = len(puzzles)
        validated = sum(1 for p in puzzles if p["status"] == "validated")
        rejected = sum(1 for p in puzzles if p["status"] == "rejected")
        pending = total - validated - rejected

        difficulties = [p["difficulty"] for p in puzzles if p["difficulty"] is not None]
        avg_difficulty = round(sum(difficulties) / len(difficulties), 1) if difficulties else None

        return render_template(
            "dashboard.html",
            puzzles=puzzles,
            total=total,
            validated=validated,
            rejected=rejected,
            pending=pending,
            avg_difficulty=avg_difficulty,
        )

    @app.route("/puzzle/<puzzle_dir>")
    def puzzle_detail(puzzle_dir: str):
        ds = app.config["DATASET_DIR"]
        puzzle_path = os.path.join(ds, puzzle_dir)

        # Load metadata
        meta_path = os.path.join(puzzle_path, "metadata.json")
        if not os.path.exists(meta_path):
            return "Puzzle not found", 404
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Load eval items and organize
        items_path = os.path.join(puzzle_path, "eval_items.json")
        items = []
        if os.path.exists(items_path):
            with open(items_path, "r") as f:
                items = json.load(f)
        organized = organize_eval_items(items)

        # Load validation state
        validation = load_validation(ds)
        val = validation.get(puzzle_dir, {})

        # Images
        images = get_puzzle_images(ds, puzzle_dir)

        # Navigation: prev / next / next-pending
        all_puzzles = sorted(
            d for d in os.listdir(ds)
            if os.path.isdir(os.path.join(ds, d))
            and os.path.exists(os.path.join(ds, d, "metadata.json"))
        )
        idx = all_puzzles.index(puzzle_dir) if puzzle_dir in all_puzzles else -1
        prev_puzzle = all_puzzles[idx - 1] if idx > 0 else None
        next_puzzle = all_puzzles[idx + 1] if idx < len(all_puzzles) - 1 else None

        next_pending = None
        for i in range(idx + 1, len(all_puzzles)):
            if validation.get(all_puzzles[i], {}).get("status", "pending") == "pending":
                next_pending = all_puzzles[i]
                break
        # Wrap around if needed
        if next_pending is None:
            for i in range(0, idx):
                if validation.get(all_puzzles[i], {}).get("status", "pending") == "pending":
                    next_pending = all_puzzles[i]
                    break

        starred = set(val.get("starred", []))

        return render_template(
            "puzzle_detail.html",
            puzzle_dir=puzzle_dir,
            metadata=metadata,
            organized=organized,
            val=val,
            images=images,
            prev_puzzle=prev_puzzle,
            next_puzzle=next_puzzle,
            next_pending=next_pending,
            num_items=len(items),
            starred=starred,
        )

    @app.route("/puzzle/<puzzle_dir>/validate", methods=["POST"])
    def validate_puzzle(puzzle_dir: str):
        ds = app.config["DATASET_DIR"]
        validation = load_validation(ds)

        status = request.form.get("status", "pending")
        difficulty = request.form.get("difficulty")
        notes = request.form.get("notes", "").strip()

        existing = validation.get(puzzle_dir, {})
        validation[puzzle_dir] = {
            "status": status,
            "difficulty": int(difficulty) if difficulty else None,
            "notes": notes,
            "starred": existing.get("starred", []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_validation(ds, validation)

        # Redirect based on which button was clicked
        if request.form.get("action") == "save_next":
            # Find next pending
            all_puzzles = sorted(
                d for d in os.listdir(ds)
                if os.path.isdir(os.path.join(ds, d))
                and os.path.exists(os.path.join(ds, d, "metadata.json"))
            )
            idx = all_puzzles.index(puzzle_dir) if puzzle_dir in all_puzzles else -1
            for i in list(range(idx + 1, len(all_puzzles))) + list(range(0, idx)):
                if validation.get(all_puzzles[i], {}).get("status", "pending") == "pending":
                    return redirect(url_for("puzzle_detail", puzzle_dir=all_puzzles[i]))
            # No more pending — go to dashboard
            return redirect(url_for("dashboard"))

        return redirect(url_for("puzzle_detail", puzzle_dir=puzzle_dir))

    @app.route("/puzzle/<puzzle_dir>/star", methods=["POST"])
    def toggle_star(puzzle_dir: str):
        ds = app.config["DATASET_DIR"]
        eval_id = request.json.get("eval_id", "")
        if not eval_id:
            return jsonify({"error": "missing eval_id"}), 400

        validation = load_validation(ds)
        entry = validation.setdefault(puzzle_dir, {})
        starred = entry.get("starred", [])

        if eval_id in starred:
            starred.remove(eval_id)
            is_starred = False
        else:
            starred.append(eval_id)
            is_starred = True

        entry["starred"] = starred
        save_validation(ds, validation)
        return jsonify({"eval_id": eval_id, "starred": is_starred})

    @app.route("/dataset-image/<path:filepath>")
    def dataset_image(filepath: str):
        ds = app.config["DATASET_DIR"]
        return send_from_directory(ds, filepath)

    return app


# ---------------------------------------------------------------------------
# Entry point for CLI
# ---------------------------------------------------------------------------

def run_verify(dataset_dir: str, port: int = 5050, debug: bool = True) -> None:
    """Launch the verification web app."""
    app = create_app(dataset_dir)
    print(f"Serving verification app for: {os.path.abspath(dataset_dir)}")
    print(f"Open http://localhost:{port}/dashboard")
    app.run(host="127.0.0.1", port=port, debug=debug)
