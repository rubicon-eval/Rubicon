#!/usr/bin/env python3
"""Browser-based GUI for finding and replacing text in prompts across RQ files.

Usage:
    uv run python -m scripts.edit_prompts_gui dataset_hard/
    uv run python -m scripts.edit_prompts_gui dataset_hard/ --port 7861
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr


# ── Data helpers ─────────────────────────────────────────────

def find_rq_files(dataset: Path) -> dict[str, Path]:
    rq_files = {}
    for f in sorted(dataset.glob("rq*.json")):
        rq_files[f.stem] = f
    return rq_files


def load_items(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def format_prompt(items: list[dict], index: int = 0) -> str:
    return items[index]["prompt"]


def build_full_preview(prompt: str, replacements: list[tuple[str, str]]) -> str:
    """Apply replacements and return the modified prompt."""
    for old, new in replacements:
        if old:
            prompt = prompt.replace(old, new)
    return prompt


def build_stats(rq_name: str, items: list[dict], replacements: list[tuple[str, str]]) -> str:
    """Count affected items in the selected RQ file."""
    active = [(o, n) for o, n in replacements if o]
    if not active:
        return "No replacements defined."
    affected = sum(
        1 for item in items
        if any(o in item["prompt"] for o, _ in active)
    )
    return f"  {rq_name}: {affected}/{len(items)} items affected"


def apply_replacements(dataset: Path, rq_name: str, rq_path: Path, replacements: list[tuple[str, str]]) -> str:
    """Apply replacements to a single RQ file and log to replacements.json."""
    active = [(o, n) for o, n in replacements if o]
    if not active:
        return "No replacements to apply."
    items = load_items(rq_path)
    total_modified = 0
    for item in items:
        original = item["prompt"]
        modified = original
        for old, new in active:
            modified = modified.replace(old, new)
        if modified != original:
            item["prompt"] = modified
            total_modified += 1

    if total_modified:
        with open(rq_path, "w") as f:
            json.dump(items, f, indent=2)
            f.write("\n")

    # Append to replacements.json for bookkeeping
    log_path = dataset / "replacements.json"
    history = []
    if log_path.exists():
        with open(log_path) as f:
            history = json.load(f)
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rq_file": rq_name,
        "replacements": [{"old": o, "new": n} for o, n in active],
        "items_modified": total_modified,
    })
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
        f.write("\n")

    return f"Done. {total_modified} prompts modified in {rq_name}.json\nLog saved to {log_path}"


# ── GUI builder ──────────────────────────────────────────────

def build_gui(dataset: Path):
    rq_files = find_rq_files(dataset)
    if not rq_files:
        print(f"No rq*.json files found in {dataset}")
        sys.exit(1)

    rq_names = list(rq_files.keys())
    rq_labels = [f"{name} ({len(load_items(rq_files[name]))} items)" for name in rq_names]

    with gr.Blocks(title="Prompt Editor") as app:
        gr.Markdown(f"## Prompt Editor — `{dataset}`")

        # ── Step 1: Browse ───────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Browse Prompts")
                rq_dropdown = gr.Dropdown(
                    choices=rq_labels, label="Select RQ file", value=rq_labels[0]
                )
                item_slider = gr.Slider(
                    minimum=0, maximum=0, step=1, value=0, label="Item index"
                )
            with gr.Column(scale=3):
                prompt_display = gr.Textbox(
                    label="Prompt", lines=20, interactive=False
                )

        # ── Step 2: Replacements (editable table) ────────
        gr.Markdown("### 2. Define Replacements")
        gr.Markdown("*Right-click a row to insert/delete. Click a cell to edit.*")
        repl_table = gr.Dataframe(
            value=[["", ""]],
            headers=["Find", "Replace with"],
            datatype=["str", "str"],
            interactive=True,
            column_widths=["50%", "50%"],
        )

        # ── Step 3: Preview ──────────────────────────────
        gr.Markdown("### 3. Preview")
        preview_btn = gr.Button("Preview Changes", variant="secondary")
        stats_display = gr.Textbox(label="Matches in selected RQ", lines=3, interactive=False)
        full_preview = gr.Textbox(label="Modified prompt (preview)", lines=20, interactive=False)

        # ── Step 4: Apply ────────────────────────────────
        gr.Markdown("### 4. Apply")
        apply_btn = gr.Button("Apply Replacements", variant="primary")
        result_display = gr.Textbox(label="Result", lines=5, interactive=False)

        # ── Event handlers ───────────────────────────────

        def on_rq_change(rq_label):
            idx = rq_labels.index(rq_label)
            name = rq_names[idx]
            items = load_items(rq_files[name])
            max_idx = len(items) - 1
            return (
                format_prompt(items, 0),
                gr.update(maximum=max_idx, value=0),
            )

        def on_item_change(rq_label, item_idx):
            idx = rq_labels.index(rq_label)
            name = rq_names[idx]
            items = load_items(rq_files[name])
            return format_prompt(items, int(item_idx))

        def parse_table(table_data):
            """Extract (old, new) pairs from dataframe data."""
            replacements = []
            if table_data is None:
                return replacements
            for row in table_data.values.tolist() if hasattr(table_data, 'values') else table_data:
                old = str(row[0] or "").strip()
                new = str(row[1] or "").strip()
                if old:
                    replacements.append((old, new))
            return replacements

        def on_preview(rq_label, item_idx, table_data):
            replacements = parse_table(table_data)

            idx = rq_labels.index(rq_label)
            name = rq_names[idx]
            items = load_items(rq_files[name])
            prompt = items[int(item_idx)]["prompt"]

            stats = build_stats(name, items, replacements)
            full = build_full_preview(prompt, replacements)
            return stats, full

        def on_apply(rq_label, table_data):
            replacements = parse_table(table_data)
            idx = rq_labels.index(rq_label)
            name = rq_names[idx]
            return apply_replacements(dataset, name, rq_files[name], replacements)

        # Wire events
        rq_dropdown.change(
            on_rq_change, inputs=[rq_dropdown], outputs=[prompt_display, item_slider]
        )
        item_slider.change(
            on_item_change, inputs=[rq_dropdown, item_slider], outputs=[prompt_display]
        )
        preview_btn.click(
            on_preview,
            inputs=[rq_dropdown, item_slider, repl_table],
            outputs=[stats_display, full_preview],
        )
        apply_btn.click(
            on_apply,
            inputs=[rq_dropdown, repl_table],
            outputs=[result_display],
        )

        # Load initial prompt
        app.load(
            on_rq_change, inputs=[rq_dropdown], outputs=[prompt_display, item_slider]
        )

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Browser GUI for find-and-replace across RQ prompt files."
    )
    parser.add_argument("dataset", type=Path, help="Path to filtered dataset directory")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on (default 7860)")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        print(f"Error: {args.dataset} is not a directory")
        sys.exit(1)

    app = build_gui(args.dataset)
    app.launch(server_port=args.port, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
