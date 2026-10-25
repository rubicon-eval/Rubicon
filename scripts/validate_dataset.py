"""Validate the released evaluation records, images and source checksums offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUZZLES = ("chain_vault", "mosaic_grid", "tide_grid")
TASKS = ("rq1_main", "rq2_hard_ablation_deadlock", "rq2_hard_ablation_solvable",
         "rq2_hard_deadlock", "rq2_hard_solvable", "rq3_hard", "rq3_hard_ablation_final")


def validate(dataset: Path, manifest: dict) -> dict:
    from PIL import Image
    for name, digest in manifest["files"].items():
        path = dataset.joinpath(*Path(name).parts[1:])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or changed release file: {name}")
    total = 0
    images = set()
    for puzzle in PUZZLES:
        base = dataset / f"dataset_{puzzle}"
        for task in TASKS:
            rows = json.loads((base / f"{task}.json").read_text())
            if len(rows) != 100:
                raise ValueError(f"Expected 100 records in {puzzle}/{task}, got {len(rows)}")
            seen = set()
            for row in rows:
                if not row["prompt"].strip() or row["eval_id"] in seen:
                    raise ValueError(f"Empty prompt or duplicate eval_id in {puzzle}/{task}")
                seen.add(row["eval_id"])
                if not isinstance(row["ground_truth"], dict) or not isinstance(row["metadata"], dict):
                    raise ValueError("Ground truth and metadata must be objects")
                image = (base / row["image_path"]).resolve()
                if not image.is_relative_to(base.resolve()):
                    raise ValueError("Image reference escapes the dataset directory")
                images.add(image)
            total += len(rows)
    for image in images:
        with Image.open(image) as opened:
            opened.verify()
    return {"records": total, "images": len(images), "domains": len(PUZZLES), "task_variants": len(TASKS)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parent / "data" / "dataset-manifest.json").read_text())
    print(json.dumps(validate(args.dataset_dir, manifest), indent=2))


if __name__ == "__main__":
    main()
