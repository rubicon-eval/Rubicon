"""Build viewer-ready Parquet files and a byte-preserving original ZIP. Requires [data]."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from datasets import Dataset, Features, Image, Value

ROOT = Path(__file__).resolve().parents[1]


def export(dataset_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    features = Features({
        "eval_id": Value("string"), "puzzle": Value("string"), "task": Value("string"),
        "puzzle_dir": Value("string"), "rq": Value("string"), "variant": Value("string"),
        "image": Image(), "image_path": Value("string"), "prompt": Value("string"),
        "ground_truth_json": Value("string"), "metadata_json": Value("string"),
    })
    counts = {}
    for domain in sorted(dataset_dir.glob("dataset_*")):
        records = []
        for path in sorted(domain.glob("rq*.json")):
            for row in json.loads(path.read_text()):
                records.append({
                    **{key: row[key] for key in ("eval_id", "puzzle_dir", "rq", "variant", "image_path", "prompt")},
                    "puzzle": domain.name.removeprefix("dataset_"), "task": path.stem,
                    "image": {"bytes": (domain / row["image_path"]).read_bytes(), "path": row["image_path"]},
                    "ground_truth_json": json.dumps(row["ground_truth"], ensure_ascii=False, sort_keys=True),
                    "metadata_json": json.dumps(row["metadata"], ensure_ascii=False, sort_keys=True),
                })
        Dataset.from_list(records, features=features).to_parquet(output / f"{domain.name.removeprefix('dataset_')}.parquet")
        counts[domain.name] = len(records)
    with zipfile.ZipFile(output / "rubicon-original.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dataset_dir.rglob("*")):
            if path.is_file():
                archive.write(path, "dataset/" + path.relative_to(dataset_dir).as_posix())
    (output / "dataset-manifest.json").write_bytes((Path(__file__).resolve().parent / "data" / "dataset-manifest.json").read_bytes())
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.dataset_dir, args.output), indent=2))
