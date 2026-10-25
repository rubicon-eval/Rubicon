"""Download and verify the original RUBICON evaluation files from Hugging Face."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract_verified(archive: Path, output: Path, manifest: dict) -> int:
    expected = manifest["files"]
    with zipfile.ZipFile(archive) as source:
        members = [item for item in source.infolist() if not item.is_dir()]
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("Archive contents do not match the release manifest")
        # Verify everything before writing into the destination.
        with tempfile.TemporaryDirectory(prefix="rubicon-") as temporary:
            staging = Path(temporary)
            for item in members:
                relative = Path(item.filename)
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "dataset":
                    raise ValueError(f"Unsafe archive path: {item.filename}")
                content = source.read(item)
                if hashlib.sha256(content).hexdigest() != expected[item.filename]:
                    raise ValueError(f"Checksum mismatch: {item.filename}")
                target = staging.joinpath(*relative.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            if output.is_symlink() or (output.exists() and any(output.iterdir())):
                raise FileExistsError(f"Destination must be empty: {output}")
            shutil.copytree(staging, output, dirs_exist_ok=True)
    return len(members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    parser.add_argument("--revision", default="main", help="Hugging Face revision; checksums always pin the source dataset")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded original ZIP instead of the network")
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parent / "data" / "dataset-manifest.json").read_text())
    if args.archive:
        archive = args.archive
    else:
        from huggingface_hub import hf_hub_download
        archive = Path(hf_hub_download("ajirs/rubicon", "rubicon-original.zip", repo_type="dataset", revision=args.revision))
    count = extract_verified(archive, args.output.resolve(), manifest)
    print(f"Verified and extracted {count} files to {args.output}")


if __name__ == "__main__":
    main()
