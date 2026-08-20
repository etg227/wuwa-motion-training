from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from common import character_root, video_info, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a local combat video into a character motion dataset")
    parser.add_argument("--character", required=True)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--source", choices=("self", "community", "other"), default="self")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    source = args.video.resolve()
    if not source.is_file():
        raise SystemExit(f"video not found: {source}")

    root = character_root(args.character)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{stamp}_{source.stem}"
    destination = root / "videos" / f"{base}{source.suffix.lower()}"
    counter = 1
    while destination.exists():
        destination = root / "videos" / f"{base}_{counter}{source.suffix.lower()}"
        counter += 1
    shutil.copy2(source, destination)

    info = video_info(destination)
    metadata = {
        "version": 1,
        "character": args.character,
        "source": args.source,
        "note": args.note,
        "original_name": source.name,
        "video": str(destination.relative_to(root)),
        "sha256": sha256(destination),
        **info,
    }
    meta_path = destination.with_suffix(destination.suffix + ".json")
    write_json(meta_path, metadata)

    print(destination)
    print(f"fps={info['fps']:.3f} frames={info['frames']} duration={info['duration_ms']/1000:.2f}s")
    print("下一步：用 mark_cycles.py 标记同一个动作/姿态重复出现的位置。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
