from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import character_root


def _resolve_reference(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _collect_referenced_videos(root: Path) -> tuple[set[Path], list[str]]:
    referenced: set[Path] = set()
    errors: list[str] = []

    annotation_dir = root / "annotations"
    for path in sorted(annotation_dir.glob("*.cycles.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            video = data.get("video")
            if not isinstance(video, str) or not video.strip():
                raise ValueError("missing video field")
            referenced.add(_resolve_reference(video, root))
        except Exception as exc:
            errors.append(f"annotation {path.name}: {exc}")

    # root manifest + future per-mode manifests are all treated as keep references.
    for path in sorted(root.glob("**/manifest.jsonl")):
        try:
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                video = row.get("video")
                if isinstance(video, str) and video.strip():
                    referenced.add(_resolve_reference(video, root))
        except Exception as exc:
            errors.append(f"manifest {path.relative_to(root)}: {exc}")

    return referenced, errors


def _auto_stems(video_dir: Path) -> list[str]:
    stems: set[str] = set()
    for path in video_dir.glob("auto_*"):
        name = path.name
        if name.endswith(".inputs.jsonl"):
            stems.add(name[: -len(".inputs.jsonl")])
        elif name.endswith(".session.json"):
            stems.add(name[: -len(".session.json")])
        elif name.endswith(".mp4"):
            stems.add(path.stem)
    return sorted(stems)


def _group_files(video_dir: Path, stem: str) -> list[Path]:
    return [
        video_dir / f"{stem}.mp4",
        video_dir / f"{stem}.inputs.jsonl",
        video_dir / f"{stem}.session.json",
    ]


def _group_size(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            pass
    return total


def scan(character: str):
    root = character_root(character)
    video_dir = root / "videos"
    referenced, errors = _collect_referenced_videos(root)

    if errors:
        raise RuntimeError(
            "训练元数据存在解析异常；为避免误删，已取消清理：\n- " + "\n- ".join(errors)
        )

    # 如果目录里完全没有任何可证明的训练引用，不自动把所有 auto_* 都判成垃圾。
    if not referenced:
        raise RuntimeError(
            "没有找到任何 annotation/manifest 对录像的引用。为避免误删，已取消清理。"
        )

    keep = []
    delete = []
    for stem in _auto_stems(video_dir):
        files = _group_files(video_dir, stem)
        video = files[0].resolve()
        item = {
            "stem": stem,
            "files": files,
            "bytes": _group_size(files),
        }
        if video in referenced:
            keep.append(item)
        else:
            delete.append(item)
    return root, keep, delete


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely find/delete historical auto-training recordings not referenced by annotations/manifests"
    )
    parser.add_argument("--character", required=True)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="actually delete unreferenced auto_* mp4/input/session triplets; default is report only",
    )
    args = parser.parse_args()

    root, keep, delete = scan(args.character)
    print(f"character root: {root}")
    print(f"KEEP groups: {len(keep)}")
    for item in keep:
        print(f"  KEEP   {item['stem']}  {item['bytes'] / 1024 / 1024:.1f} MB")

    total_delete = sum(item["bytes"] for item in delete)
    print(f"DELETE candidates: {len(delete)}  total={total_delete / 1024 / 1024:.1f} MB")
    for item in delete:
        print(f"  DELETE {item['stem']}  {item['bytes'] / 1024 / 1024:.1f} MB")

    if not args.delete:
        print("\nDry-run only. Review the list, then rerun with --delete to remove DELETE candidates.")
        return 0

    deleted_files = 0
    freed = 0
    for item in delete:
        for path in item["files"]:
            try:
                if path.is_file():
                    size = path.stat().st_size
                    path.unlink()
                    freed += size
                    deleted_files += 1
            except OSError as exc:
                print(f"WARN failed to delete {path}: {exc}")

    print(
        f"deleted {deleted_files} files from {len(delete)} recording groups; "
        f"freed {freed / 1024 / 1024:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
