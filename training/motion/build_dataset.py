from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import character_root, crop_roi, read_json


def extract_cycle(cap, start, end, fps, target_fps, roi, image_size):
    if end <= start + 2:
        return None
    step = max(1.0, fps / target_fps)
    indexes = np.arange(start, end, step)
    frames = []
    phases = []
    for frame_index_f in indexes:
        frame_index = min(end - 1, int(round(frame_index_f)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        crop = crop_roi(frame, roi)
        crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        frames.append(crop)
        phases.append((frame_index - start) / max(1, end - start))
    if len(frames) < 8:
        return None
    return np.asarray(frames, dtype=np.uint8), np.asarray(phases, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cycle-phase training samples; no basic-attack stage count is required")
    parser.add_argument("--character", required=True)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--min-cycle", type=float, default=0.6, help="minimum cycle duration in seconds")
    parser.add_argument("--max-cycle", type=float, default=8.0, help="maximum cycle duration in seconds")
    args = parser.parse_args()

    root = character_root(args.character)
    annotation_files = sorted((root / "annotations").glob("*.cycles.json"))
    if not annotation_files:
        raise SystemExit(f"no annotations found in {root / 'annotations'}")

    cycle_dir = root / "cycles"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for annotation_path in annotation_files:
        annotation = read_json(annotation_path)
        video = Path(annotation["video"])
        if not video.is_file():
            print(f"skip missing video: {video}")
            continue
        roi = tuple(annotation["roi"])
        fps = float(annotation["video_info"]["fps"])
        boundaries = [int(item["frame"]) for item in annotation["boundaries"]]
        cap = cv2.VideoCapture(str(video))
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
            duration = (end - start) / fps
            if not (args.min_cycle <= duration <= args.max_cycle):
                print(f"skip {annotation_path.stem} cycle {index}: duration={duration:.3f}s")
                continue
            result = extract_cycle(cap, start, end, fps, args.target_fps, roi, args.size)
            if result is None:
                continue
            frames, phases = result
            cycle_name = f"{annotation_path.stem}_{index:03d}.npz"
            target = cycle_dir / cycle_name
            np.savez_compressed(target, frames=frames, phases=phases)
            manifest.append({
                "cycle": str(target.relative_to(root)),
                "video": str(video),
                "annotation": str(annotation_path.relative_to(root)),
                "start_frame": start,
                "end_frame": end,
                "duration_s": duration,
                "frames": int(len(frames)),
                "target_fps": args.target_fps,
                "image_size": args.size,
            })
        cap.release()

    manifest_path = root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for item in manifest:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"built {len(manifest)} cycles -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
