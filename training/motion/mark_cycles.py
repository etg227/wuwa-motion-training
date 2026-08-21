from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import character_root, crop_roi, parse_roi, video_info, write_json

WINDOW = "Motion cycle marker"


def read_frame(cap, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def pose_descriptor(frame, roi):
    crop = crop_roi(frame, roi)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    vector = gray.astype(np.float32).reshape(-1)
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    if norm > 1e-6:
        vector /= norm
    return vector


def similarity(a, b) -> float:
    return float(np.dot(a, b))


def auto_suggest(cap, boundaries, total_frames, roi, search_ratio=0.20):
    if len(boundaries) < 2:
        print("至少先手动标两个相同姿态，再按 G 自动建议。")
        return boundaries
    first, second = boundaries[0], boundaries[1]
    period = second - first
    if period < 5:
        print("前两个标记太近，无法估算一轮平A。")
        return boundaries
    reference_frame = read_frame(cap, first)
    if reference_frame is None:
        return boundaries
    reference = pose_descriptor(reference_frame, roi)
    result = sorted(set(boundaries[:2]))
    expected = second + period
    radius = max(3, round(period * search_ratio))

    while expected < total_frames - 1:
        start = max(result[-1] + max(2, period // 2), expected - radius)
        end = min(total_frames - 1, expected + radius)
        best_index = None
        best_score = -2.0
        # 大视频逐帧 seek 很慢；每2帧粗搜一次，再在最佳点附近细搜。
        coarse = []
        for index in range(start, end + 1, 2):
            frame = read_frame(cap, index)
            if frame is None:
                continue
            score = similarity(reference, pose_descriptor(frame, roi))
            coarse.append((score, index))
        if not coarse:
            break
        _, coarse_best = max(coarse)
        for index in range(max(start, coarse_best - 2), min(end, coarse_best + 2) + 1):
            frame = read_frame(cap, index)
            if frame is None:
                continue
            score = similarity(reference, pose_descriptor(frame, roi))
            if score > best_score:
                best_score, best_index = score, index
        if best_index is None:
            break
        result.append(best_index)
        print(f"建议 cycle boundary: frame={best_index}, similarity={best_score:.3f}")
        expected = best_index + period
    return result


def draw_overlay(frame, frame_index, fps, boundaries, paused, roi):
    view = frame.copy()
    h, w = view.shape[:2]
    x, y, rw, rh = roi
    p1 = (round(x * w), round(y * h))
    p2 = (round((x + rw) * w), round((y + rh) * h))
    cv2.rectangle(view, p1, p2, (255, 255, 255), 2)
    text = f"frame {frame_index}  t={frame_index/fps:.3f}s  marks={len(boundaries)}  {'PAUSE' if paused else 'PLAY'}"
    cv2.putText(view, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(view, "SPACE mark | G auto | X undo | A/D +-1 | J/L +-10 | P play | S save | Q quit", (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    if boundaries:
        nearest = min(boundaries, key=lambda value: abs(value - frame_index))
        cv2.putText(view, f"nearest mark={nearest} ({nearest/fps:.3f}s)", (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def save_annotation(character, video, roi, boundaries, info):
    root = character_root(character)
    target = root / "annotations" / f"{video.stem}.cycles.json"
    value = {
        "version": 1,
        "character": character,
        "video": str(video.resolve()),
        "roi": list(roi),
        "boundary_semantics": "same_pose_recurrence; each consecutive pair is one full basic-attack cycle",
        "boundaries": [
            {"frame": int(index), "time_ms": round(index / info["fps"] * 1000)}
            for index in sorted(set(boundaries))
        ],
        "video_info": info,
    }
    write_json(target, value)
    print(f"saved: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark repeated-pose boundaries for a continuous basic-attack video")
    parser.add_argument("--character", required=True)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--roi", help="normalized x,y,w,h; default focuses on central character area")
    args = parser.parse_args()

    video = args.video.resolve()
    info = video_info(video)
    fps = info["fps"]
    total = info["frames"]
    roi = parse_roi(args.roi)
    cap = cv2.VideoCapture(str(video))
    boundaries = []
    current = 0
    paused = True

    print("不需要数平A段数。只要在同一个容易辨认的动作/姿态每次出现时按 SPACE。")
    print("先标前两个相同姿态后可按 G，让程序按周期 + 视觉相似度建议后续边界。")

    while True:
        frame = read_frame(cap, current)
        if frame is None:
            break
        cv2.imshow(WINDOW, draw_overlay(frame, current, fps, boundaries, paused, roi))
        delay = 1 if not paused else 0
        key = cv2.waitKey(delay) & 0xFF
        if key == 255 and not paused:
            current = min(total - 1, current + 1)
            if current >= total - 1:
                paused = True
            continue
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        elif key == ord(" "):
            if current not in boundaries:
                boundaries.append(current)
                boundaries.sort()
                print(f"mark frame={current} t={current/fps:.3f}s")
        elif key == ord("x"):
            if boundaries:
                removed = boundaries.pop()
                print(f"undo frame={removed}")
        elif key == ord("g"):
            paused = True
            boundaries = auto_suggest(cap, boundaries, total, roi)
        elif key == ord("a"):
            paused = True
            current = max(0, current - 1)
        elif key == ord("d"):
            paused = True
            current = min(total - 1, current + 1)
        elif key == ord("j"):
            paused = True
            current = max(0, current - 10)
        elif key == ord("l"):
            paused = True
            current = min(total - 1, current + 10)
        elif key == ord("s"):
            save_annotation(args.character, video, roi, boundaries, info)

    cap.release()
    cv2.destroyAllWindows()
    if len(boundaries) >= 2:
        save_annotation(args.character, video, roi, boundaries, info)
    else:
        print("少于两个边界，没有生成 cycle 标注。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
