"""UP 视频逐帧标注器：把攻略视频里的操作标成 axis_timeline.json。

用法：
    py -3.12 training/axis/annotate_video.py --video path/to/up.mp4 \
        [--out axis_timeline.json] [--start-slot 1] [--note "来源说明"]

交互：
    , / .   上一帧 / 下一帧
    J / L   -10 / +10 帧
    P       播放 / 暂停
    动作 token（在当前帧记一个事件）：
        A 普攻(左键)   V 下落A       E/Q/R/F/W 对应技能键
        Z 重击         I 变奏入场（对齐标记，不产生输入）
        1 / 2 / 3      切到对应槽位（同时更新“当前站场”）
    X       撤销最后一个事件
    S       保存
    Esc     保存并退出

标注的是“动作在画面上出现”的帧；不要在标注时人工预估真实按键提前量。
动作/转场 lead 在 compile_axis.py 中统一反推。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timeline import SWITCH_TARGET, AxisVideoEvent, Timeline, frame_to_ms

WINDOW = "UP axis annotator"

TOKEN_KEYS = {
    ord("a"): "a",
    ord("v"): "fall_a",
    ord("e"): "e",
    ord("q"): "q",
    ord("r"): "r",
    ord("z"): "z",
    ord("f"): "f",
    ord("w"): "w",
    ord("i"): "intro",
    ord("1"): "s1",
    ord("2"): "s2",
    ord("3"): "s3",
}


def read_frame(cap, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def draw_overlay(frame, current, fps, timeline: Timeline, slot: int, paused: bool):
    view = frame.copy()
    text = (
        f"frame={current} t={current / fps:.3f}s slot={slot} "
        f"events={len(timeline.events)} {'PAUSED' if paused else 'PLAY'}"
    )
    cv2.putText(
        view, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        view,
        "A atk V fall E/Q/R/F/W Z heavy I intro 1/2/3 swap | ,/. +-1 J/L +-10 "
        "P play X undo S save ESC save+quit",
        (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    for offset, event in enumerate(timeline.sorted_events()[-6:]):
        cv2.putText(
            view,
            f"{event.ms / 1000:7.3f}s  slot{event.slot}  {event.action}",
            (18, 92 + offset * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (180, 255, 180), 1, cv2.LINE_AA,
        )
    return view


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate an UP video into an axis timeline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--start-slot", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    video = Path(args.video)
    out = Path(args.out) if args.out else video.with_suffix(".axis_timeline.json")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or total <= 0:
        raise SystemExit(f"invalid video metadata: fps={fps} frames={total}")
    if fps < 45:
        print(
            f"注意：源视频只有 {fps:.1f}fps，单帧量化约 {1000 / fps:.0f}ms；"
            "极限衔接建议找 60fps 投稿。"
        )

    if out.exists():
        timeline = Timeline.load(out)
        if abs(timeline.fps - fps) > 0.01:
            raise SystemExit(
                f"已有标注 fps={timeline.fps:.3f} 与当前视频 fps={fps:.3f} 不一致；"
                "请确认没有把别的视频复用到同一个 timeline"
            )
        print(f"继续已有标注：{out}（{len(timeline.events)} 个事件）")
    else:
        timeline = Timeline(
            video=video.name,
            fps=fps,
            start_slot=args.start_slot,
            source_note=args.note,
        )

    slot = timeline.start_slot
    for event in timeline.sorted_events():
        if event.action in SWITCH_TARGET:
            slot = SWITCH_TARGET[event.action]

    # 断点续标直接回到最后一个事件，而不是每次重新从第 0 帧寻找进度。
    sorted_events = timeline.sorted_events()
    current = sorted_events[-1].frame if sorted_events else 0
    current = max(0, min(current, total - 1))
    if sorted_events:
        print(f"已跳到最后标注：frame {current} ({current / fps:.3f}s)")

    paused = True
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    while True:
        frame = read_frame(cap, current)
        if frame is None:
            current = max(0, min(current, total - 1))
            frame = read_frame(cap, current)
            if frame is None:
                break
        cv2.imshow(WINDOW, draw_overlay(frame, current, fps, timeline, slot, paused))
        key = cv2.waitKey(0 if paused else max(1, int(1000 / fps))) & 0xFF

        if key in TOKEN_KEYS:
            token = TOKEN_KEYS[key]
            event_slot = slot
            timeline.events.append(AxisVideoEvent(
                frame=current,
                ms=frame_to_ms(current, fps),
                action=token,
                slot=event_slot,
            ))
            if token in SWITCH_TARGET:
                slot = SWITCH_TARGET[token]
            print(
                f"+ {token} @frame {current} ({frame_to_ms(current, fps)}ms) "
                f"slot={event_slot}"
            )
        elif key == ord(","):
            current = max(0, current - 1)
            paused = True
        elif key == ord("."):
            current = min(total - 1, current + 1)
            paused = True
        elif key == ord("j"):
            current = max(0, current - 10)
            paused = True
        elif key == ord("l"):
            current = min(total - 1, current + 10)
            paused = True
        elif key == ord("p"):
            paused = not paused
        elif key == ord("x"):
            if timeline.events:
                removed = timeline.events.pop()
                print(f"- 撤销 {removed.action} @frame {removed.frame}")
                slot = timeline.start_slot
                for event in timeline.sorted_events():
                    if event.action in SWITCH_TARGET:
                        slot = SWITCH_TARGET[event.action]
        elif key == ord("s"):
            timeline.save(out)
            print(f"已保存 {len(timeline.events)} 个事件 -> {out}")
        elif key == 27:  # Esc；Q 留给技能 Q 标注。
            timeline.save(out)
            print(f"已保存并退出 -> {out}")
            break

        if not paused:
            current += 1
            if current >= total:
                current = total - 1
                paused = True

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
