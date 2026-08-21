"""把 axis_timeline.json 编译成可执行产物。

用法：
    py -3.12 training/axis/compile_axis.py --timeline up.axis_timeline.json \
        [--lead-ms 0] [--action-lead e=80,r=110,s2=60] \
        [--transition-lead e:s2=95,a:s3=55] \
        [--hold z=550,w=180] [--macro-name UP_MACRO]

输出：
    1. 视觉事件反推的估计输入时间表；
    2. RawInputTimeline 宏源码；
    3. 按站场分组的状态机轴节点草表；
    4. 编译告警。

lead 优先级：transition > action > global。
统一 global lead 只会移动“宏原点对应的视频时刻”，不会改变动作间相对 gap；真正
影响极限衔接的是 action/transition lead 的差值。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timeline import (
    ACTIONS,
    Timeline,
    compile_axis_nodes,
    compile_macro,
    estimate_press_schedule,
    macro_source,
)


def parse_named_ints(text: str | None, *, transitions: bool = False) -> dict:
    if not text:
        return {}
    values = {}
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        key, separator, raw_value = part.partition("=")
        if not separator:
            raise SystemExit(f"invalid mapping {part!r}; expected key=ms")
        key = key.strip()
        try:
            value = int(raw_value.strip())
        except ValueError as exc:
            raise SystemExit(f"invalid ms value in {part!r}") from exc

        if transitions:
            before, transition_separator, after = key.partition(":")
            if not transition_separator:
                raise SystemExit(
                    f"invalid transition {key!r}; expected from:to=ms, e.g. e:s2=95"
                )
            before = before.strip()
            after = after.strip()
            if before not in ACTIONS or after not in ACTIONS:
                raise SystemExit(f"unknown transition action: {before}:{after}")
            values[f"{before}:{after}"] = value
        else:
            if key not in ACTIONS or ACTIONS[key][0] == "marker":
                raise SystemExit(f"unknown/non-input action: {key}")
            values[key] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile an axis timeline into input schedule + macro + node report")
    parser.add_argument("--timeline", required=True)
    parser.add_argument(
        "--lead-ms", type=int, default=0,
        help="所有未单独配置动作的默认视觉→输入提前量",
    )
    parser.add_argument(
        "--action-lead", default=None,
        help="按动作配置 lead，如 e=80,r=110,s2=60",
    )
    parser.add_argument(
        "--transition-lead", default=None,
        help="按动作转场配置 lead，如 e:s2=95,a:s3=55；优先于 action lead",
    )
    parser.add_argument(
        "--hold", default=None,
        help="覆盖默认按住时长，如 z=550,w=180",
    )
    parser.add_argument("--macro-name", default="UP_MACRO")
    args = parser.parse_args()

    path = Path(args.timeline)
    timeline = Timeline.load(path)
    action_leads = parse_named_ints(args.action_lead)
    transition_leads = parse_named_ints(args.transition_lead, transitions=True)
    holds = parse_named_ints(args.hold)

    schedule, _ = estimate_press_schedule(
        timeline,
        lead_ms=args.lead_ms,
        action_leads=action_leads,
        transition_leads=transition_leads,
    )
    steps, warnings = compile_macro(
        timeline,
        lead_ms=args.lead_ms,
        hold_overrides=holds,
        action_leads=action_leads,
        transition_leads=transition_leads,
    )
    nodes = compile_axis_nodes(timeline)

    origin_ms = schedule[0].source_press_ms if schedule else 0
    schedule_rows = [
        {
            "macro_ms": row.macro_ms,
            "video_visual_ms": row.event.ms,
            "estimated_input_ms": row.source_press_ms,
            "lead_ms": row.lead_ms,
            "from_action": row.previous_action,
            "action": row.event.action,
            "slot": row.event.slot,
            "frame": row.event.frame,
        }
        for row in schedule
    ]

    sections = [
        f"# 来源: {timeline.video} ({timeline.fps:.2f}fps)  事件数: {len(timeline.events)}",
        f"# macro_origin_video_ms: {origin_ms}",
        f"# global_lead_ms: {args.lead_ms}",
    ]
    if action_leads:
        sections.append(f"# action_leads: {json.dumps(action_leads, ensure_ascii=False, sort_keys=True)}")
    if transition_leads:
        sections.append(
            f"# transition_leads: {json.dumps(transition_leads, ensure_ascii=False, sort_keys=True)}"
        )
    if timeline.source_note:
        sections.append(f"# {timeline.source_note}")

    sections.append("\n## 1. 估计输入时间表\n")
    sections.append(json.dumps(schedule_rows, ensure_ascii=False, indent=2))

    sections.append("\n## 2. RawInputTimeline 宏源码\n")
    sections.append(macro_source(steps, args.macro_name))

    sections.append("\n## 3. 状态机轴节点草表（视频视觉 gap）\n")
    sections.append(json.dumps(nodes, ensure_ascii=False, indent=2))

    sections.append("\n## 4. 编译告警\n")
    if warnings:
        sections.extend(f"- {line}" for line in warnings)
    else:
        sections.append("(无)")

    report = "\n".join(sections)
    print(report)
    out = path.with_suffix(".compiled.txt")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\n已写入 {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
