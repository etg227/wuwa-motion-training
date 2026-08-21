"""把 axis_timeline.json 编译成可执行产物。

用法：
    py -3.12 training/axis/compile_axis.py --timeline up.axis_timeline.json \
        [--lead-ms 0] [--hold z=550,w=180] [--macro-name UP_MACRO]

输出（打印到 stdout，同时写同名 .compiled.txt）：
    1. RawInputTimeline 宏源码（*_key/*_mouse 行，可直接粘贴进忌莫守式轴文件）；
    2. 状态机轴节点草表（按站场分组 + 画面实测 gap，供整理进秧千穗式轴）；
    3. 编译告警（按住截短、同帧事件等，逐条人工复查）。

--lead-ms 是输入缓冲提前量：视频标注的是动作出现的时刻，真实按键更早。
先用 0 编译实机试跑，再用 pydirect_transport_probe 实测"按下→动作出现"的
延迟，把测得值填回来重编。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timeline import Timeline, compile_axis_nodes, compile_macro, macro_source


def parse_holds(text: str | None) -> dict[str, int]:
    if not text:
        return {}
    holds = {}
    for part in text.split(","):
        token, _, value = part.partition("=")
        holds[token.strip()] = int(value)
    return holds


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile an axis timeline into macro + node report")
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--lead-ms", type=int, default=0)
    parser.add_argument("--hold", default=None,
                        help="覆盖默认按住时长，如 z=550,w=180")
    parser.add_argument("--macro-name", default="UP_MACRO")
    args = parser.parse_args()

    path = Path(args.timeline)
    timeline = Timeline.load(path)
    steps, warnings = compile_macro(
        timeline, lead_ms=args.lead_ms, hold_overrides=parse_holds(args.hold))
    nodes = compile_axis_nodes(timeline)

    sections = []
    sections.append(
        f"# 来源: {timeline.video} ({timeline.fps:.2f}fps)  事件数: "
        f"{len(timeline.events)}  lead_ms: {args.lead_ms}"
    )
    if timeline.source_note:
        sections.append(f"# {timeline.source_note}")

    sections.append("\n## 1. RawInputTimeline 宏源码\n")
    sections.append(macro_source(steps, args.macro_name))

    sections.append("\n## 2. 状态机轴节点草表\n")
    sections.append(json.dumps(nodes, ensure_ascii=False, indent=2))

    sections.append("\n## 3. 编译告警\n")
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
