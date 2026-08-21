"""UP 视频轴时间线的数据模型与编译逻辑。

工作流：annotate_video.py 逐帧标注生成 axis_timeline.json，本模块负责：
- 时间线的加载 / 保存 / 校验；
- 编译成 RawInputTimeline 宏源码（忌莫守式，事件 + 绝对间隔）；
- 汇总成状态机轴节点报告（秧千穗式，按站场角色分组 + gap 建议）。

关键概念——输入缓冲提前量（lead）：视频里标注的是"动作在画面上出现"的
时刻，真实按键要早一个输入缓冲窗口。lead_ms 从标注时刻里减去这个提前量；
默认 0（按画面原样），实机校准后再统一设置。

本模块只依赖标准库，保证在 CI 里可测；cv2 只在标注器里使用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# action token -> (输入类别, 默认按住时长 ms)
# 类别：mouse=左键、key=键盘轻点、switch=切人数字键、marker=无输入的对齐标记。
# 默认 hold 沿用忌莫守宏已实机验证的量级；z（重击）按住约半秒，w 短按前进。
ACTIONS: dict[str, tuple[str, int]] = {
    "a": ("mouse", 78),
    "fall_a": ("mouse", 78),
    "e": ("key", 78),
    "q": ("key", 78),
    "r": ("key", 78),
    "z": ("mouse", 550),
    "f": ("key", 78),
    "w": ("key", 180),
    "s1": ("switch", 78),
    "s2": ("switch", 78),
    "s3": ("switch", 78),
    "intro": ("marker", 0),
}

SWITCH_TARGET = {"s1": 1, "s2": 2, "s3": 3}
KEY_CODE = {"e": "e", "q": "q", "r": "r", "f": "f", "w": "w",
            "s1": "1", "s2": "2", "s3": "3"}

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AxisVideoEvent:
    frame: int
    ms: int
    action: str
    slot: int  # 事件发生时站场的槽位（1..3）
    note: str = ""

    def __post_init__(self):
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action token: {self.action}")
        if self.frame < 0 or self.ms < 0:
            raise ValueError("frame/ms must be >= 0")
        if not 1 <= self.slot <= 3:
            raise ValueError(f"slot must be 1..3, got {self.slot}")


@dataclass
class Timeline:
    video: str
    fps: float
    start_slot: int = 1
    source_note: str = ""
    events: list[AxisVideoEvent] = field(default_factory=list)

    def sorted_events(self) -> list[AxisVideoEvent]:
        return sorted(self.events, key=lambda event: (event.ms, event.frame))

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "video": self.video,
            "fps": self.fps,
            "start_slot": self.start_slot,
            "source_note": self.source_note,
            "events": [
                {"frame": e.frame, "ms": e.ms, "action": e.action,
                 "slot": e.slot, "note": e.note}
                for e in self.sorted_events()
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Timeline":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        timeline = cls(
            video=str(data["video"]),
            fps=float(data["fps"]),
            start_slot=int(data.get("start_slot", 1)),
            source_note=str(data.get("source_note", "")),
        )
        for row in data.get("events", []):
            timeline.events.append(AxisVideoEvent(
                frame=int(row["frame"]),
                ms=int(row["ms"]),
                action=str(row["action"]),
                slot=int(row["slot"]),
                note=str(row.get("note", "")),
            ))
        return timeline


def frame_to_ms(frame: int, fps: float) -> int:
    return round(frame / fps * 1000)


# ---------------------------------------------------------------------------
# 宏编译：时间线 -> RawInputTimeline 事件（down/up + 绝对间隔）


@dataclass(frozen=True)
class MacroStep:
    device: str  # key / mouse
    code: str
    action: str  # down / up
    delay_after_ms: int
    label: str


def compile_macro(
    timeline: Timeline,
    lead_ms: int = 0,
    hold_overrides: dict[str, int] | None = None,
) -> tuple[list[MacroStep], list[str]]:
    """把标注时间线编译成 down/up 事件序列。

    lead_ms：输入缓冲提前量，按键实际发生在画面时刻之前 lead_ms 毫秒。
    返回 (事件列表, 告警列表)。告警包括：按住时长与下一事件重叠被截短、
    首事件提前量被钳到 0 等。所有告警都不致命，但值得人工复查。
    """
    holds = dict(hold_overrides or {})
    inputs = [e for e in timeline.sorted_events() if ACTIONS[e.action][0] != "marker"]
    warnings: list[str] = []
    steps: list[MacroStep] = []

    press_times: list[int] = []
    for event in inputs:
        press = event.ms - lead_ms
        if press < 0:
            warnings.append(
                f"{event.action}@{event.ms}ms: lead 补偿后早于 0，钳到 0"
            )
            press = 0
        press_times.append(press)

    for index, event in enumerate(inputs):
        kind, default_hold = ACTIONS[event.action]
        hold = int(holds.get(event.action, default_hold))
        press = press_times[index]
        next_press = press_times[index + 1] if index + 1 < len(inputs) else None

        if next_press is not None:
            gap = next_press - press
            if gap <= 0:
                warnings.append(
                    f"{event.action}@{event.ms}ms 与下一事件同帧或乱序（gap={gap}ms），"
                    "wait 记 0，请人工复查标注"
                )
                wait = 0
                hold = min(hold, 10)
            elif hold >= gap:
                clamped = max(10, gap - 10)
                warnings.append(
                    f"{event.action}@{event.ms}ms: 默认按住 {hold}ms 超过与下一事件的间隔 "
                    f"{gap}ms，截短为 {clamped}ms"
                )
                hold = clamped
                wait = gap - hold
            else:
                wait = gap - hold
        else:
            wait = 0

        if kind == "mouse":
            device, code = "mouse", "left"
        else:
            device, code = "key", KEY_CODE[event.action]

        stamp = f"@{event.ms / 1000:.3f}s"
        label = f"{event.action} {stamp}" + (f" {event.note}" if event.note else "")
        steps.append(MacroStep(device, code, "down", hold, f"{label} 按下"))
        steps.append(MacroStep(device, code, "up", wait, f"{label} 抬起"))

    return steps, warnings


def macro_source(steps: list[MacroStep], name: str = "UP_MACRO") -> str:
    """输出可直接粘贴进 JimoshouAxis 风格文件的 *_key/*_mouse 源码。"""
    lines = [f"{name} = ("]
    for index in range(0, len(steps), 2):
        down, up = steps[index], steps[index + 1]
        label = down.label.removesuffix(" 按下")
        if down.device == "mouse":
            lines.append(
                f'    *_mouse({down.delay_after_ms}, {up.delay_after_ms}, "{label}"),'
            )
        else:
            lines.append(
                f'    *_key("{down.code}", {down.delay_after_ms}, '
                f'{up.delay_after_ms}, "{label}"),'
            )
    lines.append(")")
    return "\n".join(lines)


def reconstruct_press_times(steps: list[MacroStep]) -> list[int]:
    """按 delay_after 语义重建每次按下的绝对时刻，用于测试与对齐验证。"""
    at = 0
    presses = []
    for step in steps:
        if step.action == "down":
            presses.append(at)
        at += step.delay_after_ms
    return presses


# ---------------------------------------------------------------------------
# 轴节点汇总：按站场角色分组，生成状态机轴的节点草表与 gap 报告


def compile_axis_nodes(timeline: Timeline) -> list[dict]:
    """把时间线按"谁在场上"切成节点：切人事件结束当前节点并开启下一个。

    输出节点草表；具体落到 YangqianSuiAxis 风格文件时由人工整理，
    gap 字段给出节点内相邻动作与节点间切换的画面实测间隔。
    """
    nodes: list[dict] = []
    current: dict | None = None

    for event in timeline.sorted_events():
        if event.action in SWITCH_TARGET:
            if current is not None:
                current["swap_gap_ms"] = event.ms - current["events"][-1]["ms"] \
                    if current["events"] else None
                nodes.append(current)
            current = {
                "slot": SWITCH_TARGET[event.action],
                "start_ms": event.ms,
                "events": [],
                "swap_gap_ms": None,
            }
            continue

        if current is None:
            current = {
                "slot": event.slot,
                "start_ms": event.ms,
                "events": [],
                "swap_gap_ms": None,
            }
        previous_ms = current["events"][-1]["ms"] if current["events"] else None
        current["events"].append({
            "action": event.action,
            "ms": event.ms,
            "gap_ms": event.ms - previous_ms if previous_ms is not None else None,
        })

    if current is not None:
        nodes.append(current)

    for node in nodes:
        node["label"] = "".join(
            {"fall_a": "下落a", "intro": "变"}.get(row["action"], row["action"])
            for row in node["events"]
        )
    return nodes
