"""UP 视频轴时间线的数据模型与编译逻辑。

时间线里的 event.ms 始终表示视频中“动作在画面上出现”的时刻，而不是实际
按键时刻。编译时可按动作或动作转场配置视觉→输入提前量（lead），再把估计
输入时刻转换成 RawInputTimeline 相对时间轴。

lead 优先级：transition lead > action lead > global lead。
例如 e:s2 可以专门描述“E 动画中切 2”的输入提前量，而不影响其它切人。

RawInputTimeline 从最早的估计输入开始（宏 t=0）。因此视频可以保留任意片头，
不需要把视频第 0 帧误当成宏开始时刻；编译报告会保留宏原点对应的视频时刻。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# action token -> (输入类别, 默认按住时长 ms)
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
KEY_CODE = {
    "e": "e", "q": "q", "r": "r", "f": "f", "w": "w",
    "s1": "1", "s2": "2", "s3": "3",
}

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
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return round(frame / fps * 1000)


# ---------------------------------------------------------------------------
# 视觉时刻 -> 估计输入时刻


@dataclass(frozen=True)
class EstimatedPress:
    event: AxisVideoEvent
    previous_action: str | None
    lead_ms: int
    source_press_ms: int  # 相对原视频第 0 帧，可为负数
    macro_ms: int         # 最早估计输入归零后的宏时刻


def _normalize_transition_leads(
    transition_leads: dict[tuple[str, str] | str, int] | None,
) -> dict[tuple[str, str], int]:
    normalized: dict[tuple[str, str], int] = {}
    for key, value in (transition_leads or {}).items():
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError(f"invalid transition lead key: {key!r}")
            before, after = key
        else:
            before, separator, after = str(key).partition(":")
            if not separator:
                raise ValueError(
                    f"invalid transition lead key {key!r}; expected 'from:to'"
                )
        before = str(before).strip()
        after = str(after).strip()
        if before not in ACTIONS or after not in ACTIONS:
            raise ValueError(f"unknown transition action: {before}:{after}")
        if ACTIONS[after][0] == "marker":
            raise ValueError(f"transition target has no input: {before}:{after}")
        normalized[(before, after)] = int(value)
    return normalized


def estimate_press_schedule(
    timeline: Timeline,
    *,
    lead_ms: int = 0,
    action_leads: dict[str, int] | None = None,
    transition_leads: dict[tuple[str, str] | str, int] | None = None,
) -> tuple[list[EstimatedPress], list[str]]:
    """按 lead 从视觉事件反推输入时刻，并以最早输入作为宏 t=0。

    transition 使用前一个“有输入的视觉动作”作为 from；intro 等 marker 不参与。
    不同 lead 可能让输入顺序与视觉出现顺序不同，这种情况会显式告警。
    """

    default_lead = int(lead_ms)
    action_map = {str(key): int(value) for key, value in (action_leads or {}).items()}
    for action in action_map:
        if action not in ACTIONS or ACTIONS[action][0] == "marker":
            raise ValueError(f"action lead is not valid for input action: {action}")
    transition_map = _normalize_transition_leads(transition_leads)

    visual_inputs = [
        event for event in timeline.sorted_events()
        if ACTIONS[event.action][0] != "marker"
    ]
    if not visual_inputs:
        return [], []

    raw_rows: list[tuple[int, AxisVideoEvent, str | None, int, int]] = []
    previous_action: str | None = None
    for visual_index, event in enumerate(visual_inputs):
        transition_lead = (
            transition_map.get((previous_action, event.action))
            if previous_action is not None else None
        )
        effective_lead = (
            transition_lead
            if transition_lead is not None
            else action_map.get(event.action, default_lead)
        )
        source_press_ms = event.ms - effective_lead
        raw_rows.append((
            visual_index, event, previous_action,
            int(effective_lead), int(source_press_ms),
        ))
        previous_action = event.action

    ordered = sorted(raw_rows, key=lambda row: (row[4], row[0]))
    warnings: list[str] = []
    if [row[0] for row in ordered] != list(range(len(raw_rows))):
        warnings.append(
            "不同动作/转场 lead 使估计按键顺序与视觉出现顺序不同；"
            "这可能是输入缓冲，也可能是 lead 过大，请逐帧复查"
        )

    origin_ms = ordered[0][4]
    if origin_ms < 0:
        warnings.append(
            f"最早估计输入位于视频起点前 {-origin_ms}ms；宏仍以该输入归零，"
            "说明源视频前摇不足以直接验证这颗输入"
        )

    schedule = [
        EstimatedPress(
            event=event,
            previous_action=previous,
            lead_ms=effective_lead,
            source_press_ms=source_press_ms,
            macro_ms=source_press_ms - origin_ms,
        )
        for _, event, previous, effective_lead, source_press_ms in ordered
    ]
    return schedule, warnings


# ---------------------------------------------------------------------------
# 宏编译：估计输入时刻 -> RawInputTimeline 事件


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
    *,
    action_leads: dict[str, int] | None = None,
    transition_leads: dict[tuple[str, str] | str, int] | None = None,
) -> tuple[list[MacroStep], list[str]]:
    """把视觉时间线编译成 down/up 事件序列。

    第一颗 down 固定为宏 t=0。原视频中的绝对估计输入时刻由
    estimate_press_schedule() 保留，供逐帧对齐和校准使用。
    """

    holds = {str(key): int(value) for key, value in (hold_overrides or {}).items()}
    schedule, warnings = estimate_press_schedule(
        timeline,
        lead_ms=lead_ms,
        action_leads=action_leads,
        transition_leads=transition_leads,
    )
    steps: list[MacroStep] = []

    for index, row in enumerate(schedule):
        event = row.event
        kind, default_hold = ACTIONS[event.action]
        hold = int(holds.get(event.action, default_hold))
        if hold < 0:
            raise ValueError(f"hold must be >= 0 for {event.action}")

        press = row.macro_ms
        next_press = schedule[index + 1].macro_ms if index + 1 < len(schedule) else None

        if next_press is not None:
            gap = next_press - press
            if gap <= 0:
                warnings.append(
                    f"{event.action}@{event.ms}ms 与下一估计输入同刻（gap={gap}ms），"
                    "wait 记 0，请人工复查 lead/标注"
                )
                wait = 0
                hold = min(hold, 10)
            elif hold >= gap:
                # 极限轴可能只有几 ms 间隔。旧实现最少强制 10ms hold，会得到负 wait。
                # 这里保证 hold + wait == gap，且二者都不为负。
                clamped = max(1, gap - 1) if gap > 1 else 1
                clamped = min(clamped, gap)
                warnings.append(
                    f"{event.action}@{event.ms}ms: 默认按住 {hold}ms 超过与下一估计输入的间隔 "
                    f"{gap}ms，截短为 {clamped}ms"
                )
                hold = clamped
                wait = max(0, gap - hold)
            else:
                wait = gap - hold
        else:
            wait = 0

        if kind == "mouse":
            device, code = "mouse", "left"
        else:
            device, code = "key", KEY_CODE[event.action]

        stamp = f"@{event.ms / 1000:.3f}s"
        lead_text = f" lead={row.lead_ms}ms" if row.lead_ms else ""
        label = f"{event.action} {stamp}{lead_text}" + (
            f" {event.note}" if event.note else ""
        )
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
    """按 delay_after 语义重建每次按下的宏相对时刻，用于测试与对齐验证。"""
    at = 0
    presses = []
    for step in steps:
        if step.action == "down":
            presses.append(at)
        at += step.delay_after_ms
    return presses


# ---------------------------------------------------------------------------
# 轴节点汇总：按站场角色分组，保留视频视觉 gap


def compile_axis_nodes(timeline: Timeline) -> list[dict]:
    """把时间线按“谁在场上”切成节点；这里的 gap 是视频视觉时序。"""
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
