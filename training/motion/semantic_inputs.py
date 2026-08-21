from __future__ import annotations

import json
from pathlib import Path


SEMANTIC_MAP_VERSION = 1


def semantic_action(device: str, code: str) -> str | None:
    device_norm = str(device or "").strip().casefold()
    code_norm = str(code or "").strip().casefold()

    if device_norm == "mouse" and code_norm in {"left", "button.left"}:
        return "ATTACK"

    if device_norm in {"key", "keyboard"}:
        return {
            "e": "SKILL_E",
            "q": "ECHO_Q",
            "r": "LIBERATION_R",
            "1": "SWAP_1",
            "2": "SWAP_2",
            "3": "SWAP_3",
        }.get(code_norm)

    if device_norm.startswith("gamepad"):
        return {
            "x": "ATTACK",
        }.get(code_norm)

    return None


def load_semantic_events(path: Path, *, action: str | None = None, edge: str | None = None) -> list[dict]:
    events: list[dict] = []
    if not path.is_file():
        return events

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid telemetry JSON at {path}:{line_no}: {exc}") from exc

        semantic = semantic_action(raw.get("device", ""), raw.get("code", ""))
        if semantic is None:
            continue
        event = dict(raw)
        event["semantic"] = semantic

        if action is not None and semantic != action:
            continue
        if edge is not None and str(event.get("action", "")).casefold() != edge.casefold():
            continue
        events.append(event)

    events.sort(key=lambda row: (float(row.get("t_ms", 0.0)), int(row.get("frame", 0))))
    return events


def telemetry_path_for_video(video: Path) -> Path:
    return video.with_suffix(".inputs.jsonl")


def session_path_for_video(video: Path) -> Path:
    return video.with_suffix(".session.json")
