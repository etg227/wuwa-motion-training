from __future__ import annotations

import json
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "training_data" / "motion"
DEFAULT_ROI = (0.12, 0.04, 0.76, 0.92)  # x, y, w, h, normalized


def safe_character(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    if not value:
        raise ValueError("character must not be empty")
    return value


def character_root(character: str, create: bool = True) -> Path:
    root = DATA_ROOT / safe_character(character)
    if create:
        for child in ("videos", "annotations", "cycles", "models"):
            (root / child).mkdir(parents=True, exist_ok=True)
    return root


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"invalid video metadata: fps={fps}, frames={frames}")
    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration_ms": round(frames / fps * 1000),
    }


def parse_roi(value: str | None):
    if not value:
        return DEFAULT_ROI
    parts = tuple(float(item.strip()) for item in value.split(","))
    if len(parts) != 4:
        raise ValueError("ROI format must be x,y,w,h")
    x, y, w, h = parts
    if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x + w <= 1 and y + h <= 1):
        raise ValueError("ROI must stay inside normalized 0..1 frame coordinates")
    return parts


def crop_roi(frame, roi):
    height, width = frame.shape[:2]
    x, y, w, h = roi
    left = max(0, min(width - 1, round(x * width)))
    top = max(0, min(height - 1, round(y * height)))
    right = max(left + 1, min(width, round((x + w) * width)))
    bottom = max(top + 1, min(height, round((y + h) * height)))
    return frame[top:bottom, left:right]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
