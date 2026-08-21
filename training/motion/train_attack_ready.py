from __future__ import annotations

import argparse
import bisect
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch

import phase_alignment
from common import character_root, read_json, write_json
from semantic_inputs import (
    SEMANTIC_MAP_VERSION,
    load_semantic_events,
    session_path_for_video,
    telemetry_path_for_video,
)

PROFILE_BINS = 128
MIN_MODE_CYCLES = 3
MIN_WINDOW_CONFIDENCE = 0.38
MAX_ACCEPT_LEAD_MS = 260.0
MAX_PREVIOUS_ATTACK_GAP_MS = 260.0


def _circ_dist(a: float, b: float) -> float:
    d = abs((a - b) % 1.0)
    return min(d, 1.0 - d)


def _anchored_median(values: list[float], anchor: float) -> float:
    if not values:
        return anchor % 1.0
    rel = [((value - anchor + 0.5) % 1.0) - 0.5 for value in values]
    return float((anchor + float(np.median(rel))) % 1.0)


def _smooth(values: np.ndarray, radius: int = 3) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    sigma = max(1.0, radius / 1.6)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    padded = np.concatenate((values[-radius:], values, values[:radius]))
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _motion_evidence(frames: np.ndarray) -> np.ndarray:
    if len(frames) < 3:
        return np.zeros(len(frames), dtype=np.float32)
    features = np.stack([phase_alignment._frame_feature(frame, size=24) for frame in frames])
    diff = np.zeros(len(features), dtype=np.float32)
    diff[1:] = np.mean(np.abs(features[1:] - features[:-1]), axis=1)
    median = float(np.median(diff[1:]))
    mad = float(np.median(np.abs(diff[1:] - median)))
    scale = max(1.4826 * mad, median * 0.18, 1e-5)
    velocity = np.clip((diff - median) / scale, 0.0, 8.0)
    accel = np.zeros_like(diff)
    accel[1:] = np.clip(diff[1:] - diff[:-1], 0.0, None) / scale
    return (velocity + 0.40 * np.clip(accel, 0.0, 8.0)).astype(np.float32)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _video_fps(video: Path) -> float:
    session = session_path_for_video(video)
    if session.is_file():
        try:
            fps = float(read_json(session).get("capture_fps", 0.0))
            if fps > 0:
                return fps
        except Exception:
            pass
    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return fps if fps > 0 else 30.0


def _mode_manifest(root: Path, meta: dict) -> Path:
    value = meta.get("manifest")
    if isinstance(value, str) and value:
        return root / value
    return root / "modes" / f"mode_{int(meta['id'])}" / "manifest.jsonl"


def _cycle_motion_bins(root: Path, item: dict, bins: int = PROFILE_BINS) -> np.ndarray:
    with np.load(root / item["cycle"]) as data:
        frames = data["frames"].copy()
        phases = data["phases"].copy()
    evidence = _motion_evidence(frames)
    offset = float(item.get("phase_offset", 0.0))
    values = np.zeros(bins, dtype=np.float32)
    counts = np.zeros(bins, dtype=np.float32)
    for phase, score in zip(phases, evidence):
        p = ((float(phase) + offset) % 1.0) * bins
        left = int(math.floor(p)) % bins
        frac = p - math.floor(p)
        for index, weight in ((left, 1.0 - frac), ((left + 1) % bins, frac)):
            values[index] += float(score) * weight
            counts[index] += weight
    return values / np.maximum(counts, 1e-5)


def _visual_peaks(root: Path, items: list[dict]) -> list[dict]:
    rows = [_cycle_motion_bins(root, item) for item in items]
    if not rows:
        return []
    profile = _smooth(np.median(np.stack(rows), axis=0), radius=3)
    threshold = float(np.percentile(profile, 62))
    candidates = []
    radius = max(3, round(len(profile) * 0.045))
    for index, value in enumerate(profile):
        if value < threshold:
            continue
        if value < profile[(index - 1) % len(profile)] or value < profile[(index + 1) % len(profile)]:
            continue
        neighbors = [
            float(profile[(index + delta) % len(profile)])
            for delta in range(-radius, radius + 1)
            if abs(delta) > 1
        ]
        floor = float(np.percentile(neighbors, 30)) if neighbors else threshold
        prominence = max(0.0, float(value) - floor)
        candidates.append({
            "phase": index / len(profile),
            "value": float(value),
            "prominence": prominence,
            "rank": float(value) + prominence * 0.8,
        })
    candidates.sort(key=lambda row: row["rank"], reverse=True)
    strongest = float(candidates[0]["rank"]) if candidates else 0.0
    selected = []
    for row in candidates:
        if strongest and row["rank"] < strongest * 0.16:
            continue
        if any(_circ_dist(row["phase"], old["phase"]) < 0.10 for old in selected):
            continue
        selected.append(row)
        if len(selected) >= 7:
            break
    return sorted(selected, key=lambda row: row["phase"])


def _attack_events(video: Path, cache: dict):
    key = str(video)
    if key not in cache:
        events = load_semantic_events(
            telemetry_path_for_video(video), action="ATTACK", edge="down"
        )
        cache[key] = (events, [int(row.get("frame", 0)) for row in events])
    return cache[key]


def _refine_peak(root: Path, item: dict, global_phase: float) -> tuple[float, float]:
    with np.load(root / item["cycle"]) as data:
        frames = data["frames"].copy()
        phases = data["phases"].copy()
    evidence = _motion_evidence(frames)
    offset = float(item.get("phase_offset", 0.0))
    target = (global_phase - offset) % 1.0
    center = int(np.argmin([_circ_dist(float(p), target) for p in phases]))
    radius = max(2, round(len(frames) * 0.035))
    indexes = [(center + delta) % len(frames) for delta in range(-radius, radius + 1)]
    best = max(indexes, key=lambda index: float(evidence[index]))
    return (float(phases[best]) + offset) % 1.0, float(evidence[best])


def _sample_for_peak(root: Path, item: dict, candidate_phase: float, cache: dict) -> dict | None:
    video = Path(item["video"])
    events, frames = _attack_events(video, cache)
    if not events:
        return None

    start, end = int(item["start_frame"]), int(item["end_frame"])
    if end <= start + 2:
        return None
    fps = _video_fps(video)
    offset = float(item.get("phase_offset", 0.0))
    transition, motion_score = _refine_peak(root, item, candidate_phase)
    local_transition = (transition - offset) % 1.0
    transition_frame = start + local_transition * (end - start)

    index = bisect.bisect_right(frames, transition_frame + 1.0) - 1
    if index < 0:
        return None
    accepted = events[index]
    accepted_frame = float(accepted.get("frame", 0))
    lead_ms = (transition_frame - accepted_frame) / fps * 1000.0
    if lead_ms < -50.0 or lead_ms > MAX_ACCEPT_LEAD_MS:
        return None

    previous_frame = None
    previous_gap = None
    if index > 0:
        prior = float(events[index - 1].get("frame", 0))
        gap = (accepted_frame - prior) / fps * 1000.0
        if 0.0 < gap <= MAX_PREVIOUS_ATTACK_GAP_MS:
            previous_frame, previous_gap = prior, gap

    accepted_phase = (
        (accepted_frame - start) / max(1.0, end - start) + offset
    ) % 1.0
    if previous_frame is not None:
        opening_frame = (previous_frame + accepted_frame) * 0.5
        opening_phase = (
            (opening_frame - start) / max(1.0, end - start) + offset
        ) % 1.0
    else:
        opening_phase = (accepted_phase - 0.025) % 1.0

    return {
        "cycle": str(item["cycle"]),
        "video": str(video),
        "transition_phase": float(transition),
        "accepted_phase": float(accepted_phase),
        "opening_phase": float(opening_phase),
        "accepted_frame": int(round(accepted_frame)),
        "accepted_t_ms": float(accepted.get("t_ms", 0.0)),
        "previous_attack_frame": (
            int(round(previous_frame)) if previous_frame is not None else None
        ),
        "previous_gap_ms": previous_gap,
        "lead_ms": float(max(0.0, lead_ms)),
        "motion_score": motion_score,
    }


def _window(candidate: dict, samples: list[dict], cycle_count: int) -> dict | None:
    if not samples:
        return None
    transition = _anchored_median(
        [row["transition_phase"] for row in samples], candidate["phase"]
    )
    spread = float(np.median([
        _circ_dist(row["transition_phase"], transition) for row in samples
    ]))
    accepted_lead = float(np.median([
        (row["transition_phase"] - row["accepted_phase"]) % 1.0 for row in samples
    ]))
    opening_lead = float(np.median([
        (row["transition_phase"] - row["opening_phase"]) % 1.0 for row in samples
    ]))
    opening_lead = max(opening_lead, accepted_lead + 0.008)
    lead_ms = np.asarray([row["lead_ms"] for row in samples], dtype=np.float32)
    lead_mad = float(np.median(np.abs(lead_ms - np.median(lead_ms))))
    bounded = sum(row["previous_attack_frame"] is not None for row in samples)
    support = len(samples) / max(1, cycle_count)
    prominence = candidate["prominence"] / max(candidate["value"], 1e-5)
    confidence = (
        0.42 * min(1.0, support)
        + 0.22 * min(1.0, prominence * 3.0)
        + 0.18 * max(0.0, 1.0 - spread / 0.055)
        + 0.10 * max(0.0, 1.0 - lead_mad / 115.0)
        + 0.08 * bounded / max(1, len(samples))
    )
    return {
        "start_phase": float((transition - opening_lead) % 1.0),
        "accepted_phase": float((transition - accepted_lead) % 1.0),
        "transition_phase": float(transition),
        "support_cycles": len(samples),
        "total_cycles": cycle_count,
        "support_ratio": float(support),
        "median_lead_ms": float(np.median(lead_ms)),
        "lead_mad_ms": lead_mad,
        "transition_spread": spread,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "samples": samples,
    }


def _probabilities(windows: list[dict], bins: int = PROFILE_BINS) -> list[float]:
    values = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        phase = index / bins
        for window in windows:
            start = window["start_phase"]
            end = window["transition_phase"]
            total = (end - start) % 1.0
            progress = (phase - start) % 1.0
            if total < 1e-6 or progress > total:
                continue
            accepted = (window["accepted_phase"] - start) % 1.0
            if accepted <= 1e-6 or progress >= accepted:
                probability = 1.0
            else:
                x = progress / accepted
                probability = x * x * (3.0 - 2.0 * x)
            values[index] = max(values[index], float(probability))
    return _smooth(values, radius=2).clip(0.0, 1.0).tolist()


def _mode_profile(root: Path, meta: dict, items: list[dict], cache: dict) -> dict:
    usable = [
        item for item in items
        if telemetry_path_for_video(Path(item["video"])).is_file()
    ]
    peaks = _visual_peaks(root, usable)
    minimum_support = max(2, math.ceil(len(usable) * 0.45))
    windows, accepted = [], []
    for candidate in peaks:
        samples = [
            sample for item in usable
            if (sample := _sample_for_peak(root, item, candidate["phase"], cache)) is not None
        ]
        if len(samples) < minimum_support:
            continue
        window = _window(candidate, samples, len(usable))
        if window is None:
            continue
        if window["transition_spread"] > 0.060:
            continue
        if window["confidence"] < MIN_WINDOW_CONFIDENCE:
            continue
        windows.append(window)
        accepted.extend(samples)

    windows.sort(key=lambda row: row["transition_phase"])
    ready = len(usable) >= MIN_MODE_CYCLES and bool(windows)
    return {
        "id": int(meta["id"]),
        "name": meta.get("name", f"mode_{meta['id']}"),
        "stable_id": meta.get("stable_id"),
        "ready": ready,
        "raw_cycle_count": len(items),
        "telemetry_cycle_count": len(usable),
        "visual_candidate_peaks": peaks,
        "window_count": len(windows),
        "windows": windows,
        "probability_bins": PROFILE_BINS,
        "probabilities": _probabilities(windows),
        "accepted_samples": accepted,
    }


def _promote(candidate: Path, formal: Path) -> None:
    formal.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate, formal)


def _quarantine_previous(formal: Path) -> None:
    if not formal.is_file():
        return
    target = formal.with_name(f"{formal.stem}.previous_ready{formal.suffix}")
    if target.exists():
        target.unlink()
    os.replace(formal, target)
    print(f"quarantined previous ATTACK READY model -> {target}")


def train_character_attack_ready(character: str) -> dict:
    root = character_root(character)
    index_path = root / "modes" / "index.json"
    if not index_path.is_file():
        raise RuntimeError(f"missing {index_path}; train the phase model first")
    mode_index = read_json(index_path)
    ready_dir = root / "ready"
    ready_dir.mkdir(parents=True, exist_ok=True)

    if not bool(mode_index.get("router_ready")):
        result = {
            "schema": 1,
            "character": character,
            "semantic_map_version": SEMANTIC_MAP_VERSION,
            "ready_model_ready": False,
            "reason": "phase_router_not_ready",
            "modes": [],
        }
        write_json(ready_dir / "attack_ready.json", result)
        print("ATTACK READY skipped: phase router is not ready.")
        return result

    cache, profiles, accepted_rows = {}, [], []
    for meta in mode_index.get("modes", []):
        profile = _mode_profile(root, meta, _load_jsonl(_mode_manifest(root, meta)), cache)
        profiles.append(profile)
        accepted_rows.extend({
            **row, "mode_id": profile["id"], "stable_id": profile.get("stable_id")
        } for row in profile["accepted_samples"])
        print(
            f"[{profile['name']}] ATTACK READY: "
            f"telemetry_cycles={profile['telemetry_cycle_count']} "
            f"windows={profile['window_count']} ready={profile['ready']}"
        )
        for number, window in enumerate(profile["windows"], start=1):
            print(
                f"  window {number}: start={window['start_phase']:.3f} "
                f"accepted≈{window['accepted_phase']:.3f} "
                f"transition={window['transition_phase']:.3f} "
                f"support={window['support_cycles']}/{window['total_cycles']} "
                f"lead={window['median_lead_ms']:.1f}ms "
                f"confidence={window['confidence']:.3f}"
            )

    ready = bool(profiles) and all(profile["ready"] for profile in profiles)
    result = {
        "schema": 1,
        "architecture": "PhaseAttackReadyProfile-v1",
        "character": character,
        "semantic_map_version": SEMANTIC_MAP_VERSION,
        "phase_router_ready": True,
        "ready_model_ready": ready,
        "action": "ATTACK",
        "target": "CHAIN_READY",
        "notes": (
            "Pseudo-supervised: the last human ATTACK down before a repeatable visual "
            "transition is treated as the accepted candidate; the prior spam ATTACK "
            "bounds the READY opening."
        ),
        "modes": [
            {key: value for key, value in profile.items() if key != "accepted_samples"}
            for profile in profiles
        ],
    }
    write_json(ready_dir / "attack_ready.json", result)
    with (ready_dir / "accepted_attack_samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in accepted_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    candidate = root / "models" / "attack_ready.candidate.pt"
    formal = root / "models" / "attack_ready.pt"
    torch.save(result, candidate)
    if ready:
        _promote(candidate, formal)
        print(f"ATTACK READY model READY -> {formal}")
    else:
        _quarantine_previous(formal)
        print(f"ATTACK READY model not ready; candidate kept at {candidate}")
    print(
        f"ATTACK READY summary: modes={len(profiles)} "
        f"ready_modes={sum(bool(p['ready']) for p in profiles)} "
        f"accepted_samples={len(accepted_rows)} ready={ready}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Learn ATTACK/CHAIN_READY windows from phase-aligned video + human telemetry"
    )
    parser.add_argument("--character", required=True)
    args = parser.parse_args()
    train_character_attack_ready(args.character)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
