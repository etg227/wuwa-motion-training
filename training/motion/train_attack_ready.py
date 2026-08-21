from __future__ import annotations

import argparse
import bisect
import math
from pathlib import Path

import attack_ready_legacy as legacy
from semantic_inputs import telemetry_path_for_video


def _cycle_has_attack(item: dict, cache: dict) -> bool:
    video = Path(item["video"])
    events, frames = legacy._attack_events(video, cache)
    if not events:
        return False
    start = int(item["start_frame"])
    end = int(item["end_frame"])
    index = bisect.bisect_left(frames, start)
    return index < len(frames) and int(frames[index]) < end


def _mode_profile(root: Path, meta: dict, items: list[dict], cache: dict) -> dict:
    mode_name = meta.get("name", f"mode_{meta['id']}")
    telemetry_file_cycles = sum(
        telemetry_path_for_video(Path(item["video"])).is_file() for item in items
    )
    usable = [item for item in items if _cycle_has_attack(item, cache)]
    ignored = telemetry_file_cycles - len(usable)
    print(
        f"[{mode_name}] ATTACK telemetry: "
        f"input_logs={telemetry_file_cycles} actual_attack_cycles={len(usable)} "
        f"ignored_without_attack={ignored}"
    )

    peaks = legacy._visual_peaks(root, usable)
    minimum_support = max(2, math.ceil(len(usable) * 0.45)) if usable else 2
    windows, accepted = [], []
    for candidate in peaks:
        samples = [
            sample for item in usable
            if (sample := legacy._sample_for_peak(root, item, candidate["phase"], cache)) is not None
        ]
        if len(samples) < minimum_support:
            continue
        window = legacy._window(candidate, samples, len(usable))
        if window is None:
            continue
        if window["transition_spread"] > 0.060:
            continue
        if window["confidence"] < legacy.MIN_WINDOW_CONFIDENCE:
            continue
        windows.append(window)
        accepted.extend(samples)

    windows.sort(key=lambda row: row["transition_phase"])
    ready = len(usable) >= legacy.MIN_MODE_CYCLES and bool(windows)
    return {
        "id": int(meta["id"]),
        "name": mode_name,
        "stable_id": meta.get("stable_id"),
        "ready": ready,
        "raw_cycle_count": len(items),
        "telemetry_file_cycle_count": telemetry_file_cycles,
        "telemetry_cycle_count": len(usable),
        "ignored_without_attack": ignored,
        "minimum_support": minimum_support,
        "visual_candidate_peaks": peaks,
        "window_count": len(windows),
        "windows": windows,
        "probability_bins": legacy.PROFILE_BINS,
        "probabilities": legacy._probabilities(windows),
        "accepted_samples": accepted,
    }


def train_character_attack_ready(character: str) -> dict:
    original = legacy._mode_profile
    legacy._mode_profile = _mode_profile
    try:
        return legacy.train_character_attack_ready(character)
    finally:
        legacy._mode_profile = original


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Learn ATTACK/CHAIN_READY from phase-aligned video + real ATTACK telemetry"
    )
    parser.add_argument("--character", required=True)
    args = parser.parse_args()
    train_character_attack_ready(args.character)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
