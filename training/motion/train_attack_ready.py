from __future__ import annotations

import argparse
import bisect
import math
from pathlib import Path

import numpy as np

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


def _short_signed_delta(later: float, earlier: float) -> float:
    """Shortest signed circular delta from earlier -> later in [-0.5, 0.5)."""
    return float(((later - earlier + 0.5) % 1.0) - 0.5)


def _causal_phase_lead(transition: float, event_phase: float) -> float:
    """How far an input precedes a visual transition on the short circular arc.

    Video/input timestamps are quantized to frames. If the input lands one frame
    after the visual transition, treat that small negative delta as 0 rather than
    wrapping it to ~1.0 cycle.
    """
    return max(0.0, _short_signed_delta(transition, event_phase))


def _window(candidate: dict, samples: list[dict], cycle_count: int) -> dict | None:
    if not samples:
        return None

    transition = legacy._anchored_median(
        [row["transition_phase"] for row in samples], candidate["phase"]
    )
    spread = float(np.median([
        legacy._circ_dist(row["transition_phase"], transition) for row in samples
    ]))

    # IMPORTANT: do not use `(transition - event) % 1.0` here. Around phase 0,
    # a harmless -1 frame timestamp ordering would become ~0.98 and create a
    # READY window spanning almost the entire combo cycle.
    accepted_leads = [
        _causal_phase_lead(row["transition_phase"], row["accepted_phase"])
        for row in samples
    ]
    opening_leads = [
        _causal_phase_lead(row["transition_phase"], row["opening_phase"])
        for row in samples
    ]
    accepted_lead = float(np.median(accepted_leads))
    opening_lead = float(np.median(opening_leads))

    # The opening bound must precede the accepted point by at least one small
    # phase bin so the probability curve has a finite ramp even when input and
    # transition are recorded in the same 30-FPS frame.
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
        "lead_aggregation": "short-signed-causal-v2",
    }


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
        window = _window(candidate, samples, len(usable))
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
