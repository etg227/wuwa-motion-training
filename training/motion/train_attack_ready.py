from __future__ import annotations

import argparse
import bisect
import math
from pathlib import Path

import numpy as np

import attack_ready_legacy as legacy
from semantic_inputs import telemetry_path_for_video


PROFILE_ACCEPT_MARGIN = 0.006
PROFILE_MIN_RAMP = 0.010
PROFILE_LATE_FADE = 0.015
SELF_COVERAGE_THRESHOLD = 0.85
SELF_COVERAGE_MIN_RATIO = 0.70


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
    """How far an input precedes a visual transition on the short circular arc."""
    return max(0.0, _short_signed_delta(transition, event_phase))


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _profile_bounds(transition: float, samples: list[dict]) -> dict:
    """Build a burst-ready interval from all accepted samples in one visual window."""
    accepted = np.asarray([
        _short_signed_delta(transition, float(row["accepted_phase"]))
        for row in samples
    ], dtype=np.float32)
    opening = np.asarray([
        _short_signed_delta(transition, float(row["opening_phase"]))
        for row in samples
    ], dtype=np.float32)

    accepted_early = float(np.percentile(accepted, 90)) + PROFILE_ACCEPT_MARGIN
    accepted_late = float(np.percentile(accepted, 10)) - PROFILE_ACCEPT_MARGIN
    ready_early = max(PROFILE_MIN_RAMP, accepted_early)
    ready_late = min(0.0, accepted_late)

    opening_early = float(np.percentile(opening, 90)) + PROFILE_ACCEPT_MARGIN
    ramp_early = max(opening_early, ready_early + PROFILE_MIN_RAMP)
    late_end = ready_late - PROFILE_LATE_FADE

    ramp_early = min(ramp_early, 0.25)
    ready_early = min(ready_early, ramp_early - PROFILE_MIN_RAMP)
    ready_late = max(ready_late, -0.12)
    late_end = max(late_end, -0.15)

    return {
        "ramp_early_delta": float(ramp_early),
        "ready_early_delta": float(ready_early),
        "ready_late_delta": float(ready_late),
        "late_end_delta": float(late_end),
        "accepted_delta_p10": float(np.percentile(accepted, 10)),
        "accepted_delta_median": float(np.median(accepted)),
        "accepted_delta_p90": float(np.percentile(accepted, 90)),
        "opening_delta_p90": float(np.percentile(opening, 90)),
    }


def _window(candidate: dict, samples: list[dict], cycle_count: int) -> dict | None:
    if not samples:
        return None

    transition = legacy._anchored_median(
        [row["transition_phase"] for row in samples], candidate["phase"]
    )
    spread = float(np.median([
        legacy._circ_dist(row["transition_phase"], transition) for row in samples
    ]))

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

    bounds = _profile_bounds(transition, samples)
    return {
        "start_phase": float((transition - bounds["ramp_early_delta"]) % 1.0),
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
        "lead_aggregation": "short-signed-causal-v3",
        "profile": {
            "architecture": "accepted-distribution-burst-v2",
            **bounds,
        },
    }


def _probabilities(windows: list[dict], bins: int = legacy.PROFILE_BINS) -> list[float]:
    """Create a robust burst-ready curve from empirical accepted distributions."""
    values = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        phase = index / bins
        probability = 0.0
        for window in windows:
            profile = window.get("profile", {})
            if not profile:
                continue
            transition = float(window["transition_phase"])
            delta = _short_signed_delta(transition, phase)
            ramp_early = float(profile["ramp_early_delta"])
            ready_early = float(profile["ready_early_delta"])
            ready_late = float(profile["ready_late_delta"])
            late_end = float(profile["late_end_delta"])

            if delta > ramp_early or delta < late_end:
                current = 0.0
            elif delta > ready_early:
                width = max(1e-6, ramp_early - ready_early)
                current = _smoothstep((ramp_early - delta) / width)
            elif delta >= ready_late:
                current = 1.0
            else:
                width = max(1e-6, ready_late - late_end)
                current = _smoothstep((delta - late_end) / width)
            probability = max(probability, current)
        values[index] = probability

    return legacy._smooth(values, radius=1).clip(0.0, 1.0).tolist()


def _profile_probability(values: list[float], phase: float) -> float:
    if not values:
        return 0.0
    count = len(values)
    position = (phase % 1.0) * count
    left = int(math.floor(position)) % count
    frac = position - math.floor(position)
    right = (left + 1) % count
    return float((1.0 - frac) * float(values[left]) + frac * float(values[right]))


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
    probabilities = _probabilities(windows)
    self_ready = [
        _profile_probability(probabilities, float(row["accepted_phase"]))
        for row in accepted
    ]
    self_median = float(np.median(self_ready)) if self_ready else 0.0
    self_ratio = (
        float(np.mean(np.asarray(self_ready) >= SELF_COVERAGE_THRESHOLD))
        if self_ready else 0.0
    )
    self_coverage_ready = bool(
        self_median >= SELF_COVERAGE_THRESHOLD
        and self_ratio >= SELF_COVERAGE_MIN_RATIO
    )
    ready = bool(
        len(usable) >= legacy.MIN_MODE_CYCLES
        and windows
        and self_coverage_ready
    )
    print(
        f"[{mode_name}] READY self-coverage: median={self_median:.3f} "
        f">={SELF_COVERAGE_THRESHOLD:.2f}: {self_ratio:.3f} "
        f"quality={'PASS' if self_coverage_ready else 'REJECT'}"
    )

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
        "probabilities": probabilities,
        "accepted_samples": accepted,
        "profile_architecture": "AcceptedDistributionBurstProfile-v2",
        "self_coverage": {
            "sample_count": len(self_ready),
            "median_ready": self_median,
            "threshold": SELF_COVERAGE_THRESHOLD,
            "ge_threshold_ratio": self_ratio,
            "minimum_ratio": SELF_COVERAGE_MIN_RATIO,
            "quality": "ready" if self_coverage_ready else "low_self_coverage",
        },
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
