from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_OBSERVATION_GAIN = 0.48
DEFAULT_SOFT_GATE = 0.10
DEFAULT_HARD_GATE = 0.22
DEFAULT_MAX_CORRECTION = 0.055
DEFAULT_REANCHOR_AFTER = 4


def signed_circular_delta(target: float, reference: float) -> float:
    """Shortest signed delta reference -> target in [-0.5, 0.5)."""
    return float(((target - reference + 0.5) % 1.0) - 0.5)


@dataclass(frozen=True)
class PhaseTrackResult:
    phase: float
    predicted_phase: float
    raw_phase: float
    residual: float
    observation_weight: float
    rejected: bool
    reanchored: bool


class CircularPhaseTracker:
    """Online phase-locked tracker for cyclic character motion.

    PhaseNet is treated as a visual observation rather than the complete state.
    Between observations the tracker advances using the learned median cycle
    duration of the current motion mode. Small visual residuals correct the state;
    isolated large jumps are rejected so one bad frame cannot move READY across a
    combo. Repeated disagreement eventually re-anchors, allowing recovery after a
    real state discontinuity instead of staying permanently locked to stale state.
    """

    def __init__(
        self,
        mode_durations_s: dict[int, float],
        *,
        observation_gain: float = DEFAULT_OBSERVATION_GAIN,
        soft_gate: float = DEFAULT_SOFT_GATE,
        hard_gate: float = DEFAULT_HARD_GATE,
        max_correction: float = DEFAULT_MAX_CORRECTION,
        reanchor_after: int = DEFAULT_REANCHOR_AFTER,
    ):
        self.mode_durations_s = {
            int(mode): max(1e-3, float(duration))
            for mode, duration in mode_durations_s.items()
        }
        self.observation_gain = float(np.clip(observation_gain, 0.0, 1.0))
        self.soft_gate = max(1e-4, float(soft_gate))
        self.hard_gate = max(self.soft_gate + 1e-4, float(hard_gate))
        self.max_correction = max(0.0, float(max_correction))
        self.reanchor_after = max(1, int(reanchor_after))
        self.reset()

    def reset(self) -> None:
        self.mode_id: int | None = None
        self.unwrapped_phase: float | None = None
        self.reject_streak = 0

    def _duration(self, mode_id: int) -> float:
        value = self.mode_durations_s.get(int(mode_id))
        if value is not None:
            return value
        if self.mode_durations_s:
            return float(np.median(list(self.mode_durations_s.values())))
        return 1.5

    def update(
        self,
        mode_id: int,
        raw_phase: float,
        dt_s: float,
        *,
        mode_confidence: float = 1.0,
    ) -> PhaseTrackResult:
        mode_id = int(mode_id)
        raw_phase = float(raw_phase % 1.0)
        dt_s = max(0.0, float(dt_s))

        if self.unwrapped_phase is None or self.mode_id != mode_id:
            self.mode_id = mode_id
            self.unwrapped_phase = raw_phase
            self.reject_streak = 0
            return PhaseTrackResult(
                phase=raw_phase,
                predicted_phase=raw_phase,
                raw_phase=raw_phase,
                residual=0.0,
                observation_weight=1.0,
                rejected=False,
                reanchored=True,
            )

        duration = self._duration(mode_id)
        predicted_unwrapped = self.unwrapped_phase + dt_s / duration
        predicted_phase = predicted_unwrapped % 1.0
        residual = signed_circular_delta(raw_phase, predicted_phase)
        absolute = abs(residual)

        if absolute <= self.soft_gate:
            gate_weight = 1.0
        elif absolute < self.hard_gate:
            gate_weight = (self.hard_gate - absolute) / (self.hard_gate - self.soft_gate)
            gate_weight *= 0.35
        else:
            gate_weight = 0.0

        confidence_weight = float(np.clip(mode_confidence, 0.45, 1.0))
        observation_weight = self.observation_gain * gate_weight * confidence_weight
        rejected = observation_weight <= 1e-6

        if rejected:
            self.reject_streak += 1
        else:
            self.reject_streak = 0

        if rejected and self.reject_streak >= self.reanchor_after:
            # Several consecutive frames disagree with the motion clock. This is
            # more likely a true state discontinuity than four independent visual
            # outliers, so re-anchor to the nearest unwrapped representation.
            nearest_raw = predicted_unwrapped + residual
            self.unwrapped_phase = nearest_raw
            self.reject_streak = 0
            return PhaseTrackResult(
                phase=float(nearest_raw % 1.0),
                predicted_phase=float(predicted_phase),
                raw_phase=raw_phase,
                residual=float(residual),
                observation_weight=1.0,
                rejected=False,
                reanchored=True,
            )

        correction = float(np.clip(
            residual * observation_weight,
            -self.max_correction,
            self.max_correction,
        ))
        self.unwrapped_phase = predicted_unwrapped + correction
        tracked = float(self.unwrapped_phase % 1.0)
        return PhaseTrackResult(
            phase=tracked,
            predicted_phase=float(predicted_phase),
            raw_phase=raw_phase,
            residual=float(residual),
            observation_weight=float(observation_weight),
            rejected=rejected,
            reanchored=False,
        )
