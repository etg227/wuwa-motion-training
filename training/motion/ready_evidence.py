from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


DEFAULT_MEMORY_FRAMES = 4
DEFAULT_DECAY = 0.72
DEFAULT_ENTER_THRESHOLD = 0.85
DEFAULT_EXIT_THRESHOLD = 0.45
DEFAULT_EXIT_FRAMES = 2


@dataclass(frozen=True)
class ReadyEvidenceResult:
    mode_id: int
    raw_ready: float
    evidence: float
    ready: bool
    entered: bool
    exited: bool
    low_streak: int


def combine_ready_evidence(
    values: list[float] | tuple[float, ...],
    *,
    decay: float = DEFAULT_DECAY,
) -> float:
    """Causally combine recent READY probabilities without looking into the future.

    The newest observation keeps full weight; older observations decay
    exponentially. Independent moderate observations can accumulate enough
    evidence to cross the READY threshold, while isolated low noise cannot.
    """
    if not values:
        return 0.0
    decay = float(np.clip(decay, 0.0, 1.0))
    miss_probability = 1.0
    for age, raw in enumerate(reversed(values)):
        probability = float(np.clip(raw, 0.0, 1.0))
        weighted = probability * (decay ** age)
        miss_probability *= 1.0 - weighted
    return float(np.clip(1.0 - miss_probability, 0.0, 1.0))


class ReadyEvidenceTracker:
    """Causal READY evidence accumulator with asymmetric hysteresis.

    This tracker intentionally does not modify phase. It consumes the READY
    probability produced by the existing phase/profile stack and answers the
    execution-level question: "is there enough recent evidence to start/keep a
    short ATTACK burst?"

    READY turns on when accumulated evidence crosses the high enter threshold.
    Once on, a brief probability dip does not immediately turn it off; evidence
    must remain below the lower exit threshold for several frames.
    """

    def __init__(
        self,
        *,
        memory_frames: int = DEFAULT_MEMORY_FRAMES,
        decay: float = DEFAULT_DECAY,
        enter_threshold: float = DEFAULT_ENTER_THRESHOLD,
        exit_threshold: float = DEFAULT_EXIT_THRESHOLD,
        exit_frames: int = DEFAULT_EXIT_FRAMES,
    ):
        self.memory_frames = max(1, int(memory_frames))
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.enter_threshold = float(np.clip(enter_threshold, 0.0, 1.0))
        self.exit_threshold = float(np.clip(exit_threshold, 0.0, 1.0))
        if self.exit_threshold >= self.enter_threshold:
            raise ValueError("exit_threshold must be lower than enter_threshold")
        self.exit_frames = max(1, int(exit_frames))
        self.reset()

    def reset(self) -> None:
        self.mode_id: int | None = None
        self.history: deque[float] = deque(maxlen=self.memory_frames)
        self.ready = False
        self.low_streak = 0

    def update(self, mode_id: int, raw_ready: float) -> ReadyEvidenceResult:
        mode_id = int(mode_id)
        raw_ready = float(np.clip(raw_ready, 0.0, 1.0))

        # A motion-mode switch changes the meaning of phase/profile entirely, so
        # do not carry stale evidence from the previous form into the new one.
        if self.mode_id != mode_id:
            self.mode_id = mode_id
            self.history.clear()
            self.ready = False
            self.low_streak = 0

        self.history.append(raw_ready)
        evidence = combine_ready_evidence(list(self.history), decay=self.decay)

        entered = False
        exited = False
        if not self.ready:
            if evidence >= self.enter_threshold:
                self.ready = True
                self.low_streak = 0
                entered = True
        else:
            if evidence < self.exit_threshold:
                self.low_streak += 1
                if self.low_streak >= self.exit_frames:
                    self.ready = False
                    self.low_streak = 0
                    exited = True
            else:
                self.low_streak = 0

        return ReadyEvidenceResult(
            mode_id=mode_id,
            raw_ready=raw_ready,
            evidence=evidence,
            ready=bool(self.ready),
            entered=entered,
            exited=exited,
            low_streak=int(self.low_streak),
        )
