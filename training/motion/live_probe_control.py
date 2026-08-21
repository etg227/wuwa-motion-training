from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadyWindowDecision:
    trigger: bool
    armed: bool
    rearmed: bool
    mode_changed: bool
    ready_probability: float
    reason: str


class ReadyWindowGate:
    """One-shot gate for a single high-READY probability island.

    The gate is intentionally conservative for live probing:
    - a high READY island can trigger at most once;
    - blocked/uncertain high islands are consumed instead of triggering late;
    - re-arming requires several clearly-low READY frames;
    - a motion-mode change does not by itself arm or disarm the gate. The current
      armed/disarmed state is preserved so a one-frame classifier mode change
      cannot swallow a valid READY island or create a duplicate trigger.
    """

    def __init__(
        self,
        *,
        enter_threshold: float = 0.85,
        rearm_threshold: float = 0.20,
        rearm_frames: int = 2,
        min_trigger_interval_s: float = 0.18,
    ):
        enter_threshold = float(enter_threshold)
        rearm_threshold = float(rearm_threshold)
        if not 0.0 <= rearm_threshold < enter_threshold <= 1.0:
            raise ValueError("require 0 <= rearm_threshold < enter_threshold <= 1")
        self.enter_threshold = enter_threshold
        self.rearm_threshold = rearm_threshold
        self.rearm_frames = max(1, int(rearm_frames))
        self.min_trigger_interval_s = max(0.0, float(min_trigger_interval_s))
        self.reset()

    def reset(self) -> None:
        self.mode_id: int | None = None
        self.armed = True
        self.low_streak = 0
        self.last_trigger_at = float("-inf")

    def update(
        self,
        mode_id: int,
        ready_probability: float,
        now_s: float,
        *,
        can_trigger: bool = True,
    ) -> ReadyWindowDecision:
        mode_id = int(mode_id)
        ready_probability = max(0.0, min(1.0, float(ready_probability)))
        now_s = float(now_s)

        initial_mode = self.mode_id is None
        mode_changed = self.mode_id is not None and self.mode_id != mode_id
        if initial_mode:
            self.mode_id = mode_id
        elif mode_changed:
            # Raw mode classification can flicker for a frame around visual
            # transitions. Preserve armed/disarmed state across that flicker.
            # Clear only the partial low streak so low evidence from two
            # different mode profiles cannot combine to re-arm a consumed island.
            self.mode_id = mode_id
            self.low_streak = 0

        rearmed = False
        if not self.armed:
            if ready_probability <= self.rearm_threshold:
                self.low_streak += 1
                if self.low_streak >= self.rearm_frames:
                    self.armed = True
                    self.low_streak = 0
                    rearmed = True
            else:
                self.low_streak = 0

        if self.armed and ready_probability >= self.enter_threshold:
            cooldown_ok = (
                now_s - self.last_trigger_at >= self.min_trigger_interval_s
            )
            trigger = bool(can_trigger and cooldown_ok)
            # Consume this high island even when blocked. A late click after the
            # model/burst/cooldown condition recovers is more dangerous than a miss.
            self.armed = False
            self.low_streak = 0
            if trigger:
                self.last_trigger_at = now_s
                reason = "trigger"
            elif not can_trigger:
                reason = "blocked_high_island"
            else:
                reason = "cooldown_high_island"
            return ReadyWindowDecision(
                trigger=trigger,
                armed=self.armed,
                rearmed=rearmed,
                mode_changed=mode_changed,
                ready_probability=ready_probability,
                reason=reason,
            )

        if rearmed:
            reason = "rearmed"
        elif mode_changed:
            reason = "mode_changed_preserve_gate"
        elif self.armed:
            reason = "armed_wait_high"
        else:
            reason = "wait_low_to_rearm"
        return ReadyWindowDecision(
            trigger=False,
            armed=self.armed,
            rearmed=rearmed,
            mode_changed=mode_changed,
            ready_probability=ready_probability,
            reason=reason,
        )
