from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import mss
import numpy as np
import pydirectinput
import torch
import win32gui
from pynput import keyboard

import replay_validate as replay
from auto_train import GAME_PROCESS, find_game_window, game_client_rect
from common import DEFAULT_ROI, character_root
from live_probe_control import ReadyWindowGate


class LiveMotionRuntime:
    """Single-frame causal wrapper around the trained mode/phase/READY stack."""

    def __init__(self, root: Path, device: torch.device, roi):
        (
            _mode_index,
            _modes,
            self.classifier,
            self.classifier_window,
            self.phase_models,
            self.phase_windows,
            self.ready_profiles,
            self.image_size,
        ) = replay._load_models(root, device)
        self.device = device
        self.roi = tuple(float(value) for value in roi)
        self.max_window = max(
            [self.classifier_window, *self.phase_windows.values()]
        )
        self.history: deque[np.ndarray] = deque(maxlen=self.max_window)
        self.seen_frames = 0

    def reset(self) -> None:
        self.history.clear()
        self.seen_frames = 0

    def infer(self, frame_bgr: np.ndarray) -> dict | None:
        started = time.perf_counter()
        small = replay._model_input(frame_bgr, self.roi, self.image_size)
        self.history.append(small)
        self.seen_frames += 1
        if self.seen_frames < self.max_window:
            return None

        clips = np.stack(self.history)[None, ...]
        with torch.inference_mode():
            if self.classifier is None:
                mode_id = 0
                mode_confidence = 1.0
            else:
                logits = self.classifier(
                    replay._clip_tensor(
                        clips,
                        self.classifier_window,
                        self.device,
                    )
                )
                probabilities = torch.softmax(logits, dim=1)
                mode_confidence_t, mode_id_t = torch.max(probabilities, dim=1)
                mode_id = int(mode_id_t.item())
                mode_confidence = float(mode_confidence_t.item())

            model = self.phase_models.get(mode_id)
            if model is None:
                return None
            vector = model(
                replay._clip_tensor(
                    clips,
                    self.phase_windows[mode_id],
                    self.device,
                )
            )
            phase = float(replay.core.vector_phase(vector).item() % 1.0)

        ready = float(
            np.clip(
                replay._ready_probability(
                    self.ready_profiles.get(mode_id),
                    phase,
                ),
                0.0,
                1.0,
            )
        )
        return {
            "mode": mode_id,
            "mode_confidence": mode_confidence,
            "phase": phase,
            "chain_ready": ready,
            "inference_ms": (time.perf_counter() - started) * 1000.0,
        }


class BurstExecutor:
    """Short PyDirect mouse burst with foreground and emergency-stop checks."""

    def __init__(
        self,
        *,
        stop_event: threading.Event,
        enabled_event: threading.Event,
        foreground_check,
        clicks: int,
        down_ms: float,
        gap_ms: float,
        dry_run: bool,
    ):
        self.stop_event = stop_event
        self.enabled_event = enabled_event
        self.foreground_check = foreground_check
        self.clicks = max(1, int(clicks))
        self.down_s = max(0.001, float(down_ms) / 1000.0)
        self.gap_s = max(0.0, float(gap_ms) / 1000.0)
        self.dry_run = bool(dry_run)
        self._state_lock = threading.Lock()
        self._busy = False
        self._mouse_down = False
        pydirectinput.PAUSE = 0

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return self._busy

    def _can_continue(self) -> bool:
        return bool(
            not self.stop_event.is_set()
            and self.enabled_event.is_set()
            and self.foreground_check()
        )

    def _interruptible_sleep(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            if not self._can_continue():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(0.005, remaining))

    def _mouse_up_safely(self) -> None:
        if self.dry_run:
            with self._state_lock:
                self._mouse_down = False
            return
        try:
            pydirectinput.mouseUp(button="left")
        except Exception:
            pass
        with self._state_lock:
            self._mouse_down = False

    def release(self) -> None:
        self._mouse_up_safely()

    def trigger(self, label: str) -> bool:
        with self._state_lock:
            if self._busy:
                return False
            self._busy = True
        thread = threading.Thread(
            target=self._run,
            args=(label,),
            name="ready-probe-burst",
            daemon=True,
        )
        thread.start()
        return True

    def _run(self, label: str) -> None:
        try:
            if self.dry_run:
                print(f"[BURST dry-run] {label}")
            for index in range(self.clicks):
                if not self._can_continue():
                    break

                if not self.dry_run:
                    pydirectinput.mouseDown(button="left")
                with self._state_lock:
                    self._mouse_down = True

                if not self._interruptible_sleep(self.down_s):
                    break
                self._mouse_up_safely()

                if index + 1 < self.clicks:
                    if not self._interruptible_sleep(self.gap_s):
                        break
        except Exception as exc:
            print(f"[BURST ERROR] {exc}")
        finally:
            self._mouse_up_safely()
            with self._state_lock:
                self._busy = False


def _select_device(value: str) -> torch.device:
    selected = value
    if selected == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(selected)


def _capture_client(screen: mss.mss, hwnd: int) -> np.ndarray | None:
    left, top, right, bottom = game_client_rect(hwnd)
    width = int(right - left)
    height = int(bottom - top)
    if width < 320 or height < 180:
        return None
    grabbed = screen.grab(
        {"left": left, "top": top, "width": width, "height": height}
    )
    # mss returns BGRA; replay._model_input expects BGR.
    frame = np.asarray(grabbed, dtype=np.uint8)[:, :, :3]
    return np.ascontiguousarray(frame)


def main() -> int:
    if sys.platform != "win32":
        raise SystemExit("live_ready_probe.py 仅支持 Windows。")

    parser = argparse.ArgumentParser(
        description=(
            "Controlled live probe: trained mode/phase -> raw CHAIN_READY -> "
            "short PyDirect ATTACK burst"
        )
    )
    parser.add_argument("--character", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ready-threshold", type=float, default=0.85)
    parser.add_argument("--rearm-threshold", type=float, default=0.20)
    parser.add_argument("--rearm-frames", type=int, default=2)
    parser.add_argument("--min-mode-confidence", type=float, default=0.85)
    parser.add_argument("--min-trigger-interval-ms", type=float, default=180.0)
    parser.add_argument("--burst-clicks", type=int, default=2)
    parser.add_argument("--mouse-down-ms", type=float, default=30.0)
    parser.add_argument("--between-click-ms", type=float, default=45.0)
    parser.add_argument("--max-triggers", type=int, default=12)
    parser.add_argument("--status-ms", type=float, default=500.0)
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        default=DEFAULT_ROI,
        metavar=("X", "Y", "W", "H"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run live inference and trigger logging without sending mouse input",
    )
    args = parser.parse_args()

    args.fps = max(5.0, min(60.0, float(args.fps)))
    args.ready_threshold = max(0.0, min(1.0, float(args.ready_threshold)))
    args.rearm_threshold = max(0.0, min(1.0, float(args.rearm_threshold)))
    args.min_mode_confidence = max(
        0.0, min(1.0, float(args.min_mode_confidence))
    )
    args.max_triggers = max(1, int(args.max_triggers))
    args.status_ms = max(100.0, float(args.status_ms))

    if args.rearm_threshold >= args.ready_threshold:
        raise SystemExit("--rearm-threshold 必须小于 --ready-threshold")

    root = character_root(args.character)
    device = _select_device(args.device)
    runtime = LiveMotionRuntime(root, device, args.roi)
    gate = ReadyWindowGate(
        enter_threshold=args.ready_threshold,
        rearm_threshold=args.rearm_threshold,
        rearm_frames=args.rearm_frames,
        min_trigger_interval_s=float(args.min_trigger_interval_ms) / 1000.0,
    )

    stop_event = threading.Event()
    enabled_event = threading.Event()
    reset_requested = threading.Event()
    foreground_state = {"hwnd": 0}

    def foreground_check() -> bool:
        hwnd = int(foreground_state["hwnd"])
        return bool(
            hwnd
            and win32gui.IsWindow(hwnd)
            and win32gui.GetForegroundWindow() == hwnd
        )

    executor = BurstExecutor(
        stop_event=stop_event,
        enabled_event=enabled_event,
        foreground_check=foreground_check,
        clicks=args.burst_clicks,
        down_ms=args.mouse_down_ms,
        gap_ms=args.between_click_ms,
        dry_run=args.dry_run,
    )

    hotkey_lock = threading.Lock()
    hotkeys_down: set[object] = set()

    def on_key_press(key):
        with hotkey_lock:
            if key in hotkeys_down:
                return
            if key in (keyboard.Key.f8, keyboard.Key.f9):
                hotkeys_down.add(key)

        if key == keyboard.Key.f8:
            if enabled_event.is_set():
                enabled_event.clear()
                executor.release()
                print("[F8] probe DISARMED")
            else:
                enabled_event.set()
                print(
                    "[F8] probe ARMED - 手动左键启动第一段普攻后松手；"
                    "后续 READY window 将尝试短 burst"
                )
            reset_requested.set()
        elif key == keyboard.Key.f9:
            enabled_event.clear()
            stop_event.set()
            executor.release()
            print("[F9] EMERGENCY STOP")
            return False

    def on_key_release(key):
        with hotkey_lock:
            hotkeys_down.discard(key)

    listener = keyboard.Listener(
        on_press=on_key_press,
        on_release=on_key_release,
    )
    listener.start()

    print(
        f"character={args.character} device={device} target_fps={args.fps:.1f} "
        f"raw_READY>={args.ready_threshold:.2f}"
    )
    print(
        f"burst={args.burst_clicks}x down={args.mouse_down_ms:.0f}ms "
        f"gap={args.between_click_ms:.0f}ms max_triggers={args.max_triggers} "
        f"{'(DRY RUN)' if args.dry_run else '(LIVE INPUT)'}"
    )
    print("F8 = arm/disarm   F9 = emergency stop")
    print(f"等待 {GAME_PROCESS} 游戏窗口；输入默认处于 DISARMED。")

    trigger_count = 0
    auto_disarm_pending = False
    was_foreground = False
    last_status = 0.0
    next_deadline = time.monotonic()
    frame_interval = 1.0 / args.fps

    try:
        with mss.mss() as screen:
            while not stop_event.is_set():
                if reset_requested.is_set():
                    runtime.reset()
                    gate.reset()
                    trigger_count = 0
                    auto_disarm_pending = False
                    reset_requested.clear()

                hwnd = int(foreground_state["hwnd"])
                if not hwnd or not win32gui.IsWindow(hwnd):
                    hwnd = find_game_window()
                    foreground_state["hwnd"] = int(hwnd)
                    runtime.reset()
                    gate.reset()
                    was_foreground = False
                    if not hwnd:
                        time.sleep(0.25)
                        next_deadline = time.monotonic()
                        continue
                    print(f"[GAME] found hwnd={hwnd}")

                foreground = foreground_check()
                if not foreground:
                    if was_foreground:
                        print("[GAME] lost foreground -> inference/input paused")
                        runtime.reset()
                        gate.reset()
                        executor.release()
                    was_foreground = False
                    time.sleep(0.04)
                    next_deadline = time.monotonic()
                    continue

                if not was_foreground:
                    print(
                        f"[GAME] foreground -> warming {runtime.max_window} real frames "
                        "before READY may trigger"
                    )
                    runtime.reset()
                    gate.reset()
                    was_foreground = True

                frame = _capture_client(screen, hwnd)
                if frame is None:
                    time.sleep(0.05)
                    continue

                prediction = runtime.infer(frame)
                now = time.monotonic()
                if prediction is not None:
                    mode_id = int(prediction["mode"])
                    confidence = float(prediction["mode_confidence"])
                    phase = float(prediction["phase"])
                    ready = float(prediction["chain_ready"])

                    if enabled_event.is_set() and not auto_disarm_pending:
                        can_trigger = bool(
                            confidence >= args.min_mode_confidence
                            and not executor.busy
                            and trigger_count < args.max_triggers
                        )
                        decision = gate.update(
                            mode_id,
                            ready,
                            now,
                            can_trigger=can_trigger,
                        )
                        if decision.trigger:
                            label = (
                                f"mode={mode_id} conf={confidence:.3f} "
                                f"phase={phase:.3f} READY={ready:.3f}"
                            )
                            if executor.trigger(label):
                                trigger_count += 1
                                print(
                                    f"[TRIGGER {trigger_count}/{args.max_triggers}] "
                                    f"{label}"
                                )
                                if trigger_count >= args.max_triggers:
                                    auto_disarm_pending = True
                        elif decision.reason == "blocked_high_island":
                            print(
                                f"[SKIP] high READY island consumed: mode={mode_id} "
                                f"conf={confidence:.3f} ready={ready:.3f} "
                                f"busy={executor.busy}"
                            )

                    if auto_disarm_pending and not executor.busy:
                        enabled_event.clear()
                        auto_disarm_pending = False
                        executor.release()
                        print(
                            f"[LIMIT] reached {args.max_triggers} triggers -> DISARMED"
                        )

                    if now - last_status >= args.status_ms / 1000.0:
                        last_status = now
                        print(
                            f"[LIVE] input={'ON' if enabled_event.is_set() else 'OFF'} "
                            f"mode={mode_id} conf={confidence:.3f} "
                            f"phase={phase:.3f} ready={ready:.3f} "
                            f"infer={prediction['inference_ms']:.1f}ms "
                            f"burst={'BUSY' if executor.busy else 'idle'} "
                            f"triggers={trigger_count}/{args.max_triggers}"
                        )
                elif now - last_status >= args.status_ms / 1000.0:
                    last_status = now
                    print(
                        f"[WARMUP] {runtime.seen_frames}/{runtime.max_window} "
                        f"input={'ON' if enabled_event.is_set() else 'OFF'}"
                    )

                next_deadline += frame_interval
                remaining = next_deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                elif remaining < -0.25:
                    next_deadline = time.monotonic()
    except KeyboardInterrupt:
        print("[CTRL+C] stop")
    finally:
        enabled_event.clear()
        stop_event.set()
        executor.release()
        listener.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
