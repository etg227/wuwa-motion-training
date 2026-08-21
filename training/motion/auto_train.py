from __future__ import annotations

import ctypes
import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import mss
import numpy as np
import psutil
import win32gui
import win32process
from pynput import keyboard, mouse

from auto_cycle import discover_cycles, save_annotation
from common import DATA_ROOT, ROOT, character_root, safe_character, write_json


GAME_PROCESS = "client-win64-shipping.exe"
CAPTURE_FPS = 30.0
TRAIN_EPOCHS = 30
XINPUT_POLL_HZ = 240.0
XINPUT_GAMEPAD_TRIGGER_THRESHOLD = 30


class XInputGamepad(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class XInputState(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.c_ulong),
        ("Gamepad", XInputGamepad),
    ]


XINPUT_BUTTONS = (
    (0x0001, "dpad_up"),
    (0x0002, "dpad_down"),
    (0x0004, "dpad_left"),
    (0x0008, "dpad_right"),
    (0x0010, "start"),
    (0x0020, "back"),
    (0x0040, "left_thumb"),
    (0x0080, "right_thumb"),
    (0x0100, "lb"),
    (0x0200, "rb"),
    (0x1000, "a"),
    (0x2000, "b"),
    (0x4000, "x"),
    (0x8000, "y"),
)


def _load_xinput():
    for dll_name in ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"):
        try:
            dll = ctypes.WinDLL(dll_name)
            get_state = dll.XInputGetState
            get_state.argtypes = [ctypes.c_ulong, ctypes.POINTER(XInputState)]
            get_state.restype = ctypes.c_ulong
            return get_state, dll_name
        except (OSError, AttributeError):
            continue
    return None, None


XINPUT_GET_STATE, XINPUT_DLL_NAME = _load_xinput() if sys.platform == "win32" else (None, None)


def _process_name_from_hwnd(hwnd: int) -> str:
    if not hwnd:
        return ""
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return psutil.Process(pid).name().casefold()
    except (psutil.Error, OSError):
        return ""


def find_game_window() -> int:
    candidates: list[int] = []

    def callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        if _process_name_from_hwnd(hwnd) != GAME_PROCESS:
            return
        try:
            left, top = win32gui.ClientToScreen(hwnd, (0, 0))
            right, bottom = win32gui.ClientToScreen(hwnd, win32gui.GetClientRect(hwnd)[2:])
            if right - left >= 640 and bottom - top >= 360:
                candidates.append(hwnd)
        except Exception:
            return

    win32gui.EnumWindows(callback, None)
    if not candidates:
        return 0
    foreground = win32gui.GetForegroundWindow()
    if foreground in candidates:
        return foreground
    return candidates[0]


def game_client_rect(hwnd: int):
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    client = win32gui.GetClientRect(hwnd)
    right, bottom = win32gui.ClientToScreen(hwnd, (client[2], client[3]))
    return left, top, right, bottom


def _key_name(key) -> str:
    try:
        if key.char is not None:
            return str(key.char)
    except AttributeError:
        pass
    try:
        return str(key.name)
    except AttributeError:
        return str(key)


def _xinput_digital_state(state: XInputState) -> dict[str, bool]:
    buttons = int(state.Gamepad.wButtons)
    result = {name: bool(buttons & mask) for mask, name in XINPUT_BUTTONS}
    result["lt"] = int(state.Gamepad.bLeftTrigger) >= XINPUT_GAMEPAD_TRIGGER_THRESHOLD
    result["rt"] = int(state.Gamepad.bRightTrigger) >= XINPUT_GAMEPAD_TRIGGER_THRESHOLD
    return result


class RecordingSession:
    def __init__(self, character: str, log):
        self.character = safe_character(character)
        self.log = log
        self.root = character_root(self.character)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = self.root / "videos" / f"auto_{stamp}.mp4"
        self.telemetry_path = self.root / "videos" / f"auto_{stamp}.inputs.jsonl"
        self.session_path = self.root / "videos" / f"auto_{stamp}.session.json"

        self.stop_event = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.gamepad_thread: threading.Thread | None = None
        self.keyboard_listener = None
        self.mouse_listener = None
        self.hwnd = 0
        self.frame_count = 0
        self.started_at = 0.0
        self.finished_at = 0.0
        self.first_frame_at = 0.0
        self.telemetry: list[dict] = []
        self.telemetry_lock = threading.Lock()
        self.error: Exception | None = None
        self.width = 0
        self.height = 0
        self.connected_gamepads: set[int] = set()

    def _game_foreground(self) -> bool:
        return bool(self.hwnd and win32gui.GetForegroundWindow() == self.hwnd)

    def _append_input(self, kind: str, code: str, action: str):
        if not self._game_foreground() or self.first_frame_at <= 0:
            return
        with self.telemetry_lock:
            self.telemetry.append(
                {
                    "frame": max(0, self.frame_count - 1),
                    "t_ms": round((time.monotonic() - self.first_frame_at) * 1000, 3),
                    "device": kind,
                    "code": code,
                    "action": action,
                }
            )

    def _on_key_press(self, key):
        self._append_input("key", _key_name(key), "down")

    def _on_key_release(self, key):
        self._append_input("key", _key_name(key), "up")

    def _on_mouse_click(self, _x, _y, button, pressed):
        name = getattr(button, "name", str(button))
        self._append_input("mouse", name, "down" if pressed else "up")

    def _gamepad_loop(self):
        if XINPUT_GET_STATE is None:
            self.log("未找到 XInput DLL；本次只记录键盘/鼠标输入。")
            return

        previous: dict[int, dict[str, bool]] = {}
        interval = 1.0 / XINPUT_POLL_HZ
        while not self.stop_event.is_set():
            loop_start = time.monotonic()
            for index in range(4):
                state = XInputState()
                result = int(XINPUT_GET_STATE(index, ctypes.byref(state)))
                if result != 0:
                    previous.pop(index, None)
                    continue

                if index not in self.connected_gamepads:
                    self.connected_gamepads.add(index)
                    self.log(f"检测到 XInput 手柄 #{index + 1}（{XINPUT_DLL_NAME}）")

                current = _xinput_digital_state(state)
                prior = previous.get(index)
                if prior is not None and self._game_foreground() and self.first_frame_at > 0:
                    for code, pressed in current.items():
                        old_pressed = prior.get(code, False)
                        if pressed == old_pressed:
                            continue
                        self._append_input(
                            f"gamepad{index}",
                            code,
                            "down" if pressed else "up",
                        )
                previous[index] = current

            remaining = interval - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    def start(self):
        self.hwnd = find_game_window()
        if not self.hwnd:
            raise RuntimeError(
                f"没有找到 {GAME_PROCESS} 游戏窗口。请先启动鸣潮，再点击‘开始录制’。"
            )

        self.started_at = time.monotonic()
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self.mouse_listener = mouse.Listener(on_click=self._on_mouse_click)
        self.keyboard_listener.start()
        self.mouse_listener.start()

        self.gamepad_thread = threading.Thread(target=self._gamepad_loop, daemon=True)
        self.gamepad_thread.start()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        writer = None
        try:
            with mss.mss() as screen:
                next_deadline = time.monotonic()
                frame_interval = 1.0 / CAPTURE_FPS

                while not self.stop_event.is_set():
                    if not win32gui.IsWindow(self.hwnd):
                        self.hwnd = find_game_window()
                        if not self.hwnd:
                            time.sleep(0.1)
                            next_deadline = time.monotonic()
                            continue

                    # 只有游戏前台时才写帧。点击停止切回训练器后不会把训练器录进去。
                    if not self._game_foreground():
                        time.sleep(0.03)
                        next_deadline = time.monotonic()
                        continue

                    left, top, right, bottom = game_client_rect(self.hwnd)
                    width = right - left
                    height = bottom - top
                    if width < 320 or height < 180:
                        time.sleep(0.05)
                        continue

                    grabbed = screen.grab(
                        {"left": left, "top": top, "width": width, "height": height}
                    )
                    frame = np.asarray(grabbed, dtype=np.uint8)[:, :, :3]
                    frame = np.ascontiguousarray(frame)

                    if writer is None:
                        self.width = width - (width % 2)
                        self.height = height - (height % 2)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(self.video_path),
                            fourcc,
                            CAPTURE_FPS,
                            (self.width, self.height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError("无法创建 MP4 录像文件。")
                        self.first_frame_at = time.monotonic()
                        self.log(
                            f"开始写入游戏画面：{self.width}x{self.height} @ {CAPTURE_FPS:.0f} FPS"
                        )

                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(
                            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
                        )

                    writer.write(frame)
                    self.frame_count += 1

                    next_deadline += frame_interval
                    remaining = next_deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    elif remaining < -0.25:
                        next_deadline = time.monotonic()

        except Exception as exc:
            self.error = exc
            self.log(f"录制线程错误：{exc}")
        finally:
            if writer is not None:
                writer.release()

    def stop(self):
        self.stop_event.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=5)
        if self.gamepad_thread is not None:
            self.gamepad_thread.join(timeout=2)

        for listener in (self.keyboard_listener, self.mouse_listener):
            if listener is not None:
                listener.stop()

        self.finished_at = time.monotonic()

        with self.telemetry_lock:
            events = list(self.telemetry)
        with self.telemetry_path.open("w", encoding="utf-8") as stream:
            for item in events:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")

        device_counts: dict[str, int] = {}
        for item in events:
            device = str(item["device"])
            device_counts[device] = device_counts.get(device, 0) + 1

        session = {
            "schema": 2,
            "character": self.character,
            "video": str(self.video_path.resolve()),
            "input_log": str(self.telemetry_path.resolve()),
            "capture_fps": CAPTURE_FPS,
            "frames": self.frame_count,
            "width": self.width,
            "height": self.height,
            "input_events": len(events),
            "input_events_by_device": device_counts,
            "xinput_dll": XINPUT_DLL_NAME,
            "xinput_gamepads": sorted(self.connected_gamepads),
            "foreground_only": True,
        }
        write_json(self.session_path, session)

        if self.error is not None:
            raise RuntimeError(str(self.error))
        if self.frame_count < int(CAPTURE_FPS * 3):
            raise RuntimeError("有效游戏录像不足 3 秒；本次文件已保存，但不会训练。")
        return self.video_path


class AutoTrainerApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("角色动作自动训练")
        self.window.geometry("760x540")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.recording: RecordingSession | None = None
        self.pipeline_thread: threading.Thread | None = None
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        self.config_path = DATA_ROOT / "auto_train_config.json"

        config = self._load_config()
        self.character_var = tk.StringVar(value=config.get("character", "Suisui"))
        self.status_var = tk.StringVar(value="准备就绪")

        outer = ttk.Frame(self.window, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text="角色动作自动训练", font=("Segoe UI", 16, "bold")).pack(anchor=tk.W)
        help_text = (
            "角色名只需第一次确认。之后每次训练只需要：开始录制 → 回到游戏连续平A/正常操作 → "
            "回来点击停止录制。停止后会自动找完整平A循环、重建数据集并训练。"
        )
        ttk.Label(outer, text=help_text, wraplength=710).pack(anchor=tk.W, pady=(6, 14))

        row = ttk.Frame(outer)
        row.pack(fill=tk.X)
        ttk.Label(row, text="角色：").pack(side=tk.LEFT)
        self.character_entry = ttk.Entry(row, textvariable=self.character_var, width=28)
        self.character_entry.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(row, textvariable=self.status_var).pack(side=tk.LEFT)

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=14)
        self.start_button = ttk.Button(buttons, text="开始录制", command=self._start_recording)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(
            buttons,
            text="停止录制并自动训练",
            command=self._stop_recording,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=10)

        ttk.Separator(outer).pack(fill=tk.X, pady=(0, 10))
        ttk.Label(outer, text="日志").pack(anchor=tk.W)
        self.log_box = tk.Text(outer, height=22, wrap=tk.WORD, state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.window.after(100, self._drain_events)

    def _load_config(self):
        if not self.config_path.is_file():
            return {}
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_config(self):
        write_json(self.config_path, {"character": safe_character(self.character_var.get())})

    def log(self, message: str):
        self.events.put(("log", message))

    def _drain_events(self):
        while True:
            try:
                kind, message = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.log_box.configure(state=tk.NORMAL)
                stamp = time.strftime("%H:%M:%S")
                self.log_box.insert(tk.END, f"[{stamp}] {message}\n")
                self.log_box.see(tk.END)
                self.log_box.configure(state=tk.DISABLED)
            elif kind == "pipeline_done":
                self.start_button.configure(state=tk.NORMAL)
                self.stop_button.configure(state=tk.DISABLED)
                self.character_entry.configure(state=tk.NORMAL)
                self.status_var.set(message)
        self.window.after(100, self._drain_events)

    def _start_recording(self):
        if self.recording is not None or (
            self.pipeline_thread is not None and self.pipeline_thread.is_alive()
        ):
            return
        try:
            character = safe_character(self.character_var.get())
            self._save_config()
            session = RecordingSession(character, self.log)
            session.start()
        except Exception as exc:
            messagebox.showerror("无法开始录制", str(exc))
            return

        self.recording = session
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.character_entry.configure(state=tk.DISABLED)
        self.status_var.set("等待/录制游戏前台")
        self.log("已开始采集。切回鸣潮后才会写入帧；离开游戏时自动暂停采集。")

    def _stop_recording(self):
        if self.recording is None:
            return
        session = self.recording
        self.recording = None
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("正在停止录像...")

        try:
            video = session.stop()
        except Exception as exc:
            self.log(f"停止录像：{exc}")
            self.start_button.configure(state=tk.NORMAL)
            self.character_entry.configure(state=tk.NORMAL)
            self.status_var.set("录像未进入训练")
            return

        with session.telemetry_lock:
            gamepad_events = sum(
                1 for item in session.telemetry if str(item.get("device", "")).startswith("gamepad")
            )
        self.log(
            f"录像完成：{video.name}，有效帧={session.frame_count}，"
            f"自动记录输入事件={len(session.telemetry)}（手柄={gamepad_events}）"
        )
        self.status_var.set("自动分析/训练中")
        self.pipeline_thread = threading.Thread(
            target=self._pipeline,
            args=(session.character, video),
            daemon=True,
        )
        self.pipeline_thread.start()

    def _run_command(self, args: list[str]):
        process = subprocess.Popen(
            args,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            self.log(line.rstrip())
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"命令退出码 {code}: {' '.join(args)}")

    def _pipeline(self, character: str, video: Path):
        try:
            self.log("步骤 1/3：自动寻找重复动作与完整平A cycle...")
            annotation = discover_cycles(character, video, analysis_fps=30.0)
            annotation_path = save_annotation(character, video, annotation)
            auto = annotation["auto_detection"]
            self.log(
                f"自动识别 {auto['cycle_count']} 个完整 cycle；"
                f"周期≈{auto['period_s']:.3f}s；"
                f"边界置信度={auto['boundary_confidence']:.3f}"
            )
            self.log(f"自动标注：{annotation_path.name}")

            self.log("步骤 2/3：重建该角色全部 cycle 数据集...")
            self._run_command(
                [
                    sys.executable,
                    str(ROOT / "training" / "motion" / "build_dataset.py"),
                    "--character",
                    character,
                ]
            )

            manifest = character_root(character) / "manifest.jsonl"
            cycle_count = 0
            if manifest.is_file():
                cycle_count = sum(
                    1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()
                )

            if cycle_count < 3:
                raise RuntimeError(
                    f"当前总共只有 {cycle_count} 个有效 cycle。录像已保留；"
                    "继续再录几轮，达到 3 个后会自动训练。"
                )

            self.log(f"步骤 3/3：自动训练 phase 模型（总 cycle={cycle_count}）...")
            self._run_command(
                [
                    sys.executable,
                    str(ROOT / "training" / "motion" / "train_phase_model.py"),
                    "--character",
                    character,
                    "--epochs",
                    str(TRAIN_EPOCHS),
                ]
            )
            model_path = character_root(character) / "models" / "phase_model.pt"
            self.log(f"训练完成：{model_path}")
            self.events.put(("pipeline_done", "训练完成，可继续录下一段"))

        except Exception as exc:
            self.log(f"自动流程结束：{exc}")
            self.events.put(("pipeline_done", "本次数据已保存；未完成训练"))

    def _on_close(self):
        if self.recording is not None:
            if not messagebox.askyesno("正在录制", "正在录制，确定停止并退出吗？"):
                return
            try:
                self.recording.stop()
            except Exception:
                pass
            self.recording = None
        self.window.destroy()

    def run(self):
        self.window.mainloop()


def main() -> int:
    if sys.platform != "win32":
        raise SystemExit("自动录制训练器当前只支持 Windows。")
    AutoTrainerApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
