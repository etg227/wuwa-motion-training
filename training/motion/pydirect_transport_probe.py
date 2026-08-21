from __future__ import annotations

import argparse
import ctypes
import sys
import time

import pydirectinput
import win32gui

from auto_train import GAME_PROCESS, find_game_window


def _is_admin() -> bool:
    if sys.platform != "win32":
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _foreground_matches(hwnd: int) -> bool:
    return bool(
        hwnd
        and win32gui.IsWindow(hwnd)
        and win32gui.GetForegroundWindow() == hwnd
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send exactly one guarded PyDirect left-click to the foreground Wuthering Waves window."
    )
    parser.add_argument("--hold-ms", type=float, default=40.0)
    parser.add_argument("--stable-ms", type=float, default=700.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--allow-non-admin",
        action="store_true",
        help="diagnostic only: allow running without elevation",
    )
    args = parser.parse_args()

    hold_s = max(0.005, float(args.hold_ms) / 1000.0)
    stable_s = max(0.1, float(args.stable_ms) / 1000.0)
    timeout_s = max(2.0, float(args.timeout))

    admin = _is_admin()
    print(f"admin={admin}")
    if not admin and not args.allow_non_admin:
        print(
            "[BLOCKED] 当前 Python/PowerShell 不是管理员权限。\n"
            "之前实机验证通过的 PyDirect 原始输入要求与游戏同/更高完整性级别。\n"
            "请用‘以管理员身份运行’打开 PowerShell/终端后再运行本脚本。"
        )
        return 2

    pydirectinput.PAUSE = 0
    pydirectinput.FAILSAFE = False

    print(
        f"等待 {GAME_PROCESS} 成为前台窗口。请用 Alt+Tab 切回游戏，不要用鼠标点击来聚焦；\n"
        f"检测到游戏连续前台 {stable_s:.1f}s 后，只会自动发送 1 次左键（按住 {hold_s*1000:.0f}ms）。"
    )

    deadline = time.monotonic() + timeout_s
    stable_since: float | None = None
    hwnd = 0
    while time.monotonic() < deadline:
        candidate = find_game_window()
        if candidate and _foreground_matches(candidate):
            if hwnd != candidate:
                hwnd = candidate
                stable_since = time.monotonic()
                print(f"[GAME] foreground hwnd={hwnd}; stabilizing...")
            elif stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= stable_s:
                break
        else:
            hwnd = candidate or 0
            stable_since = None
        time.sleep(0.03)
    else:
        print("[TIMEOUT] 没有检测到持续前台的鸣潮窗口；未发送任何输入。")
        return 3

    before = win32gui.GetForegroundWindow()
    if before != hwnd:
        print(f"[ABORT] 发送前游戏已失去前台: expected={hwnd} foreground={before}")
        return 4

    print(f"[SEND] hwnd={hwnd} left DOWN")
    pydirectinput.mouseDown(button="left")
    time.sleep(hold_s)
    pydirectinput.mouseUp(button="left")
    after = win32gui.GetForegroundWindow()
    print(f"[SEND] left UP; foreground_before={before} foreground_after={after}")

    if after != hwnd:
        print("[RESULT] 输入发送后游戏失去前台。这个问题属于输入/窗口层，不是 READY 模型。")
        return 5

    print(
        "[RESULT] PyDirect 事件已在游戏保持前台时发送。\n"
        "请只根据角色肉眼表现判断：如果角色做出一次普攻，transport=PASS；如果完全不动，transport=FAIL。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
