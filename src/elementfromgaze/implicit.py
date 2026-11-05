"""Implicit labels from normal computer use: mouse clicks and the text caret while typing.

Low-level Windows hooks only record *that* a click / key press happened and when. Key codes are
never read or stored. The caret position is asked to the foreground window when a key event is
processed (GetGUIThreadInfo works for classic Win32 text controls and many apps; browsers and
some modern apps may not expose it — then the key is simply ignored).
"""

from __future__ import annotations

import ctypes
import threading
import time
from collections import deque
from ctypes import wintypes
from dataclasses import dataclass

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE

WH_KEYBOARD_LL, WH_MOUSE_LL = 13, 14
WM_KEYDOWN, WM_SYSKEYDOWN = 0x0100, 0x0104
WM_LBUTTONDOWN, WM_RBUTTONDOWN = 0x0201, 0x0204
WM_QUIT = 0x0012


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT),
    ]


def caret_position() -> tuple[float, float] | None:
    """Screen position (physical px for a DPI-aware caller) of the focused text caret, if exposed."""
    fg = user32.GetForegroundWindow()
    if not fg:
        return None
    tid = user32.GetWindowThreadProcessId(fg, None)
    gti = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
    if not user32.GetGUIThreadInfo(tid, ctypes.byref(gti)) or not gti.hwndCaret:
        return None
    rc = gti.rcCaret
    if rc.right == rc.left and rc.bottom == rc.top:
        return None
    pt = wintypes.POINT((rc.left + rc.right) // 2, (rc.top + rc.bottom) // 2)
    if not user32.ClientToScreen(gti.hwndCaret, ctypes.byref(pt)):
        return None
    return float(pt.x), float(pt.y)


def cursor_position() -> tuple[float, float]:
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return float(pt.x), float(pt.y)


@dataclass
class InputEvent:
    kind: str                 # "click" | "key"
    t: float                  # time.monotonic()
    pos: tuple[float, float] | None = None   # click position (keys: resolved later from the caret)


class InputWatcher:
    """Runs low-level mouse + keyboard hooks on their own thread with a message loop."""

    def __init__(self) -> None:
        self.events: "deque[InputEvent]" = deque(maxlen=512)
        self._tid = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "InputWatcher":
        self._thread.start()
        self._ready.wait(2)
        return self

    def _run(self) -> None:
        self._tid = kernel32.GetCurrentThreadId()

        def mouse_proc(n, wparam, lparam):
            if n == 0 and wparam in (WM_LBUTTONDOWN, WM_RBUTTONDOWN):
                self.events.append(InputEvent("click", time.monotonic(), cursor_position()))
            return user32.CallNextHookEx(None, n, wparam, lparam)

        def key_proc(n, wparam, lparam):
            if n == 0 and wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                self.events.append(InputEvent("key", time.monotonic()))
            return user32.CallNextHookEx(None, n, wparam, lparam)

        self._mouse_cb, self._key_cb = HOOKPROC(mouse_proc), HOOKPROC(key_proc)  # keep alive
        hmod = kernel32.GetModuleHandleW(None)
        hm = user32.SetWindowsHookExW(WH_MOUSE_LL, self._mouse_cb, hmod, 0)
        hk = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._key_cb, hmod, 0)
        self._ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            pass
        for h in (hm, hk):
            if h:
                user32.UnhookWindowsHookEx(h)

    def stop(self) -> None:
        if self._tid:
            user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
        self._thread.join(timeout=1)

    def drain_ready(self, now: float, delay: float) -> list[InputEvent]:
        """Events older than `delay` s (so the gaze samples just after them exist too)."""
        out = []
        while self.events and now - self.events[0].t >= delay:
            out.append(self.events.popleft())
        return out
