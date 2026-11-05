"""Screen geometry (physical pixels, physical size) and click-through windows on Windows."""

from __future__ import annotations

import ctypes
import math

_user32 = ctypes.windll.user32
_gdi32 = ctypes.windll.gdi32


def make_dpi_aware() -> None:
    """Make every coordinate in this process a physical pixel. Call before any window opens."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        _user32.SetProcessDPIAware()


def screen_size_px() -> tuple[int, int]:
    return _user32.GetSystemMetrics(0), _user32.GetSystemMetrics(1)


def screen_size_mm() -> tuple[int, int]:
    hdc = _user32.GetDC(0)
    try:
        return _gdi32.GetDeviceCaps(hdc, 4), _gdi32.GetDeviceCaps(hdc, 6)  # HORZSIZE, VERTSIZE
    finally:
        _user32.ReleaseDC(0, hdc)


def px_per_degree(distance_mm: float = 550.0) -> float:
    """How many pixels one degree of visual angle spans at the given viewing distance."""
    w_px, _ = screen_size_px()
    w_mm, _ = screen_size_mm()
    mm_per_px = w_mm / w_px if w_mm else 0.25
    return 2 * distance_mm * math.tan(math.radians(0.5)) / mm_per_px


GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000


def make_click_through(hwnd: int) -> None:
    style = _user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    _user32.SetWindowLongW(
        hwnd, GWL_EXSTYLE,
        style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
    )
