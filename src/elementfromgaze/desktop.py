"""Gaze control for the whole Windows desktop: `efg control`.

What you look at: Windows UI Automation (the accessibility API screen readers use) lists the
clickable elements of the foreground window and the taskbar — buttons, links, menu items, tabs,
checkboxes, list items — with their screen rectangles. A background thread refreshes them (one
window takes 30–240 ms, measured), so the per-frame decision is pure arithmetic.

Which one: the same closed-form Bayesian snap as web/efg.js. Measurement g = true gaze + bias,
bias ~ N(0, diag(sx², sy²)); "looking inside element k" spreads the true gaze uniformly over its
rectangle, "looking at anything else" is uniform over the screen:
    p(g|k) = [Φ((x1-gx)/sx) - Φ((x0-gx)/sx)]·[Φ((y1-gy)/sy) - Φ((y0-gy)/sy)] / |R_k|,  p(g|0) = 1/|screen|
Snap at posterior > 0.8, release below 0.4. Two neighbours that the error can't tell apart split
the posterior and nothing is snapped: it never pretends to know.

How to click: a long blink (close both eyes and hold; the hold time and closed threshold come
from `efg gesture`, which checks that your normal blinks stay well below the hold). The element is
frozen when the eyes start closing, because closing lids drag the vertical gaze estimate down.
The older squint gesture (`"mode": "squint"` in gesture.json) is still supported.
The click is a real mouse click at the element's centre (works in any app); the cursor goes back.
"""

from __future__ import annotations

import ctypes
import json
import math
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr        # Φ, vectorised

user32 = ctypes.windll.user32

# ---------------------------------------------------------------------------------------------
@dataclass
class Element:
    rect: tuple[float, float, float, float]      # left, top, right, bottom (physical px)
    name: str
    kind: int                                     # UIA control type id


class UIAScanner:
    """Background thread: clickable elements of the foreground window + taskbar, every `period` s
    (and right away when the foreground window changes)."""

    def __init__(self, W: int, H: int, period: float = 1.5):
        self.W, self.H, self.period = W, H, period
        self.elements: list[Element] = []
        self.rects = np.zeros((0, 4))
        self.lock = threading.Lock()
        self.scan_ms = 0.0
        self._stop = threading.Event()
        self.error: str | None = None
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        try:
            import comtypes
            import comtypes.client
            comtypes.CoInitialize()
            comtypes.client.GetModule("UIAutomationCore.dll")
            from comtypes.gen import UIAutomationClient as U
        except Exception as e:     # comtypes missing / UIA unavailable
            self.error = f"UI Automation no disponible: {e}"
            return
        uia = comtypes.CoCreateInstance(U.CUIAutomation._reg_clsid_, interface=U.IUIAutomation,
                                        clsctx=comtypes.CLSCTX_INPROC_SERVER)
        clickable = uia.CreateOrConditionFromArray([uia.CreatePropertyCondition(p, True) for p in (
            U.UIA_IsInvokePatternAvailablePropertyId, U.UIA_IsTogglePatternAvailablePropertyId,
            U.UIA_IsSelectionItemPatternAvailablePropertyId, U.UIA_IsExpandCollapsePatternAvailablePropertyId)])
        # web content in browsers often exposes links/buttons without those patterns: add control types
        kinds = uia.CreateOrConditionFromArray([uia.CreatePropertyCondition(U.UIA_ControlTypePropertyId, k) for k in (
            50000, 50005, 50011, 50019, 50007, 50002, 50013, 50003, 50024, 50031, 50004)])
        # Button, Hyperlink, MenuItem, TabItem, ListItem, CheckBox, RadioButton, ComboBox, TreeItem, SplitButton, Edit
        clickable = uia.CreateOrCondition(clickable, kinds)
        cond = uia.CreateAndCondition(clickable, uia.CreatePropertyCondition(U.UIA_IsOffscreenPropertyId, False))
        cache = uia.CreateCacheRequest()
        for p in (U.UIA_BoundingRectanglePropertyId, U.UIA_NamePropertyId, U.UIA_ControlTypePropertyId):
            cache.AddProperty(p)
        last_fg, last_scan, tray, tray_at = None, 0.0, [], -1e9

        def scan(h) -> list[Element]:
            out = []
            try:
                arr = uia.ElementFromHandle(h).FindAllBuildCache(4, cond, cache)   # 4 = TreeScope_Descendants
                for i in range(arr.Length):
                    e = arr.GetElement(i)
                    r = e.CachedBoundingRectangle
                    w, hh = r.right - r.left, r.bottom - r.top
                    if w < 6 or hh < 6 or w * hh > 0.4 * self.W * self.H:
                        continue
                    if r.right < 0 or r.bottom < 0 or r.left > self.W or r.top > self.H:
                        continue
                    out.append(Element((r.left, r.top, r.right, r.bottom), e.CachedName or "", e.CachedControlType))
            except Exception:
                pass                   # windows come and go mid-scan
            return out

        while not self._stop.is_set():
            fg = user32.GetForegroundWindow()
            if fg == last_fg and time.monotonic() - last_scan < self.period:
                time.sleep(0.05)
                continue
            last_fg, last_scan = fg, time.monotonic()
            t0 = time.perf_counter()
            tw = user32.FindWindowW("Shell_TrayWnd", None)
            if tw and time.monotonic() - tray_at > 5.0:      # the taskbar barely changes: every 5 s
                tray, tray_at = scan(tw), time.monotonic()
            els: list[Element] = (scan(fg) if fg and fg != tw else []) + tray
            with self.lock:
                self.elements = els
                self.rects = np.array([e.rect for e in els], float).reshape(-1, 4)
            self.scan_ms = (time.perf_counter() - t0) * 1000

    def snapshot(self) -> tuple[list[Element], np.ndarray]:
        with self.lock:
            return self.elements, self.rects

    def close(self) -> None:
        self._stop.set()


def posterior(g, sx: float, sy: float, rects: np.ndarray, area: float, bg_weight: float = 1.0) -> np.ndarray:
    """P(looking at element k | gaze g) for every rect, closed form (see module docstring)."""
    if not len(rects):
        return np.zeros(0)
    x0, y0, x1, y1 = rects.T
    px = ndtr((x1 - g[0]) / sx) - ndtr((x0 - g[0]) / sx)
    py = ndtr((y1 - g[1]) / sy) - ndtr((y0 - g[1]) / sy)
    L = px * py / ((x1 - x0) * (y1 - y0))
    return L / (bg_weight / area + L.sum())


class Snapper:
    """Hysteresis on top of the posterior: snap > 0.8, release < 0.4 (unless another wins > 0.8)."""

    def __init__(self, snap_p: float = 0.8, release_p: float = 0.4):
        self.snap_p, self.release_p = snap_p, release_p
        self.current: tuple | None = None        # rect of the snapped element
        self.p = 0.0
        self.name = ""

    def __call__(self, g, sx, sy, elements, rects, area):
        if g is None or not len(rects):
            self.current, self.p = None, 0.0
            return None
        P = posterior(g, sx, sy, rects, area)
        k = int(np.argmax(P))
        cur = None
        if self.current is not None:
            d = np.abs(rects - np.array(self.current)).sum(1)
            j = int(np.argmin(d))
            if d[j] < 6:
                cur = j
        if cur is not None and P[cur] >= self.release_p and not (k != cur and P[k] >= self.snap_p):
            k = cur
        elif P[k] < self.snap_p:
            self.current, self.p, self.name = None, 0.0, ""
            return None
        self.current, self.p, self.name = tuple(rects[k]), float(P[k]), elements[k].name
        return self.current


# ---------------------------------------------------------------------------------------------
GESTURE_FILE = "gesture.json"


class SquintClick:
    """Squint-and-hold detector with a per-user threshold.

    score = mean(eyeSquintLeft, eyeSquintRight) - rolling baseline (median of the last ~30 s).
    Fires once when score > thr for hold_s, while not blinking (eyeBlink < blink_max) and not
    smiling (mouthSmile < smile_max); then a refractory period."""

    def __init__(self, names: list[str], thr: float = 0.25, hold_s: float = 0.3, blink_max: float = 0.6,
                 smile_max: float = 0.35, refractory_s: float = 1.0):
        ix = {n: i for i, n in enumerate(names)}
        self.i_sq = [ix["eyeSquintLeft"], ix["eyeSquintRight"]]
        self.i_bl = [ix["eyeBlinkLeft"], ix["eyeBlinkRight"]]
        self.i_sm = [ix["mouthSmileLeft"], ix["mouthSmileRight"]]
        self.thr, self.hold_s, self.blink_max, self.smile_max, self.refractory_s = thr, hold_s, blink_max, smile_max, refractory_s
        self.base: list[float] = []
        self.since: float | None = None
        self.last_fire = -1e9
        self.progress = 0.0          # 0..1, for the on-screen ring

    def __call__(self, bs: np.ndarray, t: float) -> tuple[bool, bool]:
        """-> (squinting started now, fire click now)"""
        sq = float(bs[self.i_sq].mean())
        bl = float(bs[self.i_bl].min())
        sm = float(bs[self.i_sm].mean())
        base = float(np.median(self.base)) if len(self.base) > 30 else 0.2
        on = sq - base > self.thr and bl < self.blink_max and sm < self.smile_max
        if not on:
            self.base.append(sq)
            if len(self.base) > 600:
                del self.base[:100]
            self.since, self.progress = None, 0.0
            return False, False
        started = self.since is None
        if started:
            self.since = t
        self.progress = min(1.0, (t - self.since) / self.hold_s)
        if t - self.since >= self.hold_s and t - self.last_fire > self.refractory_s:
            self.last_fire = t
            self.since = t + 1e9       # don't refire until the eyes relax
            return started, True
        return started, False


def load_gesture(path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------------------------
class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]
        _fields_ = [("mi", MOUSEINPUT), ("pad", ctypes.c_byte * 32)]
    _fields_ = [("type", wintypes.DWORD), ("i", _I)]


def click_at(x: float, y: float, restore: bool = True) -> None:
    """A real left click at screen (x, y) physical px. Marked as injected by Windows, so our own
    click-learning hook ignores it (it must: the click is where WE decided you looked)."""
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    user32.SetCursorPos(int(round(x)), int(round(y)))
    evs = (INPUT * 2)()
    for e, flag in zip(evs, (0x0002, 0x0004)):          # LEFTDOWN, LEFTUP
        e.type = 0
        e.i.mi.dwFlags = flag
    user32.SendInput(2, evs, ctypes.sizeof(INPUT))
    if restore:
        time.sleep(0.03)
        user32.SetCursorPos(pt.x, pt.y)


class LongBlinkClick:
    """Long-blink click: close both eyes for hold_s (natural blinks last 0.1-0.3 s; the default 0.45 s
    and the closed threshold come from `efg gesture`). Same interface as SquintClick. Squinting was
    tried first and failed for this user: MediaPipe's eyeSquint score went DOWN when squinting
    (0.31 -> 0.11) and overlapped normal frames, while a closed eye is unmistakable."""

    def __init__(self, names: list[str], closed_thr: float = 0.5, hold_s: float = 0.45, refractory_s: float = 1.0):
        ix = {n: i for i, n in enumerate(names)}
        self.i_bl = [ix["eyeBlinkLeft"], ix["eyeBlinkRight"]]
        self.closed_thr, self.hold_s, self.refractory_s = closed_thr, hold_s, refractory_s
        self.since: float | None = None
        self.fired = False
        self.last_fire = -1e9
        self.progress = 0.0

    def __call__(self, bs: np.ndarray, t: float) -> tuple[bool, bool]:
        closed = float(bs[self.i_bl].min()) > self.closed_thr      # BOTH eyes: a wink or a glance down is not it
        if not closed:
            self.since, self.fired, self.progress = None, False, 0.0
            return False, False
        started = self.since is None
        if started:
            self.since = t
        self.progress = min(1.0, (t - self.since) / self.hold_s)
        if not self.fired and t - self.since >= self.hold_s and t - self.last_fire > self.refractory_s:
            self.fired, self.last_fire = True, t
            return started, True
        return started, False


def make_clicker(names: list[str], gesture: dict):
    if gesture.get("mode", "blink") == "squint":
        return SquintClick(names, thr=gesture.get("thr", 0.25), hold_s=gesture.get("hold_s", 0.3))
    return LongBlinkClick(names, closed_thr=gesture.get("closed_thr", 0.5), hold_s=gesture.get("hold_s", 0.45))
