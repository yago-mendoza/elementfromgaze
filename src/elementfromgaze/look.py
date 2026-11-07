"""One look for every screen the tracker draws: calibration, warm-up, tests, the overlay, toasts.

Near-black ground, quiet greys, one copper accent (the thing to look at). Head-movement phases
use a steel blue so "move your head" never looks like "look here". Fonts ship with Windows.
"""

from __future__ import annotations

import re
import tkinter as tk

GROUND = "#111214"      # intro and message screens
PANEL = "#1A1C1F"
INK = "#ECEAE6"
DIM = "#8C9097"
FAINT = "#50545B"
RULE = "#2A2D31"
ACCENT = "#C8643B"      # copper: where to look, progress, the click ring
ACCENT_HI = "#E58A5C"
STEEL = "#6FA8C9"       # head-movement phases
GOOD = "#5DBB73"
WARN = "#D9A441"
BAD = "#E5534B"
GAZE = "#E8E4DC"        # the estimated-gaze dot: neutral, so it never competes with the target

SANS = "Segoe UI"
SANS_LIGHT = "Segoe UI Light"
SANS_SEMI = "Segoe UI Semibold"
MONO = "Consolas"


def round_rect(c: tk.Canvas, x0, y0, x1, y1, r=8, **kw):
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1,
           x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return c.create_polygon(pts, smooth=True, splinesteps=12, **kw)


def keycap(c: tk.Canvas, x, y, key: str, label: str, tags=()) -> float:
    """[KEY] label, left-aligned at x; returns the x where it ends."""
    k = c.create_text(x + 10, y, text=key, fill=INK, font=(MONO, 10), anchor="w", tags=tags)
    kb = c.bbox(k) or (x, y - 8, x + 60, y + 8)
    pad_y = (kb[3] - kb[1]) * 0.45
    box = round_rect(c, x, kb[1] - pad_y, kb[2] + 10, kb[3] + pad_y, r=6, outline=FAINT, fill=PANEL, width=1, tags=tags)
    c.tag_lower(box, k)
    t = c.create_text(kb[2] + 22, y, text=label, fill=DIM, font=(SANS, 12), anchor="w", tags=tags)
    bb = c.bbox(t)
    return bb[2] if bb else kb[2] + 22


def _below(c: tk.Canvas, item, gap: float) -> float:
    bb = c.bbox(item)
    return (bb[3] if bb else 0) + gap


_KEYS = re.compile(r"(ESPACIO|ESC|INTRO)\s+para\s+([^·\n]+)")


def split_intro(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Pull 'ESPACIO para empezar · ESC para salir' out of an intro text: (body, [(key, label)])."""
    keys, body = [], []
    for line in text.split("\n"):
        found = _KEYS.findall(line)
        if found:
            keys += [(k, v.strip()) for k, v in found]
        else:
            body.append(line)
    return "\n".join(body).strip(), keys


_DUR = re.compile(r"\s*\((?:≈|~)?\s*(\d+)\s*s\)\s*$")


def intro(c: tk.Canvas, W: int, H: int, title: str, text: str, seconds: float | None = None, tags=()) -> int:
    """The opening screen of every stage. Returns the id of the status line (caller updates it)."""
    m = _DUR.search(title)
    if m:
        title, seconds = title[:m.start()], float(m.group(1))
    body, keys = split_intro(text)
    cx = W / 2
    kick = "ELEMENTFROMGAZE" + (f"   ·   {seconds:.0f} S" if seconds and seconds < 600 else "")
    group = "intro-flow"
    tg = (group,) + tuple(tags)
    k = c.create_text(cx, 0, text=kick, fill=ACCENT, font=(MONO, 10), anchor="n", tags=tg)
    t = c.create_text(cx, _below(c, k, 14), text=title, fill=INK, font=(SANS_LIGHT, 34), anchor="n", tags=tg)
    y = _below(c, t, 22)
    c.create_line(cx - 24, y, cx + 24, y, fill=ACCENT, width=1, tags=tg)
    b = c.create_text(cx, y + 26, text=body, fill=DIM, font=(SANS, 14), justify="center", anchor="n", tags=tg)
    y = _below(c, b, 52)
    if keys:   # draw at x=0, measure, then shift to centre
        kg = (group, "keys") + tuple(tags)
        x = 0.0
        for kk, v in keys:
            x = keycap(c, x, y, kk, v, tags=kg) + 34
        bb = c.bbox("keys")
        if bb:
            c.move("keys", cx - (bb[0] + bb[2]) / 2, 0)
        c.dtag("keys", "keys")
    bb = c.bbox(group)     # centre the whole block a little above the middle
    if bb:
        c.move(group, 0, H * 0.46 - (bb[1] + bb[3]) / 2)
    c.dtag(group, group)
    return c.create_text(cx, H - 44, text="", fill=FAINT, font=(MONO, 10), tags=tags)


def status_text(face: bool, perf: str, extra: str = "") -> str:
    return ("●  cara detectada" if face else "○  no veo tu cara: más luz o acércate") + f"   ·   {perf}" + (f"   ·   {extra}" if extra else "")


def target(c: tk.Canvas, x, y, *, settle: float | None = None, light_bg=False, head=False, tags="target") -> None:
    """The point to look at: a copper disc with a pinhole (a precise place for the eye to land),
    and while settling a ring that closes onto it. Head phases: a steel double ring."""
    if head:
        c.create_oval(x - 26, y - 26, x + 26, y + 26, outline=STEEL, width=2, tags=tags)
        c.create_oval(x - 34, y - 34, x + 34, y + 34, outline=STEEL, width=1, dash=(3, 5), tags=tags)
    elif settle is not None:
        r = 9 + 22 * max(0.0, 1 - settle)
        c.create_oval(x - r, y - r, x + r, y + r, outline=ACCENT, width=2, tags=tags)
    c.create_oval(x - 7, y - 7, x + 7, y + 7, fill=ACCENT, outline="", tags=tags)
    hole = "#0B0B0C" if light_bg else "#FFFFFF"
    c.create_oval(x - 1.6, y - 1.6, x + 1.6, y + 1.6, fill=hole, outline="", tags=tags)


def ring(c: tk.Canvas, x, y, r=15, light_bg=False, tags="target") -> None:
    """A ring to follow with the eyes (pursuits, the tracking test)."""
    ink = "#141414" if light_bg else INK
    c.create_oval(x - r, y - r, x + r, y + r, outline=ink, width=3, tags=tags)
    c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=ink, outline="", tags=tags)


def message(c: tk.Canvas, W: int, H: int, text: str, kicker: str = "", tags="target") -> None:
    if kicker:
        c.create_text(W / 2, H / 2 - 70, text=kicker.upper(), fill=ACCENT, font=(MONO, 10), tags=tags)
    c.create_text(W / 2, H / 2, text=text, fill=INK, font=(SANS_LIGHT, 22), justify="center",
                  width=min(W - 200, 1000), tags=tags)


def progress(c: tk.Canvas, W: int, frac: float, light_bg=False, tags="progress") -> None:
    """A hairline across the top edge: how much of the stage is done (peripheral, never a target)."""
    c.delete(tags)
    c.create_line(0, 1, W, 1, fill="#C9C6C0" if light_bg else RULE, width=2, tags=tags)
    c.create_line(0, 1, W * max(0.0, min(1.0, frac)), 1, fill=ACCENT, width=2, tags=tags)


def warning(c: tk.Canvas, W: int, H: int, text: str, tags="quality") -> None:
    """Can't see the eyes: a thin red frame and one line, so it is noticed at once."""
    c.create_rectangle(3, 3, W - 3, H - 3, outline=BAD, width=4, tags=tags)
    t = c.create_text(W / 2, 40, text=text, fill="#FFFFFF", font=(SANS_SEMI, 12), tags=tags)
    x0, y0, x1, y1 = c.bbox(t) or (W / 2 - 80, 30, W / 2 + 80, 50)
    pill = round_rect(c, x0 - 16, y0 - 7, x1 + 16, y1 + 7, r=(y1 - y0 + 14) / 2, fill=BAD, outline="", tags=tags)
    c.tag_lower(pill, t)


def result(c: tk.Canvas, W: int, H: int, label: str, value: str, lines: list[str], tags=()) -> None:
    """End screen of a test: one big number, the detail in grey monospace, how to close."""
    cx, group = W / 2, "result-flow"
    tg = (group,) + tuple(tags)
    k = c.create_text(cx, 0, text=label.upper(), fill=ACCENT, font=(MONO, 10), anchor="n", tags=tg)
    v = c.create_text(cx, _below(c, k, 8), text=value, fill=INK, font=(SANS_LIGHT, 64), anchor="n", tags=tg)
    y = _below(c, v, 18)
    c.create_line(cx - 24, y, cx + 24, y, fill=ACCENT, tags=tg)
    t = c.create_text(cx, y + 28, text="\n".join(lines), fill=DIM, font=(MONO, 13), justify="left", anchor="n", tags=tg)
    end = keycap(c, 0, _below(c, t, 56), "ESC", "cerrar", tags=(group, "esc") + tuple(tags))
    c.move("esc", cx - end / 2, 0)
    c.dtag("esc", "esc")
    bb = c.bbox(group)
    if bb:
        c.move(group, 0, H * 0.46 - (bb[1] + bb[3]) / 2)
    c.dtag(group, group)


def button(c: tk.Canvas, x, y, w, h, label: str, lit=True, tags="target") -> None:
    """A UI-like button to look at (the warm-up's final check, the button test)."""
    round_rect(c, x - w / 2, y - h / 2, x + w / 2, y + h / 2, r=min(10, h / 4),
               fill=ACCENT if lit else PANEL, outline=ACCENT_HI if lit else RULE, width=1, tags=tags)
    c.create_text(x, y, text=label, fill="#1A1210" if lit else DIM, font=(SANS_SEMI, -max(11, int(h * 0.36))), tags=tags)   # negative = pixels, immune to DPI scaling
