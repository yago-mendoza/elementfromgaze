"""Small, modern notifications for the overlay: a dark rounded card in the bottom-right corner (like
Windows' own), an accent bar, title + one line of detail, fade in / fade out. Click-through and never
takes focus, so it can't interrupt what you are doing."""

from __future__ import annotations

import ctypes
import tkinter as tk

from . import screen

BG, FG, DIM = "#1f2226", "#f2f2f2", "#a9aeb6"
ACCENTS = {"info": "#4fc3ff", "warn": "#e8a33d", "ok": "#34c759"}


class Toaster:
    def __init__(self, root: tk.Tk, W: int, H: int):
        self.root, self.W, self.H = root, W, H
        self.win: tk.Toplevel | None = None
        self._job = None

    def show(self, title: str, body: str = "", kind: str = "info", ms: int = 6000) -> None:
        self._close_now()
        tl = tk.Toplevel(self.root)
        tl.overrideredirect(True)
        tl.attributes("-topmost", True)
        tl.attributes("-alpha", 0.0)
        tl.configure(bg=BG)
        bar = tk.Frame(tl, bg=ACCENTS.get(kind, ACCENTS["info"]), width=4)
        bar.pack(side="left", fill="y")
        box = tk.Frame(tl, bg=BG, padx=16, pady=12)
        box.pack(side="left", fill="both", expand=True)
        tk.Label(box, text=title, bg=BG, fg=FG, font=("Segoe UI Semibold", 11), anchor="w", justify="left").pack(fill="x")
        if body:
            tk.Label(box, text=body, bg=BG, fg=DIM, font=("Segoe UI", 10), anchor="w", justify="left",
                     wraplength=380).pack(fill="x", pady=(3, 0))
        tl.update_idletasks()
        w, h = max(tl.winfo_reqwidth(), 300), tl.winfo_reqheight()
        tl.geometry(f"{w}x{h}+{self.W - w - 24}+{self.H - h - 72}")
        tl.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(tl.winfo_id()) or tl.winfo_id()
        try:     # Windows 11 rounded corners (DWMWA_WINDOW_CORNER_PREFERENCE = 33, DWMWCP_ROUND = 2)
            pref = ctypes.c_int(2)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(pref), ctypes.sizeof(pref))
        except Exception:
            pass
        screen.make_click_through(hwnd)
        self.win = tl
        self._fade(tl, 0.0, 0.96, 0.12, lambda: setattr(self, "_job", self.root.after(ms, lambda: self._fade_out(tl))))

    def _fade(self, tl, a, target, step, done=None) -> None:
        if not tl.winfo_exists():
            return
        a = min(target, a + step) if step > 0 else max(target, a + step)
        tl.attributes("-alpha", a)
        if a != target:
            self.root.after(16, lambda: self._fade(tl, a, target, step, done))
        elif done:
            done()

    def _fade_out(self, tl) -> None:
        if tl.winfo_exists():
            self._fade(tl, float(tl.attributes("-alpha")), 0.0, -0.08, tl.destroy)

    def _close_now(self) -> None:
        if self._job:
            self.root.after_cancel(self._job)
            self._job = None
        if self.win is not None and self.win.winfo_exists():
            self.win.destroy()
        self.win = None
