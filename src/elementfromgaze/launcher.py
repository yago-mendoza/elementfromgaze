"""efg app: a small window and a tray icon that start every mode with one click.

The launcher never touches the camera itself: each action runs `python -m elementfromgaze ...` as a
child process (one at a time), streams its output into the window and stops it on request. Only
the child it started is ever stopped: the overlay of `dot` / `control` gets a WM_CLOSE (the same
clean exit as Ctrl+Alt+G, so the session's clicks and logs are saved), anything else is terminated.
The tray icon needs pystray; without it the window works alone.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from collections import deque
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "data" / "launcher.json"
SERVER = "http://127.0.0.1:8770"
WIN = sys.platform == "win32"

BG = "#0D0D0F"
HOVER = "#151518"
LINE = "#1B1B1F"
TEXT = "#ECE9E4"
MUTED = "#8B8A91"
FAINT = "#56555B"
ACCENT = "#C8643B"
ACCENT_DIM = "#5E3323"
LOG_BG = "#111114"
LOG_FG = "#706F76"
SANS, SANS_B, MONO = "Segoe UI", "Segoe UI Semibold", "Cascadia Mono"

# key: (subcommand args, status name, takes --rewarm, serves the web pages)
MODES = {
    "control": (["control", "--dot", "--serve"], "control del escritorio", True, True),
    "dot": (["dot", "--serve"], "solo el punto", True, True),
    "gesture": (["gesture"], "calibración del parpadeo", False, False),
    "targets": (["targets"], "prueba de botones", True, False),
    "track": (["track"], "prueba del anillo", True, False),
}
SERVER_LINE = "servidor de mirada en"     # printed by overlay.dot once the server is up
STOP_GRACE_S = 10.0                       # time for the overlay to save its session after WM_CLOSE


# --- drawing -------------------------------------------------------------------------------------

def _rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _mix(a: str, b: str, t: float) -> str:
    ca, cb = _rgb(a), _rgb(b)
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def eye_image(size: int, active: bool = False, plate: bool = False) -> Image.Image:
    """The mark: an almond eye in copper. `active` fills it; `plate` puts it on a dark rounded tile."""
    ss = 4
    S = size * ss
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    accent = _rgb(ACCENT) + (255,)
    hole = _rgb(BG) + (255,) if plate else (0, 0, 0, 0)
    if plate:
        d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.24), fill=_rgb("#17171A") + (255,))
    cx = cy = S / 2
    w = S * (0.36 if plate else 0.47)
    h = w * 0.60
    up = [(cx + x * w, cy - h * (1 - x * x) ** 1.1) for x in (-1 + 2 * i / 48 for i in range(49))]
    lo = [(cx + x * w, cy + h * (1 - x * x) ** 1.1) for x in (1 - 2 * i / 48 for i in range(49))]
    stroke = max(ss, int(S * (0.065 if plate else 0.085)))
    if active:
        d.polygon(up + lo, fill=accent)
    d.line(up + lo + [up[0], up[1]], fill=accent, width=stroke, joint="curve")
    r = h * 0.62
    if active:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=hole)
        r2 = r * 0.5
        d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=accent)
    else:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=accent)
        r2 = r * 0.36
        d.ellipse([cx - r2 + r * 0.18, cy - r2 - r * 0.18, cx + r2 + r * 0.18, cy + r2 - r * 0.18], fill=hole)
    return im.resize((size, size), Image.LANCZOS)


def toggle_image(on: bool, s: float) -> Image.Image:
    ss = 4
    w, h = round(30 * s), round(16 * s)
    W, H = w * ss, h * ss
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    lw = max(ss, round(1.1 * s * ss))
    d.rounded_rectangle([lw // 2, lw // 2, W - 1 - lw // 2, H - 1 - lw // 2], radius=H // 2,
                        fill=_rgb("#26170F" if on else "#18181C") + (255,),
                        outline=_rgb(ACCENT if on else "#34343A") + (255,), width=lw)
    k = H - 2 * round(3.2 * s * ss)
    x0 = W - (H - k) // 2 - k if on else (H - k) // 2
    y0 = (H - k) // 2
    d.ellipse([x0, y0, x0 + k, y0 + k], fill=_rgb(ACCENT if on else "#66656C") + (255,))
    return im.resize((w, h), Image.LANCZOS)


def dot_image(color: str, s: float, glow: float = 0.0) -> Image.Image:
    ss = 4
    n = round(14 * s)
    N = n * ss
    im = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if glow > 0:
        d.ellipse([0, 0, N - 1, N - 1], fill=_rgb(color) + (round(70 * glow),))
    r = N * 0.2
    d.ellipse([N / 2 - r, N / 2 - r, N / 2 + r, N / 2 + r], fill=_rgb(color) + (255,))
    return im.resize((n, n), Image.LANCZOS)


# --- Windows helpers -----------------------------------------------------------------------------

def _dpi_aware() -> None:
    if not WIN:
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("elementfromgaze.launcher")
    except Exception:
        pass


def _dark_frame(root: tk.Tk) -> None:
    """Dark title bar whose colour matches the window (Windows 11; harmless elsewhere)."""
    if not WIN:
        return
    try:
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        dwm = ctypes.windll.dwmapi.DwmSetWindowAttribute
        one = ctypes.c_int(1)
        dwm(hwnd, 20, ctypes.byref(one), ctypes.sizeof(one))
        r, g, b = _rgb(BG)
        cap = ctypes.c_int(r | g << 8 | b << 16)
        dwm(hwnd, 35, ctypes.byref(cap), ctypes.sizeof(cap))
        r, g, b = _rgb("#232327")
        bor = ctypes.c_int(r | g << 8 | b << 16)
        dwm(hwnd, 34, ctypes.byref(bor), ctypes.sizeof(bor))
    except Exception:
        pass


def window_rect(root: tk.Tk) -> tuple[int, int, int, int]:
    """Screen rectangle of the whole window, title bar included (for a screenshot of it alone)."""
    if WIN:
        try:
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            rc = wt.RECT()
            ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rc), ctypes.sizeof(rc))
            return rc.left, rc.top, rc.right, rc.bottom
        except Exception:
            pass
    x, y = root.winfo_rootx(), root.winfo_rooty()
    return x, y, x + root.winfo_width(), y + root.winfo_height()


def _work_area() -> tuple[int, int, int, int] | None:
    if not WIN:
        return None
    rc = wt.RECT()
    if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rc), 0):
        return rc.left, rc.top, rc.right, rc.bottom
    return None


class _PE32(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


def process_tree(pid: int) -> set[int]:
    """`pid` and its descendants. A venv's python.exe is a stub that runs the real interpreter as a
    child, so the tracker's windows belong to a grandchild of the process the launcher started."""
    k = ctypes.windll.kernel32
    k.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    snap = k.CreateToolhelp32Snapshot(0x2, 0)
    parent: dict[int, int] = {}
    try:
        e = _PE32()
        e.dwSize = ctypes.sizeof(e)
        ok = k.Process32FirstW(ctypes.c_void_p(snap), ctypes.byref(e))
        while ok:
            parent[e.th32ProcessID] = e.th32ParentProcessID
            ok = k.Process32NextW(ctypes.c_void_p(snap), ctypes.byref(e))
    finally:
        k.CloseHandle(ctypes.c_void_p(snap))
    tree, grew = {pid}, True
    while grew:
        grew = False
        for c, par in parent.items():
            if par in tree and c not in tree and c != par:
                tree.add(c)
                grew = True
    return tree


def overlay_windows(pid: int) -> list[int]:
    """Visible, layered, screen-sized top-level windows of `pid` or its children: the gaze overlay."""
    if not WIN:
        return []
    pids = process_tree(pid)
    u = ctypes.windll.user32
    u.GetWindowLongW.restype = ctypes.c_long
    sw, sh = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def each(hwnd, _):
        owner = wt.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value in pids and u.IsWindowVisible(hwnd) and u.GetWindowLongW(hwnd, -20) & 0x80000:
            rc = wt.RECT()
            u.GetWindowRect(hwnd, ctypes.byref(rc))
            if rc.right - rc.left >= 0.8 * sw and rc.bottom - rc.top >= 0.8 * sh:
                found.append(hwnd)
        return True

    u.EnumWindows(each, 0)
    return found


# --- the child process ---------------------------------------------------------------------------

def _child_python() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


class Runner:
    """One child at a time. Output lines and the exit arrive on `events` as ('line'|'exit', proc, x)."""

    def __init__(self, events: queue.Queue) -> None:
        self.events = events
        self.proc: subprocess.Popen | None = None
        self.stopping = False

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, argv: list[str]) -> subprocess.Popen:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if WIN else 0
        proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags)
        self.proc, self.stopping = proc, False
        threading.Thread(target=self._read, args=(proc,), daemon=True).start()
        return proc

    def _read(self, proc: subprocess.Popen) -> None:
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\r\n").split("\r")[-1].rstrip()
            if line.strip():
                self.events.put(("line", proc, line))
        self.events.put(("exit", proc, proc.wait()))

    def stop(self) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None or self.stopping:
            return
        self.stopping = True
        threading.Thread(target=self._stop, args=(proc,), daemon=True).start()

    @staticmethod
    def _stop(proc: subprocess.Popen) -> None:
        wins = overlay_windows(proc.pid)
        for hwnd in wins:       # same path as Ctrl+Alt+G: root.destroy(), then the session is saved
            ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)
        if wins:
            try:
                proc.wait(STOP_GRACE_S)
                return
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is None:
            proc.kill()


# --- the window ----------------------------------------------------------------------------------

class Row:
    """A full-width clickable line: thin mark, title, one-line explanation, action text on the right."""

    def __init__(self, app: "Launcher", parent: tk.Widget, title: str, desc: str,
                 actions: list[tuple[str, callable]] | None = None, on_click=None) -> None:
        self.app = app
        px = app.px
        self.on_click = on_click
        self.frame = tk.Frame(parent, bg=BG, cursor="hand2" if on_click else "")
        self.mark = tk.Frame(self.frame, bg=BG, width=max(2, px(2)))
        self.mark.pack(side="left", fill="y")
        self.right = tk.Frame(self.frame, bg=BG)
        self.right.pack(side="right", padx=(px(8), px(16)))
        self.body = tk.Frame(self.frame, bg=BG)
        self.body.pack(side="left", fill="both", expand=True, padx=(px(14), 0), pady=(px(6), px(7)))
        self.title = tk.Label(self.body, text=title, font=(SANS_B, 11), fg=TEXT, bg=BG, anchor="w")
        self.title.pack(fill="x")
        self.desc = tk.Label(self.body, text=desc, font=(SANS, 9), fg=MUTED, bg=BG, anchor="w")
        self.desc.pack(fill="x")
        self.links: list[tk.Label] = []
        self.active = False
        self.hover = False
        for text, fn in actions or []:
            self.add_link(text, fn)
        self._clickables = [self.frame, self.mark, self.body, self.title, self.desc, self.right]
        for w in self._clickables:
            w.bind("<Enter>", self._enter, add="+")
            w.bind("<Leave>", self._leave, add="+")
            if on_click:
                w.bind("<Button-1>", lambda e: self.on_click(), add="+")

    def add_link(self, text: str, fn) -> tk.Label:
        lab = tk.Label(self.right, text=text, font=(SANS, 9), fg=FAINT, bg=BG, cursor="hand2",
                       padx=self.app.px(2))
        lab.pack(side="left", padx=(self.app.px(10), 0))
        lab.bind("<Enter>", lambda e: (self._enter(e), lab.configure(fg=ACCENT)), add="+")
        lab.bind("<Leave>", lambda e: (self._leave(e), self.paint()), add="+")
        lab.bind("<Button-1>", lambda e: fn())
        self.links.append(lab)
        return lab

    def _enter(self, _e=None) -> None:
        if not self.hover:
            self.hover = True
            self.paint()

    def _leave(self, _e=None) -> None:
        self.frame.after(15, self._check_leave)

    def _check_leave(self) -> None:
        w = self.frame.winfo_containing(*self.frame.winfo_pointerxy())
        while w is not None and w is not self.frame:
            w = w.master
        if w is None and self.hover:
            self.hover = False
            self.paint()

    def paint(self) -> None:
        bg = HOVER if self.hover else BG
        self.mark.configure(bg=ACCENT if self.active else (ACCENT_DIM if self.hover else bg))
        stack = [self.body, self.right]
        self.frame.configure(bg=bg)
        while stack:
            w = stack.pop()
            try:
                w.configure(bg=bg)
            except tk.TclError:
                pass
            stack.extend(w.winfo_children())
        under = self.frame.winfo_containing(*self.frame.winfo_pointerxy()) if self.hover else None
        for lab in self.links:
            live = lab.cget("text") == "En marcha"
            lab.configure(fg=ACCENT if live or lab is under else (MUTED if self.hover else FAINT))


class Launcher:
    def __init__(self, tray: bool = True) -> None:
        _dpi_aware()
        self.events: queue.Queue = queue.Queue()
        self.runner = Runner(self.events)
        self.key: str | None = None          # mode of the running child
        self.serving = False                 # the child's web server is up
        self.pending_url: str | None = None  # open once the server is up
        self.next: tuple[str, str | None] | None = None   # start this when the current child exits
        self.quitting = False
        self.log: deque[tuple[str, str]] = deque(maxlen=6)
        cfg = self._load()
        self.hd = bool(cfg.get("hd", True))
        self.click = bool(cfg.get("click", False))
        self.rewarm = False                  # one shot: applies to the next start only

        self.root = root = tk.Tk()
        root.withdraw()
        self.s = root.winfo_fpixels("1i") / 96.0
        root.title("elementfromgaze")
        root.configure(bg=BG)
        root.resizable(False, False)
        self._icons = {"plate": ImageTk.PhotoImage(eye_image(self.px(64), plate=True)),
                       "head": ImageTk.PhotoImage(eye_image(self.px(24))),
                       "on": ImageTk.PhotoImage(toggle_image(True, self.s)),
                       "off": ImageTk.PhotoImage(toggle_image(False, self.s)),
                       "idle": ImageTk.PhotoImage(dot_image("#47464C", self.s))}
        self._pulse = [ImageTk.PhotoImage(dot_image(_mix(ACCENT_DIM, ACCENT, 0.5 + 0.5 * math.sin(i / 16 * 2 * math.pi)),
                                                    self.s, glow=0.5 + 0.5 * math.sin(i / 16 * 2 * math.pi)))
                       for i in range(16)]
        self._pulse_i = 0
        root.iconphoto(True, self._icons["plate"])
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.hide)
        self.tray = self._start_tray() if tray else None
        self._refresh()
        self._place()
        root.deiconify()
        _dark_frame(root)
        root.after(50, self._pump)
        root.after(80, self._animate)

    def px(self, v: float) -> int:
        return round(v * self.s)

    # --- layout ----------------------------------------------------------------------------------

    def _build(self) -> None:
        px, root = self.px, self.root
        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True, padx=px(18), pady=(px(16), px(14)))
        outer.configure(width=px(440))

        head = tk.Frame(outer, bg=BG)
        head.pack(fill="x", padx=px(16), pady=(0, px(4)))
        tk.Label(head, image=self._icons["head"], bg=BG).pack(side="left", padx=(0, px(12)))
        names = tk.Frame(head, bg=BG)
        names.pack(side="left", fill="x")
        tk.Label(names, text="elementfromgaze", font=(SANS_B, 13), fg=TEXT, bg=BG, anchor="w").pack(fill="x")
        tk.Label(head, text="mirada con la webcam", font=(SANS, 9), fg=FAINT, bg=BG).pack(
            side="right", pady=(px(3), 0))

        self.rows: dict[str, Row] = {}
        self._section(outer, "Mirada")
        r = self._row(outer, "control", "Controlar el escritorio con la mirada",
                      "El puntero salta a los botones de cualquier app.", "Iniciar")
        sub = tk.Frame(r.body, bg=BG)
        sub.pack(fill="x", pady=(px(5), px(1)))
        self.click_img = tk.Label(sub, image=self._icons["off"], bg=BG, cursor="hand2")
        self.click_img.pack(side="left")
        self.click_lab = tk.Label(sub, text="Clic con parpadeo largo", font=(SANS, 9), fg=MUTED, bg=BG,
                                  cursor="hand2")
        self.click_lab.pack(side="left", padx=(px(8), 0))
        for w in (self.click_img, self.click_lab):
            w.bind("<Button-1>", lambda e: self.toggle_click())
            w.bind("<Enter>", r._enter, add="+")
            w.bind("<Leave>", r._leave, add="+")
        self._line(outer)
        self._row(outer, "dot", "Solo el punto", "Un punto donde miras, servido también a las páginas web.",
                  "Iniciar")
        self._line(outer)
        self._row(outer, "read", "Prueba de lectura", "Página de lectura en el navegador; arranca el punto si falta.",
                  "Abrir")
        self._line(outer)
        self._row(outer, "demo", "Demo web", "Botones magnéticos, menú circular y texto en el navegador.", "Abrir")

        self._section(outer, "Calibración y pruebas")
        self._row(outer, "gesture", "Calibrar parpadeo", "Unos 20 s: tus parpadeos normales y largos, para el clic.",
                  "Iniciar")
        self._line(outer)
        prec = Row(self, outer, "Pruebas de precisión", "Mira botones que se encienden o sigue un anillo.")
        prec.frame.pack(fill="x")
        self.rows["targets"] = self.rows["track"] = prec
        self.link_targets = prec.add_link("Botones", lambda: self.action("targets"))
        self.link_track = prec.add_link("Anillo", lambda: self.action("track"))

        self._section(outer, "Ajustes")
        opts = tk.Frame(outer, bg=BG)
        opts.pack(fill="x", padx=px(16), pady=(px(6), 0))
        self.hd_img = self._toggle_item(opts, "Cámara HD", "", self.toggle_hd)
        self.rewarm_img = self._toggle_item(opts, "Recalibrar postura", "al arrancar", self.toggle_rewarm)

        foot = tk.Frame(outer, bg=BG)
        foot.pack(fill="x", pady=(px(14), 0))
        tk.Frame(foot, bg=LINE, height=1).pack(fill="x", padx=px(16), pady=(0, px(10)))
        bar = tk.Frame(foot, bg=BG)
        bar.pack(fill="x", padx=px(16))
        self.status_dot = tk.Label(bar, image=self._icons["idle"], bg=BG)
        self.status_dot.pack(side="left")
        self.status = tk.Label(bar, text="En reposo", font=(SANS, 10), fg=MUTED, bg=BG, anchor="w")
        self.status.pack(side="left", padx=(px(8), 0))
        self.stop_box = tk.Frame(bar, bg="#26262B", padx=1, pady=1)
        self.stop_box.pack(side="right")
        self.stop_btn = tk.Label(self.stop_box, text="Detener", font=(SANS_B, 9), fg=FAINT, bg=BG,
                                 padx=px(14), pady=px(4))
        self.stop_btn.pack()
        self.stop_btn.bind("<Button-1>", lambda e: self.stop())
        self.stop_btn.bind("<Enter>", lambda e: self._stop_hover(True))
        self.stop_btn.bind("<Leave>", lambda e: self._stop_hover(False))

        self.logbox = tk.Text(foot, height=4, bg=LOG_BG, fg=LOG_FG, font=(MONO, 8), bd=0, relief="flat",
                              highlightthickness=0, wrap="none", padx=px(12), pady=px(8), cursor="arrow",
                              insertwidth=0, selectbackground="#2A2A30", width=1)
        self.logbox.pack(fill="x", padx=px(16), pady=(px(10), 0))
        self.logbox.tag_configure("cmd", foreground="#9A9AA1")
        self.logbox.tag_configure("note", foreground=ACCENT)
        self.logbox.tag_configure("hint", foreground=FAINT, font=(SANS, 9))
        self._push_all()
        tk.Label(foot, text="Ctrl+Alt+R recalibra · Ctrl+Alt+flechas mueven el punto · Ctrl+Alt+G termina",
                 font=(SANS, 8), fg=FAINT, bg=BG, anchor="w").pack(fill="x", padx=px(16), pady=(px(8), 0))
        # fixed width: the rows' text never stretches the window
        outer.update_idletasks()
        self._size = (px(476), outer.winfo_reqheight() + px(30))

    def _section(self, parent: tk.Widget, text: str) -> None:
        tk.Label(parent, text=text.upper(), font=(SANS_B, 8), fg=FAINT, bg=BG, anchor="w").pack(
            fill="x", padx=self.px(16), pady=(self.px(10), self.px(1)))

    def _line(self, parent: tk.Widget) -> None:
        tk.Frame(parent, bg=LINE, height=1).pack(fill="x", padx=(self.px(16), self.px(16)))

    def _row(self, parent, key, title, desc, act) -> Row:
        r = Row(self, parent, title, desc, [(act, lambda: self.action(key))], on_click=lambda: self.action(key))
        r.frame.pack(fill="x")
        self.rows[key] = r
        return r

    def _toggle_item(self, parent, title, hint, fn) -> tk.Label:
        """Switch + name + a faint hint, side by side; the whole item toggles."""
        px = self.px
        box = tk.Frame(parent, bg=BG, cursor="hand2")
        box.pack(side="left", padx=(0, px(26)))
        img = tk.Label(box, image=self._icons["off"], bg=BG, cursor="hand2")
        img.pack(side="left")
        name = tk.Label(box, text=title, font=(SANS, 10), fg=TEXT, bg=BG, cursor="hand2")
        name.pack(side="left", padx=(px(9), 0))
        tip = tk.Label(box, text=hint, font=(SANS, 9), fg=FAINT, bg=BG, cursor="hand2")
        if hint:
            tip.pack(side="left", padx=(px(6), 0))
        for w in (box, img, name, tip):
            w.bind("<Button-1>", lambda e: fn())
            w.bind("<Enter>", lambda e: tip.configure(fg=MUTED))
            w.bind("<Leave>", lambda e: tip.configure(fg=FAINT))
        return img

    def _place(self) -> None:
        self.root.update_idletasks()
        w, h = self._size
        wa = _work_area()
        if wa:
            x, y = wa[2] - w - self.px(24), max(wa[1], wa[3] - h - self.px(56))
        else:
            x, y = 80, 80
        self.root.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y)}")

    def _stop_hover(self, on: bool) -> None:
        live = self.runner.running
        self.stop_box.configure(bg=ACCENT if on and live else ("#3A3A40" if live else "#202024"))

    # --- actions ---------------------------------------------------------------------------------

    def action(self, key: str) -> None:
        if self.quitting:
            return
        if key in ("read", "demo"):
            url = f"{SERVER}/{key}"
            if self.runner.running and self.key in MODES and MODES[self.key][3] and not self.runner.stopping:
                if self.serving:
                    webbrowser.open(url)
                else:
                    self.pending_url = url
                    self._note("abriré la página cuando el servidor esté listo")
                return
            return self._switch("dot", url)
        if self.runner.running and self.key == key and not self.runner.stopping:
            return
        self._switch(key, None)

    def _switch(self, key: str, url: str | None) -> None:
        if self.runner.running:
            self.next = (key, url)
            self._note("detengo el modo actual")
            self.stop()
        else:
            self._start(key, url)

    def argv(self, key: str) -> list[str]:
        args, _, warm, _ = MODES[key]
        args = list(args)
        if key == "control" and self.click:
            args.append("--click")
        if warm and self.rewarm:
            args.append("--rewarm")
        return ([_child_python(), "-m", "elementfromgaze"] + (["--hd"] if self.hd else []) + args)

    def _start(self, key: str, url: str | None = None, argv: list[str] | None = None) -> None:
        argv = argv or self.argv(key)
        if key in MODES and MODES[key][2] and self.rewarm:
            self.rewarm = False            # one shot
        try:
            self.runner.start(argv)
        except Exception as e:
            self._note(f"no pude arrancar: {e}")
            self.key = None
            self._refresh()
            return
        self.key, self.serving, self.pending_url = key, False, url
        self.log.clear()
        self._push("efg " + " ".join(argv[3:]), "cmd")
        self._refresh()

    def stop(self) -> None:
        if self.runner.running:
            self.runner.stop()
            self._refresh()

    def toggle_click(self) -> None:
        self.click = not self.click
        self._save()
        self._refresh()

    def toggle_hd(self) -> None:
        self.hd = not self.hd
        self._save()
        self._refresh()

    def toggle_rewarm(self) -> None:
        self.rewarm = not self.rewarm
        self._refresh()

    def show(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(200, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

    def hide(self) -> None:
        if self.tray is None:
            return self.quit()
        self.root.withdraw()
        if not getattr(self, "_told_tray", False):
            self._told_tray = True
            try:
                self.tray.notify("Sigue en la bandeja del sistema. Salir está en su menú.", "elementfromgaze")
            except Exception:
                pass

    def quit(self) -> None:
        self.quitting = True
        self.next = None
        if self.runner.running:
            self._note("detengo el modo actual antes de salir")
            self.stop()
            self.root.after(int((STOP_GRACE_S + 3) * 1000), self._final)
        else:
            self._final()

    def _final(self) -> None:
        if self.runner.running:
            self.runner.proc.kill()
        if self.tray is not None:
            try:
                self.tray.stop()
            except Exception:
                pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # --- events from the child and the tray -------------------------------------------------------

    def _pump(self) -> None:
        try:
            while True:
                kind, proc, x = self.events.get_nowait()
                if kind == "call":
                    x()
                elif proc is not self.runner.proc:
                    continue
                elif kind == "line":
                    self._push(x)
                    if SERVER_LINE in x:
                        self.serving = True
                        if self.pending_url:
                            webbrowser.open(self.pending_url)
                            self.pending_url = None
                elif kind == "exit":
                    self._exited(x)
        except queue.Empty:
            pass
        try:
            self.root.after(50, self._pump)
        except tk.TclError:
            pass

    def _exited(self, code: int) -> None:
        was = self.key
        self.key, self.serving, self.pending_url = None, False, None
        if code not in (0, None) and not self.runner.stopping:
            self._note(f"terminó con código {code}")
        elif was:
            self._note("detenido" if self.runner.stopping else "terminado")
        self.runner.stopping = False
        if self.quitting:
            return self._final()
        nxt, self.next = self.next, None
        if nxt:
            self._start(*nxt)
        self._refresh()

    # --- painting --------------------------------------------------------------------------------

    def _push(self, line: str, tag: str = "") -> None:
        self.log.append((line, tag))
        self._push_all()

    def _push_all(self) -> None:
        box = self.logbox
        box.configure(state="normal")
        box.delete("1.0", "end")
        for i, (text, t) in enumerate(self.log):
            box.insert("end", ("\n" if i else "") + text, t or ())
        if not self.log:
            box.insert("end", "Elige una acción; aquí aparece lo que va diciendo.", "hint")
        box.see("end")
        box.configure(state="disabled")

    def _note(self, text: str) -> None:
        self._push(text, "note")

    def _refresh(self) -> None:
        live = self.runner.running
        key = self.key if live else None
        for k, row in self.rows.items():
            if k in ("targets", "track"):
                continue
            row.active = key == k
            row.links[0].configure(text="En marcha" if row.active else ("Abrir" if k in ("read", "demo") else "Iniciar"))
            row.paint()
        prec = self.rows["targets"]
        prec.active = key in ("targets", "track")
        self.link_targets.configure(text="En marcha" if key == "targets" else "Botones")
        self.link_track.configure(text="En marcha" if key == "track" else "Anillo")
        prec.paint()
        self.click_img.configure(image=self._icons["on" if self.click else "off"])
        self.hd_img.configure(image=self._icons["on" if self.hd else "off"])
        self.rewarm_img.configure(image=self._icons["on" if self.rewarm else "off"])
        if live:
            name = MODES[key][1] if key in MODES else "proceso"
            self.status.configure(text=("Deteniendo " if self.runner.stopping else "En marcha: ") + name,
                                  fg=TEXT)
            self.stop_btn.configure(fg=TEXT, cursor="hand2")
        else:
            self.status_dot.configure(image=self._icons["idle"])
            self.status.configure(text="En reposo", fg=MUTED)
            self.stop_btn.configure(fg=FAINT, cursor="")
        self._stop_hover(False)
        if self.tray is not None:
            try:
                self.tray.icon = self._tray_img[bool(live)]
                self.tray.title = "elementfromgaze: " + (MODES[key][1] if live and key in MODES else "en reposo")
                self.tray.update_menu()
            except Exception:
                pass

    def _animate(self) -> None:
        if self.runner.running:
            self._pulse_i = (self._pulse_i + 1) % len(self._pulse)
            self.status_dot.configure(image=self._pulse[self._pulse_i])
        try:
            self.root.after(90, self._animate)
        except tk.TclError:
            pass

    # --- tray ------------------------------------------------------------------------------------

    def _start_tray(self):
        try:
            import pystray
        except Exception:
            return None
        self._tray_img = {False: eye_image(64), True: eye_image(64, active=True)}
        M, SEP = pystray.MenuItem, pystray.Menu.SEPARATOR

        def do(fn):
            return lambda icon, item: self.events.put(("call", None, fn))

        def on(*keys):
            return lambda item: self.runner.running and self.key in keys

        menu = pystray.Menu(
            M("Mostrar ventana", do(self.show), default=True),
            SEP,
            M("Controlar el escritorio", do(lambda: self.action("control")), checked=on("control")),
            M("Solo el punto", do(lambda: self.action("dot")), checked=on("dot")),
            M("Prueba de lectura", do(lambda: self.action("read"))),
            M("Demo web", do(lambda: self.action("demo"))),
            SEP,
            M("Calibrar parpadeo", do(lambda: self.action("gesture")), checked=on("gesture")),
            M("Prueba de botones", do(lambda: self.action("targets")), checked=on("targets")),
            M("Prueba del anillo", do(lambda: self.action("track")), checked=on("track")),
            SEP,
            M("Clic con parpadeo largo", do(self.toggle_click), checked=lambda i: self.click),
            M("Cámara HD", do(self.toggle_hd), checked=lambda i: self.hd),
            M("Recalibrar postura", do(self.toggle_rewarm), checked=lambda i: self.rewarm),
            SEP,
            M("Detener", do(self.stop), enabled=lambda i: self.runner.running),
            M("Salir", do(self.quit)),
        )
        icon = pystray.Icon("elementfromgaze", self._tray_img[False], "elementfromgaze: en reposo", menu)
        threading.Thread(target=icon.run, daemon=True).start()
        return icon

    # --- settings --------------------------------------------------------------------------------

    @staticmethod
    def _load() -> dict:
        try:
            return json.loads(SETTINGS.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self) -> None:
        try:
            SETTINGS.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS.write_text(json.dumps({"hd": self.hd, "click": self.click}), encoding="utf-8")
        except Exception:
            pass

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            if self.runner.running:
                self.runner.proc.kill()


def main() -> None:
    Launcher().run()


if __name__ == "__main__":
    main()
