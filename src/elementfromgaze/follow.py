"""`efg follow`: calibrate by looking at your own mouse cursor, with the error shrinking live.

You move the mouse wherever you like and keep looking at it. Every frame where the cursor is
slow (the eye leads a fast-moving cursor, so fast moments are recorded but not trusted) is a
label. A coverage grid shows which parts of the screen still lack data; a blue ring shows what
the model currently predicts, and the model refits every few seconds on everything so far, so
you watch the ring converge onto the cursor.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from . import screen
from .app import INTRO_BG, Segment, Stage, train
from .filters import FixationFilter
from .implicit import cursor_position
from .tracker import FOLLOW_SLOW_PX_S, USER_MODEL, GazeModel, load_sessions

GRID_X, GRID_Y = 8, 5
CELL_NEED = 30           # slow, valid samples per cell to call it covered
REFIT_EVERY_S = 4.0
PRIOR_KEEP = 2500        # older sessions kept in each live refit; the new data is all kept


def _lerp(c0: str, c1: str, f: float) -> str:
    a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * f):02x}" for x, y in zip(a, b))


class FollowStage(Stage):
    title = "Calibración siguiendo el ratón"
    intro_text = (
        "Mueve el ratón por TODA la pantalla y míralo siempre (el punto rojo es tu cursor).\n"
        "Haz pausas cortas sobre sitios distintos: esas pausas son las mejores etiquetas.\n"
        "Rellena las casillas, sobre todo las de abajo y las esquinas. El aro azul es lo que el\n"
        "modelo cree que miras: verás cómo se acerca al cursor a medida que aprende.\n\n"
        "ESPACIO para empezar y para terminar · ESC para salir sin guardar"
    )

    def __init__(self) -> None:
        screen.make_dpi_aware()
        self.cursor: "deque[tuple[float, float, float]]" = deque(maxlen=400)
        self.cells = np.zeros((GRID_Y, GRID_X), int)
        self.X_new: list[np.ndarray] = []
        self.Y_new: list[np.ndarray] = []
        self.P_new: list[np.ndarray] = []
        self.live: "deque[tuple[float, np.ndarray, np.ndarray]]" = deque(maxlen=200)
        self.model = GazeModel.load() if USER_MODEL.exists() else None
        self.fits = 0
        self._fitting = False
        self._last_fit = time.monotonic()
        self._ring = FixationFilter(radius=90)
        self._ring_p: tuple[float, float] | None = None
        # With an eye network, refits fine-tune its top layers live (replaying older sessions so it
        # doesn't forget them); without one, a fast feature model is refitted from scratch.
        self.tune = self.model is not None and getattr(self.model, "appearance", None) is not None
        try:
            prior = load_sessions(with_patches=self.tune)
            rows = np.where(prior["has_P"])[0] if self.tune else np.arange(len(prior["X"]))
            idx = rows[np.linspace(0, len(rows) - 1, min(PRIOR_KEEP, len(rows))).astype(int)]
            self.prior = (prior["X"][idx], prior["Y"][idx], prior["P"][idx] if self.tune else None)
        except SystemExit:
            self.prior = None
            self.tune = False
        self.ppd = screen.px_per_degree()
        super().__init__([Segment("follow", 3600.0, self._cursor_at, 0.0, bg=INTRO_BG)])
        self.root.bind("<space>", lambda e: self._start() if self.started_at is None else self._done())
        self._grid_drawn_at = 0.0

    # --- cursor history ---------------------------------------------------
    def _poll_cursor(self) -> None:
        x, y = cursor_position()
        self.cursor.append((time.monotonic(), x, y))

    def _cursor_near(self, t_abs: float):
        if not self.cursor:
            return None
        return min(self.cursor, key=lambda c: abs(c[0] - t_abs))

    def _cursor_at(self, lt: float) -> tuple[float, float]:
        c = self._cursor_near(self.started_at + lt)
        return (c[1], c[2]) if c else (self.W / 2, self.H / 2)

    def _speed_at(self, t_abs: float) -> float:
        pts = [c for c in self.cursor if abs(c[0] - t_abs) <= 0.08]
        if len(pts) < 2:
            return 0.0
        dt = pts[-1][0] - pts[0][0]
        return float(np.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]) / dt) if dt > 1e-3 else 0.0

    # --- Stage hooks ------------------------------------------------------
    def extra_for(self, s, seg, lt) -> dict:
        return {"cursor_speed": self._speed_at(self.started_at + lt)}

    def on_sample(self, s, target, seg, valid) -> None:
        slow = self._speed_at(s.t) < FOLLOW_SLOW_PX_S
        if valid and slow:
            self.X_new.append(s.features)
            self.Y_new.append(np.asarray(target, np.float32))
            self.P_new.append(s.patches)
            cx = min(int(target[0] / self.W * GRID_X), GRID_X - 1)
            cy = min(int(target[1] / self.H * GRID_Y), GRID_Y - 1)
            if 0 <= cx and 0 <= cy:
                self.cells[cy, cx] += 1
        if self.model is not None and valid:
            p = self.model.predict_full(s.features, s.patches)[0]
            self._ring_p = self._ring(float(p[0]), float(p[1]))
            if slow:
                self.live.append((s.t, p, np.asarray(target, np.float32)))
        self._maybe_refit()

    def _maybe_refit(self) -> None:
        if self._fitting or time.monotonic() - self._last_fit < REFIT_EVERY_S or len(self.X_new) < 60:
            return
        self._fitting = True
        Xn, Yn = np.array(self.X_new), np.array(self.Y_new)
        Pn = np.array(self.P_new) if self.tune else None

        def work():
            try:
                if self.tune:
                    import copy

                    from .appearance import finetune_copy
                    # half new data (all of it, recent first), half replay of older sessions
                    k = min(len(Xn), 1500)
                    rep = np.random.default_rng(self.fits).choice(len(self.prior[0]), min(k, len(self.prior[0])), replace=False)
                    X = np.vstack([Xn[-k:], self.prior[0][rep]])
                    Y = np.vstack([Yn[-k:], self.prior[1][rep]])
                    P = np.concatenate([Pn[-k:], self.prior[2][rep]])
                    tuned = finetune_copy(self.model.appearance, P, X, Y, steps=40)
                    m = copy.copy(self.model)
                    m.appearance = tuned
                    self.model = m          # atomic swap: the ring uses the new network from now on
                    self.fits += 1
                    return
                if self.prior is not None:
                    X, Y = np.vstack([self.prior[0], Xn]), np.vstack([self.prior[1], Yn])
                else:
                    X, Y = Xn, Yn
                # fast member only, so the ring updates every few seconds; `efg train` fits the full ensemble
                self.model = GazeModel(kind="linkrr").fit(X, Y, (self.W, self.H))
                self.fits += 1
            finally:
                self._last_fit = time.monotonic()
                self._fitting = False

        threading.Thread(target=work, daemon=True).start()

    def _done(self) -> None:
        self._bounds[-1] = time.monotonic() - self.started_at   # ends the open segment next tick

    # --- drawing ----------------------------------------------------------
    def _tick(self) -> None:
        self._poll_cursor()
        super()._tick()

    def _draw(self, seg: Segment, lt: float) -> None:
        c = self.canvas
        now = time.monotonic()
        if now - self._grid_drawn_at > 0.25:
            self._grid_drawn_at = now
            c.delete("grid")
            cw, ch = self.W / GRID_X, self.H / GRID_Y
            for j in range(GRID_Y):
                for i in range(GRID_X):
                    f = min(self.cells[j, i] / CELL_NEED, 1.0)
                    c.create_rectangle(i * cw + 2, j * ch + 2, (i + 1) * cw - 2, (j + 1) * ch - 2,
                                       fill=_lerp(INTRO_BG, "#1f5a35", f), outline="#2c2c2c", tags="grid")
            c.tag_lower("grid")
            recent = [np.linalg.norm(p - t) for (ts, p, t) in self.live if now - ts < 3.0]
            err = f"{np.median(recent) / self.ppd:.1f}°" if len(recent) > 10 else "—"
            done = int((self.cells >= CELL_NEED).sum())
            c.delete("hud")
            c.create_text(self.W / 2, 22, fill="#d0d0d0", font=("Segoe UI", 13), tags="hud",
                          text=f"casillas {done}/{GRID_X * GRID_Y} · error ahora {err} · "
                               f"modelo actualizado {self.fits} veces · ESPACIO terminar · ESC cancelar")
        c.delete("target")
        if self._ring_p is not None:
            x, y = self._ring_p
            c.create_oval(x - 18, y - 18, x + 18, y + 18, outline="#2fa8e0", width=3, tags="target")
        x, y = self.cursor[-1][1:] if self.cursor else (self.W / 2, self.H / 2)
        c.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#ff3b30", outline="", tags="target")
        c.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5, fill="#ffffff", outline="", tags="target")


def follow(note: str = "") -> None:
    stage = FollowStage()
    stage.run()
    if stage.aborted or not len(stage.rec):
        print("cancelado, no se guarda nada")
        return
    W, H = stage.W, stage.H
    path = stage.rec.save(W, H, "follow", note, stage.worker.stats())
    covered = int((stage.cells >= CELL_NEED).sum())
    print(f"guardadas {len(stage.rec)} muestras ({len(stage.X_new)} lentas y válidas, "
          f"{covered}/{GRID_X * GRID_Y} casillas) en {path}")
    train()
