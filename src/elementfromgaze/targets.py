"""`efg targets`: can you pick a button just by looking at it, and from what size up?

A full screen of buttons (like a UI) in blocks of decreasing size. One button lights up;
you look at it. The live `efg dot` pipeline runs underneath (hidden by default, so the eye
doesn't chase it) and every frame is logged. Per trial we decide which button you meant:

  frame hit   the filtered dot is INSIDE the right button (raw pixel, no help)
  snap hit    the nearest button to the dot is the right one (UI-aware decoding)
  dwell hit   one decision per trial: median dot over the 1 s dwell -> nearest button

Then a size curve from every trial pooled: for a regular grid of buttons h tall and 2h wide,
which share of dwell decisions would land in the right cell. That is the real spec:
"it works for buttons of X° (Y px) and up".

Phase 'tgt' is not a training phase, so this stays an honest test.
"""

from __future__ import annotations

import json
import random
import time

import numpy as np

from . import screen
from .app import INTRO_BG, Segment, Stage, message
from .filters import GazeSmoother
from .overlay import OFF_MARGIN_PX
from .tracker import DATA, GazeModel

SIZES_DEG = (3.0, 2.0, 1.5, 1.0)       # button height; width is twice that
ASPECT = 2.0
GAP = 0.18                             # gap between buttons, as a fraction of the height
TRIAL_S, SETTLE_S = 1.6, 0.6           # look for 1.6 s; the first 0.6 s (saccade + settle) is not judged
TRIALS = 14
WORDS = ("Abrir", "Guardar", "Copiar", "Pegar", "Enviar", "Buscar", "Cerrar", "Editar", "Nuevo", "Borrar",
         "Ver", "Ajustes", "Ayuda", "Salir", "Subir", "Bajar", "Filtrar", "Ordenar", "Compartir", "Imprimir")


class Grid:
    """Buttons of height h (px) tiling the screen, centred, with margins."""

    def __init__(self, W, H, h):
        self.h, self.w, g = h, h * ASPECT, h * GAP
        self.pitch = (self.w + g, self.h + g)
        nx = max(1, int((W - 2 * g) // self.pitch[0]))
        ny = max(1, int((H - 2 * g) // self.pitch[1]))
        x0 = (W - nx * self.pitch[0] + g) / 2 + self.w / 2
        y0 = (H - ny * self.pitch[1] + g) / 2 + self.h / 2
        self.nx, self.ny = nx, ny
        self.centres = np.array([(x0 + i * self.pitch[0], y0 + j * self.pitch[1]) for j in range(ny) for i in range(nx)])

    def inside(self, p) -> int:
        d = np.abs(self.centres - p)
        k = np.where((d[:, 0] <= self.w / 2) & (d[:, 1] <= self.h / 2))[0]
        return int(k[0]) if len(k) else -1

    def nearest(self, p) -> int:
        # nearest in "button units" so a wide button is not penalised horizontally
        d = (self.centres - p) / (self.w, self.h)
        return int(np.argmin((d ** 2).sum(1)))

    def pick(self, n, rng) -> list[int]:
        """n buttons spread over the screen: stratify by a 4x3 coarse grid, then random within cells."""
        cells = {}
        for k, (x, y) in enumerate(self.centres):
            cells.setdefault((int(4 * x / (self.centres[:, 0].max() + 1)), int(3 * y / (self.centres[:, 1].max() + 1))), []).append(k)
        keys = list(cells)
        rng.shuffle(keys)
        out = [rng.choice(cells[keys[i % len(keys)]]) for i in range(n)]
        return out


class TargetsStage(Stage):
    title = "¿Qué botón miras?"
    intro_text = ("Se ilumina un botón: míralo hasta que se ilumine otro.\n"
                  "Los botones se van haciendo más pequeños.\n"
                  "Mira el botón con naturalidad (donde mirarías para pulsarlo).\n\n"
                  "ESPACIO para empezar · ESC para salir sin guardar")

    def __init__(self, W, H, model: GazeModel, show: bool = False, seed: int | None = None):
        self.model, self.show = model, show
        from .adapt import AffineKalman
        self.aff = AffineKalman.seeded(W, H, model)      # same start as `efg dot`
        self.fixf = GazeSmoother(still_s=0.2, meas_sd_px=50, accel_sd_px_s2=1500, lead_s=0.08)   # same as `efg dot`
        ppd = screen.px_per_degree()
        rng = random.Random(seed if seed is not None else time.time_ns())
        self.grids, self.trial = [], {}      # id(segment) -> (grid index, button index)
        segs = []
        for gi, deg in enumerate(SIZES_DEG):
            g = Grid(W, H, deg * ppd)
            g.deg = deg
            g.words = [rng.choice(WORDS) for _ in range(len(g.centres))]
            self.grids.append(g)
            segs.append(message(f"Botones de {deg:g}° ({g.h:.0f} px de alto) · {len(g.centres)} en pantalla", 2.0))
            self.trial[id(segs[-1])] = (gi, -1)
            prev = -1
            for b in g.pick(TRIALS, rng):
                if b == prev:
                    continue
                prev = b
                cx, cy = g.centres[b]
                segs.append(Segment("tgt", TRIAL_S, (lambda t, x=cx, y=cy: (x, y)), SETTLE_S, bg=INTRO_BG))
                self.trial[id(segs[-1])] = (gi, b)
        self._drawn_grid, self._lit = -1, -1
        self._dot: tuple[float, float] | None = None
        self._ids: list[int] = []
        super().__init__(segs)

    # same pipeline as `efg dot` without click learning: network -> eyes gate -> off-screen -> smoother
    def extra_for(self, s, seg, lt) -> dict:
        gi, b = self.trial.get(id(seg), (-1, -1))
        base = self.aff.apply(self.model.predict_full(s.features, s.patches)[0])
        out = (np.nan, np.nan)
        outside = bool(base[0] < -OFF_MARGIN_PX or base[0] > self.W + OFF_MARGIN_PX
                       or base[1] < -OFF_MARGIN_PX or base[1] > self.H + OFF_MARGIN_PX)
        if s.eyes_ok and not outside:
            out = self.fixf(float(np.clip(base[0], 0, self.W)), float(np.clip(base[1], 0, self.H)), s.t)
            self._dot = out
        elif outside:
            self._dot = None
        return {"base": base, "out": np.asarray(out, float), "grid": gi, "button": b}

    def _draw_grid(self, gi):
        c = self.canvas
        c.delete("grid")
        g = self.grids[gi]
        fs = max(7, int(g.h * 0.32))
        self._ids = []
        for (x, y), w in zip(g.centres, g.words):
            r = c.create_rectangle(x - g.w / 2, y - g.h / 2, x + g.w / 2, y + g.h / 2, fill="#2b2b2b",
                                   outline="#3d3d3d", tags="grid")
            c.create_text(x, y, text=w, fill="#8a8a8a", font=("Segoe UI", fs), tags="grid")
            self._ids.append(r)
        self._drawn_grid, self._lit = gi, -1

    def _draw(self, seg: Segment, lt: float) -> None:
        c = self.canvas
        c.delete("target")
        gi, b = self.trial.get(id(seg), (-1, -1))
        if seg.phase == "msg":
            c.delete("grid")
            self._drawn_grid = -1
            c.create_text(self.W / 2, self.H / 2, text=seg.message, fill="#eeeeee", font=("Segoe UI", 20),
                          justify="center", tags="target")
            return
        if gi != self._drawn_grid:
            self._draw_grid(gi)
        if b != self._lit:
            if self._lit >= 0:
                c.itemconfigure(self._ids[self._lit], fill="#2b2b2b", outline="#3d3d3d")
            c.itemconfigure(self._ids[b], fill="#e8a33d", outline="#ffd27a")
            self._lit = b
        if self.show and self._dot is not None:
            g = self.grids[gi]
            k = g.nearest(np.asarray(self._dot))
            x, y = g.centres[k]
            c.create_rectangle(x - g.w / 2 - 3, y - g.h / 2 - 3, x + g.w / 2 + 3, y + g.h / 2 + 3,
                               outline="#4fc3ff", width=3, tags="target")

    def _draw_preview(self, target):
        if self.started_at is None:      # only on the intro: during trials it would cover buttons
            super()._draw_preview(target)

    def finish(self) -> None:
        self.report = summarize(self.rec, self.grids)
        c, W, H = self.canvas, self.W, self.H
        c.delete("all")
        c.configure(bg=INTRO_BG)
        c.create_text(W / 2, H / 2, text=format_report(self.report) + "\n\nESC para cerrar", fill="#eeeeee",
                      font=("Consolas", 15), justify="left")
        self.root.unbind("<space>")


# ---------------------------------------------------------------------------
def trials(rec, grids):
    """One row per trial: grid, button, target, judged dot samples."""
    ph = np.array(rec.cols["phase"])
    lt = np.array(rec.cols["local_t"])
    seg = np.array(rec.cols["seg"])
    eyes = np.array(rec.cols["eyes_ok"])
    out = np.array(rec.extra["out"], float)
    gi = np.array(rec.extra["grid"])
    bi = np.array(rec.extra["button"])
    rows = []
    for s in np.unique(seg[ph == "tgt"]):
        k = (seg == s) & (lt >= SETTLE_S) & eyes & np.isfinite(out).all(1)
        if k.sum() < 5:
            continue
        g, b = int(gi[k][0]), int(bi[k][0])
        rows.append({"grid": g, "button": b, "target": grids[g].centres[b], "dots": out[k], "t": float(np.array(rec.cols["t"])[k][-1])})
    return rows


def size_curve(rows, ppd, sizes_deg=(0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)):
    """Share of dwell decisions inside the right cell of a regular grid of h x 2h buttons."""
    e = np.array([np.median(r["dots"], 0) - r["target"] for r in rows])
    out = {}
    for d in sizes_deg:
        h = d * ppd
        hx, hy = h * ASPECT * (1 + GAP / ASPECT) / 2, h * (1 + GAP) / 2
        out[d] = float(np.mean((np.abs(e[:, 0]) <= hx) & (np.abs(e[:, 1]) <= hy)))
    return out, e


def with_clicks(rows, W, H):
    """What `efg dot` with click learning would have shown: every trial ends in a 'click' on the lit
    button, and the global correction (adapt.AffineKalman) only uses PAST trials."""
    from .adapt import AffineKalman

    aff, out = AffineKalman(W, H), []
    for r in rows:
        out.append(dict(r, dots=np.array([aff.apply(p) for p in r["dots"]])))
        aff.add(np.median(r["dots"], 0), r["target"], r["t"])
    return out


def summarize(rec, grids) -> dict | None:
    if not len(rec) or "out" not in rec.extra:
        return None
    ppd = screen.px_per_degree()
    rows = trials(rec, grids)
    if len(rows) < 5:
        return None
    rep = {"blocks": {}}
    for gi, g in enumerate(grids):
        rr = [r for r in rows if r["grid"] == gi]
        if not rr:
            continue
        frame = np.mean([np.mean([g.inside(p) == r["button"] for p in r["dots"]]) for r in rr])
        snap = np.mean([np.mean([g.nearest(p) == r["button"] for p in r["dots"]]) for r in rr])
        dwell = np.mean([g.nearest(np.median(r["dots"], 0)) == r["button"] for r in rr])
        err = np.median([np.linalg.norm(np.median(r["dots"], 0) - r["target"]) for r in rr]) / ppd
        rep["blocks"][f"{g.deg:g}"] = {"frame": round(float(frame), 3), "snap": round(float(snap), 3),
                                       "dwell": round(float(dwell), 3), "err_deg": round(float(err), 2),
                                       "n": len(rr), "h_px": round(g.h)}
    curve, e = size_curve(rows, ppd)
    rep["curve"] = {f"{k:g}": round(v, 3) for k, v in curve.items()}
    W, H = screen.screen_size_px()
    cl = with_clicks(rows, W, H)[5:]          # after the first 5 clicks
    if len(cl) >= 5:
        c2, e2 = size_curve(cl, ppd)
        rep["curve_clicks"] = {f"{k:g}": round(v, 3) for k, v in c2.items()}
        rep["dwell_err_deg_clicks"] = round(float(np.median(np.linalg.norm(e2, axis=1)) / ppd), 2)
    rep["dwell_err_deg"] = round(float(np.median(np.linalg.norm(e, axis=1)) / ppd), 2)
    rep["bias_px"] = [int(round(v)) for v in np.median(e, 0)]
    rep["spread_deg"] = [round(float(v) / ppd, 2) for v in np.percentile(np.abs(e), 75, axis=0)]
    return rep


def format_report(rep) -> str:
    if not rep:
        return "sin datos suficientes"
    ppd = screen.px_per_degree()
    lines = ["tamaño      en el botón   + imán   decisión 1 s   error",
             "                (frames)  (frames)  (por intento)"]
    for deg, b in rep["blocks"].items():
        lines.append(f"{deg:>4s}° {b['h_px']:4d}px   {100 * b['frame']:5.0f}%   {100 * b['snap']:5.0f}%   "
                     f"{100 * b['dwell']:7.0f}%      {b['err_deg']:.2f}°   (n={b['n']})")
    lines.append("")
    lines.append("curva (todos los intentos, rejilla de botones h × 2h):")
    lines.append("  " + "  ".join(f"{k}°:{100 * v:.0f}%" for k, v in rep["curve"].items()))
    need = [float(k) for k, v in rep["curve"].items() if v >= 0.9]
    if need:
        lines.append(f"  → 90% de aciertos desde botones de {min(need):g}° ({min(need) * ppd:.0f} px) de alto")
    if rep.get("curve_clicks"):
        lines.append(f"aprendiendo de tus clics (simulado, tras 5): error {rep['dwell_err_deg_clicks']}°")
        lines.append("  " + "  ".join(f"{k}°:{100 * v:.0f}%" for k, v in rep["curve_clicks"].items()))
    lines.append(f"error por intento {rep['dwell_err_deg']}° · sesgo ({rep['bias_px'][0]:+d}, {rep['bias_px'][1]:+d}) px"
                 f" · dispersión (x, y) {rep['spread_deg'][0]}°, {rep['spread_deg'][1]}°")
    return "\n".join(lines)


def targets(note: str = "", warm: bool = True, rewarm: bool = False, show: bool = False) -> None:
    from .track import warm_model, warmup

    from .tracker import load_user_model
    model = load_user_model(warm)
    if warm:
        model = warmup(model) if rewarm else warm_model(model)
    screen.make_dpi_aware()
    W, H = screen.screen_size_px()
    stage = TargetsStage(W, H, model, show=show)
    stage.run()
    if not len(stage.rec) or (stage.aborted and not getattr(stage, "report", None)):
        print("cancelado, no se guarda nada")
        return
    rep = getattr(stage, "report", None) or summarize(stage.rec, stage.grids)
    path = stage.rec.save(W, H, "targets", note, stage.worker.stats())
    print(format_report(rep))
    if rep:
        rep |= {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "session": path, "note": note, "show": show}
        with open(DATA / "targets_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rep, ensure_ascii=False) + "\n")
    print("guardada:", path)
