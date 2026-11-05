"""`efg track`: follow a moving ring while the screen shows exactly what `efg dot` would show.

Continuous ground truth over the whole screen (including the bottom edge, swept line by line),
with the live dot pipeline drawn on top so you see the error as it happens, and every frame
logged: raw network output, filtered dot, target. Half of the pursuit runs with the red dot
HIDDEN: seeing your own estimate tempts the eye to chase it, and comparing shown vs hidden
measures that feedback effect. Phases are 'trackp' (moving) / 'trackf' (static): not used for
training, so this stays an honest test.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

from . import screen
from .app import BGS, INTRO_BG, PURSUIT_LAG_S, Segment, Stage, message
from .filters import GazeSmoother
from .overlay import OFF_MARGIN_PX
from .tracker import DATA, GazeModel


def _ink(bg: str) -> str:
    """Ring colour that stands out on the segment's background (white on white was invisible)."""
    r, g, b = (int(bg[i:i + 2], 16) for i in (1, 3, 5))
    return "#111111" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "#ffffff"


def _lissajous(W, H, offset):
    cx, cy, ax, ay = W / 2, H / 2, W * 0.45, H * 0.44
    return lambda t: (cx + ax * math.sin(2 * math.pi * 0.067 * (t + offset)),
                      cy + ay * math.sin(2 * math.pi * 0.109 * (t + offset) + 0.6))


def _raster(W, H, rows, speed):
    """Horizontal sweeps, alternating direction, one row after another (top to bottom)."""
    x0, x1 = 0.05 * W, 0.95 * W
    L = x1 - x0
    per = L / speed

    def path(t):
        k = min(int(t // per), len(rows) - 1)
        f = min((t - k * per) / per, 1.0)
        x = x0 + f * L if k % 2 == 0 else x1 - f * L
        return x, rows[k] * H

    return path, per * len(rows)


class TrackStage(Stage):
    title = "Seguimiento con el punto real"
    intro_text = ("Sigue con la mirada el ARO todo el rato.\n"
                  "El punto rojo es lo que `dot` cree que miras: no lo persigas, mira siempre el aro.\n"
                  "A ratos el punto rojo se oculta a propósito.\n\n"
                  "ESPACIO para empezar · ESC para salir sin guardar")

    def __init__(self, W, H, model: GazeModel):
        self.model = model
        self.fixf = GazeSmoother(still_s=0.2, meas_sd_px=50, accel_sd_px_s2=1500, lead_s=0.08)   # same as `efg dot`
        self.shown: dict[int, bool] = {}
        self._dot: tuple[float, float] | None = None
        segs, shown = [message("Sigue el ARO. El punto rojo es lo que `dot` cree que miras.", 3.5)], [False]
        segs.append(Segment("trackp", 24.0, _lissajous(W, H, 0.0), 1.0, lag=PURSUIT_LAG_S, bg=BGS[0])); shown.append(True)
        segs.append(Segment("trackp", 24.0, _lissajous(W, H, 7.3), 0.5, lag=PURSUIT_LAG_S, bg=BGS[0])); shown.append(False)
        path, dur = _raster(W, H, [0.06, 0.28, 0.5, 0.72, 0.94], 330.0)
        segs.append(message("Ahora barridos de lado a lado, de arriba abajo.", 2.5)); shown.append(False)
        segs.append(Segment("trackp", dur, path, 1.0, lag=PURSUIT_LAG_S, bg=BGS[2])); shown.append(True)
        segs.append(message("Ahora puntos quietos.", 2.0)); shown.append(False)
        for k, (fx, fy) in enumerate([(0.1, 0.92), (0.5, 0.95), (0.9, 0.92), (0.1, 0.08), (0.5, 0.05), (0.9, 0.08), (0.5, 0.5)]):
            segs.append(Segment("trackf", 2.2, (lambda t, x=fx * W, y=fy * H: (x, y)), 0.9, bg=BGS[k % 2]))
            shown.append(True)
        self._shown = {id(sg): sh for sg, sh in zip(segs, shown)}
        super().__init__(segs)

    # the dot pipeline, without learning: network -> blink/eyes gate -> off-screen -> fixation filter
    def extra_for(self, s, seg, lt) -> dict:
        base = self.model.predict_full(s.features, s.patches)[0]
        out = (np.nan, np.nan)
        outside = bool(base[0] < -OFF_MARGIN_PX or base[0] > self.W + OFF_MARGIN_PX
                       or base[1] < -OFF_MARGIN_PX or base[1] > self.H + OFF_MARGIN_PX)
        if s.eyes_ok and not outside:
            out = self.fixf(float(np.clip(base[0], 0, self.W)), float(np.clip(base[1], 0, self.H)), s.t)
            self._dot = out
        elif outside:
            self._dot = None
        return {"base": base, "out": np.asarray(out, float), "outside": outside,
                "shown": self._shown.get(id(seg), False)}

    def _draw(self, seg: Segment, lt: float) -> None:
        c = self.canvas
        c.delete("target")
        if seg.phase == "msg":
            c.create_text(self.W / 2, self.H / 2, text=seg.message, fill="#eeeeee", font=("Segoe UI", 20),
                          justify="center", tags="target")
            return
        x, y = seg.target(lt)
        ink = _ink(seg.bg)
        c.create_oval(x - 15, y - 15, x + 15, y + 15, outline=ink, width=3, tags="target")
        c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=ink, outline="", tags="target")
        if self._shown.get(id(seg)) and self._dot is not None:
            px, py = self._dot
            c.create_oval(px - 9, py - 9, px + 9, py + 9, fill="#ff2d20", outline="", tags="target")

    def finish(self) -> None:
        self.report = summarize(self.rec, self.W, self.H)
        c, W, H = self.canvas, self.W, self.H
        c.delete("all")
        c.configure(bg=INTRO_BG)
        r = self.report
        if r:
            lines = [f"error mediano del punto {r['all']['median_deg']}° · sesgo medio ({r['all']['bias_px'][0]:+d}, {r['all']['bias_px'][1]:+d}) px"]
            for band in ("arriba", "medio", "abajo"):
                b = r["bands"].get(band)
                if b:
                    lines.append(f"{band:7s}: error {b['median_deg']}° · sesgo ({b['bias_px'][0]:+d}, {b['bias_px'][1]:+d}) px")
            if r.get("shown") and r.get("hidden"):
                lines.append(f"punto visible {r['shown']['median_deg']}° · oculto {r['hidden']['median_deg']}°")
            c.create_text(W / 2, H / 2, text="\n".join(lines) + "\n\nESC para cerrar", fill="#eeeeee",
                          font=("Consolas", 16), justify="left")
        self.root.unbind("<space>")


def _stats(err, bias, ppd):
    return {"median_deg": round(float(np.median(err) / ppd), 2), "n": int(len(err)),
            "bias_px": [int(round(float(v))) for v in bias]}


def summarize(rec, W, H) -> dict | None:
    if not len(rec):
        return None
    ppd = screen.px_per_degree()
    ph = np.array(rec.cols["phase"])
    lt = np.array(rec.cols["local_t"])
    valid = np.array(rec.cols["valid"])
    Y = np.array(rec.cols["targets"], float)
    out = np.array(rec.extra["out"], float)
    shown = np.array(rec.extra["shown"])
    ok = valid & np.isin(ph, ("trackp", "trackf")) & np.isfinite(out).all(1)
    if ok.sum() < 30:
        return None
    d = out[ok] - Y[ok]
    e = np.linalg.norm(d, axis=1)
    rep = {"all": _stats(e, d.mean(0), ppd), "bands": {}}
    yb = Y[ok][:, 1]
    for name, lo, hi in (("arriba", 0, H / 3), ("medio", H / 3, 2 * H / 3), ("abajo", 2 * H / 3, H + 1)):
        k = (yb >= lo) & (yb < hi)
        if k.sum() > 20:
            rep["bands"][name] = _stats(e[k], d[k].mean(0), ppd)
    sp = shown[ok] & (ph[ok] == "trackp")
    hp = ~shown[ok] & (ph[ok] == "trackp")
    if sp.sum() > 20 and hp.sum() > 20:
        rep["shown"] = _stats(e[sp], d[sp].mean(0), ppd)
        rep["hidden"] = _stats(e[hp], d[hp].mean(0), ppd)
    rep["off_screen_frames"] = float(np.mean(np.array(rec.extra["outside"])[ok]))
    return rep


def track(note: str = "", warm: bool = True, rewarm: bool = False) -> None:
    model = GazeModel.load()
    if warm:   # measure what `efg dot` shows, i.e. after the posture warm-up
        model = warmup(model) if rewarm else warm_model(model)
    screen.make_dpi_aware()
    W, H = screen.screen_size_px()
    stage = TrackStage(W, H, model)
    stage.run()
    if not len(stage.rec) or (stage.aborted and not getattr(stage, "report", None)):
        print("cancelado, no se guarda nada")
        return
    rep = getattr(stage, "report", None) or summarize(stage.rec, W, H)
    path = stage.rec.save(W, H, "track", note, stage.worker.stats())
    if rep:
        rep |= {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "session": path, "note": note}
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        with open(DATA / "track_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rep, ensure_ascii=False) + "\n")
    print("guardada:", path)


# ---------------------------------------------------------------------------
class WarmupStage(Stage):
    """~45 s following a ring (no dot shown): fresh labels in the posture you are in right now."""

    title = "Calentamiento (≈50 s)"
    intro_text = ("Sigue con la mirada el ARO hasta que termine.\n"
                  "Siéntate como vas a usar el ordenador: así el modelo se ajusta a tu postura de ahora.\n\n"
                  "ESPACIO para empezar · ESC para saltarlo")

    def __init__(self, W, H):
        path, dur = _raster(W, H, [0.06, 0.3, 0.55, 0.8, 0.95], 380.0)
        segs = [Segment("pursuit", 18.0, _lissajous(W, H, 3.1), 1.0, lag=PURSUIT_LAG_S, bg=BGS[0]),
                Segment("pursuit", dur, path, 0.8, lag=PURSUIT_LAG_S, bg=BGS[2])]
        # corners: the extremes were the worst after warm-up (top-right 3.6°), and pursuit rarely
        # dwells there
        for k, (fx, fy) in enumerate([(0.04, 0.05), (0.96, 0.05), (0.96, 0.95), (0.04, 0.95)]):
            segs.append(Segment("fix", 1.9, (lambda t, x=fx * W, y=fy * H: (x, y)), 0.7, bg=BGS[k % 2]))
        super().__init__(segs)

    def _draw(self, seg: Segment, lt: float) -> None:
        c = self.canvas
        c.delete("target")
        x, y = seg.target(lt)
        ink = _ink(seg.bg)
        c.create_oval(x - 15, y - 15, x + 15, y + 15, outline=ink, width=3, tags="target")
        c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=ink, outline="", tags="target")


WARM_MAX_AGE_S = 30 * 60


def _warm_path():
    from .tracker import USER_MODEL
    return USER_MODEL.with_name(USER_MODEL.stem + "_warm.pkl")


def warm_model(model: GazeModel) -> GazeModel:
    """The warmed-up model from the last 30 min if there is one, else run a warm-up now."""
    from .tracker import USER_MODEL

    p = _warm_path()
    if (p.exists() and time.time() - p.stat().st_mtime < WARM_MAX_AGE_S
            and p.stat().st_mtime > USER_MODEL.stat().st_mtime):      # newer than the base model
        age = (time.time() - p.stat().st_mtime) / 60
        print(f"uso el calentamiento de hace {age:.0f} min (si cambiaste de postura: --rewarm)")
        return GazeModel.load(p)
    return warmup(model)


def warmup(model: GazeModel, rounds: int = 8) -> GazeModel:   # 8 > 4: 1.45° -> 1.25° on 3 tracks
    """Fine-tune the eye network's top layers on a fresh ~45 s pursuit (replaying the calibration
    sessions 1:1). Measured on a far, head-down posture never seen in calibration: 4.3° -> 1.1-1.2°
    on curves and static points not used for the fit."""
    import copy

    from .appearance import finetune_copy
    from .tracker import load_sessions

    if getattr(model, "appearance", None) is None:
        print("sin red de ojos: no hay calentamiento")
        return model
    screen.make_dpi_aware()
    W, H = screen.screen_size_px()
    stage = WarmupStage(W, H)
    stage.run()
    if stage.aborted or len(stage.rec) < 200:
        print("calentamiento saltado")
        return model
    c = stage.rec.cols
    ok = np.array(c["valid"])
    X, Y = np.array(c["features"])[ok], np.array(c["targets"], np.float32)[ok]
    P = np.array(c["patches"])[ok]
    print(f"calentamiento: {ok.sum()} muestras válidas · ajustando la red a tu postura…")
    pr = load_sessions(with_patches=True)
    rows = np.where(pr["has_P"])[0]
    rng = np.random.default_rng(0)
    app = model.appearance
    t0 = time.monotonic()
    for k in range(rounds):
        rep = rng.choice(rows, min(len(X), len(rows)), replace=False)
        app = finetune_copy(app, np.concatenate([P, pr["P"][rep]]), np.vstack([X, pr["X"][rep]]),
                            np.vstack([Y, pr["Y"][rep]]), steps=40, seed=k)
    tuned = copy.copy(model)
    tuned.appearance = app
    print(f"listo en {time.monotonic() - t0:.0f}s")
    stage.rec.save(W, H, "warmup", "calentamiento de `efg dot`", stage.worker.stats())
    tuned.save(_warm_path())
    return tuned
