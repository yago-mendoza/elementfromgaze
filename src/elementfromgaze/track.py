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
import random
import time

import numpy as np

from . import look, screen
from .app import BGS, INTRO_BG, PURSUIT_LAG_S, Segment, Stage, message
from .adapt import AffineKalman
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
        self.aff = AffineKalman.seeded(W, H, model)          # same start as `efg dot` (no click learning here)
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
        base = self.aff.apply(self.model.predict_full(s.features, s.patches)[0])
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
            look.message(c, self.W, self.H, seg.message)
            return
        x, y = seg.target(lt)
        look.ring(c, x, y, light_bg=_ink(seg.bg) != "#ffffff")
        if self._shown.get(id(seg)) and self._dot is not None:
            px, py = self._dot
            c.create_oval(px - 9, py - 9, px + 9, py + 9, fill=look.BAD, outline="", tags="target")

    def finish(self) -> None:
        self.report = summarize(self.rec, self.W, self.H)
        c, W, H = self.canvas, self.W, self.H
        c.delete("all")
        c.configure(bg=INTRO_BG)
        r = self.report
        if r:
            lines = [f"sesgo medio ({r['all']['bias_px'][0]:+d}, {r['all']['bias_px'][1]:+d}) px"]
            for band in ("arriba", "medio", "abajo"):
                b = r["bands"].get(band)
                if b:
                    lines.append(f"{band:7s} {b['median_deg']}°   sesgo ({b['bias_px'][0]:+d}, {b['bias_px'][1]:+d}) px")
            if r.get("shown") and r.get("hidden"):
                lines.append(f"punto visible {r['shown']['median_deg']}°   oculto {r['hidden']['median_deg']}°")
            look.result(c, W, H, "error mediano", f"{r['all']['median_deg']}°", lines)
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
    from .tracker import load_user_model
    model = load_user_model(warm)
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
HEAD_S, HEAD_AMP = 8.5, 12.0     # head-turn segment: seconds, guide amplitude (deg)


def head_target(t: float) -> tuple[float, float]:
    """Guide for the head turn: (yaw, pitch) in degrees. A smooth horizontal swing, then vertical,
    ~3.6 s per axis, starting and ending at rest."""
    import math
    t = max(0.0, t - 0.6)
    if t < 3.6:
        return HEAD_AMP * math.sin(2 * math.pi * t / 3.6), 0.0
    t -= 3.6
    if t < 3.6:
        return 0.0, 0.75 * HEAD_AMP * math.sin(2 * math.pi * t / 3.6)
    return 0.0, 0.0


CHECK_POINTS = [(fx, fy) for fy in (0.08, 0.5, 0.92) for fx in (0.08, 0.5, 0.92)]
CHECK_S, CHECK_SETTLE_S = 1.6, 0.6


class WarmupStage(Stage):
    """~45 s following a ring (no dot shown): fresh labels in the posture you are in right now.
    Then ~15 s looking at 9 lit buttons: NOT used to fine-tune, but to measure what the fine-tuned
    network still gets wrong on real fixations (on `targets` it left a 1.5° vertical bias), which
    seeds the global correction of `efg dot` so it is right from the first second, before any click."""

    title = "Calentamiento (≈78 s)"
    intro_text = ("Sigue con la mirada el ARO; al final, mira cada BOTÓN que se ilumine.\n"
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
        # head turns while fixating: the dot drifted ~2-3 px per degree of head yaw (measured on the
        # head-motion sessions); 6 s of this in the CURRENT posture teaches the network to hold still
        segs.append(message("Mira el aro todo el rato y mueve la cabeza: lleva tu marca naranja al punto azul.", 2.6))
        segs.append(Segment("vor", HEAD_S, (lambda t, x=W / 2, y=H * 0.45: (x, y)), 0.6, bg=BGS[0]))
        pts = CHECK_POINTS[:]
        random.shuffle(pts)
        segs.append(message("Ahora mira cada botón que se ilumine.", 1.8))
        for fx, fy in pts:
            segs.append(Segment("check", CHECK_S, (lambda t, x=fx * W, y=fy * H: (x, y)), CHECK_SETTLE_S, bg=INTRO_BG))
        self.btn_h = 1.5 * screen.px_per_degree()
        self.head = {"base": [], "now": None, "pairs": []}      # live head pose for the head-turn guide
        super().__init__(segs)

    def on_sample(self, s, target, seg, valid) -> None:
        if seg.phase != "vor":
            self.head["base"].clear()
            return
        from .features import head_euler
        he = np.asarray(head_euler(s.matrix), float)[:2]          # pitch, yaw (deg)
        if len(self.head["base"]) < 6:
            self.head["base"].append(he)
            return
        rel = he - np.median(self.head["base"], 0)
        self.head["now"] = rel
        lt = s.t - self.started_at - self._bounds[self.segments.index(seg)]
        self.head["pairs"].append((head_target(lt), rel))

    def _head_signs(self) -> np.ndarray:
        """Which way is 'left' for MediaPipe's angles vs the screen: learned from the first second of
        following the guide (flip an axis whose actual motion anti-correlates with the guide)."""
        P = self.head["pairs"]
        sg = np.ones(2)
        if len(P) > 15:
            d = np.array([p[0] for p in P])           # (yaw, pitch) guide
            a = np.array([p[1][::-1] for p in P])     # (yaw, pitch) actual
            for i in range(2):
                if d[:, i].std() > 1 and a[:, i].std() > 0.5 and np.corrcoef(d[:, i], a[:, i])[0, 1] < -0.3:
                    sg[i] = -1
        return sg

    def _draw(self, seg: Segment, lt: float) -> None:
        c = self.canvas
        c.delete("target")
        if seg.phase == "msg":
            look.message(c, self.W, self.H, seg.message)
            return
        x, y = seg.target(lt)
        if seg.phase == "check":     # the button is centred on the target (points sit far enough from the edges)
            h = self.btn_h
            look.button(c, x, y, 2 * h, h, "Aceptar")
            return
        if seg.phase == "vor":         # head-turn guide: where to point the nose, and where it points
            R = 150
            c.create_oval(x - R, y - R, x + R, y + R, outline=look.RULE, width=1, tags="target")
            c.create_oval(x - R / 2, y - R / 2, x + R / 2, y + R / 2, outline=look.RULE, width=1, dash=(2, 6), tags="target")
            c.create_line(x - R, y, x + R, y, fill=look.RULE, tags="target")
            c.create_line(x, y - R, x, y + R, fill=look.RULE, tags="target")
            gy, gp = head_target(lt)
            gx_, gy_ = x + gy / HEAD_AMP * R, y - gp / HEAD_AMP * R
            c.create_oval(gx_ - 10, gy_ - 10, gx_ + 10, gy_ + 10, fill=look.STEEL, outline="", tags="target")
            if self.head["now"] is not None:
                sg = self._head_signs()
                ay, ap = self.head["now"][1] * sg[0], self.head["now"][0] * sg[1]
                ax_, ay_ = x + np.clip(ay / HEAD_AMP, -1.3, 1.3) * R, y - np.clip(ap / HEAD_AMP, -1.3, 1.3) * R
                c.create_oval(ax_ - 15, ay_ - 15, ax_ + 15, ay_ + 15, outline=look.ACCENT, width=3, tags="target")
            c.create_text(x, y + R + 34, text="mira el aro blanco  ·  mueve la cabeza para llevar el círculo naranja al punto azul",
                          fill=look.DIM, font=(look.SANS, 13), tags="target")
        look.ring(c, x, y, light_bg=_ink(seg.bg) != "#ffffff")


def check_correction(model: GazeModel, rec, W, H) -> dict | None:
    """Global correction (shift + bounded per-axis gain) from the 9 button fixations, measured
    with the already fine-tuned network: one 'click' per button, as `efg dot` would learn them."""
    c = rec.cols
    ph, lt, seg = np.array(c["phase"]), np.array(c["local_t"]), np.array(c["seg"])
    k = (ph == "check") & (lt >= CHECK_SETTLE_S) & np.array(c["eyes_ok"])
    if k.sum() < 40:
        return None
    pred = model.predict_full(np.array(c["features"])[k], np.array(c["patches"])[k])
    Y, T, S = np.array(c["targets"], float)[k], np.array(c["t"])[k], seg[k]
    aff, errs = AffineKalman.for_check(W, H), []
    for sg in np.unique(S):
        q = S == sg
        if q.sum() < 5:
            continue
        m = np.median(pred[q], 0)
        errs.append(m - Y[q][0])
        aff.add(m, Y[q][0], float(T[q][-1]))
    if len(errs) < 5:
        return None
    e = np.array(errs)
    ppd = screen.px_per_degree()
    print(f"comprobación con botones: error {np.median(np.linalg.norm(e, axis=1)) / ppd:.2f}° · sesgo "
          f"({np.median(e[:, 0]):+.0f}, {np.median(e[:, 1]):+.0f}) px -> corrección {aff.describe()} "
          f"({aff.accepted}/{len(e)} aceptados)")
    return aff.state()


def pose_summary(head_euler, features) -> dict:
    """What the warm-up was fitted to: head pitch/yaw (deg), eye width (distance proxy), lid opening."""
    return {"pitch": float(np.median(head_euler[:, 0])), "yaw": float(np.median(head_euler[:, 1])),
            "eye_w": float(np.median(features[:, 6])), "lid": float(np.median((features[:, 2] + features[:, 5]) / 2))}


def pose_drift(now: dict, ref: dict) -> list[str]:
    """Reasons the current posture no longer matches the warm-up. Thresholds from 2026-11-06: a
    reused warm-up with pitch -13 -> -21°, eye width -10%, lids -12% gave 2.35° instead of 0.9°."""
    out = []
    if abs(now["pitch"] - ref["pitch"]) > 5:
        out.append(f"cabeza {'más baja' if now['pitch'] < ref['pitch'] else 'más alta'} ({now['pitch'] - ref['pitch']:+.0f}°)")
    if abs(now["yaw"] - ref["yaw"]) > 6:
        out.append(f"cabeza girada ({now['yaw'] - ref['yaw']:+.0f}°)")
    r = now["eye_w"] / ref["eye_w"]
    if abs(r - 1) > 0.07:
        out.append("más lejos" if r < 1 else "más cerca")
    if now["lid"] / ref["lid"] < 0.85:
        out.append("ojos más cerrados")
    return out


WARM_MAX_AGE_S = 30 * 60


def _bank_dir():
    """Posture bank: every warm-up is kept with the posture it was fitted to; `dot` switches to
    the one that matches how you sit now, so a posture you have warmed up once never needs it again
    (until `efg train` makes a new base model, which makes the bank stale)."""
    from .tracker import USER_MODEL
    return USER_MODEL.with_name(USER_MODEL.stem + "_bank")


def pose_distance(a: dict, b: dict) -> float:
    """Same units as pose_drift's thresholds: 1.0 ~ 'just different enough to matter'."""
    return float(np.sqrt(((a["pitch"] - b["pitch"]) / 5) ** 2 + ((a["yaw"] - b["yaw"]) / 6) ** 2
                         + (np.log(a["eye_w"] / b["eye_w"]) / 0.07) ** 2 + ((a["lid"] / b["lid"] - 1) / 0.15) ** 2))


def bank_entries() -> list[tuple[dict, "Path"]]:
    """(warm_pose, path) of every banked warm-up newer than the base model."""
    import pickle

    from .tracker import USER_MODEL
    out = []
    d = _bank_dir()
    if not d.exists() or not USER_MODEL.exists():
        return out
    for p in sorted(d.glob("warm_*.pkl")):
        if p.stat().st_mtime <= USER_MODEL.stat().st_mtime:
            continue
        try:
            pose = getattr(pickle.loads(p.read_bytes()), "warm_pose", None)
        except Exception:
            continue
        if pose:
            out.append((pose, p))
    return out


def _warm_path():
    from .tracker import USER_MODEL
    return USER_MODEL.with_name(USER_MODEL.stem + "_warm.pkl")


def warm_model(model: GazeModel) -> GazeModel:
    """The warmed-up model from the last 30 min if there is one, else run a warm-up now."""
    from .tracker import USER_MODEL

    p = _warm_path()
    bank = bank_entries()
    if bank and not (p.exists() and time.time() - p.stat().st_mtime < WARM_MAX_AGE_S):
        # an older warm-up: start from the latest; `dot` switches to the bank entry that matches
        # your posture after ~10 s, or tells you it is a new posture
        print(f"banco de posturas: {len(bank)} calentamientos; elijo el tuyo en unos segundos")
        return GazeModel.load(bank[-1][1])
    if (p.exists() and time.time() - p.stat().st_mtime < WARM_MAX_AGE_S
            and p.stat().st_mtime > USER_MODEL.stat().st_mtime):      # newer than the base model
        m = GazeModel.load(p)
        if getattr(m, "warm_pose", None) is None:      # pre-drift-check warm-up: redo it
            return warmup(model)
        age = (time.time() - p.stat().st_mtime) / 60
        print(f"uso el calentamiento de hace {age:.0f} min (si cambiaste de postura: --rewarm; `dot` avisa si te mueves)")
        return m
    return warmup(model)


def adapt(model: GazeModel, P, X, Y, rounds: int = 8, screen_px=None) -> GazeModel:
    """The warm-up's fine-tune: top layers on the warm-up, replaying the user's own sessions 1:1.
    A NEW user (model.generic, from generic.py) first gets the screen geometry fitted, 16 rounds
    instead of 8 (1.86° -> 1.62°), and replays strangers (every fine-tune of theirs does)."""
    import copy

    from .appearance import finetune_copy
    from .tracker import load_sessions

    app = model.appearance
    new = getattr(model, "generic", False)
    if new:
        from .generic import fit_geometry
        app = fit_geometry(app, X, Y)
        rounds = max(rounds, 16)
    try:
        pr = load_sessions(with_patches=True)
        rows = np.where(pr["has_P"])[0]
    except SystemExit:                     # no sessions of their own yet
        pr, rows = None, np.zeros(0, int)
    rng = np.random.default_rng(0)
    for k in range(rounds):
        rep = rng.choice(rows, min(len(X), len(rows)), replace=False)
        if len(rep):
            app = finetune_copy(app, np.concatenate([P, pr["P"][rep]]), np.vstack([X, pr["X"][rep]]),
                                np.vstack([Y, pr["Y"][rep]]), steps=40, seed=k)
        else:
            app = finetune_copy(app, P, X, Y, steps=40, seed=k)
    tuned = copy.copy(model)
    tuned.appearance = app
    if new:
        tuned.generic = False              # geometry fitted: from now on it is this user's model
        tuned.fit(X, Y, screen_px or getattr(model, "screen", None))   # feature fallback regressor
    return tuned


def warmup(model: GazeModel, rounds: int = 8) -> GazeModel:   # 8 > 4: 1.45° -> 1.25° on 3 tracks
    """Fine-tune the eye network's top layers on a fresh ~45 s pursuit (replaying the calibration
    sessions 1:1). Measured on a far, head-down posture never seen in calibration: 4.3° -> 1.1-1.2°
    on curves and static points not used for the fit."""
    if getattr(model, "appearance", None) is None:
        print("sin red de ojos: no hay calentamiento")
        return model
    screen.make_dpi_aware()
    W, H = screen.screen_size_px()
    stage = WarmupStage(W, H)
    stage.run()
    if stage.aborted or len(stage.rec) < 200:
        if getattr(model, "generic", False):
            raise SystemExit("la primera vez el calentamiento es imprescindible: ajusta la red a tus ojos y tu pantalla")
        print("calentamiento saltado")
        return model
    c = stage.rec.cols
    ok = np.array(c["valid"]) & np.isin(np.array(c["phase"]), ("pursuit", "fix", "vor"))
    X, Y = np.array(c["features"])[ok], np.array(c["targets"], np.float32)[ok]
    P = np.array(c["patches"])[ok]
    print(f"calentamiento: {ok.sum()} muestras válidas · ajustando la red a tu postura…")
    t0 = time.monotonic()
    tuned = adapt(model, P, X, Y, rounds, (W, H))
    tuned.start_aff = check_correction(tuned, stage.rec, W, H)
    tuned.warm_pose = pose_summary(np.array(c["head_euler"])[ok], np.array(c["features"])[ok])
    print(f"listo en {time.monotonic() - t0:.0f}s")
    stage.rec.save(W, H, "warmup", "calentamiento de `efg dot`", stage.worker.stats())
    tuned.save(_warm_path())
    bank = _bank_dir()
    bank.mkdir(parents=True, exist_ok=True)
    tuned.save(bank / f"warm_{time.strftime('%Y%m%d-%H%M%S')}.pkl")
    return tuned
