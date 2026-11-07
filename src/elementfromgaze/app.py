"""Full-screen routines: calibrate, evaluate, and the red-dot overlay.

Recording principle: store everything raw (frames, all landmarks, blendshapes, head pose,
target position at capture time, segment/local time) and only *flag* which samples look
valid. Every filtering choice (settle window, pursuit lag, blink threshold) can be redone
offline without asking the user to sit through another session.
"""

from __future__ import annotations

import base64
import json
import math
import queue
import random
import time
import tkinter as tk
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from . import screen
from .features import BLENDSHAPE_NAMES, Sample, head_euler
from .filters import BlinkDetector
from .tracker import DATA, SESSIONS, USER_MODEL, FeatureWorker, GazeModel, load_sessions

from . import look

PROTOCOL = "v2"
INTRO_BG = look.GROUND
# Real use mixes dark and bright UIs, and screen light changes how the face is lit:
# cycle backgrounds so the model never learns one lighting.
BGS = ["#1b1b1b", "#f0f0f0", "#7a7a7a"]
PURSUIT_LAG_S = 0.10
# Blinks: MediaPipe's eyeBlink score rises when you look down or lean back (0.45-0.8 with
# eyes open), so an absolute threshold discards whole sessions. BlinkDetector looks for a
# sudden drop of lid opening against your own last ~1.5 s instead.


def lid_opening(s: Sample) -> float:
    return float((s.features[2] + s.features[5]) / 2)


def _is_light(bg: str) -> bool:
    return int(bg[1:3], 16) > 0x99


@dataclass
class Segment:
    phase: str                                       # fix | vor | pursuit | eval | msg
    duration: float
    target: Callable[[float], tuple[float, float]] | None
    record_from: float = 0.0
    lag: float = 0.0
    bg: str = INTRO_BG
    message: str = ""


def fixation(phase, x, y, bg, dur=1.6, settle=0.7) -> Segment:
    return Segment(phase, dur, lambda t, x=x, y=y: (x, y), settle, bg=bg)


def pursuit(W, H, dur, t_offset, bg) -> Segment:
    cx, cy, ax, ay = W / 2, H / 2, W * 0.44, H * 0.42

    def path(t: float) -> tuple[float, float]:
        t += t_offset
        return (cx + ax * math.sin(2 * math.pi * 0.071 * t),
                cy + ay * math.sin(2 * math.pi * 0.113 * t + 0.6))

    return Segment("pursuit", dur, path, 1.0 if t_offset == 0 else 0.0, lag=PURSUIT_LAG_S, bg=bg)


def message(text: str, dur: float) -> Segment:
    return Segment("msg", dur, None, message=text)


@dataclass
class Recorder:
    cols: dict = field(default_factory=lambda: {k: [] for k in (
        "features", "crops", "patches", "landmarks", "blendshapes", "matrix", "head_euler", "brightness",
        "t", "seg", "local_t", "phase", "target_now", "targets", "valid", "bg", "blink", "eyes_ok")})
    jpegs: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)      # optional per-sample columns (all-or-none per key)
    save_frames: bool = True
    ir: list = field(default_factory=list)          # (monotonic t, JPEG) lit IR frames, same clock as `t`

    def add(self, s: Sample, *, seg_i, local_t, seg: Segment, target_now, target, valid, extra=None) -> None:
        for k, v in (extra or {}).items():
            self.extra.setdefault(k, []).append(v)
        c = self.cols
        c["features"].append(s.features)
        c["crops"].append(s.crops)
        c["patches"].append(s.patches)
        c["landmarks"].append(s.landmarks)
        c["blendshapes"].append(s.blendshapes)
        c["matrix"].append(s.matrix)
        c["head_euler"].append(head_euler(s.matrix))
        c["brightness"].append(float(s.frame[::8, ::8].mean()) if s.frame is not None else float("nan"))
        if s.frame is not None:
            self._frame_size = s.frame.shape[1::-1]
        c["t"].append(s.t)
        c["seg"].append(seg_i)
        c["local_t"].append(local_t)
        c["phase"].append(seg.phase)
        c["target_now"].append(target_now)
        c["targets"].append(target)
        c["valid"].append(valid)
        c["bg"].append(seg.bg)
        c["blink"].append(s.blink)
        c["eyes_ok"].append(s.eyes_ok)
        if self.save_frames and s.frame is None:
            self.jpegs.append(np.zeros(0, np.uint8))        # frame already dropped (old label rows)
        elif self.save_frames:
            ok, buf = cv2.imencode(".jpg", s.frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.jpegs.append(buf.ravel() if ok else np.zeros(0, np.uint8))

    def __len__(self) -> int:
        return len(self.cols["t"])

    def save(self, W: int, H: int, tag: str, note: str, perf: str) -> str:
        sid = f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
        d = SESSIONS / sid
        d.mkdir(parents=True, exist_ok=True)
        arr = {k: np.array(v) for k, v in self.cols.items()}
        for k in ("features", "landmarks", "blendshapes", "matrix", "target_now", "targets", "head_euler"):
            arr[k] = arr[k].astype(np.float32)
        arr["crops"] = arr["crops"].astype(np.uint8)
        arr["patches"] = arr["patches"].astype(np.uint8)
        for k, v in self.extra.items():
            if len(v) == len(self):
                arr[f"x_{k}"] = np.array(v)
        np.savez_compressed(d / "samples.npz", **arr, screen=np.array([W, H]),
                            blendshape_names=np.array(BLENDSHAPE_NAMES))
        if self.jpegs:
            offs = np.cumsum([0] + [len(j) for j in self.jpegs])
            np.savez(d / "frames.npz", data=np.concatenate(self.jpegs), offsets=offs)
        if self.ir:
            offs = np.cumsum([0] + [len(r[1]) for r in self.ir])
            np.savez(d / "irframes.npz", t=np.array([r[0] for r in self.ir]), lit=np.array([r[2] for r in self.ir]),
                     data=np.concatenate([r[1] for r in self.ir]), offsets=offs)

        he = arr["head_euler"]
        eye_w = arr["features"][:, 6]
        meta = {
            "id": sid, "tag": tag, "note": note, "protocol": PROTOCOL,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "screen_px": [W, H], "screen_mm": list(screen.screen_size_mm()),
            "px_per_degree": screen.px_per_degree(), "perf": perf,
            "n_samples": len(self), "n_valid": int(arr["valid"].sum()),
            "auto": {
                "brightness_mean": float(np.nanmean(arr["brightness"])),
                "eye_width_median": float(np.median(eye_w)),          # distance proxy: bigger = closer
                "pitch_p5_p95": np.percentile(he[:, 0], [5, 95]).round(1).tolist(),
                "yaw_p5_p95": np.percentile(he[:, 1], [5, 95]).round(1).tolist(),
                "roll_p5_p95": np.percentile(he[:, 2], [5, 95]).round(1).tolist(),
                "mp_blink_median": float(np.median(arr["blink"])),
                "eyes_ok_rate": float(arr["eyes_ok"].mean()),
                "invalid_rate": float(1 - arr["valid"].mean()),
                "frame_size": list(self._frame_size) if getattr(self, "_frame_size", None) else None,
                "ir_frames": len(self.ir), "ir_lit": int(sum(r[2] for r in self.ir)),
            },
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(d)


class Stage:
    """A full-screen Tk window that plays Segments and records every sample against them."""

    title = "Calibración"
    intro_text = ("Mira siempre el punto naranja; leerás instrucciones entre fases.\n"
                  "ESPACIO para empezar · ESC para salir sin guardar")

    def __init__(self, segments: list[Segment]):
        screen.make_dpi_aware()
        self.W, self.H = screen.screen_size_px()
        self.segments = segments
        self.samples: "queue.Queue[Sample]" = queue.Queue()
        self.rec = Recorder()
        self.root = tk.Tk()
        self.root.configure(bg=INTRO_BG)
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.canvas = tk.Canvas(self.root, bg=INTRO_BG, highlightthickness=0, cursor="none")
        self.canvas.pack(fill="both", expand=True)
        self.root.bind("<Escape>", lambda e: self._abort())
        self.root.bind("<space>", lambda e: self._start())
        self.started_at: float | None = None
        self.aborted = False
        self.blinker = BlinkDetector()
        self._last_quality = (0.0, True)
        self._last_frame: np.ndarray | None = None
        self._preview_img = None          # Tk keeps only a weak ref: hold the PhotoImage here
        self._preview_at = 0.0
        self._preview_top = False
        self._cur_seg = -1
        acc, self._bounds = 0.0, []
        for s in segments:
            self._bounds.append(acc)
            acc += s.duration
        self._bounds.append(acc)
        self.worker = FeatureWorker(self.samples.put)
        from .ircam import open_ir
        self.ir = open_ir(record=True)     # recorded alongside, same clock, for the IR experiments
        self._intro()

    def _segment_at(self, t: float) -> tuple[int, float] | None:
        b = self._bounds
        for i in range(len(self.segments)):
            if b[i] <= t < b[i + 1]:
                return i, t - b[i]
        return None

    def _intro(self) -> None:
        self._status_id = look.intro(self.canvas, self.W, self.H, self.title, self.intro_text, self._bounds[-1])
        self.root.after(200, self._intro_tick)

    def _intro_tick(self) -> None:
        if self.started_at is not None or self.aborted:
            return
        ir = f"IR {self.ir.fps():.1f} fps" if self.ir else ""
        self.canvas.itemconfigure(self._status_id, text=look.status_text(self.worker.face_seen, self.worker.stats(), ir),
                                  fill=look.FAINT if self.worker.face_seen else look.BAD)
        while not self.samples.empty():
            s = self.samples.get_nowait()
            self._last_frame, self._last_quality = s.frame, (time.monotonic(), s.eyes_ok)
        self._draw_preview(None)
        self.root.after(100, self._intro_tick)

    PREVIEW_W, PREVIEW_H = 200, 150

    def _draw_preview(self, target: tuple[float, float] | None) -> None:
        """Small mirrored camera view so you can check, from the corner of your eye, that you
        are still in frame. Green border = both eyes seen; red = not. It sits at the bottom
        centre and moves to the top while the dot is in the lower half, so it never covers it."""
        now = time.monotonic()
        if now - self._preview_at < 0.1 or self._last_frame is None:
            return
        self._preview_at = now
        if target is not None:   # hysteresis: switch side only when the dot is clearly past the middle
            if target[1] > self.H * 0.6:
                self._preview_top = True
            elif target[1] < self.H * 0.4:
                self._preview_top = False
        pw, ph = self.PREVIEW_W, self.PREVIEW_H
        small = cv2.resize(self._last_frame, (pw, ph), interpolation=cv2.INTER_AREA)[:, ::-1]
        small = (small * 0.8).astype(np.uint8)       # dimmed: visible, but not a magnet for the eye
        ok, png = cv2.imencode(".png", small)
        if not ok:
            return
        self._preview_img = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        x0 = self.W / 2 - pw / 2
        y0 = 12 if self._preview_top else self.H - ph - 12
        t_q, eyes_ok = self._last_quality
        good = self.worker.face_seen and (eyes_ok or now - t_q > 0.5)
        c = self.canvas
        c.delete("preview")
        c.create_image(x0, y0, image=self._preview_img, anchor="nw", tags="preview")
        c.create_rectangle(x0 - 2, y0 - 2, x0 + pw + 2, y0 + ph + 2, outline=look.GOOD if good else look.BAD,
                           width=2, tags="preview")

    def _draw(self, seg: Segment, lt: float) -> None:
        c = self.canvas
        c.delete("target")
        if seg.phase == "msg":
            look.message(c, self.W, self.H, seg.message)
            return
        x, y = seg.target(lt)
        light = _is_light(seg.bg)
        if seg.phase == "vor":
            look.target(c, x, y, head=True, light_bg=light)
        elif seg.phase == "pursuit":
            look.target(c, x, y, light_bg=light)
        else:
            look.target(c, x, y, settle=lt / max(seg.record_from, 1e-3), light_bg=light)

    def _start(self) -> None:
        if self.started_at is None:
            self.canvas.delete("all")
            while not self.samples.empty():
                self.samples.get_nowait()
            self.started_at = time.monotonic()
            self._tick()

    def _abort(self) -> None:
        self.aborted = True
        self.root.destroy()

    def _tick(self) -> None:
        now = time.monotonic() - self.started_at
        while not self.samples.empty():
            s = self.samples.get_nowait()
            hit = self._segment_at(s.t - self.started_at)
            if hit is None:
                continue
            i, lt = hit
            seg = self.segments[i]
            if seg.target is None:
                continue
            is_blink = self.blinker(lid_opening(s), s.eye_blink)
            valid = lt >= seg.record_from and not is_blink and s.eyes_ok
            self._last_quality = (time.monotonic(), s.eyes_ok)
            self._last_frame = s.frame
            target = seg.target(max(lt - seg.lag, 0.0))
            self.rec.add(s, seg_i=i, local_t=lt, seg=seg, target_now=seg.target(lt), target=target, valid=valid,
                         extra=self.extra_for(s, seg, lt))
            self.on_sample(s, target, seg, valid)
        cur = self._segment_at(now)
        if cur is None:
            self.finish()
            return
        i, lt = cur
        seg = self.segments[i]
        if i != self._cur_seg:
            self._cur_seg = i
            bg = seg.bg if seg.phase != "msg" else INTRO_BG
            self.canvas.configure(bg=bg)
        self._draw(seg, lt)
        look.progress(self.canvas, self.W, now / self._bounds[-1], light_bg=seg.phase != "msg" and _is_light(seg.bg))
        self._draw_quality()
        if seg.phase == "msg":
            self.canvas.delete("preview")
        else:
            self._draw_preview(seg.target(lt) if seg.target is not None else None)
        self.root.after(8, self._tick)

    def _draw_quality(self) -> None:
        """Red frame while the camera can't see both eyes properly, so you notice at once."""
        t_q, ok = self._last_quality
        bad = (not self.worker.face_seen) or (not ok and time.monotonic() - t_q < 0.5)
        self.canvas.delete("quality")
        if bad:
            look.warning(self.canvas, self.W, self.H, "no veo bien tus ojos")

    def on_sample(self, s, target, seg, valid) -> None:
        pass

    def extra_for(self, s, seg, lt) -> dict | None:
        return None

    def finish(self) -> None:
        self.root.destroy()

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self.worker.close()
            if self.ir is not None:
                t0 = self.started_at or 0.0
                self.rec.ir = [r for r in self.ir.take() if r[0] >= t0]
                self.ir.close()


# ---------------------------------------------------------------------------
def calibration_protocol(W: int, H: int) -> list[Segment]:
    grid = [(fx, fy) for fy in (0.07, 0.5, 0.93) for fx in (0.05, 0.5, 0.95)]
    grid += [(0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72)]
    random.shuffle(grid)
    segs = [fixation("fix", fx * W, fy * H, BGS[k % 3]) for k, (fx, fy) in enumerate(grid)]

    segs.append(message(
        "Ahora el punto se quedará QUIETO (con un aro azul).\n\n"
        "No dejes de mirarlo y mueve la cabeza despacio:\n"
        "arriba y abajo, a los lados, inclínala, acércate y aléjate.", 6.0))
    vor = [(0.5, 0.5), (0.18, 0.22), (0.82, 0.22), (0.18, 0.78), (0.82, 0.78)]
    random.shuffle(vor)
    segs += [Segment("vor", 6.0, (lambda t, x=fx * W, y=fy * H: (x, y)), 0.8, bg=BGS[k % 3])
             for k, (fx, fy) in enumerate(vor)]

    segs.append(message("Ahora sigue con la mirada el punto que se mueve.\nCabeza natural, sin forzar.", 3.5))
    segs += [pursuit(W, H, 11.0, 0.0, BGS[0]), pursuit(W, H, 11.0, 11.0, BGS[1])]
    return segs


def calibrate(tag: str = "normal", note: str = "") -> None:
    screen.make_dpi_aware()
    W, H = screen.screen_size_px()
    stage = Stage(calibration_protocol(W, H))
    stage.run()
    if stage.aborted or not len(stage.rec):
        print("calibración cancelada, no se guarda nada")
        return
    path = stage.rec.save(W, H, tag, note, stage.worker.stats())
    print(f"guardadas {len(stage.rec)} muestras en {path}  ({stage.worker.stats()})")
    train()


MOTION_TEXT = {
    "portatil": "Mira el punto SIN dejar de mirarlo y mueve el PORTÁTIL despacio:\n"
                "gíralo a los lados, inclina la pantalla, acércalo y aléjalo, súbelo y bájalo.",
    "cabeza": "Mira el punto SIN dejar de mirarlo y mueve la CABEZA despacio:\n"
              "a los lados, arriba y abajo, inclínala, acércate y aléjate.",
    "ladeo": "Mira el punto SIN dejar de mirarlo y LADEA la cabeza, oreja hacia el hombro:\n"
             "despacio a un lado, quédate un momento, luego al otro. Ladeos grandes y pequeños.",
}


def motion(mode: str = "portatil", tag: str | None = None, note: str = "", n: int = 7, dur: float = 9.0) -> None:
    """Fixate static points while the laptop (or head) moves: pure pose-invariance data.
    The target is fixed on the screen, so any movement of the prediction is pose sensitivity."""
    screen.make_dpi_aware()
    W, H = screen.screen_size_px()
    pts = [(0.5, 0.5), (0.12, 0.15), (0.88, 0.15), (0.12, 0.85), (0.88, 0.85), (0.5, 0.15), (0.5, 0.85),
           (0.12, 0.5), (0.88, 0.5)][:n if mode != "ladeo" else 9]
    random.shuffle(pts)
    segs = [message(MOTION_TEXT[mode] + "\n\nCada punto dura unos segundos; luego salta a otro.", 6.0)]
    segs += [Segment("vor", dur, (lambda t, x=fx * W, y=fy * H: (x, y)), 0.8, bg=BGS[k % 3])
             for k, (fx, fy) in enumerate(pts)]
    stage = type("MotionStage", (Stage,), {"title": f"Movimiento: {mode}"})(segs)
    stage.run()
    if stage.aborted or not len(stage.rec):
        print("cancelado, no se guarda nada")
        return
    path = stage.rec.save(W, H, tag or f"mov-{mode}", note, stage.worker.stats())
    print(f"guardadas {len(stage.rec)} muestras en {path}")
    train()


def _perimeter_path(W: int, H: int, inset: float, speed: float):
    """Point moving along the screen border (clockwise from top-left) at `speed` px/s."""
    x0, y0, x1, y1 = inset * W, inset * H, (1 - inset) * W, (1 - inset) * H
    sides = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    lens = [math.dist(a, b) for a, b in sides]
    total = sum(lens)

    def path(t: float) -> tuple[float, float]:
        s = (t * speed) % total
        for (a, b), L in zip(sides, lens):
            if s <= L:
                f = s / L
                return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            s -= L
        return sides[0][0]

    return path, total / speed


def edges(note: str = "") -> None:
    """Follow a dot around the screen border (extreme-range labels, where the model compresses most),
    then look away from the screen on cue (negatives for an 'not looking at the screen' detector)."""
    screen.make_dpi_aware()
    W, H = screen.screen_size_px()
    segs = [message("Sigue con la mirada el punto que recorre el BORDE de la pantalla.\n"
                    "Cabeza natural. Dará dos vueltas.", 4.0)]
    for k, inset in enumerate((0.02, 0.06)):
        path, lap = _perimeter_path(W, H, inset, speed=320.0)
        segs.append(Segment("pursuit", lap, path, 1.0, lag=PURSUIT_LAG_S, bg=BGS[k % 2]))
    segs.append(message("Ahora, cuando la pantalla se ponga ROJA, mira FUERA de la pantalla\n"
                        "(al teclado, a un lado, arriba, a la pared...). Cuando vuelva el punto, míralo.", 6.0))
    off = lambda t: (-1.0, -1.0)   # sentinel target: not on the screen
    spots = [(0.5, 0.5), (0.2, 0.8), (0.8, 0.2), (0.5, 0.9)]
    for k, (fx, fy) in enumerate(spots):
        segs.append(Segment("off", 4.0, off, 0.8, bg="#8a1c1c"))
        segs.append(fixation("fix", fx * W, fy * H, BGS[k % 3], dur=2.0, settle=0.8))
    stage = type("EdgeStage", (Stage,), {"title": "Bordes y mirar fuera"})(segs)
    stage.run()
    if stage.aborted or not len(stage.rec):
        print("cancelado, no se guarda nada")
        return
    print("guardada:", stage.rec.save(W, H, "edges", note, stage.worker.stats()))
    train()


def train(appearance: bool = True, force: bool = False) -> GazeModel | None:
    import os
    if os.environ.get("EFG_NO_TRAIN") and not force:
        print("sesión guardada · sin reentrenar (--no-train): corre `efg train` al terminar")
        return None
    d = load_sessions(with_patches=appearance)
    screen.make_dpi_aware()
    W, H = screen.screen_size_px()
    model = GazeModel().fit(d["X"], d["Y"], (W, H))
    ppd = screen.px_per_degree()
    err = np.linalg.norm(model.predict(d["X"]) - d["Y"], axis=1)
    print(f"modelo base entrenado con {len(d['X'])} muestras de {len(d['names'])} sesiones · "
          f"error de ajuste (optimista) {np.median(err) / ppd:.1f}°")
    # A user who started from the strangers' base keeps it: a network trained on their own few
    # sessions alone would be far worse (the base + warm-up: ~2°). Retraining that mixes strangers
    # in is the next step; until then their eye network only moves through warm-up fine-tunes.
    from .generic import uses_strangers
    old = GazeModel.load() if USER_MODEL.exists() else None
    if appearance and old is not None and uses_strangers(getattr(old, "appearance", None)):
        model.appearance = old.appearance
        for k in ("generic", "start_aff"):
            if hasattr(old, k):
                setattr(model, k, getattr(old, k))
        print("red de ojos: se mantiene la de base (entrenada con otras personas) + tus calentamientos")
    elif appearance:
        has = d["has_P"]
        if has.sum() >= 1500:
            print(f"entrenando la red de ojos con {int(has.sum())} muestras (unos minutos)…")
            model.fit_appearance(d["P"][has], d["X"][has], d["Y"][has])
            e2 = np.linalg.norm(model.predict_full(d["X"][has], d["P"][has]) - d["Y"][has], axis=1)
            print(f"red de ojos lista · error de ajuste (optimista) {np.median(e2) / ppd:.1f}°")
        else:
            print("pocas muestras con imagen de ojos todavía: se usa solo el modelo base")
    model.save()
    return model


class EvalStage(Stage):
    title = "Evaluación"

    def __init__(self, segments, model: GazeModel):
        self.model = model
        self.per_point: dict[int, list] = {}
        super().__init__(segments)

    def on_sample(self, s, target, seg, valid):
        if valid:
            p = self.model.predict_full(s.features, s.patches)[0]
            self.per_point.setdefault(id(seg), []).append((p, np.array(target)))

    def finish(self):
        c, W, H = self.canvas, self.W, self.H
        c.delete("all")
        c.configure(bg=INTRO_BG)
        ppd = screen.px_per_degree()
        errs = []
        for pts in self.per_point.values():
            preds = np.array([p for p, _ in pts])
            tgt = pts[0][1]
            med = np.median(preds, axis=0)
            errs.append(float(np.linalg.norm(med - tgt)))
            c.create_line(*tgt, *med, fill="#777777")
            c.create_oval(tgt[0] - 6, tgt[1] - 6, tgt[0] + 6, tgt[1] + 6, outline="#ffffff", width=2)
            c.create_oval(med[0] - 5, med[1] - 5, med[0] + 5, med[1] + 5, fill="#ff3b30", outline="")
            for p in preds[::3]:
                c.create_oval(p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1, fill="#ff8a80", outline="")
        if errs:
            e = np.array(errs)
            self.report = {
                "points": len(e), "median_px": round(float(np.median(e))), "mean_px": round(float(e.mean())),
                "median_deg": round(float(np.median(e) / ppd), 2), "mean_deg": round(float(e.mean() / ppd), 2),
                "px_per_degree": round(ppd, 1), "screen": [W, H], "perf": self.worker.stats(),
                "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            c.create_text(
                W / 2, 40, fill="#eeeeee", font=("Segoe UI", 16),
                text=f"error mediano {self.report['median_px']}px ({self.report['median_deg']}°) · "
                     f"medio {self.report['mean_px']}px ({self.report['mean_deg']}°) · ESC para cerrar",
            )
        self.root.unbind("<space>")


def evaluate(n: int = 12, tag: str = "normal", note: str = "") -> None:
    model = GazeModel.load()
    screen.make_dpi_aware()
    W, H = screen.screen_size_px()
    rng = random.Random()
    pts = [(rng.uniform(0.06, 0.94) * W, rng.uniform(0.08, 0.92) * H) for _ in range(n)]
    segs = [fixation("eval", x, y, BGS[k % 3], dur=1.9, settle=0.8) for k, (x, y) in enumerate(pts)]
    stage = EvalStage(segs, model)
    stage.run()
    if getattr(stage, "report", None):
        n_before = len(load_sessions()["names"])
        stage.report |= {"tag": tag, "note": note, "trained_on_sessions": n_before}
        print(json.dumps(stage.report, indent=2, ensure_ascii=False))
        DATA.mkdir(parents=True, exist_ok=True)
        with open(DATA / "eval_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(stage.report, ensure_ascii=False) + "\n")
        print("guardada:", stage.rec.save(W, H, f"eval-{tag}", note, stage.worker.stats()))


def dot(radius: int = 12, learn: bool = True, warm: bool = True, rewarm: bool = False, serve: bool = False,
        show: bool = True, control: bool = False, click: bool = False) -> None:
    from .overlay import dot as _dot

    _dot(radius, learn, warm, rewarm, serve, show, control, click)
