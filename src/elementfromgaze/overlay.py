"""`efg dot`: the red dot over the whole screen, learning from your clicks and typing as you go.

Per frame:   eye network (or feature model) -> global deformation (Kalman) -> local field -> blink gate
             -> fixation filter
Three speeds of learning: the deformation Kalman moves with every label (seconds); every ~60 new
labelled samples the network's top layers fine-tune in the background on them plus a replay of the
calibration sessions (minutes), then the fast correctors reset since the network now holds it;
`efg train` refits everything (sessions).
Per click:   gaze samples from 350 ms before to 50 ms after = one label at the cursor
Per keypress (with an exposed text caret, max ~3/s): one label at the caret, lower confidence
On exit:     every label is saved as an 'implicit' session, so `efg train` folds it into the base model.
"""

from __future__ import annotations

import ctypes
import threading
import time
import tkinter as tk
from collections import deque

import numpy as np

from . import screen
from .adapt import AffineKalman, ResidualCorrector, pose_vector
from .app import Recorder, Segment, lid_opening
from .features import Sample, head_euler
from .filters import BlinkDetector, FixationFilter, GazeSmoother, Spring2D
from .implicit import InputWatcher, caret_position, cursor_position
from .tracker import USER_MODEL, FeatureWorker, GazeModel

TRANSPARENT = "#010101"
VK = {"ctrl": 0x11, "alt": 0x12, "g": 0x47}
VK_KEYS = {"left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "r": 0x52}
NUDGE_PX = 12            # Ctrl+Alt+arrow: move the dot this much (your "+n")
RECHECK_POINTS = [(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
RECHECK_S, RECHECK_SETTLE_S = 1.4, 0.5
# Frames are time-stamped when PROCESSED, ~0.13 s after exposure (fitted on 4 track sessions:
# best lag 0.10-0.16 s). Event windows below are in EYE time and shifted by this to sample time.
LATENCY_S = 0.13
LABEL_DELAY_S = 0.55     # process an event once the samples of its whole window exist
WINDOWS = {"click": (-0.30, -0.05, 1.0), "caret": (0.0, 0.35, 0.6)}  # eye time: from, to, confidence
# "gaze": labels posted by a web page when its attractor posterior is unambiguous (see web/efg.js);
# window and confidence come with each label
KEY_MIN_GAP_S = 0.35
OFF_MARGIN_PX = 90      # prediction this far outside the screen = not looking at it (edges data:
OFF_FRAMES = 3           # ~95% of look-aways caught, <1% false alarms)
TYPING_RADIUS_PX = 240   # ~4°: below this, movement while typing is treated as shake
TYPING_HOLD_S = 0.8
TUNE_EVERY = 60          # new labelled samples (~6 clicks or ~10 caret labels) per fine-tune
TUNE_MIN_GAP_S = 20.0


def _hotkey_down() -> bool:
    k = ctypes.windll.user32.GetAsyncKeyState
    return all(k(v) & 0x8000 for v in VK.values())


def _combo(key: str) -> bool:
    """Ctrl+Alt+<key> held right now."""
    k = ctypes.windll.user32.GetAsyncKeyState
    return bool(k(VK["ctrl"]) & 0x8000 and k(VK["alt"]) & 0x8000 and k(VK_KEYS[key]) & 0x8000)


SD_DEG = 0.9             # per-fixation error after warm-up + check (targets, 2026-11-06): sent to web pages
SD_X_DEG, SD_Y_DEG = 0.7, 1.0   # same test, per axis: vertical is the hard one


def dot(radius: int = 12, learn: bool = True, warm: bool = True, rewarm: bool = False, serve: bool = False,
        show: bool = True, control: bool = False, click: bool = False) -> None:
    if not USER_MODEL.exists():
        raise SystemExit("no hay modelo todavía: corre `efg calibrate` primero")
    model = GazeModel.load()
    if warm:   # posture is the biggest source of error: adapt to the one you're in now
        from .track import warm_model, warmup
        model = warmup(model) if rewarm else warm_model(model)
    state = {"model": model}
    screen.make_dpi_aware()
    W, H = screen.screen_size_px()
    server = None
    if serve:
        from .web import GazeServer
        server = GazeServer(log_path=USER_MODEL.parent / "web_log.jsonl")
        print(f"servidor de mirada en {server.url} · página de prueba: {server.url}/demo")
    ppd = screen.px_per_degree()
    sd_px, sdx_px, sdy_px = SD_DEG * ppd, SD_X_DEG * ppd, SD_Y_DEG * ppd
    scanner = snapper = squint = None
    ctl = {"frozen": None, "fire": None, "snap": None, "clicks": 0}
    if control:
        from .desktop import GESTURE_FILE, Snapper, SquintClick, UIAScanner, load_gesture
        scanner, snapper = UIAScanner(W, H), Snapper()
        gest = load_gesture(USER_MODEL.parent / GESTURE_FILE)
        if click:
            if not gest:
                print("   aviso: sin `efg gesture` uso un umbral genérico para entrecerrar; mejor grábalo (20 s)")
        print("control del escritorio: botones de la ventana activa y la barra de tareas"
              + (" · ENTRECIERRA los ojos ~0,3 s para hacer clic" if click else " · (clic desactivado: --click)"))

    aff = AffineKalman.seeded(W, H, model)   # global deformation (fast, few labels); starts from the warm-up check
    corr = ResidualCorrector()       # local residual field on top (needs many labels)
    blinker = BlinkDetector()
    # Kalman while the gaze moves (no lag), fixation filter once it's still (steady); see filters.py
    fixf = GazeSmoother(still_s=0.2, meas_sd_px=50, accel_sd_px_s2=1500, lead_s=0.08)
    # Typing shakes the laptop, so the camera and the dot shake. While keys are being pressed the
    # eyes sit on the text anyway: a wider, longer fixation filter holds the dot still.
    fixf_typing = FixationFilter(radius=TYPING_RADIUS_PX, window=20)
    typing = {"last_key": -1e9}
    lock = threading.Lock()
    # (t, base, h, sample, blink): 15 s, because reading labels are decided 2-10 s after the line was read
    recent: "deque[tuple]" = deque(maxlen=450)
    gen = {"n": 0}     # bumped whenever the correction is reset/replaced: pages re-sync their own filters
    posture: "deque[tuple]" = deque(maxlen=400)  # ~20 s of (pitch, yaw, eye width, lid) for the drift check
    warm_pose = getattr(model, "warm_pose", None)
    from .track import bank_entries, pose_distance
    bank = bank_entries() if warm else []
    wp = {"pose": warm_pose, "bank_checked": False}
    from .track import pose_drift      # here, not at the top: track imports overlay
    labels_hist: "deque[tuple]" = deque(maxlen=40)   # (feature rows, patch rows, h, pos, t, conf) to re-fit after a fine-tune
    latest: dict = {}
    stats = {"keys_no_caret": 0, "labels": 0}
    rec = Recorder()
    # Diagnostic trace of every frame (no images): what the model said, what was shown, whether
    # it was held, and where the mouse was. This is what lets us debug "it never reaches the bottom".
    trace = Recorder(save_frames=False)
    trace_seg = Segment("dotlog", 0.0, None)
    TRACE_MAX = 30 * 60 * 20

    def on_sample(s: Sample) -> None:
        nonlocal squint
        if control and click and s.blendshapes is not None:
            if squint is None:
                from .features import BLENDSHAPE_NAMES
                g = load_gesture(USER_MODEL.parent / "gesture.json")
                squint = SquintClick(list(BLENDSHAPE_NAMES), thr=g.get("thr", 0.25), hold_s=g.get("hold_s", 0.3))
            started, fire = squint(s.blendshapes, s.t)
            if started:
                ctl["frozen"] = ctl["snap"]        # lids narrowing shift the estimate: decide NOW
            if fire and ctl["frozen"] is not None:
                ctl["fire"] = ctl["frozen"]
        base = state["model"].predict_full(s.features, s.patches)[0]
        h = pose_vector(head_euler(s.matrix), float(s.features[6]))
        is_blink = blinker(lid_opening(s), s.eye_blink)
        held = is_blink or not s.eyes_ok
        if recheck["on"] and not held and time.monotonic() - recheck["t0"] >= RECHECK_SETTLE_S:
            recheck["got"].append(base)
        if not held:
            he = head_euler(s.matrix)
            posture.append((he[0], he[1], float(s.features[6]), lid_opening(s)))
        with lock:
            recent.append((s.t, base, h, s, held))
            if len(recent) > 60 and recent[-61][3].frame is not None:
                recent[-61][3].frame = None      # keep full HD frames for the last ~3 s only (RAM)
            a = aff.apply(base)
            c = a - base + corr.correction(a, h, s.t)
        out = latest.get("p", (np.nan, np.nan))
        p_raw = base + c
        outside = (p_raw[0] < -OFF_MARGIN_PX or p_raw[0] > W + OFF_MARGIN_PX
                   or p_raw[1] < -OFF_MARGIN_PX or p_raw[1] > H + OFF_MARGIN_PX)
        latest["off_run"] = latest.get("off_run", 0) + 1 if (outside and not held) else 0
        if latest["off_run"] >= OFF_FRAMES:            # looking at the keyboard / away: hide the dot
            latest["off_t"] = s.t
        if not held and not outside:
            p = p_raw
            x, y = float(np.clip(p[0], 0, W)), float(np.clip(p[1], 0, H))
            calm, still = fixf(x, y, s.t), fixf_typing(x, y)      # both see every frame: no jump on switch
            out = still if s.t - typing["last_key"] < TYPING_HOLD_S else calm
            latest["p"], latest["t"] = out, s.t
        cur = cursor_position()
        if server is not None:
            ok = bool(not held and not outside and latest.get("off_run", 0) < OFF_FRAMES and np.isfinite(out[0]))
            server.push({"t": round(s.t, 3), "x": round(float(out[0]), 1) if ok else None,
                         "y": round(float(out[1]), 1) if ok else None, "ok": ok, "cx": int(cur[0]), "cy": int(cur[1]),
                         "rx": round(float(p_raw[0]), 1) if ok else None, "ry": round(float(p_raw[1]), 1) if ok else None,
                         "ey": round(float(c[1]), 1), "gen": gen["n"],
                         "sd": round(sd_px, 1), "sdx": round(sdx_px, 1), "sdy": round(sdy_px, 1), "W": W, "H": H})
        if len(trace) < TRACE_MAX:
            trace.add(s, seg_i=0, local_t=0.0, seg=trace_seg, target_now=cur, target=cur, valid=False,
                      extra={"base": base, "corr": c, "out": out, "blink": is_blink, "held": held, "outside": outside})

    # --- window ----------------------------------------------------------
    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry(f"{W}x{H}+0+0")
    root.configure(bg=TRANSPARENT)
    root.attributes("-transparentcolor", TRANSPARENT)
    root.attributes("-topmost", True)
    canvas = tk.Canvas(root, bg=TRANSPARENT, highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    dot_id = canvas.create_oval(0, 0, 0, 0, fill="#ff2d20", outline="#ffffff", width=2, state="hidden")
    spring = Spring2D(omega=28.0)        # display physics: glide, don't teleport
    lens_id = canvas.create_rectangle(0, 0, 0, 0, outline="#4fc3ff", width=3, state="hidden")
    ring_id = canvas.create_arc(0, 0, 0, 0, start=90, extent=0, style="arc", outline="#e8a33d", width=4, state="hidden")
    lens_spring = Spring2D(omega=30.0)
    lens_spring.p = None

    def looking_away_now(now: float) -> bool:
        return now - latest.get("off_t", -1e9) < 0.25
    keys_prev: dict = {}
    recheck = {"on": False, "i": 0, "t0": 0.0, "got": [], "res": []}

    def pressed(key: str) -> bool:      # rising edge of Ctrl+Alt+key
        now_down = _combo(key)
        was = keys_prev.get(key, False)
        keys_prev[key] = now_down
        return now_down and not was

    def nudge(dx: float, dy: float) -> None:
        with lock:
            aff.theta[2] += dx / W
            aff.theta[5] += dy / H
            gen["n"] += 1
        print(f"   empujón ({dx:+.0f}, {dy:+.0f}) px · {aff.describe()}")

    def recheck_tick(now: float) -> None:
        """Ctrl+Alt+R: 5 buttons, ~7 s, without leaving `dot`. Their residuals replace the global
        correction (same as the warm-up's 9-button check)."""
        canvas.delete("recheck")
        i = recheck["i"]
        if i >= len(RECHECK_POINTS):
            res = recheck["res"]
            recheck["on"] = False
            if len(res) >= 3:
                with lock:
                    chk = AffineKalman.for_check(W, H)       # clean labels: looser gain prior
                    for p, lab, t in res:
                        chk.add(p, lab, t)
                    aff.adopt(chk)
                    gen["n"] += 1
                    corr.forget()
                    desc = aff.describe()
                e = np.median([np.linalg.norm(p - lab) for p, lab, _ in res]) / screen.px_per_degree()
                print(f"   recalibrado con {len(res)} botones (error antes {e:.2f}°) · {desc}")
            else:
                print("   recalibración: no vi bien tus ojos, sin cambios")
            return
        fx, fy = RECHECK_POINTS[i]
        x, y = fx * W, fy * H
        h = 1.5 * screen.px_per_degree()
        canvas.create_rectangle(x - h, y - h / 2, x + h, y + h / 2, fill="#e8a33d", outline="#ffd27a", width=2, tags="recheck")
        canvas.create_text(x, y, text="mira aquí", fill="#1b1b1b", font=("Segoe UI", int(h * 0.28)), tags="recheck")
        lt = now - recheck["t0"]
        if lt >= RECHECK_S:
            if len(recheck["got"]) >= 5:
                recheck["res"].append((np.median(recheck["got"], 0), np.array([x, y]), now))
            recheck["i"] += 1
            recheck["t0"], recheck["got"] = now, []
    root.update_idletasks()
    screen.make_click_through(ctypes.windll.user32.GetParent(root.winfo_id()))

    def flash(pos, ok: bool) -> None:
        x, y = pos
        i = canvas.create_oval(x - 16, y - 16, x + 16, y + 16, outline="#34c759" if ok else "#8e8e93", width=2)
        root.after(350, lambda: canvas.delete(i))

    # --- live fine-tuning of the eye network from accepted labels -------------
    tune = {"X": [], "Y": [], "P": [], "n_new": 0, "busy": False, "last": time.monotonic(), "fits": 0, "prior": None}
    can_tune = learn and getattr(state["model"], "appearance", None) is not None

    def load_prior() -> None:
        from .tracker import load_sessions
        try:
            d = load_sessions(with_patches=True)
        except SystemExit:
            return
        rows = np.where(d["has_P"])[0]
        idx = rows[np.linspace(0, len(rows) - 1, min(3000, len(rows))).astype(int)]
        tune["prior"] = (d["X"][idx], d["Y"][idx], d["P"][idx])

    if can_tune:
        threading.Thread(target=load_prior, daemon=True).start()

    def maybe_tune(now: float) -> None:
        if (not can_tune or tune["busy"] or tune["prior"] is None or tune["n_new"] < TUNE_EVERY
                or now - tune["last"] < TUNE_MIN_GAP_S):
            return
        tune["busy"], tune["n_new"] = True, 0
        k = min(len(tune["X"]), 1500)
        Xn, Yn, Pn = np.array(tune["X"][-k:]), np.array(tune["Y"][-k:]), np.array(tune["P"][-k:])

        def work():
            import copy

            from .appearance import finetune_copy
            try:
                Xp, Yp, Pp = tune["prior"]
                # Click/typing labels come clustered (a few spots, a line of text): fine-tuning on them
                # alone drifts the rest of the screen (simulated: 0.4° -> up to 1.0°). So: 4x replay
                # of the calibration sessions, a smaller step, fewer iterations.
                rep = np.random.default_rng(tune["fits"]).choice(len(Xp), min(max(4 * k, 600), len(Xp)), replace=False)
                tuned = finetune_copy(state["model"].appearance, np.concatenate([Pn, Pp[rep]]),
                                      np.vstack([Xn, Xp[rep]]), np.vstack([Yn, Yp[rep]]), steps=30, lr=3e-4,
                                      seed=tune["fits"])
                m = copy.copy(state["model"])
                m.appearance = tuned
                # Re-fit the fast correctors with the NEW network instead of wiping them: the fine-tune
                # (heavy replay, few steps) absorbs only part of a posture bias, and wiping threw the
                # rest away after every ~6 clicks.
                hist = list(labels_hist)
                refit = [(np.median(tuned.predict(P, X), 0), h, pos, t, conf) for X, P, h, pos, t, conf in hist]
                with lock:
                    state["model"] = m
                    gen["n"] += 1
                    aff.forget()
                    corr.forget()
                    for p, h, pos, t, conf in refit:
                        if aff.add(p, pos, t, conf):
                            corr.add(aff.apply(p), h, pos, t, conf)
                tune["fits"] += 1
            finally:
                tune["last"] = time.monotonic()
                tune["busy"] = False

        threading.Thread(target=work, daemon=True).start()

    worker = FeatureWorker(on_sample)
    watcher = InputWatcher().start() if learn else None
    last_key_label = [0.0]
    tick_t = [time.monotonic()]
    drift = {"checked": time.monotonic(), "warned": -1e9}
    last_print = [time.monotonic()]
    print("punto rojo activo · aprende de clics y escritura" if learn else "punto rojo activo",
          "· Ctrl+Alt+G salir · Ctrl+Alt+flechas empujar · Ctrl+Alt+R recalibrar (5 botones)")

    def absorb(kind: str, t_ev: float, pos, window: tuple | None = None, sd_px: tuple | None = None,
               y_only: bool = False) -> None:
        """window given explicitly = already in sample time (web labels); else WINDOWS (eye time)."""
        if window is None:
            lo, hi, conf = WINDOWS[kind]
            lo, hi = lo + LATENCY_S, hi + LATENCY_S
        else:
            lo, hi, conf = window
        with lock:
            rows = [r for r in recent if t_ev + lo <= r[0] <= t_ev + hi and not r[4]]
        if len(rows) < 3:
            return
        p = np.median([r[1] for r in rows], axis=0)
        h = np.median([r[2] for r in rows], axis=0)
        exo = kind in ("click", "caret")     # exogenous labels; gaze-derived ones (page snaps, reading) are not
        with lock:
            if y_only:
                pos = (float(aff.apply(p)[0]), pos[1])       # zero x innovation
            ok = aff.add(p, pos, t_ev, conf, sd_px)
            if ok and exo:
                corr.add(aff.apply(p), h, pos, t_ev, conf)
                if rows[0][3].patches is not None:
                    labels_hist.append((np.array([r[3].features for r in rows]), np.array([r[3].patches for r in rows]),
                                        h, pos, t_ev, conf))
        seg = Segment(kind, 0.0, None)
        idx = stats["labels"]
        stats["labels"] += 1
        for r in rows:
            rec.add(r[3], seg_i=idx, local_t=1.0, seg=seg, target_now=pos, target=pos, valid=ok)
            # the eye network learns only from EXOGENOUS labels (clicks, caret): page snaps are the
            # tracker's own decisions, and 4-10% of them name the wrong element in simulation — a
            # pseudo-label confirmation loop if they reached the network
            if ok and can_tune and exo and r[3].patches is not None:
                tune["X"].append(r[3].features)
                tune["Y"].append(np.asarray(pos, np.float32))
                tune["P"].append(r[3].patches)
                tune["n_new"] += 1
        if exo:
            flash(pos, ok)
        else:
            stats["gaze_labels"] = stats.get("gaze_labels", 0) + int(ok)

    def tick() -> None:
        if _hotkey_down():
            root.destroy()
            return
        now = time.monotonic()
        for key, (dx, dy) in {"left": (-NUDGE_PX, 0), "right": (NUDGE_PX, 0), "up": (0, -NUDGE_PX), "down": (0, NUDGE_PX)}.items():
            if pressed(key):
                nudge(dx, dy)
        if pressed("r") and not recheck["on"]:
            recheck.update(on=True, i=0, t0=now, got=[], res=[])
            print("   recalibración rápida: mira cada botón naranja")
        if recheck["on"]:
            recheck_tick(now)
        if watcher:
            typing["last_key"] = max(typing["last_key"], watcher.last_key)   # live: the label queue lags 0.55 s
            for ev in watcher.drain_ready(now, LABEL_DELAY_S):
                if ev.kind != "click":
                    typing["last_key"] = max(typing["last_key"], ev.t)
                if ev.kind == "click":
                    absorb("click", ev.t, ev.pos)
                elif ev.t - last_key_label[0] >= KEY_MIN_GAP_S:
                    pos = caret_position()
                    if pos is None or not (0 <= pos[0] < W and 0 <= pos[1] < H):
                        stats["keys_no_caret"] += 1
                        continue
                    last_key_label[0] = ev.t
                    absorb("caret", ev.t, pos)
        if server is not None and learn:
            while not server.labels.empty():
                lab = server.labels.get_nowait()
                if 0 <= lab["y"] < H and 0 < lab["t1"] - lab["t0"] <= 3.0 and (lab["axis"] == "y" or 0 <= lab["x"] < W):
                    if lab["axis"] == "y":       # reading: the line is known, the x is not
                        sd = (1e4, max(lab.get("sdy", 30.0), 25.0))
                        conf = min(lab["conf"], 0.3)
                    else:
                        sd = (lab["sdx"], lab["sdy"]) if "sdx" in lab else None
                        conf = lab["conf"]
                    absorb(lab["kind"], lab["t1"], (lab["x"], lab["y"]), (lab["t0"] - lab["t1"], 0.0, conf), sd,
                           y_only=lab["axis"] == "y")
        maybe_tune(now)
        if wp["pose"] and now - drift["checked"] > 5 and len(posture) > 200:
            drift["checked"] = now
            a = np.array(posture)
            cur = {"pitch": float(np.median(a[:, 0])), "yaw": float(np.median(a[:, 1])),
                   "eye_w": float(np.median(a[:, 2])), "lid": float(np.median(a[:, 3]))}
            why = pose_drift(cur, wp["pose"])
            if bank and (why or not wp["bank_checked"]):
                # posture bank: the warm-up fitted closest to how you sit now
                wp["bank_checked"] = True
                d, (pose, path) = min(((pose_distance(cur, b[0]), b) for b in bank), key=lambda z: z[0])
                if not pose_drift(cur, pose) and pose is not wp["pose"]:
                    m = GazeModel.load(path)
                    with lock:
                        state["model"] = m
                        aff.adopt(AffineKalman.seeded(W, H, m))
                        corr.forget()
                        labels_hist.clear()
                        gen["n"] += 1
                    wp["pose"] = pose
                    why = []
                    print(f"   banco de posturas: uso el calentamiento {path.stem[5:]} (distancia {d:.2f})")
            if why and now - drift["warned"] > 90:
                drift["warned"] = now
                msg = ("Postura nueva (" + ", ".join(why) + ").  Ctrl+Alt+R recalibra en 7 s · "
                       "`--rewarm` la añade al banco")
                print("   ⚠", msg)
                gen["n"] += 1
                canvas.delete("toast")
                canvas.create_rectangle(W / 2 - 430, 14, W / 2 + 430, 54, fill="#202328", outline="#e8a33d", width=2, tags="toast")
                canvas.create_text(W / 2, 34, text=msg, fill="#f0f0f0", font=("Segoe UI", 12), tags="toast")
                root.after(7000, lambda: canvas.delete("toast"))
        p = latest.get("p")
        looking_away = now - latest.get("off_t", -1e9) < 0.25
        dt = now - tick_t[0]
        tick_t[0] = now
        if control:
            els, rects = scanner.snapshot()
            g = latest.get("p") if (latest.get("p") is not None and now - latest.get("t", -1e9) < 0.6
                                    and not looking_away_now(now)) else None
            r = snapper(None if g is None else np.asarray(g, float), sdx_px, sdy_px, els, rects, W * H)
            ctl["snap"] = r
            if r is not None:
                tgt = [r[0] - 5, r[1] - 5, r[2] + 5, r[3] + 5]
                if canvas.itemcget(lens_id, "state") == "hidden":
                    lens_spring.reset(tgt)
                l, t_, rr, b = lens_spring(tgt, dt)
                canvas.coords(lens_id, l, t_, rr, b)
                canvas.itemconfigure(lens_id, state="normal")
            else:
                canvas.itemconfigure(lens_id, state="hidden")
            prog = squint.progress if squint is not None else 0.0
            if prog > 0 and ctl["frozen"] is not None:
                fx0, fy0, fx1, fy1 = ctl["frozen"]
                cx, cy, rad = (fx0 + fx1) / 2, (fy0 + fy1) / 2, 0.5 * max(fx1 - fx0, fy1 - fy0) + 14
                canvas.coords(ring_id, cx - rad, cy - rad, cx + rad, cy + rad)
                canvas.itemconfigure(ring_id, extent=-359.9 * prog, state="normal")
            else:
                canvas.itemconfigure(ring_id, state="hidden")
            if ctl["fire"] is not None:
                fx0, fy0, fx1, fy1 = ctl["fire"]
                ctl["fire"], ctl["frozen"] = None, None
                from .desktop import click_at
                click_at((fx0 + fx1) / 2, (fy0 + fy1) / 2)
                ctl["clicks"] += 1
        if show and p is not None and now - latest["t"] < 0.6 and not looking_away and not recheck["on"]:
            if canvas.itemcget(dot_id, "state") == "hidden":
                spring.reset(p)          # reappearing: start where the gaze is, don't fly in
            x, y = spring(p, dt)
            canvas.coords(dot_id, x - radius, y - radius, x + radius, y + radius)
            canvas.itemconfigure(dot_id, state="normal")
        else:
            canvas.itemconfigure(dot_id, state="hidden")
        root.attributes("-topmost", True)
        if now - last_print[0] > 10:
            last_print[0] = now
            with lock:
                desc = aff.describe()
            print(f"   {worker.stats()} · etiquetas ✓{aff.accepted} ✗{aff.rejected} · "
                  f"teclas sin caret {stats['keys_no_caret']} · por mirada {stats.get('gaze_labels', 0)} · {desc}"
                  + (f" · {len(scanner.elements)} botones ({scanner.scan_ms:.0f} ms)" + (f" · {scanner.error}" if scanner.error else "") if scanner else "")
                  + (f" · red ajustada {tune['fits']} veces" if can_tune else ""))
        root.after(16, tick)

    root.after(16, tick)
    # Ctrl+C in the terminal: close the window cleanly so the sessions below still get saved
    import signal
    signal.signal(signal.SIGINT, lambda *_: root.after(0, root.destroy))
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        if watcher:
            watcher.stop()
        worker.close()
        if len(trace):
            print("registro de diagnóstico:", trace.save(W, H, "dotlog", "traza por fotograma de `efg dot`", worker.stats()))
        if scanner is not None:
            scanner.close()
            print(f"control: {ctl['clicks']} clics con la mirada")
        if len(rec):
            path = rec.save(W, H, "implicit", "clics y escritura durante `efg dot`", worker.stats())
            print(f"guardadas {stats['labels']} etiquetas implícitas ({len(rec)} muestras) en {path}")
            print("corre `efg train` para incorporarlas al modelo base")
