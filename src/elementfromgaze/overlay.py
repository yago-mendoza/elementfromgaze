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
# Recalibration (Ctrl+Alt+R, or automatic on a new posture): follow a ring along the SAME path as the
# warm-up's first 45 s (18 s curve + raster sweeps top to bottom) + 3 still points, then re-fit the
# network (4 rounds, background) and bank the posture. Measured by re-fitting with the first N s of
# real warm-ups on 5 held-out sessions: 12 s 1.77°, 20 s 1.62°, 30 s 1.60°, 45 s 1.33° (= full
# warm-up 1.34°; close reading posture 2.44° -> 1.28°). What matters is covering the whole screen,
# edges included; 4 rounds beat 8. Long, but once per posture: the bank remembers it.
RECAL_POINT_S, RECAL_SETTLE_S, RECAL_LAG_S = 1.3, 0.5, 0.10
RECAL_POINTS = [(0.12, 0.14), (0.88, 0.14), (0.5, 0.88)]
RECAL_ROUNDS = 4
RECAL_CURVE_S, RECAL_ROWS, RECAL_SPEED = 18.0, [0.06, 0.3, 0.55, 0.8, 0.95], 380.0


def _recal_plan(W: int, H: int):
    """(path(t), duration): the warm-up's curve, then its raster sweeps."""
    from .track import _lissajous, _raster
    curve = _lissajous(W, H, 3.1)
    raster, dur = _raster(W, H, RECAL_ROWS, RECAL_SPEED)
    total = RECAL_CURVE_S + dur

    def path(t: float) -> tuple[float, float]:
        return curve(t) if t < RECAL_CURVE_S else raster(t - RECAL_CURVE_S)

    return path, total
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
TUNE_MIN_ZONES = 4       # of a 3x3 grid: the network only generalises from spread-out labels


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
    ctl = {"frozen": None, "fire": None, "snap": None, "clicks": 0, "hist": deque(maxlen=40)}   # hist: (t, snapped rect)
    if control:
        from .desktop import GESTURE_FILE, Snapper, UIAScanner, load_gesture, make_clicker
        scanner, snapper = UIAScanner(W, H), Snapper()
        gest = load_gesture(USER_MODEL.parent / GESTURE_FILE)
        if click:
            if not gest:
                print("   aviso: sin `efg gesture` uso valores genéricos para el parpadeo largo; mejor grábalo (25 s)")
        how = "ENTRECIERRA ~0,3 s" if gest.get("mode") == "squint" else f"PARPADEO LARGO (~{gest.get('hold_s', 0.45):.2g} s)"
        print("control del escritorio: botones de la ventana activa y la barra de tareas"
              + (f" · {how} para hacer clic" if click else " · (clic desactivado: --click)"))

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
                squint = make_clicker(list(BLENDSHAPE_NAMES), load_gesture(USER_MODEL.parent / GESTURE_FILE))
            started, fire = squint(s.blendshapes, s.t)
            if started:
                # closing lids drag the estimate down for ~0.1-0.2 s before the gesture registers:
                # decide on the element snapped just BEFORE that (minus the camera latency)
                now_m = time.monotonic()
                prior = [r for (tt, r) in list(ctl["hist"]) if tt <= now_m - 0.25]
                ctl["frozen"] = prior[-1] if prior else ctl["snap"]
            if fire and ctl["frozen"] is not None:
                ctl["fire"] = ctl["frozen"]
        base = state["model"].predict_full(s.features, s.patches)[0]
        h = pose_vector(head_euler(s.matrix), float(s.features[6]))
        is_blink = blinker(lid_opening(s), s.eye_blink)
        held = is_blink or not s.eyes_ok
        if recheck["on"] and not held:
            te = s.t - LATENCY_S - recheck["t0"]          # when the eye saw what this frame shows
            tgt, kind = None, None
            if 0.8 <= te < recal_T:
                tgt, kind = recal_path(te - RECAL_LAG_S), "pursuit"
            elif te >= recal_T:
                k, lt = divmod(te - recal_T, RECAL_POINT_S)
                if int(k) < len(RECAL_POINTS) and lt >= RECAL_SETTLE_S:
                    fx, fy = RECAL_POINTS[int(k)]
                    tgt, kind = (fx * W, fy * H), f"fix{int(k)}"
            if tgt is not None:
                recheck["rows"].append((s, base, np.asarray(tgt, float), kind))
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
    recheck = {"on": False, "t0": 0.0, "rows": [], "busy": False}   # rows: (sample, base, target, kind)
    recal_path, recal_T = _recal_plan(W, H)
    rechk_rec = Recorder()     # recheck fixations: clean labels in NEW postures -> training data

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

    def current_pose() -> dict | None:
        if len(posture) < 100:
            return None
        a = np.array(posture)
        return {"pitch": float(np.median(a[:, 0])), "yaw": float(np.median(a[:, 1])),
                "eye_w": float(np.median(a[:, 2])), "lid": float(np.median(a[:, 3]))}

    def bank_this_posture() -> bool:
        """After a recheck: keep (current network + this correction) for this posture, so the next
        time you sit like this the bank picks it up with no recalibration."""
        import copy

        from .track import _bank_dir
        pose = current_pose()
        if pose is None or not warm:
            return False
        m = copy.copy(state["model"])
        m.start_aff, m.warm_pose = aff.state(), pose
        d = _bank_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"warm_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
        m.save(path)
        bank.append((pose, path))
        wp["pose"] = pose
        return True

    def start_recheck(auto: bool = False) -> None:
        if recheck["on"] or recheck["busy"]:
            return
        recheck.update(on=True, t0=time.monotonic(), rows=[])
        print("   recalibración" + (" automática" if auto else "") + f": sigue el aro naranja (≈{recal_T + 4:.0f} s)")

    def finish_recheck() -> None:
        """Re-fit the network to this posture (background), then the global correction from the 3
        still points measured with the NEW network; bank the result; keep the frames as training data."""
        rows = [r for r in recheck["rows"] if r[0].patches is not None]
        pts = [r for r in rows if r[3].startswith("fix")]
        recheck["on"] = False
        if len(rows) < 120 or len({r[3] for r in pts}) < 2:
            print("   recalibración: no vi bien tus ojos, sin cambios")
            toaster.show("Recalibración sin cambios", "No vi bien tus ojos", "warn", 3500)
            return
        ppd_ = screen.px_per_degree()
        err_before = np.median([np.linalg.norm(aff.apply(r[1]) - r[2]) for r in pts]) / ppd_
        for i, (smp, _, tgt, kind) in enumerate(rows):             # training data for `efg train`
            if smp.frame is not None:
                seg = Segment("pursuit" if kind == "pursuit" else "fix", 0.0, None)
                rechk_rec.add(smp, seg_i=0 if kind == "pursuit" else 1 + int(kind[3:]), local_t=1.0, seg=seg,
                              target_now=tuple(tgt), target=tuple(tgt), valid=True)
        recheck["busy"] = True
        toaster.show("Ajustando la red a esta postura…", "Unos segundos; puedes seguir usando el ordenador", "info", 4000)

        def work():
            import copy
            try:
                m = state["model"]
                X = np.array([r[0].features for r in rows])
                P = np.array([r[0].patches for r in rows])
                Y = np.array([r[2] for r in rows], np.float32)
                app = getattr(m, "appearance", None)
                if app is not None and tune.get("prior") is not None:
                    from .appearance import finetune_copy
                    Xp, Yp, Pp = tune["prior"]
                    rng = np.random.default_rng(int(time.time()))
                    for k in range(RECAL_ROUNDS):          # same recipe as the warm-up, half the rounds
                        rep = rng.choice(len(Xp), min(len(X), len(Xp)), replace=False)
                        app = finetune_copy(app, np.concatenate([P, Pp[rep]]), np.vstack([X, Xp[rep]]),
                                            np.vstack([Y, Yp[rep]]), steps=40, seed=k)
                    m = copy.copy(m)
                    m.appearance = app
                    tuned = True
                else:
                    tuned = False
                chk = AffineKalman.for_check(W, H)
                for kind in sorted({r[3] for r in pts}):
                    q = [i for i, r in enumerate(rows) if r[3] == kind]
                    pred = m.predict_full(X[q], P[q]) if tuned else np.array([rows[i][1] for i in q])
                    chk.add(np.median(pred, 0), rows[q[0]][2], time.monotonic())
                with lock:
                    state["model"] = m
                    aff.adopt(chk)
                    corr.forget()
                    labels_hist.clear()
                    gen["n"] += 1
                saved = bank_this_posture()
                msg = f"error antes {err_before:.1f}°" + (" · red ajustada" if tuned else "") + (" · postura guardada" if saved else "")
                print(f"   recalibrado: {msg} · {aff.describe()}")
                recheck["toast"] = ("Recalibrado", msg, "ok", 3500)     # shown by the Tk thread
            except Exception as e:      # never kill the overlay over a recalibration
                print("   recalibración falló:", e)
            finally:
                recheck["busy"] = False

        threading.Thread(target=work, daemon=True).start()

    def recheck_tick(now: float) -> None:
        canvas.delete("recheck")
        t = now - recheck["t0"]
        total = recal_T + len(RECAL_POINTS) * RECAL_POINT_S
        if t >= total + 0.3:
            finish_recheck()
            return
        if t < recal_T:
            x, y = recal_path(t)
            r = 15
        else:
            fx, fy = RECAL_POINTS[min(int((t - recal_T) // RECAL_POINT_S), len(RECAL_POINTS) - 1)]
            x, y = fx * W, fy * H
            r = 15 + 14 * max(0.0, 1 - ((t - recal_T) % RECAL_POINT_S) / RECAL_SETTLE_S)   # shrinking ring
        canvas.create_oval(x - r, y - r, x + r, y + r, outline="#e8a33d", width=4, tags="recheck")
        canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#e8a33d", outline="", tags="recheck")
    root.update_idletasks()
    screen.make_click_through(ctypes.windll.user32.GetParent(root.winfo_id()))
    from .toast import Toaster
    toaster = Toaster(root, W, H)

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
        k = min(len(tune["X"]), 1500)
        Xn, Yn, Pn = np.array(tune["X"][-k:]), np.array(tune["Y"][-k:]), np.array(tune["P"][-k:])
        # Clicks piled up in one spot only say something about THAT spot: fine-tuning on them bent
        # the rest of the screen ("it rotates everything else", user, 2026-11-06). The network only
        # adapts once the labels cover >= 4 of 9 screen zones, and each zone weighs the same.
        zone = (np.clip(Yn[:, 0] / W * 3, 0, 2.999).astype(int) * 3 + np.clip(Yn[:, 1] / H * 3, 0, 2.999).astype(int))
        zones, counts = np.unique(zone, return_counts=True)
        if len(zones) < TUNE_MIN_ZONES:
            tune["last"] = now          # re-check after TUNE_MIN_GAP_S, not every frame
            if not tune.get("told"):
                print(f"   la red espera clics más repartidos ({len(zones)}/9 zonas; hacen falta {TUNE_MIN_ZONES})")
                tune["told"] = True
            return
        cap = int(max(np.median(counts), 20))
        keep = np.concatenate([np.where(zone == z)[0][-cap:] for z in zones])
        Xn, Yn, Pn = Xn[keep], Yn[keep], Pn[keep]
        k = len(keep)
        tune["busy"], tune["n_new"], tune["told"] = True, 0, False

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
            start_recheck()
        if recheck["on"]:
            recheck_tick(now)
        if recheck.get("toast"):
            toaster.show(*recheck["toast"])
            recheck["toast"] = None
        if watcher:
            typing["last_key"] = max(typing["last_key"], watcher.last_key)   # live: the label queue lags 0.55 s
            for ev in watcher.drain_ready(now, LABEL_DELAY_S):
                if ev.kind != "click":
                    typing["last_key"] = max(typing["last_key"], ev.t)
                if ev.kind == "click":
                    absorb("click", ev.t, ev.pos)
                elif ev.t - last_key_label[0] >= KEY_MIN_GAP_S:
                    ca = watcher.caret        # accessibility caret (Chromium & co.) if it moved with this key
                    pos = (ca[0], ca[1]) if ca is not None and -0.1 <= ca[3] - ev.t <= 0.45 else caret_position()
                    if pos is None or not (0 <= pos[0] < W and 0 <= pos[1] < H):
                        stats["keys_no_caret"] += 1
                        continue
                    last_key_label[0] = ev.t
                    # per-axis noise from the literature (EyeTyper '18): typists' gaze sits near the
                    # caret in x, much looser in y (they glance below / at the keys)
                    absorb("caret", ev.t, pos, sd_px=(1.3 * ppd, 2.5 * ppd))
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
                    toaster.show("Postura reconocida", f"Uso tu calentamiento del {path.stem[11:13]}/{path.stem[9:11]} a las "
                                 f"{path.stem[14:16]}:{path.stem[16:18]}", "info", 3500)
            if why and now - drift["warned"] > 90:
                drift["warned"] = now
                # a new posture: the 2026-11-06 reading test ran 37% closer than the warm-up with a
                # 3.5° vertical error and only a toast nobody acted on. So recalibrate by itself
                # (5 buttons, 7 s) and bank the posture, after a short heads-up.
                print("   ⚠ postura nueva (" + ", ".join(why) + "): recalibro en 3 s")
                gen["n"] += 1
                toaster.show("Postura nueva: " + ", ".join(why),
                             "Recalibro en 3 s: sigue el aro naranja (≈45 s). Después recuerdo esta postura.", "warn", 3000)
                root.after(3000, lambda: start_recheck(auto=True))
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
            ctl["hist"].append((now, r))
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
        if len(rechk_rec):
            print("recalibraciones guardadas como datos de entrenamiento:",
                  rechk_rec.save(W, H, "recheck", "botones de Ctrl+Alt+R / postura nueva", worker.stats()))
        if len(rec):
            path = rec.save(W, H, "implicit", "clics y escritura durante `efg dot`", worker.stats())
            print(f"guardadas {stats['labels']} etiquetas implícitas ({len(rec)} muestras) en {path}")
            print("corre `efg train` para incorporarlas al modelo base")
