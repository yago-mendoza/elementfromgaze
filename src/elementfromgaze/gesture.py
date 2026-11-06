"""`efg gesture`: ~25 s to learn YOUR click gesture, a long blink, so it fires on it and never on
normal blinks.

Records MediaPipe's blendshapes while you look normally, blink normally on cue, and close your eyes
for about half a second on cue. "Closed" = min(eyeBlinkLeft, eyeBlinkRight) above a threshold set
halfway between your open eyes (99th percentile) and the middle of your long blinks. The hold time
goes between your longest normal blink and your shortest long one. Saved to data/gesture.json.

(Squinting was the first idea: for this user MediaPipe's eyeSquint score went DOWN when squinting
and overlapped normal frames, so it is kept only as an option, mode "squint".)
"""

from __future__ import annotations

import json

import numpy as np

from . import screen
from .app import Segment, Stage
from .tracker import USER_MODEL

TEXT = {"g_normal": "Mira el punto con normalidad", "g_rest": "relaja",
        "g_blink": "PARPADEA normal", "g_long": "CIERRA los ojos medio segundo"}


class GestureStage(Stage):
    title = "Gesto de clic: parpadeo largo (≈25 s)"
    intro_text = ("Sigue las instrucciones del centro de la pantalla.\n"
                  "Cuando ponga CIERRA, cierra los dos ojos como medio segundo y ábrelos.\n\n"
                  "ESPACIO para empezar · ESC para salir")

    def __init__(self, W, H):
        c = (lambda t, x=W / 2, y=H / 2: (x, y))
        segs = [Segment("g_normal", 6.0, c, 1.0)]
        for _ in range(4):
            segs += [Segment("g_blink", 1.0, c, 0.0), Segment("g_rest", 1.0, c, 0.0)]
        for _ in range(5):
            segs += [Segment("g_long", 1.6, c, 0.0), Segment("g_rest", 1.2, c, 0.0)]
        super().__init__(segs)

    def _draw(self, seg: Segment, lt: float) -> None:
        cv = self.canvas
        cv.delete("target")
        x, y = seg.target(lt)
        cv.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#ff3b30", outline="", tags="target")
        hot = seg.phase in ("g_blink", "g_long")
        cv.create_text(x, y - 70, text=TEXT[seg.phase], fill="#e8a33d" if hot else "#9a9a9a",
                       font=("Segoe UI", 26 if hot else 16), tags="target")


def _runs(closed: np.ndarray, t: np.ndarray, seg: np.ndarray) -> list[tuple[int, float]]:
    """(segment, duration) of every run of closed frames."""
    out, start = [], None
    for i in range(len(closed)):
        if closed[i] and start is None:
            start = i
        end = (not closed[i]) or i == len(closed) - 1 or seg[i + 1] != seg[i]
        if start is not None and end:
            j = i if closed[i] else i - 1
            out.append((int(seg[start]), float(t[j] - t[start]) + 0.05))     # +1 frame
            start = None
    return out


def analyse(rec) -> dict | None:
    from .features import BLENDSHAPE_NAMES

    if not len(rec):
        return None
    ix = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}
    B = np.array(rec.cols["blendshapes"], float)
    ph = np.array(rec.cols["phase"])
    lt = np.array(rec.cols["local_t"])
    t = np.array(rec.cols["t"], float)
    seg = np.array(rec.cols["seg"])
    bl = B[:, [ix["eyeBlinkLeft"], ix["eyeBlinkRight"]]].min(1)
    open_hi = float(np.percentile(bl[ph == "g_normal"], 99))
    long_mid = bl[(ph == "g_long") & (lt > 0.3) & (lt < 0.8)]
    if len(long_mid) < 5:
        return None
    long_med = float(np.median(long_mid))
    closed_thr = (open_hi + long_med) / 2
    runs = _runs(bl > closed_thr, t, seg)
    phase_of = {int(s): p for s, p in zip(seg, ph)}
    nat = [d for s, d in runs if phase_of.get(s) in ("g_blink", "g_normal", "g_rest")]
    lng = [max([d for s2, d in runs if s2 == s] or [0.0]) for s in np.unique(seg[ph == "g_long"])]
    nat_hi = float(max(nat)) if nat else 0.25
    lng_lo = float(np.percentile(lng, 20)) if lng else 0.0
    separable = long_med > open_hi + 0.1 and lng_lo > nat_hi + 0.1
    hold = float(np.clip((nat_hi + lng_lo) / 2, 0.35, 0.8)) if separable else 0.5
    return {"mode": "blink", "closed_thr": round(closed_thr, 3), "hold_s": round(hold, 2), "separable": bool(separable),
            "open_p99": round(open_hi, 3), "long_median": round(long_med, 3),
            "normal_blinks_s": [round(d, 2) for d in sorted(nat)], "long_blinks_s": [round(d, 2) for d in lng],
            "long_caught": round(float(np.mean(np.array(lng) >= hold)) if lng else 0.0, 2),
            "normal_over": round(float(np.mean(np.array(nat) >= hold)) if nat else 0.0, 2)}


def gesture() -> None:
    screen.make_dpi_aware()
    W, H = screen.screen_size_px()
    st = GestureStage(W, H)
    st.run()
    if st.aborted:
        print("cancelado")
        return
    rep = analyse(st.rec)
    if not rep:
        print("no hay datos suficientes (¿se te veían los ojos?)")
        return
    (USER_MODEL.parent / "gesture.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    if rep["separable"]:
        print(f"bien: un parpadeo de {rep['hold_s']} s o más hace clic; tus parpadeos normales no llegan "
              f"({100 * rep['long_caught']:.0f}% de los largos lo superan, {100 * rep['normal_over']:.0f}% de los normales)")
    else:
        print("ojo: tus parpadeos largos y normales se parecen; repite cerrando un poco más de tiempo")
