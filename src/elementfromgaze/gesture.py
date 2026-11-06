"""`efg gesture`: 20 s to learn YOUR squint, so squint-to-click fires on it and not on blinks or smiles.

Records MediaPipe's blendshapes while you look normally, squint on cue (narrow the eyes, don't
close them), blink on cue and smile on cue. The squint score is mean(eyeSquintLeft/Right) minus
your normal baseline; the threshold goes halfway between the highest non-squint frames (99th
percentile of normal + rest + blinks + smiles, with the same blink/smile gates the detector uses)
and the lower quarter of the squint frames. Saved to data/gesture.json.
"""

from __future__ import annotations

import json

import numpy as np

from . import screen
from .app import INTRO_BG, Segment, Stage, message
from .tracker import USER_MODEL

BLINK_MAX, SMILE_MAX = 0.6, 0.35

TEXT = {"g_normal": "Mira el punto con normalidad", "g_squint": "ENTRECIERRA los ojos (sin cerrarlos)",
        "g_rest": "relaja", "g_blink": "PARPADEA", "g_smile": "SONRÍE"}


class GestureStage(Stage):
    title = "Gesto de clic: entrecerrar los ojos (≈25 s)"
    intro_text = ("Sigue las instrucciones del centro de la pantalla.\n"
                  "Entrecerrar = achinar los ojos como para ver algo lejano, SIN llegar a cerrarlos.\n\n"
                  "ESPACIO para empezar · ESC para salir")

    def __init__(self, W, H):
        c = (lambda t, x=W / 2, y=H / 2: (x, y))
        segs = [Segment("g_normal", 6.0, c, 1.0)]
        for _ in range(5):
            segs += [Segment("g_squint", 1.3, c, 0.0), Segment("g_rest", 1.3, c, 0.0)]
        for _ in range(4):
            segs += [Segment("g_blink", 0.9, c, 0.0), Segment("g_rest", 0.9, c, 0.0)]
        for _ in range(2):
            segs += [Segment("g_smile", 1.6, c, 0.0), Segment("g_rest", 1.4, c, 0.0)]
        super().__init__(segs)

    def _draw(self, seg: Segment, lt: float) -> None:
        cv = self.canvas
        cv.delete("target")
        x, y = seg.target(lt)
        cv.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#ff3b30", outline="", tags="target")
        hot = seg.phase in ("g_squint", "g_blink", "g_smile")
        cv.create_text(x, y - 70, text=TEXT[seg.phase], fill="#e8a33d" if hot else "#9a9a9a",
                       font=("Segoe UI", 26 if hot else 16), tags="target")


def analyse(rec) -> dict | None:
    from .features import BLENDSHAPE_NAMES

    if not len(rec):
        return None
    ix = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}
    B = np.array(rec.cols["blendshapes"], float)
    ph = np.array(rec.cols["phase"])
    lt = np.array(rec.cols["local_t"])
    sq = B[:, [ix["eyeSquintLeft"], ix["eyeSquintRight"]]].mean(1)
    bl = B[:, [ix["eyeBlinkLeft"], ix["eyeBlinkRight"]]].min(1)
    sm = B[:, [ix["mouthSmileLeft"], ix["mouthSmileRight"]]].mean(1)
    base = float(np.median(sq[ph == "g_normal"]))
    score = sq - base
    gated = (bl < BLINK_MAX) & (sm < SMILE_MAX)          # what the detector would even consider
    pos = score[(ph == "g_squint") & (lt >= 0.4) & gated]
    neg = score[np.isin(ph, ("g_normal", "g_rest", "g_blink", "g_smile")) & gated]
    if len(pos) < 10 or len(neg) < 30:
        return None
    hi_neg, lo_pos = float(np.percentile(neg, 99)), float(np.percentile(pos, 25))
    separable = lo_pos > hi_neg
    thr = (hi_neg + lo_pos) / 2 if separable else hi_neg + 0.03
    caught = float(np.mean(pos > thr))
    false = float(np.mean(neg > thr))
    return {"thr": round(thr, 3), "hold_s": 0.3, "baseline": round(base, 3), "separable": bool(separable),
            "squint_p25": round(lo_pos, 3), "other_p99": round(hi_neg, 3),
            "squint_frames_caught": round(caught, 3), "other_frames_over": round(false, 4),
            "blinks_gated": float(np.mean(bl[ph == "g_blink"] >= BLINK_MAX)),
            "smiles_gated": float(np.mean(sm[ph == "g_smile"] >= SMILE_MAX))}


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
        print(f"bien: tu entrecerrar se distingue (umbral {rep['thr']}); "
              f"{100 * rep['squint_frames_caught']:.0f}% de los fotogramas de entrecerrar lo superan, "
              f"{100 * rep['other_frames_over']:.2f}% de los demás")
    else:
        print("ojo: tu entrecerrar se solapa con gestos normales; prueba a entrecerrar más marcado y repite")
