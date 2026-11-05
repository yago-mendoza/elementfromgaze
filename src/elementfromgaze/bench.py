"""Offline benchmark: leave-one-session-out, scored per static target (median prediction vs target).

This is the honest number: every session is predicted by a model that never saw it,
so a new condition (distance, light, pose) shows up as the error it really causes.
"""

from __future__ import annotations

import json

import numpy as np

from . import screen
from .tracker import BLENDSHAPES, EYES, POSE, GazeModel, load_sessions

STATIC = ("fix", "vor", "eval")
CONFIGS = {
    "eyes+bs linkrr": (EYES + BLENDSHAPES, "linkrr"),
    "eyes+bs lin": (EYES + BLENDSHAPES, "lin"),
    "eyes+bs mlp": (EYES + BLENDSHAPES, "mlp"),
    "all mlp": (list(range(24)), "mlp"),
    "all linkrr": (list(range(24)), "linkrr"),
    "ensemble": (list(range(24)), "ens"),
}


def per_target_errors(pred: np.ndarray, Y: np.ndarray, seg: np.ndarray) -> np.ndarray:
    out = []
    for s in np.unique(seg):
        m = seg == s
        out.append(np.linalg.norm(np.median(pred[m], axis=0) - np.median(Y[m], axis=0)))
    return np.array(out)


def bench() -> dict:
    screen.make_dpi_aware()
    ppd = screen.px_per_degree()
    d = load_sessions()
    names = d["names"]
    if len(names) < 2:
        raise SystemExit("hacen falta al menos 2 sesiones")
    results: dict = {}
    header = f"{'sesión (no vista)':34s}" + "".join(f"{c:>18s}" for c in CONFIGS)
    print(header + "\n" + "-" * len(header))
    for i, name in enumerate(names):
        test = (d["session"] == i) & np.isin(d["phase"], STATIC)
        train = d["session"] != i
        if test.sum() < 10 or train.sum() < 50:
            continue
        row = {}
        for cname, (cols, kind) in CONFIGS.items():
            m = GazeModel(cols, kind).fit(d["X"][train], d["Y"][train], (0, 0))
            e = per_target_errors(m.predict(d["X"][test]), d["Y"][test], d["seg"][test])
            row[cname] = float(np.median(e) / ppd)
        results[name] = row
        print(f"{name:34s}" + "".join(f"{row[c]:17.2f}°" for c in CONFIGS))
    if results:
        med = {c: float(np.median([r[c] for r in results.values()])) for c in CONFIGS}
        print("-" * len(header) + f"\n{'mediana':34s}" + "".join(f"{med[c]:17.2f}°" for c in CONFIGS))
        results["_median"] = med
    return results


if __name__ == "__main__":
    print(json.dumps(bench(), indent=2))
