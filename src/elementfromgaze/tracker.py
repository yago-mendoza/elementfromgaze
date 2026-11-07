"""Background worker (camera -> features) and the per-user gaze regressor."""

from __future__ import annotations

import os
import pickle
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .camera import Camera
from .features import FeatureExtractor, Sample

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA = ROOT / "data"
SESSIONS = DATA / "sessions"
LANDMARKER = MODELS / "face_landmarker.task"
# One model per camera resolution: the 4:3 640x480 and 16:9 1080p images have different fields of
# view and aspect, so eye features in normalised image coordinates aren't comparable across them.
RES = os.environ.get("EFG_RES", "640x480").lower()
USER_MODEL = DATA / ("user_model.pkl" if RES == "640x480" else f"user_model_{RES}.pkl")


class FeatureWorker:
    """Runs camera + landmarker on a thread and hands every Sample to `on_sample`."""

    def __init__(self, on_sample: Callable[[Sample], None], camera_index: int = 0):
        self.on_sample = on_sample
        res = os.environ.get("EFG_RES", "640x480")
        w, h = (int(v) for v in res.lower().split("x"))
        self.camera = Camera(camera_index, w, h)
        self.extractor = FeatureExtractor(LANDMARKER)
        self.proc_ms: deque[float] = deque(maxlen=60)
        self.frame_ts: deque[float] = deque(maxlen=60)
        self.face_seen = False
        self.frames = 0
        self.frames_face = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        seq = 0
        while self._running:
            seq, frame = self.camera.next(seq)
            if frame is None:
                continue
            t0 = time.perf_counter()
            sample = self.extractor(frame)
            self.proc_ms.append((time.perf_counter() - t0) * 1000)
            self.frame_ts.append(time.monotonic())
            self.face_seen = sample is not None
            self.frames += 1
            self.frames_face += sample is not None
            if sample is not None:
                self.on_sample(sample)

    def stats(self) -> str:
        fps = 0.0
        if len(self.frame_ts) > 1:
            fps = (len(self.frame_ts) - 1) / max(self.frame_ts[-1] - self.frame_ts[0], 1e-6)
        ms = float(np.mean(self.proc_ms)) if self.proc_ms else 0.0
        cover = self.frames_face / self.frames if self.frames else 0.0
        return f"{fps:4.1f} fps, {ms:4.1f} ms/frame, cara {cover:.0%}"

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        self.camera.close()
        self.extractor.close()


# Feature groups (see features.py for the layout).
EYES = list(range(0, 6))          # iris u/v + lid opening, both eyes
BLENDSHAPES = list(range(7, 15))  # eyeLook* scores
POSE = list(range(15, 24))        # head rotation + translation
# With few sessions, head pose correlates with time instead of gaze and the model
# extrapolates wildly when the head moves (11° vs 1.2-2.7° on the first eval).
# Pose comes back once multi-pose / VOR data makes it independent of the target.
DEFAULT_COLS = EYES + BLENDSHAPES


class LinPlusKRR:
    """Ridge for the global shape + RBF kernel ridge on its residual.

    A bare RBF kernel ridge has no intercept: where data is thin (screen edges, the bottom
    half that early sessions lost) it shrinks predictions toward (0, 0) = top-left corner.
    Here, away from the data the kernel term fades to 0 and the linear map takes over,
    which extrapolates sensibly to the edges.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 0.05):
        self.alpha, self.gamma = alpha, gamma

    def fit(self, X, Y):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.lin = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(Z, Y)
        self.krr = KernelRidge(kernel="rbf", alpha=self.alpha, gamma=self.gamma).fit(Z, Y - self.lin.predict(Z))
        return self

    def predict(self, X):
        Z = self.sc.transform(X)
        return self.lin.predict(Z) + self.krr.predict(Z)


class Ensemble:
    """Average of members, each on its own feature columns: pose-blind linkrr (robust in the
    usual posture) + pose-aware mlp and linkrr (robust when the head/laptop moves)."""

    MEMBERS = (("eyes_bs", "linkrr"), ("all", "mlp"), ("all", "linkrr"))

    def fit(self, X, Y):
        self.models = []
        for cols, kind in self.MEMBERS:
            c = EYES + BLENDSHAPES if cols == "eyes_bs" else list(range(X.shape[1]))
            self.models.append((c, make_regressor(kind).fit(X[:, c], Y)))
        return self

    def predict(self, X):
        return np.mean([m.predict(X[:, c]) for c, m in self.models], axis=0)


def make_regressor(kind: str = "linkrr"):
    if kind == "ens":
        return Ensemble()
    if kind == "geo":
        from .geometry import GeoModel
        return GeoModel()
    if kind == "geores":
        from .geometry import GeoPlusResidual
        return GeoPlusResidual()
    if kind == "linkrr":
        return LinPlusKRR()
    if kind == "mlp":
        # Small MLP (24 -> 64 -> 64 -> 2, ~6k params, <0.1 ms/frame). Unlike polynomial features
        # it can use head pose without exploding off-distribution, given varied-pose data.
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.neural_network import MLPRegressor

        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-2, learning_rate_init=1e-3,
                           max_iter=600, early_stopping=True, n_iter_no_change=25, random_state=0)
        return make_pipeline(StandardScaler(), TransformedTargetRegressor(mlp, transformer=StandardScaler()))
    if kind == "lin":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 30)))
    if kind == "poly":
        return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                             RidgeCV(alphas=np.logspace(-2, 5, 30)))
    return make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", alpha=1.0, gamma=0.05))


class GazeModel:
    """features -> screen pixel, fitted per user from calibration sessions."""

    def __init__(self, cols: list[int] | None = None, kind: str = "ens") -> None:
        # the ensemble picks its own columns per member, so it gets all of them
        self.cols = list(cols or (list(range(24)) if kind in ("ens", "geo", "geores") else DEFAULT_COLS))
        self.kind = kind
        self.pipe = make_regressor(kind)
        self.screen: tuple[int, int] | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, screen: tuple[int, int]) -> "GazeModel":
        self.screen = screen
        if len(X) > 4000:  # kernel ridge is O(n^2) in memory; subsample evenly
            idx = np.linspace(0, len(X) - 1, 4000).astype(int)
            X, Y = X[idx], Y[idx]
        self.pipe.fit(X[:, self.cols], Y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.pipe.predict(np.atleast_2d(x)[:, self.cols])

    # Appearance network (eye images). Optional: the feature model above is the fallback.
    def fit_appearance(self, P: np.ndarray, X: np.ndarray, Y: np.ndarray) -> "GazeModel":
        from .appearance import AppearanceModel
        self.appearance = AppearanceModel().fit(P, X, Y)
        return self

    def predict_full(self, x: np.ndarray, patches: np.ndarray | None) -> np.ndarray:
        app = getattr(self, "appearance", None)
        if app is None or patches is None:
            return self.predict(x)
        return app.predict(patches, x)

    def save(self, path: Path = USER_MODEL) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path = USER_MODEL) -> "GazeModel":
        return pickle.loads(path.read_bytes())


def load_user_model(warm: bool = True) -> GazeModel:
    """The user's model; for someone new, the strangers' base (generic.py), which needs a warm-up."""
    if USER_MODEL.exists():
        m = GazeModel.load()
    else:
        from .generic import new_user_model
        m = new_user_model()
        if m is None:
            raise SystemExit("no hay modelo todavía: corre `efg calibrate` primero")
        m.save()
        print("primera vez: empiezo con una red entrenada con otras personas; el calentamiento la ajusta a ti")
    if getattr(m, "generic", False) and not warm:
        raise SystemExit("la primera vez hace falta el calentamiento (quita --no-warmup)")
    return m


TRAIN_PHASES = ("fix", "vor", "pursuit", "eval", "click", "caret", "follow")


def session_paths() -> list[Path]:
    """Legacy sessions are single .npz files; protocol v2 sessions are folders with samples.npz."""
    legacy = list(SESSIONS.glob("*.npz"))
    folders = [p / "samples.npz" for p in SESSIONS.iterdir() if (p / "samples.npz").exists()] if SESSIONS.exists() else []
    return sorted(legacy + folders, key=lambda p: p.parent.name if p.name == "samples.npz" else p.stem)


def session_name(p: Path) -> str:
    return p.parent.name if p.name == "samples.npz" else p.stem


def session_res(p: Path) -> str:
    """'WxH' of the colour frames a session was recorded at (sessions before this field: 640x480)."""
    meta = (p.parent if p.name == "samples.npz" else p.with_suffix("")) / "meta.json"
    try:
        import json
        fs = json.loads(meta.read_text(encoding="utf-8"))["auto"].get("frame_size")
        return f"{fs[0]}x{fs[1]}" if fs else "640x480"
    except (OSError, KeyError, ValueError):
        return "640x480"


CACHE = DATA / "cache"


def session_patches(p: Path, d) -> np.ndarray | None:
    """Eye patches for every sample: stored in newer sessions, else rebuilt from frames (and cached)."""
    if "patches" in d:
        return d["patches"]
    if p.name != "samples.npz":
        return None
    cache = CACHE / f"{p.parent.name}_patches.npz"
    if cache.exists():
        return np.load(cache)["P"]
    from .appearance import patches_from_frames
    P = patches_from_frames(p.parent, d["landmarks"])
    if P is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, P=P)
    return P


def load_sessions(phases: tuple[str, ...] = TRAIN_PHASES, only_valid: bool = True, res: str | None = RES,
                  with_patches: bool = False) -> dict:
    paths = [p for p in session_paths() if res is None or session_res(p) == res]
    if not paths:
        raise SystemExit("no calibration sessions yet: run `efg calibrate` first")
    parts: dict[str, list[np.ndarray]] = {"X": [], "Y": [], "session": [], "phase": [], "seg": [], "P": [], "has_P": []}
    for i, p in enumerate(paths):
        d = np.load(p)
        keep = np.isin(d["phase"], phases)
        if only_valid and "local_t" in d:
            keep &= recompute_valid(d)
        parts["X"].append(d["features"][keep])
        parts["Y"].append(d["targets"][keep])
        parts["session"].append(np.full(int(keep.sum()), i))
        parts["phase"].append(d["phase"][keep])
        seg = d["seg"][keep] if "seg" in d else _legacy_segments(d["targets"])[keep]
        parts["seg"].append(seg)
        if with_patches:
            from .appearance import PATCH_H, PATCH_W
            P = session_patches(p, d)
            n = int(keep.sum())
            parts["P"].append(P[keep] if P is not None else np.zeros((n, 2, PATCH_H, PATCH_W), np.uint8))
            parts["has_P"].append(np.full(n, P is not None))
    if not with_patches:
        parts.pop("P")
        parts.pop("has_P")
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["names"] = [session_name(p) for p in paths]
    return out


SETTLE_S = {"fix": 0.7, "vor": 0.8, "eval": 0.8, "pursuit": 0.0}
FOLLOW_SLOW_PX_S = 350.0


def recompute_valid(d) -> np.ndarray:
    """Validity from raw columns, so filtering rules can change after recording."""
    from .filters import blink_mask

    settle = np.array([SETTLE_S.get(str(ph), 0.0) for ph in d["phase"]])
    from .features import eyes_in_frame

    opening = (d["features"][:, 2] + d["features"][:, 5]) / 2
    names = list(d["blendshape_names"])
    bs = d["blendshapes"]
    eye_blink = np.minimum(bs[:, names.index("eyeBlinkLeft")], bs[:, names.index("eyeBlinkRight")])
    quality = eyes_in_frame(d["landmarks"]) & (d["crops"].reshape(len(d["crops"]), 2, -1).max(-1) > 0).all(1)
    valid = (d["local_t"] >= settle) & ~blink_mask(opening, eye_blink) & quality
    # implicit labels keep the online verdict: a rejected click means you looked elsewhere
    implicit = np.isin(d["phase"], ("click", "caret"))
    valid &= ~implicit | d["valid"]
    if "x_cursor_speed" in d:   # follow-the-mouse: the eye leads a moving cursor; trust slow moments
        valid &= (d["phase"] != "follow") | (d["x_cursor_speed"] < FOLLOW_SLOW_PX_S)
    return valid


def _legacy_segments(targets: np.ndarray) -> np.ndarray:
    """Old files have no segment index: a new segment starts whenever the target jumps."""
    jumps = np.r_[0, (np.linalg.norm(np.diff(targets, axis=0), axis=1) > 40).astype(int)]
    return np.cumsum(jumps)
