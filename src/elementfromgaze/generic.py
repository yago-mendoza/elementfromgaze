"""A base that works for ANYONE: eye networks trained on other people, adapted by the warm-up.

Data: MPIIFaceGaze (15 people, laptop webcams, CC BY-NC-SA 4.0 — so the weights built from it are
CC BY-NC-SA 4.0 too). Each image goes through exactly our pipeline (MediaPipe landmarks + pose,
corner-warped eye patches); the label is the TRUE 3D gaze direction in the camera frame, from the
eye midpoint to the 3D gaze target. The network is the same as the personal one; its eye-in-head
angles, rotated by the head pose, are compared with that direction (no screen involved), so
people with different screens and cameras can be mixed.

A new user then: (1) the warm-up fits the screen geometry (physics fit on the warm-up data),
(2) the top layers fine-tune on the warm-up with REPLAY of the strangers (16 rounds).

Measured 2026-11-07 with the author as the new person (the base never saw him), 5 held-out sessions:
  no stranger replay, 8 rounds 2.19° · replay, 8 rounds 1.86° · replay, 16 rounds 1.62-2.13°
  (two seeds; 2-network ensemble 1.98°) — vs 1.34° for his personal network trained on 29 of his
  own sessions. Unfreezing conv blocks in the warm-up: tie (1 block) or worse (all). On 5 unseen
  MPII people (cross-day): 8.3° cold, 4.8° after 150 images (the personal network: 12.6° / 5.8°).
  More and better faces (ETH-XGaze, volunteers on the target laptop) are the next lever.

    python -m elementfromgaze generic prepare <MPIIFaceGaze/Data> # -> data/external/mpii_proc/pXX.npz
    python -m elementfromgaze generic build                       # -> models/generic_base.pkl + replay
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .tracker import DATA

ROOT = Path(__file__).resolve().parents[2]
GENERIC_BASE = ROOT / "models" / "generic_base.pkl"
GENERIC_REPLAY = ROOT / "models" / "generic_replay.npz"
PROC = DATA / "external" / "mpii_proc"
U = 57.2958 / 1.634           # unit vector -> the pixel loss's "100 px" units (1.634° at 61.2 px/°)
EYE_IDX = [33, 133, 362, 263]


def prepare(root: str | Path, out: Path = PROC) -> None:
    """MPIIFaceGaze original images -> our features, patches and the 3D gaze direction (OpenCV axes)."""
    import cv2
    import mediapipe as mp
    import scipy.io as sio
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    from .appearance import eye_patches
    from .features import (GAZE_BLENDSHAPES, L_CORNERS, L_IRIS, L_LIDS, R_CORNERS, R_IRIS, R_LIDS,
                           _eye_local, ensure_model, eyes_in_frame)
    from .tracker import LANDMARKER

    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(LANDMARKER))),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    for pdir in sorted(root.glob("p[0-9][0-9]")):
        dst = out / f"{pdir.name}.npz"
        if dst.exists():
            continue
        cam = sio.loadmat(pdir / "Calibration" / "Camera.mat")
        K, dist = cam["cameraMatrix"].astype(np.float64), cam["distCoeffs"].astype(np.float64)
        rows = [l.split() for l in open(pdir / f"{pdir.name}.txt") if l.strip()]
        P, X, D, OK, DAY = [], [], [], [], []
        for f in rows:
            img = cv2.imread(str(pdir / f[0]))
            if img is None:
                continue
            H, W = img.shape[:2]
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if not res.face_landmarks:
                continue
            pts = np.array([(q.x, q.y, q.z) for q in res.face_landmarks[0]], np.float32)
            iso = pts.copy()
            iso[:, 1] *= H / W
            uR, vR, oR, wR = _eye_local(iso, R_CORNERS, R_LIDS, R_IRIS)
            uL, vL, oL, wL = _eye_local(iso, L_CORNERS, L_LIDS, L_IRIS)
            bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
            M = np.asarray(res.facial_transformation_matrixes[0], np.float32)
            X.append(np.array([uR, vR, oR, uL, vL, oL, (wR + wL) / 2, *[bs.get(n, 0.0) for n in GAZE_BLENDSHAPES],
                               *M[:3, :2].T.ravel(), *(M[:3, 3] / 50.0)], np.float32))
            v = np.array(f[1:27], np.float64)           # last column is "left"/"right"
            fc, gt = v[20:23], v[23:26]                  # face centre, 3D gaze target (camera mm)
            mid = (pts[EYE_IDX, :2] * [W, H]).mean(0)    # eye midpoint, back-projected at the face depth
            n = cv2.undistortPoints(mid.reshape(1, 1, 2), K, dist).reshape(2)
            d = gt - np.array([n[0], n[1], 1.0]) * fc[2]
            D.append(d / np.linalg.norm(d))
            P.append(eye_patches(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), pts))
            OK.append(bool(eyes_in_frame(pts)) and min(bs.get("eyeBlinkLeft", 0), bs.get("eyeBlinkRight", 0)) < 0.5)
            DAY.append(f[0].split("/")[0])
        np.savez_compressed(dst, patches=np.array(P, np.uint8), X=np.array(X, np.float32),
                            dir=np.array(D, np.float32), ok=np.array(OK), day=np.array(DAY))
        print(f"{pdir.name}: {len(P)} de {len(rows)} imágenes", flush=True)


def load(people, proc: Path = PROC):
    P, X, D, G = [], [], [], []
    for i in people:
        z = np.load(proc / f"p{i:02d}.npz")
        k = z["ok"]
        P.append(z["patches"][k]); X.append(z["X"][k]); D.append(z["dir"][k]); G.append(np.full(int(k.sum()), i))
    return [np.concatenate(a) for a in (P, X, D, G)]


def cv_dir(R, ang):
    """eye-in-head yaw/pitch -> gaze direction in OpenCV camera axes (MediaPipe's pose is OpenGL)"""
    import torch
    yaw, pitch = ang[:, 0], ang[:, 1]
    cp = torch.cos(pitch)
    d = torch.stack([torch.sin(yaw) * cp, torch.sin(pitch), torch.cos(yaw) * cp], 1)
    g = torch.einsum("nij,nj->ni", R, d)
    return torch.stack([g[:, 0], -g[:, 1], -g[:, 2]], 1)


def angle_loss(net, x, R, D, bias=None):
    """smooth-L1 between predicted and true unit gaze directions, in the pixel loss's units"""
    import torch.nn.functional as F
    from .appearance import IN_H, IN_W
    B = x.shape[0]
    ang = net.head(net.towers[0](x.reshape(B * 2, 1, IN_H, IN_W)).reshape(B, 128))
    if bias is not None:
        ang = ang + bias
    return F.smooth_l1_loss(cv_dir(R, ang) * U, D * U)


def train(people=range(10), epochs: int = 20, seed: int = 0, lr: float = 2e-3, batch: int = 128, log=print):
    """one eye network on strangers only (per-person angle bias absorbs each one's kappa)"""
    import torch

    from .appearance import AppearanceModel, _build_net, _crop, _norm, _pose
    from .geometry import N_PARAMS

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    P, X, D, G = load(people)
    R, Dt = torch.from_numpy(_pose(X)[0]), torch.from_numpy(D)
    m = AppearanceModel(epochs=epochs, seed=seed)
    m._geo_p, m._n_mod = np.zeros(N_PARAMS), 1          # the screen geometry comes from the warm-up
    m.net = net = _build_net(1, m._geo_p)
    bias = torch.nn.Parameter(torch.zeros(int(G.max()) + 1, 2))
    opt = torch.optim.AdamW(list(net.parameters()) + [bias], lr=lr, weight_decay=1e-4)
    total = epochs * int(np.ceil(len(X) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.15)
    t0 = time.monotonic()
    for s in range(total):
        net.train()
        ix = rng.integers(0, len(X), batch)
        x = torch.from_numpy(_crop(_norm(P[ix]), True, rng))
        loss = angle_loss(net, x, R[ix], Dt[ix], bias[torch.from_numpy(G[ix])])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and s % 1000 == 0:
            log(f"   red genérica (semilla {seed}): paso {s}/{total} · pérdida {loss.item():.3f} · {time.monotonic() - t0:.0f}s")
    net.eval()
    return m


def build(people=range(10), seeds=(0, 1), per_person: int = 400) -> None:
    """models/generic_base.pkl (ensemble of `seeds`) + models/generic_replay.npz (strangers to replay)"""
    import pickle

    from .appearance import AppearanceEnsemble

    members = [train(people, seed=s) for s in seeds]
    base = AppearanceEnsemble(members) if len(members) > 1 else members[0]
    GENERIC_BASE.parent.mkdir(parents=True, exist_ok=True)
    GENERIC_BASE.write_bytes(pickle.dumps(base))
    write_replay(people, per_person)
    print(f"guardado {GENERIC_BASE} y {GENERIC_REPLAY}")


def write_replay(people=range(10), per_person: int = 400) -> None:
    rng = np.random.default_rng(0)
    P, X, D, G = load(people)
    idx = np.concatenate([rng.choice(np.where(G == g)[0], min(per_person, int((G == g).sum())), replace=False)
                          for g in np.unique(G)])
    np.savez_compressed(GENERIC_REPLAY, patches=P[idx], X=X[idx], dir=D[idx])


_REPLAY = None


def replay():
    """(patches, features, directions) of strangers, or None when the file is missing"""
    global _REPLAY
    if _REPLAY is None and GENERIC_REPLAY.exists():
        z = np.load(GENERIC_REPLAY)
        _REPLAY = (z["patches"], z["X"], z["dir"])
    return _REPLAY


def new_user_model():
    """A GazeModel for someone with no model of their own: the strangers' base, marked `generic` so
    the warm-up fits the screen geometry, runs longer and replays strangers."""
    import pickle

    from .tracker import GazeModel

    if not GENERIC_BASE.exists():
        return None
    m = GazeModel(kind="poly")
    m.appearance = pickle.loads(GENERIC_BASE.read_bytes())
    for a in [m.appearance, *getattr(m.appearance, "members", [])]:
        a.generic_replay = True      # every fine-tune of this user's networks also replays strangers
    m.generic = True                 # until the first warm-up has fitted the screen geometry
    return m


def uses_strangers(app) -> bool:
    return app is not None and any(getattr(a, "generic_replay", False) for a in [app, *getattr(app, "members", [])])


def fit_geometry(app, X: np.ndarray, Y: np.ndarray):
    """the warm-up's physics fit of eye position, scale and screen mapping, copied into every network"""
    import copy

    import torch

    from .appearance import AppearanceEnsemble
    from .geometry import P_E, P_M, P_S, GeoModel

    geo = GeoModel().fit(X, Y).p
    members = app.members if isinstance(app, AppearanceEnsemble) else [app]
    out = []
    for mm in members:
        mm = copy.deepcopy(mm)
        mm._geo_p = geo
        g0 = torch.tensor(np.r_[geo[P_E], geo[P_S], geo[P_M]], dtype=torch.float32)
        with torch.no_grad():
            mm.net.g.copy_(g0)
            mm.net.g0.copy_(g0)
        out.append(mm)
    return AppearanceEnsemble(out) if len(out) > 1 else out[0]


def main(argv: list[str]) -> None:
    if argv and argv[0] == "prepare" and len(argv) > 1:
        prepare(argv[1])
    elif argv and argv[0] == "build":
        build()
    else:
        print(__doc__)
