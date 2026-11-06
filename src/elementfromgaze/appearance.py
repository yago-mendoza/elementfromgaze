"""Appearance model: a small CNN reads eye-in-head rotation from the eye images; head pose enters
only through the gaze-ray geometry (geometry.py), never as a learned input.

Why: MediaPipe's iris landmarks and eyeLook* blendshapes are noisy, pose-contaminated summaries of
the eye. The pixels hold more. Leave-one-session-out on the user's data: 3.5-7.8° -> 1.7-4.1° on
640x480, 2.9° -> 1.9° on 1080p, with the dot moving about half as much when the head turns.

Eye patches are normalised before the network sees them: a similarity warp puts both eye corners at
fixed positions (removes head roll and distance), and the left eye is mirrored so one tower serves
both eyes. ~110k parameters, ~1 ms per frame on CPU.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

PATCH_H, PATCH_W = 44, 72        # stored size; training random-crops 36x60, inference takes the centre
IN_H, IN_W = 36, 60
CORNER_DIST = 36.0               # px between the eye corners in the patch
R_OUT, R_IN, L_IN, L_OUT = 33, 133, 362, 263


def _warp(gray: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dst = np.float32([[PATCH_W / 2 - CORNER_DIST / 2, PATCH_H / 2], [PATCH_W / 2 + CORNER_DIST / 2, PATCH_H / 2]])
    M, _ = cv2.estimateAffinePartial2D(np.float32([a, b]), dst)
    if M is None:
        return np.zeros((PATCH_H, PATCH_W), np.uint8)
    return cv2.warpAffine(gray, M, (PATCH_W, PATCH_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def eye_patches(gray: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """(H, W) gray frame + (478, >=2) normalised landmarks -> (2, 44, 72) uint8: right eye, left eye mirrored."""
    h, w = gray.shape
    pts = landmarks[:, :2] * np.array([w, h], np.float32)
    right = _warp(gray, pts[R_OUT], pts[R_IN])
    left = _warp(gray, pts[L_IN], pts[L_OUT])[:, ::-1]
    return np.stack([right, left])


def patches_from_frames(session_dir: Path, landmarks: np.ndarray) -> np.ndarray | None:
    """Recompute patches from a session's stored JPEG frames (sessions recorded before patches were saved)."""
    f = session_dir / "frames.npz"
    if not f.exists():
        return None
    z = np.load(f)
    data, offs = z["data"], z["offsets"]
    if len(offs) - 1 != len(landmarks):
        return None
    out = np.zeros((len(landmarks), 2, PATCH_H, PATCH_W), np.uint8)
    for i in range(len(landmarks)):
        img = cv2.imdecode(data[offs[i]:offs[i + 1]], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out[i] = eye_patches(img, landmarks[i])
    return out


# ---------------------------------------------------------------------------
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _build_net(n_mod: int, geo_p: np.ndarray, pose_in: bool = False, binoc: bool = False):
    torch, nn = _torch()
    from .geometry import P_E, P_M, P_PITCH, P_S, P_YAW

    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            def blk(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
            # adaptive pool: 36x60 input gives 4x7 already (no-op); larger patches pool down to it
            self.c = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 48), nn.AdaptiveAvgPool2d((4, 7)))
            self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(48 * 4 * 7, 64), nn.ReLU())

        def forward(self, x):
            return self.fc(self.c(x))

    class Net(nn.Module):
        """Each modality: a pair of eye patches through its own tower -> eye-in-head yaw/pitch ->
        gaze ray (rotated and placed by the head pose) -> hit point on the screen plane -> px."""

        def __init__(self):
            super().__init__()
            self.towers = nn.ModuleList([Tower() for _ in range(n_mod)])
            # pose_in: the eye-in-head angle head also sees the head pose (rotation + position). The
            # same eye angle LOOKS different from another head pitch / distance (lid occlusion,
            # foreshortening); without this the head has to guess the posture from the patch alone.
            self.pose_in = pose_in
            self.head = nn.Sequential(nn.Linear(128 * n_mod + (12 if pose_in else 0), 64), nn.ReLU(), nn.Linear(64, 2))
            with torch.no_grad():
                if pose_in:
                    self.head[0].weight[:, 128 * n_mod:].mul_(0.1)
                self.head[2].weight.mul_(0.1)
                self.head[2].bias.copy_(torch.tensor([geo_p[P_YAW][0], geo_p[P_PITCH][0]], dtype=torch.float32))
            g0 = torch.tensor(np.r_[geo_p[P_E], geo_p[P_S], geo_p[P_M]], dtype=torch.float32)
            self.register_buffer("g0", g0.clone())
            self.g = nn.Parameter(g0.clone())
            # binoc: each eye its own ray (own angle head + confidence), from its own origin (head
            # origin ± a learned half inter-eye offset), both supervised; fused by confidence. A far,
            # foreshortened or lid-covered eye can be down-weighted; disagreement = uncertainty.
            self.binoc = binoc
            if binoc:
                pin = 64 * n_mod + (12 if pose_in else 0)
                self.eye_heads = nn.ModuleList([nn.Sequential(nn.Linear(pin, 48), nn.ReLU(), nn.Linear(48, 3)) for _ in range(2)])
                with torch.no_grad():
                    for h in self.eye_heads:
                        h[2].weight.mul_(0.1)
                        h[2].bias.copy_(torch.tensor([geo_p[P_YAW][0], geo_p[P_PITCH][0], 0.0], dtype=torch.float32))
                self.half_ipd = nn.Parameter(torch.zeros(3))
            self.last = None

        def _hit(self, yaw, pitch, R, t, off):
            cp = torch.cos(pitch)
            d = torch.stack([torch.sin(yaw) * cp, torch.sin(pitch), torch.cos(yaw) * cp], 1)
            g = torch.einsum("nij,nj->ni", R, d)
            e, s, m = self.g[0:3], self.g[3], self.g[4:10]
            o = s * t + torch.einsum("nij,j->ni", R, e + off)
            lam = -o[:, 2] / g[:, 2].clamp(min=0.15)
            px, py = o[:, 0] + lam * g[:, 0], o[:, 1] + lam * g[:, 1]
            return torch.stack([m[0] * px + m[1] * py + m[2], m[3] * px + m[4] * py + m[5]], 1)

        def forward(self, xs, R, t):
            B = xs[0].shape[0]
            if self.binoc:
                fe = torch.cat([tw(x.reshape(B * 2, 1, *x.shape[2:])).reshape(B, 2, 64) for tw, x in zip(self.towers, xs)], 2)
                pts, conf = [], []
                for k in range(2):
                    fk = fe[:, k]
                    if self.pose_in:
                        fk = torch.cat([fk, R.reshape(B, 9), t / 50.0], 1)
                    a = self.eye_heads[k](fk)
                    pts.append(self._hit(a[:, 0], a[:, 1], R, t, self.half_ipd * (1 if k == 0 else -1)))
                    conf.append(a[:, 2])
                w = torch.softmax(torch.stack(conf, 1), 1)
                fused = w[:, :1] * pts[0] + w[:, 1:] * pts[1]
                self.last = (pts[0], pts[1], w)
                return fused
            f = torch.cat([tw(x.reshape(B * 2, 1, *x.shape[2:])).reshape(B, 128) for tw, x in zip(self.towers, xs)], 1)
            if self.pose_in:
                f = torch.cat([f, R.reshape(B, 9), t / 50.0], 1)
            ang = self.head(f)
            yaw, pitch = ang[:, 0], ang[:, 1]
            cp = torch.cos(pitch)
            d = torch.stack([torch.sin(yaw) * cp, torch.sin(pitch), torch.cos(yaw) * cp], 1)
            g = torch.einsum("nij,nj->ni", R, d)
            e, s, m = self.g[0:3], self.g[3], self.g[4:10]
            o = s * t + torch.einsum("nij,j->ni", R, e)
            lam = -o[:, 2] / g[:, 2].clamp(min=0.15)
            px, py = o[:, 0] + lam * g[:, 0], o[:, 1] + lam * g[:, 1]
            return torch.stack([m[0] * px + m[1] * py + m[2], m[3] * px + m[4] * py + m[5]], 1)

        def prior(self):
            scale = torch.tensor([1, 1, 1, 0.05, 64, 64, 100, 64, 64, 100.0])
            return (((self.g - self.g0) / scale) ** 2).sum() * 0.5

    return Net()


def _norm(P: np.ndarray) -> np.ndarray:
    P = P.astype(np.float32)
    return (P - P.mean(axis=(-2, -1), keepdims=True)) / (P.std(axis=(-2, -1), keepdims=True) + 1e-3)


def _crop(P: np.ndarray, train: bool, rng: np.random.Generator | None = None) -> np.ndarray:
    oy, ox = (PATCH_H - IN_H) // 2, (PATCH_W - IN_W) // 2
    if not train:
        return np.ascontiguousarray(P[..., oy:oy + IN_H, ox:ox + IN_W])
    B = P.shape[0]
    dy = rng.integers(0, PATCH_H - IN_H + 1, (B, 2))
    dx = rng.integers(0, PATCH_W - IN_W + 1, (B, 2))
    out = np.empty((B, 2, IN_H, IN_W), np.float32)
    for i in range(B):
        for k in range(2):
            out[i, k] = P[i, k, dy[i, k]:dy[i, k] + IN_H, dx[i, k]:dx[i, k] + IN_W]
    out *= rng.uniform(0.85, 1.15, (B, 2, 1, 1)).astype(np.float32)      # contrast jitter
    out += rng.normal(0, 0.08, out.shape).astype(np.float32)             # sensor noise
    return out


def _mods(P) -> list[np.ndarray]:
    """One patch array per camera: a bare array is the colour camera alone."""
    return list(P) if isinstance(P, (list, tuple)) else [P]


def _pose(X: np.ndarray):
    c0, c1 = X[:, 15:18], X[:, 18:21]
    return np.stack([c0, c1, np.cross(c0, c1)], -1).astype(np.float32), (X[:, 21:24] * 50).astype(np.float32)


_LIVE_THREADS_SET = False


class AppearanceModel:
    """eye patches (+ head pose from the feature vector) -> screen px."""

    def __init__(self, epochs: int = 20, batch: int = 128, lr: float = 2e-3, seed: int = 0, pose_in: bool = False,
                 binoc: bool = False):
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed
        self._pose_in, self._binoc = pose_in, binoc

    def fit(self, P, X: np.ndarray, Y: np.ndarray, log=print) -> "AppearanceModel":
        """P: (N, 2, 44, 72) patches, or a list of them (one per camera, e.g. [colour, IR])."""
        torch, _ = _torch()
        import torch.nn.functional as F

        from .geometry import GeoModel

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Pn = [_norm(p) for p in _mods(P)]
        self._geo_p = GeoModel().fit(X, Y).p
        self._n_mod = len(Pn)
        self.net = _build_net(self._n_mod, self._geo_p, getattr(self, "_pose_in", False), getattr(self, "_binoc", False))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        steps_per = int(np.ceil(len(X) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs * steps_per, pct_start=0.15)
        R, t = (torch.from_numpy(a) for a in _pose(X))
        Yt = torch.from_numpy(Y.astype(np.float32))
        t0 = time.monotonic()
        for ep in range(self.epochs):
            self.net.train()
            perm = rng.permutation(len(X))
            tot = 0.0
            for b in range(0, len(X), self.batch):
                ix = perm[b:b + self.batch]
                if len(ix) < 8:
                    continue
                xs = [torch.from_numpy(_crop(p[ix], True, rng)) for p in Pn]
                loss = F.smooth_l1_loss(self.net(xs, R[ix], t[ix]) / 100, Yt[ix] / 100) + self.net.prior() / len(X)
                if self.net.binoc:            # each eye must hit the target on its own too
                    p0, p1, _ = self.net.last
                    loss = loss + 0.25 * (F.smooth_l1_loss(p0 / 100, Yt[ix] / 100) + F.smooth_l1_loss(p1 / 100, Yt[ix] / 100))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item() * len(ix)
            if log and (ep % 5 == 4 or ep == self.epochs - 1):
                log(f"   red de ojos: época {ep + 1}/{self.epochs} · pérdida {tot / len(X):.3f} · {time.monotonic() - t0:.0f}s")
        self.net.eval()
        return self

    # the network class is built inside a function (torch is imported lazily), so pickle its weights
    def __getstate__(self):
        st = self.__dict__.copy()
        if "net" in st:
            st["net"] = {k: v.cpu().numpy() for k, v in self.net.state_dict().items()}
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        if isinstance(st.get("net"), dict):
            torch, _ = _torch()
            self.net = _build_net(st["_n_mod"], st["_geo_p"], st.get("_pose_in", False), st.get("_binoc", False))
            self.net.load_state_dict({k: torch.from_numpy(v) for k, v in st["net"].items()})
            self.net.eval()

    # Test-time augmentation: the centre crop plus 4 shifted ones, averaged. The shifts are the same
    # jitter the network was trained with, so each sees a slightly different eye position and their
    # errors partly cancel. Measured after warm-up on 3 tracks: 1.45° -> 1.35° on pursuit.
    TTA_SHIFTS = ((0, 0), (-2, 0), (2, 0), (0, -3), (0, 3))

    def predict(self, P, X: np.ndarray, tta: bool = True) -> np.ndarray:
        torch, _ = _torch()
        global _LIVE_THREADS_SET
        if len(np.atleast_2d(X)) == 1 and not _LIVE_THREADS_SET:   # live use: don't fight MediaPipe for cores
            torch.set_num_threads(1)
            _LIVE_THREADS_SET = True
        X = np.atleast_2d(X)
        Ps = [np.asarray(p)[None] if np.asarray(p).ndim == 3 else np.asarray(p) for p in _mods(P)]
        R, t = _pose(X)
        out = []
        shifts = self.TTA_SHIFTS if tta else ((0, 0),)
        oy, ox = (PATCH_H - IN_H) // 2, (PATCH_W - IN_W) // 2
        k = len(shifts)
        with torch.inference_mode():
            for b in range(0, len(X), 512 // k):
                e = min(b + 512 // k, len(X))
                xs = []
                for p in Ps:
                    pn = _norm(p[b:e])
                    xs.append(torch.from_numpy(np.concatenate(
                        [np.ascontiguousarray(pn[..., oy + dy:oy + dy + IN_H, ox + dx:ox + dx + IN_W]) for dy, dx in shifts])))
                Rb = torch.from_numpy(np.tile(R[b:e], (k, 1, 1)))
                tb = torch.from_numpy(np.tile(t[b:e], (k, 1)))
                out.append(self.net(xs, Rb, tb).numpy().reshape(k, e - b, 2).mean(0))
        return np.concatenate(out)


def finetune_copy(model: AppearanceModel, P, X: np.ndarray, Y: np.ndarray,
                  steps: int = 40, batch: int = 64, lr: float = 5e-4, seed: int = 0,
                  freeze_conv: bool = True) -> AppearanceModel:
    """A fine-tuned COPY of `model` (the live one keeps predicting meanwhile; the caller swaps it in).

    Only the top of the network moves: the per-eye fully connected layer, the head and the
    geometry. The convolutional layers, which learned what an eye looks like from every session,
    stay frozen, so a few minutes of live data adapt the mapping without forgetting the rest.
    """
    import copy

    torch, _ = _torch()
    import torch.nn.functional as F

    m = copy.deepcopy(model)
    net = m.net
    net.eval()                        # BatchNorm keeps its statistics; no dropout noise in a few steps
    if freeze_conv:
        for tw in net.towers:
            for prm in tw.c.parameters():
                prm.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    Pn = [_norm(p) for p in _mods(P)]
    R, t = (torch.from_numpy(a) for a in _pose(X))
    Yt = torch.from_numpy(Y.astype(np.float32))
    for _ in range(steps):
        ix = rng.integers(0, len(X), min(batch, len(X)))
        xs = [torch.from_numpy(_crop(p[ix], True, rng)) for p in Pn]
        loss = F.smooth_l1_loss(net(xs, R[ix], t[ix]) / 100, Yt[ix] / 100) + net.prior() / max(len(X), 1000)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for prm in net.parameters():
        prm.requires_grad_(True)
    net.eval()
    return m
