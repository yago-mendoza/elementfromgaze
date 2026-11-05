"""Face -> gaze feature vector, using MediaPipe's Face Landmarker (tasks API).

Per frame we keep three kinds of signal:
  * iris position inside each eye, in an eye-local frame (corner-to-corner axis)
  * eyelid opening (the upper lid follows vertical gaze, which helps a lot)
  * head pose (rotation + translation from the facial transformation matrix)
plus MediaPipe's own eyeLook* blendshapes, and two small grayscale eye crops
kept for training an appearance model later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# MediaPipe face mesh indices. "R" is the subject's right eye (image left).
R_CORNERS, L_CORNERS = (33, 133), (362, 263)
R_LIDS, L_LIDS = (159, 145), (386, 374)
R_IRIS, L_IRIS = 468, 473
R_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
L_EYE_RING = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

GAZE_BLENDSHAPES = [
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
]

CROP_H, CROP_W = 24, 48


@dataclass
class Sample:
    t: float
    features: np.ndarray        # (N_FEATURES,)
    crops: np.ndarray           # (2, CROP_H, CROP_W) uint8
    blink: float                # max(eyeBlinkLeft, eyeBlinkRight)
    # raw signal, kept so features/models can be re-derived later without recollecting
    landmarks: np.ndarray | None = None    # (478, 3) normalised
    blendshapes: np.ndarray | None = None  # (52,) in BLENDSHAPE_NAMES order
    matrix: np.ndarray | None = None       # (4, 4) facial transformation
    frame: np.ndarray | None = None        # BGR frame it came from
    eyes_ok: bool = True                   # both eyes fully inside the frame, crops non-empty
    eye_blink: float = 0.0                 # min(eyeBlinkLeft, eyeBlinkRight): both eyes must close
    patches: np.ndarray | None = None      # (2, 44, 72) normalised eye patches for the appearance model


EYE_MARGIN = 0.02


def eyes_in_frame(landmarks: np.ndarray) -> np.ndarray:
    """(N, 478, 3) or (478, 3) -> bool: every eye-contour landmark inside the image with a margin.
    When an eye leaves the frame MediaPipe still returns landmarks for it — invented ones."""
    pts = landmarks[..., R_EYE_RING + L_EYE_RING, :2]
    return ((pts > EYE_MARGIN) & (pts < 1 - EYE_MARGIN)).all(axis=(-1, -2))


BLENDSHAPE_NAMES: list[str] = []


def ensure_model(path: Path) -> Path:
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def head_euler(M: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from MediaPipe's facial transformation matrix."""
    import math

    R = M[:3, :3]
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.asin(float(np.clip(-R[2, 0], -1, 1))))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


def _eye_local(pts: np.ndarray, corners, lids, iris) -> tuple[float, float, float, float]:
    a, b = pts[corners[0], :2], pts[corners[1], :2]
    axis = b - a
    width = float(np.linalg.norm(axis)) + 1e-9
    ex = axis / width
    ey = np.array([-ex[1], ex[0]])
    rel = pts[iris, :2] - (a + b) / 2
    u, v = float(rel @ ex) / width, float(rel @ ey) / width
    opening = float(np.linalg.norm(pts[lids[0], :2] - pts[lids[1], :2])) / width
    return u, v, opening, width


def _crop(gray: np.ndarray, pts_px: np.ndarray, ring: list[int]) -> np.ndarray:
    p = pts_px[ring]
    cx, cy = p.mean(axis=0)
    w = (p[:, 0].max() - p[:, 0].min()) * 1.5
    h = w * CROP_H / CROP_W
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1 = int(cx + w / 2), int(cy + h / 2)
    H, W = gray.shape
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 - x0 < 4:
        return np.zeros((CROP_H, CROP_W), np.uint8)
    return cv2.equalizeHist(cv2.resize(gray[y0:y1, x0:x1], (CROP_W, CROP_H), interpolation=cv2.INTER_AREA))


class FeatureExtractor:
    def __init__(self, model_path: Path):
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._t0 = time.monotonic()
        self._last_ts = -1

    def __call__(self, frame_bgr: np.ndarray) -> Sample | None:
        t = time.monotonic()
        ts = max(int((t - self._t0) * 1000), self._last_ts + 1)
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.face_landmarks:
            return None

        pts = np.array([(p.x, p.y, p.z) for p in res.face_landmarks[0]], np.float32)
        H, W = frame_bgr.shape[:2]
        # Normalised coords are anisotropic (x/W, y/H); work in pixel-proportional units.
        pts_iso = pts.copy()
        pts_iso[:, 1] *= H / W

        uR, vR, oR, wR = _eye_local(pts_iso, R_CORNERS, R_LIDS, R_IRIS)
        uL, vL, oL, wL = _eye_local(pts_iso, L_CORNERS, L_LIDS, L_IRIS)

        cats = res.face_blendshapes[0]
        if not BLENDSHAPE_NAMES:
            BLENDSHAPE_NAMES.extend(c.category_name for c in cats)
        bs = {c.category_name: c.score for c in cats}
        look = [bs.get(n, 0.0) for n in GAZE_BLENDSHAPES]
        blink = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))

        M = np.asarray(res.facial_transformation_matrixes[0], np.float32)
        rot = M[:3, :2].T.ravel()            # first two columns pin the rotation
        trans = M[:3, 3] / 50.0              # cm-ish -> order 1

        feats = np.array(
            [uR, vR, oR, uL, vL, oL, (wR + wL) / 2, *look, *rot, *trans], np.float32
        )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        pts_px = pts[:, :2] * np.array([W, H], np.float32)
        crops = np.stack([_crop(gray, pts_px, R_EYE_RING), _crop(gray, pts_px, L_EYE_RING)])
        from .appearance import eye_patches
        patches = eye_patches(gray, pts)
        return Sample(
            t=t, features=feats, crops=crops, blink=blink,
            landmarks=pts, blendshapes=np.array([bs.get(n, 0.0) for n in BLENDSHAPE_NAMES], np.float32),
            matrix=M, frame=frame_bgr,
            eyes_ok=bool(eyes_in_frame(pts)) and bool(crops[0].any() and crops[1].any()),
            eye_blink=min(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)),
            patches=patches,
        )

    def close(self) -> None:
        self._lm.close()


N_FEATURES = 7 + len(GAZE_BLENDSHAPES) + 6 + 3
