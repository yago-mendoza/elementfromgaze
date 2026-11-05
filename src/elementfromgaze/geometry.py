"""Geometric gaze model: head pose and eye rotation compose physically into a ray that hits the screen.

Why: a black-box regressor sees "iris moved" but can't tell whether the eye rotated or the head
did, so it either ignores pose (the dot moves when the laptop moves, 15-68 px per degree) or
overfits it (11° on a new session). Here pose enters as physics, and only ~40 numbers are learned.

Camera space (MediaPipe facial transformation matrix, OpenGL convention): cm, x = image right,
y = up, z = towards the viewer; the face sits at z ≈ -50 and its canonical +z axis points at the
camera. On a laptop the camera lies in the screen plane, so the screen is the plane z = 0.

    yaw_e, pitch_e = A · f + a0                      eye-in-head angles from iris/lid/blendshape features
    d_head = (sin yaw cos pitch, sin pitch, cos yaw cos pitch)
    g = R · d_head                                    gaze direction in camera space
    o = s · t + R · e                                 eye origin: head translation (scale s) + offset e
    P = o + (-o_z / g_z) · g                          hit point on the screen plane (cm)
    pixel = M · [P_x, P_y, 1]                         cm -> px, initialised from the panel's physics

Fitted per user by robust nonlinear least squares over every calibration sample.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import screen

# Eye-in-head features. Measured on the user's VOR data (fixating while the head turns), MediaPipe's
# eyeLook* blendshapes change as much per degree of head rotation as per degree of gaze (|ratio|
# 1.05-1.06, physics says 1): they are true eye-in-head angles. The iris-vs-eye-corner offsets only
# respond 28-60% (vertical even with the wrong sign): pose-contaminated, so they are left out here.
BS_COLS = list(range(7, 15))
LID_COLS = [2, 5]
EYE_COLS = BS_COLS + LID_COLS
ROT0, ROT1, TRANS = slice(15, 18), slice(18, 21), slice(21, 24)
N_EYE = len(EYE_COLS)
# parameter vector layout
P_YAW = slice(0, N_EYE + 1)
P_PITCH = slice(N_EYE + 1, 2 * (N_EYE + 1))
P_E = slice(2 * (N_EYE + 1), 2 * (N_EYE + 1) + 3)
P_S = 2 * (N_EYE + 1) + 3
P_M = slice(P_S + 1, P_S + 7)
N_PARAMS = P_S + 7


def _pose(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c0, c1 = X[:, ROT0], X[:, ROT1]
    R = np.stack([c0, c1, np.cross(c0, c1)], axis=-1)     # (N, 3, 3), columns = canonical axes
    return R, X[:, TRANS] * 50.0                          # features store trans / 50


class GeoModel:
    def __init__(self, reg: float = 20.0, screen_px: tuple[int, int] | None = None):
        self.reg = reg
        self.screen_px = screen_px

    # -- physics ----------------------------------------------------------
    def _physical_M(self) -> np.ndarray:
        screen.make_dpi_aware()   # otherwise Windows reports the *scaled* size and the physics is off
        W, H = self.screen_px or screen.screen_size_px()
        w_mm, _ = screen.screen_size_mm()
        ppcm = W / (w_mm / 10.0) if w_mm else 64.0
        cam_above_cm = 0.8
        # camera +x (image right) is the user's left = screen left; camera +y is up = screen top
        return np.array([-ppcm, 0.0, W / 2.0, 0.0, -ppcm, -cam_above_cm * ppcm])

    def _forward(self, p: np.ndarray, F: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        yaw = F @ p[P_YAW]
        pitch = F @ p[P_PITCH]
        cp = np.cos(pitch)
        d = np.stack([np.sin(yaw) * cp, np.sin(pitch), np.cos(yaw) * cp], axis=1)
        g = np.einsum("nij,nj->ni", R, d)
        o = p[P_S] * t + np.einsum("nij,j->ni", R, p[P_E])
        gz = np.maximum(g[:, 2], 0.15)
        s = -o[:, 2] / gz
        Px, Py = o[:, 0] + s * g[:, 0], o[:, 1] + s * g[:, 1]
        m = p[P_M]
        return np.stack([m[0] * Px + m[1] * Py + m[2], m[3] * Px + m[4] * Py + m[5]], axis=1)

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (X[:, EYE_COLS] - self.mu) / self.sd
        return np.c_[np.ones(len(X)), Z]

    # -- API --------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GeoModel":
        self.mu = X[:, EYE_COLS].mean(0)
        self.sd = X[:, EYE_COLS].std(0) + 1e-6
        F = self._design(X)
        R, t = _pose(X)
        M0 = self._physical_M()
        p0 = np.zeros(N_PARAMS)
        p0[P_S] = 1.0
        p0[P_M] = M0
        reg_idx = np.r_[np.arange(1, N_EYE + 1), np.arange(N_EYE + 2, 2 * (N_EYE + 1))]
        w_reg = self.reg / np.sqrt(len(X))

        def resid(p):
            r = (self._forward(p, F, R, t) - Y).ravel()
            prior = np.r_[
                w_reg * 100 * p[reg_idx],                    # eye-angle weights: small (radians per sd)
                w_reg * 30 * p[P_E],                         # eye offset: a few cm at most
                w_reg * 300 * (p[P_S] - 1.0),                # translation scale near 1
                w_reg * 2 * (p[P_M] - M0) / np.r_[64, 64, 100, 64, 64, 100],  # px mapping near physics
            ]
            return np.r_[r, prior]

        # Hard physical bounds: without them the optimiser "cheats" (distance scale -> 0, eyes 30 cm
        # away from the head) and turns the physics back into an unconstrained regression.
        lo = np.full(N_PARAMS, -np.inf)
        hi = np.full(N_PARAMS, np.inf)
        lo[P_E], hi[P_E] = [-4.0, -2.0, -6.0], [4.0, 6.0, 6.0]
        lo[P_S], hi[P_S] = 0.7, 1.4
        ppcm, W = -M0[0], 2 * M0[2]
        lo[P_M] = [-1.4 * ppcm, -0.25 * ppcm, W / 2 - 250, -0.25 * ppcm, -1.4 * ppcm, M0[5] - 4 * ppcm]
        hi[P_M] = [-0.7 * ppcm, 0.25 * ppcm, W / 2 + 250, 0.25 * ppcm, -0.7 * ppcm, M0[5] + 3 * ppcm]
        sol = least_squares(resid, p0, bounds=(lo, hi), loss="soft_l1", f_scale=80.0, max_nfev=300, x_scale="jac")
        self.p = sol.x
        self.cost_ = sol.cost
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        R, t = _pose(X)
        return self._forward(self.p, self._design(X), R, t)

    def describe(self) -> str:
        p = self.p
        return (f"offset ojo (cm) {np.round(p[P_E], 2).tolist()} · escala dist {p[P_S]:.2f} · "
                f"px/cm x {-p[P_M][0]:.1f} y {-p[P_M][4]:.1f} · centro {p[P_M][2]:.0f},{p[P_M][5]:.0f}")


class GeoPlusResidual:
    """Geometry for pose and distance; a pose-blind kernel model mops up the eye nonlinearity."""

    def __init__(self, res_alpha: float = 3.0):
        self.res_alpha = res_alpha

    def fit(self, X, Y):
        from .tracker import BLENDSHAPES, EYES, LinPlusKRR

        self.geo = GeoModel().fit(X, Y)
        self.cols = EYES + BLENDSHAPES
        self.res = LinPlusKRR(alpha=self.res_alpha).fit(X[:, self.cols], Y - self.geo.predict(X))
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.geo.predict(X) + self.res.predict(X[:, self.cols])
