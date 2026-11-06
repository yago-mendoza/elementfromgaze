"""Online correction from implicit labels (clicks, text caret).

You almost always look at what you click and at where you type, so every click / keystroke is
a free, noisy label of where the gaze truly is. A single global offset does NOT work (measured:
it fixes the normal posture, 2.6°->1.2° with 3 clicks, but makes 'near'/'far' postures worse),
because the error depends on where you look and on head pose. So the correction is a field:

    c(p, h, t) = g(h, t) + l(p, h, t)

    g = Σ w_i r_i / (λg + Σ w_i)                       pose-local bias        w_i = c_i · K_h(h, h_i) · 2^-(t-t_i)/T
    l = Σ w_i k_i (r_i - g) / (λl + Σ w_i k_i)          screen-local residual  k_i = K_p(p, p_i)

r_i = label - base prediction. λ are pseudo-counts: with no nearby evidence c -> 0 (trust the
base model), with lots of it c -> the weighted mean residual. Labels that still disagree after
correction get down-weighted (Cauchy) or rejected (gate): you were looking at the keyboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def pose_vector(head_euler_deg, eye_width: float) -> np.ndarray:
    """Head pose in units where 1 ~ "different enough to need a different correction"."""
    p, y, r = head_euler_deg
    return np.array([p / 8.0, y / 8.0, r / 10.0, math.log(max(eye_width, 1e-4) / 0.035) / 0.15], np.float32)


@dataclass
class CorrectorConfig:
    sigma_px: float = 380.0       # screen-space reach of one label
    half_life_s: float = 900.0    # labels fade over ~15 min (lighting, posture drift)
    lam_global: float = 1.5
    lam_local: float = 1.0
    gate_px: float = 520.0        # ~8.5°: further than this after correction = not looking there
    robust_px: float = 260.0      # Cauchy scale for down-weighting doubtful labels
    max_labels: int = 3000


class ResidualCorrector:
    def __init__(self, cfg: CorrectorConfig | None = None):
        self.cfg = cfg or CorrectorConfig()
        self.P = np.zeros((0, 2), np.float32)
        self.H = np.zeros((0, 4), np.float32)
        self.R = np.zeros((0, 2), np.float32)
        self.T = np.zeros(0, np.float64)
        self.C = np.zeros(0, np.float32)
        self.accepted = 0
        self.rejected = 0

    def forget(self) -> None:
        """Drop the stored labels (the network has just absorbed them by fine-tuning)."""
        self.P, self.H, self.R = self.P[:0], self.H[:0], self.R[:0]
        self.T, self.C = self.T[:0], self.C[:0]

    def __len__(self) -> int:
        return len(self.T)

    def correction(self, p, h, t: float) -> np.ndarray:
        if not len(self):
            return np.zeros(2, np.float32)
        c = self.cfg
        dh = ((self.H - h) ** 2).sum(1)
        w = self.C * np.exp(-0.5 * dh) * np.exp2(-(t - self.T) / c.half_life_s)
        g = (w[:, None] * self.R).sum(0) / (c.lam_global + w.sum())
        dp = ((self.P - np.asarray(p, np.float32)) ** 2).sum(1)
        wl = w * np.exp(-0.5 * dp / c.sigma_px ** 2)
        l = (wl[:, None] * (self.R - g)).sum(0) / (c.lam_local + wl.sum())
        return (g + l).astype(np.float32)

    def add(self, p, h, label, t: float, confidence: float = 1.0) -> bool:
        p, label = np.asarray(p, np.float32), np.asarray(label, np.float32)
        miss = float(np.linalg.norm(label - (p + self.correction(p, h, t))))
        if miss > self.cfg.gate_px:
            self.rejected += 1
            return False
        conf = confidence / (1.0 + (miss / self.cfg.robust_px) ** 2)
        self.P = np.vstack([self.P, p])[-self.cfg.max_labels:]
        self.H = np.vstack([self.H, h])[-self.cfg.max_labels:]
        self.R = np.vstack([self.R, label - p])[-self.cfg.max_labels:]
        self.T = np.append(self.T, t)[-self.cfg.max_labels:]
        self.C = np.append(self.C, conf)[-self.cfg.max_labels:]
        self.accepted += 1
        return True


class AffineKalman:
    """Global screen deformation tracked by a Kalman filter: corrected = A · p + b.

    The errors measured on a fresh session are mostly one smooth deformation of the whole screen
    (e.g. vertical gain 0.57, horizontal 0.80 when sitting further back than in calibration): a
    linear correction fitted on 11 points took 3.7° -> 1.7° on the 12th. Six numbers, so a handful
    of clicks pins them down, and every point on screen benefits, not only those near a click.

    State θ = [a11, a12, b1, a21, a22, b2] in screen-fraction units, prior = identity. It is a
    random walk (process noise) so old labels fade and a posture change is followed within a few
    labels: the integral term of a PID, with the gain chosen by the filter instead of by hand.
    Labels whose innovation is implausible (Mahalanobis > gate) are rejected: you clicked while
    looking elsewhere.
    """

    # Gains are DIAGONAL and tightly bounded. Real labels come in clusters (a few clicks in one
    # area, a line of text): with a full free affine, 4 clustered clicks squashed the whole screen
    # onto a diagonal (measured live). Cross terms are therefore pinned to ~0, and the per-axis gains
    # get a prior sd of 0.1: on the `targets` session (vertical gain 0.85 after warm-up) clicks took
    # the per-trial error 1.67° -> 1.05° shift-only -> 0.97° with these gains, while 6 clicks
    # clustered in a 150 px blob on a perfect model cost no more than shift-only (1.20° vs 1.23°
    # worst case); sd 0.2 was 0.92° but 1.70° in that stress test. In real use (23 clicks, a posture
    # change, heavy-tailed labels: clicks while looking elsewhere, up to 8° off) sd 0.1 HURT
    # (2.67° vs 2.35° shift-only), 0.05 was neutral (2.37°): 0.05 it is, and the absolute gate went
    # 650 -> 400 px (6.5°) so those clicks are rejected.
    def __init__(self, W: int, H: int, label_sd_px: float = 120.0, prior_gain_sd: float = 0.05,
                 prior_shift_sd: float = 0.2, forget_s: float = 600.0, gate: float = 5.0,
                 abs_gate_px: float = 400.0):
        self.W, self.H = float(W), float(H)
        self.theta = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.P0 = np.diag([prior_gain_sd ** 2, 1e-12, prior_shift_sd ** 2, 1e-12, prior_gain_sd ** 2, prior_shift_sd ** 2])
        self.P = self.P0.copy()
        self.R0 = np.diag([(label_sd_px / W) ** 2, (label_sd_px / H) ** 2])
        self.forget_s, self.gate, self.abs_gate_px = forget_s, gate, abs_gate_px
        self.t = None
        self.accepted = self.rejected = 0

    def _norm(self, p):
        return np.array([p[0] / self.W - 0.5, p[1] / self.H - 0.5])

    def _H(self, q):
        return np.array([[q[0], q[1], 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, q[0], q[1], 1.0]])

    def _predict_to(self, t: float) -> None:
        if self.t is not None and t > self.t:
            # random walk scaled so that, without labels, uncertainty returns to the prior in ~forget_s
            self.P = self.P + self.P0 * ((t - self.t) / self.forget_s)
        self.t = t if self.t is None else max(self.t, t)

    def apply(self, p) -> np.ndarray:
        q = self._norm(p)
        z = self._H(q) @ self.theta
        return np.array([(z[0] + 0.5) * self.W, (z[1] + 0.5) * self.H])

    def add(self, p, label, t: float, confidence: float = 1.0) -> bool:
        self._predict_to(t)
        q, y = self._norm(p), self._norm(label)
        Hm = self._H(q)
        innov = y - Hm @ self.theta
        S = Hm @ self.P @ Hm.T + self.R0 / max(confidence, 1e-3)
        m2 = float(innov @ np.linalg.solve(S, innov))
        innov_px = np.hypot(innov[0] * self.W, innov[1] * self.H)
        if m2 > self.gate ** 2 or innov_px > self.abs_gate_px:
            self.rejected += 1
            return False
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.theta = self.theta + K @ innov
        self.P = (np.eye(6) - K @ Hm) @ self.P
        self.accepted += 1
        return True

    def forget(self) -> None:
        """Back to identity (the network has just absorbed these labels by fine-tuning)."""
        self.theta = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.P = self.P0.copy()

    def state(self) -> dict:
        return {"theta": self.theta.copy(), "P": self.P.copy()}

    @classmethod
    def seeded(cls, W: int, H: int, model=None) -> "AffineKalman":
        """A fresh filter, started from the correction the warm-up's button check measured
        (model.start_aff) when there is one."""
        k = cls(W, H)
        st = getattr(model, "start_aff", None)
        if st is not None:
            k.theta, k.P = np.array(st["theta"], float), np.array(st["P"], float)
        return k

    def describe(self) -> str:
        a11, a12, b1, a21, a22, b2 = self.theta
        return f"ganancia x{a11:.2f} y{a22:.2f} · desplazamiento ({b1 * self.W:+.0f}, {b2 * self.H:+.0f}) px"
