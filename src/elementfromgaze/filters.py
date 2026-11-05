"""One Euro filter: smooth when the gaze rests, responsive when it jumps."""

from __future__ import annotations

import math
import numpy as np


class _LowPass:
    def __init__(self) -> None:
        self.y: float | None = None

    def __call__(self, x: float, alpha: float) -> float:
        self.y = x if self.y is None else alpha * x + (1 - alpha) * self.y
        return self.y


class OneEuro:
    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x, self._dx = _LowPass(), _LowPass()
        self._t: float | None = None
        self._prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._t is None:
            self._t, self._prev = t, x
            return self._x(x, 1.0)
        dt = max(t - self._t, 1e-3)
        self._t = t
        dx = (x - self._prev) / dt
        self._prev = x
        edx = self._dx(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(x, self._alpha(cutoff, dt))


BLINK_JUMP = 0.20     # eyeBlink score above its own rolling median
BLINK_RATIO = 0.75    # lid opening below this fraction of its rolling median


def _median(xs) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


class BlinkDetector:
    """A blink needs BOTH kinds of evidence, each relative to your own recent baseline:

      * MediaPipe's eyeBlink score (min of both eyes) jumps > BLINK_JUMP above its median
      * lid opening (geometric) drops below BLINK_RATIO of its median

    Either signal alone fires on head/laptop motion and on looking down (measured: 17-19% of
    calibration frames and 69% of a laptop-motion session flagged; with both, 4% = 7-10
    blinks/min, which is the normal screen-use rate). Flagged frames don't update the baselines;
    `hold` frames after a blink are flagged too while the eye re-settles.
    """

    def __init__(self, window: int = 31, hold: int = 2, max_run: int = 15):
        from collections import deque

        self.op: "deque[float]" = deque(maxlen=window)
        self.bl: "deque[float]" = deque(maxlen=window)
        self.hold, self.max_run = hold, max_run
        self.run = self.hold_left = 0

    def __call__(self, opening: float, eye_blink: float) -> bool:
        if len(self.op) >= 5:
            closing = (eye_blink - _median(self.bl) > BLINK_JUMP) and (opening < BLINK_RATIO * _median(self.op))
        else:
            closing = False
        if closing:
            self.run += 1
            if self.run <= self.max_run:
                self.hold_left = self.hold
                return True
            self.op.clear()                     # stuck: accept it as the new normal
            self.bl.clear()
        self.run = 0
        self.op.append(opening)
        self.bl.append(eye_blink)
        if self.hold_left > 0:
            self.hold_left -= 1
            return True
        return False


def blink_mask(opening: "np.ndarray", eye_blink: "np.ndarray", window: int = 31, dilate: int = 2) -> "np.ndarray":
    """Offline version with *centred* rolling medians, dilated both ways (offline we can also
    drop the half-closed frames just before the detection)."""
    import numpy as np

    h = window // 2
    n = len(opening)
    med_op = np.array([np.median(opening[max(0, i - h): i + h + 1]) for i in range(n)])
    med_bl = np.array([np.median(eye_blink[max(0, i - h): i + h + 1]) for i in range(n)])
    m = (eye_blink - med_bl > BLINK_JUMP) & (opening < BLINK_RATIO * med_op)
    out = m.copy()
    for k in range(1, dilate + 1):
        out[:-k] |= m[k:]
        out[k:] |= m[:-k]
    return out


class FixationFilter:
    """Gaze is fixations joined by saccades, so smooth like that (an I-DT style filter).

    Points within `radius` of the current fixation are averaged into it (robustly, over the
    last `window` frames so slow drift is still followed). A single far point is treated as
    noise; `confirm` consecutive far points that agree with each other mean a saccade, and the
    output jumps there at once instead of sliding.
    """

    def __init__(self, radius: float = 110.0, window: int = 10, confirm: int = 2):
        from collections import deque

        self.radius, self.confirm = radius, confirm
        self.fix: "deque[tuple[float, float]]" = deque(maxlen=window)
        self.cand: list[tuple[float, float]] = []

    @staticmethod
    def _center(pts) -> tuple[float, float]:
        xs, ys = sorted(p[0] for p in pts), sorted(p[1] for p in pts)
        n = len(xs)
        if n >= 5:  # trimmed mean: drop the extremes on each axis
            xs, ys = xs[1:-1], ys[1:-1]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        if not self.fix:
            self.fix.append((x, y))
            return x, y
        cx, cy = self._center(self.fix)
        if math.hypot(x - cx, y - cy) <= self.radius:
            self.fix.append((x, y))
            self.cand.clear()
        else:
            if self.cand and math.hypot(x - self.cand[-1][0], y - self.cand[-1][1]) > self.radius:
                self.cand.clear()
            self.cand.append((x, y))
            if len(self.cand) >= self.confirm:
                self.fix.clear()
                self.fix.extend(self.cand)
                self.cand.clear()
        return self._center(self.fix)

    def spread(self) -> float:
        if len(self.fix) < 3:
            return self.radius
        cx, cy = self._center(self.fix)
        return sorted(math.hypot(p[0] - cx, p[1] - cy) for p in self.fix)[len(self.fix) // 2]


class OneEuro2D:
    def __init__(self, **kw) -> None:
        self.fx, self.fy = OneEuro(**kw), OneEuro(**kw)

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self.fx(x, t), self.fy(y, t)


class GazeKalman:
    """Constant-velocity Kalman filter on screen position, with saccade resets and look-ahead.

    The fixation filter above is steady but slides behind anything that moves: measured on a
    pursuit, the shown dot trailed the target by 0.35 s while the raw network trailed by 0.15 s.
    Here velocity is part of the state, so a smoothly moving gaze is followed without lag while a
    still one is averaged; a saccade (innovation far outside what the filter expects, twice in a
    row) resets the state instead of being smoothed; and the output is extrapolated `lead_s`
    ahead (only by the velocity the filter trusts) to hide the camera + network latency.
    """

    def __init__(self, meas_sd_px: float = 45.0, accel_sd_px_s2: float = 1500.0, lead_s: float = 0.12,
                 saccade_sigma: float = 4.0, confirm: int = 2):
        self.r = meas_sd_px ** 2
        self.q = accel_sd_px_s2 ** 2
        self.lead = lead_s
        self.gate = saccade_sigma
        self.confirm = confirm
        self.x = None          # (2, 2): rows = axis, cols = [position, velocity]
        self.P = None          # (2, 2, 2): per-axis covariance
        self.t = None
        self.far = 0

    def reset(self, z, t) -> None:
        self.x = np.array([[z[0], 0.0], [z[1], 0.0]])
        self.P = np.tile(np.diag([self.r, 400.0 ** 2]), (2, 1, 1))
        self.t = t
        self.far = 0

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        z = np.array([x, y])
        if self.x is None or t - self.t > 0.5:
            self.reset(z, t)
            return x, y
        dt = max(t - self.t, 1e-3)
        self.t = t
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        for a in range(2):
            self.x[a] = F @ self.x[a]
            self.P[a] = F @ self.P[a] @ F.T + Q
        innov = z - self.x[:, 0]
        S = self.P[:, 0, 0] + self.r
        if np.any(np.abs(innov) > self.gate * np.sqrt(S)):
            self.far += 1
            if self.far >= self.confirm:          # a saccade: jump, don't smooth across it
                self.reset(z, t)
                return x, y
            return self._out()
        self.far = 0
        for a in range(2):
            K = self.P[a][:, 0] / S[a]
            self.x[a] = self.x[a] + K * innov[a]
            self.P[a] = self.P[a] - np.outer(K, self.P[a][0])
        return self._out()

    def _out(self) -> tuple[float, float]:
        # look ahead only with velocity the filter is sure about (|v| > 2 sd), else noise would be amplified
        out = []
        for a in range(2):
            v, vsd = self.x[a, 1], math.sqrt(max(self.P[a][1, 1], 1e-9))
            v_trust = v if abs(v) > 2 * vsd else 0.0
            out.append(self.x[a, 0] + self.lead * v_trust)
        return out[0], out[1]


class GazeSmoother:
    """Two regimes, like the eye: while the gaze moves (smooth pursuit, drift) the velocity Kalman
    follows it without lag; once the Kalman sees no trustworthy velocity for `still_s`, the robust
    fixation filter takes over and holds the dot steady. Saccades reset both.

    Replayed on a recorded pursuit + fixation test: moving 1.78° -> 1.3° and lag 0.35 s -> ~0.1 s
    versus the fixation filter alone, with the same steadiness on still targets.
    """

    def __init__(self, still_s: float = 0.15, **kalman):
        self.kf = GazeKalman(**kalman)
        self.ff = FixationFilter(radius=140, window=12)
        self.still_s = still_s
        self.still_since = None

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        k = self.kf(x, y, t)
        f = self.ff(x, y)
        moving = False
        for a in range(2):
            v, vsd = self.kf.x[a, 1], math.sqrt(max(self.kf.P[a][1, 1], 1e-9))
            moving |= abs(v) > 2 * vsd and abs(v) > 60.0       # px/s: ~1°/s, below that is drift/noise
        if moving:
            self.still_since = None
            return k
        if self.still_since is None:
            self.still_since = t
        return f if t - self.still_since >= self.still_s else k
