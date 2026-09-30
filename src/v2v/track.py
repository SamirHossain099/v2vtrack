"""Online methods. Each has step(i, f) -> ranked candidate beams (first = served) and feedback(i, f, fb).

f is the causal feature dict for the stream (phi, speed, range, moving); fb carries what the receiver learns after
serving: the power on the served beam, the powers on any locally swept beams, and on sweep frames the full vector.
"""
import numpy as np

from .geo import C_CENTRE, N_BEAMS, NOMINAL, delta_from_beam, rank_beams, wrap


class Method:
    name = "base"

    def __init__(self, K, delta0=0.0, **kw):
        self.K, self.delta = K, float(delta0)

    def step(self, i, f):
        return rank_beams(f["phi"][i], self.delta, self.K)[0]

    def feedback(self, i, f, fb):
        pass


class Static(Method):
    """The map with a fixed box rotation (the challenge baseline's constant shift, in angle)."""
    name = "static"


class BeamHold(Method):
    """Serve the last known best beam (the beam-hold baseline of Luo et al.)."""
    name = "beam-hold"

    def __init__(self, K, **kw):
        super().__init__(K)
        self.last = None

    def step(self, i, f):
        if self.last is None:
            return rank_beams(f["phi"][i], 0.0, self.K)[0]
        b = self.last
        offs = np.array([0, 1, -1, 2, -2])
        return np.clip(b % N_BEAMS + offs, 0, N_BEAMS - 1) + (b // N_BEAMS) * N_BEAMS

    def feedback(self, i, f, fb):
        if fb.get("sweep_beam") is not None:
            self.last = int(fb["sweep_beam"])
        elif fb.get("local_best") is not None:
            self.last = int(fb["local_best"])


class IndexOffsetMA(Method):
    """ViBe-style: track a beam-index offset between the map and the observed best beam by a moving average."""
    name = "index-offset-ma"

    def __init__(self, K, alpha=0.2, **kw):
        super().__init__(K)
        self.off, self.alpha = 0.0, alpha

    def step(self, i, f):
        c = rank_beams(f["phi"][i], 0.0, self.K)
        arr = c // N_BEAMS
        return np.clip(c % N_BEAMS + int(round(self.off)), 0, N_BEAMS - 1) + arr * N_BEAMS

    def feedback(self, i, f, fb):
        obs = fb.get("sweep_beam")
        if obs is None and fb.get("local_interior", False):
            obs = fb.get("local_best")
        if obs is None:
            return
        pred = rank_beams(f["phi"][i], 0.0, self.K)[0][0]
        if obs // N_BEAMS != pred // N_BEAMS:
            return
        d = (obs % N_BEAMS) - (pred % N_BEAMS)
        self.off = (1 - self.alpha) * self.off + self.alpha * d


class KalmanDelta(Method):
    """Box rotation as a random-walk state, measured at sweeps (and optionally at local sweeps) through the inverse map.

    Measurement noise: one beam of quantisation (1/K rad) plus the GPS bearing error at the current range
    (sigma_pos / range), added in quadrature. Process noise q per second sets how fast the offset may drift.
    """
    name = "kalman"

    def __init__(self, K, delta0=0.0, q_deg_per_sqrt_s=0.5, sigma_pos_m=0.7, P0_deg=5.0, use_local=True,
                 hold_still=True, hold_max_frames=50, **kw):
        super().__init__(K, delta0)
        self.q = np.radians(q_deg_per_sqrt_s) ** 2
        self.P = np.radians(P0_deg) ** 2
        self.sig_pos, self.use_local = sigma_pos_m, use_local
        self.hold_still, self.hold_max = hold_still, hold_max_frames
        self.agree_local = kw.get("agree_local", 3)
        self.t_last, self.last_obs, self.last_obs_i = None, None, None

    def step(self, i, f):
        # at standstill the geometry does not move, so the last observed best beam is the best guess we have
        if self.hold_still and not f["moving"][i] and self.last_obs is not None and i - self.last_obs_i <= self.hold_max:
            b = self.last_obs
            offs = np.array([0, 1, -1, 2, -2])
            return np.clip(b % N_BEAMS + offs, 0, N_BEAMS - 1) + (b // N_BEAMS) * N_BEAMS
        return rank_beams(f["phi"][i], self.delta, self.K)[0]

    def feedback(self, i, f, fb):
        t = fb["t"]
        if self.t_last is not None:
            self.P += self.q * max(t - self.t_last, 0.0)
        self.t_last = t
        obs = fb.get("sweep_beam")
        is_local = False
        if obs is None and self.use_local and fb.get("local_interior", False):
            obs, is_local = fb.get("local_best"), True
        if obs is not None:
            self.last_obs, self.last_obs_i = int(obs), i
        if obs is None or not f["moving"][i]:
            return
        if is_local:
            # a local maximum only measures the rotation if it was found near where the map looks; one found
            # around a held beam that has wandered onto a reflection would drag the rotation with it
            m0 = int(rank_beams(f["phi"][i], self.delta, self.K)[0][0])
            if obs // N_BEAMS != m0 // N_BEAMS or abs(obs - m0) > self.agree_local:
                return
        z = float(delta_from_beam(f["phi"][i], int(obs), self.K))
        r = (1.0 / self.K) ** 2 + (self.sig_pos / max(f["range"][i], 1.0)) ** 2
        y = wrap(z - self.delta)
        if abs(y) > np.radians(20):  # a reflection or a wrong-array sweep, not a rotation
            return
        g = self.P / (self.P + r)
        self.delta = float(wrap(self.delta + g * y))
        self.P *= (1 - g)


class RobustLS(Method):
    """01's recalibrator: box rotation re-fitted by a robust mean of the last W sweep-implied rotations."""
    name = "robust-window"

    def __init__(self, K, delta0=0.0, window=20, **kw):
        super().__init__(K, delta0)
        self.buf, self.window = [], window

    def feedback(self, i, f, fb):
        obs = fb.get("sweep_beam")
        if obs is None and fb.get("local_interior", False):
            obs = fb.get("local_best")
        if obs is None or not f["moving"][i]:
            return
        z = float(delta_from_beam(f["phi"][i], int(obs), self.K))
        if abs(wrap(z - self.delta)) > np.radians(20):
            return
        self.buf.append(z)
        self.buf = self.buf[-self.window:]
        if len(self.buf) >= 3:
            self.delta = float(np.angle(np.exp(1j * np.array(self.buf)).mean()))
            # circular median would be sturdier; the 20 deg gate above removes the gross outliers first


class KalmanHybrid(KalmanDelta):
    """The tracked map for recovery and array choice; the last observed best beam for fine alignment.

    Serve the last observed best beam (sweep or interior local maximum) while it is recent and agrees with the
    tracked map to within `agree` beams on the same array; otherwise serve the map. Beam-hold alone loses the
    beam and never recovers (E2); the map alone cannot resolve the last beam or two (GPS bearing noise).
    """
    name = "kalman-hybrid"

    def __init__(self, K, delta0=0.0, agree=3, hold_max_frames=30, **kw):
        super().__init__(K, delta0, hold_max_frames=hold_max_frames, **kw)
        self.agree = agree

    def step(self, i, f):
        cand = rank_beams(f["phi"][i], self.delta, self.K)[0]
        recent = self.last_obs is not None and i - self.last_obs_i <= self.hold_max
        if recent and not f["moving"][i]:
            b = self.last_obs
        elif recent and self.last_obs // N_BEAMS == cand[0] // N_BEAMS and abs(self.last_obs - cand[0]) <= self.agree:
            b = self.last_obs
        else:
            return cand
        offs = np.array([0, 1, -1, 2, -2])
        return np.clip(b % N_BEAMS + offs, 0, N_BEAMS - 1) + (b // N_BEAMS) * N_BEAMS


METHODS = {m.name: m for m in (Static, BeamHold, IndexOffsetMA, KalmanDelta, RobustLS, KalmanHybrid)}
