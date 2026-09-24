"""Ten-class, three-tier jammer taxonomy.

Every jammer exposes ``step(t, history) -> mask`` where ``mask`` is a boolean
array of length M marking jammed channels for slot t.  ``history`` is the list
of channels the transmitter has used so far (spread transmissions append -1),
which the adaptive (Tier-3) jammers exploit.
"""
from __future__ import annotations
import numpy as np

# Fraction of the band a wideband barrage jammer denies. Set above the DSSS
# processing-gain tolerance (strategies.DSSS_TOL) so the taxonomy spans both
# sides of that margin instead of only the narrowband side.
BARRAGE_FRAC = 0.70

TIERS = {
    "constant": "low", "random": "low", "pulsed": "low", "barrage": "low",
    "sweep": "mid", "comb": "mid", "partial_band": "mid",
    "reactive": "top", "learning": "top", "smart_partial": "top",
}


class Jammer:
    def __init__(self, M: int, seed: int = 0):
        self.M = M
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def _empty(self):
        return np.zeros(self.M, dtype=bool)


# ---- Tier 1: fixed ------------------------------------------------------
class ConstantJammer(Jammer):
    def step(self, t, history):
        m = self._empty(); m[0] = True; return m


class RandomChannelJammer(Jammer):
    def step(self, t, history):
        m = self._empty(); m[self.rng.integers(self.M)] = True; return m


class PulsedJammer(Jammer):
    def __init__(self, M, seed=0, period=4, duty=2):
        super().__init__(M, seed); self.period, self.duty = period, duty

    def step(self, t, history):
        m = self._empty()
        if t % self.period < self.duty:
            m[0] = True
        return m


class BarrageJammer(Jammer):
    """Wideband barrage: noise spread over a large fraction of the band.

    A fresh random subset of ``frac`` of the channels is denied each slot. This
    is the brute-force counter to spread spectrum -- it exceeds a DSSS
    receiver's processing-gain margin, which no narrowband jammer in this
    taxonomy does. The price the adversary pays, diluting a fixed transmit
    power across the whole band, is a power-domain effect and is therefore not
    represented in this occupancy-level model; see the limitations discussion.
    """
    def __init__(self, M, seed=0, frac=BARRAGE_FRAC):
        super().__init__(M, seed)
        self.w = max(1, int(frac * M))

    def step(self, t, history):
        m = self._empty()
        m[self.rng.choice(self.M, size=min(self.w, self.M), replace=False)] = True
        return m


# ---- Tier 2: pattern ----------------------------------------------------
class SweepJammer(Jammer):
    def step(self, t, history):
        m = self._empty(); m[t % self.M] = True; return m


class CombJammer(Jammer):
    def __init__(self, M, seed=0, teeth=5):
        super().__init__(M, seed)
        self.chans = np.linspace(0, M, teeth, endpoint=False).astype(int)

    def step(self, t, history):
        m = self._empty(); m[self.chans] = True; return m


class PartialBandJammer(Jammer):
    def __init__(self, M, seed=0, frac=0.30):
        super().__init__(M, seed)
        self.w = max(1, int(frac * M)); self.b = 0

    def step(self, t, history):
        m = self._empty(); m[self.b:self.b + self.w] = True; return m


# ---- Tier 3: intelligent ------------------------------------------------
class ReactiveJammer(Jammer):
    """Follower: jams the last-used channel and its immediate neighbours."""
    def step(self, t, history):
        m = self._empty()
        if history:
            c = history[-1]
            if c >= 0:
                for d in (-1, 0, 1):
                    m[(c + d) % self.M] = True
        return m


class LearningJammer(Jammer):
    """Predictive: models the transmitter's channel statistics and jams the
    top-k most likely next channels.  A public/deterministic strategy is
    predicted exactly through the supplied ``oracle``; an unpredictable
    (cryptographic) strategy defeats it, leaving only a 1/M chance."""
    def __init__(self, M, seed=0, k=1, oracle=None):
        super().__init__(M, seed); self.k = k; self.oracle = oracle
        self.counts = np.zeros(M)

    def reset(self):
        self.counts[:] = 0.0

    def step(self, t, history):
        m = self._empty()
        if self.oracle is not None:
            m[self.oracle(t) % self.M] = True          # exact prediction
            return m
        if history and history[-1] >= 0:
            self.counts[history[-1]] += 1.0
        top = np.argsort(self.counts)[::-1][:self.k]
        m[top] = True
        return m


class SmartPartialBandJammer(Jammer):
    """Places a jamming block over the transmitter's most active region."""
    def __init__(self, M, seed=0, frac=0.30):
        super().__init__(M, seed); self.w = max(1, int(frac * M))
        self.counts = np.zeros(M)

    def reset(self):
        self.counts[:] = 0.0

    def step(self, t, history):
        if history and history[-1] >= 0:
            self.counts[history[-1]] += 1.0
        # block centred on the busiest channel
        centre = int(np.argmax(self.counts)) if self.counts.any() else 0
        b = max(0, min(self.M - self.w, centre - self.w // 2))
        m = self._empty(); m[b:b + self.w] = True; return m


def make_jammer(name: str, M: int, seed: int = 0, oracle=None) -> Jammer:
    if name == "constant": return ConstantJammer(M, seed)
    if name == "random": return RandomChannelJammer(M, seed)
    if name == "pulsed": return PulsedJammer(M, seed)
    if name == "barrage": return BarrageJammer(M, seed)
    if name == "sweep": return SweepJammer(M, seed)
    if name == "comb": return CombJammer(M, seed)
    if name == "partial_band": return PartialBandJammer(M, seed)
    if name == "reactive": return ReactiveJammer(M, seed)
    if name == "learning": return LearningJammer(M, seed, oracle=oracle)
    if name == "smart_partial": return SmartPartialBandJammer(M, seed)
    raise ValueError(name)


ALL_JAMMERS = list(TIERS.keys())
