"""Hopping / spread-spectrum strategies compared in the study.

Single-channel hoppers pick one channel per slot; spread strategies occupy the
band.  Each ``step`` returns (channel, avoided, throughput).  ``peek`` exposes
the next channel to an adversary IFF the strategy is public/deterministic; the
cryptographic hopper returns None because the sequence needs the secret key.
"""
from __future__ import annotations
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from .. import config

DSSS_TOL = 0.34            # processing-gain fraction DSSS tolerates
USS_RENDEZVOUS = 0.5      # coordination overhead of uncoordinated SS (Popper)


class Strategy:
    kind = "single"
    name = "base"
    def reset(self): pass
    def peek(self, t): return None


class FixedChannel(Strategy):
    name = "fixed"
    def __init__(self, M): self.M = M
    def channel(self, t): return 0
    def peek(self, t): return 0
    def step(self, t, jammed):
        c = 0; ok = not jammed[c]
        return c, ok, 1.0 if ok else 0.0


class RandomHop(Strategy):
    """Public pseudo-random hopping: deterministic given a *public* seed, so an
    adversary who knows the seed can predict it exactly."""
    name = "random"
    def __init__(self, M, seed=0): self.M, self.seed = M, seed
    def channel(self, t):
        return int(np.random.default_rng(self.seed + t).integers(self.M))
    def peek(self, t): return self.channel(t)
    def step(self, t, jammed):
        c = self.channel(t); ok = not jammed[c]
        return c, ok, 1.0 if ok else 0.0


class CryptoHop(Strategy):
    """AES-CTR keyed hopping: channel = AES_k(counter) mod M.  Unpredictable to
    any adversary lacking the key (peek returns None)."""
    name = "crypto"
    def __init__(self, M, key=None):
        self.M = M
        self.key = key or bytes(range(config.AES_KEY_BYTES))
    def channel(self, t):
        enc = Cipher(algorithms.AES(self.key), modes.ECB()).encryptor()
        ct = enc.update(t.to_bytes(16, "big")) + enc.finalize()
        return int.from_bytes(ct[:4], "big") % self.M
    def peek(self, t): return None                    # needs the secret key
    def step(self, t, jammed):
        c = self.channel(t); ok = not jammed[c]
        return c, ok, 1.0 if ok else 0.0


class DSSS(Strategy):
    """Direct-sequence spread spectrum: robust to narrowband jamming up to the
    processing-gain fraction, degrades under wideband/barrage jamming."""
    kind = "spread"; name = "dsss"
    def __init__(self, M): self.M = M
    def step(self, t, jammed):
        frac = float(jammed.mean())
        tput = 1.0 if frac <= DSSS_TOL else max(0.0, 1 - (frac - DSSS_TOL) / (1 - DSSS_TOL))
        return -1, tput > 0.5, tput


class UncoordinatedSS(Strategy):
    """Uncoordinated spread spectrum (Popper/Strasser anti-jam anchor): jam
    resistant without a shared key, at the price of rendezvous overhead."""
    name = "uss"
    def __init__(self, M, seed=0): self.M, self.seed = M, seed
    def channel(self, t):
        return int(np.random.default_rng(self.seed + 7919 * t).integers(self.M))
    def peek(self, t): return None                    # no shared sequence to follow
    def step(self, t, jammed):
        c = self.channel(t); ok = not jammed[c]
        return c, ok, (USS_RENDEZVOUS if ok else 0.0)


def make_strategy(name, M, seed=0):
    if name == "fixed": return FixedChannel(M)
    if name == "random": return RandomHop(M, seed)
    if name == "crypto": return CryptoHop(M)
    if name == "dsss": return DSSS(M)
    if name == "uss": return UncoordinatedSS(M, seed)
    raise ValueError(name)


NONLEARNED = ["crypto", "random", "fixed", "dsss", "uss"]
