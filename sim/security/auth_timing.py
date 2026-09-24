"""Authenticated-timing protocol and the spoof/replay adversaries.

A beacon broadcasts m = {id, T_tx, nonce} with an HMAC over m under the shared
key.  The drone accepts a measurement only if (i) the HMAC verifies, (ii) the
echoed nonce equals the fresh nonce it just issued, and (iii) the arrival time
is consistent with T_tx plus the geometric time-of-flight, within tolerance.

Security layers (increasing):
  L0 none          : fixed channel, no authentication
  L1 crypto        : cryptographic hopping only (no message auth)
  L2 crypto+sig    : hopping + HMAC signature only (no nonce / no ToF check)
  L3 auth-timing   : hopping + HMAC + fresh nonce + time-of-flight check (full)

The L2 vs L3 contrast is the central point: a valid signature does NOT stop a
meaconing replay from shifting the timing; the nonce + ToF binding does.
"""
from __future__ import annotations
import hmac, hashlib, os
import numpy as np
from .. import config

LAYERS = ["L0_none", "L1_crypto", "L2_crypto_sig", "L3_auth_timing"]
TOF_TOL_SIGMA = 3.0               # ToF tolerance in units of range-noise sigma


def _tag(key, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()


class Beacon:
    def __init__(self, bid: int, key: bytes):
        self.bid, self.key = bid, key

    def emit(self, t_tx: float, nonce: bytes):
        msg = f"{self.bid}|{t_tx:.9f}|".encode() + nonce
        return {"bid": self.bid, "t_tx": t_tx, "nonce": nonce,
                "msg": msg, "tag": _tag(self.key, msg)}


def verify(pkt, key, arrival: float, geo_delay: float, current_nonce: bytes,
           sigma_r: float, layer: str) -> bool:
    """Return True if the drone accepts the packet under the given layer."""
    if layer == "L0_none":
        return True
    if layer == "L1_crypto":
        return True                                     # reception implies right channel; no auth
    # signature check (L2, L3)
    if not hmac.compare_digest(pkt["tag"], _tag(key, pkt["msg"])):
        return False
    if layer == "L2_crypto_sig":
        return True                                     # signature valid -> accepted (no timing bind)
    # L3: fresh nonce + time-of-flight plausibility
    if not hmac.compare_digest(pkt["nonce"], current_nonce):
        return False
    tol = TOF_TOL_SIGMA * sigma_r / config.C_LIGHT      # tolerance in seconds
    return abs(arrival - (pkt["t_tx"] + geo_delay)) <= tol
