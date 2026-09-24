"""Map SNR and bandwidth to a range-estimation standard deviation.

Uses the classical time-of-arrival Cramer-Rao bound (Gezici et al., 2005):

    sigma_tau  >=  1 / (2*sqrt(2)*pi * sqrt(SNR) * beta)

where beta is the effective (RMS) bandwidth of the signal.  For a flat
spectrum of two-sided bandwidth B, beta = B / sqrt(3).  The range std is
sigma_r = c * sigma_tau.  This single relation ties the measurement noise
used by the estimators to the CRLB used as the benchmark, so the two are
mutually consistent.
"""
from __future__ import annotations
import numpy as np
from .. import config


def snr_linear(snr_db: float) -> float:
    return 10.0 ** (snr_db / 10.0)


def rms_bandwidth(b_hz: float = config.BANDWIDTH_HZ) -> float:
    return b_hz / np.sqrt(3.0)


def sigma_tau(snr_db: float, b_hz: float = config.BANDWIDTH_HZ) -> float:
    beta = rms_bandwidth(b_hz)
    return 1.0 / (2.0 * np.sqrt(2.0) * np.pi * np.sqrt(snr_linear(snr_db)) * beta)


def sigma_range(snr_db: float, b_hz: float = config.BANDWIDTH_HZ) -> float:
    """Range-estimation std (metres) for a given SNR and bandwidth."""
    return config.C_LIGHT * sigma_tau(snr_db, b_hz)
