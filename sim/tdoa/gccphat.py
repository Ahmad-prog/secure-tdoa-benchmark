"""Signal-level GCC-PHAT delay estimation, used to validate the range-noise
model against the measurement-level Gaussian assumption (proposal 4.4)."""
from __future__ import annotations
import numpy as np
from .. import config
from . import noise as noisemod


def bandlimited_signal(n: int, fs: float, b_hz: float,
                       gen: np.random.Generator) -> np.ndarray:
    """Unit-power real signal whose energy is confined to [0, b_hz]."""
    x = gen.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    X[f > b_hz] = 0.0
    y = np.fft.irfft(X, n=n)
    return y / np.sqrt(np.mean(y ** 2))


def delay_signal(x: np.ndarray, tau: float, fs: float) -> np.ndarray:
    """Fractional delay by tau seconds via a frequency-domain phase ramp."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return np.fft.irfft(X * np.exp(-1j * 2 * np.pi * f * tau), n=n)


def gcc_phat(x1: np.ndarray, x2: np.ndarray, fs: float,
             interp: int = 16) -> float:
    """Estimate the delay of x1 relative to x2 (seconds) with PHAT weighting."""
    n = len(x1) + len(x2)
    X1 = np.fft.rfft(x1, n=n)
    X2 = np.fft.rfft(x2, n=n)
    R = X1 * np.conj(X2)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n=interp * n)
    max_shift = interp * n // 2
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift / float(interp * fs)


def validate_noise_model(snr_db_list=config.SNR_DB_SWEEP, n: int = 4096,
                         trials: int = 200, seed: int = config.BASE_SEED):
    """Empirical range-error std from GCC-PHAT vs the analytic sigma_range.

    Returns a list of dicts (one per SNR) with empirical and analytic std and
    their ratio.  A near-constant ratio across SNR confirms the model's
    1/sqrt(SNR) scaling law that links the estimator noise to the CRLB.
    """
    gen = np.random.default_rng(seed)
    fs = 4.0 * config.BANDWIDTH_HZ
    out = []
    for snr_db in snr_db_list:
        snr = noisemod.snr_linear(snr_db)
        errs = []
        for _ in range(trials):
            s = bandlimited_signal(n, fs, config.BANDWIDTH_HZ, gen)
            tau = gen.uniform(-20, 20) / fs           # true delay (samples->s)
            x1 = delay_signal(s, tau, fs)
            npow = 1.0 / snr
            x1n = x1 + gen.normal(0, np.sqrt(npow), n)
            x2n = s + gen.normal(0, np.sqrt(npow), n)
            est = gcc_phat(x1n, x2n, fs)
            errs.append((est - tau) * config.C_LIGHT)
        emp = float(np.std(errs))
        ana = noisemod.sigma_range(snr_db)
        out.append({"snr_db": snr_db, "sigma_r_emp_m": emp,
                    "sigma_r_analytic_m": ana,
                    "ratio": emp / ana if ana else np.nan})
    return out
