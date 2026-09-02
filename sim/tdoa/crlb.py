"""Cramer-Rao lower bound for TDOA multilateration (Chan & Ho / Gezici)."""
from __future__ import annotations
import numpy as np
from .geometry import true_ranges, tdoa_covariance


def jacobian(p: np.ndarray, beacons: np.ndarray, ref: int = 0) -> np.ndarray:
    """d(h_i)/d(p) for h_i = ||p-s_i|| - ||p-s_ref||  ->  (N-1, 3)."""
    diff = p[None, :] - beacons
    unit = diff / np.linalg.norm(diff, axis=1, keepdims=True)
    idx = [i for i in range(len(beacons)) if i != ref]
    return unit[idx] - unit[ref]


def crlb_cov(p: np.ndarray, beacons: np.ndarray, sigma_r: float,
             ref: int = 0) -> np.ndarray:
    """Position CRLB covariance (3x3) = (G^T Sigma^-1 G)^-1."""
    G = jacobian(p, beacons, ref)
    Sigma = tdoa_covariance(len(beacons), sigma_r, ref)
    W = np.linalg.inv(Sigma)
    F = G.T @ W @ G
    return np.linalg.inv(F)


def crlb_rmse(p: np.ndarray, beacons: np.ndarray, sigma_r: float,
              ref: int = 0) -> float:
    """sqrt(trace(CRLB)) = lower bound on position RMSE (metres)."""
    return float(np.sqrt(np.trace(crlb_cov(p, beacons, sigma_r, ref))))
