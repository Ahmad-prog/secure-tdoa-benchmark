"""Beacon geometry and the forward TDOA measurement model."""
from __future__ import annotations
import numpy as np
from .. import config


def hexagon_beacons(n: int = config.N_BEACONS,
                    radius: float = config.BEACON_RADIUS,
                    heights=None) -> np.ndarray:
    """Return an (n, 3) array of beacon coordinates on a regular polygon.

    ``heights`` gives per-beacon z (mast/terrain diversity); it is cycled if
    shorter than n and defaults to the configured pattern.
    """
    if heights is None:
        heights = config.BEACON_HEIGHTS
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    z = np.array([heights[i % len(heights)] for i in range(n)])
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang), z])


def random_drone_position(gen: np.random.Generator,
                          radius: float = config.BEACON_RADIUS,
                          alt: float = config.DRONE_ALT) -> np.ndarray:
    """A random 3-D drone position inside the constellation footprint."""
    r = radius * 0.8 * np.sqrt(gen.random())
    th = 2 * np.pi * gen.random()
    return np.array([r * np.cos(th), r * np.sin(th),
                     alt * (0.5 + gen.random())])


def true_ranges(p: np.ndarray, beacons: np.ndarray) -> np.ndarray:
    return np.linalg.norm(beacons - p[None, :], axis=1)


def range_differences(p: np.ndarray, beacons: np.ndarray, ref: int = 0) -> np.ndarray:
    """Noise-free range differences d_i = r_i - r_ref for i != ref."""
    r = true_ranges(p, beacons)
    idx = [i for i in range(len(beacons)) if i != ref]
    return r[idx] - r[ref]


def measure_tdoa(p: np.ndarray, beacons: np.ndarray, sigma_r: float,
                 gen: np.random.Generator, ref: int = 0,
                 nlos_mask: np.ndarray | None = None,
                 nlos_bias_mean: float = config.NLOS_BIAS_MEAN_M
                 ) -> np.ndarray:
    """Simulate measured range differences.

    Each beacon range carries independent Gaussian estimation noise of std
    ``sigma_r``.  Beacons flagged in ``nlos_mask`` additionally carry a
    one-sided (exponential) positive excess-delay bias, so NLOS inflates the
    error above the (Gaussian) CRLB, exactly the behaviour reported for real
    3-D UAV TDOA.
    """
    r = true_ranges(p, beacons)
    noisy = r + gen.normal(0.0, sigma_r, size=r.shape)
    if nlos_mask is not None:
        bias = gen.exponential(nlos_bias_mean, size=r.shape) * nlos_mask
        noisy = noisy + bias
    idx = [i for i in range(len(beacons)) if i != ref]
    return noisy[idx] - noisy[ref]


def tdoa_covariance(n_beacons: int, sigma_r: float, ref: int = 0) -> np.ndarray:
    """Covariance of the range-difference vector.

    d_i = (r_i + n_i) - (r_ref + n_ref).  With i.i.d. per-range noise of
    variance sigma_r^2:  Cov(d_i, d_j) = sigma_r^2 (I + 11^T).
    """
    m = n_beacons - 1
    return sigma_r ** 2 * (np.eye(m) + np.ones((m, m)))
