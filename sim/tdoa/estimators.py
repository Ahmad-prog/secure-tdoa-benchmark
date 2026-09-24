"""TDOA position estimators: Chan-Ho closed form, Taylor-series (ML) refine,
and an IRLS/Huber robust variant for NLOS mitigation."""
from __future__ import annotations
import numpy as np
from .. import config
from .geometry import tdoa_covariance


def chan_ho(meas: np.ndarray, beacons: np.ndarray, sigma_r: float,
            ref: int = 0) -> np.ndarray:
    """Chan-Ho first-stage closed-form estimate (weighted least squares)."""
    idx = [i for i in range(len(beacons)) if i != ref]
    s_ref = beacons[ref]
    S = beacons[idx]
    d = meas
    K = np.sum(beacons ** 2, axis=1)
    A = np.column_stack([2.0 * (S - s_ref), 2.0 * d])
    b = K[idx] - K[ref] - d ** 2
    W = np.linalg.inv(tdoa_covariance(len(beacons), sigma_r, ref))
    z, *_ = np.linalg.lstsq(W @ A, W @ b, rcond=None)
    return z[:3]


def _h(p, beacons, ref):
    r = np.linalg.norm(beacons - p[None, :], axis=1)
    idx = [i for i in range(len(beacons)) if i != ref]
    return r[idx] - r[ref]


def taylor_refine(p0, meas, beacons, W, ref: int = 0, iters: int = 40,
                  tol: float = 1e-4, bound: float = 5.0) -> np.ndarray:
    """Damped (Levenberg-Marquardt) weighted-least-squares ML refinement.

    ``W`` is the measurement weighting matrix (inverse covariance, possibly
    robustly re-weighted).  LM damping with cost-controlled acceptance keeps
    the search stable at the weak elevation angles of a ground constellation.
    """
    from .crlb import jacobian
    span = bound * config.BEACON_RADIUS

    def cost(p):
        r = meas - _h(p, beacons, ref)
        return float(r @ W @ r)

    p = p0.astype(float).copy()
    lam, c = 1e-3, cost(p0.astype(float))
    for _ in range(iters):
        G = jacobian(p, beacons, ref)
        resid = meas - _h(p, beacons, ref)
        A = G.T @ W @ G
        g = G.T @ W @ resid
        improved = False
        for _inner in range(8):
            try:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            except np.linalg.LinAlgError:
                lam *= 5.0
                continue
            p_new = np.clip(p + step, -span, span)
            c_new = cost(p_new)
            if c_new < c:
                p, c, lam, improved = p_new, c_new, max(lam * 0.5, 1e-9), True
                break
            lam *= 5.0
        if not improved or np.linalg.norm(step) < tol:
            break
    return p


def _clip(p, bound: float = 5.0):
    """Constrain a fix to a sane operational box (prevents rare divergence)."""
    span = bound * config.BEACON_RADIUS
    return np.array([np.clip(p[0], -span, span), np.clip(p[1], -span, span),
                     np.clip(p[2], -50.0, 2 * span)])


def estimate(meas, beacons, sigma_r, ref: int = 0) -> np.ndarray:
    """Chan-Ho initialiser + Taylor ML refinement (full covariance weighting)."""
    W = np.linalg.inv(tdoa_covariance(len(beacons), sigma_r, ref))
    p0 = chan_ho(meas, beacons, sigma_r, ref)
    if not np.all(np.isfinite(p0)):
        p0 = np.array([0.0, 0.0, config.DRONE_ALT])
    return _clip(taylor_refine(p0, meas, beacons, W, ref))


def _solve_subset(beacons_arr, meas_arr, sigma_r, lref):
    W = np.linalg.inv(tdoa_covariance(len(beacons_arr), sigma_r, lref))
    p0 = chan_ho(meas_arr, beacons_arr, sigma_r, lref)
    if not np.all(np.isfinite(p0)):
        p0 = np.array([0.0, 0.0, config.DRONE_ALT])
    return taylor_refine(p0, meas_arr, beacons_arr, W, lref)


def estimate_robust(meas, beacons, sigma_r, ref: int = 0, iters: int = 40,
                    subset: int = 5, thresh_mult: float = 3.0,
                    seed: int = 0) -> np.ndarray:
    """RANSAC NLOS mitigation: find the largest self-consistent beacon set.

    TDOA yields every pairwise range difference, so we can re-reference freely
    (g[i]=r_i-r_0 measured; diff wrt any k is g[i]-g[k]).  We sample beacon
    subsets and references, fit, score inliers across all beacons, keep the
    best consensus, then refit on the inliers.  This rejects NLOS-biased
    beacons even when the nominal reference is itself in NLOS.
    """
    n = len(beacons)
    g = np.concatenate([[0.0], meas])                 # g[i] = r_i - r_0
    thresh = thresh_mult * np.sqrt(2.0) * sigma_r
    gen = np.random.default_rng(seed)

    def diffs(indices, kref):
        others = [i for i in indices if i != kref]
        return beacons[[kref] + others], np.array([g[i] - g[kref] for i in others]), others

    def inliers_of(p, kref):
        pred = np.linalg.norm(beacons - p[None, :], axis=1)
        pred = pred - pred[kref]
        res = pred - (g - g[kref])
        return np.abs(res) < thresh

    best_mask, best_p = None, None
    for _ in range(iters):
        kref = int(gen.integers(n))
        pool = [i for i in range(n) if i != kref]
        sel = list(gen.choice(pool, size=min(subset - 1, len(pool)), replace=False))
        bs, ms, _ = diffs([kref] + sel, kref)
        try:
            p = _solve_subset(bs, ms, sigma_r, 0)
        except Exception:
            continue
        mask = inliers_of(p, kref)
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask, best_p = mask, p
    if best_mask is None or best_mask.sum() < subset:
        return estimate(meas, beacons, sigma_r, ref)
    # refit on the inlier consensus set, referenced to an inlier beacon
    idx = np.where(best_mask)[0]
    kref = int(idx[0])
    bs, ms, _ = diffs(list(idx), kref)
    return _clip(_solve_subset(bs, ms, sigma_r, 0))
