"""Experiment 1 -- TDOA positioning accuracy vs the Cramer-Rao lower bound.

Addresses RO1 / RQ1: positioning accuracy against the CRLB.
Produces:  data/tdoa_rmse_vs_snr.csv, data/tdoa_gccphat_validation.csv,
           figures/fig_tdoa_crlb.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import config, utils
from ..plotstyle import COL_W, PAGE_W
from ..tdoa import geometry as geo, crlb, estimators, noise as noisemod, gccphat


def run(n_trials: int = config.N_TRIALS):
    beacons = geo.hexagon_beacons()
    # method label -> (nlos?, estimator fn)
    conditions = [
        ("LOS", False, estimators.estimate),
        ("NLOS", True, estimators.estimate),
        ("NLOS+mitigation", True, estimators.estimate_robust),
    ]
    rows = []
    for snr_db in config.SNR_DB_SWEEP:
        sigma_r = noisemod.sigma_range(snr_db)
        for cond, nlos, est in conditions:
            gen = utils.rng(config.BASE_SEED + int(snr_db) * 7 + (2 if "mit" in cond else int(nlos)))
            err, crlb_vals = [], []
            for _ in range(n_trials):
                p = geo.random_drone_position(gen)
                mask = ((gen.random(len(beacons)) < config.NLOS_FRACTION).astype(float)
                        if nlos else None)
                m = geo.measure_tdoa(p, beacons, sigma_r, gen, nlos_mask=mask)
                phat = est(m, beacons, sigma_r)
                err.append(float(np.linalg.norm(phat - p)))
                crlb_vals.append(crlb.crlb_rmse(p, beacons, sigma_r))
            err = np.array(err)
            rmse = float(np.sqrt(np.mean(err ** 2)))
            rows.append({"snr_db": snr_db, "condition": cond,
                         "sigma_r_m": sigma_r, "rmse_m": rmse,
                         "median_m": float(np.median(err)),
                         "cep95_m": float(np.percentile(err, 95)),
                         "crlb_m": float(np.mean(crlb_vals)),
                         "efficiency": float(np.mean(crlb_vals)) / rmse})
    df = pd.DataFrame(rows)
    utils.save_table(df, "tdoa_rmse_vs_snr")

    val = pd.DataFrame(gccphat.validate_noise_model())
    utils.save_table(val, "tdoa_gccphat_validation")

    _plot(df)
    _summary(df, val)
    return df, val


def _plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    # LOS: RMSE hugs the CRLB. NLOS error is heavy-tailed, so median is the
    # meaningful statistic (standard practice in the localization literature).
    los = df[df.condition == "LOS"]
    ax.semilogy(los.snr_db, los.rmse_m, "-o", label="LOS (RMSE)")
    nlos = df[df.condition == "NLOS"]
    ax.semilogy(nlos.snr_db, nlos.median_m, "--s", label="NLOS (median)")
    mit = df[df.condition == "NLOS+mitigation"]
    ax.semilogy(mit.snr_db, mit.median_m, "-.^", label="NLOS + mitigation (median)")
    ax.semilogy(los.snr_db, los.crlb_m, "k:", label="CRLB")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Position error (m)")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(loc="lower left")
    fig.tight_layout()
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_tdoa_crlb.png")
    plt.close(fig)
    print("  wrote figures/fig_tdoa_crlb.png")


def _summary(df, val):
    print("\n  --- TDOA summary @20 dB ---")
    for cond in ["LOS", "NLOS", "NLOS+mitigation"]:
        r = df[(df.condition == cond) & (df.snr_db == 20)].iloc[0]
        print(f"   {cond:16}: RMSE {r.rmse_m:8.2f} m | median {r.median_m:7.2f} m "
              f"| CRLB {r.crlb_m:6.2f} m | eff {r.efficiency:.2f}")
    print(f"   GCC-PHAT/analytic sigma_r ratio (mean): {val.ratio.mean():.2f}")


if __name__ == "__main__":
    run()
