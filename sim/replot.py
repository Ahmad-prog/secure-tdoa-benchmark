"""Regenerate every figure from already-saved results.

    python -m sim.replot

Reads `results/data/*.csv` and the released dataset `.npz`, then calls the same
plotting routines the experiments use, so there is one plotting implementation
rather than two that can drift. Use this after a plotting change, or after a
long run whose figures were drawn by an older revision of the code -- it costs
seconds instead of re-running the study.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config
from .experiments import (exp_dataset, exp_tdoa, exp_antijam, exp_scaling,
                          exp_security, exp_cost, exp_sync)


def _csv(name):
    path = config.DATA_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def main():
    done, skipped = [], []

    npz = config.DATA_DIR / "spectrum_occupancy_dataset.npz"
    if npz.exists():
        z = np.load(npz)
        arrays = {k: z[k] for k in z.files if z[k].ndim == 2}
        exp_dataset._plot(arrays, int(z["M"]), int(z["slots"]))
        done.append("dataset")
    else:
        skipped.append("dataset")

    for name, fn, table in [
        ("tdoa",     exp_tdoa._plot,      "tdoa_rmse_vs_snr"),
        ("antijam",  exp_antijam._heatmap, "antijam_results"),
        ("scaling",  exp_scaling._plots,  "scaling_results"),
        ("security", exp_security._plot,  "security_results"),
        ("cost",     exp_cost._plot,      "cost_results"),
        ("sync",     exp_sync._plot,      "sync_results"),
    ]:
        df = _csv(table)
        if df is None:
            skipped.append(name); continue
        fn(df)
        done.append(name)

    print(f"\nreplotted: {', '.join(done) or 'nothing'}")
    if skipped:
        print(f"skipped (no saved results): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
