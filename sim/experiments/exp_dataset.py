"""Experiment 0 -- build and release the labelled spectrum-occupancy dataset
(RO2 data: the released labelled spectrum-occupancy dataset).

For every jammer class a per-slot x per-channel occupancy matrix is generated
(clean=0 / jammed=1) and saved as an .npz release, with per-class statistics
and a spectrogram figure.

Produces: data/spectrum_occupancy_dataset.npz, data/dataset_stats.csv,
          figures/fig_dataset.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import config, utils, spectrum
from ..jammers import taxonomy


def run(M=32, slots=128, seed=config.BASE_SEED):
    arrays, rows = {}, []
    for jam in taxonomy.ALL_JAMMERS:
        occ = spectrum.occupancy_matrix(jam, M, slots, seed)
        arrays[jam] = occ
        rows.append({"jammer": jam, "tier": taxonomy.TIERS[jam],
                     "slots": slots, "channels": M,
                     "jammed_fraction": float(occ.mean()),
                     "mean_channels_jammed_per_slot": float(occ.sum(1).mean())})
    np.savez_compressed(config.DATA_DIR / "spectrum_occupancy_dataset.npz",
                        **arrays, M=M, slots=slots, window=config.WINDOW)
    df = pd.DataFrame(rows)
    utils.save_table(df, "dataset_stats")
    print(f"  wrote data/spectrum_occupancy_dataset.npz "
          f"({len(arrays)} classes, {slots}x{M})")
    _plot(arrays, M, slots)
    return df


def _plot(arrays, M, slots):
    # The grid is derived from the class count: a hard-coded 3x3 silently
    # dropped the tenth class when the taxonomy grew.
    names = [j for j in taxonomy.ALL_JAMMERS if j in arrays]
    ncol = 5 if len(names) > 9 else 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.5 * nrow),
                             squeeze=False)
    for ax, jam in zip(axes.ravel(), names):
        ax.imshow(arrays[jam].T, aspect="auto", cmap="Greys", vmin=0, vmax=1,
                  interpolation="nearest", origin="lower")
        ax.set_title(f"{jam}  [{taxonomy.TIERS[jam]} tier]", fontsize=9)
        ax.set_xlabel("Slot index", fontsize=8)
        ax.set_ylabel("Channel index", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(names):]:          # blank any unused cell
        ax.axis("off")
    fig.suptitle(f"Spectrum-occupancy dataset: jammed channels black, clean "
                 f"white ({slots} slots x {M} channels per class)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_dataset.png", dpi=200)
    plt.close(fig)
    print("  wrote figures/fig_dataset.png")


if __name__ == "__main__":
    run()
