"""Experiment 0 -- build and release the labelled spectrum-occupancy dataset
(RO2 data, DRAC C1/C4 dataset relevance).

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
    fig, axes = plt.subplots(3, 3, figsize=(9, 7))
    for ax, jam in zip(axes.ravel(), taxonomy.ALL_JAMMERS):
        ax.imshow(arrays[jam].T, aspect="auto", cmap="Greys",
                  interpolation="nearest", origin="lower")
        ax.set_title(f"{jam}  [{taxonomy.TIERS[jam]}]", fontsize=9)
        ax.set_xlabel("slot", fontsize=7); ax.set_ylabel("channel", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("Spectrum-occupancy dataset: jammed (black) vs clean per slot",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_dataset.png", dpi=200)
    plt.close(fig)
    print("  wrote figures/fig_dataset.png")


if __name__ == "__main__":
    run()
