"""Experiment 2 -- anti-jamming benchmark of cryptographic hopping against the
cyber baselines across the nine-class taxonomy and channel scales.

Addresses RQ2: the cyber-technique comparison and the cryptographic anchor.
Produces: data/antijam_results.csv, figures/fig_antijam_heatmap.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import config, utils, spectrum
from ..jammers import taxonomy
from ..hopping import strategies as strat


def run(scales=config.CHANNEL_SCALES, n_seeds=config.N_SEEDS):
    rows = []
    for M in scales:
        for jam in taxonomy.ALL_JAMMERS:
            for s in strat.NONLEARNED:
                av, tp = [], []
                for k in range(n_seeds):
                    a, t = spectrum.evaluate(s, jam, M, config.BASE_SEED + k)
                    av.append(a); tp.append(t)
                av, tp = np.array(av), np.array(tp)
                rows.append({
                    "M": M, "jammer": jam, "tier": taxonomy.TIERS[jam],
                    "strategy": s,
                    "avoidance": av.mean(), "avoidance_sd": av.std(ddof=1),
                    "throughput": tp.mean(), "throughput_sd": tp.std(ddof=1),
                })
    df = pd.DataFrame(rows)
    utils.save_table(df, "antijam_results")
    _heatmap(df)
    _summary(df)
    return df


# Readable axis labels for the strategy keys used in the results tables.
PRETTY = {
    "crypto": "Cryptographic hop",
    "random": "Public random hop",
    "fixed": "Fixed channel",
    "dsss": "DSSS",
    "uss": "Uncoordinated SS",
}


def _heatmap(df, M=config.CHANNEL_SCALES[0]):
    sub = df[df.M == M].pivot(index="strategy", columns="jammer", values="avoidance")
    sub = sub.reindex(index=strat.NONLEARNED, columns=taxonomy.ALL_JAMMERS)
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    im = ax.imshow(sub.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sub.columns)))
    ax.set_xticklabels(sub.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(sub.index)))
    ax.set_yticklabels([PRETTY.get(s, s) for s in sub.index])
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.values[i, j]
            # white on the dark ends of the ramp, black in the middle
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if (v < 0.22 or v > 0.88) else "black")
    ax.set_xlabel("Jammer class")
    ax.set_ylabel("Hopping / spreading strategy")
    ax.set_title(f"Jamming-avoidance rate by strategy and jammer "
                 f"(M={M} channels; 1.00 = never jammed)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Jamming-avoidance rate", fontsize=8)
    fig.tight_layout()
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_antijam_heatmap.png", dpi=200)
    plt.close(fig)
    print("  wrote figures/fig_antijam_heatmap.png")


def _summary(df):
    print("\n  --- avoidance by strategy (mean over jammers) ---")
    for M in config.CHANNEL_SCALES:
        g = df[df.M == M].groupby("strategy").avoidance.mean()
        print(f"   M={M:5}: " + "  ".join(f"{k}={v:.2f}" for k, v in g.items()))
    print("\n  --- crypto vs learning jammer (unpredictability) ---")
    lc = df[(df.jammer == "learning")].pivot(index="M", columns="strategy", values="avoidance")
    print(lc[["crypto", "random", "fixed"]].round(3).to_string())


if __name__ == "__main__":
    run()
