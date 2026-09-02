"""Experiment 3 -- deep-learning-at-scale study (RO2/RO3, DRAC C3).

Trains the DQN under a fixed wall-clock budget as the channel count scales from
10 to 2320, against an adaptive fraction-jamming adversary, and compares with
training-free cryptographic hopping.  Shows (a) the avoidance crossover and
(b) the O(M) training-cost blow-up.

Produces: data/scaling_results.csv, figures/fig_scaling_crossover.png,
          figures/fig_scaling_cost.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import config, utils, spectrum
from ..jammers import taxonomy
from ..hopping import dqn

JAMMER = "smart_partial"          # adaptive, jams ~30% of the band
BUDGET_S = 20.0
N_SEEDS_DQN = 3


def run(scales=config.CHANNEL_SCALES):
    rows = []
    for M in scales:
        dq_av, secs, steps, conv = [], [], [], []
        for k in range(N_SEEDS_DQN):
            fac = lambda s, M=M: taxonomy.make_jammer(JAMMER, M, seed=s + 101)
            r = dqn.train(fac, M, max_steps=200000, time_budget=BUDGET_S,
                          seed=config.BASE_SEED + k)
            dq_av.append(r["final_avoidance"]); secs.append(r["train_seconds"])
            steps.append(r["steps"]); conv.append(r["converged_step"])
            params = r["params"]
        cav = np.mean([spectrum.evaluate("crypto", JAMMER, M, config.BASE_SEED + k)[0]
                       for k in range(N_SEEDS_DQN)])
        dq_av = np.array(dq_av); steps = np.array(steps); secs = np.array(secs)
        rows.append({
            "M": M, "params": params,
            "dqn_avoidance": dq_av.mean(), "dqn_avoidance_sd": dq_av.std(ddof=1),
            "crypto_avoidance": float(cav),
            "train_seconds": secs.mean(), "steps": steps.mean(),
            "steps_per_sec": (steps / secs).mean(),
            "converged": int(sum(c is not None for c in conv)),
        })
        print(f"   M={M:5} DQN={dq_av.mean():.2f}+-{dq_av.std(ddof=1):.2f} "
              f"crypto={cav:.2f} steps/s={rows[-1]['steps_per_sec']:.0f} "
              f"params={params}")
    df = pd.DataFrame(rows)
    utils.save_table(df, "scaling_results")
    _plots(df)
    return df


def _plots(df):
    # crossover
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.errorbar(df.M, df.dqn_avoidance, yerr=df.dqn_avoidance_sd, marker="o",
                capsize=3, label="DQN (20 s train budget)")
    ax.plot(df.M, df.crypto_avoidance, "s--", label="Cryptographic hopping (training-free)")
    ax.set_xscale("log"); ax.set_xlabel("Number of channels M")
    ax.set_ylabel("Jamming-avoidance rate"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("Learned vs cryptographic hopping as channels scale")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend()
    fig.tight_layout()
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_scaling_crossover.png", dpi=200)
    plt.close(fig)

    # cost
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.loglog(df.M, df.params, "o-", label="DQN parameters (O(M))")
    ax2 = ax.twinx()
    ax2.loglog(df.M, df.steps_per_sec, "s--", color="tab:red",
               label="training steps / s")
    ax.set_xlabel("Number of channels M"); ax.set_ylabel("Parameter count")
    ax2.set_ylabel("Training steps per second", color="tab:red")
    ax.set_title("Deep-learning training cost vs channel count")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_scaling_cost.png", dpi=200)
    plt.close(fig)
    print("  wrote figures/fig_scaling_crossover.png, fig_scaling_cost.png")


if __name__ == "__main__":
    run()
