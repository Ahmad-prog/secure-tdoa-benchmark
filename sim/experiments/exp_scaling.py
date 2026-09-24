"""Experiment 3 -- deep-learning-at-scale study (RO2/RO3).

Scales the channel count from 10 to 2320 against an adaptive fraction-jamming
adversary and compares the learned policy with training-free cryptographic
hopping under two training regimes:

  * equal wall clock -- every M gets the same training seconds (BUDGET_S).
    This is the deployment-realistic regime: a fixed compute allowance.
  * equal gradient steps -- every M gets the same number of updates
    (STEP_MATCHED), however long that takes. This removes the objection that
    the wide-band models are simply being starved of compute, and isolates the
    degradation that is due to the learning problem itself rather than to the
    budget.

Reporting both is the point: the gap between the arms is the cost of the
budget, and whatever degradation survives the step-matched arm is intrinsic.

Produces: data/scaling_results.csv, figures/fig_scaling_crossover.png,
          figures/fig_scaling_cost.png, figures/fig_scaling_convergence.png
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
BUDGET_S = 20.0                   # equal-wall-clock arm
N_SEEDS_DQN = 10                  # seeds for the (cheap) budget arm
STEP_MATCHED = 15000              # equal-gradient-update arm
N_SEEDS_STEP = 3                  # seeds for the (expensive) step-matched arm


def _train_arm(M, n_seeds, *, time_budget, max_steps):
    """Train n_seeds models at channel count M and collect per-seed outcomes."""
    av, secs, steps, conv = [], [], [], []
    params = None
    for k in range(n_seeds):
        fac = lambda s, M=M: taxonomy.make_jammer(JAMMER, M, seed=s + 101)
        r = dqn.train(fac, M, max_steps=max_steps, time_budget=time_budget,
                      seed=config.BASE_SEED + k)
        av.append(r["final_avoidance"]); secs.append(r["train_seconds"])
        steps.append(r["steps"]); conv.append(r["converged_step"])
        params = r["params"]
    return {
        "avoidance": np.array(av), "seconds": np.array(secs),
        "steps": np.array(steps), "converged": conv, "params": params,
    }


def run(scales=config.CHANNEL_SCALES, resume=True):
    # Each channel count costs up to hours, so every completed M is checkpointed
    # immediately. A crash, a reboot, or a kill then costs one M, not the run.
    ckpt = config.DATA_DIR / "scaling_partial.csv"
    rows = []
    if resume and ckpt.exists():
        rows = pd.read_csv(ckpt).to_dict("records")
        done = {int(r["M"]) for r in rows}
        if done:
            print(f"   resuming: {sorted(done)} already complete")
        scales = [m for m in scales if m not in done]

    for M in scales:
        budget = _train_arm(M, N_SEEDS_DQN, time_budget=BUDGET_S, max_steps=200000)
        matched = _train_arm(M, N_SEEDS_STEP, time_budget=None, max_steps=STEP_MATCHED)

        cav = np.mean([spectrum.evaluate("crypto", JAMMER, M, config.BASE_SEED + k)[0]
                       for k in range(N_SEEDS_DQN)])

        # Steps to first reach 0.90 avoidance; None means never reached within
        # the arm's allowance, which is itself the finding at large M.
        reached = [c for c in matched["converged"] if c is not None]
        rows.append({
            "M": M, "params": budget["params"],
            # arm A -- equal wall clock
            "dqn_avoidance": budget["avoidance"].mean(),
            "dqn_avoidance_sd": budget["avoidance"].std(ddof=1),
            "train_seconds": budget["seconds"].mean(),
            "steps": budget["steps"].mean(),
            "steps_per_sec": (budget["steps"] / budget["seconds"]).mean(),
            "converged": int(sum(c is not None for c in budget["converged"])),
            "n_seeds_budget": N_SEEDS_DQN,
            # arm B -- equal gradient steps
            "dqn_steps_avoidance": matched["avoidance"].mean(),
            "dqn_steps_avoidance_sd": matched["avoidance"].std(ddof=1),
            "steps_arm_seconds": matched["seconds"].mean(),
            "steps_arm_steps": matched["steps"].mean(),
            "steps_arm_converged": len(reached),
            "steps_to_converge": float(np.mean(reached)) if reached else np.nan,
            "n_seeds_matched": N_SEEDS_STEP,
            # training-free reference
            "crypto_avoidance": float(cav),
        })
        r = rows[-1]
        print(f"   M={M:5} budget={r['dqn_avoidance']:.2f}+-{r['dqn_avoidance_sd']:.2f} "
              f"step-matched={r['dqn_steps_avoidance']:.2f}+-{r['dqn_steps_avoidance_sd']:.2f} "
              f"crypto={cav:.2f} steps/s={r['steps_per_sec']:.0f} params={r['params']}",
              flush=True)
        pd.DataFrame(rows).to_csv(ckpt, index=False)      # checkpoint this M

    df = pd.DataFrame(rows).sort_values("M").reset_index(drop=True)
    utils.save_table(df, "scaling_results")
    _plots(df)
    return df


def _plots(df):
    # crossover -- both training regimes against the training-free baseline
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.errorbar(df.M, df.dqn_avoidance, yerr=df.dqn_avoidance_sd, marker="o",
                capsize=3, label=f"DQN (equal wall clock, {BUDGET_S:.0f} s)")
    ax.errorbar(df.M, df.dqn_steps_avoidance, yerr=df.dqn_steps_avoidance_sd,
                marker="^", ls="-.", capsize=3, color="tab:green",
                label=f"DQN (equal steps, {STEP_MATCHED:,})")
    ax.plot(df.M, df.crypto_avoidance, "s--", color="tab:orange",
            label="Cryptographic hopping (training-free)")
    ax.set_xscale("log"); ax.set_xlabel("Number of channels M")
    ax.set_ylabel("Jamming-avoidance rate"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("Learned vs cryptographic hopping as channels scale")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(fontsize=8)
    fig.tight_layout()
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_scaling_crossover.png", dpi=200)
    plt.close(fig)

    # cost -- two single-axis panels. A twin-axis version of this plot cannot be
    # read without a legend telling you which curve belongs to which scale.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.9))
    axL.loglog(df.M, df.params, "o-", color="tab:blue")
    axL.set_xlabel("Number of channels M")
    axL.set_ylabel("Trainable parameters")
    axL.set_title("Model size grows as $O(M)$", fontsize=10)
    axL.grid(True, which="both", ls=":", alpha=0.5)
    for x, y in zip(df.M, df.params):
        axL.annotate(f"{int(y):,}", (x, y), textcoords="offset points",
                     xytext=(0, -13), fontsize=7, ha="center")

    axR.loglog(df.M, df.steps_per_sec, "s-", color="tab:red")
    axR.set_xlabel("Number of channels M")
    axR.set_ylabel("Training throughput (gradient steps per second)")
    axR.set_title("Training throughput collapses", fontsize=10)
    axR.grid(True, which="both", ls=":", alpha=0.5)
    for x, y in zip(df.M, df.steps_per_sec):
        axR.annotate(f"{y:,.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), fontsize=7, ha="center")
    fig.suptitle("Deep-learning training cost vs channel count", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_scaling_cost.png", dpi=200)
    plt.close(fig)

    # convergence -- one unit only (gradient steps). Plotting steps and seconds
    # on a shared axis makes the y-value ambiguous.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ok = df.dropna(subset=["steps_to_converge"])
    missed = df[df.steps_to_converge.isna()]
    if not ok.empty:
        ax.loglog(ok.M, ok.steps_to_converge, "o-", color="tab:purple",
                  label="Gradient steps to first reach 0.90 avoidance")
        for x, y in zip(ok.M, ok.steps_to_converge):
            ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), fontsize=7, ha="center")
    if not missed.empty:
        ceiling = (ok.steps_to_converge.max() if not ok.empty else 1.0)
        ax.scatter(missed.M, [ceiling * 3] * len(missed), marker="x", s=70,
                   color="tab:red", zorder=4,
                   label=f"Never reached 0.90 within {STEP_MATCHED:,} steps")
    ax.set_xlabel("Number of channels M")
    ax.set_ylabel("Gradient steps to reach 0.90 avoidance")
    ax.set_title("Training cost of a fixed competence level")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_scaling_convergence.png", dpi=200)
    plt.close(fig)
    print("  wrote figures/fig_scaling_crossover.png, fig_scaling_cost.png, "
          "fig_scaling_convergence.png")


if __name__ == "__main__":
    run()
