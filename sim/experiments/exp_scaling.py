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
from ..plotstyle import COL_W, PAGE_W
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
    def _bars(mean, sd):
        # avoidance is a rate in [0, 1]; a symmetric +-sd bar would run past it
        return [np.minimum(sd, mean), np.minimum(sd, 1.0 - mean)]

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.errorbar(df.M, df.dqn_avoidance,
                yerr=_bars(df.dqn_avoidance, df.dqn_avoidance_sd), marker="o",
                capsize=2, label=f"DQN, equal time ({BUDGET_S:.0f} s)")
    ax.errorbar(df.M, df.dqn_steps_avoidance,
                yerr=_bars(df.dqn_steps_avoidance, df.dqn_steps_avoidance_sd),
                marker="^", ls="-.", capsize=2, color="tab:green",
                label=f"DQN, equal steps ({STEP_MATCHED:,})")
    ax.plot(df.M, df.crypto_avoidance, "s--", color="tab:orange",
            label="Crypto hopping (no training)")
    ax.set_xscale("log"); ax.set_xlabel("Number of channels $M$")
    ax.set_ylabel("Jamming-avoidance rate"); ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    # lower left is the one region no line or error bar passes through
    ax.legend(loc="lower left")
    fig.tight_layout()
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_scaling_crossover.png")
    plt.close(fig)

    # cost -- two single-axis panels. A twin-axis version of this plot cannot be
    # read without a legend telling you which curve belongs to which scale.
    # Point labels go on the side of the marker the line does NOT leave from:
    # below-right of a rising curve, above-right of a falling one. The last
    # point of a rising curve flips to above-left to stay inside the axes.
    def _label_points(ax, xs, ys, fmt, rising):
        last = len(xs) - 1
        for i, (x, y) in enumerate(zip(xs, ys)):
            if rising and i == last:
                off, ha, va = (-5, 4), "right", "bottom"
            elif rising:
                off, ha, va = (5, -3), "left", "top"
            else:
                off, ha, va = (5, 3), "left", "bottom"
            ax.annotate(fmt(y), (x, y), textcoords="offset points", xytext=off,
                        ha=ha, va=va, fontsize=6.5)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(0.8 * PAGE_W, 2.3))  # 0.8 textwidth
    axL.loglog(df.M, df.params, "o-", color="tab:blue")
    axL.set_xlabel("Number of channels $M$")
    axL.set_ylabel("Trainable parameters")
    axL.set_title("Model size grows as $O(M)$")
    axL.grid(True, which="both", ls=":", alpha=0.5)
    _label_points(axL, df.M, df.params, lambda v: f"{int(v):,}", rising=True)

    axR.loglog(df.M, df.steps_per_sec, "s-", color="tab:red")
    axR.set_xlabel("Number of channels $M$")
    axR.set_ylabel("Gradient steps per second")
    axR.set_title("Training throughput collapses")
    axR.grid(True, which="both", ls=":", alpha=0.5)
    _label_points(axR, df.M, df.steps_per_sec, lambda v: f"{v:,.0f}", rising=False)
    for a in (axL, axR):
        a.margins(x=0.22, y=0.15)             # room for the point labels
    fig.tight_layout(w_pad=1.5)
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_scaling_cost.png")
    plt.close(fig)

    # convergence -- one unit only (gradient steps). Plotting steps and seconds
    # on a shared axis makes the y-value ambiguous.
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ok = df.dropna(subset=["steps_to_converge"])
    missed = df[df.steps_to_converge.isna()]
    if not ok.empty:
        ax.loglog(ok.M, ok.steps_to_converge, "o-", color="tab:purple",
                  label="Steps to first reach 0.90 avoidance")
        for x, y in zip(ok.M, ok.steps_to_converge):
            ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points",
                        xytext=(6, -3), ha="left", fontsize=6.5)
    # The training limit is drawn as a line, and runs that never converged sit
    # ON it: they were cut off there, so plotting them anywhere else would
    # invent a value.
    ax.axhline(STEP_MATCHED, ls="--", color="0.45", lw=0.9,
               label=f"Training limit ({STEP_MATCHED:,} steps)")
    if not missed.empty:
        ax.scatter(missed.M, [STEP_MATCHED] * len(missed), marker="x", s=40,
                   color="tab:red", zorder=4, linewidths=1.5,
                   label="Limit reached, 0.90 never reached")
    lo = ok.steps_to_converge.min() if not ok.empty else STEP_MATCHED / 10
    ax.set_ylim(lo / 1.6, STEP_MATCHED * 3.2)    # headroom above the limit for the legend
    ax.set_xlim(df.M.min() / 1.6, df.M.max() * 1.6)
    ax.set_xlabel("Number of channels $M$")
    ax.set_ylabel("Gradient steps to 0.90 avoidance")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_scaling_convergence.png")
    plt.close(fig)
    print("  wrote figures/fig_scaling_crossover.png, fig_scaling_cost.png, "
          "fig_scaling_convergence.png")

if __name__ == "__main__":
    run()
