"""Experiment 7 -- statistical significance of the headline comparisons
(paired tests, 95% CIs, effect sizes, Bonferroni correction).

Produces: data/significance_results.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config, utils, spectrum
from ..jammers import taxonomy
from ..hopping import dqn
from ..analysis import stats


def _antijam_seeds(strategy, jammer, M, seeds):
    return np.array([spectrum.evaluate(strategy, jammer, M, config.BASE_SEED + k)[0]
                     for k in range(seeds)])


def _dqn_seeds(jammer, M, seeds, budget):
    out = []
    for k in range(seeds):
        fac = lambda s, M=M, jammer=jammer: taxonomy.make_jammer(jammer, M, seed=s + 101)
        out.append(dqn.train(fac, M, max_steps=200000, time_budget=budget,
                             seed=config.BASE_SEED + k)["final_avoidance"])
    return np.array(out)


def run(dqn_seeds=8):
    rows, pvals = [], []

    # (A) crypto vs cyber baselines vs the intelligent (learning) jammer, M=100
    M = 100
    crypto = _antijam_seeds("crypto", "learning", M, config.N_SEEDS)
    for base in ["random", "uss", "dsss"]:
        b = _antijam_seeds(base, "learning", M, config.N_SEEDS)
        r = stats.compare(crypto, b); r.update(comparison=f"crypto>{base}",
                                               jammer="learning", M=M)
        rows.append(r); pvals.append(r["t_p"])

    # (B) crypto vs DQN across the crossover (smart_partial adversary)
    for M, budget in [(10, 8.0), (100, 20.0), (1000, 20.0)]:
        d = _dqn_seeds("smart_partial", M, dqn_seeds, budget)
        c = _antijam_seeds("crypto", "smart_partial", M, dqn_seeds)
        r = stats.compare(c, d); r.update(comparison="crypto>dqn",
                                          jammer="smart_partial", M=M)
        rows.append(r); pvals.append(r["t_p"])
        print(f"   M={M:5} crypto={c.mean():.2f} dqn={d.mean():.2f} "
              f"diff={r['mean_diff']:+.2f} p={r['t_p']:.1e}")

    corr = stats.bonferroni(pvals)
    for r, pc in zip(rows, corr):
        r["t_p_bonferroni"] = pc
        r["significant_0.05"] = pc < 0.05
    df = pd.DataFrame(rows)
    utils.save_table(df, "significance_results")
    return df


if __name__ == "__main__":
    run()
