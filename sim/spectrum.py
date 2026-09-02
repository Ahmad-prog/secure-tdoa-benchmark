"""Spectrum access environment: run one hopper against one jammer and score
jamming-avoidance rate and normalized throughput.  Also builds the labelled
spectrum-occupancy dataset consumed by the deep-learning model."""
from __future__ import annotations
import numpy as np
from . import config
from .jammers import taxonomy
from .hopping import strategies as strat


def run_episode(strategy, jammer, slots=config.SLOTS_PER_EPISODE):
    if hasattr(jammer, "reset"):
        jammer.reset()
    strategy.reset()
    history, avoided, tput = [], 0.0, 0.0
    for t in range(slots):
        jammed = jammer.step(t, history)
        ch, ok, tp = strategy.step(t, jammed)
        avoided += float(ok); tput += tp
        history.append(ch)
    return avoided / slots, tput / slots


def evaluate(strategy_name, jammer_name, M, seed):
    """One seeded episode: build strategy + jammer (wiring the learning
    jammer's oracle to public strategies) and return (avoidance, throughput)."""
    s = strat.make_strategy(strategy_name, M, seed=seed)
    oracle = None
    if jammer_name == "learning" and strategy_name in ("fixed", "random"):
        oracle = s.peek                                  # adversary predicts public hop
    j = taxonomy.make_jammer(jammer_name, M, seed=seed + 101, oracle=oracle)
    return run_episode(s, j)


def occupancy_matrix(jammer_name, M, slots, seed):
    """Per-slot, per-channel jammed/clean label matrix (the DL input dataset).

    A neutral uniform hopper drives any adaptive jammer so the trace reflects a
    realistic transmitter interaction (proposal 4.3)."""
    j = taxonomy.make_jammer(jammer_name, M, seed=seed)
    s = strat.RandomHop(M, seed=seed + 1)
    if hasattr(j, "reset"):
        j.reset()
    hist = []
    occ = np.zeros((slots, M), dtype=np.uint8)
    for t in range(slots):
        mask = j.step(t, hist)
        occ[t] = mask.astype(np.uint8)
        hist.append(s.channel(t))
    return occ
