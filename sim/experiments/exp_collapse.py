"""Experiment 8 -- why the learned hopper fails at large M: policy collapse.

Trains the DQN against the adaptive smart-partial jammer and, at fixed step
counts, probes the GREEDY policy it would deploy: its avoidance rate, how many
distinct channels it uses over an episode, and the share of picks going to its
single most-used channel.

Finding: at M=100 training diversifies the policy and avoidance rises; at
M=1000 training collapses it onto a single channel, which the tracking jammer
then covers permanently. The untrained network, whose argmax shifts with the
input, is effectively a pseudo-random hopper and avoids about as well as
cryptographic hopping -- training removes that unpredictability before it has
seen enough samples per channel to learn anything better. (Only the chosen
action's value is updated, so each of M channels needs its own samples.)

Produces: data/policy_collapse.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config, utils
from ..hopping.dqn import DQNAgent
from ..jammers import taxonomy

JAMMER = "smart_partial"
SCALES = [100, 1000]
CHECKPOINTS = [0, 220, 3000]      # untrained / equal-time budget at M=1000 / longer


def _probe(agent, M, seed):
    """Deploy the greedy policy for one episode against a fresh jammer."""
    jam = taxonomy.make_jammer(JAMMER, M, seed=seed + 777 + 101)
    jam.reset()
    window = np.zeros((agent.W, M), np.uint8)
    hist, picks, clean = [], [], 0
    for t in range(config.SLOTS_PER_EPISODE):
        a = int(np.argmax(agent.q(window.ravel())))
        jammed = jam.step(t, hist)
        clean += 0 if jammed[a] else 1
        hist.append(a); picks.append(a)
        window = np.roll(window, -1, 0); window[-1] = jammed.astype(np.uint8)
    picks = np.asarray(picks)
    return {"avoidance": clean / len(picks),
            "distinct_channels": int(len(np.unique(picks))),
            "top_channel_share": float(np.bincount(picks, minlength=M).max() / len(picks))}


def _trace(M, seed):
    """Same training loop as hopping.dqn.train, pausing at each checkpoint."""
    agent = DQNAgent(M, seed=seed)
    jam = taxonomy.make_jammer(JAMMER, M, seed=seed + 101)
    eps, step, rows = 1.0, 0, []
    todo = list(CHECKPOINTS)
    if todo[0] == 0:
        rows.append({"M": M, "steps": 0, "epsilon": eps, **_probe(agent, M, seed)})
        todo.pop(0)
    while todo:
        jam.reset()
        window = np.zeros((agent.W, M), np.uint8); hist = []
        for t in range(config.SLOTS_PER_EPISODE):
            s = window.ravel(); a = agent.act(s, eps)
            jammed = jam.step(t, hist); r = 0.0 if jammed[a] else 1.0
            agent.store(s.copy(), a, r); agent.train_step()
            hist.append(a)
            window = np.roll(window, -1, 0); window[-1] = jammed.astype(np.uint8)
            eps = max(0.05, eps * 0.9995); step += 1
            if todo and step == todo[0]:
                rows.append({"M": M, "steps": step, "epsilon": eps,
                             **_probe(agent, M, seed)})
                todo.pop(0)
                if not todo:
                    break
    return rows


def run(seed=config.BASE_SEED):
    rows = []
    for M in SCALES:
        rows += _trace(M, seed)
        for r in rows[-len(CHECKPOINTS):]:
            print(f"   M={M:5} steps={r['steps']:5} avoid={r['avoidance']:.2f} "
                  f"distinct={r['distinct_channels']:3} top-share={r['top_channel_share']:.2f}",
                  flush=True)
    df = pd.DataFrame(rows)
    utils.save_table(df, "policy_collapse")
    return df


if __name__ == "__main__":
    run()
