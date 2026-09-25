"""Experiment 5 -- per-decision compute and energy: cryptographic hopping vs
deep-learning inference (RO3: the cost and energy limits of both approaches).

Measures AES-block throughput and NN forward-pass cost empirically, converts to
energy with standard per-operation figures, and tabulates the gap.

Produces: data/cost_results.csv, figures/fig_cost.png
"""
from __future__ import annotations
import time, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .. import config, utils
from ..plotstyle import COL_W, PAGE_W

E_MAC = 3.7e-12                   # J per multiply-accumulate (Horowitz, 45 nm class)
E_AES_BLOCK_HW = 1.3e-11          # J per AES-128 block on hardware AES (~0.8 pJ/byte)
HIDDEN = 64
WINDOW = config.WINDOW


def _aes_throughput(n=200_000):
    key = os.urandom(16)
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    block = os.urandom(16)
    t0 = time.perf_counter()
    for _ in range(n):
        enc.update(block)
    dt = time.perf_counter() - t0
    return n / dt                 # blocks per second (library, single core)


def _nn_forward_time(M, reps=200):
    x = np.zeros(WINDOW * M, np.float32)
    W1 = np.zeros((WINDOW * M, HIDDEN), np.float32)
    b1 = np.zeros(HIDDEN, np.float32)
    W2 = np.zeros((HIDDEN, M), np.float32)
    b2 = np.zeros(M, np.float32)
    t0 = time.perf_counter()
    for _ in range(reps):
        a1 = np.maximum(x @ W1 + b1, 0.0)
        _ = a1 @ W2 + b2
    return (time.perf_counter() - t0) / reps


def run(scales=config.CHANNEL_SCALES):
    aes_bps = _aes_throughput()
    rows = []
    for M in scales:
        macs = HIDDEN * M * (WINDOW + 1)               # forward-pass MACs
        nn_t = _nn_forward_time(M)
        rows.append({
            "M": M,
            "crypto_ops": 1,                            # one AES block per hop
            "crypto_energy_J": E_AES_BLOCK_HW,
            "crypto_time_s": 1.0 / aes_bps,
            "dqn_macs": macs,
            "dqn_energy_J": macs * E_MAC,
            "dqn_time_s": nn_t,
            "energy_ratio": macs * E_MAC / E_AES_BLOCK_HW,
        })
    df = pd.DataFrame(rows)
    df.attrs["aes_blocks_per_sec"] = aes_bps
    utils.save_table(df, "cost_results")
    _plot(df)
    _summary(df, aes_bps)
    return df


def _plot(df):
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.loglog(df.M, df.dqn_energy_J * 1e6, "o-", label="DQN inference")
    ax.axhline(df.crypto_energy_J.iloc[0] * 1e6, ls="--", color="tab:green",
               label="Cryptographic hop (AES block)")
    ax.set_xlabel("Number of channels $M$")
    ax.set_ylabel("Energy per decision ($\\mu$J)")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(loc="upper left")
    fig.tight_layout()
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_cost.png")
    plt.close(fig)
    print("  wrote figures/fig_cost.png")


def _summary(df, aes_bps):
    print(f"\n  --- cost summary (AES lib throughput {aes_bps/1e6:.1f} Mblock/s) ---")
    for _, r in df.iterrows():
        print(f"   M={r.M:5}: crypto {r.crypto_energy_J*1e12:6.1f} pJ | "
              f"DQN {r.dqn_energy_J*1e6:8.3f} uJ ({int(r.dqn_macs):>9} MACs) | "
              f"ratio {r.energy_ratio:,.0f}x")


if __name__ == "__main__":
    run()
