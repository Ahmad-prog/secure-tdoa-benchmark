"""Experiment 4 -- attack-success rate of spoofing and replay adversaries under
each security layer (RO4: replay defence and the signature-only ablation).

Produces: data/security_results.csv, figures/fig_security.png
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import config, utils
from ..security import auth_timing as at
from ..tdoa import noise as noisemod

# a harmful meaconing replay adds enough delay to corrupt the TDOA timing
HARMFUL_DELAY_SIGMA = 20.0        # added range-equivalent delay, in sigma units


def _spoof_success(layer, M, sigma_r, gen):
    """Injection spoof: guess the active (crypto) channel and forge auth."""
    key = os.urandom(config.HMAC_KEY_BYTES)
    if layer == "L0_none":
        return 1.0                                     # fixed channel, no auth
    # crypto hopping: attacker must land on the active channel
    guessed = int(gen.integers(M)); active = int(gen.integers(M))
    if guessed != active:
        return 0.0
    if layer == "L1_crypto":
        return 1.0                                     # right channel, no auth needed
    # L2/L3 require a valid HMAC the attacker cannot forge without the key
    forged = os.urandom(32)
    pkt = {"msg": b"spoofed", "tag": forged,
           "nonce": os.urandom(16), "t_tx": 0.0, "bid": 0}
    nonce = os.urandom(16)
    return 1.0 if at.verify(pkt, key, 0.0, 0.0, nonce, sigma_r, layer) else 0.0


def _replay_success(layer, M, sigma_r, gen):
    """Meaconing: capture a genuine packet, re-transmit with an added delay."""
    key = os.urandom(config.HMAC_KEY_BYTES)
    nonce_issued = os.urandom(16)                      # drone's fresh nonce this round
    beacon = at.Beacon(0, key)
    geo_delay = 1200.0 / config.C_LIGHT                # ~1200 m range
    t_tx = 0.0
    # genuine packet the attacker captured earlier used an OLD nonce
    old_nonce = os.urandom(16)
    pkt = beacon.emit(t_tx, old_nonce)
    added = HARMFUL_DELAY_SIGMA * sigma_r / config.C_LIGHT
    arrival = t_tx + geo_delay + added                 # replay shifts arrival late
    accepted = at.verify(pkt, key, arrival, geo_delay, nonce_issued, sigma_r, layer)
    # 'success' = accepted AND actually corrupts timing (harmful)
    harmful = (added * config.C_LIGHT) > at.TOF_TOL_SIGMA * sigma_r
    return 1.0 if (accepted and harmful) else 0.0


def run(scales=config.CHANNEL_SCALES, n=2000):
    sigma_r = noisemod.sigma_range(config.SNR_DB_DEFAULT)
    rows = []
    for M in scales:
        gen = utils.rng(config.BASE_SEED + M)
        for layer in at.LAYERS:
            sp = np.mean([_spoof_success(layer, M, sigma_r, gen) for _ in range(n)])
            rp = np.mean([_replay_success(layer, M, sigma_r, gen) for _ in range(n)])
            rows.append({"M": M, "layer": layer,
                         "spoof_success": sp, "replay_success": rp})
    df = pd.DataFrame(rows)
    utils.save_table(df, "security_results")
    _plot(df)
    _summary(df)
    return df


def _plot(df, M=config.CHANNEL_SCALES[1]):
    sub = df[df.M == M]
    x = np.arange(len(at.LAYERS)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    b1 = ax.bar(x - w / 2, sub.spoof_success, w, label="Spoofing (forged beacon)")
    b2 = ax.bar(x + w / 2, sub.replay_success, w,
                label="Replay / meaconing (delayed re-transmission)")
    # A 0.00 bar has no height, so label every bar: otherwise "defeated" and
    # "not measured" look identical.
    ax.bar_label(b1, fmt="%.2f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels([l.replace("_", "\n") for l in at.LAYERS])
    ax.set_xlabel("Security layer applied (cumulative)")
    ax.set_ylabel("Attack-success rate")
    ax.set_ylim(0, 1.40)                      # headroom so the legend clears the bars
    ax.set_title(f"Attack success by security layer (M={M} channels)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=8,
              framealpha=0.95)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    # Placed in the empty upper-right quadrant: the bars occupy the left three
    # groups up to y=1.0 and the legend sits above y=1.25.
    ax.annotate("a valid signature\nstill loses to replay",
                xy=(2 + w / 2, 1.04), xytext=(2.52, 1.09), fontsize=8,
                color="crimson", ha="left",
                arrowprops=dict(arrowstyle="->", color="crimson"))
    fig.tight_layout()
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_security.png", dpi=200)
    plt.close(fig)
    print("  wrote figures/fig_security.png")


def _summary(df):
    print("\n  --- attack-success rate (M=100) ---")
    sub = df[df.M == 100]
    for _, r in sub.iterrows():
        print(f"   {r.layer:16} spoof={r.spoof_success:.3f}  replay={r.replay_success:.3f}")


if __name__ == "__main__":
    run()
