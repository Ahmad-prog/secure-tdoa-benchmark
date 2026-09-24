"""Experiment 6 -- synchronization drift tolerance for cryptographic de-hopping
(RO3: how hop synchronization is ensured and what drift it tolerates).

Correct de-hopping needs beacon and drone to share the hop index.  If the clock
offset (plus jitter) exceeds a guard fraction of the hop dwell, they land on
different channels and de-hop fails.  We sweep the offset and report the
maximum tolerable offset and the resync latency for several hop rates.

Produces: data/sync_results.csv, figures/fig_sync.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .. import config, utils

HOP_RATES = [100, 1000, 10000]     # hops/s
GUARD_FRAC = 0.5                    # de-hop ok while |offset| < guard*dwell
JITTER_FRAC = 0.05                  # timing jitter as a fraction of dwell
RESYNC_SLOTS = 8                    # late-entry acquisition length


def run():
    gen = utils.rng(config.BASE_SEED)
    rows = []
    for hr in HOP_RATES:
        dwell = 1.0 / hr                              # s
        jitter = JITTER_FRAC * dwell
        guard = GUARD_FRAC * dwell
        offsets = np.linspace(0, 1.2 * dwell, 60)
        succ = []
        for off in offsets:
            draws = off + gen.normal(0, jitter, 4000)
            succ.append(float(np.mean(np.abs(draws) < guard)))
        succ = np.array(succ)
        # max tolerable offset: largest offset with >=99% de-hop success
        ok = offsets[succ >= 0.99]
        max_tol = float(ok.max()) if ok.size else 0.0
        for off, s in zip(offsets, succ):
            rows.append({"hop_rate": hr, "dwell_us": dwell * 1e6,
                         "offset_us": off * 1e6, "success": s})
        print(f"   {hr:6} hops/s: dwell {dwell*1e6:7.1f} us | "
              f"max tolerable offset {max_tol*1e6:7.2f} us | "
              f"resync latency {RESYNC_SLOTS*dwell*1e3:.2f} ms")
    df = pd.DataFrame(rows)
    utils.save_table(df, "sync_results")
    _plot(df)
    return df


def _plot(df):
    # Plotted against ABSOLUTE offset. Normalising by dwell collapses every hop
    # rate onto one curve, which hides the point: the faster the hopping, the
    # tighter the clock requirement in real time.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for hr in HOP_RATES:
        sub = df[(df.hop_rate == hr) & (df.offset_us > 0)]
        dwell = float(sub.dwell_us.iloc[0])
        ax.semilogx(sub.offset_us, sub.success, "-o", ms=3.5,
                    label=f"{hr:,} hops/s (dwell {dwell:,.0f} $\\mu$s)")
    ax.axhline(0.99, ls=":", color="k", alpha=0.6)
    ax.text(ax.get_xlim()[0] * 1.1, 0.965, "99% de-hop success",
            fontsize=8, color="0.3")
    ax.set_xlabel("Clock offset between beacon and drone ($\\mu$s, log scale)")
    ax.set_ylabel("De-hop success rate")
    ax.set_ylim(-0.05, 1.12)
    ax.set_title("Hop-synchronization drift tolerance by hop rate")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    for d in (config.FIG_DIR, config.PAPER_FIG_DIR):
        fig.savefig(d / "fig_sync.png", dpi=200)
    plt.close(fig)
    print("  wrote figures/fig_sync.png")


if __name__ == "__main__":
    run()
