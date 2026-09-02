"""Reproduce the whole study end to end (fixed seeds).

    python -m sim.run_all            # everything
    python -m sim.run_all --quick    # skip the slow DQN sweeps

Each experiment writes its CSV(s) to sim/results/data and figures to
sim/results/figures (and paper/figures).
"""
from __future__ import annotations
import argparse, time
from .experiments import (exp_dataset, exp_tdoa, exp_antijam, exp_scaling,
                          exp_security, exp_cost, exp_sync, exp_significance)


STAGES = [
    ("dataset  (C1)", exp_dataset.run, False),
    ("tdoa     (RO1/RQ1, C1)", exp_tdoa.run, False),
    ("antijam  (RQ2, C2/C6)", exp_antijam.run, False),
    ("security (RO4, C5/C9)", exp_security.run, False),
    ("cost     (C11/P1)", exp_cost.run, False),
    ("sync     (C7)", exp_sync.run, False),
    ("scaling  (RO2, C3)", exp_scaling.run, True),
    ("signif.  (IRB stats)", exp_significance.run, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip slow DQN stages")
    args = ap.parse_args()
    for name, fn, slow in STAGES:
        if slow and args.quick:
            print(f"[skip] {name}"); continue
        print(f"\n===== {name} =====")
        t0 = time.perf_counter()
        fn()
        print(f"  ({time.perf_counter()-t0:.1f}s)")
    print("\nAll stages complete.  Results in sim/results/.")


if __name__ == "__main__":
    main()
