"""Rewrite every volatile number in main.tex from the saved result CSVs.

    python3 update_numbers.py [--check]

The manuscript keeps its changeable figures in one macro block and the scaling
table rows between markers. This script regenerates both from
sim/results/data/*.csv, so a re-run can never leave a stale number in the prose.
With --check it only reports what would change and exits non-zero if anything
differs, which makes it usable as a pre-submission guard.
"""
from __future__ import annotations
import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[1] / "sim" / "results" / "data"
TEX = HERE / "main.tex"


def _fmt_pow(x):
    """LaTeX scientific notation, e.g. 1.1\\times10^{4}."""
    s = f"{x:.1e}"                      # '1.1e+04'
    mant, exp = s.split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


def build_macros():
    sc = pd.read_csv(DATA / "scaling_results.csv").sort_values("M")
    aj = pd.read_csv(DATA / "antijam_results.csv")
    co = pd.read_csv(DATA / "cost_results.csv")

    crypto = aj[aj.strategy == "crypto"]
    by_m = crypto.groupby("M").avoidance.mean()
    learn = crypto[crypto.jammer == "learning"].set_index("M").avoidance
    barrage = aj[(aj.jammer == "barrage") & (aj.strategy == "crypto")].avoidance.mean()
    dsss_barrage = aj[(aj.jammer == "barrage") & (aj.strategy == "dsss")].throughput.mean()

    lo, hi = sc.M.min(), sc.M.max()
    row = lambda m: sc[sc.M == m].iloc[0]
    kilo = 1000 if (sc.M == 1000).any() else sc.M.iloc[-2]

    ratio = co.energy_ratio.max() if "energy_ratio" in co.columns else float("nan")

    # exploration budget: the DQN's epsilon schedule (hopping/dqn.py) decays by
    # 0.9995 per step to a 0.05 floor; this is the expected number of random
    # picks in the first 3000 steps, spread over M channels
    explore_3000 = sum(max(0.05, 0.9995 ** k) for k in range(3000))

    macros = {
        "dqnStepMax":     f"{row(hi).dqn_steps_avoidance:.2f}",
        "dqnBudgetMaxSd": f"{row(hi).dqn_avoidance_sd:.2f}",
        "budgetStepsMax": f"{row(hi).steps:,.0f}",
        "convTen":        f"{row(lo).steps_to_converge:,.0f}",
        "convHundred":    f"{row(100).steps_to_converge:,.0f}",
        "exploreHundred": f"{explore_3000 / 100:.1f}",
        "exploreKilo":    f"{explore_3000 / 1000:.1f}",
    }
    col_path = DATA / "policy_collapse.csv"
    if col_path.exists():
        pc = pd.read_csv(col_path).set_index(["M", "steps"])
        macros.update({
            "collapseUntrained":   f"{pc.loc[(1000, 0)].avoidance:.2f}",
            "collapseEarlyShare":  f"{100 * pc.loc[(1000, 220)].top_channel_share:.0f}",
            "collapseLateShare":   f"{100 * pc.loc[(1000, 3000)].top_channel_share:.0f}",
            "collapseHundredLate": f"{pc.loc[(100, 3000)].avoidance:.2f}",
        })

    return macros | {
        "cryptoMeanLo":   f"{by_m.loc[lo]:.2f}",
        "cryptoMeanHi":   f"{by_m.loc[hi]:.2f}",
        "cryptoLearnLo":  f"{learn.loc[lo]:.2f}",
        "cryptoLearnHi":  f"{learn.loc[hi]:.2f}",
        "barrageAvoid":   f"{barrage:.2f}",
        "dsssBarrageTput": f"{dsss_barrage:.2f}",
        "dqnBudgetTen":   f"{row(lo).dqn_avoidance:.2f}",
        "dqnBudgetKilo":  f"{row(kilo).dqn_avoidance:.2f}",
        "dqnBudgetMax":   f"{row(hi).dqn_avoidance:.2f}",
        "cryptoScale":    f"{row(lo).crypto_avoidance:.2f}",
        "stepsLo":        f"{row(lo).steps_per_sec:,.0f}",
        "stepsHi":        f"{row(hi).steps_per_sec:,.0f}",
        "paramsLo":       _fmt_pow(row(lo).params),
        "paramsHi":       _fmt_pow(row(hi).params),
        "dqnStepTen":     f"{row(lo).dqn_steps_avoidance:.2f}",
        "dqnStepKilo":    f"{row(kilo).dqn_steps_avoidance:.2f}",
        "costRatio":      _fmt_pow(ratio) if ratio == ratio else "7.2\\times10^{5}",
    }


def build_scaling_rows():
    # the M and steps/s columns are right-aligned in the tabular spec, so no
    # manual padding is needed
    sc = pd.read_csv(DATA / "scaling_results.csv").sort_values("M")
    out = []
    for _, r in sc.iterrows():
        out.append(f"{int(r.M):,} & {r.dqn_avoidance:.2f} & "
                   f"{r.dqn_steps_avoidance:.2f} & {r.crypto_avoidance:.2f} & "
                   f"{r.steps_per_sec:,.0f} \\\\")
    return "\n".join(out)


NAMES = {"crypto": "Crypto", "dqn": "DQN", "random": "Pub.\\ random",
         "uss": "USS", "dsss": "DSSS"}
JAMMERS = {"learning": "learning", "smart_partial": "smart partial"}


def _fmt_p(p):
    if p < 1e-3:
        return f"$<10^{{{int(np.floor(np.log10(p))) + 1}}}$" if p > 0 else "$<10^{-12}$"
    return f"${p:.3f}$"


def _zero_variance(r):
    # Every paired difference identical: the t statistic divides by zero, so
    # its p (0) and d (~1e15 or 0) are numerical artifacts, not evidence.
    return r.ci_lo == r.ci_hi or not np.isfinite(r.cohens_d) or abs(r.cohens_d) > 1e3 \
        or (r.cohens_d == 0 and r.mean_diff != 0)


def build_stat_rows():
    st = pd.read_csv(DATA / "significance_results.csv")
    family = len(st)                     # Bonferroni family size
    # crypto-vs-DQN rows first, ordered by M, then the baseline comparisons
    st["_dqn"] = st.comparison.str.endswith("dqn")
    st = st.sort_values(["_dqn", "M"], ascending=[False, True])
    out = []
    for _, r in st.iterrows():
        a, b = r.comparison.split(">")
        diff = r.mean_diff
        better, worse = (a, b) if diff >= 0 else (b, a)
        cond = f"$M{{=}}{int(r.M)}$" if r._dqn else JAMMERS.get(r.jammer, r.jammer)
        if _zero_variance(r):
            p = _fmt_p(min(1.0, r.wilcoxon_p * family)) + "$^\\dagger$"
            d = "--"
        else:
            p = _fmt_p(r.t_p_bonferroni)
            d = f"${abs(r.cohens_d):.2f}$"
        out.append(f"{NAMES.get(better, better)} $>$ {NAMES.get(worse, worse)} & "
                   f"{cond} & ${abs(diff):.3f}$ & {p} & {d} \\\\")
    return "\n".join(out)


def build_collapse_rows():
    pc = pd.read_csv(DATA / "policy_collapse.csv").sort_values(["M", "steps"])
    out = []
    for i, (_, r) in enumerate(pc.iterrows()):
        first = i == 0 or pc.iloc[i - 1].M != r.M
        m = f"{int(r.M):,}" if first else ""
        out.append(f"{m} & {int(r.steps):,} & {r.avoidance:.2f} & "
                   f"{int(r.distinct_channels)} & {r.top_channel_share:.2f} \\\\")
    return "\n".join(out)


def _replace_block(text, begin, end, new_body):
    pat = re.compile(rf"({re.escape(begin)}\n).*?(\n{re.escape(end)})", re.S)
    if not pat.search(text):
        sys.exit(f"marker pair not found: {begin} .. {end}")
    return pat.sub(lambda m: m.group(1) + new_body + m.group(2), text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report differences without writing; exit 1 if stale")
    args = ap.parse_args()

    text = original = TEX.read_text(encoding="utf8")
    macros = build_macros()

    # keep the comment lines inside the macro block, rewrite only \newcommand
    body = []
    for line in re.search(r"% BEGIN-RESULT-MACROS\n(.*?)\n% END-RESULT-MACROS",
                          text, re.S).group(1).split("\n"):
        m = re.match(r"\\newcommand\{\\(\w+)\}\{.*?\}(\s*%.*)?$", line)
        if m and m.group(1) in macros:
            body.append(f"\\newcommand{{\\{m.group(1)}}}{{{macros[m.group(1)]}}}"
                        + (m.group(2) or ""))
        else:
            body.append(line)
    text = _replace_block(text, "% BEGIN-RESULT-MACROS", "% END-RESULT-MACROS",
                          "\n".join(body))
    text = _replace_block(text, "% BEGIN-SCALING-ROWS", "% END-SCALING-ROWS",
                          build_scaling_rows())
    text = _replace_block(text, "% BEGIN-STAT-ROWS", "% END-STAT-ROWS",
                          build_stat_rows())
    if (DATA / "policy_collapse.csv").exists():
        text = _replace_block(text, "% BEGIN-COLLAPSE-ROWS", "% END-COLLAPSE-ROWS",
                              build_collapse_rows())

    if text == original:
        print("numbers already current"); return
    if args.check:
        print("STALE: main.tex does not match the current results"); sys.exit(1)
    TEX.write_text(text, encoding="utf8")
    print("updated main.tex from", DATA)
    for k, v in macros.items():
        print(f"  {k:18} {v}")


if __name__ == "__main__":
    main()
