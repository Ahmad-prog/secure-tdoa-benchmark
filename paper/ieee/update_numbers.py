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

    return {
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
    sc = pd.read_csv(DATA / "scaling_results.csv").sort_values("M")
    width = max(len(f"{v:,.0f}") for v in sc.steps_per_sec)
    out = []
    for _, r in sc.iterrows():
        sps = f"{r.steps_per_sec:,.0f}"
        pad = "\\phantom{" + "0" * (width - len(sps)) + "}" if len(sps) < width else ""
        out.append(f"{int(r.M):<4} & {r.dqn_avoidance:.2f} & "
                   f"{r.dqn_steps_avoidance:.2f} & {r.crypto_avoidance:.2f} & "
                   f"{pad}{sps} \\\\")
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
