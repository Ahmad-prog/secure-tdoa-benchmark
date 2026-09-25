"""Significance testing helpers (paired t-test, Wilcoxon, bootstrap CI,
Cohen's d) used across the study."""
from __future__ import annotations
import numpy as np
from scipy import stats


def compare(a, b, n_boot=10000, seed=0):
    """Paired comparison of a vs b (a-b>0 means a better).  Returns a dict."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    mean_diff = float(d.mean())
    # bootstrap 95% CI of the mean difference
    g = np.random.default_rng(seed)
    boot = d[g.integers(0, n, (n_boot, n))].mean(1)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    # paired t-test
    if np.allclose(d, 0):
        t_p = 1.0
    else:
        t_p = float(stats.ttest_rel(a, b).pvalue)
    # Wilcoxon signed-rank (falls back when all-zero differences)
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        w_p = 1.0
    sd = d.std(ddof=1) if n > 1 else 0.0
    cohen_d = mean_diff / sd if sd > 0 else 0.0
    return {"n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": mean_diff, "ci_lo": ci[0], "ci_hi": ci[1],
            "t_p": t_p, "wilcoxon_p": w_p, "cohens_d": float(cohen_d)}


def bonferroni(pvals):
    """Bonferroni-corrected p-values (family-wise error control)."""
    m = len(pvals)
    return [min(1.0, p * m) for p in pvals]
