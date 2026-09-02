"""Shared helpers: seeding, timing, and results I/O (CSV/JSON)."""
from __future__ import annotations
import json, time, contextlib
import numpy as np
import pandas as pd
from . import config


def rng(seed: int) -> np.random.Generator:
    """A fresh, independent PCG64 generator for a given integer seed."""
    return np.random.default_rng(seed)


@contextlib.contextmanager
def timer():
    """with timer() as t: ...  ; then t() -> elapsed wall-clock seconds."""
    t0 = time.perf_counter()
    box = {}
    yield lambda: box.get("dt", time.perf_counter() - t0)
    box["dt"] = time.perf_counter() - t0


def save_table(df: pd.DataFrame, name: str) -> None:
    path = config.DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")


def save_json(obj, name: str) -> None:
    path = config.DATA_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  wrote {path}")


def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(config.DATA_DIR / f"{name}.csv")
