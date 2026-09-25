# Secure GPS-Denied TDOA Positioning — Reproducible Benchmark

Code, datasets, and paper for **"Benchmarking Deep-Reinforcement-Learning
Anti-Jamming Against Cryptographic Frequency Hopping for GPS-Denied Drone
Positioning: A Reproducible Empirical Study."**

A drone that has lost GPS can still locate itself by timing signals from ground
beacons (TDOA multilateration) — but only if the radio link survives jamming,
spoofing, and replay. This benchmark puts two defences head-to-head under one
simulator: a **learned** deep-Q anti-jamming model (reproduced from the
literature) versus **training-free cryptographic frequency hopping**, and adds
an authenticated-timing layer that secures the position estimate itself.

Everything runs on CPU with fixed seeds — no GPU, no PyTorch (the deep-Q agent
is a compact NumPy network, which makes its O(M) scaling cost a measured result
rather than an assertion).

## Reproduce

```bash
pip install -r requirements.txt
python -m sim.run_all            # full study (several hours; the
                                 # step-matched deep-learning arm dominates)
python -m sim.run_all --quick    # skip the slow deep-learning sweeps (~15 min)
```

Outputs land in `sim/results/`: CSVs, the released
`spectrum_occupancy_dataset.npz`, and all figures.

## Headline findings (fixed-seed, reproducible)

| Question | Result |
|---|---|
| TDOA accuracy | Maximum-likelihood estimator **reaches the Cramér–Rao bound** in line-of-sight from 10 dB upward; RANSAC cuts non-line-of-sight median error up to **20×** |
| Learned vs cryptographic | Deep-Q **beats** crypto at 10–100 channels but **fails** at 1000–2320: **0.00** avoidance even with 15,000 training updates, vs **0.70** for crypto, which needs no training |
| Why it fails | At 1000 channels training **collapses the policy onto one channel**, which an adaptive jammer tracks; the untrained network avoids 0.72 of slots (`sim/experiments/exp_collapse.py`) |
| Intelligent adversary | Crypto avoidance ≈ **1.0**; predictable (keyless) hopping drops to **0.0** |
| Replay/meaconing | A valid signature still loses (**1.0**); authenticated timing (HMAC + nonce + time-of-flight) rejects every simulated spoof and replay, **provided the drone can predict each beacon's time of flight to within ~6 ns** — the simulation supplies it, a deployment must |
| Cost | Crypto **13 pJ**/decision vs deep-learning up to **9.3 µJ** (≈ 7×10⁵×), plus training |

## Layout

- `sim/` — simulator: `tdoa/`, `jammers/`, `hopping/`, `security/`, `analysis/`, `experiments/`, `run_all.py`
- `sim/results/` — CSV results, released dataset (`.npz`), figures
- `paper/` — LaTeX source, references, and the compiled PDF

## Paper

`paper/ieee/main.pdf` is the current IEEE two-column manuscript; build it with
`pdflatex main.tex` inside `paper/ieee/`. Its result numbers are generated from
`sim/results/data/*.csv` by `paper/ieee/update_numbers.py`, and
`python3 update_numbers.py --check` fails if the manuscript is out of date.
`paper/main.pdf` is the earlier single-column preprint and is superseded.

## Citation

```
M. Ahmad and G. Mustafa, "Benchmarking Deep-Reinforcement-Learning Anti-Jamming
Against Cryptographic Frequency Hopping for GPS-Denied Drone Positioning:
A Reproducible Empirical Study," 2026.
```

## License

MIT — see `LICENSE`.
