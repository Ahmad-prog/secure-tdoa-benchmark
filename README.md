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
| TDOA accuracy | Maximum-likelihood estimator **reaches the Cramér–Rao bound** in line-of-sight; RANSAC cuts non-line-of-sight median error up to **20×** |
| Learned vs cryptographic | Deep-Q **beats** crypto at 10–100 channels but **collapses** at 1000–2320 (0.02 vs 0.70) under a fixed compute budget; crypto is unaffected |
| Intelligent adversary | Crypto avoidance ≈ **1.0**; predictable (keyless) hopping drops to **0.0** |
| Replay/meaconing | A valid signature still loses (**1.0**); authenticated timing (HMAC + nonce + time-of-flight) drives spoof and replay to **0.0** |
| Cost | Crypto **13 pJ**/decision vs deep-learning up to **9.3 µJ** (≈ 7×10⁵×), plus training |

## Layout

- `sim/` — simulator: `tdoa/`, `jammers/`, `hopping/`, `security/`, `analysis/`, `experiments/`, `run_all.py`
- `sim/results/` — CSV results, released dataset (`.npz`), figures
- `paper/` — LaTeX source, references, and the compiled PDF

## Paper

`paper/main.pdf` (single-column preprint). Build with `tectonic paper/main.tex`
(or upload `paper/` to Overleaf, compiler pdfLaTeX).

## Citation

```
M. Ahmad and G. Mustafa, "Benchmarking Deep-Reinforcement-Learning Anti-Jamming
Against Cryptographic Frequency Hopping for GPS-Denied Drone Positioning:
A Reproducible Empirical Study," 2026.
```

## License

MIT — see `LICENSE`.
