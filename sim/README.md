# Secure GPS-Denied TDOA Positioning — Simulation Study

Self-contained, CPU-only, reproducible simulation for *"Benchmarking
Deep-Reinforcement-Learning Anti-Jamming Against Cryptographic Frequency Hopping
for GPS-Denied Drone Positioning."*  No GPU, no PyTorch — the deep-Q agent is a
compact NumPy implementation, which also makes its O(M) scaling cost the object
of study rather than an assertion.

## Reproduce

```bash
python -m sim.run_all            # full study (several hours; the step-matched
                                 # deep-learning arm at M=2320 dominates)
python -m sim.run_all --quick    # skip the slow DQN sweeps (~15 min)
```

Outputs: `results/data/*.csv`, `results/data/spectrum_occupancy_dataset.npz`,
`results/figures/*.png` (also copied to `../paper/figures`).

## Layout

| Module | Purpose |
|---|---|
| `tdoa/` | geometry, SNR→range-noise, CRLB, Chan-Ho + Taylor, RANSAC NLOS mitigation, GCC-PHAT |
| `jammers/taxonomy.py` | ten-class, three-tier jammer taxonomy (narrowband through wideband barrage) |
| `hopping/strategies.py` | crypto (AES-CTR), DSSS, uncoordinated-SS, random, fixed |
| `hopping/dqn.py` | NumPy deep-Q anti-jamming agent |
| `spectrum.py` | jammer-vs-hopper environment + occupancy dataset |
| `security/auth_timing.py` | HMAC + nonce + time-of-flight defense; spoof/replay adversaries |
| `analysis/stats.py` | paired t-test, Wilcoxon, bootstrap CI, Cohen's d, Bonferroni |
| `experiments/` | one runnable experiment per result |

## Key results (fixed seeds)

- **TDOA**: LOS estimator hits the CRLB (efficiency ≈ 1.0); GCC-PHAT validates
  the noise model (empirical/analytic σ_r ≈ 1.03); RANSAC cuts NLOS median error
  up to 20×.
- **Anti-jam**: crypto avoidance rises with M and reaches ≈ 1.0 vs the
  intelligent jammer, where predictable random/fixed collapse to 0.0. Wideband
  barrage is the one class that defeats DSSS, by exceeding its processing-gain
  margin.
- **Scaling**: reported under two training regimes — equal wall clock (a fixed
  compute allowance) and equal gradient steps (identical updates at every M).
  The gap between them is the cost of the budget; what survives the step-matched
  arm is intrinsic to the learning problem. Crypto needs no training at all.
- **Security**: a valid signature still loses to replay (success 1.0); the
  authenticated-timing layer drives spoof and replay success to 0.0.
- **Cost**: crypto 13 pJ/decision vs DQN up to 9.3 µJ (≈ 7×10⁵×), plus training.
