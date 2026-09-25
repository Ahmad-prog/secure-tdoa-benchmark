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
python -m sim.run_all --quick    # skip the slow DQN stages (~15 min)
python -m sim.replot             # redraw every figure from saved results (seconds)
```

Outputs: `results/data/*.csv`, `results/data/spectrum_occupancy_dataset.npz`,
`results/figures/*.png` (also copied to `../paper/figures`). The scaling stage
checkpoints each channel count, so an interrupted run resumes where it stopped.

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
| `plotstyle.py` | print-size figure style: figures are drawn at the width they print |
| `replot.py` | regenerates every figure from saved results |

## Key results (fixed seeds)

- **TDOA**: LOS estimator reaches the CRLB from 10 dB upward (efficiency
  0.97–1.06); RANSAC cuts NLOS median error up to 20×. Results use the analytic
  range-noise model; the signal-level GCC-PHAT check agrees with it only near
  20 dB (its error stays at 0.43–0.85 m across SNR).
- **Anti-jam**: crypto avoidance rises with M and reaches ≈ 1.0 vs the
  intelligent jammer, where predictable random/fixed collapse to 0.0. Wideband
  barrage is the one class that defeats DSSS, by exceeding its processing-gain
  margin.
- **Scaling**: the DQN beats crypto at M ≤ 100 but fails at M ≥ 1000 under both
  equal wall clock and equal gradient steps — at M=1000 and M=2320, 15,000
  updates give 0.00 avoidance on every seed. Crypto needs no training at all.
- **Why** (`experiments/exp_collapse.py`): at M=1000 training collapses the
  greedy policy onto a single channel, which the tracking jammer then covers.
  The untrained network avoids 0.72 of slots; after 3,000 updates it avoids
  none. Only the chosen action's value is updated, so each of M channels needs
  its own samples, and at M=1000 there are fewer than two random trials per
  channel in the first 3,000 steps.
- **Security**: a valid signature still loses to replay (success 1.0); the
  authenticated-timing layer rejects every simulated spoof and replay. The
  simulation gives the drone the true time of flight and an aligned clock, so
  this checks the protocol logic; a deployment must predict each beacon's time
  of flight to within ~6 ns at 20 dB.
- **Cost**: crypto 13 pJ/decision vs DQN up to 9.3 µJ (≈ 7×10⁵×), plus training.
