# Secure GPS-Denied TDOA Positioning — Simulation Study

Self-contained, CPU-only, reproducible simulation for the thesis *"Benchmarking
Deep-Reinforcement-Learning Anti-Jamming Against Cryptographic Frequency Hopping
for GPS-Denied Drone Positioning."*  No GPU, no PyTorch — the deep-Q agent is a
compact NumPy implementation, which also makes its O(M) scaling cost the object
of study rather than an assertion.

## Reproduce

```bash
python -m sim.run_all            # full study (~15 min on 12 CPU cores)
python -m sim.run_all --quick    # skip the slow DQN sweeps
```

Outputs: `results/data/*.csv`, `results/data/spectrum_occupancy_dataset.npz`,
`results/figures/*.png` (also copied to `../paper/figures`).

## Layout

| Module | Purpose | Covers |
|---|---|---|
| `tdoa/` | geometry, SNR→range-noise, CRLB, Chan-Ho + Taylor, RANSAC NLOS mitigation, GCC-PHAT | RO1/RQ1, C1 |
| `jammers/taxonomy.py` | nine-class, three-tier jammer taxonomy | C3 |
| `hopping/strategies.py` | crypto (AES-CTR), DSSS, uncoordinated-SS, random, fixed | C2/C6 |
| `hopping/dqn.py` | NumPy deep-Q anti-jamming agent | RO2/C3 |
| `spectrum.py` | jammer-vs-hopper environment + occupancy dataset | C1/C4 |
| `security/auth_timing.py` | HMAC + nonce + time-of-flight defense; spoof/replay adversaries | RO4, C5/C9 |
| `analysis/stats.py` | paired t-test, Wilcoxon, bootstrap CI, Cohen's d, Bonferroni | IRB stats |
| `experiments/` | one runnable experiment per result | — |

## Key results (fixed seeds)

- **TDOA**: LOS estimator hits the CRLB (efficiency ≈ 1.0); GCC-PHAT validates
  the noise model (empirical/analytic σ_r ≈ 1.03); RANSAC cuts NLOS median error
  up to 20×.
- **Anti-jam**: crypto avoidance rises with M and reaches ≈ 1.0 vs the
  intelligent jammer, where predictable random/fixed collapse to 0.0.
- **Scaling**: DQN beats crypto at M≤100 but collapses at M≥1000 under a fixed
  compute budget (training throughput 1801→5 steps/s); crypto is unaffected.
- **Security**: a valid signature still loses to replay (success 1.0); the
  authenticated-timing layer drives spoof and replay success to 0.0.
- **Cost**: crypto 13 pJ/decision vs DQN up to 9.3 µJ (≈ 7×10⁵×), plus training.
