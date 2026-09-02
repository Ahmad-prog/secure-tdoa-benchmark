"""Central configuration and physical constants for the secure-TDOA study.

All experiments import their defaults from here so that a single edit changes
the whole study consistently (reproducibility requirement, DRAC C1/C4).
"""
from __future__ import annotations
import numpy as np

# ---- physical constants -------------------------------------------------
C_LIGHT = 299_792_458.0          # m/s

# ---- TDOA geometry (metres) --------------------------------------------
# Six ground beacons on a regular hexagon of radius R at ground level; the
# drone flies inside the constellation at altitude H.
N_BEACONS = 8                    # redundancy enables RANSAC-based NLOS rejection
BEACON_RADIUS = 1_000.0          # m
BEACON_Z = 0.0
# Mast/terrain height diversity makes the vertical coordinate observable and
# keeps GDOP reasonable (coplanar ground beacons alone leave altitude nearly
# unobservable -- the altitude-bias effect reported by Dickerson et al.).
BEACON_HEIGHTS = [0.0, 40.0, 10.0, 60.0, 20.0, 45.0, 5.0, 55.0]
DRONE_ALT = 300.0                # m (matches the altitude studied by Dickerson et al.)

# ---- signal / channel ---------------------------------------------------
BANDWIDTH_HZ = 10e6              # 10 MHz positioning waveform
SNR_DB_DEFAULT = 20.0
SNR_DB_SWEEP = [0, 5, 10, 15, 20, 25, 30]

# NLOS excess-delay model: a fraction of beacons carry a positive range bias
# drawn from an exponential distribution (one-sided, as physical NLOS is).
NLOS_FRACTION = 0.34             # ~2 of 6 beacons in NLOS
NLOS_BIAS_MEAN_M = 30.0          # mean excess range (m)

# ---- Monte-Carlo --------------------------------------------------------
N_TRIALS = 500                   # per condition for TDOA RMSE
N_SEEDS = 30                     # independent seeds for anti-jam / security stats
BASE_SEED = 20260830

# ---- frequency hopping / spectrum --------------------------------------
CHANNEL_SCALES = [10, 100, 1000, 2320]   # 2320 = SINCGARS 30-88 MHz @ 25 kHz
SLOTS_PER_EPISODE = 500
WINDOW = 16                      # sliding-window depth for the DL spectrogram input

# ---- crypto -------------------------------------------------------------
AES_KEY_BYTES = 16               # AES-128
HMAC_KEY_BYTES = 32

# ---- results ------------------------------------------------------------
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "results" / "data"
FIG_DIR = ROOT / "results" / "figures"
PAPER_FIG_DIR = ROOT.parent / "paper" / "figures"
for _d in (DATA_DIR, FIG_DIR, PAPER_FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
