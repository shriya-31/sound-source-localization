"""
env_check.py

One-off script to confirm the LSO brainstem model runs end-to-end in the
`lso` conda environment, before building the ipsi/contra-level wrapper
on top of it. Not part of the final wrapper -- just a smoke test.
"""

import sys
from pathlib import Path

# lso/ and Cochlea_NEF/ both use bare, cwd-relative imports (e.g.
# `from config import SimulationConfig` in brainstem.py, `from
# cochlear_nef_model import CochlearModel` in main.ipynb), so both
# directories need to be on sys.path regardless of where this script
# is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lso"))
sys.path.insert(0, str(REPO_ROOT / "Cochlea_NEF"))

import numpy as np
from config import SimulationConfig
from cochlear_nef_model import CochlearModel
from brainstem import BrainstemModel

config = SimulationConfig()

cochlea_L = CochlearModel(
    fs=config.fs, n_channels=config.n_channels,
    low_freq=config.f_low, high_freq=config.f_high,
    neurons_per_channel=30,
)
cochlea_R = CochlearModel(
    fs=config.fs, n_channels=config.n_channels,
    low_freq=config.f_low, high_freq=config.f_high,
    neurons_per_channel=30,
)
brainstem = BrainstemModel(config=config, cochlea_L=cochlea_L, cochlea_R=cochlea_R)

# Identical tone in both ears (no ILD) -- this just checks the pipeline
# runs without error, not tuning behavior yet.
n_samples = int(config.duration_s * config.fs)
t = np.arange(n_samples) / config.fs
tone = 0.1 * np.sin(2 * np.pi * 20000 * t)

results = brainstem.run(tone, tone)

print("ild shape:", results.ild.shape)
print("ild_scalar:", results.ild_scalar)
print("lso_L range:", results.lso_L.min(), "-", results.lso_L.max())
print("lso_R range:", results.lso_R.min(), "-", results.lso_R.max())
