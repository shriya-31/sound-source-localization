"""
level_difference_wrapper.py

Adapts the real LSO brainstem model (BrainstemModel) to the same
get_spike_rate(ipsi_level, contra_level, frequency) interface used by
MockNeuron in validation/level_difference_coding.py, so the existing
Fig. 3.25 analysis pipeline (find_preferred_frequency, find_threshold,
find_centre_and_bandwidth) can run against real model output instead
of a hard-coded placeholder.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lso"))
sys.path.insert(0, str(REPO_ROOT / "Cochlea_NEF"))

import numpy as np
from config import SimulationConfig
from cochlear_nef_model import CochlearModel
from brainstem import BrainstemModel
from stimulus import StimulusGenerator


class LSOWrapper:
    """
    ipsi = left ear, contra = right ear (fixed, arbitrary convention --
    the brainstem circuit is bilaterally symmetric). get_spike_rate
    always reads out lso_L as "the ipsi-side response."
    """

    def __init__(self, config=None, neurons_per_channel=30):
        self.config = config or SimulationConfig()

        self.cochlea_L = CochlearModel(
            fs=self.config.fs, n_channels=self.config.n_channels,
            low_freq=self.config.f_low, high_freq=self.config.f_high,
            neurons_per_channel=neurons_per_channel,
        )
        self.cochlea_R = CochlearModel(
            fs=self.config.fs, n_channels=self.config.n_channels,
            low_freq=self.config.f_low, high_freq=self.config.f_high,
            neurons_per_channel=neurons_per_channel,
        )
        self.brainstem = BrainstemModel(
            config=self.config, cochlea_L=self.cochlea_L, cochlea_R=self.cochlea_R,
        )
        self.stimulus_gen = StimulusGenerator(self.config)
        self.cfs = self.cochlea_L.cfs

    @staticmethod
    def _level_to_amplitude(level_db, ref_amplitude=1.0):
        """
        dB -> waveform amplitude. No dB/SPL reference exists anywhere
        else in this codebase, and none is needed here: find_threshold
        and find_centre_and_bandwidth (validation/level_difference_coding.py)
        only fit the response curve's shape relative to itself, not an
        absolute physical SPL, so any monotonic mapping works.
        """
        return ref_amplitude * 10 ** (level_db / 20.0)

    def get_spike_rate(self, ipsi_level, contra_level, frequency):
        """Same interface as MockNeuron.get_spike_rate."""
        ipsi_tone = self._level_to_amplitude(ipsi_level) * self.stimulus_gen.generate_pure_tone(frequency)
        contra_tone = self._level_to_amplitude(contra_level) * self.stimulus_gen.generate_pure_tone(frequency)

        results = self.brainstem.run(left_signal=ipsi_tone, right_signal=contra_tone)

        # Channel is looked up from the known Greenwood CF map, not by
        # picking whichever channel has the largest response -- raw
        # lso_L magnitude is NOT reliably comparable across channels.
        # KNOWN CAVEAT (see lso/test_wrapper_sanity.py Test 3): a tone
        # at one channel's exact CF does not always produce the largest
        # response in that channel vs. others. Likely cause: CochlearModel
        # (Cochlea_NEF/cochlear_nef_model.py) auto-scales each channel's
        # NEF ensemble radius to that channel's own envelope max
        # (`ch_radius = max(ch_max, 1e-3) * 1.2`), so a channel driven
        # only by filter leakage gets a tiny radius and a correspondingly
        # hyper-sensitive population, which can inflate its decoded
        # output relative to channels with genuine strong drive. Doesn't
        # block this pipeline since find_centre_and_bandwidth only fits
        # each channel's response relative to its own max (matching how
        # Fisch (2025) normalizes per-neuron too), but worth revisiting
        # if we ever need cross-channel magnitude to be meaningful.
        ch = int(np.argmin(np.abs(self.cfs - frequency)))
        return results.lso_L[ch]
