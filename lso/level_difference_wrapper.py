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
        # magnitude is NOT guaranteed comparable across channels (each
        # channel's upstream cochlear ensemble auto-scales its own NEF
        # radius to its own envelope max -- see CochlearModel in
        # Cochlea_NEF/cochlear_nef_model.py). ORIGINAL CAVEAT (see
        # lso/test_wrapper_sanity.py Test 3) was observed using the old
        # NEF-decoded lso_L value; re-check whether it still holds now
        # that get_spike_rate reads real per-neuron spike rates instead.
        # Doesn't block this pipeline either way since find_centre_and_bandwidth
        # only fits each channel's response relative to its own max
        # (matching how Fisch (2025) normalizes per-neuron too).
        ch = int(np.argmin(np.abs(self.cfs - frequency)))
        return results.lso_L_rate_hz[ch]
