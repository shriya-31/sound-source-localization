"""
stimulus.py

Acoustic stimulus generation module. This file contains utilities for generating acoustic signals used as inputs for the auditory brainstem model.
"""

import numpy as np
from scipy.signal import butter, sosfilt
from config import SimulationConfig

class StimulusGenerator:
    """
    Generate acoustic stimuli for auditory localization experiments.

    Parameters
    config: SimulationConfig
        Simulation configuration containing sampling frequency, duration, and frequency range parameters.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def generate_pure_tone(self, frequency: float) -> np.ndarray:
        """
        Generate a sinusoidal pure tone.

        Parameters
        frequency: float
            Tone frequency in Hz.
        Returns
        np.ndarray
            Generated waveform with amplitude range [-1, 1].
        Notes
        The generated signal is monaural and must be spatialized using HRIRs before binaural processing.
        """
        n_samples = int(self.config.duration_s * self.config.fs)
        time = np.arange(n_samples) / self.config.fs
        signal = np.sin(2 * np.pi * frequency * time)
        return signal.astype(np.float64)



    def generate_noise(self, seed: int = 0) -> np.ndarray:
        """
        Generate band-pass filtered white noise.
        The noise spectrum is limited to the frequency range.

        Parameters
        seed: int, optional
            Random seed for reproducibility.
        Returns
        np.ndarray
            Normalized band-pass noise waveform.
        """
        rng = np.random.default_rng(seed)
        n_samples = int(self.config.duration_s * self.config.fs)
        noise = rng.standard_normal(n_samples)
        sos = butter(N=4,Wn=[self.config.f_low / (self.config.fs / 2),self.config.f_high / (self.config.fs / 2)],btype="band",output="sos")
        filtered_noise = sosfilt(sos,noise)
        filtered_noise /= (np.max(np.abs(filtered_noise))+1e-12)
        return filtered_noise.astype(np.float64)

    def generate(self,stimulus_type: str,frequency: float = None,seed: int = 0) -> np.ndarray:
        """
        General stimulus generation interface.

        Parameters
        stimulus_type : str
            Type of stimulus: "tone", "noise"
        frequency : float, optional
            Frequency of tone in Hz.
        seed : int, optional
            Random seed for noise generation.
        Returns
        np.ndarray
            Generated monaural acoustic signal.
        """ 
        if stimulus_type.lower() == "tone":
            if frequency is None:
                raise ValueError("Frequency must be provided for tone stimulus.")
            return self.generate_pure_tone(frequency)
        elif stimulus_type.lower() == "noise":
            return self.generate_noise(seed)
        else:
            raise ValueError(f"Unknown stimulus type: {stimulus_type}")