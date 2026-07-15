"""
spatializer.py
Binaural spatialization module.
This module applies head-related impulse responses (HRIRs) to convert a monaural acoustic stimulus into a binaural signal corresponding to a specific sound source location.
"""

import numpy as np
from scipy.signal import fftconvolve
from config import SimulationConfig

class Spatializer:
    """
    Apply HRIR-based spatial filtering.

    Parameters
    hrir: np.ndarray
        HRIR dataset. Expected shape: (2, elevation, azimuth, samples)
        where:
            dimension 0:
                0 - left ear
                1 - right ear
            dimension 1:
                elevation index
            dimension 2:
                azimuth index
            dimension 3:
                impulse response samples
    config: SimulationConfig
        Simulation configuration.
    """
    def __init__(self,hrir: np.ndarray,config: SimulationConfig):
        self.hrir = hrir
        self.config = config

    @staticmethod
    def angle_to_idx(angle_deg: float) -> int:
        """
        Convert azimuth angle into HRIR dataset index.
        Parameters
        angle_deg: float
            Sound source azimuth in degrees.
            Convention: -180...180 degrees
        Returns
        int
            Corresponding HRIR azimuth index.
        Notes
        The mouse HRTF dataset used here contains 200 azimuth positions with approximately 1.8 degree resolution.
        """
        angle_360 = angle_deg % 360
        index = int(np.floor(angle_360 / 1.8 + 0.5))
        return index % 200

    def spatialize(self, mono_signal: np.ndarray, angle_deg: float):
        """
        Generate binaural sound using HRIR convolution.
        Parameters
        mono_signal: np.ndarray
            Monaural acoustic waveform.
        angle_deg: float
            Azimuth of sound source.
        Returns
        tuple[np.ndarray, np.ndarray]
            left_signal: Sound arriving at the left ear.
            right_signal: Sound arriving at the right ear.
        """
        idx = self.angle_to_idx(angle_deg)
        elevation = self.config.elevation_idx
        left_hrir = self.hrir[0,elevation,idx,:]
        right_hrir = self.hrir[1,elevation,idx,:]
        left_signal = fftconvolve(mono_signal,left_hrir,mode="same")
        right_signal = fftconvolve(mono_signal,right_hrir,mode="same")
        return (left_signal.astype(np.float64),right_signal.astype(np.float64))

    def make_stereo(self, mono_signal: np.ndarray,angle_deg: float) -> np.ndarray:
        """
        Create stereo waveform.
        Parameters
        mono_signal: np.ndarray
            Monaural input.
        angle_deg: float
            Source azimuth.
        Returns
        np.ndarray
            Stereo waveform with shape (samples, 2)
            column 0: left ear
            column 1: right ear
        """
        left, right = self.spatialize(mono_signal,angle_deg)
        stereo = np.column_stack([left,right])
        return stereo