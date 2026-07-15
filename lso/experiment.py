"""
Interface for evaluating the auditory brainstem model over multiple source azimuths.
"""

import gc
import numpy as np
from results import SimulationResults

class LocalizationExperiment:
    """
    Run the auditory localization model for a range of source azimuths.
    """
    def __init__(self, stimulus, spatializer, cochlea_L,cochlea_R,brainstem):
        self.stimulus = stimulus
        self.spatializer = spatializer
        self.cochlea_L = cochlea_L
        self.cochlea_R = cochlea_R
        self.brainstem = brainstem

    def run(self, angles):
        """
        Evaluate the neural ILD representation for every azimuth.
        Parameters
        angles: ndarray
            Source azimuths in degrees.
        Returns
        dict
        """
        n_channels = self.brainstem.config.n_channels
        neural_ild = np.zeros((n_channels, len(angles)))
        broadband = np.zeros(len(angles))
        cfs = None
        for i, angle in enumerate(angles):
            print(f"Running {angle:.1f}°")
            left, right = self.spatializer.spatialize(self.stimulus, angle)
            
            self.cochlea_L.set_audio(left)
            self.cochlea_R.set_audio(right)

            results = self.brainstem.run(self.cochlea_L, self.cochlea_R)

            neural_ild[:, i] = results.ild
            broadband[i] = results.ild_scalar

            if cfs is None:
                cfs = results.cfs

            gc.collect()

        return {
            "angles": angles,
            "ild": neural_ild,
            "broadband": broadband,
            "cfs": cfs,
        }