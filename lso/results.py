"""
results.p
Data structures for storing auditory brainstem simulation outputs.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationResults:
    """
    Container for auditory brainstem model outputs.

    Parameters
    ild: np.ndarray
        Neural ILD representation. Positive values indicate stronger activity in the right LSO, while negative values indicate stronger activity in the left LSO.
    ild_scalar: float
        Mean ILD value across all frequency channels.
    cfs: np.ndarray
        Cochlear characteristic frequencies.
    sbc_L: np.ndarray
        Left spherical bushy cell population activity.
    sbc_R: np.ndarray
        Right spherical bushy cell population activity.
    mntb_L: np.ndarray
        Left medial nucleus of trapezoid body activity.
    mntb_R: np.ndarray
        Right medial nucleus of trapezoid body activity.
    lso_L: np.ndarray
        Left lateral superior olive activity.
    lso_R: np.ndarray
        Right lateral superior olive activity.
    """
    ild: np.ndarray
    ild_scalar: float
    cfs: np.ndarray
    sbc_L: np.ndarray
    sbc_R: np.ndarray
    mntb_L: np.ndarray
    mntb_R: np.ndarray
    lso_L: np.ndarray
    lso_R: np.ndarray