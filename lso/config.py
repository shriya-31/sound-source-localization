"""
config.py
Configuration parameters for the auditory brainstem model.
All simulation, neural, and cochlear parameters are stored in the
SimulationConfig dataclass to avoid the use of global variables.
"""

from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """
    Configuration of the auditory localization simulation.

    Parameters

    fs: int
        Audio sampling frequency in Hz.
    duration_s: float
        Duration of the acoustic stimulus.
    elevation_idx: int
        Elevation index used when selecting HRIRs from the dataset.
    n_channels: int
        Number of cochlear frequency channels.
    f_low: float
        Lowest frequency (Hz).
    f_high: float
        Highest frequency (Hz).
    dt_sim: float
        Simulation time step used by Nengo (seconds).
    warmup_s: float
        Initial transient period discarded before averaging neural activity.
    smooth_ms: float
        Low-pass filter time constant used when probing neural signals.
    n_sbc: int
        Number of neurons representing each spherical bushy cell population.
    n_mntb: int
        Number of neurons representing each MNTB population.
    n_lso: int
        Number of neurons representing each LSO population.
    tau_excit: float
        Synaptic time constant of excitatory connections.
    tau_inhib: float
        Synaptic time constant of inhibitory connections.
    tau_rc: float
        Membrane RC time constant of LIF neurons.
    tau_ref: float
        Absolute refractory period of LIF neurons.
    w_sbc_mntb: float
        Excitatory connection weight from SBC to MNTB.
    w_sbc_lso: float
        Excitatory connection weight from SBC to LSO.
    w_mntb_lso: float
        Inhibitory connection weight from MNTB to LSO.
    """
    
    #Audio
    fs: int = 500_000 #
    duration_s: float = 0.1

    #HRIR
    elevation_idx: int = 0 #

    #Cochlea
    n_channels: int = 48
    f_low: float = 5_000 #
    f_high: float = 80_000 #

    #Simulation
    dt_sim: float = 0.0001
    warmup_s: float = 0.02
    smooth_ms: float = 10.0

    #Neural populations
    n_sbc: int = 100
    n_mntb: int = 100
    n_lso: int = 200

    #Synapses
    tau_excit: float = 0.002
    tau_inhib: float = 0.002

    #LIF neurons
    tau_rc: float = 0.005
    tau_ref: float = 0.001

    #Connection weights
    w_sbc_mntb: float = 1.0
    w_sbc_lso: float = 1.0
    w_mntb_lso: float = 1.2