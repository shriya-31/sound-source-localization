"""
brainstem.py
Spiking neural network model of the mouse auditory brainstem.
"""

import numpy as np
import nengo
from config import SimulationConfig
from results import SimulationResults

class BrainstemModel:
    """
    Nengo implementation of the auditory brainstem ILD pathway.
    Parameters
    config : SimulationConfig
        Simulation parameters.
    cochlear_model_class: class
        External cochlear implementation.
    """
    def __init__(self,config: SimulationConfig,cochlea_L,cochlea_R):
        self.config = config
        self.cochlea_L = cochlea_L
        self.cochlea_R = cochlea_R
        self.model = None
        self.probes = {}

    def build_network(self):
        """
        Construct the complete Nengo auditory brainstem network.
        Returns
        nengo.Network
            Built neural network.
        """
        config = self.config
        identity = np.eye(config.n_channels)
        intercepts = nengo.dists.Uniform(0.0,0.8)
        with nengo.Network(seed=0) as model:
            #Cochlear stage
            cochlea_L_net, cochlea_L_probes = (self.cochlea_L.build())
            cochlea_R_net, cochlea_R_probes = (self.cochlea_R.build())
            #SBC populations
            sbc_L = nengo.networks.EnsembleArray(
                n_neurons=config.n_sbc,
                n_ensembles=config.n_channels,
                radius=3.0,
                encoders=nengo.dists.Choice([[1.0]]),
                intercepts=intercepts
            )
            sbc_R = nengo.networks.EnsembleArray(
                n_neurons=config.n_sbc,
                n_ensembles=config.n_channels,
                radius=3.0,
                encoders=nengo.dists.Choice([[1.0]]),
                intercepts=intercepts
            )
            #MNTB populations
            mntb_L = nengo.networks.EnsembleArray(
                n_neurons=config.n_mntb,
                n_ensembles=config.n_channels,
                radius=3.0,
                encoders=nengo.dists.Choice([[1.0]]),
                intercepts=intercepts
            )
            mntb_R = nengo.networks.EnsembleArray(
                n_neurons=config.n_mntb,
                n_ensembles=config.n_channels,
                radius=3.0,
                encoders=nengo.dists.Choice([[1.0]]),
                intercepts=intercepts
            )
            #LSO populations
            lso_L = nengo.networks.EnsembleArray(
                n_neurons=config.n_lso,
                n_ensembles=config.n_channels,
                radius=2.0,
                encoders=nengo.dists.Choice([[1.0]]),
                intercepts=intercepts
            )
            lso_R = nengo.networks.EnsembleArray(
                n_neurons=config.n_lso,
                n_ensembles=config.n_channels,
                radius=2.0,
                encoders=nengo.dists.Choice([[1.0]]),
                intercepts=intercepts
            )

            #Cochlea - SBC
            for ch in range(config.n_channels):
                nengo.Connection(cochlea_L_probes["ensembles"][ch],sbc_L.input[ch],synapse=config.tau_excit)
                nengo.Connection(cochlea_R_probes["ensembles"][ch],sbc_R.input[ch],synapse=config.tau_excit)
            #SBC-MNTB
            #Contralateral inhibition pathway
            nengo.Connection(sbc_R.output,mntb_L.input,synapse=config.tau_excit,transform=identity*config.w_sbc_mntb)
            nengo.Connection(sbc_L.output,mntb_R.input,synapse=config.tau_excit,transform=identity*config.w_sbc_mntb)

            #SBC - LSO excitation
            nengo.Connection(sbc_L.output,lso_L.input,synapse=config.tau_excit,transform=identity*config.w_sbc_lso)
            nengo.Connection(sbc_R.output,lso_R.input,synapse=config.tau_excit,transform=identity*config.w_sbc_lso)

            #MNTB - LSO inhibition
            nengo.Connection(mntb_L.output,lso_L.input,synapse=config.tau_inhib,transform=-identity*config.w_mntb_lso)
            nengo.Connection(mntb_R.output,lso_R.input,synapse=config.tau_inhib,transform=-identity*config.w_mntb_lso)

            #Neural ILD representation
            ild_node = nengo.Node(size_in=config.n_channels)
            nengo.Connection(lso_L.output,ild_node,synapse=config.tau_excit,transform=-identity)
            nengo.Connection(lso_R.output,ild_node,synapse=config.tau_excit,transform=identity)

            #Probes
            smooth_tau = (config.smooth_ms / 1000)
            self.probes["ild"] = nengo.Probe(ild_node,synapse=smooth_tau)
            self.probes["sbc_L"] = nengo.Probe(sbc_L.output,synapse=smooth_tau)
            self.probes["sbc_R"] = nengo.Probe(sbc_R.output,synapse=smooth_tau)
            self.probes["mntb_L"] = nengo.Probe(mntb_L.output,synapse=smooth_tau)
            self.probes["mntb_R"] = nengo.Probe(mntb_R.output,synapse=smooth_tau)
            self.probes["lso_L"] = nengo.Probe(lso_L.output,synapse=smooth_tau)
            self.probes["lso_R"] = nengo.Probe(lso_R.output,synapse=smooth_tau)

        self.model = model
        return model

    def run(self,left_signal: np.ndarray,right_signal: np.ndarray) -> SimulationResults:
        """
        Run auditory brainstem simulation.
        Parameters
        left_signal: np.ndarray
            Left ear waveform.
        right_signal: np.ndarray
            Right ear waveform.
        Returns
        SimulationResults
            Neural activity recorded from the model.
        """
        self.build_network()
        with nengo.Simulator(self.model,dt=self.config.dt_sim,progress_bar=False) as sim:
            sim.run(self.config.duration_s)
        warmup_samples = int(self.config.warmup_s /self.config.dt_sim)
        sl = slice(warmup_samples,None)
        ild = (sim.data[self.probes["ild"]][sl].mean(axis=0))
        sbc_L = (sim.data[self.probes["sbc_L"]][sl].mean(axis=0))
        sbc_R = (sim.data[self.probes["sbc_R"]][sl].mean(axis=0))
        mntb_L = (sim.data[self.probes["mntb_L"]][sl].mean(axis=0))
        mntb_R = (sim.data[self.probes["mntb_R"]][sl].mean(axis=0))
        lso_L = (sim.data[self.probes["lso_L"]][sl].mean(axis=0))
        lso_R = (sim.data[self.probes["lso_R"]][sl].mean(axis=0))

        return SimulationResults(
            ild=ild,
            ild_scalar=float(ild.mean()),
            cfs=self.cochlea_L.cfs,
            sbc_L=sbc_L,
            sbc_R=sbc_R,
            mntb_L=mntb_L,
            mntb_R=mntb_R,
            lso_L=lso_L,
            lso_R=lso_R
        )