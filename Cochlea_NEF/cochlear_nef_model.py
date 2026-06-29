"""
Biologically realistic cochlear model in Nengo (NEF).

Pipeline (mirrors real cochlear physiology):

  audio --> [Gammatone filterbank]      <- basilar membrane mechanics
            (ERB-spaced on Greenwood
             tonotopic map)
        --> [Half-wave rectify +        <- inner hair cell transduction
             compressive nonlinearity]
        --> [One NEF ensemble per       <- auditory nerve / spiral ganglion,
             channel, tonotopically        tonotopically organized
             ordered]
        --> spikes / decoded envelope per channel

Each ensemble is a population of LIF neurons that uses NEF encoders/decoders
to represent a scalar: the instantaneous rectified, compressed amplitude of
the basilar-membrane motion at that cochlear place. Wiring N independent
ensembles -- rather than one big ensemble -- mirrors the real cochlea, where
each frequency channel is carried by its own dedicated population of spiral
ganglion neurons with no mixing between channels at this stage.
"""

import numpy as np
import nengo

from cochlear_filterbank import GammatoneFilterbank, make_erb_spaced_freqs


def hair_cell_transduce(x, compression_exponent=0.4, gain=5.0):
    """
    Inner hair cell transduction applied to one gammatone channel's output.

    Two well-established nonlinearities are applied:
      1. Half-wave rectification: stereocilia bending only opens
         mechanotransduction channels for deflection in one direction,
         so hair cells respond mainly to the positive half of basilar
         membrane motion.
      2. Compressive nonlinearity (power-law, exponent < 1): the inner
         ear's active cochlear amplifier gives a highly compressive
         input/output function (loud sounds are compressed much more
         than soft ones) -- this is why human hearing covers ~120 dB of
         intensity. We approximate it with a power-law compression on
         the rectified signal.
    """
    rectified = np.maximum(x, 0.0)
    compressed = gain * (rectified ** compression_exponent)
    return compressed


class CochlearModel:
    """
    Builds a Nengo network with one ensemble per cochlear channel.

    Parameters
    ----------
    fs : sample rate of the audio (Hz)
    n_channels : number of tonotopic channels (cochlear places)
    low_freq, high_freq : frequency range covered by the channels (Hz)
    neurons_per_channel : LIF neurons per auditory-nerve ensemble
    """

    def __init__(self, fs, n_channels=32, low_freq=100, high_freq=8000,
                 neurons_per_channel=40):
        self.fs = fs
        self.n_channels = n_channels
        self.cfs, self.positions = make_erb_spaced_freqs(low_freq, high_freq, n_channels)
        self.filterbank = GammatoneFilterbank(fs, self.cfs)
        self.neurons_per_channel = neurons_per_channel

        self._audio = None
        self._channel_envelopes = None  # precomputed hair-cell output, indexed by channel then time
        self._dt = None

    def set_audio(self, audio):
        """Run audio through the mechanical + hair-cell stages (offline,
        since this is a fixed deterministic filter, not something the NEF
        needs to compute) and cache per-channel envelopes for the Nengo
        Nodes to look up during simulation."""
        self._audio = np.asarray(audio)
        mech_out = self.filterbank.process(self._audio)
        self._channel_envelopes = hair_cell_transduce(mech_out)
        self._dt = 1.0 / self.fs

    def _lookup(self, channel_idx):
        """Return a function of t that interpolates the precomputed
        hair-cell envelope for one channel -- this is what drives that
        channel's Nengo ensemble."""
        env = self._channel_envelopes[channel_idx]
        t_axis = np.arange(len(env)) * self._dt
        max_t = t_axis[-1] if len(t_axis) else 0.0

        def f(t):
            if self._channel_envelopes is None:
                return 0.0
            idx = int(min(t, max_t) / self._dt)
            idx = min(idx, len(env) - 1)
            return env[idx]
        return f

    def build(self):
        """
        Construct the Nengo network. Must call set_audio() first.
        Returns (network, probes) where probes is a dict with:
          'spikes': list of spike probes, one per channel
          'envelope': list of decoded-output probes, one per channel
          'input': list of raw hair-cell-drive probes, one per channel
        """
        if self._channel_envelopes is None:
            raise RuntimeError("Call set_audio(audio) before build()")

        net = nengo.Network(label="Cochlear Model (NEF)")
        probes = {"spikes": [], "envelope": [], "input": [], "nodes": [], "ensembles": []}

        with net:
            for ch in range(self.n_channels):
                drive_fn = self._lookup(ch)
                node = nengo.Node(drive_fn, label=f"cochlea_in_ch{ch}_{self.cfs[ch]:.0f}Hz")

                # Each channel gets its OWN representational radius, scaled to
                # that channel's own envelope range. A single global radius
                # shared across all channels would waste dynamic range on
                # quieter (often high-CF) channels, leaving many neurons'
                # intercepts above any input they ever receive -- exactly the
                # kind of failure that makes weaker channels go silent even
                # though they should be responsive at their own scale.
                ch_max = float(np.max(self._channel_envelopes[ch]))
                ch_radius = max(ch_max, 1e-3) * 1.2

                # Auditory-nerve / spiral-ganglion ensemble for this channel.
                # Encoders fixed to +1: hair cells only ever excite their
                # afferent nerve fibers (a nonnegative drive), so unlike a
                # generic NEF population we don't want half the neurons
                # tuned to negative values of a signal that's never negative.
                ens = nengo.Ensemble(
                    n_neurons=self.neurons_per_channel,
                    dimensions=1,
                    radius=ch_radius,
                    encoders=nengo.dists.Choice([[1]]),
                    intercepts=nengo.dists.Uniform(-0.2, 0.8),
                    max_rates=nengo.dists.Uniform(100, 250),
                    # tau_ref ~0.75ms matches the absolute refractory period
                    # used for auditory nerve fibers in the standard
                    # Zilany/Bruce/Carney periphery model (Bruce et al. 2018,
                    # Hearing Research) -- real ANFs cannot fire much faster
                    # than ~1/tau_ref even under saturating drive.
                    neuron_type=nengo.LIF(tau_ref=0.00075, tau_rc=0.01),
                    label=f"AN_ch{ch}_{self.cfs[ch]:.0f}Hz",
                )

                nengo.Connection(node, ens, synapse=0.002)

                probes["input"].append(nengo.Probe(node))
                probes["spikes"].append(nengo.Probe(ens.neurons))
                probes["envelope"].append(nengo.Probe(ens, synapse=0.01))
                probes["nodes"].append(node)
                probes["ensembles"].append(ens)

        self.net = net
        self.probes = probes
        return net, probes


if __name__ == "__main__":
    fs = 20000
    model = CochlearModel(fs=fs, n_channels=8, low_freq=100, high_freq=4000,
                           neurons_per_channel=30)

    t = np.arange(0, 0.3, 1 / fs)
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1 kHz pure tone test
    model.set_audio(audio)
    net, probes = model.build()

    with nengo.Simulator(net, progress_bar=False) as sim:
        sim.run(0.3)

    print("Channel CFs (Hz):", np.round(model.cfs, 1))
    rates = []
    for ch in range(model.n_channels):
        # Nengo's neuron probes report instantaneous rate (1/dt on a spike
        # bin, 0 otherwise), so the time-average IS the mean firing rate --
        # no extra division by dt or duration needed.
        spikes = sim.data[probes["spikes"][ch]]
        rate = spikes.mean(axis=0).mean()  # Hz, averaged over neurons in ensemble
        rates.append(rate)
    rates = np.array(rates)
    print("Mean firing rate per channel (Hz):", np.round(rates, 1))
    print("Peak-responding channel:", np.argmax(rates), "CF =", model.cfs[np.argmax(rates)])
