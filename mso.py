import numpy as np
import matplotlib.pyplot as plt
import nengo
from nengo.neurons import NeuronType

"""
Prototype of ITD sensitivity model based on Grothe et al. (2010).  
"""

class StepStimulus:
    def __init__(self, amplitude, delay, duration):
        self.amplitude = amplitude
        self.delay = delay
        self.duration = duration

    def __call__(self, t):
        if t > self.delay and t < self.delay + self.duration:
            return self.amplitude
        else:
            return 0


class Branch(NeuronType):
    """
    A main branch of a MSO neuron, modelled as a non-rectified linear neuron.

    Parameters
    ----------
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    def __init__(self, initial_state=None):
        super().__init__(initial_state)

        self.amplitude = 1

    def gain_bias(self, max_rates, intercepts):
        """Determine gain and bias by shifting and scaling the lines."""
        max_rates = np.asarray(np.atleast_1d(max_rates), dtype=float)
        intercepts = np.asarray(np.atleast_1d(intercepts), dtype=float)
        gain = max_rates / (1 - intercepts)
        bias = -intercepts * gain
        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        """Compute the inverse of gain_bias."""
        intercepts = -bias / gain
        max_rates = gain * (1 - intercepts)
        return max_rates, intercepts

    def step(self, dt, J, output):
        output[...] = J


model = nengo.Network()
with model:
    ipsi_envelope = nengo.Node(StepStimulus(amplitude=1, delay=0.12, duration=.02))
    contra_envelope = nengo.Node(StepStimulus(amplitude=1, delay=0.1, duration=.02))

    n = 50
    contra_interneurons = nengo.Ensemble(n_neurons=n, dimensions=1)
    ipsi_branches = nengo.Ensemble(n_neurons=n, dimensions=1, neuron_type=Branch(), encoders=np.ones((n,1)), bias=np.zeros(n), gain=4*np.ones(n))
    contra_branches = nengo.Ensemble(n_neurons=n, dimensions=1, neuron_type=Branch(), encoders=np.ones((n,1)), bias=np.zeros(n), gain=4*np.ones(n))
    MSO = nengo.Ensemble(
        n_neurons=n, dimensions=1, encoders=np.ones((n,1)),
        bias=-3+.25*np.random.randn(n),
        gain=5*np.random.rand(n)
    )

    nengo.Connection(ipsi_envelope, ipsi_branches, synapse=.002, transform=[[.1]])

    nengo.Connection(contra_envelope, contra_branches, synapse=.004)
    nengo.Connection(contra_envelope, contra_interneurons, synapse=.001)
    nengo.Connection(contra_interneurons, contra_branches, synapse=.001, transform=[[-1]])

    nengo.Connection(ipsi_branches.neurons, MSO.neurons, synapse=None, transform=np.eye(n))
    nengo.Connection(contra_branches.neurons, MSO.neurons, synapse=None, transform=np.eye(n))

    ispi_probe = nengo.Probe(ipsi_branches, synapse=0.01)
    contra_probe = nengo.Probe(contra_branches, synapse=0.01)
    MSO_probe = nengo.Probe(MSO, synapse=0.01)


if __name__ == "__main__":
    sim = nengo.Simulator(model)
    sim.run(1)
    plt.plot(sim.trange(), sim.data[ispi_probe])
    plt.plot(sim.trange(), sim.data[contra_probe])
    plt.plot(sim.trange(), sim.data[MSO_probe])
    plt.legend(['Ipsi', 'Contra', 'MSO'])
    plt.show()

