import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lso"))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from level_difference_wrapper import LSOWrapper

"""
Draft code for finding distribution of LSO neuron ILD tuning centre and bandwidth as in Fisch (2025).

TODO:
- test
- plot model distribution with Fisch distribution
- wrapper for sound localization model
"""


class MockNeuron:
    def __init__(self, preferred_frequency=50000, max_rate=300):
        """
        Linear-nonlinear approximation of example neuron in Fisch (2025) Fig. 3.25.
        :param preferred_frequency: sound frequency with strongest response
        :param max_rate: saturated spike rate
        """
        self.preferred_frequency = preferred_frequency
        self.frequency_width = preferred_frequency * .2
        self.max_rate = max_rate
        alpha = -np.log(9) # 10% response level of logistic sigmoid

        # linear parameters from solving equations at example points in Fisch (2025)
        self.a = -0.3*alpha
        self.b = .0667*alpha
        self.d = 16*alpha

    def get_spike_rate(self, ipsi_level, contra_level, frequency):
        level_difference_drive = self.a*ipsi_level + self.b*contra_level + self.d

        df = frequency - self.preferred_frequency
        frequency_gain = np.exp(-df**2/2/self.frequency_width**2)

        return self.max_rate * frequency_gain / (1 + np.exp(-level_difference_drive))


def find_preferred_frequency(neuron):
    def gaussian(x, amplitude, mean, sd):
        """
        amplitude: Peak height
        mean: Peak center on the x-axis
        sd: Standard deviation
        """
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * sd ** 2))

    frequencies = np.linspace(0, 100000, 10) # Hz
    responses = [neuron.get_spike_rate(70, 0, f) for f in frequencies]
    popt, cov = curve_fit(gaussian, frequencies, responses, p0=[200, 50000, 10000])
    return popt[1]


def logsig(x, slope, bias, max):
    drive = slope * x + bias
    return max / (1 + np.exp(-drive))


def find_threshold(neuron, frequency):
    amplitudes = np.linspace(0, 90, 10) # dB
    responses = [neuron.get_spike_rate(a, 0, frequency) for a in amplitudes]
    popt, cov = curve_fit(logsig, amplitudes, responses, p0=[1, 1, float(np.max(responses))], bounds=([0, -250, 0], [20, 250, 1000]))
    slope, bias, max = popt
    return (-np.log(9) - bias) / slope # threshold (10% of max response) according to sigmoid model


def find_centre_and_bandwidth(neuron, frequency=None, plot=False):
    # frequency=None: discover it via find_preferred_frequency, as before
    # (appropriate when the neuron's CF is unknown, e.g. MockNeuron).
    # frequency given: use it directly, skipping discovery -- appropriate
    # for LSOWrapper, where the CF is already known exactly (wrapper.cfs)
    # and a Gaussian-fit discovery risks a bad fit (see the channel-gain
    # caveat noted in lso/level_difference_wrapper.py).

    freq = frequency if frequency is not None else find_preferred_frequency(neuron)
    threshold = find_threshold(neuron, freq)

    ipsi = threshold + 20
    contras = np.linspace(0, 100, 21)
    rates = np.array([neuron.get_spike_rate(ipsi, contra, freq) for contra in contras])

    popt, cov = curve_fit(logsig, contras, rates, p0=[-1, 0, float(np.max(rates))], bounds=([-10, -100, 0], [0, 100, 1000]))
    slope, bias, max = popt

    centre = -bias/slope
    bandwidth = -2 * np.log(9) / slope

    if plot:
        plt.plot(contras, rates)
        plt.xlabel('Contralateral level (dB)')
        plt.ylabel('Spike rate')
        plt.title('Centre: {} Bandwidth: {}'.format(centre, bandwidth))
        plt.show()

    return centre, bandwidth, ipsi


# --- Snapshots of the original (pre-fix) versions, commit d401ffe -------
# Kept only for the 3-case comparison in __main__ below. MockNeuron,
# find_preferred_frequency, and logsig are unchanged from that commit,
# so only the two functions that actually differ (hardcoded p0=200, no
# frequency param, 2-value return) are duplicated here.

def find_threshold_og(neuron, frequency):
    amplitudes = np.linspace(0, 90, 10) # dB
    responses = [neuron.get_spike_rate(a, 0, frequency) for a in amplitudes]
    popt, cov = curve_fit(logsig, amplitudes, responses, p0=[1, 1, 200], bounds=([0, -250, 0], [20, 250, 1000]))
    slope, bias, max = popt
    return (-np.log(9) - bias) / slope


def find_centre_and_bandwidth_og(neuron, plot=False):
    freq = find_preferred_frequency(neuron)
    threshold = find_threshold_og(neuron, freq)

    ipsi = threshold + 20
    contras = np.linspace(0, 100, 21)
    rates = np.array([neuron.get_spike_rate(ipsi, contra, freq) for contra in contras])

    popt, cov = curve_fit(logsig, contras, rates, p0=[-1, 0, 200], bounds=([-10, -100, 0], [0, 100, 1000]))
    slope, bias, max = popt

    centre = -bias/slope
    bandwidth = -2 * np.log(9) / slope

    if plot:
        plt.plot(contras, rates)
        plt.xlabel('Contralateral level (dB)')
        plt.ylabel('Spike rate')
        plt.title('Centre: {} Bandwidth: {}'.format(centre, bandwidth))
        plt.show()

    return centre, bandwidth


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent

    # Case 1: OG pipeline + MockNeuron
    neuron = MockNeuron()
    centre, bandwidth = find_centre_and_bandwidth_og(neuron, plot=True)
    plt.savefig(out_dir / "case1_og_mockneuron.png", dpi=150)
    plt.close()
    print(f"[1] OG + MockNeuron: centre={centre:.2f} dB, bandwidth={bandwidth:.2f} dB (no ipsi returned -> no ILD50 in the original code)")

    # Case 2: New pipeline + MockNeuron
    neuron = MockNeuron()
    centre, bandwidth, ipsi = find_centre_and_bandwidth(neuron, plot=True)
    plt.savefig(out_dir / "case2_new_mockneuron.png", dpi=150)
    plt.close()
    print(f"[2] New + MockNeuron: ILD50={ipsi - centre:.2f} dB, dynamic range={bandwidth:.2f} dB")

    # Case 3: New pipeline + LSOWrapper (slow -- real simulation, ~31 runs)
    wrapper = LSOWrapper()
    ch = 39  # CF ~50078 Hz -- high-frequency channel, contrast to ch 24
    freq = float(wrapper.cfs[ch])
    centre, bandwidth, ipsi = find_centre_and_bandwidth(wrapper, frequency=freq, plot=True)
    plt.savefig(out_dir / "case3_new_lsowrapper.png", dpi=150)
    plt.close()
    print(f"[3] New + LSOWrapper (ch{ch}, CF={freq:.0f} Hz): ILD50={ipsi - centre:.2f} dB, dynamic range={bandwidth:.2f} dB")



